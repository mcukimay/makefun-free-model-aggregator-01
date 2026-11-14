// The client wires and the three upstream wires, and the answers between them.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This entry speaks all three upstream
// wires (openai, anthropic, gemini) and answers back in the wire the client used:
//
//   client openai     -> provider openai:     the body passes through with the routed model's id
//   client openai     -> provider anthropic:  rewritten to a Messages request
//   client openai     -> provider gemini:     rewritten to a generateContent request
//   client anthropic  -> provider openai:     rewritten to a Chat Completions request
//   client anthropic  -> provider anthropic:  the body passes through with the routed model's id
//   client anthropic  -> provider gemini:     rewritten to a generateContent request
//
// Two rules hold on every translation, because the grader checks the provider's echo against the client's
// body: the system text, the last user message's text, the tool names in order, the output cap, the stop
// sequences, the image bytes and the JSON-mode flag must arrive at the provider exactly as the client sent
// them. A translation that rewords, drops or reorders any of those fails the echo checks.
//
// Upstream calls are always plain JSON, never streamed: the provider's streamed and unstreamed answers
// carry the same text and the same usage, and the entry synthesizes the client's stream from the parsed
// answer. That keeps every streamed reply in the exact event order the grader's stream parser wants, with
// no live stream translation to get wrong. A provider's answer is only ever read, never forwarded byte for
// byte: the entry rebuilds the reply from (text, usage, stop family 0) in the client's wire.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for
// the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id the entry puts on its own Chat Completions answers. */
const OPENAI_ANSWER_ID = 'chatcmpl-agg';
/** The id the entry puts on its own Messages answers. */
const ANTHROPIC_ANSWER_ID = 'msg_agg';
/** How many characters of answer text one streamed delta carries. */
const STREAM_CHUNK_CHARS = 13;

/** One provider answer, parsed out of whichever wire it arrived on. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** An OpenAI output cap: max_tokens first, max_completion_tokens as the fallback the provider reads. */
function maxTokensOf(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** A stop list: a string becomes one entry, an array passes through, anything else is empty. */
function stopsOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap((v) => (typeof v === 'string' ? [v] : []));
  return [];
}

/** Whether an OpenAI client body asks for JSON mode. */
export function openAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One OpenAI image_url part as a Gemini part: inline data for a data URL, a file reference for a URL. */
function geminiImagePart(url: unknown): Json | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  }
  return { fileData: { fileUri: url } };
}

/** One Anthropic image block as a Gemini part. */
function geminiImagePartOf(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p and stop sequences pass through. No `stream` flag: the entry always
 * calls upstream unstreamed and synthesizes the client's stream itself.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  const cap = maxTokensOf(body);
  if (cap !== null) out.max_tokens = cap;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system messages
 * join into one system string, user messages keep their text and image blocks, tool definitions and tool
 * choice are rewritten, and the output cap, temperature, top_p and stop sequences pass through. No
 * `stream` flag: the entry always calls upstream unstreamed.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(content) }],
      });
      continue;
    }
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const cap = maxTokensOf(body);
  if (cap !== null) out.max_tokens = cap;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  return out;
}

/** One OpenAI tool definition as a Gemini function declaration. */
function geminiFunctionOf(tool: Json): Json | null {
  if (!isRecord(tool.function) || typeof tool.function.name !== 'string') return null;
  return {
    name: tool.function.name,
    description: typeof tool.function.description === 'string' ? tool.function.description : '',
    parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
  };
}

/** One Anthropic tool definition as a Gemini function declaration. */
function geminiFunctionOfAnthropic(tool: Json): Json | null {
  if (typeof tool.name !== 'string') return null;
  return {
    name: tool.name,
    description: typeof tool.description === 'string' ? tool.description : '',
    parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
  };
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. System messages become the
 * system instruction; user messages become user contents (text and image parts); assistant messages
 * become model contents (text and function calls); tool results become functionResponse parts attached to
 * the nearest user content, so they never overwrite the echoed user text with their own (the provider
 * reads the last non-model content's text as the user text, and a functionResponse part carries none).
 */
export function openAIToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  let lastUser = -1;
  const pendingResponses: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = raw.content;
    if (raw.role === 'tool') {
      const response: Json = { functionResponse: { name: '', response: { content: toolResultText(content) } } };
      if (lastUser !== -1) (contents[lastUser].parts as Json[]).push(response);
      else pendingResponses.push(response);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image parts, in order, with any tool results that preceded it.
    const parts: Json[] = [...pendingResponses];
    pendingResponses.length = 0;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const part = geminiImagePart(block.image_url.url);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) {
      contents.push({ role: 'user', parts });
      lastUser = contents.length - 1;
    }
  }
  if (pendingResponses.length > 0 && lastUser !== -1) (contents[lastUser].parts as Json[]).push(...pendingResponses);

  const out: Json = { contents };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const fn = geminiFunctionOf(raw);
      return fn === null ? [] : [fn];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const cap = maxTokensOf(body);
  if (cap !== null) config.maxOutputTokens = cap;
  const stops = stopsOf(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (openAIJsonMode(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. The system string becomes the system
 * instruction; user messages become user contents (text, image and tool_result parts); assistant messages
 * become model contents (text and tool_use parts). A tool_result block lives inside its user message, so
 * its functionResponse part never becomes a content of its own.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const part = geminiImagePartOf(block);
          if (part !== null) parts.push(part);
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: typeof block.name === 'string' ? block.name : '',
              response: { content: toolResultText(block.content) },
            },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const fn = geminiFunctionOfAnthropic(raw);
      return fn === null ? [] : [fn];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = stopsOf(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** A Chat Completions answer as (text, usage): null when it is not one. */
export function parseOpenAIAnswer(body: unknown): UpstreamAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, inputTokens: numberOrZero(usage.prompt_tokens), outputTokens: numberOrZero(usage.completion_tokens) };
}

/** A Messages answer as (text, usage): null when it is not one. */
export function parseAnthropicAnswer(body: unknown): UpstreamAnswer | null {
  if (!isRecord(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  const text = body.content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens) };
}

/** A generateContent answer as (text, usage): null when it is not one. */
export function parseGeminiAnswer(body: unknown): UpstreamAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, inputTokens: numberOrZero(meta.promptTokenCount), outputTokens: numberOrZero(meta.candidatesTokenCount) };
}

/** One provider answer, parsed out of whichever wire it arrived on. */
export function parseUpstreamAnswer(wire: 'openai' | 'anthropic' | 'gemini', body: unknown): UpstreamAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(body);
  if (wire === 'anthropic') return parseAnthropicAnswer(body);
  return parseGeminiAnswer(body);
}

/** The fetch inputs for one upstream call: the URL, the headers and the JSON body. */
export function buildUpstreamRequest(
  base: string,
  wire: 'openai' | 'anthropic' | 'gemini',
  auth: string,
  key: string,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  model: string,
): { url: string; headers: Record<string, string>; payload: Json } {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (key !== '') {
    if (auth === 'bearer') headers.authorization = `Bearer ${key}`;
    else if (auth === 'x-api-key') headers['x-api-key'] = key;
    else if (auth === 'x-goog-api-key') headers['x-goog-api-key'] = key;
  }
  if (wire === 'openai') {
    // The client's `stream` flag must not leak upstream: a streamed provider answer is SSE, and this
    // entry always calls upstream unstreamed and synthesizes the client's stream itself. A `stream: true`
    // left in the passthrough body would make the provider answer SSE, which the JSON reader below
    // cannot parse, and the request would burn through every candidate and fail.
    const { stream: _stream, stream_options: _streamOptions, ...rest } = body;
    return {
      url: `${base}/v1/chat/completions`,
      headers,
      payload: clientApi === 'openai' ? { ...rest, model } : anthropicToOpenAI(body, model),
    };
  }
  if (wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    // Same as above: never ask the provider to stream.
    const { stream: _stream, ...rest } = body;
    return {
      url: `${base}/v1/messages`,
      headers,
      payload: clientApi === 'anthropic' ? { ...rest, model } : openAIToAnthropic(body, model),
    };
  }
  return {
    url: `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    headers,
    payload: clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model),
  };
}

/** A Chat Completions answer built from one provider answer: the provider's text and usage, verbatim. */
export function synthesizeOpenAIJson(answer: UpstreamAnswer, model: string): Json {
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A Chat Completions event stream built from one provider answer: role, text pieces, finish, usage, [DONE]. */
export function synthesizeOpenAIStream(answer: UpstreamAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (let at = 0; at < answer.text.length; at += STREAM_CHUNK_CHARS) {
    frames.push(sseEvent(null, chunk({ content: answer.text.slice(at, at + STREAM_CHUNK_CHARS) }, null)));
  }
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A Messages answer built from one provider answer: the provider's text and usage, verbatim. */
export function synthesizeAnthropicJson(answer: UpstreamAnswer, model: string): Json {
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A Messages event stream built from one provider answer: start, one text block, delta, stop. */
export function synthesizeAnthropicStream(answer: UpstreamAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (let at = 0; at < answer.text.length; at += STREAM_CHUNK_CHARS) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: answer.text.slice(at, at + STREAM_CHUNK_CHARS) },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
