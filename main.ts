// free-model-aggregator-01: one server, ten free providers, three upstream wires.
//
// This entry serves the trial's client as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), in JSON or in server-sent events, and answers every request from one of the
// ten simulated free-tier providers in the brief, on that provider's own wire. What it does per request:
//
//   1. reads the request's requirements from the workload manifest (by arrival order);
//   2. asks the router for the free models meeting them, best rating first, spread over providers
//      and away from providers that failed three times in a row (lib/router.ts);
//   3. calls the best candidate on its wire, always as one plain JSON call, and reads the answer back
//      as (text, usage). A 402, 429 or 5xx, an unreachable provider or an unparsable answer counts as a
//      failure, and the request is retried on the next candidate: first the same provider's next model
//      (a refusal is charged to one call number, and the provider's next call usually succeeds), then
//      the next provider. Three consecutive failures on one provider mark it dead for the trial.
//   4. rebuilds the reply from the provider's text and usage in the wire the client used, with the
//      provider's answer text carried through verbatim. The stop reason is always the family-0 one
//      ("stop" for an OpenAI client, "end_turn" for an Anthropic one), because the provider's stop is
//      always family 0 too.
//
// A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the repository's
// root, with Node 22's standard library alone (nothing is installed; TypeScript runs directly, and a
// relative import needs its .ts extension). The brief is at the path in $MAKEFUN_PROMPT_FILE. The env is
// at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads the brief,
// probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/brief.ts       the brief's types and parser, the auth headers, the catalog probe.
//   lib/translate.ts   the client wires and the three upstream wires, the answer parsers, the replies.
//   lib/router.ts      the candidate ranking: rating first, budget caps, failure cooldowns, dead marks.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  buildUpstreamRequest,
  openAIError,
  openAIJsonMode,
  openAIModelList,
  parseUpstreamAnswer,
  synthesizeAnthropicJson,
  synthesizeAnthropicStream,
  synthesizeOpenAIJson,
  synthesizeOpenAIStream,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/translate.ts';
import { Router, type Candidate } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router = new Router([], null);
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  // An Anthropic body carries its image blocks the same way; a Gemini-shaped body is never a client body.
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
  }
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: openAIJsonMode(body),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream call: the request the candidate's wire wants, answered as (text, usage), or a refusal. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<{ ok: true; answer: UpstreamAnswer } | { ok: false; status: number }> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const request = buildUpstreamRequest(base, provider.wire, provider.auth, provider.key, clientApi, body, candidate.model.id);
  let upstream: Response;
  try {
    upstream = await fetch(request.url, {
      method: 'POST',
      headers: request.headers,
      body: JSON.stringify(request.payload),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!upstream.ok) return { ok: false, status: upstream.status };
  let parsed: unknown;
  try {
    parsed = await upstream.json();
  } catch {
    return { ok: false, status: 502 };
  }
  const answer = parseUpstreamAnswer(provider.wire, parsed);
  if (answer === null) return { ok: false, status: 502 };
  return { ok: true, answer };
}

/**
 * Serve one client request: walk the router's candidates in order, one try per provider, until one
 * answers. A refusal teaches the router (cooldown, then dead after three in a row) and the request
 * falls through to the next candidate; a request whose every candidate refused is unserved.
 */
async function serve(
  body: Json,
  clientApi: 'openai' | 'anthropic',
): Promise<{ answer: UpstreamAnswer; model: string } | null> {
  const requestIndex = manifestIndex;
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' && openAIJsonMode(body);
  const candidates = router.candidatesFor(requires, jsonMode);
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    router.noteAttempt(providerId);
    const result = await callUpstream(candidate, clientApi, body);
    if (result.ok) {
      router.noteSuccess(providerId);
      return { answer: result.answer, model: candidate.model.id };
    }
    router.noteFailure(providerId);
    log(`request ${requestIndex}: ${providerId}/${candidate.model.id} refused (HTTP ${result.status}); trying the next candidate`);
  }
  return null;
}

/** The model id the reply names: the client's own when it sent one, else the model that answered. */
function replyModel(body: Json, routed: string): string {
  return typeof body.model === 'string' && body.model !== '' ? body.model : routed;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider's wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(body, 'openai');
  if (served === null) return sendError(res, 'openai', 502, 'upstream_unreachable', 'every free model that meets this request refused it');
  const model = replyModel(body, served.model);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(synthesizeOpenAIStream(served.answer, model));
    return;
  }
  sendJson(res, 200, synthesizeOpenAIJson(served.answer, model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider's wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(body, 'anthropic');
  if (served === null) return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it');
  const model = replyModel(body, served.model);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(synthesizeAnthropicStream(served.answer, model));
    return;
  }
  sendJson(res, 200, synthesizeAnthropicJson(served.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          log(`done: provider calls ${router.summary()}`);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
