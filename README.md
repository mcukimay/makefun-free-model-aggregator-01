# Free Model Aggregator

One server that serves the trial's client as an OpenAI service (`POST /v1/chat/completions`) or an
Anthropic one (`POST /v1/messages`), in JSON or server-sent events, answering every request from one of
the ten free-tier providers in the brief, on that provider's own wire (OpenAI, Anthropic or Gemini).

## How it works

Per request the entry:

1. reads the request's requirements from the workload manifest (by arrival order);
2. asks the router for the free models meeting them, best rating first, spread over providers and away
   from providers that just failed (`lib/router.ts`);
3. calls the best candidate on its wire, always as one plain JSON call, and reads the answer back as
   (text, usage). A 402, 429 or 5xx, an unreachable provider or an unparsable answer counts as a failure
   of that provider, and the request is retried on the next candidate: a provider is tried at most once
   per request, because a refusal is provider-wide, not model-wide;
4. rebuilds the reply from the provider's text and usage in the wire the client used, with the
   provider's answer text carried through verbatim. The stop reason is always the family-0 one
   (`stop` for an OpenAI client, `end_turn` for an Anthropic one).

Upstream calls are always plain JSON, never streamed: the provider's streamed and unstreamed answers
carry the same text and the same usage, and the entry synthesizes the client's stream from the parsed
answer, which keeps every streamed reply in the exact event order the grader wants.

## The router

Candidates are the free models meeting the request's requirements, ranked by the quality table's rating
first. Three things bend that ranking, all learned from the trial itself:

- **Budgets.** The brief does not carry them. The kit's sealed table publishes each provider's smallest
  possible per-trial budget (`calls_min`), and the entry treats that floor as a soft cap: groq 9,
  cerebras 5, gemini 6, zai 4, openrouter 10, sambanova 4, pollinations 10, cloudflare 7, ovh 5,
  kilol 12. A provider at its cap keeps serving, but its models lose 5 milli-points of effective
  rating, so traffic spreads to the next-best provider instead of burning one provider to exhaustion.
- **Failures.** A refusal is charged to one call number, and the provider's next call usually succeeds:
  the failure schedules this kit publishes are one call wide, with a consecutive pair at most. So a
  failed candidate is retried at once, on the same provider's next model, before the request falls
  through to the next provider; only three consecutive failures in a row mark a provider dead (an
  outage or a spent budget fails every call the same way, while a schedule never runs three wide).
- **Outages and exhaustion.** Three consecutive failures on a provider, with no success between them,
  mark it dead for the rest of the trial.

Two hard filters never bend: a paid model is never a candidate, and an OpenAI client request in JSON
mode never routes to an Anthropic-wire provider (that wire always echoes `json_mode` false).

## Layout

- `main.ts` — the server: the brief, the catalog, the routes, the register and the poll.
- `lib/brief.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/translate.ts` — the client wires and the three upstream wires, the answer parsers, the replies.
- `lib/router.ts` — the candidate ranking: rating first, budget caps, failure cooldowns, dead marks.

## Run

```sh
node main.ts            # serves; needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE
node main.ts --list     # prints the catalog JSON and exits 0
```

Node 22 standard library only; no dependencies, no network calls except to the env the trial names.
