# Free Model Aggregator — v2

One server that serves the trial's client as an OpenAI service (`POST /v1/chat/completions`) or an
Anthropic one (`POST /v1/messages`), in JSON or server-sent events, answering every request from one of
the ten free-tier providers in the brief, on that provider's own wire (OpenAI, Anthropic or Gemini).

v2 of the v1 entry, which scored 99/100 on the practice grade (quality 99, served, wire, failover and
catalog all 100) at 25/32/29 upstream calls over the three practice trials. 100/100 is not on the
table: the 1475 model lives on one provider whose budget tops out at 12 calls against 17 requests that
want it, so quality tops out at 99 and the score with it. The final ranks the top 60 by mean score
over three fresh instances, ties broken by fewer provider calls, so v2's job is 99 at the fewest
calls. A simulator that replicates the kit's generator and env (validated by reproducing v1's
25/32/29 exactly) puts v2 at 23/23/24 calls on the practice trials and a mean of ~23.1 over 1,000
drawn instances, against v1's ~27.6, with the score at 99 on every one of those instances.

## How it works

Per request the entry:

1. reads the request's requirements from the workload manifest (by arrival order);
2. asks the router for the free models meeting them, ranked by effective rating: the quality
   table's rating minus 150 milli-points per unit of the provider's next-call failure
   probability, computed from the kit's published budget ranges (`lib/router.ts`). A fresh
   provider's first call never refuses, so the first call to every provider is free; a second
   call to a four-call provider is dear, while a second call to a twelve-call provider is
   cheap. Past 400 milli-weight of cumulative loss the router ranks purely by rating, so a
   bad trial cannot drift into quality 97, where the score itself would fall;
3. calls the best candidate on its wire, always as one plain JSON call, and reads the answer back as
   (text, usage). A refusal's message names its cause: a spent budget or an outage marks that
   provider dead at once; a scheduled refusal is retried on the same model, because the failure
   schedules are one call wide and the next call usually lands; a second scheduled refusal in a
   row moves the request on to the next candidate. A refusal that names no cause (network error,
   unparsable answer) moves on without a retry;
4. rebuilds the reply from the provider's text and usage in the wire the client used, with the
   provider's answer text carried through verbatim. The stop reason is always the family-0 one
   (`stop` for an OpenAI client, `end_turn` for an Anthropic one).

Upstream calls are always plain JSON, never streamed: the provider's streamed and unstreamed answers
carry the same text and the same usage, and the entry synthesizes the client's stream from the parsed
answer, which keeps every streamed reply in the exact event order the grader wants.

## The router

Candidates are the free models meeting the request's requirements, ranked by effective rating
(rating minus 150 × the marginal probability the provider's next call refuses). That probability
is computed from the kit's sealed table alone: each provider's per-trial budget is drawn in a
published range, it carries one or two scheduled failures in [2, budget-1], and one provider in
ten goes out at a point in that same range. The ranking spends the safe calls first and only pays
for a risky one when the rating gap is worth it. Quality has slack to spend here: the score's
quality component floors at 99 while the weighted loss stays under 382 milli-weight, and even a
1470 in place of a 1475 costs 5 per unit of weight, so trading a little rating for a whole
avoided failure is a win on the tie-break, which is provider calls.

Two hard filters never bend: a paid model is never a candidate, and an OpenAI client request in
JSON mode never routes to an Anthropic-wire provider (that wire always echoes json_mode false,
so the echo check would fail whatever the model).

## Files

- `main.ts` — the server: the brief, the catalog, the routes, the register and the poll.
- `lib/brief.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/translate.ts` — the client wires and the three upstream wires, the answer parsers, the replies.
- `lib/router.ts` — the candidate ranking: risk-adjusted rating, loss guard, refusal causes.
- `LICENSE` — MIT.
