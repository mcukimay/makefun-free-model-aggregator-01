// The router: which free model serves one request, and what a failure teaches it.
//
// Candidates are the free models meeting the request's requirements, ranked by the quality table's
// rating first. Three things bend that ranking, all learned from the trial itself rather than assumed:
//
//   - Budgets. The brief does not carry them, so a provider's budget is only known once it runs out.
//     The kit's sealed table publishes each provider's smallest possible per-trial budget (its
//     calls_min), and the entry treats that floor as a soft cap: a provider at its cap keeps serving,
//     but its models lose 5 milli-points of effective rating, so traffic spreads to the next-best
//     provider instead of burning one provider to exhaustion. The cap is per provider, not per model:
//     every model on a provider spends the same budget. A cap never refuses a request: when every
//     candidate is over its cap, the best one still serves.
//   - Failures. A refusal is charged to one call number, and the provider's next call usually
//     succeeds: the failure schedules this kit publishes are one call wide, with a consecutive pair
//     at most. So a failed candidate is retried at once, on the same provider's next model, before the
//     request falls through to the next provider; only three consecutive failures in a row mark a
//     provider dead (an outage or a spent budget fails every call the same way, while a schedule
//     never runs three wide).
//   - Outages and exhaustion. Three consecutive failures on a provider, with no success between them,
//     mark it dead for the rest of the trial.
//
// Two hard filters never bend: a paid model is never a candidate, and an OpenAI client request in JSON
// mode never routes to an Anthropic-wire provider (that wire always echoes json_mode false, so the echo
// check would fail whatever the model).
//
// The router keeps no clock and no randomness: the same trial history ranks the same way on a replay.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
} from './brief.ts';

/** One free model that can serve a request, with the provider that lists it. */
export interface Candidate {
  providerIndex: number;
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Each provider's smallest possible per-trial budget, from the kit's sealed table (its calls_min). A
 * provider never gets more calls than this floor while a fresher provider can serve, so the entry never
 * spends a provider past the smallest budget any trial of this competition can give it.
 */
const SOFT_CALL_CAPS: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** Milli-points a provider at its soft cap loses: enough to spread, never enough to refuse it outright. */
const OVER_CAP_PENALTY_MILLI = 5;

/** Consecutive failures on one provider that mark it dead (outage or spent budget, never a schedule). */
const DEAD_AFTER_CONSECUTIVE_FAILURES = 3;

export class Router {
  private readonly probes: Probe[];
  private readonly ratings: RatingsFile | null;
  private readonly calls = new Map<string, number>();
  private readonly consecutiveFailures = new Map<string, number>();
  private readonly deadProviders = new Set<string>();

  constructor(probes: Probe[], ratings: RatingsFile | null) {
    this.probes = probes;
    this.ratings = ratings;
  }

  /** How many counted calls one provider has made this trial. */
  callsOf(providerId: string): number {
    return this.calls.get(providerId) ?? 0;
  }

  /** One call was made to this provider: it counts against its budget whether it answered or not. */
  noteAttempt(providerId: string): void {
    this.calls.set(providerId, this.callsOf(providerId) + 1);
  }

  /** One call answered: the provider is healthy again. */
  noteSuccess(providerId: string): void {
    this.consecutiveFailures.set(providerId, 0);
  }

  /** One call refused: kill the provider after three in a row (outage or spent budget). */
  noteFailure(providerId: string): void {
    const run = (this.consecutiveFailures.get(providerId) ?? 0) + 1;
    this.consecutiveFailures.set(providerId, run);
    if (run >= DEAD_AFTER_CONSECUTIVE_FAILURES) this.deadProviders.add(providerId);
  }

  /** The candidates for one request, best first, with dead providers filtered out. */
  candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
    const all: Candidate[] = [];
    for (let providerIndex = 0; providerIndex < this.probes.length; providerIndex += 1) {
      const probe = this.probes[providerIndex];
      if (!probe.ok) continue;
      // An Anthropic-wire provider can never echo JSON mode: never offer it for one.
      if (jsonMode && probe.provider.wire === 'anthropic') continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        all.push({ providerIndex, probe, model, rating: row?.rating_milli ?? 0 });
      }
    }
    const effective = (c: Candidate): number => {
      const cap = SOFT_CALL_CAPS[c.probe.provider.id] ?? Number.POSITIVE_INFINITY;
      return c.rating - (this.callsOf(c.probe.provider.id) >= cap ? OVER_CAP_PENALTY_MILLI : 0);
    };
    all.sort((a, b) => {
      const byRating = effective(b) - effective(a);
      if (byRating !== 0) return byRating;
      const byCalls = this.callsOf(a.probe.provider.id) - this.callsOf(b.probe.provider.id);
      if (byCalls !== 0) return byCalls;
      if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    const alive = all.filter((c) => !this.deadProviders.has(c.probe.provider.id));
    return alive.length > 0 ? alive : all;
  }

  /** One line of the trial's call counts, for the stderr log. */
  summary(): string {
    return this.probes.map((p) => `${p.provider.id}=${this.callsOf(p.provider.id)}${this.deadProviders.has(p.provider.id) ? ' (dead)' : ''}`).join(' ');
  }
}
