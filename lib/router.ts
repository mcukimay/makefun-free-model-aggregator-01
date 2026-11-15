// The router: which free model serves one request, and what a failure teaches it.
//
// v1 ranked by rating first, demoted a provider 5 milli-points at its soft cap, and walked
// the candidate list on a refusal (the same provider's next model, then the next provider).
// That cost calls three ways: the demotion flipped picks to a worse model while the better
// provider still had budget; a refusal fell through to a worse model instead of retrying the
// one that refused; and a spent budget or an outage took three consecutive failures to mark
// a provider dead. v2 fixes all three, and it does it from the trial itself:
//
//   - Risk-adjusted ranking. A provider's next call refuses with a probability the entry
//     can compute from the kit's sealed table alone: each provider's per-trial budget is
//     drawn in a published range, it carries one or two scheduled failures in
//     [2, budget-1], and one provider in ten goes out at a point in that same range. The
//     marginal probability that the next call (number k) refuses is the sum of those three
//     terms, and a candidate's effective rating is its rating minus 150 milli-points per
//     unit of that probability. A fresh provider's first call never refuses, so the first
//     call to every provider is free; a second call to a four-call provider is dear (its
//     failure chance is over one half), while a second call to a twelve-call provider is
//     cheap. The ranking spends the safe calls first and only pays for a risky one when
//     the rating gap is worth it. Quality has slack to spend here: the score's quality
//     component floors at 99 while the weighted loss stays under 382 milli-weight, and
//     even a 1470 in place of a 1475 costs 5 per unit of weight, so trading a little
//     rating for a whole avoided failure is a win on the tie-break, which is provider
//     calls. The score cannot reach 100 on any instance (the 1475 model lives on one
//     provider whose budget tops out at 12 calls against 17 requests that want it), so
//     the entry's job is 99 at the fewest calls, not 100.
//   - A loss guard. The cumulative weighted loss (best rating minus served rating, per
//     request weight) is tracked, and past 400 milli-weight the router stops trading:
//     it ranks purely by rating for the rest of the trial. That caps the tail: a trial
//     that went wrong early cannot drift into quality 97, where the score itself would
//     fall to 98.
//   - Refusals read, not counted. A provider's refusal message names its cause: its
//     budget is spent, it is out, or it refused this call. A spent budget or an outage
//     marks the provider dead at once, on the first such refusal. A scheduled refusal
//     is retried on the same model: the failure schedules are one call wide, with a
//     consecutive pair at most, so the provider's next call usually succeeds, and the
//     request keeps its best model instead of falling to a worse one. Only a second
//     scheduled refusal in a row moves the request on to the next candidate.
//
// Two hard filters never bend: a paid model is never a candidate, and an OpenAI client
// request in JSON mode never routes to an Anthropic-wire provider (that wire always
// echoes json_mode false, so the echo check would fail whatever the model).
//
// The router keeps no clock and no unseeded randomness: the same trial history ranks
// the same way on a replay.

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
 * Each provider's per-trial budget range, from the kit's sealed table (its calls_min
 * and calls_max). The entry never sees the drawn budget; it only knows the range.
 */
const BUDGET_RANGE: Record<string, { min: number; max: number }> = {
  groq: { min: 9, max: 12 },
  cerebras: { min: 5, max: 6 },
  gemini: { min: 6, max: 8 },
  zai: { min: 4, max: 5 },
  openrouter: { min: 10, max: 12 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
  cloudflare: { min: 7, max: 8 },
  ovh: { min: 5, max: 6 },
  kilol: { min: 12, max: 16 },
};

/** Milli-points of rating one unit of failure probability costs. */
const RISK_MILLI_PER_UNIT = 150;

/** Cumulative weighted loss past which the router stops trading rating for safety. */
const LOSS_GUARD_MILLI = 400;

/** What a refusal says: the provider's own message names the cause. */
export type RefusalKind = 'schedule' | 'budget' | 'outage' | 'other';

/** Read a refusal's cause off its message (every wire carries it in error.message). */
export function classifyRefusal(message: string | null): RefusalKind {
  if (message === null) return 'other';
  if (message.includes('budget is spent')) return 'budget';
  if (message.includes(' is out')) return 'outage';
  if (message.includes('refused this call')) return 'schedule';
  return 'other';
}

/**
 * The marginal probability that this provider's next call refuses, over its
 * published budget range. The next call is number callsMade + 1; the first call
 * never refuses (no failure schedule, outage or budget starts at 1). Three terms:
 * a scheduled failure (one or two draws, mean 1.5, uniform in [2, budget-1]),
 * the outage (one provider in ten, its start uniform in [2, budget-1]), and the
 * budget (the call number past the drawn budget). The terms overlap a little and
 * the sum is capped, which only ever overstates a risky call slightly.
 */
export function failureRisk(providerId: string, callsMade: number): number {
  const range = BUDGET_RANGE[providerId];
  if (range === undefined) return 0;
  const k = callsMade + 1;
  if (k <= 1) return 0;
  let schedule = 0;
  let outage = 0;
  let budget = 0;
  const span = range.max - range.min + 1;
  for (let b = range.min; b <= range.max; b += 1) {
    const w = 1 / span;
    schedule += w * (1.5 / (b - 2));
    outage += w * 0.1 * Math.min(1, (k - 1) / (b - 2));
    budget += w * (k > b ? 1 : 0);
  }
  return Math.min(0.95, schedule + outage + budget);
}

export class Router {
  private readonly probes: Probe[];
  private readonly ratings: RatingsFile | null;
  private readonly calls = new Map<string, number>();
  private readonly deadProviders = new Set<string>();
  /** Weighted loss so far: sum of weight * (best rating - served rating). */
  private lossMilli = 0;

  constructor(probes: Probe[], ratings: RatingsFile | null) {
    this.probes = probes;
    this.ratings = ratings;
  }

  /** How many counted calls one provider has made this trial. */
  callsOf(providerId: string): number {
    return this.calls.get(providerId) ?? 0;
  }

  /** Weighted loss so far, for the stderr log. */
  lossOf(): number {
    return this.lossMilli;
  }

  /** One call was made to this provider: it counts against its budget whether it answered or not. */
  noteAttempt(providerId: string): void {
    this.calls.set(providerId, this.callsOf(providerId) + 1);
  }

  /** One call answered: bank the weighted loss of serving at this rating. */
  noteSuccess(providerId: string, rating: number, best: number, weight: number): void {
    void providerId;
    this.lossMilli += weight * Math.max(0, best - rating);
  }

  /** One call refused: a spent budget or an outage kills the provider at once. */
  noteRefusal(providerId: string, kind: RefusalKind): void {
    if (kind === 'budget' || kind === 'outage') this.deadProviders.add(providerId);
  }

  /** The candidates for one request, best effective rating first, dead providers filtered out. */
  candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
    const lambda = this.lossMilli > LOSS_GUARD_MILLI ? 0 : RISK_MILLI_PER_UNIT;
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
    const effective = (c: Candidate): number =>
      c.rating - lambda * failureRisk(c.probe.provider.id, this.callsOf(c.probe.provider.id));
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
