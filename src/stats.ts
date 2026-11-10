/**
 * Statistics for comparing measurement outcomes.
 *
 * The question an experiment ends on is "does the device agree with the
 * simulator?" — and the honest answer needs two numbers: how far apart the
 * distributions are, and how far apart they would be from shot noise alone.
 * This module gives both: distances and fidelities between outcome maps,
 * confidence intervals on counts, a bootstrap for any statistic, and
 * `compareOutcomes` / `compareExperiments`, which put the distance next to its
 * noise floor.
 *
 * Every function accepts `Outcomes` — a `Distribution`, a probability map, or
 * raw counts — and normalises. Bitstrings have qubit 0 leftmost.
 */

import { checkOptions } from './circuit.js'
import { isDistribution } from './experiment.js'
import type { Experiment, Outcomes } from './experiment.js'
import { makePrng } from './prng.js'

// ── Normalisation ────────────────────────────────────────────────────────────

type Probs = ReadonlyMap<string, number>

function probsOf(o: Outcomes): Readonly<Record<string, number>> {
  return isDistribution(o) ? o.probs : o
}

/** Outcomes as a normalised map, with the bitstring width they share. */
function normalise(o: Outcomes, where: string): { probs: Probs; width: number } {
  const raw = probsOf(o)
  let total = 0, width = -1
  for (const [bits, v] of Object.entries(raw)) {
    if (!/^[01]+$/.test(bits)) throw new TypeError(`${where}: '${bits}' is not a bitstring`)
    if (!(v >= 0)) throw new TypeError(`${where}: weight of '${bits}' is ${v}`)
    if (width === -1) width = bits.length
    else if (bits.length !== width) throw new TypeError(`${where}: bitstrings of mixed width (${width} and ${bits.length})`)
    total += v
  }
  if (!(total > 0)) throw new TypeError(`${where}: outcomes are empty`)
  const probs = new Map<string, number>()
  for (const [bits, v] of Object.entries(raw)) if (v > 0) probs.set(bits, v / total)
  return { probs, width }
}

function pair(p: Outcomes, q: Outcomes, where: string): { p: Probs; q: Probs; width: number } {
  const a = normalise(p, where), b = normalise(q, where)
  if (a.width !== b.width) throw new TypeError(`${where}: bitstring widths differ (${a.width} vs ${b.width})`)
  return { p: a.probs, q: b.probs, width: a.width }
}

function keys(p: Probs, q: Probs): Set<string> { return new Set([...p.keys(), ...q.keys()]) }

// ── Distances ────────────────────────────────────────────────────────────────

/**
 * Total variation distance ½·Σ|p − q| ∈ [0, 1]: the largest difference in
 * probability the two distributions assign to any event.
 */
export function tvd(p: Outcomes, q: Outcomes): number {
  const n = pair(p, q, 'tvd')
  let d = 0
  for (const k of keys(n.p, n.q)) d += Math.abs((n.p.get(k) ?? 0) - (n.q.get(k) ?? 0))
  return Math.min(1, d / 2)
}

/**
 * Classical (Hellinger) fidelity (Σ√(p·q))² ∈ [0, 1]: 1 when identical, 0
 * when disjoint. For two measurement distributions of the same circuit this is
 * the usual hardware-vs-ideal figure of merit.
 */
export function hellingerFidelity(p: Outcomes, q: Outcomes): number {
  const n = pair(p, q, 'hellingerFidelity')
  let s = 0
  for (const k of keys(n.p, n.q)) s += Math.sqrt((n.p.get(k) ?? 0) * (n.q.get(k) ?? 0))
  return Math.min(1, s * s)
}

/**
 * Kullback–Leibler divergence Σ p·ln(p/q) in nats. Infinite when `q` gives
 * zero probability to an outcome `p` can produce.
 */
export function klDivergence(p: Outcomes, q: Outcomes): number {
  const n = pair(p, q, 'klDivergence')
  let s = 0
  for (const [k, pk] of n.p) {
    const qk = n.q.get(k) ?? 0
    if (qk === 0) return Infinity
    s += pk * Math.log(pk / qk)
  }
  return Math.max(0, s)
}

/**
 * Linear cross-entropy benchmarking fidelity 2ⁿ·Σ p_ideal(x)·p_measured(x) − 1.
 * Reads 1 when the device samples the ideal distribution exactly, 0 for
 * uniform noise. `ideal` should be exact probabilities (`exactProbs()`).
 */
export function linearXeb(ideal: Outcomes, measured: Outcomes): number {
  const n = pair(ideal, measured, 'linearXeb')
  let s = 0
  for (const [k, pk] of n.p) s += pk * (n.q.get(k) ?? 0)
  return 2 ** n.width * s - 1
}

/**
 * ⟨Z…Z⟩ over the given qubits: Σ p(x)·(−1)^(parity of x on those qubits).
 * All qubits when `qubits` is omitted.
 */
export function zExpectation(outcomes: Outcomes, qubits?: readonly number[]): number {
  const { probs, width } = normalise(outcomes, 'zExpectation')
  const qs = qubits ?? Array.from({ length: width }, (_, i) => i)
  for (const q of qs)
    if (!Number.isInteger(q) || q < 0 || q >= width) throw new RangeError(`zExpectation: qubit ${q} is out of range for width ${width}`)
  let e = 0
  for (const [bits, p] of probs) {
    let parity = 0
    for (const q of qs) if (bits[q] === '1') parity ^= 1
    e += parity ? -p : p
  }
  return e
}

// ── Intervals ────────────────────────────────────────────────────────────────

export interface Interval {
  readonly low: number
  readonly high: number
}

/**
 * Wilson score interval for a proportion from `k` successes in `n` trials.
 * Well behaved at k = 0 and k = n, where the naive ±z·√(p(1−p)/n) collapses to
 * zero width. `z` = 1.96 for 95%.
 */
export function wilson(k: number, n: number, z = 1.959964): Interval {
  if (!Number.isInteger(n) || n < 1) throw new RangeError(`wilson: n must be a positive integer, got ${n}`)
  if (!(k >= 0 && k <= n)) throw new RangeError(`wilson: k must lie in [0, ${n}], got ${k}`)
  const p = k / n, z2 = z * z
  const centre = (p + z2 / (2 * n)) / (1 + z2 / n)
  const half = (z / (1 + z2 / n)) * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
  return { low: Math.max(0, centre - half), high: Math.min(1, centre + half) }
}

/**
 * Wilson interval for the probability of one bitstring in a set of outcomes.
 * `shots` is taken from a `Distribution`, else must be given.
 */
export function probInterval(outcomes: Outcomes, bits: string, options: { shots?: number; z?: number } = {}): Interval {
  checkOptions(options, ['shots', 'z'], 'probInterval')
  const shots = isDistribution(outcomes) ? outcomes.shots : options.shots
  if (shots === undefined) throw new TypeError('probInterval: shots are needed for an interval on a probability map')
  const { probs } = normalise(outcomes, 'probInterval')
  return wilson(Math.round((probs.get(bits) ?? 0) * shots), shots, options.z)
}

// ── Bootstrap ────────────────────────────────────────────────────────────────

export interface BootstrapOptions {
  /** Resamples. Default 500. */
  resamples?: number
  /** Two-sided confidence for `low`/`high`. Default 0.95. */
  confidence?: number
  seed?: number
}

export interface BootstrapResult {
  /** The statistic on the outcomes as given. */
  readonly estimate: number
  /** Mean of the statistic over resamples. */
  readonly mean: number
  /** Standard deviation of the statistic over resamples: its standard error. */
  readonly stderr: number
  /** Percentile interval at the requested confidence. */
  readonly low: number
  readonly high: number
}

/** Draw `shots` outcomes from `probs` and return the empirical probabilities. */
function resample(probs: Probs, shots: number, rng: () => number): Record<string, number> {
  const labels = [...probs.keys()], cdf: number[] = []
  let acc = 0
  for (const l of labels) { acc += probs.get(l)!; cdf.push(acc) }
  const counts = new Map<string, number>()
  for (let s = 0; s < shots; s++) {
    const u = rng() * acc
    let lo = 0, hi = cdf.length - 1
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid]! < u) lo = mid + 1; else hi = mid }
    const l = labels[lo]!
    counts.set(l, (counts.get(l) ?? 0) + 1)
  }
  const out: Record<string, number> = {}
  for (const [l, c] of counts) out[l] = c / shots
  return out
}

/**
 * Parametric bootstrap of any statistic of a measured distribution: redraw
 * `shots` outcomes from the empirical probabilities, recompute the statistic,
 * repeat. `shots` is taken from a `Distribution`, else must be given.
 *
 * @example
 * // Error bar on ⟨ZZ⟩ for a Bell state measured with 1000 shots
 * bootstrapOutcomes(bell.run({ shots: 1000 }), o => zExpectation(o))
 * // { estimate: 1, mean: 1, stderr: 0, low: 1, high: 1 }   — Bell has no parity noise
 */
export function bootstrapOutcomes(
  outcomes: Outcomes,
  statistic: (sample: Readonly<Record<string, number>>) => number,
  options: BootstrapOptions & { shots?: number } = {},
): BootstrapResult {
  checkOptions(options, ['resamples', 'confidence', 'seed', 'shots'], 'bootstrapOutcomes')
  const { resamples = 500, confidence = 0.95, seed } = options
  const shots = isDistribution(outcomes) ? outcomes.shots : options.shots
  if (shots === undefined || !Number.isInteger(shots) || shots < 1)
    throw new TypeError('bootstrapOutcomes: a positive integer shot count is needed')
  if (!Number.isInteger(resamples) || resamples < 2)
    throw new RangeError(`bootstrapOutcomes: resamples must be an integer ≥ 2, got ${resamples}`)
  if (!(confidence > 0 && confidence < 1))
    throw new RangeError(`bootstrapOutcomes: confidence must lie in (0, 1), got ${confidence}`)

  const { probs } = normalise(outcomes, 'bootstrapOutcomes')
  const estimate = statistic(Object.fromEntries(probs))
  const rng = makePrng(seed)
  const values: number[] = []
  for (let b = 0; b < resamples; b++) values.push(statistic(resample(probs, shots, rng)))
  values.sort((a, b) => a - b)
  const mean = values.reduce((s, v) => s + v, 0) / values.length
  const stderr = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1))
  const q = (f: number): number => values[Math.min(values.length - 1, Math.max(0, Math.round(f * (values.length - 1))))]!
  return { estimate, mean, stderr, low: q((1 - confidence) / 2), high: q(1 - (1 - confidence) / 2) }
}

/**
 * How far from `reference` a `shots`-shot sample of `reference` itself
 * typically lands, in total variation distance. This is the noise floor a
 * measured TVD must clear before it means anything.
 */
export function shotNoiseTvd(reference: Outcomes, shots: number, options: BootstrapOptions = {}): BootstrapResult {
  checkOptions(options, ['resamples', 'confidence', 'seed'], 'shotNoiseTvd')
  const ref = probsOf(reference)
  return bootstrapOutcomes(ref, sample => tvd(sample, ref), { ...options, shots })
}

// ── Comparison ───────────────────────────────────────────────────────────────

export interface CompareOptions extends BootstrapOptions {
  /** Shots behind `observed`, when it is a plain probability map. */
  shots?: number
}

export interface Comparison {
  readonly tvd: number
  readonly fidelity: number
  /** Expected TVD from shot noise alone — a sample of the reference against the reference — when the shot count is known. */
  readonly shotNoise: BootstrapResult | undefined
  /** Standard error of `tvd` itself, from resampling the observed distribution. */
  readonly tvdStderr: number | undefined
  /** `tvd` minus the shot-noise mean: the distance the device, not the sampling, is responsible for. */
  readonly excess: number | undefined
  /**
   * `excess` over the combined standard error √(shotNoise.stderr² + tvdStderr²).
   * Beyond ~3 the deviation is real. Both terms matter: against a reference that
   * is a single bitstring the shot-noise floor is exactly zero, and only the
   * observed sample's own uncertainty keeps the ratio finite.
   */
  readonly sigma: number | undefined
}

/**
 * Put a measured distribution next to a reference — usually hardware against
 * `exactProbs()` — and say whether the gap exceeds shot noise.
 *
 * @example
 * const ideal = bell.exactProbs()
 * const qpu   = countsToProbs(counts, 2, 1000)
 * compareOutcomes(qpu, ideal, { shots: 1000 })
 * // { tvd: 0.04, fidelity: 0.998, shotNoise: { mean: 0.018, … }, excess: 0.022, sigma: 2.4 }
 */
export function compareOutcomes(observed: Outcomes, reference: Outcomes, options: CompareOptions = {}): Comparison {
  checkOptions(options, ['shots', 'resamples', 'confidence', 'seed'], 'compareOutcomes')
  const { shots: shotsOpt, ...boot } = options
  const shots = isDistribution(observed) ? observed.shots : shotsOpt
  const d = tvd(observed, reference)
  const fidelity = hellingerFidelity(observed, reference)
  if (shots === undefined)
    return { tvd: d, fidelity, shotNoise: undefined, tvdStderr: undefined, excess: undefined, sigma: undefined }
  const shotNoise = shotNoiseTvd(reference, shots, boot)
  const ref = probsOf(reference)
  const tvdStderr = bootstrapOutcomes(observed, s => tvd(s, ref), { ...boot, shots }).stderr
  const excess = d - shotNoise.mean
  const se = Math.hypot(shotNoise.stderr, tvdStderr)
  const sigma = se > 1e-12 ? excess / se : Math.abs(excess) > 1e-12 ? Math.sign(excess) * Infinity : 0
  return { tvd: d, fidelity, shotNoise, tvdStderr, excess, sigma }
}

export interface PointComparison<P> extends Comparison {
  readonly index: number
  readonly params: P
}

export interface ExperimentComparison<P> {
  readonly points: readonly PointComparison<P>[]
  readonly meanTvd: number
  readonly maxTvd: number
  readonly meanFidelity: number
  /** Points whose `sigma` exceeds `threshold` (default 3), i.e. disagree beyond shot noise. */
  readonly beyondNoise: readonly number[]
}

/**
 * Compare two experiment records point by point: `observed` (say, a hardware
 * sweep) against `reference` (the same sweep on a simulator). Points must line
 * up — same count, same parameters in the same order. Shot noise is taken from
 * the observed points' shot counts.
 */
export function compareExperiments<P>(
  observed: Experiment<P>,
  reference: Experiment<P>,
  options: BootstrapOptions & { threshold?: number } = {},
): ExperimentComparison<P> {
  checkOptions(options, ['resamples', 'confidence', 'seed', 'threshold'], 'compareExperiments')
  const { threshold = 3, ...boot } = options
  if (observed.points.length !== reference.points.length)
    throw new TypeError(`compareExperiments: ${observed.points.length} observed points vs ${reference.points.length} reference points`)
  const points = observed.points.map((o, i) => {
    const r = reference.points[i]!
    if (JSON.stringify(o.params) !== JSON.stringify(r.params))
      throw new TypeError(`compareExperiments: parameters differ at index ${i}: ${JSON.stringify(o.params)} vs ${JSON.stringify(r.params)}`)
    const opts: CompareOptions = { ...boot }
    if (o.shots !== undefined) opts.shots = o.shots
    return { index: o.index, params: o.params, ...compareOutcomes(o.probs, r.probs, opts) }
  })
  const n = points.length
  return {
    points,
    meanTvd: n ? points.reduce((s, p) => s + p.tvd, 0) / n : 0,
    maxTvd: n ? Math.max(...points.map(p => p.tvd)) : 0,
    meanFidelity: n ? points.reduce((s, p) => s + p.fidelity, 0) / n : 1,
    beyondNoise: points.filter(p => p.sigma !== undefined && p.sigma > threshold).map(p => p.index),
  }
}
