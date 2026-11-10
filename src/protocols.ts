/**
 * Hardware characterisation protocols: the CHSH inequality and randomized
 * benchmarking.
 *
 * Each protocol is split into three pure pieces — circuit generation, analysis
 * of outcomes, and a runner that threads an executor through both — so one
 * protocol definition runs unchanged against any local backend or a QPU. The
 * circuits carry no explicit measurement: every shot sampler in ket, and the
 * IonQ API, read out all qubits at the end, so the generated circuits are
 * submittable as they are.
 *
 * Qubit 0 is the leftmost character of every bitstring, as everywhere in ket.
 */

import { Circuit, checkOptions } from './circuit.js'
import { isDistribution, resolveOutcomes } from './experiment.js'
import type { Outcomes, Executor } from './experiment.js'
import { CliffordSim } from './clifford.js'
import { makePrng } from './prng.js'

// ── Shared ───────────────────────────────────────────────────────────────────

export type { Outcomes, Executor, ExecutorResult, RunReport } from './experiment.js'

function probsOf(o: Outcomes): Readonly<Record<string, number>> {
  return isDistribution(o) ? o.probs : o
}

function shotsOf(o: Outcomes): number | undefined {
  return isDistribution(o) ? o.shots : undefined
}

/** Σ p(s)·f(s) over the outcomes, normalised so an un-normalised histogram still works. */
function expectOver(o: Outcomes, where: string, f: (bits: string) => number): number {
  let num = 0, den = 0
  for (const [bits, p] of Object.entries(probsOf(o))) {
    num += p * f(bits)
    den += p
  }
  if (den <= 0) throw new TypeError(`${where}: outcomes are empty`)
  return num / den
}

function checkQubit(q: number, width: number, where: string): void {
  if (!Number.isInteger(q) || q < 0 || q >= width)
    throw new RangeError(`${where}: qubit ${q} is out of range for width ${width}`)
}

// ── CHSH ─────────────────────────────────────────────────────────────────────

/**
 * Measurement angles for a CHSH experiment, in radians, each selecting the
 * observable cos θ·Z + sin θ·X. `a`/`a2` are Alice's two settings, `b`/`b2`
 * Bob's.
 */
export interface ChshAngles {
  readonly a: number
  readonly a2: number
  readonly b: number
  readonly b2: number
}

/** The angles that maximise S for |Φ⁺⟩: a = 0, a' = π/2, b = π/4, b' = 3π/4 → S = 2√2. */
export const CHSH_OPTIMAL: ChshAngles = Object.freeze({
  a: 0, a2: Math.PI / 2, b: Math.PI / 4, b2: 3 * Math.PI / 4,
})

export interface ChshOptions {
  /** The pair measured — Alice's qubit then Bob's. Default `[0, 1]`. */
  qubits?: readonly [number, number]
  /** Total qubits in each circuit. Default: one past the highest index in `qubits`. */
  width?: number
  /** Measurement settings. Default {@link CHSH_OPTIMAL}. */
  angles?: ChshAngles
}

export interface ChshResult {
  /** S = E(a,b) − E(a,b') + E(a',b) + E(a',b'). Local hidden variables bound |S| ≤ 2; quantum mechanics reaches 2√2. */
  readonly S: number
  /** The four correlators, in the order of {@link chshCircuits}. */
  readonly correlations: readonly [number, number, number, number]
  /** Standard error on S from shot noise, or `undefined` when the shot count is unknown. */
  readonly stderr: number | undefined
  /** How many standard errors S lies above the classical bound of 2. Negative when it does not. */
  readonly sigma: number | undefined
}

/**
 * The four CHSH circuits, in the order (a,b), (a,b'), (a',b), (a',b').
 *
 * Each prepares |Φ⁺⟩ on the pair and rotates both qubits so a computational
 * basis readout measures the chosen observable: Ry(−θ) followed by Z measures
 * cos θ·Z + sin θ·X.
 *
 * @example
 * const [ab] = chshCircuits()
 * ab.exactProbs()  // { '00': 0.427, '11': 0.427, '01': 0.073, '10': 0.073 }
 */
export function chshCircuits(options: ChshOptions = {}): readonly Circuit[] {
  checkOptions(options, ['qubits', 'width', 'angles'], 'chshCircuits')
  const [qa, qb] = options.qubits ?? [0, 1]
  const width = options.width ?? Math.max(qa, qb) + 1
  checkQubit(qa, width, 'chshCircuits'); checkQubit(qb, width, 'chshCircuits')
  if (qa === qb) throw new TypeError(`chshCircuits: the two qubits must differ (got ${qa})`)
  const { a, a2, b, b2 } = options.angles ?? CHSH_OPTIMAL
  const settings: readonly [number, number][] = [[a, b], [a, b2], [a2, b], [a2, b2]]
  return settings.map(([ta, tb]) =>
    new Circuit(width).h(qa).cnot(qa, qb).ry(-ta, qa).ry(-tb, qb),
  )
}

/** The S value an ideal |Φ⁺⟩ gives for these angles: Σ ± cos(θₐ − θ_b). */
export function chshIdeal(angles: ChshAngles = CHSH_OPTIMAL): number {
  const { a, a2, b, b2 } = angles
  return Math.cos(a - b) - Math.cos(a - b2) + Math.cos(a2 - b) + Math.cos(a2 - b2)
}

/**
 * Compute S from the outcomes of the four {@link chshCircuits}, in the same
 * order. `qubits` must match the pair the circuits were built for.
 *
 * `shots` is only needed for error bars when the outcomes are plain
 * probability maps; a `Distribution` carries its own.
 */
export function chshAnalyze(
  outcomes: readonly Outcomes[],
  options: { qubits?: readonly [number, number]; shots?: number } = {},
): ChshResult {
  checkOptions(options, ['qubits', 'shots'], 'chshAnalyze')
  if (outcomes.length !== 4)
    throw new TypeError(`chshAnalyze: expected outcomes for 4 settings, got ${outcomes.length}`)
  const [qa, qb] = options.qubits ?? [0, 1]
  const E = outcomes.map((o, i) => expectOver(o, `chshAnalyze: setting ${i}`, bits => {
    const x = bits[qa], y = bits[qb]
    if (x === undefined || y === undefined)
      throw new RangeError(`chshAnalyze: bitstring '${bits}' is too short for qubits ${qa}, ${qb}`)
    return x === y ? 1 : -1
  })) as [number, number, number, number]
  const S = E[0] - E[1] + E[2] + E[3]

  // Each correlator is a mean of ±1 samples, so Var(E) = (1 − E²)/N; the
  // settings are independent experiments and their variances add.
  let variance = 0, known = true
  outcomes.forEach((o, i) => {
    const n = shotsOf(o) ?? options.shots
    if (n === undefined || n <= 0) { known = false; return }
    variance += (1 - E[i]! ** 2) / n
  })
  const stderr = known ? Math.sqrt(variance) : undefined
  const sigma  = stderr === undefined ? undefined
               : stderr > 0           ? (S - 2) / stderr
               : S > 2 ? Infinity : S < 2 ? -Infinity : 0
  return { S, correlations: E, stderr, sigma }
}

/**
 * Generate, run and analyse a CHSH experiment in one call.
 *
 * @example
 * // Local, noiseless — S → 2√2
 * await runChsh(c => c.run({ shots: 4096 }))
 *
 * // Same experiment on hardware
 * await runChsh(async c => {
 *   const { counts, shots } = await runIonQ(c.toIonQ(), { apiKey, target: 'qpu.forte-1', shots: 1000 })
 *   return countsToProbs(counts, c.qubits, shots)
 * }, { shots: 1000 })
 */
export async function runChsh(
  execute: Executor,
  options: ChshOptions & { shots?: number } = {},
): Promise<ChshResult & { readonly circuits: readonly Circuit[] }> {
  checkOptions(options, ['qubits', 'width', 'angles', 'shots'], 'runChsh')
  const { shots, ...rest } = options
  const circuits = chshCircuits(rest)
  const outcomes: Outcomes[] = []
  for (let i = 0; i < circuits.length; i++) outcomes.push(resolveOutcomes(await execute(circuits[i]!, i)))
  const analyzeOpts: { qubits?: readonly [number, number]; shots?: number } = {}
  if (rest.qubits !== undefined) analyzeOpts.qubits = rest.qubits
  if (shots !== undefined) analyzeOpts.shots = shots
  return { ...chshAnalyze(outcomes, analyzeOpts), circuits }
}

// ── Clifford group ───────────────────────────────────────────────────────────

type CGate = { readonly name: 'h' | 's' | 'sdg'; readonly a: number }
           | { readonly name: 'cnot'; readonly a: number; readonly b: number }

interface CliffordElement {
  readonly word: readonly CGate[]
  readonly key:  string
}

function applyToSim(sim: CliffordSim, g: CGate): void {
  switch (g.name) {
    case 'h':    sim.h(g.a); break
    case 's':    sim.s(g.a); break
    case 'sdg':  sim.si(g.a); break
    case 'cnot': sim.cnot(g.a, g.b); break
  }
}

function applyToCircuit(c: Circuit, g: CGate, map: readonly number[]): Circuit {
  const a = map[g.a]!
  switch (g.name) {
    case 'h':    return c.h(a)
    case 's':    return c.s(a)
    case 'sdg':  return c.sdg(a)
    case 'cnot': return c.cnot(a, map[g.b]!)
  }
}

function inverseWord(word: readonly CGate[]): CGate[] {
  return [...word].reverse().map(g =>
    g.name === 's'   ? { name: 'sdg', a: g.a } :
    g.name === 'sdg' ? { name: 's',   a: g.a } : g)
}

function tableauKey(sim: CliffordSim): string { return sim.tableau().join('') }

const GROUP_CACHE = new Map<number, { elements: readonly CliffordElement[]; index: ReadonlyMap<string, number> }>()

/**
 * Enumerate the n-qubit Clifford group (mod phase) by breadth-first search
 * over {H, S, S†, CNOT}, keyed on the stabilizer tableau. Each element's word
 * is a shortest one found by the search, which bounds the inverse step of an
 * RB sequence to a fixed small depth.
 */
function cliffordElements(n: number): { elements: readonly CliffordElement[]; index: ReadonlyMap<string, number> } {
  const cached = GROUP_CACHE.get(n)
  if (cached) return cached
  if (n !== 1 && n !== 2)
    throw new RangeError(`Clifford group enumeration supports 1 or 2 qubits, got ${n}`)

  const gens: CGate[] = []
  for (let q = 0; q < n; q++) gens.push({ name: 'h', a: q }, { name: 's', a: q }, { name: 'sdg', a: q })
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) if (a !== b) gens.push({ name: 'cnot', a, b })

  const elements: CliffordElement[] = []
  const index = new Map<string, number>()
  const push = (word: readonly CGate[]): void => {
    const sim = new CliffordSim(n)
    for (const g of word) applyToSim(sim, g)
    const key = tableauKey(sim)
    if (index.has(key)) return
    index.set(key, elements.length)
    elements.push({ word, key })
  }
  push([])
  for (let head = 0; head < elements.length; head++) {
    const { word } = elements[head]!
    for (const g of gens) push([...word, g])
  }
  const out = { elements, index }
  GROUP_CACHE.set(n, out)
  return out
}

/**
 * Every element of the 1- or 2-qubit Clifford group (mod global phase) as a
 * circuit over {H, S, S†, CNOT}: 24 elements for one qubit, 11,520 for two.
 * Order is deterministic and shortest-word-first.
 *
 * @example
 * cliffordGroup(1).length  // 24
 * cliffordGroup(2).length  // 11520
 */
export function cliffordGroup(qubits: 1 | 2): readonly Circuit[] {
  const map = qubits === 1 ? [0] : [0, 1]
  return cliffordElements(qubits).elements.map(e =>
    e.word.reduce((c, g) => applyToCircuit(c, g, map), new Circuit(qubits)))
}

// ── Randomized benchmarking ──────────────────────────────────────────────────

export interface RbOptions {
  /** The qubit, or pair of qubits, under test. Default `[0]`. */
  qubits?: readonly number[]
  /** Total qubits in each circuit. Default: one past the highest index in `qubits`. */
  width?: number
  /** Clifford sequence lengths m to sample at. At least three distinct values are needed to fit. */
  lengths: readonly number[]
  /** Random sequences per length. Default 10. */
  samples?: number
  /** Seed for the sequence generator; sequences are reproducible given the seed. */
  seed?: number
}

export interface RbSequence {
  /** Number of random Cliffords before the inverting one. */
  readonly length: number
  /** Index of this sequence among those at the same length. */
  readonly sample: number
  readonly circuit: Circuit
}

/**
 * Standard randomized benchmarking sequences: m uniformly random Cliffords
 * followed by the single Clifford that inverts their product, so an ideal
 * device returns the benchmarked qubits to |0…0⟩ with certainty.
 *
 * Cliffords are drawn uniformly from the full group (24 or 11,520 elements),
 * and the inverse is looked up rather than synthesised, so every sequence
 * ends in one bounded-depth Clifford regardless of m.
 *
 * @example
 * const seqs = rbSequences({ lengths: [1, 2, 4, 8, 16, 32], samples: 20, seed: 1 })
 * seqs.length                       // 120
 * seqs[0]!.circuit.exactProbs()     // { '0': 1 } — noiseless, the inverse undoes everything
 */
/** The benchmarked qubits and register width an `RbOptions` denotes — one resolution shared by `rbSequences` and `runRb`. */
function resolveRbQubits(options: RbOptions, where: string): { qs: readonly number[]; n: 1 | 2; width: number } {
  const qs = options.qubits ?? [0]
  const n = qs.length
  if (n !== 1 && n !== 2)
    throw new RangeError(`${where}: benchmark 1 or 2 qubits, got ${n}`)
  if (new Set(qs).size !== n)
    throw new TypeError(`${where}: qubits must be distinct, got [${qs.join(', ')}]`)
  const width = options.width ?? Math.max(...qs) + 1
  for (const q of qs) checkQubit(q, width, where)
  return { qs, n, width }
}

const RB_OPTION_KEYS = ['qubits', 'width', 'lengths', 'samples', 'seed'] as const

export function rbSequences(options: RbOptions): readonly RbSequence[] {
  checkOptions(options, RB_OPTION_KEYS, 'rbSequences')
  const { qs, n, width } = resolveRbQubits(options, 'rbSequences')
  const { lengths, samples = 10, seed } = options
  if (lengths.length === 0) throw new TypeError('rbSequences: lengths must be non-empty')
  for (const m of lengths)
    if (!Number.isInteger(m) || m < 1) throw new RangeError(`rbSequences: lengths must be integers ≥ 1, got ${m}`)
  if (!Number.isInteger(samples) || samples < 1)
    throw new RangeError(`rbSequences: samples must be an integer ≥ 1, got ${samples}`)

  const { elements, index } = cliffordElements(n)
  const rng = makePrng(seed)
  const out: RbSequence[] = []
  for (const length of lengths) {
    for (let sample = 0; sample < samples; sample++) {
      let circuit = new Circuit(width)
      const net = new CliffordSim(n)
      for (let i = 0; i < length; i++) {
        const { word } = elements[Math.floor(rng() * elements.length)]!
        for (const g of word) {
          applyToSim(net, g)
          circuit = applyToCircuit(circuit, g, qs)
        }
      }
      const netIdx = index.get(tableauKey(net))
      if (netIdx === undefined) throw new Error('rbSequences: composed Clifford missing from the group table')
      for (const g of inverseWord(elements[netIdx]!.word)) circuit = applyToCircuit(circuit, g, qs)
      out.push({ length, sample, circuit })
    }
  }
  return out
}

/**
 * Probability that every benchmarked qubit read 0 — the RB survival
 * probability. Other qubits in the bitstring are ignored.
 */
export function rbSurvival(outcomes: Outcomes, qubits: readonly number[] = [0]): number {
  return expectOver(outcomes, 'rbSurvival', bits => {
    for (const q of qubits) {
      const b = bits[q]
      if (b === undefined) throw new RangeError(`rbSurvival: bitstring '${bits}' is too short for qubit ${q}`)
      if (b !== '0') return 0
    }
    return 1
  })
}

export interface RbPoint {
  readonly length: number
  readonly survival: number
}

export interface RbFitOptions {
  /** Number of qubits benchmarked — sets the dimension in r = (1 − p)(1 − 2⁻ⁿ). Default 1. */
  qubits?: number
  /** Bootstrap resamples for the error bar on `errorPerClifford`. Default 200; 0 disables. */
  bootstrap?: number
  /** Seed for the bootstrap. */
  seed?: number
  /**
   * Hold the asymptote B at this value instead of fitting it — typically 2⁻ⁿ,
   * the ideal-SPAM value. Use it when the lengths do not reach the plateau:
   * over a short range A·pᵐ + B is nearly linear, and the data cannot
   * separate a small decay with B ≈ ½ from a smaller one with B ≈ 0. The
   * bootstrap error bar does not see that ambiguity, since it is a property
   * of the model, not of the sample.
   */
  fixB?: number
}

export interface RbFit {
  /** Fitted depolarising parameter per Clifford. */
  readonly p: number
  /** Fitted amplitude: survival(m) = A·pᵐ + B. */
  readonly A: number
  /** Fitted asymptote. Ideal SPAM gives A = 1 − 2⁻ⁿ, B = 2⁻ⁿ. */
  readonly B: number
  /** Average error per Clifford, r = (1 − p)(1 − 2⁻ⁿ). */
  readonly errorPerClifford: number
  /** Bootstrap standard error on `errorPerClifford`, or `undefined` when bootstrapping was disabled. */
  readonly errorPerCliffordStderr: number | undefined
  /** Average Clifford fidelity, 1 − r. */
  readonly fidelity: number
  /** Number of qubits the fit assumed. */
  readonly qubits: number
  /** Mean survival at each length with its standard error over sequences. */
  readonly points: readonly { readonly length: number; readonly mean: number; readonly stderr: number; readonly samples: number }[]
}

/**
 * Least-squares A, B for fixed p with both held in [0, 1], and the residual
 * sum of squares. The box matters: a decay that looks linear over the sampled
 * lengths is fit unconstrained by p → 1 with A → +∞ and B → −∞, the Taylor
 * limit of A·pᵐ + B. A survival probability has A, B ∈ [0, 1], and inside that
 * box the same data picks out the genuine p. The objective is convex, so the
 * constrained minimum is the unconstrained one when feasible, and otherwise lies
 * on one of the four edges, each a clamped one-dimensional fit.
 */
function linearFit(pts: readonly RbPoint[], p: number, fixB?: number): { A: number; B: number; sse: number } {
  const n = pts.length
  let sx = 0, sy = 0, sxx = 0, sxy = 0
  for (const { length, survival } of pts) {
    const x = p ** length
    sx += x; sy += survival; sxx += x * x; sxy += x * survival
  }
  const sse = (A: number, B: number): number => {
    let e = 0
    for (const { length, survival } of pts) e += (survival - A * p ** length - B) ** 2
    return e
  }
  const clamp = (v: number): number => Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0))

  if (fixB !== undefined) {
    const A = sxx > 0 ? clamp((sxy - fixB * sx) / sxx) : 0
    return { A, B: fixB, sse: sse(A, fixB) }
  }

  const det = n * sxx - sx * sx
  if (Math.abs(det) >= 1e-12 * Math.max(1, n * sxx)) {
    const A = (n * sxy - sx * sy) / det, B = (sy - A * sx) / n
    if (A >= 0 && A <= 1 && B >= 0 && B <= 1) return { A, B, sse: sse(A, B) }
  }
  const edges: [number, number][] = [
    [0, clamp(sy / n)],                                   // A = 0
    [1, clamp((sy - sx) / n)],                            // A = 1
    [sxx > 0 ? clamp(sxy / sxx) : 0, 0],                  // B = 0
    [sxx > 0 ? clamp((sxy - sx) / sxx) : 0, 1],           // B = 1
  ]
  let best = edges[0]!, bestSse = Infinity
  for (const e of edges) {
    const v = sse(e[0], e[1])
    if (v < bestSse) { bestSse = v; best = e }
  }
  return { A: best[0], B: best[1], sse: bestSse }
}

/**
 * Fit A·pᵐ + B by exhaustive search on p ∈ [0, 1] followed by golden-section
 * refinement. The scan runs downward from p = 1 so that a curve with no
 * visible decay — every p fits it equally well — is reported as p = 1, no
 * error, rather than as whichever end of the interval came first.
 */
function fitDecay(pts: readonly RbPoint[], fixB?: number): { p: number; A: number; B: number } {
  let best = 1, bestSse = Infinity
  for (let i = 1000; i >= 0; i--) {
    const p = i / 1000
    const { sse } = linearFit(pts, p, fixB)
    if (sse < bestSse) { bestSse = sse; best = p }
  }
  let lo = Math.max(0, best - 0.001), hi = Math.min(1, best + 0.001)
  const phi = (Math.sqrt(5) - 1) / 2
  let c = hi - phi * (hi - lo), d = lo + phi * (hi - lo)
  let fc = linearFit(pts, c, fixB).sse, fd = linearFit(pts, d, fixB).sse
  for (let i = 0; i < 40; i++) {
    if (fc < fd) { hi = d; d = c; fd = fc; c = hi - phi * (hi - lo); fc = linearFit(pts, c, fixB).sse }
    else         { lo = c; c = d; fc = fd; d = lo + phi * (hi - lo); fd = linearFit(pts, d, fixB).sse }
  }
  const p = (lo + hi) / 2
  const { A, B } = linearFit(pts, p, fixB)
  return { p, A, B }
}

/**
 * Fit the RB decay survival(m) = A·pᵐ + B to per-sequence survival
 * probabilities and report the error per Clifford.
 *
 * Pass every sequence's point rather than per-length means: the fit weights
 * lengths by how many sequences they have, and the bootstrap resamples
 * sequences within each length.
 */
export function rbFit(points: readonly RbPoint[], options: RbFitOptions = {}): RbFit {
  checkOptions(options, ['qubits', 'bootstrap', 'seed', 'fixB'], 'rbFit')
  const { qubits = 1, bootstrap = 200, seed, fixB } = options
  if (!Number.isInteger(qubits) || qubits < 1)
    throw new RangeError(`rbFit: qubits must be a positive integer, got ${qubits}`)
  if (fixB !== undefined && !(fixB >= 0 && fixB <= 1))
    throw new RangeError(`rbFit: fixB must lie in [0, 1], got ${fixB}`)
  const byLength = new Map<number, number[]>()
  for (const { length, survival } of points) {
    if (!Number.isInteger(length) || length < 0) throw new TypeError(`rbFit: length must be a non-negative integer, got ${length}`)
    if (!Number.isFinite(survival)) throw new TypeError(`rbFit: survival at m=${length} is not a number`)
    const arr = byLength.get(length) ?? []
    arr.push(survival)
    byLength.set(length, arr)
  }
  const needed = fixB === undefined ? 3 : 2
  if (byLength.size < needed)
    throw new TypeError(`rbFit: need at least ${needed} distinct lengths to fit A·pᵐ${fixB === undefined ? ' + B' : ''}, got ${byLength.size}`)

  const scale = 1 - 2 ** -qubits
  const { p, A, B } = fitDecay(points, fixB)
  const r = (1 - p) * scale

  let stderr: number | undefined
  if (bootstrap > 0) {
    const rng = makePrng(seed)
    const groups = [...byLength.entries()]
    const rs: number[] = []
    for (let b = 0; b < bootstrap; b++) {
      const sample: RbPoint[] = []
      for (const [length, arr] of groups)
        for (let i = 0; i < arr.length; i++) sample.push({ length, survival: arr[Math.floor(rng() * arr.length)]! })
      rs.push((1 - fitDecay(sample, fixB).p) * scale)
    }
    const mean = rs.reduce((s, v) => s + v, 0) / rs.length
    stderr = Math.sqrt(rs.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, rs.length - 1))
  }

  const summary = [...byLength.entries()].sort((a, b) => a[0] - b[0]).map(([length, arr]) => {
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length
    const variance = arr.length > 1 ? arr.reduce((s, v) => s + (v - mean) ** 2, 0) / (arr.length - 1) : 0
    return { length, mean, stderr: Math.sqrt(variance / arr.length), samples: arr.length }
  })

  return {
    p, A, B,
    errorPerClifford: r,
    errorPerCliffordStderr: stderr,
    fidelity: 1 - r,
    qubits,
    points: summary,
  }
}

export interface RbResult extends RbFit {
  readonly sequences: readonly RbSequence[]
  /** One survival probability per sequence, in `sequences` order. */
  readonly survivals: readonly RbPoint[]
}

/**
 * Generate, run and fit a randomized benchmarking experiment in one call.
 * Circuits are executed one after another; to submit them concurrently, use
 * `rbSequences`, `rbSurvival` and `rbFit` directly.
 *
 * @example
 * const rb = await runRb(c => c.runClifford({ shots: 1000, noise: { p1: 0.002 } }), {
 *   lengths: [1, 2, 4, 8, 16, 32, 64], samples: 20, seed: 7,
 * })
 * rb.errorPerClifford  // ≈ 0.004 — three gates per Clifford on average, 2p/3 average error each
 *
 * Choose lengths long enough that the survival visibly decays. A flat curve
 * leaves p unidentified, and `errorPerCliffordStderr` will be large to say so.
 */
export async function runRb(
  execute: Executor,
  options: RbOptions & Omit<RbFitOptions, 'qubits'>,
): Promise<RbResult> {
  checkOptions(options, [...RB_OPTION_KEYS, 'bootstrap', 'fixB'], 'runRb')
  const { bootstrap, fixB, ...rbOpts } = options
  const sequences = rbSequences(rbOpts)
  const { qs } = resolveRbQubits(rbOpts, 'runRb')
  const survivals: RbPoint[] = []
  for (let i = 0; i < sequences.length; i++) {
    const seq = sequences[i]!
    survivals.push({ length: seq.length, survival: rbSurvival(resolveOutcomes(await execute(seq.circuit, i)), qs) })
  }
  // One seed drives both the sequence draw and the bootstrap.
  const fitOpts: RbFitOptions = { qubits: qs.length }
  if (bootstrap !== undefined) fitOpts.bootstrap = bootstrap
  if (fixB !== undefined) fitOpts.fixB = fixB
  if (rbOpts.seed !== undefined) fitOpts.seed = rbOpts.seed
  return { ...rbFit(survivals, fitOpts), sequences, survivals }
}
