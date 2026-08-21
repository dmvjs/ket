/**
 * Hybrid statevector driver.
 *
 * Circuits start sparse and stay sparse for as long as that pays. A state that
 * densifies is promoted once to a `Float64Array` and runs there for the rest of
 * the circuit. Promotion is one-way: a dense state is never demoted, because the
 * fill test would cost a full scan per gate to save work the dense kernel is
 * already fast at.
 *
 * The two regimes this exists to serve, both real:
 *
 *   - GHZ-20 holds two amplitudes through the whole circuit. Sparse runs it in
 *     microseconds; dense would touch 1M slots per gate for nothing.
 *   - A depth-4 random circuit on 16 qubits fills all 65,536 amplitudes within
 *     the first layer. Sparse then pays a Map rebuild, a Set of visited keys and
 *     an object allocation per amplitude per gate; dense pays none of it.
 *
 * Every function here mirrors the signature of its `statevector.ts` counterpart
 * so the caller's dispatch table does not change shape.
 */

import { AMP_EPSILON, type Complex } from './complex.js'
import {
  cdfSample, denseSampleEach, denseClone, denseCollapse, denseDecay, denseNnz, denseNorm2,
  denseProbabilities, denseProbsEach, denseProbOne, denseSample, denseScale, denseScaleBranch,
  denseZero, DENSE_QUBIT_LIMIT, fromSparse, guardSparseGrowth, MAX_DENSE_QUBITS,
  resolvePolicy, SERIAL_EXEC, toSparse,
  type DenseExec, type DenseOptions, type DensePolicy, type DenseState,
} from './dense.js'
import {
  applyCNOT, applyControlled, applyCsrSwap, applyCSwap, applySingle, applySWAP,
  applyToffoli, applyTwo, applyUnitary, probabilities,
  type Gate2x2, type Gate4x4, type StateVector,
} from './statevector.js'

/**
 * Default promotion point: at least 1/8 full.
 *
 * The dense kernel measured ~100x cheaper per amplitude than the sparse one, so
 * break-even sits near 2ⁿ/100, and this promotes just short of it. Promoting at
 * 2ⁿ/8 instead — twelve times later than break-even — was measurably expensive:
 * every amplitude added past the crossover costs a BigInt key, a boxed complex
 * and a Map slot to build, and then gets copied into the dense buffer anyway.
 *
 * Measured across dense, sparse, partial-occupancy, QFT and Grover circuits at
 * n = 10…22, fill = 64 is faster than fill = 8 everywhere and never slower: 4.3x
 * on a 75%-occupancy state at n = 16, 3.9x at n = 22 (766ms -> 195ms), 1.9x on
 * QFT-16, and identical on genuinely sparse states, which still never promote.
 * Going further to 128 over-promotes and regresses at n = 18, so this sits at 64.
 *
 * Override per call via `RunOptions.dense`.
 */
export const DEFAULT_SV_POLICY: DensePolicy = { fill: 64, maxQubits: MAX_DENSE_QUBITS }

/** Resolve caller-supplied statevector dense options against the defaults. */
export const svPolicy = (opts?: DenseOptions): DensePolicy => resolvePolicy(opts, DEFAULT_SV_POLICY)

/**
 * Where dense gates run, and whether the buffer they run on can be shared.
 *
 * Travels with the state for the same reason the promotion policy does: two
 * concurrent runs may legitimately want different execution strategies, and the
 * library is otherwise free of mutable globals.
 *
 * `shared` has to be decided before the state exists, not at the gate that
 * finally needs it — a `SharedArrayBuffer` cannot be adopted after allocation,
 * so a state that might be promoted into a worker pool must be allocated shared
 * from the start. It is kept separate from `exec` because the two can disagree:
 * a run that asked for workers on a host that cannot spawn them gets the serial
 * executor over an ordinary buffer.
 */
export interface DenseRuntime {
  readonly exec: DenseExec
  readonly shared: boolean
}

/** Run every gate on the calling thread, over an ordinary buffer. */
export const SERIAL_RUNTIME: DenseRuntime = { exec: SERIAL_EXEC, shared: false }

/**
 * A statevector in whichever representation currently suits it.
 *
 * The sparse variant carries the promotion policy so it travels with the state
 * rather than living in module scope — the library is otherwise free of mutable
 * globals, and two concurrent runs may legitimately want different thresholds.
 * Both variants carry the runtime, because a sparse state has to know how it
 * would be promoted before it is.
 */
export type SimState =
  | {
      readonly kind: 'sparse'; readonly sv: StateVector; readonly n: number
      readonly policy: DensePolicy; readonly rt: DenseRuntime
    }
  | { readonly kind: 'dense'; readonly d: DenseState; readonly rt: DenseRuntime }

/** |0…0⟩ over n qubits, sparse. */
export const simZero = (
  n: number, policy: DensePolicy = DEFAULT_SV_POLICY, rt: DenseRuntime = SERIAL_RUNTIME,
): SimState => ({ kind: 'sparse', sv: new Map([[0n, { re: 1, im: 0 }]]), n, policy, rt })

/** Wrap an existing sparse state. */
/**
 * Adopt a sparse state, copying it first.
 *
 * Sparse gates mutate the map they are given, matching what the dense kernel has
 * always done. The copy is what makes that safe: it draws the ownership line at
 * the point the simulation takes the state, so a caller-supplied initial state is
 * never written through.
 */
export const simFromSparse = (
  sv: StateVector, n: number, policy: DensePolicy = DEFAULT_SV_POLICY, rt: DenseRuntime = SERIAL_RUNTIME,
): SimState => ({ kind: 'sparse', sv: new Map(sv), n, policy, rt })

/** True once the state has densified enough to be worth moving. */
function shouldPromote(sv: StateVector, n: number, policy: DensePolicy): boolean {
  return n <= policy.maxQubits && sv.size * policy.fill > 2 ** n
}

/** Apply the fill test to a freshly-computed sparse state, promoting if warranted. */
function settle(sv: StateVector, n: number, policy: DensePolicy, rt: DenseRuntime): SimState {
  if (shouldPromote(sv, n, policy)) return { kind: 'dense', d: fromSparse(sv, n, rt.shared), rt }
  guardSparseGrowth(sv.size, n, policy, DEFAULT_SV_POLICY, 'statevector')
  return { kind: 'sparse', sv, n, policy, rt }
}

/** Force the dense representation regardless of fill. Exposed for testing. */
export function simPromote(s: SimState): SimState {
  if (s.kind === 'dense') return s
  return { kind: 'dense', d: fromSparse(s.sv, s.n, s.rt.shared), rt: s.rt }
}

/** Which backend is currently live — for tests and diagnostics. */
export const simKind = (s: SimState): 'sparse' | 'dense' => s.kind

/** Non-negligible amplitude count. */
export const simNnz = (s: SimState): number =>
  s.kind === 'sparse' ? s.sv.size : denseNnz(s.d)

/**
 * Visit every non-negligible amplitude as (index, re, im).
 *
 * Lets callers that only want to read amplitudes skip `simToSparse`, which for a
 * dense state builds a 2ⁿ-entry Map purely to be iterated and thrown away.
 */
export function simForEach(s: SimState, fn: (idx: number, re: number, im: number) => void): void {
  if (s.kind === 'sparse') {
    for (const [i, a] of s.sv) fn(Number(i), a.re, a.im)
    return
  }
  const { data } = s.d
  const total = 1 << s.d.n
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    if (re * re + im * im >= AMP_EPSILON * AMP_EPSILON) fn(i, re, im)
  }
}

/** Materialise as the sparse map the public API returns. */
export const simToSparse = (s: SimState): StateVector =>
  s.kind === 'sparse' ? s.sv : toSparse(s.d)

/**
 * Probabilities keyed by basis index.
 *
 * Reads a dense state directly rather than going through `simToSparse`, so the
 * hot `run()` and `exactProbs()` paths never build the intermediate map.
 */
export const simProbabilities = (s: SimState): Map<bigint, number> =>
  s.kind === 'sparse' ? probabilities(s.sv) : denseProbabilities(s.d)

/**
 * Visit each non-negligible probability in ascending index order, without
 * materialising a map of them.
 *
 * The index type follows the representation rather than being normalised to one
 * of them, because normalising is the cost worth avoiding: a dense state can
 * report a plain `number`, and a caller re-keying every amplitude — `exactProbs`
 * builds 2ⁿ bitstrings — would otherwise allocate a `BigInt` per amplitude only
 * to stringify it. Sparse indices stay `bigint`: those states are not bounded by
 * the dense addressing limit and genuinely need the width.
 *
 * The fork is deliberately visible in the callback's type, so a caller cannot
 * forget that one branch exists. See {@link simSampleEach} for the same trade on
 * the sampling path.
 */
export function simProbsEach(s: SimState, emit: (idx: number | bigint, p: number) => void): void {
  if (s.kind === 'dense') { denseProbsEach(s.d, emit); return }
  for (const [idx, p] of probabilities(s.sv)) emit(idx, p)
}

// ── Gate application ──────────────────────────────────────────────────────────
//
// Dense states mutate in place and return the same object; sparse states rebuild
// and re-test for promotion.

export function simSingle(s: SimState, q: number, gate: Gate2x2): SimState {
  if (s.kind === 'dense') { s.rt.exec.single(s.d, q, gate); return s }
  return settle(applySingle(s.sv, q, gate), s.n, s.policy, s.rt)
}

export function simCNOT(s: SimState, control: number, target: number): SimState {
  if (s.kind === 'dense') { s.rt.exec.cnot(s.d, control, target); return s }
  // A permutation cannot change the support size, so no promotion test is needed.
  return { kind: 'sparse', sv: applyCNOT(s.sv, control, target), n: s.n, policy: s.policy, rt: s.rt }
}

export function simSWAP(s: SimState, a: number, b: number): SimState {
  if (s.kind === 'dense') { s.rt.exec.swap(s.d, a, b); return s }
  return { kind: 'sparse', sv: applySWAP(s.sv, a, b), n: s.n, policy: s.policy, rt: s.rt }
}

export function simToffoli(s: SimState, c1: number, c2: number, target: number): SimState {
  if (s.kind === 'dense') { s.rt.exec.toffoli(s.d, c1, c2, target); return s }
  return { kind: 'sparse', sv: applyToffoli(s.sv, c1, c2, target), n: s.n, policy: s.policy, rt: s.rt }
}

export function simCSwap(s: SimState, control: number, a: number, b: number): SimState {
  if (s.kind === 'dense') { s.rt.exec.cswap(s.d, control, a, b); return s }
  return { kind: 'sparse', sv: applyCSwap(s.sv, control, a, b), n: s.n, policy: s.policy, rt: s.rt }
}

export function simControlled(s: SimState, control: number, target: number, gate: Gate2x2): SimState {
  if (s.kind === 'dense') { s.rt.exec.controlled(s.d, control, target, gate); return s }
  return settle(applyControlled(s.sv, control, target, gate), s.n, s.policy, s.rt)
}

export function simTwo(s: SimState, a: number, b: number, gate: Gate4x4): SimState {
  if (s.kind === 'dense') { s.rt.exec.two(s.d, a, b, gate); return s }
  return settle(applyTwo(s.sv, a, b, gate), s.n, s.policy, s.rt)
}

export function simCsrSwap(s: SimState, control: number, a: number, b: number): SimState {
  if (s.kind === 'dense') { s.rt.exec.csrswap(s.d, control, a, b); return s }
  return settle(applyCsrSwap(s.sv, control, a, b), s.n, s.policy, s.rt)
}

export function simUnitary(s: SimState, qs: readonly number[], matrix: readonly (readonly Complex[])[]): SimState {
  if (s.kind === 'dense') { s.rt.exec.unitary(s.d, qs, matrix); return s }
  return settle(applyUnitary(s.sv, qs, matrix), s.n, s.policy, s.rt)
}

// ── Measurement and channel primitives ────────────────────────────────────────
//
// The noise / mid-circuit path projects, rescales and samples between gates.
// Sparse implementations rebuild a Map; dense ones mutate in place.

/** Independent copy. Kraus channels trial every operator, so they need one per branch. */
export const simClone = (s: SimState): SimState =>
  s.kind === 'sparse'
    ? { kind: 'sparse', sv: new Map(s.sv), n: s.n, policy: s.policy, rt: s.rt }
    // A clone is scratch for a Kraus trial, never handed to the pool, so it is
    // deliberately unshared even when the state it came from is.
    : { kind: 'dense', d: denseClone(s.d), rt: SERIAL_RUNTIME }

/** Total ⟨ψ|ψ⟩ — not 1 after an unnormalised Kraus operator. */
export function simNorm2(s: SimState): number {
  if (s.kind === 'dense') return denseNorm2(s.d)
  let t = 0
  for (const a of s.sv.values()) t += a.re * a.re + a.im * a.im
  return t
}

/** Multiply every amplitude by a real scalar. */
export function simScale(s: SimState, f: number): SimState {
  if (s.kind === 'dense') { denseScale(s.d, f); return s }
  const next: StateVector = new Map()
  for (const [i, a] of s.sv) next.set(i, { re: a.re * f, im: a.im * f })
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy, rt: s.rt }
}

/** Probability that qubit q reads 1. */
export function simProbOne(s: SimState, q: number): number {
  if (s.kind === 'dense') return denseProbOne(s.d, q)
  const mask = 1n << BigInt(q)
  let p = 0
  for (const [i, a] of s.sv) if ((i & mask) !== 0n) p += a.re * a.re + a.im * a.im
  return p
}

/** Project qubit q onto `outcome` and renormalise by `inv`. */
export function simCollapse(s: SimState, q: number, outcome: 0 | 1, inv: number): SimState {
  if (s.kind === 'dense') { denseCollapse(s.d, q, outcome, inv); return s }
  const mask = 1n << BigInt(q)
  const next: StateVector = new Map()
  for (const [i, a] of s.sv) {
    if (((i & mask) !== 0n) === (outcome === 1)) next.set(i, { re: a.re * inv, im: a.im * inv })
  }
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy, rt: s.rt }
}

/** Scale the |1⟩ branch of qubit q by `sOne`, the |0⟩ branch by `sZero`. */
export function simScaleBranch(s: SimState, q: number, sOne: number, sZero: number): SimState {
  if (s.kind === 'dense') { denseScaleBranch(s.d, q, sOne, sZero); return s }
  const mask = 1n << BigInt(q)
  const next: StateVector = new Map()
  for (const [i, a] of s.sv) {
    const f = (i & mask) !== 0n ? sOne : sZero
    next.set(i, { re: a.re * f, im: a.im * f })
  }
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy, rt: s.rt }
}

/** Amplitude-damping jump: |1⟩ amplitudes move to their |0⟩ partner, scaled by `inv`. */
export function simDecay(s: SimState, q: number, inv: number): SimState {
  if (s.kind === 'dense') { denseDecay(s.d, q, inv); return s }
  const mask = 1n << BigInt(q)
  const next: StateVector = new Map()
  for (const [i, a] of s.sv) {
    if ((i & mask) !== 0n) next.set(i ^ mask, { re: a.re * inv, im: a.im * inv })
  }
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy, rt: s.rt }
}

/** Sample one basis index. Both representations walk indices ascending, so a
 *  given RNG draw yields the same outcome either way. */
export function simSample(s: SimState, rand: number): bigint {
  if (s.kind === 'dense') return denseSample(s.d, rand)
  const sorted = Array.from(s.sv.entries()).toSorted(([a], [b]) => (a < b ? -1 : 1))
  let cum = 0
  for (const [idx, amp] of sorted) {
    cum += amp.re * amp.re + amp.im * amp.im
    if (rand <= cum) return idx
  }
  return sorted.at(-1)?.[0] ?? 0n
}

/**
 * Draw `shots` samples from the final state, calling `emit` per shot.
 *
 * `run()` used to sample by materialising `Map<bigint, number>` probabilities,
 * copying them to an array and sorting them back into the ascending order they
 * were already in, all to read a few thousand values out. For a dense state that
 * is 2ⁿ BigInt keys and a 2ⁿ-element sort per call, which measured at 25–54% of
 * total wall-clock at n = 20–22.
 *
 * Both representations produce the same cumulative distribution in ascending
 * index order, so a given draw yields the same outcome either way — the dense
 * path just gets there without boxing anything.
 */
export function simSampleEach(
  s: SimState, shots: number, rng: () => number, emit: (idx: bigint) => void,
): void {
  if (s.kind === 'dense') {
    // One BigInt per shot instead of one per amplitude.
    denseSampleEach(s.d, shots, rng, i => emit(BigInt(i)))
    return
  }
  // A sparse state holds only its support, so its table is already O(support)
  // and there is nothing to stream away.
  const sorted = Array.from(s.sv.entries()).toSorted(([a], [b]) => (a < b ? -1 : 1))
  if (sorted.length === 0) return
  const idx = sorted.map(([i]) => i)
  const cdf = new Float64Array(sorted.length)
  let cum = 0
  for (let i = 0; i < sorted.length; i++) {
    const a = sorted[i]![1]
    cum += a.re * a.re + a.im * a.im
    cdf[i] = cum
  }
  cdf[cdf.length - 1] = 1
  const last = idx[idx.length - 1]!
  for (let i = 0; i < shots; i++) emit(idx[cdfSample(cdf, rng())] ?? last)
}

/** Re-export so callers can build a dense zero state without importing dense.js. */
export { denseZero, DENSE_QUBIT_LIMIT, MAX_DENSE_QUBITS }
export type { DenseExec, DenseOptions, DensePolicy }
