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
  denseClone, denseCNOT, denseCollapse, denseControlled, denseCsrSwap, denseCSwap,
  denseDecay, denseNnz, denseNorm2, denseProbabilities, denseProbOne, denseSample,
  denseScale, denseScaleBranch, denseSingle, denseSWAP, denseToffoli, denseTwo,
  denseUnitary, denseZero, fromSparse, guardSparseGrowth, MAX_DENSE_QUBITS, resolvePolicy, toSparse,
  type DenseOptions, type DensePolicy, type DenseState,
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
 * A statevector in whichever representation currently suits it.
 *
 * The sparse variant carries the promotion policy so it travels with the state
 * rather than living in module scope — the library is otherwise free of mutable
 * globals, and two concurrent runs may legitimately want different thresholds.
 * The dense variant does not need it: promotion is one-way.
 */
export type SimState =
  | { readonly kind: 'sparse'; readonly sv: StateVector; readonly n: number; readonly policy: DensePolicy }
  | { readonly kind: 'dense';  readonly d: DenseState }

/** |0…0⟩ over n qubits, sparse. */
export const simZero = (n: number, policy: DensePolicy = DEFAULT_SV_POLICY): SimState =>
  ({ kind: 'sparse', sv: new Map([[0n, { re: 1, im: 0 }]]), n, policy })

/** Wrap an existing sparse state. */
export const simFromSparse = (sv: StateVector, n: number, policy: DensePolicy = DEFAULT_SV_POLICY): SimState =>
  ({ kind: 'sparse', sv, n, policy })

/** True once the state has densified enough to be worth moving. */
function shouldPromote(sv: StateVector, n: number, policy: DensePolicy): boolean {
  return n <= policy.maxQubits && sv.size * policy.fill > 2 ** n
}

/** Apply the fill test to a freshly-computed sparse state, promoting if warranted. */
function settle(sv: StateVector, n: number, policy: DensePolicy): SimState {
  if (shouldPromote(sv, n, policy)) return { kind: 'dense', d: fromSparse(sv, n) }
  guardSparseGrowth(sv.size, n, policy, DEFAULT_SV_POLICY, 'statevector')
  return { kind: 'sparse', sv, n, policy }
}

/** Force the dense representation regardless of fill. Exposed for testing. */
export function simPromote(s: SimState): SimState {
  if (s.kind === 'dense') return s
  return { kind: 'dense', d: fromSparse(s.sv, s.n) }
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

// ── Gate application ──────────────────────────────────────────────────────────
//
// Dense states mutate in place and return the same object; sparse states rebuild
// and re-test for promotion.

export function simSingle(s: SimState, q: number, gate: Gate2x2): SimState {
  if (s.kind === 'dense') { denseSingle(s.d, q, gate); return s }
  return settle(applySingle(s.sv, q, gate), s.n, s.policy)
}

export function simCNOT(s: SimState, control: number, target: number): SimState {
  if (s.kind === 'dense') { denseCNOT(s.d, control, target); return s }
  // A permutation cannot change the support size, so no promotion test is needed.
  return { kind: 'sparse', sv: applyCNOT(s.sv, control, target), n: s.n, policy: s.policy }
}

export function simSWAP(s: SimState, a: number, b: number): SimState {
  if (s.kind === 'dense') { denseSWAP(s.d, a, b); return s }
  return { kind: 'sparse', sv: applySWAP(s.sv, a, b), n: s.n, policy: s.policy }
}

export function simToffoli(s: SimState, c1: number, c2: number, target: number): SimState {
  if (s.kind === 'dense') { denseToffoli(s.d, c1, c2, target); return s }
  return { kind: 'sparse', sv: applyToffoli(s.sv, c1, c2, target), n: s.n, policy: s.policy }
}

export function simCSwap(s: SimState, control: number, a: number, b: number): SimState {
  if (s.kind === 'dense') { denseCSwap(s.d, control, a, b); return s }
  return { kind: 'sparse', sv: applyCSwap(s.sv, control, a, b), n: s.n, policy: s.policy }
}

export function simControlled(s: SimState, control: number, target: number, gate: Gate2x2): SimState {
  if (s.kind === 'dense') { denseControlled(s.d, control, target, gate); return s }
  return settle(applyControlled(s.sv, control, target, gate), s.n, s.policy)
}

export function simTwo(s: SimState, a: number, b: number, gate: Gate4x4): SimState {
  if (s.kind === 'dense') { denseTwo(s.d, a, b, gate); return s }
  return settle(applyTwo(s.sv, a, b, gate), s.n, s.policy)
}

export function simCsrSwap(s: SimState, control: number, a: number, b: number): SimState {
  if (s.kind === 'dense') { denseCsrSwap(s.d, control, a, b); return s }
  return settle(applyCsrSwap(s.sv, control, a, b), s.n, s.policy)
}

export function simUnitary(s: SimState, qs: readonly number[], matrix: readonly (readonly Complex[])[]): SimState {
  if (s.kind === 'dense') { denseUnitary(s.d, qs, matrix); return s }
  return settle(applyUnitary(s.sv, qs, matrix), s.n, s.policy)
}

// ── Measurement and channel primitives ────────────────────────────────────────
//
// The noise / mid-circuit path projects, rescales and samples between gates.
// Sparse implementations rebuild a Map; dense ones mutate in place.

/** Independent copy. Kraus channels trial every operator, so they need one per branch. */
export const simClone = (s: SimState): SimState =>
  s.kind === 'sparse' ? { kind: 'sparse', sv: new Map(s.sv), n: s.n, policy: s.policy } : { kind: 'dense', d: denseClone(s.d) }

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
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy }
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
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy }
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
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy }
}

/** Amplitude-damping jump: |1⟩ amplitudes move to their |0⟩ partner, scaled by `inv`. */
export function simDecay(s: SimState, q: number, inv: number): SimState {
  if (s.kind === 'dense') { denseDecay(s.d, q, inv); return s }
  const mask = 1n << BigInt(q)
  const next: StateVector = new Map()
  for (const [i, a] of s.sv) {
    if ((i & mask) !== 0n) next.set(i ^ mask, { re: a.re * inv, im: a.im * inv })
  }
  return { kind: 'sparse', sv: next, n: s.n, policy: s.policy }
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

/** Re-export so callers can build a dense zero state without importing dense.js. */
export { denseZero, MAX_DENSE_QUBITS }
export type { DenseOptions, DensePolicy }
