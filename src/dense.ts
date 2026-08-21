/**
 * Dense statevector backend.
 *
 * The sparse `Map<bigint, Complex>` backend in `statevector.ts` is unbeatable on
 * states that stay sparse — a GHZ chain holds two amplitudes at any width. It is
 * a poor fit once a state densifies: every gate allocates a fresh `Map`, a
 * `Set<bigint>` of visited keys, and one `{re, im}` object per amplitude, with
 * BigInt arithmetic in the inner loop.
 *
 * This module is the other half of that trade. State lives in one contiguous
 * `Float64Array` with real and imaginary parts interleaved — `data[2i]` and
 * `data[2i+1]` hold amplitude `i`. Gates mutate it in place, so a gate costs
 * 2ⁿ unboxed f64 operations with no allocation at all. V8 keeps the whole inner
 * loop in registers.
 *
 * Index convention matches the sparse backend exactly: qubit q is bit q of the
 * amplitude index (q0 = LSB).
 *
 * `hybrid.ts` decides which representation a circuit runs on.
 */

import { AMP_EPSILON, type Complex } from './complex.js'
import type { Gate2x2, Gate4x4, StateVector } from './statevector.js'

/**
 * Amplitude-magnitude-squared cutoff, matching `isNegligible` on the sparse side
 * so a state means the same thing in either representation.
 */
const AMP_MIN_NORM2 = AMP_EPSILON * AMP_EPSILON

/**
 * Dense state over `n` qubits.
 *
 * `data.length === 2^(n+1)` — two f64 slots (re, im) per amplitude. Mutated in
 * place by every operation in this module; nothing here returns a new state.
 */
export interface DenseState {
  readonly n: number
  readonly data: Float64Array
}

/**
 * Hard architectural ceiling on the dense backend: 30 qubits.
 *
 * Not a memory bound — a policy cannot raise it, because past it the *indexing*
 * stops working. Every kernel here addresses amplitude `i` at slots `i << 1` and
 * `(i << 1) | 1`, and bounds its walk with `1 << n`. Both are int32 operations:
 *
 *   n = 30:  max index 1,073,741,823  →  i << 1 = 2,147,483,646   ok
 *   n = 31:  max index 2,147,483,647  →  i << 1 =            -2   negative
 *
 * At n = 31 every loop bound goes negative and every kernel silently iterates
 * zero times, returning an untouched buffer with no error raised. Rejecting the
 * policy up front is the only way that failure becomes visible.
 *
 * The shift is not incidental and cannot simply be widened: replacing `i << 1`
 * with `i * 2` to reach 31+ measured 1.9x slower, because the multiply leaves
 * V8's int32 fast path. Going wider needs chunked or two-level addressing, which
 * is a different design rather than a constant change.
 */
export const DENSE_QUBIT_LIMIT = 30

/**
 * Largest n this backend will allocate *without being asked*: 2^26 amplitudes ×
 * 16 bytes = 1 GiB. Beyond it the sparse backend stays in charge however full
 * the state gets, until the caller raises the ceiling via {@link DenseOptions}.
 *
 * This sits below {@link DENSE_QUBIT_LIMIT} on purpose, and the gap is not
 * timidity — it is where the two representations actually cross. A sparse entry
 * costs ~100 bytes against 16 for a dense amplitude, so dense only becomes the
 * cheaper way to hold a state above ~16% occupancy. Promotion fires at 1/64,
 * which is 1.6%. In that band dense is *faster* but roughly ten times larger, so
 * promotion trades memory for speed rather than saving both.
 *
 * That trade is worth making silently at 1 GiB and not at 16. Promoting a
 * 30-qubit state the moment it crossed 1.6% fill would swap a 1.7 GB sparse map
 * for a 16 GiB buffer that the caller never asked for. Anyone who does want the
 * top of the range is chasing it deliberately — `dense: { maxQubits: 30 }`,
 * usually alongside `workers` — and saying so costs them one option.
 */
export const MAX_DENSE_QUBITS = 26

/**
 * Per-call override for when a state is moved to the dense representation.
 *
 * Both fields are advisory bounds on memory versus speed, so the useful reasons
 * to set them are concrete: a constrained environment that cannot afford the
 * default ceiling, or a large machine where paying more memory to go faster is
 * the right trade.
 */
export interface DenseOptions {
  /**
   * Promote once the state exceeds `1 / fill` of full occupancy. Higher values
   * promote sooner. Default 64 for statevectors, 32 for density matrices.
   */
  fill?: number
  /**
   * Largest qubit count for which a dense buffer will be allocated at all.
   * Beyond it the sparse representation is used however full the state gets.
   * Default 24 for statevectors (256 MiB), 12 for density matrices (256 MiB).
   */
  maxQubits?: number
}

/** {@link DenseOptions} with defaults filled in. */
export interface DensePolicy {
  readonly fill: number
  readonly maxQubits: number
}

/**
 * Entry count past which a sparse state is treated as a memory hazard.
 *
 * A sparse entry costs roughly 100 bytes once the BigInt key, the boxed complex
 * value and Map overhead are counted, so ~4M entries is already several hundred
 * megabytes. Used only to catch the case where promotion was disabled by an
 * override and the state filled up anyway — see `guardSparseGrowth`.
 */
export const SPARSE_ENTRY_LIMIT = 1 << 22

/**
 * Fail fast when an override has turned off promotion for a state that is
 * filling up and small enough that dense would have handled it.
 *
 * `dense.maxQubits` exists to *bound* memory. Honouring a lowered ceiling all
 * the way into a multi-gigabyte sparse map would invert that intent and end in a
 * V8 heap abort with no indication of the cause, so this raises a diagnostic
 * instead.
 *
 * Deliberately narrow. It fires only when all three hold:
 *   - the state has grown past the hazard threshold,
 *   - promotion is off because the caller lowered `maxQubits`,
 *   - the default ceiling would have allowed dense at this width.
 *
 * So default runs are unaffected, and so are genuinely wide circuits where a
 * dense buffer was never an option and sparse is the only way to proceed.
 */
export function guardSparseGrowth(
  size: number, n: number, policy: DensePolicy, defaults: DensePolicy, what: string,
): void {
  if (size <= SPARSE_ENTRY_LIMIT) return
  if (n <= policy.maxQubits) return          // promotion still available
  if (n > defaults.maxQubits) return         // dense was never viable here
  throw new RangeError(
    `${what}: ${size.toLocaleString()} non-zero entries at ${n} qubits with dense promotion disabled ` +
    `(dense.maxQubits = ${policy.maxQubits}). The sparse representation would need several gigabytes. ` +
    `Raise dense.maxQubits to at least ${n} to use the dense backend, or reduce the circuit.`,
  )
}

/** Resolve caller options against a backend's defaults, rejecting nonsense. */
export function resolvePolicy(opts: DenseOptions | undefined, fallback: DensePolicy): DensePolicy {
  const fill = opts?.fill ?? fallback.fill
  const maxQubits = opts?.maxQubits ?? fallback.maxQubits
  if (!(fill > 0)) throw new RangeError(`dense.fill must be > 0 (got ${fill})`)
  if (!Number.isInteger(maxQubits) || maxQubits < 0) {
    throw new RangeError(`dense.maxQubits must be a non-negative integer (got ${maxQubits})`)
  }
  if (maxQubits > DENSE_QUBIT_LIMIT) {
    throw new RangeError(
      `dense.maxQubits must be at most ${DENSE_QUBIT_LIMIT} (got ${maxQubits}). ` +
      `The dense backend addresses amplitudes with int32 arithmetic, which overflows above ` +
      `${DENSE_QUBIT_LIMIT} qubits and would silently leave the state untouched.`,
    )
  }
  return { fill, maxQubits }
}

/**
 * Allocate the amplitude buffer for an n-qubit dense state, or explain why not.
 *
 * A `Float64Array` that the host cannot back throws `RangeError: Array buffer
 * allocation failed`, which says nothing about the circuit that asked for it.
 * Since a dense promotion is an implicit consequence of a state filling up
 * rather than something the caller wrote, the bare message is close to
 * undiagnosable — hence restating it in terms of qubits, bytes and the knob.
 */
function allocAmplitudes(n: number, shared: boolean): Float64Array {
  const slots = 2 ** (n + 1)
  const bytes = slots * 8
  try {
    return shared
      ? new Float64Array(new SharedArrayBuffer(bytes))
      : new Float64Array(slots)
  } catch (cause) {
    throw new RangeError(
      `dense statevector: cannot allocate ${(bytes / 2 ** 30).toFixed(1)} GiB for ${n} qubits ` +
      `(2^${n} amplitudes × 16 bytes). Lower dense.maxQubits to keep this circuit on the sparse ` +
      `backend, or reduce the qubit count.`,
      { cause },
    )
  }
}

/**
 * |0…0⟩ over n qubits.
 *
 * `shared` backs the amplitudes with a `SharedArrayBuffer` so worker threads can
 * operate on the same state — see `dense-parallel.ts`. Every kernel here is
 * indifferent to which it gets; `Float64Array` behaves identically over either.
 */
export function denseZero(n: number, shared = false): DenseState {
  const data = allocAmplitudes(n, shared)
  data[0] = 1
  return { n, data }
}

/** Materialise a sparse state as dense. Cost is O(2ⁿ) for the allocation. */
export function fromSparse(sv: StateVector, n: number, shared = false): DenseState {
  const d = { n, data: allocAmplitudes(n, shared) }
  for (const [idx, amp] of sv) {
    const i = Number(idx) << 1
    d.data[i]     = amp.re
    d.data[i + 1] = amp.im
  }
  return d
}

/**
 * Convert back to the sparse map the public API returns.
 *
 * Drops negligible amplitudes on the same threshold the sparse backend uses, so
 * a state that round-trips through dense has the same support it would have had
 * staying sparse.
 */
export function toSparse(d: DenseState): StateVector {
  const sv: StateVector = new Map()
  const { data } = d
  const total = 1 << d.n
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    if (re * re + im * im >= AMP_MIN_NORM2) sv.set(BigInt(i), { re, im })
  }
  return sv
}

/**
 * Probability below which a basis state counts as absent.
 *
 * Deliberately not {@link AMP_MIN_NORM2}: that bounds an *amplitude* worth
 * keeping, this bounds a *probability* worth reporting. The sparse backend's
 * `probabilities` uses the same figure, so which representation a state ended in
 * cannot change which keys come back.
 */
const PROB_EPSILON = 1e-14

/**
 * Visit each non-negligible probability in ascending index order.
 *
 * Yields a plain `number` index, which is the whole point: a caller that only
 * wants to *look at* the distribution should not have to allocate a `BigInt` and
 * a map entry per amplitude to do it. `exactProbs` re-keys 2ⁿ probabilities as
 * bitstrings, and going via {@link denseProbabilities} first meant building a
 * 2ⁿ-entry `Map` purely to walk it once and discard it — measured at n = 22 that
 * intermediate map was 890ms of a 3.8s call.
 *
 * A `number` is always enough here: dense states are capped at
 * {@link DENSE_QUBIT_LIMIT} qubits, far below the 2⁵³ where an integer index
 * would stop being exact.
 */
export function denseProbsEach(d: DenseState, emit: (i: number, p: number) => void): void {
  const { data } = d
  const total = 1 << d.n
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    const p = re * re + im * im
    if (p > PROB_EPSILON) emit(i, p)
  }
}

/**
 * Probability of each basis state, keyed as the sparse backend keys them.
 *
 * Built on {@link denseProbsEach} so the walk and the threshold have one
 * definition; the `BigInt` and the map entry per amplitude are what a caller
 * pays for wanting a `Map`, and are why {@link denseProbsEach} exists.
 */
export function denseProbabilities(d: DenseState): Map<bigint, number> {
  const probs = new Map<bigint, number>()
  denseProbsEach(d, (i, p) => probs.set(BigInt(i), p))
  return probs
}

/**
 * Cumulative probability over all 2ⁿ basis states, ascending by index.
 *
 * The sampling path wants a CDF to binary-search, and used to get there by way
 * of {@link denseProbabilities} — a `Map` with a `BigInt` key and a boxed number
 * per amplitude — which was then copied to an array and sorted back into the
 * ascending order it was already built in. All of that to read `shots` values
 * out of it.
 *
 * Building the CDF straight into a `Float64Array` skips the map, the boxing, the
 * 2ⁿ BigInt allocations and the sort. Measured against the old path on a fully
 * dense state: 194ms → 1.4ms at n = 20, 1383ms → 3.9ms at n = 22. Since that
 * work sat on every `run()`, it was 25–54% of total wall-clock at those widths.
 *
 * `cdf[i]` is the probability of drawing an index ≤ i, so `run()` converts a
 * uniform draw to a basis index with one binary search and one BigInt — per
 * shot, rather than per amplitude. The final entry is pinned to exactly 1 so a
 * draw arbitrarily close to 1 cannot fall off the end through rounding.
 */
export function denseCdf(d: DenseState): Float64Array {
  const { data } = d
  const total = 1 << d.n
  const cdf = new Float64Array(total)
  let cum = 0
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    cum += re * re + im * im
    cdf[i] = cum
  }
  if (total > 0) cdf[total - 1] = 1
  return cdf
}

/**
 * Smallest index whose cumulative probability reaches `r`, by binary search.
 *
 * Matches the sparse sampler's ascending walk, so a given RNG draw picks the
 * same outcome in either representation.
 */
export function cdfSample(cdf: Float64Array, r: number): number {
  let lo = 0, hi = cdf.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (cdf[mid]! < r) lo = mid + 1
    else hi = mid
  }
  return lo
}

/**
 * States at or below which sampling materialises a {@link denseCdf}.
 *
 * 4096 amplitudes is a 32 KiB table — small enough that holding it is free, and
 * the regime where binary search beats the streaming walk below, because a tiny
 * state with a huge shot count pays for one pass per *amplitude* either way but
 * only log₂(4096) = 12 comparisons per shot.
 */
export const CDF_MAX_STATE = 1 << 12

/**
 * Draw `shots` samples, calling `emit` once per shot with the basis index.
 *
 * Building a full CDF costs 8 bytes per amplitude *on top of* the 16 the state
 * already occupies — 512 MiB alongside a 1 GiB state at n = 26, and 4 GiB at
 * n = 29, which is exactly where the extra allocation is least affordable.
 *
 * Above {@link CDF_MAX_STATE} this streams instead: draw every uniform up front,
 * sort them, then walk the state once, emitting outcomes as the running
 * cumulative probability passes each draw. Memory becomes O(shots) rather than
 * O(2ⁿ), and the state is still touched exactly once.
 *
 * Sorting reorders which shot receives which draw, but not the multiset of
 * outcomes — and `Distribution` is a histogram, with classical-register
 * accumulation likewise order-independent, so a seeded run is unchanged.
 *
 * Both branches reproduce the pinned-CDF convention exactly: the last index
 * absorbs any draw that rounding leaves above the final cumulative sum, so a
 * draw arbitrarily close to 1 can never fall off the end.
 */
export function denseSampleEach(
  d: DenseState, shots: number, rng: () => number, emit: (idx: number) => void,
): void {
  const { data } = d
  const total = 1 << d.n
  if (total <= 0 || shots <= 0) return

  if (total <= CDF_MAX_STATE) {
    const cdf = denseCdf(d)
    for (let i = 0; i < shots; i++) emit(cdfSample(cdf, rng()))
    return
  }

  const draws = new Float64Array(shots)
  for (let i = 0; i < shots; i++) draws[i] = rng()
  draws.sort()                       // numeric by default on a TypedArray

  let cum = 0
  let j = 0
  // Stop one short: the final index is pinned, matching `denseCdf`.
  for (let i = 0; i < total - 1 && j < shots; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    cum += re * re + im * im
    while (j < shots && draws[j]! <= cum) { emit(i); j++ }
  }
  while (j < shots) { emit(total - 1); j++ }
}

/** Number of non-negligible amplitudes. */
export function denseNnz(d: DenseState): number {
  const { data } = d
  const total = 1 << d.n
  let count = 0
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    if (re * re + im * im >= AMP_MIN_NORM2) count++
  }
  return count
}

// ── Work decomposition ────────────────────────────────────────────────────────
//
// Every kernel below walks a flat *work space*: a contiguous range of integers
// [0, W) where each value names one independent unit of work — one amplitude
// pair for a single-qubit gate, one 2-bit context for a two-qubit gate, and so
// on. A work item never touches an amplitude another item touches, so any
// partition of [0, W) can run concurrently and in any order, and the result is
// bit-identical to running the whole range on one thread.
//
// That property is what `dense-parallel.ts` divides across worker threads. It is
// also why each kernel takes `(lo, hi)`: the serial call passes the whole range,
// a worker passes its slice, and both go through the same code.
//
// The work index is turned into an amplitude index by *inserting a zero bit* at
// each acted-on qubit position, low to high — `spread1`/`spread2` inline that
// below. Enumerating contexts this way rather than scanning all 2ⁿ indices and
// rejecting the ones that do not apply is also what makes the controlled and
// permutation kernels cheap: a CNOT acts on a quarter of the index space, and
// this visits exactly that quarter instead of testing every index.
//
// Sizes are computed as `(1 << n) >>> k` rather than `1 << (n - k)` so a state
// narrower than the gate yields 0 work items instead of a negative shift count.

/** Work-space size for a gate acting on `k` qubits of an `n`-qubit state. */
export const denseWork = (n: number, k: number): number => (1 << n) >>> k

// ── Gate application ──────────────────────────────────────────────────────────

/**
 * Apply a 2×2 unitary to qubit q, over work items [lo, hi) of {@link denseWork}(n, 1).
 *
 * Work item `p` is the amplitude pair {k, k | 2^q} where k is `p` with a zero
 * bit inserted at position q. Both amplitudes are read into locals before either
 * is written, so the update is simultaneous rather than sequential.
 */
export function denseSingle(
  d: DenseState, q: number, [[a, b], [c, e]]: Gate2x2,
  lo = 0, hi = denseWork(d.n, 1),
): void {
  const { data } = d
  const stride = 1 << q
  const lowMask = stride - 1
  const ar = a.re, ai = a.im, br = b.re, bi = b.im
  const cr = c.re, ci = c.im, dr = e.re, di = e.im

  for (let p = lo; p < hi; p++) {
    const k = ((p >>> q) << (q + 1)) | (p & lowMask)
    const i0 = k << 1, i1 = (k + stride) << 1
    const r0 = data[i0]!, m0 = data[i0 + 1]!
    const r1 = data[i1]!, m1 = data[i1 + 1]!
    data[i0]     = ar * r0 - ai * m0 + br * r1 - bi * m1
    data[i0 + 1] = ar * m0 + ai * r0 + br * m1 + bi * r1
    data[i1]     = cr * r0 - ci * m0 + dr * r1 - di * m1
    data[i1 + 1] = cr * m0 + ci * r0 + dr * m1 + di * r1
  }
}

/**
 * Expand a two-qubit work index into the amplitude index with zero bits at both
 * `qa` and `qb`. Returns the pieces the caller needs to build the four corners.
 */
function spread2Masks(qa: number, qb: number): { lowMask: number; midMask: number; lo: number; hi: number } {
  const lo = qa < qb ? qa : qb
  const hi = qa < qb ? qb : qa
  return { lowMask: (1 << lo) - 1, midMask: (1 << (hi - 1 - lo)) - 1, lo, hi }
}

/**
 * Expand a three-qubit work index into the amplitude index with zero bits at all
 * three positions. Three ascending inserts; hot enough to inline, cold enough
 * not to warrant the two-qubit treatment.
 */
function spread3(p: number, a: number, b: number, c: number): number {
  let i = p
  i = ((i >>> a) << (a + 1)) | (i & ((1 << a) - 1))
  i = ((i >>> b) << (b + 1)) | (i & ((1 << b) - 1))
  i = ((i >>> c) << (c + 1)) | (i & ((1 << c) - 1))
  return i
}

/** Sort three bit positions ascending — the order `spread3` requires. */
const sort3 = (x: number, y: number, z: number): [number, number, number] => {
  const s = [x, y, z].sort((m, n) => m - n)
  return [s[0]!, s[1]!, s[2]!]
}

/** Swap the amplitudes at indices `i` and `j`. */
function swapAmps(data: Float64Array, i: number, j: number): void {
  const a = i << 1, b = j << 1
  const r = data[a]!, m = data[a + 1]!
  data[a] = data[b]!; data[a + 1] = data[b + 1]!
  data[b] = r;        data[b + 1] = m
}

/** Controlled-X. Pure permutation over the quarter of indices with control set. */
export function denseCNOT(
  d: DenseState, control: number, target: number,
  lo = 0, hi = denseWork(d.n, 2),
): void {
  const { data } = d
  const cm = 1 << control, tm = 1 << target
  const { lowMask, midMask, lo: l, hi: h } = spread2Masks(control, target)
  for (let p = lo; p < hi; p++) {
    const i = ((p & lowMask) | (((p >>> l) & midMask) << (l + 1)) | ((p >>> (h - 1)) << (h + 1))) | cm
    swapAmps(data, i, i | tm)
  }
}

/** SWAP two qubits. Pure permutation over the |10⟩ member of each differing pair. */
export function denseSWAP(
  d: DenseState, qa: number, qb: number,
  lo = 0, hi = denseWork(d.n, 2),
): void {
  if (qa === qb) return
  const { data } = d
  const ma = 1 << qa, mb = 1 << qb
  const { lowMask, midMask, lo: l, hi: h } = spread2Masks(qa, qb)
  for (let p = lo; p < hi; p++) {
    const i = ((p & lowMask) | (((p >>> l) & midMask) << (l + 1)) | ((p >>> (h - 1)) << (h + 1))) | ma
    swapAmps(data, i, (i ^ ma) | mb)
  }
}

/** Toffoli (CCX). Pure permutation over the eighth with both controls set. */
export function denseToffoli(
  d: DenseState, c1: number, c2: number, target: number,
  lo = 0, hi = denseWork(d.n, 3),
): void {
  const { data } = d
  const m1 = 1 << c1, m2 = 1 << c2, tm = 1 << target
  const [x, y, z] = sort3(c1, c2, target)
  for (let p = lo; p < hi; p++) {
    const i = spread3(p, x, y, z) | m1 | m2
    swapAmps(data, i, i | tm)
  }
}

/** Fredkin (CSWAP). Pure permutation. */
export function denseCSwap(
  d: DenseState, control: number, qa: number, qb: number,
  lo = 0, hi = denseWork(d.n, 3),
): void {
  const { data } = d
  const cm = 1 << control, ma = 1 << qa, mb = 1 << qb
  const [x, y, z] = sort3(control, qa, qb)
  for (let p = lo; p < hi; p++) {
    const i = spread3(p, x, y, z) | cm | ma
    swapAmps(data, i, (i ^ ma) | mb)
  }
}

/** Controlled 2×2 unitary: apply `gate` to `target` where `control` is set. */
export function denseControlled(
  d: DenseState, control: number, target: number, [[a, b], [c, e]]: Gate2x2,
  lo = 0, hi = denseWork(d.n, 2),
): void {
  if (control === target) throw new TypeError(`control and target qubits must differ (got ${control})`)
  const { data } = d
  const cm = 1 << control, tm = 1 << target
  const { lowMask, midMask, lo: l, hi: h } = spread2Masks(control, target)
  const ar = a.re, ai = a.im, br = b.re, bi = b.im
  const cr = c.re, ci = c.im, dr = e.re, di = e.im

  for (let p = lo; p < hi; p++) {
    const i = ((p & lowMask) | (((p >>> l) & midMask) << (l + 1)) | ((p >>> (h - 1)) << (h + 1))) | cm
    const i0 = i << 1, i1 = (i | tm) << 1
    const r0 = data[i0]!, m0 = data[i0 + 1]!
    const r1 = data[i1]!, m1 = data[i1 + 1]!
    data[i0]     = ar * r0 - ai * m0 + br * r1 - bi * m1
    data[i0 + 1] = ar * m0 + ai * r0 + br * m1 + bi * r1
    data[i1]     = cr * r0 - ci * m0 + dr * r1 - di * m1
    data[i1 + 1] = cr * m0 + ci * r0 + dr * m1 + di * r1
  }
}

/**
 * Apply a 4×4 unitary to qubits (a, b).
 *
 * Local index ordering matches the sparse backend: qubit `a` is the MSB of the
 * 2-bit local index, so rows/columns run |00⟩, |01⟩, |10⟩, |11⟩ with `a` first.
 */
export function denseTwo(
  d: DenseState, qa: number, qb: number, gate: Gate4x4,
  ctxLo = 0, ctxHi = denseWork(d.n, 2),
): void {
  const { data } = d
  const ma = 1 << qa, mb = 1 << qb
  const g = flatten4(gate)

  // Enumerate the 2ⁿ⁻² contexts directly with `spread`, rather than walking all
  // 2ⁿ indices and rejecting three quarters of them on a branch.
  const lo = qa < qb ? qa : qb
  const hi = qa < qb ? qb : qa
  const lowMask = (1 << lo) - 1
  const midMask = (1 << (hi - 1 - lo)) - 1

  const g00r = g[0]!,  g00i = g[1]!,  g01r = g[2]!,  g01i = g[3]!
  const g02r = g[4]!,  g02i = g[5]!,  g03r = g[6]!,  g03i = g[7]!
  const g10r = g[8]!,  g10i = g[9]!,  g11r = g[10]!, g11i = g[11]!
  const g12r = g[12]!, g12i = g[13]!, g13r = g[14]!, g13i = g[15]!
  const g20r = g[16]!, g20i = g[17]!, g21r = g[18]!, g21i = g[19]!
  const g22r = g[20]!, g22i = g[21]!, g23r = g[22]!, g23i = g[23]!
  const g30r = g[24]!, g30i = g[25]!, g31r = g[26]!, g31i = g[27]!
  const g32r = g[28]!, g32i = g[29]!, g33r = g[30]!, g33i = g[31]!

  for (let c = ctxLo; c < ctxHi; c++) {
    const i = (c & lowMask) | (((c >>> lo) & midMask) << (lo + 1)) | ((c >>> (hi - 1)) << (hi + 1))
    const p0 = i << 1, p1 = (i | mb) << 1, p2 = (i | ma) << 1, p3 = (i | ma | mb) << 1
    const r0 = data[p0]!, m0 = data[p0 + 1]!
    const r1 = data[p1]!, m1 = data[p1 + 1]!
    const r2 = data[p2]!, m2 = data[p2 + 1]!
    const r3 = data[p3]!, m3 = data[p3 + 1]!

    data[p0]     = g00r * r0 - g00i * m0 + g01r * r1 - g01i * m1 + g02r * r2 - g02i * m2 + g03r * r3 - g03i * m3
    data[p0 + 1] = g00r * m0 + g00i * r0 + g01r * m1 + g01i * r1 + g02r * m2 + g02i * r2 + g03r * m3 + g03i * r3
    data[p1]     = g10r * r0 - g10i * m0 + g11r * r1 - g11i * m1 + g12r * r2 - g12i * m2 + g13r * r3 - g13i * m3
    data[p1 + 1] = g10r * m0 + g10i * r0 + g11r * m1 + g11i * r1 + g12r * m2 + g12i * r2 + g13r * m3 + g13i * r3
    data[p2]     = g20r * r0 - g20i * m0 + g21r * r1 - g21i * m1 + g22r * r2 - g22i * m2 + g23r * r3 - g23i * m3
    data[p2 + 1] = g20r * m0 + g20i * r0 + g21r * m1 + g21i * r1 + g22r * m2 + g22i * r2 + g23r * m3 + g23i * r3
    data[p3]     = g30r * r0 - g30i * m0 + g31r * r1 - g31i * m1 + g32r * r2 - g32i * m2 + g33r * r3 - g33i * m3
    data[p3 + 1] = g30r * m0 + g30i * r0 + g31r * m1 + g31i * r1 + g32r * m2 + g32i * r2 + g33r * m3 + g33i * r3
  }
}

/**
 * Controlled-√iSWAP on (a, b) when `control` is set.
 *
 * |c00⟩ and |c11⟩ are untouched; the |c01⟩/|c10⟩ subspace mixes through
 * [[1/√2, i/√2], [i/√2, 1/√2]].
 */
export function denseCsrSwap(
  d: DenseState, control: number, qa: number, qb: number,
  lo = 0, hi = denseWork(d.n, 3),
): void {
  const { data } = d
  const cm = 1 << control, ma = 1 << qa, mb = 1 << qb
  const s = Math.SQRT1_2
  const [x, y, z] = sort3(control, qa, qb)

  for (let w = lo; w < hi; w++) {
    // Act once per (control=1, a=0, b=1) representative of each mixing pair.
    const i = spread3(w, x, y, z) | cm | mb
    const j = (i ^ mb) | ma           // partner: a=1, b=0
    const p = i << 1, q = j << 1
    const r0 = data[p]!, m0 = data[p + 1]!
    const r1 = data[q]!, m1 = data[q + 1]!
    // out0 = s·in0 + i·s·in1 ; out1 = i·s·in0 + s·in1
    data[p]     = s * r0 - s * m1
    data[p + 1] = s * m0 + s * r1
    data[q]     = s * r1 - s * m0
    data[q + 1] = s * m1 + s * r0
  }
}

/**
 * Apply a 2^k × 2^k unitary to `qs` (qs[0] = MSB of the local index).
 *
 * Allocates two scratch buffers of length 2^k, reused across all contexts.
 */
export function denseUnitary(
  d: DenseState, qs: readonly number[], matrix: readonly (readonly Complex[])[],
  ctxLo = 0, ctxHi = denseWork(d.n, qs.length),
): void {
  const k = qs.length
  const dim = 1 << k
  const { data } = d

  // Flatten the matrix once: [re, im] per entry, row-major. Reading it as f64
  // from a contiguous buffer keeps the inner product off the object heap, where
  // every `matrix[r][c].re` was a pointer chase.
  const g = new Float64Array(dim * dim * 2)
  for (let r = 0; r < dim; r++) {
    const row = matrix[r]!
    for (let c = 0; c < dim; c++) {
      const v = row[c]!
      g[(r * dim + c) * 2]     = v.re
      g[(r * dim + c) * 2 + 1] = v.im
    }
  }

  // Byte offset of each local basis state relative to its context index. The
  // gate's first qubit is the high bit of the local index, matching applyUnitary.
  const offset = new Int32Array(dim)
  for (let local = 0; local < dim; local++) {
    let o = 0
    for (let bit = 0; bit < k; bit++) {
      if ((local >> (k - 1 - bit)) & 1) o |= 1 << qs[bit]!
    }
    offset[local] = o
  }

  // Positions to re-insert as zero bits when expanding a context counter, low to
  // high, so `spread` below walks contexts without testing every index in 2ⁿ.
  const lowMasks = Int32Array.from([...qs].sort((a, b) => a - b), q => (1 << q) - 1)

  const re = new Float64Array(dim), im = new Float64Array(dim)

  for (let ctx = ctxLo; ctx < ctxHi; ctx++) {
    let i = ctx
    for (let h = 0; h < k; h++) {
      const lowMask = lowMasks[h]!
      i = (i & lowMask) | ((i & ~lowMask) << 1)
    }

    for (let local = 0; local < dim; local++) {
      const p = (i | offset[local]!) << 1
      re[local] = data[p]!; im[local] = data[p + 1]!
    }

    let gi = 0
    for (let r = 0; r < dim; r++) {
      let sr = 0, si = 0
      for (let c = 0; c < dim; c++) {
        const ar = g[gi]!, ai = g[gi + 1]!
        gi += 2
        sr += ar * re[c]! - ai * im[c]!
        si += ar * im[c]! + ai * re[c]!
      }
      const p = (i | offset[r]!) << 1
      data[p] = sr; data[p + 1] = si
    }
  }
}

// ── Execution strategy ────────────────────────────────────────────────────────

/**
 * How a dense gate gets applied — the seam between one thread and many.
 *
 * `hybrid.ts` calls through this rather than the kernels directly, so the
 * decision about *where* the work runs is made once, when the state is created,
 * instead of at every gate. {@link SERIAL_EXEC} runs the kernel on the calling
 * thread; the pool in `dense-parallel.ts` splits the same kernel's work space
 * across workers. Both produce bit-identical states — see the note on work
 * decomposition above for why that is guaranteed rather than hoped for.
 */
export interface DenseExec {
  single(d: DenseState, q: number, gate: Gate2x2): void
  controlled(d: DenseState, control: number, target: number, gate: Gate2x2): void
  cnot(d: DenseState, control: number, target: number): void
  swap(d: DenseState, a: number, b: number): void
  toffoli(d: DenseState, c1: number, c2: number, target: number): void
  cswap(d: DenseState, control: number, a: number, b: number): void
  csrswap(d: DenseState, control: number, a: number, b: number): void
  two(d: DenseState, a: number, b: number, gate: Gate4x4): void
  unitary(d: DenseState, qs: readonly number[], matrix: readonly (readonly Complex[])[]): void
}

/**
 * Run every gate on the calling thread. The default, and the reference.
 *
 * The kernels are referenced directly rather than wrapped: their range
 * parameters are optional and default to the whole work space, which is exactly
 * what running serially means.
 */
export const SERIAL_EXEC: DenseExec = {
  single:     denseSingle,
  controlled: denseControlled,
  cnot:       denseCNOT,
  swap:       denseSWAP,
  toffoli:    denseToffoli,
  cswap:      denseCSwap,
  csrswap:    denseCsrSwap,
  two:        denseTwo,
  unitary:    denseUnitary,
}

// ── Measurement and channel primitives ────────────────────────────────────────
//
// These back the noise / mid-circuit path, where the state is repeatedly
// projected, rescaled and sampled between gates.

/** Independent copy — needed by Kraus channels, which trial every operator. */
export const denseClone = (d: DenseState): DenseState =>
  ({ n: d.n, data: d.data.slice() })

/** Total ⟨ψ|ψ⟩. Not 1 after an unnormalised Kraus operator. */
export function denseNorm2(d: DenseState): number {
  const { data } = d
  let s = 0
  for (let i = 0; i < data.length; i += 2) s += data[i]! * data[i]! + data[i + 1]! * data[i + 1]!
  return s
}

/** Multiply every amplitude by a real scalar. */
export function denseScale(d: DenseState, f: number): void {
  const { data } = d
  for (let i = 0; i < data.length; i++) data[i]! *= f
}

/** Probability that qubit q reads 1. */
export function denseProbOne(d: DenseState, q: number): number {
  const { data } = d
  const total = 1 << d.n
  const mask = 1 << q
  let p = 0
  for (let i = 0; i < total; i++) {
    if ((i & mask) !== 0) {
      const re = data[i << 1]!, im = data[(i << 1) | 1]!
      p += re * re + im * im
    }
  }
  return p
}

/**
 * Project onto `outcome` on qubit q and rescale the surviving branch by `inv`.
 * The other branch is zeroed.
 */
export function denseCollapse(d: DenseState, q: number, outcome: 0 | 1, inv: number): void {
  const { data } = d
  const total = 1 << d.n
  const mask = 1 << q
  for (let i = 0; i < total; i++) {
    const keep = ((i & mask) !== 0) === (outcome === 1)
    const p = i << 1
    if (keep) { data[p]! *= inv; data[p + 1]! *= inv }
    else      { data[p] = 0;     data[p + 1] = 0 }
  }
}

/** Scale the |1⟩ branch of qubit q by `sOne` and the |0⟩ branch by `sZero`. */
export function denseScaleBranch(d: DenseState, q: number, sOne: number, sZero: number): void {
  const { data } = d
  const total = 1 << d.n
  const mask = 1 << q
  for (let i = 0; i < total; i++) {
    const s = (i & mask) !== 0 ? sOne : sZero
    const p = i << 1
    data[p]! *= s; data[p + 1]! *= s
  }
}

/**
 * Amplitude-damping jump on qubit q: every |1⟩ amplitude moves to its |0⟩
 * partner scaled by `inv`, and the original |1⟩ slot is cleared. Each |1⟩ index
 * has exactly one |0⟩ partner, so this is a move, never an accumulation.
 */
export function denseDecay(d: DenseState, q: number, inv: number): void {
  const { data } = d
  const total = 1 << d.n
  const stride = 1 << q
  for (let i = 0; i < total; i++) {
    if ((i & stride) !== 0) continue          // visit each pair from its |0⟩ member
    const p0 = i << 1, p1 = (i | stride) << 1
    data[p0]     = data[p1]! * inv
    data[p0 + 1] = data[p1 + 1]! * inv
    data[p1] = 0; data[p1 + 1] = 0
  }
}

/**
 * Sample one basis index.
 *
 * Walks indices in ascending order, matching the sparse sampler's sorted walk,
 * so a given RNG draw picks the same outcome in either representation.
 */
export function denseSample(d: DenseState, rand: number): bigint {
  const { data } = d
  const total = 1 << d.n
  let cum = 0
  let last = 0
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    const p = re * re + im * im
    if (p <= 0) continue
    last = i
    cum += p
    if (rand <= cum) return BigInt(i)
  }
  return BigInt(last)
}

/** Pack a 4×4 complex matrix into a flat f64 array: [re, im] per entry, row-major. */
function flatten4(gate: Gate4x4): Float64Array {
  const g = new Float64Array(32)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const v = gate[r]![c]!
      g[(r << 3) | (c << 1)]     = v.re
      g[(r << 3) | (c << 1) | 1] = v.im
    }
  }
  return g
}
