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

import type { Complex } from './complex.js'
import type { Gate2x2, Gate4x4, StateVector } from './statevector.js'

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
 * Largest n this backend will allocate by default: 2^24 amplitudes × 16 bytes =
 * 256 MiB. Above this the sparse backend stays in charge regardless of fill — a
 * dense buffer would be a worse problem than a slow one. Override per call with
 * {@link DenseOptions}.
 */
export const MAX_DENSE_QUBITS = 24

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
   * Promote once the state exceeds `1 / fill` of full occupancy. Lower values
   * promote sooner. Default 8 for statevectors, 32 for density matrices.
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

/** Resolve caller options against a backend's defaults, rejecting nonsense. */
export function resolvePolicy(opts: DenseOptions | undefined, fallback: DensePolicy): DensePolicy {
  const fill = opts?.fill ?? fallback.fill
  const maxQubits = opts?.maxQubits ?? fallback.maxQubits
  if (!(fill > 0)) throw new RangeError(`dense.fill must be > 0 (got ${fill})`)
  if (!Number.isInteger(maxQubits) || maxQubits < 0) {
    throw new RangeError(`dense.maxQubits must be a non-negative integer (got ${maxQubits})`)
  }
  return { fill, maxQubits }
}

/** |0…0⟩ over n qubits. */
export function denseZero(n: number): DenseState {
  const data = new Float64Array(2 ** (n + 1))
  data[0] = 1
  return { n, data }
}

/** Materialise a sparse state as dense. Cost is O(2ⁿ) for the allocation. */
export function fromSparse(sv: StateVector, n: number): DenseState {
  const d = { n, data: new Float64Array(2 ** (n + 1)) }
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
    if (re * re + im * im >= 1e-14) sv.set(BigInt(i), { re, im })
  }
  return sv
}

/** Probability of each basis state, keyed as the sparse backend keys them. */
export function denseProbabilities(d: DenseState): Map<bigint, number> {
  const probs = new Map<bigint, number>()
  const { data } = d
  const total = 1 << d.n
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    const p = re * re + im * im
    if (p > 1e-14) probs.set(BigInt(i), p)
  }
  return probs
}

/** Number of non-negligible amplitudes. */
export function denseNnz(d: DenseState): number {
  const { data } = d
  const total = 1 << d.n
  let count = 0
  for (let i = 0; i < total; i++) {
    const re = data[i << 1]!, im = data[(i << 1) | 1]!
    if (re * re + im * im >= 1e-14) count++
  }
  return count
}

// ── Gate application ──────────────────────────────────────────────────────────

/**
 * Apply a 2×2 unitary to qubit q.
 *
 * Walks the index space in blocks of 2·stride, pairing each index with the one
 * that has bit q set. Both amplitudes are read into locals before either is
 * written, so the update is simultaneous rather than sequential.
 */
export function denseSingle(d: DenseState, q: number, [[a, b], [c, e]]: Gate2x2): void {
  const { data } = d
  const stride = 1 << q
  const total  = 1 << d.n
  const ar = a.re, ai = a.im, br = b.re, bi = b.im
  const cr = c.re, ci = c.im, dr = e.re, di = e.im

  for (let base = 0; base < total; base += stride << 1) {
    for (let k = base; k < base + stride; k++) {
      const i0 = k << 1, i1 = (k + stride) << 1
      const r0 = data[i0]!, m0 = data[i0 + 1]!
      const r1 = data[i1]!, m1 = data[i1 + 1]!
      data[i0]     = ar * r0 - ai * m0 + br * r1 - bi * m1
      data[i0 + 1] = ar * m0 + ai * r0 + br * m1 + bi * r1
      data[i1]     = cr * r0 - ci * m0 + dr * r1 - di * m1
      data[i1 + 1] = cr * m0 + ci * r0 + dr * m1 + di * r1
    }
  }
}

/** Controlled-X. Pure permutation — swap the two halves where control is set. */
export function denseCNOT(d: DenseState, control: number, target: number): void {
  const { data } = d
  const total = 1 << d.n
  const cm = 1 << control, tm = 1 << target
  for (let i = 0; i < total; i++) {
    // Visit each pair once by only acting on the member with target bit clear.
    if ((i & cm) !== 0 && (i & tm) === 0) {
      const j = i | tm
      const a = i << 1, b = j << 1
      const r = data[a]!, m = data[a + 1]!
      data[a] = data[b]!; data[a + 1] = data[b + 1]!
      data[b] = r;        data[b + 1] = m
    }
  }
}

/** SWAP two qubits. Pure permutation. */
export function denseSWAP(d: DenseState, qa: number, qb: number): void {
  if (qa === qb) return
  const { data } = d
  const total = 1 << d.n
  const ma = 1 << qa, mb = 1 << qb
  for (let i = 0; i < total; i++) {
    // Only the |10⟩ member of each differing pair, so each swap happens once.
    if ((i & ma) !== 0 && (i & mb) === 0) {
      const j = (i ^ ma) | mb
      const a = i << 1, b = j << 1
      const r = data[a]!, m = data[a + 1]!
      data[a] = data[b]!; data[a + 1] = data[b + 1]!
      data[b] = r;        data[b + 1] = m
    }
  }
}

/** Toffoli (CCX). Pure permutation. */
export function denseToffoli(d: DenseState, c1: number, c2: number, target: number): void {
  const { data } = d
  const total = 1 << d.n
  const m1 = 1 << c1, m2 = 1 << c2, tm = 1 << target
  for (let i = 0; i < total; i++) {
    if ((i & m1) !== 0 && (i & m2) !== 0 && (i & tm) === 0) {
      const j = i | tm
      const a = i << 1, b = j << 1
      const r = data[a]!, m = data[a + 1]!
      data[a] = data[b]!; data[a + 1] = data[b + 1]!
      data[b] = r;        data[b + 1] = m
    }
  }
}

/** Fredkin (CSWAP). Pure permutation. */
export function denseCSwap(d: DenseState, control: number, qa: number, qb: number): void {
  const { data } = d
  const total = 1 << d.n
  const cm = 1 << control, ma = 1 << qa, mb = 1 << qb
  for (let i = 0; i < total; i++) {
    if ((i & cm) !== 0 && (i & ma) !== 0 && (i & mb) === 0) {
      const j = (i ^ ma) | mb
      const a = i << 1, b = j << 1
      const r = data[a]!, m = data[a + 1]!
      data[a] = data[b]!; data[a + 1] = data[b + 1]!
      data[b] = r;        data[b + 1] = m
    }
  }
}

/** Controlled 2×2 unitary: apply `gate` to `target` where `control` is set. */
export function denseControlled(d: DenseState, control: number, target: number, [[a, b], [c, e]]: Gate2x2): void {
  if (control === target) throw new TypeError(`control and target qubits must differ (got ${control})`)
  const { data } = d
  const total = 1 << d.n
  const cm = 1 << control, tm = 1 << target
  const ar = a.re, ai = a.im, br = b.re, bi = b.im
  const cr = c.re, ci = c.im, dr = e.re, di = e.im

  for (let i = 0; i < total; i++) {
    if ((i & cm) === 0 || (i & tm) !== 0) continue
    const j = i | tm
    const i0 = i << 1, i1 = j << 1
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
export function denseTwo(d: DenseState, qa: number, qb: number, gate: Gate4x4): void {
  const { data } = d
  const total = 1 << d.n
  const ma = 1 << qa, mb = 1 << qb
  const g = flatten4(gate)
  const re = new Float64Array(4), im = new Float64Array(4)

  for (let i = 0; i < total; i++) {
    if ((i & ma) !== 0 || (i & mb) !== 0) continue   // iterate contexts only
    const bases = [i, i | mb, i | ma, i | ma | mb]

    for (let k = 0; k < 4; k++) {
      const p = bases[k]! << 1
      re[k] = data[p]!; im[k] = data[p + 1]!
    }
    for (let r = 0; r < 4; r++) {
      let sr = 0, si = 0
      for (let c = 0; c < 4; c++) {
        const gr = g[(r << 3) | (c << 1)]!, gi = g[(r << 3) | (c << 1) | 1]!
        sr += gr * re[c]! - gi * im[c]!
        si += gr * im[c]! + gi * re[c]!
      }
      const p = bases[r]! << 1
      data[p] = sr; data[p + 1] = si
    }
  }
}

/**
 * Controlled-√iSWAP on (a, b) when `control` is set.
 *
 * |c00⟩ and |c11⟩ are untouched; the |c01⟩/|c10⟩ subspace mixes through
 * [[1/√2, i/√2], [i/√2, 1/√2]].
 */
export function denseCsrSwap(d: DenseState, control: number, qa: number, qb: number): void {
  const { data } = d
  const total = 1 << d.n
  const cm = 1 << control, ma = 1 << qa, mb = 1 << qb
  const s = Math.SQRT1_2

  for (let i = 0; i < total; i++) {
    // Act once per (control=1, a=0, b=1) representative of each mixing pair.
    if ((i & cm) === 0 || (i & ma) !== 0 || (i & mb) === 0) continue
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
export function denseUnitary(d: DenseState, qs: readonly number[], matrix: readonly (readonly Complex[])[]): void {
  const k = qs.length
  const dim = 1 << k
  const { data } = d
  const total = 1 << d.n
  const masks = qs.map(q => 1 << q)
  const allMask = masks.reduce((acc, m) => acc | m, 0)

  const re = new Float64Array(dim), im = new Float64Array(dim)
  const bases = new Int32Array(dim)

  for (let i = 0; i < total; i++) {
    if ((i & allMask) !== 0) continue   // contexts only

    for (let local = 0; local < dim; local++) {
      let g = i
      for (let bit = 0; bit < k; bit++) {
        if ((local >> (k - 1 - bit)) & 1) g |= masks[bit]!
      }
      bases[local] = g
      const p = g << 1
      re[local] = data[p]!; im[local] = data[p + 1]!
    }

    for (let r = 0; r < dim; r++) {
      const row = matrix[r]!
      let sr = 0, si = 0
      for (let c = 0; c < dim; c++) {
        const g = row[c]!
        sr += g.re * re[c]! - g.im * im[c]!
        si += g.re * im[c]! + g.im * re[c]!
      }
      const p = bases[r]! << 1
      data[p] = sr; data[p + 1] = si
    }
  }
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
