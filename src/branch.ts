/**
 * Branch backend: the state as a sum of product states.
 *
 * Each branch holds one unnormalised 2-vector per qubit, so a branch costs O(n)
 * numbers regardless of what basis its qubits are in. A gate on k qubits is
 * applied to the k-qubit product inside each branch and the result is split
 * back into product states by fixing all but one of the k qubits in the
 * computational basis — a controlled gate whose control is a basis state in
 * that branch therefore never splits, and a cphase splits only when both sides
 * are superposed. Branches that agree on every basis-state qubit are summed.
 *
 * Exact for any circuit. Cheap exactly when the state is a short sum of
 * product states, which is a different regime from the ones the statevector
 * (few non-zero amplitudes), MPS (low entanglement across a line) and
 * stabilizer-rank (few stabilizer terms) backends cover. The motivating case is
 * Shor's: inside the Beauregard arithmetic every register is an integer or the
 * Fourier transform of one, both product states, and the branch count equals
 * the multiplicative order being found — so the cost of simulating the circuit
 * honestly is the period, and nothing else.
 */
import type { Complex } from './complex.js'
import type { FlatOp } from './circuit.js'

const EPS = 1e-12
const SQ2 = Math.SQRT1_2

type Mat = readonly (readonly Complex[])[]

export interface BranchOptions {
  /** Throw once the number of branches would exceed this. Default: unbounded. */
  maxBranches?: number
}

export class BranchState {
  readonly n: number
  /** Each branch: Float64Array(4n) laid out [re0, im0, re1, im1] per qubit. */
  branches: Float64Array[]
  /** Largest branch count reached so far. */
  peakBranches = 1
  readonly #cap: number

  constructor(n: number, options: BranchOptions = {}) {
    this.n = n
    this.#cap = options.maxBranches ?? Infinity
    const v = new Float64Array(4 * n)
    for (let q = 0; q < n; q++) v[4 * q] = 1
    this.branches = [v]
  }

  /** Start from a computational basis state (bitstring, q0 leftmost). */
  static fromBitstring(bits: string, options?: BranchOptions): BranchState {
    const s = new BranchState(bits.length, options)
    const v = s.branches[0]!
    for (let q = 0; q < bits.length; q++) {
      if (bits[q] === '1') { v[4 * q] = 0; v[4 * q + 2] = 1 }
    }
    return s
  }

  clone(): BranchState {
    const s = new BranchState(this.n, { maxBranches: this.#cap })
    s.branches = this.branches.map(v => new Float64Array(v))
    s.peakBranches = this.peakBranches
    return s
  }

  // ── Per-qubit helpers ─────────────────────────────────────────────────────

  /** 0 or 1 if qubit `q` is a computational basis state in branch `v`, else -1. */
  static basis(v: Float64Array, q: number): 0 | 1 | -1 {
    const o = 4 * q
    const n0 = v[o]! * v[o]! + v[o + 1]! * v[o + 1]!
    const n1 = v[o + 2]! * v[o + 2]! + v[o + 3]! * v[o + 3]!
    return n1 <= EPS * n0 ? 0 : n0 <= EPS * n1 ? 1 : -1
  }

  /** Squared norm of the whole branch: product of per-qubit squared norms. */
  static weight(v: Float64Array): number {
    let w = 1
    for (let i = 0; i < v.length; i += 4) w *= v[i]! * v[i]! + v[i + 1]! * v[i + 1]! + v[i + 2]! * v[i + 2]! + v[i + 3]! * v[i + 3]!
    return w
  }

  #grew(): void {
    if (this.branches.length > this.peakBranches) this.peakBranches = this.branches.length
    if (this.branches.length > this.#cap) {
      throw new RangeError(`runBranch: ${this.branches.length} branches exceeds maxBranches ${this.#cap} — the state is not a short sum of product states; use run() or runMps()`)
    }
  }

  // ── Gates ─────────────────────────────────────────────────────────────────

  /**
   * Apply a 2^k × 2^k matrix to qubits `qs` (qs[0] is the most significant
   * index bit) in every branch, splitting as needed.
   */
  apply(qs: readonly number[], m: Mat): void {
    const k = qs.length, dim = 1 << k
    const out: Float64Array[] = []
    const inRe = new Float64Array(dim), inIm = new Float64Array(dim)
    const outRe = new Float64Array(dim), outIm = new Float64Array(dim)
    for (const v of this.branches) {
      // Product of the k local vectors → local input amplitudes.
      for (let i = 0; i < dim; i++) {
        let ar = 1, ai = 0
        for (let j = 0; j < k; j++) {
          const bit = (i >> (k - 1 - j)) & 1
          const o = 4 * qs[j]! + 2 * bit
          const br = v[o]!, bi = v[o + 1]!
          const nr = ar * br - ai * bi
          ai = ar * bi + ai * br; ar = nr
        }
        inRe[i] = ar; inIm[i] = ai
      }
      for (let r = 0; r < dim; r++) {
        let sr = 0, si = 0
        const row = m[r]!
        for (let c = 0; c < dim; c++) {
          const g = row[c]!
          if (g.re === 0 && g.im === 0) continue
          sr += g.re * inRe[c]! - g.im * inIm[c]!
          si += g.re * inIm[c]! + g.im * inRe[c]!
        }
        outRe[r] = sr; outIm[r] = si
      }
      // Pick the free qubit whose fixing of the others yields the fewest
      // branches. Ties go to the last qubit — the target of a controlled gate —
      // so a control in superposition is what gets split, and a register the
      // gate merely phases keeps its product structure.
      let bestF = k - 1, bestCount = Infinity
      for (let f = k - 1; f >= 0; f--) {
        let count = 0
        for (let p = 0; p < dim; p++) {
          if ((p >> (k - 1 - f)) & 1) continue            // enumerate patterns with free bit = 0
          const p1 = p | (1 << (k - 1 - f))
          if (outRe[p]! * outRe[p]! + outIm[p]! * outIm[p]! + outRe[p1]! * outRe[p1]! + outIm[p1]! * outIm[p1]! > 1e-28) count++
        }
        if (count < bestCount) { bestCount = count; bestF = f }
      }
      const fbit = 1 << (k - 1 - bestF)
      for (let p = 0; p < dim; p++) {
        if (p & fbit) continue
        const p1 = p | fbit
        const w = outRe[p]! * outRe[p]! + outIm[p]! * outIm[p]! + outRe[p1]! * outRe[p1]! + outIm[p1]! * outIm[p1]!
        if (w <= 1e-28) continue
        const b = bestCount === 1 ? v : new Float64Array(v)
        for (let j = 0; j < k; j++) {
          const o = 4 * qs[j]!
          if (j === bestF) { b[o] = outRe[p]!; b[o + 1] = outIm[p]!; b[o + 2] = outRe[p1]!; b[o + 3] = outIm[p1]! }
          else { const bit = (p >> (k - 1 - j)) & 1; b[o] = b[o + 1] = b[o + 2] = b[o + 3] = 0; b[o + 2 * bit] = 1 }
        }
        out.push(b)
      }
    }
    this.branches = out
    this.#grew()
  }

  /** Exchange the vectors of two qubits in every branch — never splits. */
  swap(a: number, b: number): void {
    for (const v of this.branches) {
      for (let i = 0; i < 4; i++) { const t = v[4 * a + i]!; v[4 * a + i] = v[4 * b + i]!; v[4 * b + i] = t }
    }
  }

  // ── Merging and measurement ───────────────────────────────────────────────

  /**
   * Sum branches that agree on every basis-state qubit and share at most one
   * superposed qubit. Without this, interference between branches would be
   * carried as separate terms forever and the count would grow without bound.
   */
  merge(): void {
    if (this.branches.length < 2) return
    const map = new Map<string, Float64Array>()
    const keep: Float64Array[] = []
    const n = this.n
    for (const v of this.branches) {
      let key = '', free = -1, ok = true
      for (let q = 0; q < n; q++) {
        const b = BranchState.basis(v, q)
        if (b === -1) { if (free !== -1) { ok = false; break } free = q; key += '*' } else key += b
      }
      if (!ok) { keep.push(v); continue }
      const prev = map.get(key)
      if (!prev) { map.set(key, v); continue }
      // Fold each branch's basis-qubit scalars into the free qubit (or qubit 0
      // when fully basis), then add the two free vectors.
      const f = free === -1 ? 0 : free
      const fold = (w: Float64Array): [number, number] => {
        let sr = 1, si = 0
        for (let q = 0; q < n; q++) {
          if (q === f) continue
          const o = 4 * q, b = BranchState.basis(w, q) === 1 ? 2 : 0
          const ar = w[o + b]!, ai = w[o + b + 1]!
          const nr = sr * ar - si * ai; si = sr * ai + si * ar; sr = nr
          w[o] = w[o + 1] = w[o + 2] = w[o + 3] = 0; w[o + b] = 1
        }
        return [sr, si]
      }
      const [pr, pi] = fold(prev), [vr, vi] = fold(v)
      const o = 4 * f
      const r0 = prev[o]!, i0 = prev[o + 1]!, r1 = prev[o + 2]!, i1 = prev[o + 3]!
      prev[o]     = pr * r0 - pi * i0 + vr * v[o]!     - vi * v[o + 1]!
      prev[o + 1] = pr * i0 + pi * r0 + vr * v[o + 1]! + vi * v[o]!
      prev[o + 2] = pr * r1 - pi * i1 + vr * v[o + 2]! - vi * v[o + 3]!
      prev[o + 3] = pr * i1 + pi * r1 + vr * v[o + 3]! + vi * v[o + 2]!
    }
    for (const v of map.values()) if (BranchState.weight(v) > 1e-24) keep.push(v)
    this.branches = keep
  }

  /** True when every pair of branches differs on some qubit where both are basis states. */
  #orthogonal(): boolean {
    const n = this.n, bs = this.branches
    for (let i = 0; i < bs.length; i++) {
      for (let j = i + 1; j < bs.length; j++) {
        let orth = false
        for (let q = 0; q < n && !orth; q++) {
          const a = BranchState.basis(bs[i]!, q), b = BranchState.basis(bs[j]!, q)
          orth = a !== -1 && b !== -1 && a !== b
        }
        if (!orth) return false
      }
    }
    return true
  }

  /**
   * Probability that qubit `q` reads 1. Orthogonal branches add their weights;
   * otherwise the full Gram sum over branch pairs is taken, which is exact but
   * quadratic in the branch count.
   */
  prob1(q: number): number {
    const bs = this.branches, n = this.n, o = 4 * q
    if (this.#orthogonal()) {
      let p1 = 0, tot = 0
      for (const v of bs) {
        const w = BranchState.weight(v)
        const nq = v[o]! * v[o]! + v[o + 1]! * v[o + 1]! + v[o + 2]! * v[o + 2]! + v[o + 3]! * v[o + 3]!
        tot += w; p1 += w * (v[o + 2]! * v[o + 2]! + v[o + 3]! * v[o + 3]!) / nq
      }
      return tot > 0 ? p1 / tot : 0
    }
    let p1 = 0, tot = 0
    for (let i = 0; i < bs.length; i++) {
      for (let j = 0; j < bs.length; j++) {
        const a = bs[i]!, b = bs[j]!
        // ⟨a|b⟩ restricted: product over qubits ≠ q of ⟨a_q'|b_q'⟩, times ⟨a_q|1⟩⟨1|b_q⟩ (p1) or ⟨a_q|b_q⟩ (tot)
        let gr = 1, gi = 0
        for (let qq = 0; qq < n; qq++) {
          if (qq === q) continue
          const oo = 4 * qq
          const ir = a[oo]! * b[oo]! + a[oo + 1]! * b[oo + 1]! + a[oo + 2]! * b[oo + 2]! + a[oo + 3]! * b[oo + 3]!
          const ii = a[oo]! * b[oo + 1]! - a[oo + 1]! * b[oo]! + a[oo + 2]! * b[oo + 3]! - a[oo + 3]! * b[oo + 2]!
          const nr = gr * ir - gi * ii; gi = gr * ii + gi * ir; gr = nr
          if (gr === 0 && gi === 0) break
        }
        const one = a[o + 2]! * b[o + 2]! + a[o + 3]! * b[o + 3]!       // real part of ⟨a_q|1⟩⟨1|b_q⟩
        const oneI = a[o + 2]! * b[o + 3]! - a[o + 3]! * b[o + 2]!
        const all = one + a[o]! * b[o]! + a[o + 1]! * b[o + 1]!
        const allI = oneI + a[o]! * b[o + 1]! - a[o + 1]! * b[o]!
        p1 += gr * one - gi * oneI
        tot += gr * all - gi * allI
      }
    }
    return tot > 0 ? Math.min(1, Math.max(0, p1 / tot)) : 0
  }

  /** Project qubit `q` onto `bit` and renormalise by the given probability. */
  project(q: number, bit: 0 | 1, p: number): void {
    const o = 4 * q, s = p > 0 ? 1 / Math.sqrt(p) : 0
    for (const v of this.branches) {
      if (bit) { v[o] = v[o + 1] = 0; v[o + 2]! *= s; v[o + 3]! *= s }
      else     { v[o + 2] = v[o + 3] = 0; v[o]! *= s; v[o + 1]! *= s }
    }
    this.branches = this.branches.filter(v => BranchState.weight(v) > 1e-24)
    if (this.branches.length === 0) throw new Error('runBranch: state vanished after projection (numerical underflow)')
  }

  /** Measure qubit `q`, collapsing the state. */
  measure(q: number, rand: () => number): 0 | 1 {
    this.merge()
    const p1 = this.prob1(q)
    const bit: 0 | 1 = rand() < p1 ? 1 : 0
    this.project(q, bit, bit ? p1 : 1 - p1)
    return bit
  }

  /** Sample every qubit (destructive). Returns the basis index with qubit 0 as bit 0. */
  sample(rand: () => number): bigint {
    let idx = 0n
    for (let q = 0; q < this.n; q++) if (this.measure(q, rand)) idx |= 1n << BigInt(q)
    return idx
  }

  /** Expand to a sparse amplitude map. Exponential — for small circuits and tests. */
  amplitudes(): Map<bigint, Complex> {
    const out = new Map<bigint, Complex>()
    const n = this.n, dim = 1 << n
    for (const v of this.branches) {
      for (let i = 0; i < dim; i++) {
        let ar = 1, ai = 0
        for (let q = 0; q < n && (ar !== 0 || ai !== 0); q++) {
          const o = 4 * q + 2 * ((i >> q) & 1)
          const nr = ar * v[o]! - ai * v[o + 1]!; ai = ar * v[o + 1]! + ai * v[o]!; ar = nr
        }
        if (ar === 0 && ai === 0) continue
        const prev = out.get(BigInt(i))
        out.set(BigInt(i), prev ? { re: prev.re + ar, im: prev.im + ai } : { re: ar, im: ai })
      }
    }
    for (const [k, a] of out) if (Math.hypot(a.re, a.im) < 1e-12) out.delete(k)
    return out
  }
}

// ── Op dispatch ──────────────────────────────────────────────────────────────

const C = (re: number, im = 0): Complex => ({ re, im })
const Z = C(0), O = C(1)
const X_MAT: Mat = [[Z, O], [O, Z]]
/** |01⟩,|10⟩ block of the controlled √SWAP on (a, b), control = MSB. */
const CSRSWAP: Mat = (() => {
  const m: Complex[][] = Array.from({ length: 8 }, (_, r) => Array.from({ length: 8 }, (_, c) => r === c ? O : Z))
  // control=1 half: indices 4..7 ↔ |c=1, a, b⟩; swap block on |01⟩ (5) and |10⟩ (6)
  m[5] = [Z, Z, Z, Z, Z, C(SQ2), C(0, SQ2), Z]
  m[6] = [Z, Z, Z, Z, Z, C(0, SQ2), C(SQ2), Z]
  return m
})()
function controlledMat(g: Mat): Mat {   // control = MSB
  return [[O, Z, Z, Z], [Z, O, Z, Z], [Z, Z, g[0]![0]!, g[0]![1]!], [Z, Z, g[1]![0]!, g[1]![1]!]]
}
function cregValue(cregs: Map<string, boolean[]>, name: string): number {
  return (cregs.get(name) ?? []).reduce((acc, b, i) => acc | (b ? 1 << i : 0), 0)
}

/** Apply flattened ops to `s`, handling measure/reset/if with `rand`. */
export function applyBranchOps(s: BranchState, ops: readonly FlatOp[], cregs: Map<string, boolean[]>, rand: () => number): void {
  for (const op of ops) {
    switch (op.kind) {
      case 'single':     s.apply([op.q], op.gate); break
      case 'controlled': s.apply([op.control, op.target], controlledMat(op.gate)); break
      case 'cnot':       s.apply([op.control, op.target], controlledMat(X_MAT)); break
      case 'two':        s.apply([op.a, op.b], op.gate); break
      case 'swap':       s.swap(op.a, op.b); break
      case 'toffoli': {
        const m: Complex[][] = Array.from({ length: 8 }, (_, r) => Array.from({ length: 8 }, (_, c) => r === c ? O : Z))
        m[6] = m[7]!.map(() => Z); m[7] = m[6].map(() => Z); m[6]![7] = O; m[7]![6] = O
        s.apply([op.c1, op.c2, op.target], m); break }
      case 'cswap': {
        const m: Complex[][] = Array.from({ length: 8 }, (_, r) => Array.from({ length: 8 }, (_, c) => r === c ? O : Z))
        m[5] = m[5]!.map(() => Z); m[6] = m[6]!.map(() => Z); m[5]![6] = O; m[6]![5] = O
        s.apply([op.control, op.a, op.b], m); break }
      case 'csrswap':    s.apply([op.control, op.a, op.b], CSRSWAP); break
      case 'unitary':    s.apply(op.qubits, op.matrix); break
      case 'barrier':    break
      case 'measure': {
        const bit = s.measure(op.q, rand)
        const arr = cregs.get(op.creg) ?? []
        arr[op.bit] = bit === 1
        cregs.set(op.creg, arr)
        break }
      case 'reset': if (s.measure(op.q, rand)) s.apply([op.q], X_MAT); break
      case 'if': if (cregValue(cregs, op.creg) === op.value) applyBranchOps(s, op.ops as readonly FlatOp[], cregs, rand); break
    }
  }
}
