/**
 * Phase-sensitive Clifford simulator in CH-form.
 *
 * An n-qubit stabilizer state is represented as
 *
 *   |φ⟩ = ω · U_C · U_H |s⟩                                              (Eq. 42)
 *
 * where U_H = H(v) is a tensor product of Hadamards selected by v ∈ {0,1}ⁿ, and
 * U_C is a C-type Clifford (a product of S, CZ, CX, so U_C|0ⁿ⟩ = |0ⁿ⟩) described
 * by its stabilizer tableau
 *
 *   U_C⁻¹ Z_p U_C = Z(G_p)        U_C⁻¹ X_p U_C = i^{γ_p} X(F_p) Z(M_p)  (Eq. 43)
 *
 * Unlike the CHP tableau in `clifford.ts`, this form tracks the global phase ω.
 * That is the whole point: CHP knows the stabilizer *group* and so determines the
 * state only up to phase, which makes ⟨x|φ⟩ meaningless. Here ⟨x|φ⟩ is exact, and
 * amplitudes are what a stabilizer-rank (Clifford+T) simulator superposes.
 *
 * Costs: O(n) per S/CZ/CX, O(n²) per H, O(n²) per amplitude, O(n²) per sample.
 *
 * Reference: Bravyi, Browne, Calpin, Campbell, Gosset, Howard, "Simulation of
 * quantum circuits by low-rank stabilizer decompositions", Quantum 3, 181 (2019),
 * Section 4.1. Equation numbers throughout refer to that paper.
 *
 * Unlike `Circuit`, gate methods mutate and return `this`. That is a deliberate
 * departure from the library's immutable style: a stabilizer-rank decomposition
 * holds up to millions of these, and cloning every one on every gate would cost
 * more than the simulation. Use `clone()` where a copy is wanted.
 */
import { c, ZERO, type Complex } from './complex.js'
import { expSum, dyadic } from './exp-sum.js'

/** i^k for k ∈ Z₄, as [re, im]. */
const POW_I: readonly (readonly [number, number])[] = [[1, 0], [0, 1], [-1, 0], [0, -1]]

/**
 * Parity of a word's popcount.
 *
 * Row dot products need Σ_w popcount(x_w) mod 2, and since
 * popcount(a ⊕ b) ≡ popcount(a) + popcount(b) (mod 2), the words can be XOR-folded
 * first and counted once.
 */
function parity(x: number): number {
  x ^= x >>> 16; x ^= x >>> 8; x ^= x >>> 4; x ^= x >>> 2; x ^= x >>> 1
  return x & 1
}

export class StabilizerCH {
  readonly n: number

  // Fields use TypeScript `private` rather than `#`: the hot paths destructure
  // (`const { n, W, _F } = this`), which `#` names do not support, and `walker()`
  // captures them in a closure. Methods use `#` since neither applies there.
  /** Words per tableau row, ⌈n/32⌉. */
  private readonly W: number
  // Bit-packed binary tableau blocks, ⌈n/32⌉ words per row; row p holds the
  // image of qubit p (Eq. 43). Packing costs 3n²/8 bytes instead of 3n², which
  // is what sets how many terms a stabilizer-rank decomposition can hold.
  // Bits at columns ≥ n are never written, so the tail of each row stays zero
  // and row-wise XOR and popcount need no masking.
  private readonly _F: Uint32Array
  private readonly _G: Uint32Array
  private readonly _M: Uint32Array
  /** Phase vector γ ∈ Z₄ⁿ. */
  private readonly _g: Uint8Array
  /** H-layer selector v and basis string s. */
  private readonly _v: Uint8Array
  private readonly _s: Uint8Array
  /** Global phase ω. */
  private _wRe: number
  private _wIm: number

  /** Initialise |0ⁿ⟩: G = F = I, M = 0, γ = v = s = 0, ω = 1. */
  constructor(n: number) {
    if (!Number.isInteger(n) || n < 1)
      throw new RangeError(`qubit count must be a positive integer, got ${n}`)
    this.n = n
    const W = (n + 31) >>> 5
    this.W = W
    this._F = new Uint32Array(n * W)
    this._G = new Uint32Array(n * W)
    this._M = new Uint32Array(n * W)
    this._g = new Uint8Array(n)
    this._v = new Uint8Array(n)
    this._s = new Uint8Array(n)
    this._wRe = 1
    this._wIm = 0
    for (let i = 0; i < n; i++) {
      const bit = 1 << (i & 31), w = i * W + (i >>> 5)
      this._F[w] = (this._F[w] ?? 0) | bit
      this._G[w] = (this._G[w] ?? 0) | bit
    }
  }

  /** Deep copy — stabilizer-rank simulation branches states, so this is hot. */
  clone(): StabilizerCH {
    const o = new StabilizerCH(this.n)
    o._F.set(this._F); o._G.set(this._G); o._M.set(this._M)
    o._g.set(this._g); o._v.set(this._v); o._s.set(this._s)
    o._wRe = this._wRe; o._wIm = this._wIm
    return o
  }

  /**
   * Multiply the global phase by (re + i·im).
   *
   * @internal Phase bookkeeping for the gate implementations. Callers outside
   * this class can desynchronise ω from the tableau and break normalisation.
   */
  scaleOmega(re: number, im: number): this {
    const wr = this._wRe, wi = this._wIm
    this._wRe = wr * re - wi * im
    this._wIm = wr * im + wi * re
    return this
  }

  #q(q: number): void {
    if (!Number.isInteger(q) || q < 0 || q >= this.n)
      throw new RangeError(`qubit ${q} out of range for ${this.n}-qubit state`)
  }

  // ── Tableau updates ─────────────────────────────────────────────────────────
  // Verbatim from the L[Γ] / R[Γ] rule table at the end of Section 4.1.
  // R[Γ] is U_C ← U_C Γ (used to absorb Proposition 4's W_C).
  // L[Γ] is U_C ← Γ U_C (used to apply a gate to the state).

  /** R[S_q] */
  #rS(q: number): void {
    const { n, W, _F, _M, _g } = this
    const wq = q >>> 5, bq = 1 << (q & 31)
    for (let p = 0; p < n; p++) {
      const i = p * W + wq
      if ((_F[i] ?? 0) & bq) {
        _M[i] = (_M[i] ?? 0) ^ bq
        _g[p] = ((_g[p] ?? 0) + 3) & 3
      }
    }
  }

  /** R[CZ_{q,r}] */
  #rCZ(q: number, r: number): void {
    const { n, W, _F, _M, _g } = this
    const wq = q >>> 5, bq = 1 << (q & 31)
    const wr = r >>> 5, br = 1 << (r & 31)
    for (let p = 0; p < n; p++) {
      const base = p * W
      const fq = ((_F[base + wq] ?? 0) & bq) !== 0
      const fr = ((_F[base + wr] ?? 0) & br) !== 0
      if (fr) _M[base + wq] = (_M[base + wq] ?? 0) ^ bq
      if (fq) _M[base + wr] = (_M[base + wr] ?? 0) ^ br
      if (fq && fr) _g[p] = ((_g[p] ?? 0) + 2) & 3
    }
  }

  /** R[CX_{q,r}] — control q, target r. */
  #rCX(q: number, r: number): void {
    const { n, W, _F, _G, _M } = this
    const wq = q >>> 5, bq = 1 << (q & 31)
    const wr = r >>> 5, br = 1 << (r & 31)
    for (let p = 0; p < n; p++) {
      const base = p * W
      if ((_G[base + wr] ?? 0) & br) _G[base + wq] = (_G[base + wq] ?? 0) ^ bq
      if ((_F[base + wq] ?? 0) & bq) _F[base + wr] = (_F[base + wr] ?? 0) ^ br
      if ((_M[base + wr] ?? 0) & br) _M[base + wq] = (_M[base + wq] ?? 0) ^ bq
    }
  }

  /** L[S_q] */
  #lS(q: number): void {
    const { W, _G, _M, _g } = this
    const base = q * W
    for (let w = 0; w < W; w++) _M[base + w] = (_M[base + w] ?? 0) ^ (_G[base + w] ?? 0)
    _g[q] = ((_g[q] ?? 0) + 3) & 3
  }

  /** L[CZ_{q,r}] */
  #lCZ(q: number, r: number): void {
    const { W, _G, _M } = this
    const bq = q * W, br = r * W
    for (let w = 0; w < W; w++) {
      const gq = _G[bq + w] ?? 0, gr = _G[br + w] ?? 0
      _M[bq + w] = (_M[bq + w] ?? 0) ^ gr
      _M[br + w] = (_M[br + w] ?? 0) ^ gq
    }
  }

  /** L[CX_{q,r}] — control q, target r. */
  #lCX(q: number, r: number): void {
    const { W, _F, _G, _M, _g } = this
    const bq = q * W, br = r * W
    // γ_q ← γ_q + γ_r + 2(M Fᵀ)_{q,r}, evaluated on the pre-update tableau.
    let acc = 0
    for (let w = 0; w < W; w++) acc ^= (_M[bq + w] ?? 0) & (_F[br + w] ?? 0)
    _g[q] = ((_g[q] ?? 0) + (_g[r] ?? 0) + 2 * parity(acc)) & 3
    for (let w = 0; w < W; w++) {
      _G[br + w] = (_G[br + w] ?? 0) ^ (_G[bq + w] ?? 0)
      _F[bq + w] = (_F[bq + w] ?? 0) ^ (_F[br + w] ?? 0)
      _M[bq + w] = (_M[bq + w] ?? 0) ^ (_M[br + w] ?? 0)
    }
  }

  // ── Clifford gates ──────────────────────────────────────────────────────────

  /** S = diag(1, i). */
  s(q: number): this { this.#q(q); this.#lS(q); return this }

  /** S† = S³. */
  sdg(q: number): this { this.#q(q); this.#lS(q); this.#lS(q); this.#lS(q); return this }

  /** Z = S². */
  z(q: number): this { this.#q(q); this.#lS(q); this.#lS(q); return this }

  /** X = H·Z·H. */
  x(q: number): this { return this.h(q).z(q).h(q) }

  /** Y = i·X·Z. */
  y(q: number): this { return this.z(q).x(q).scaleOmega(0, 1) }

  /** Controlled-Z (symmetric in its arguments). */
  cz(a: number, b: number): this {
    this.#q(a); this.#q(b)
    if (a === b) throw new RangeError('cz requires distinct qubits')
    this.#lCZ(a, b)
    return this
  }

  /** CNOT with control `a`, target `b`. */
  cx(a: number, b: number): this {
    this.#q(a); this.#q(b)
    if (a === b) throw new RangeError('cx requires distinct qubits')
    this.#lCX(a, b)
    return this
  }

  /** SWAP via three CNOTs. */
  swap(a: number, b: number): this { return this.cx(a, b).cx(b, a).cx(a, b) }

  /**
   * Hadamard on qubit p — the only gate that cannot be absorbed into the C-layer.
   *
   * Commuting H_p = 2^{-1/2}(X_p + Z_p) through U_C U_H via Eq. (43) splits the
   * state into two basis terms (Eq. 47); Proposition 4 folds them back into a
   * single CH-form. O(n²).
   */
  h(p: number): this {
    this.#q(p)
    const { n, W, _F, _G, _M, _g, _v, _s } = this
    const t = new Uint8Array(n), u = new Uint8Array(n)
    const base = p * W
    let alpha = 0, beta = 0
    for (let j = 0; j < n; j++) {
      const vj = _v[j] ?? 0, sj = _s[j] ?? 0, nv = vj ^ 1
      const w = j >>> 5, sh = j & 31
      const gpj = ((_G[base + w] ?? 0) >>> sh) & 1
      const fpj = ((_F[base + w] ?? 0) >>> sh) & 1
      const mpj = ((_M[base + w] ?? 0) >>> sh) & 1
      t[j] = sj ^ (gpj & vj)                                              // Eq. (48)
      u[j] = sj ^ (fpj & nv) ^ (mpj & vj)
      alpha ^= gpj & nv & sj                                              // Eq. (49)
      beta ^= (mpj & nv & sj) ^ (fpj & vj & (mpj ^ sj))
    }

    const gp = _g[p] ?? 0
    let same = true
    for (let j = 0; j < n; j++) if (t[j] !== u[j]) { same = false; break }

    if (same) {
      // ω' = ω·2^{-1/2}[(-1)^α + i^{γ_p}(-1)^β], s' = t, layers unchanged.
      const [ir, ii] = POW_I[gp]!
      const sa = alpha ? -1 : 1, sb = beta ? -1 : 1
      this.scaleOmega(Math.SQRT1_2 * (sa + sb * ir), Math.SQRT1_2 * (sb * ii))
      _s.set(t)
      return this
    }

    // Factor out (-1)^α so the residual relative phase is a pure power of i.
    this.scaleOmega(alpha ? -Math.SQRT1_2 : Math.SQRT1_2, 0)
    this.#merge(t, u, (gp + 2 * (alpha ^ beta)) & 3)
    return this
  }

  /**
   * Proposition 4: rewrite U_H(|t⟩ + i^δ|u⟩) as ω·W_C·W_H|s'⟩ for distinct t, u.
   *
   * Applies CX/CZ gates that collapse the disagreement between t and u onto a
   * single qubit q, leaving a one-qubit superposition there which is exactly
   * ω·S^a·H^b|c⟩. Everything is absorbed by right-multiplication into U_C. O(n).
   */
  #merge(t: Uint8Array, u: Uint8Array, delta: number): void {
    const { n, _v, _s } = this
    const V0: number[] = [], V1: number[] = []
    for (let i = 0; i < n; i++) {
      if (t[i] === u[i]) continue
      if ((_v[i] ?? 0) === 0) V0.push(i); else V1.push(i)
    }

    let q: number
    if (V0.length > 0) {
      q = V0[0]!
      for (let k = 1; k < V0.length; k++) this.#rCX(q, V0[k]!)
      for (const i of V1) this.#rCZ(q, i)
    } else {
      q = V1[0]!
      for (let k = 1; k < V1.length; k++) this.#rCX(V1[k]!, q)
    }

    // y and z now differ only at q; s'_i = y_i elsewhere.
    const tq = t[q] ?? 0
    const src = tq === 1 ? u : t
    for (let i = 0; i < n; i++) if (i !== q) _s[i] = src[i] ?? 0

    // (y_q, z_q) is (1,0) when t_q = 1, else (0,1). Normalise to |0⟩ + i^{d}|1⟩.
    let d = delta
    if (tq === 1) {
      const [pr, pi] = POW_I[delta]!
      this.scaleOmega(pr, pi)
      d = (4 - delta) & 3
    }

    // H^{v_q}(|0⟩ + i^d|1⟩) = ω·S^a·H^b|c⟩.
    let a: number, b: number, cc: number, wr: number, wi: number
    if ((_v[q] ?? 0) === 0) {
      a = d & 1; b = 1; cc = (d >> 1) & 1; wr = Math.SQRT2; wi = 0
    } else if (d === 0) {
      a = 0; b = 0; cc = 0; wr = Math.SQRT2; wi = 0
    } else if (d === 2) {
      a = 0; b = 0; cc = 1; wr = Math.SQRT2; wi = 0
    } else {
      a = 1; b = 1; cc = d === 1 ? 1 : 0; wr = 1; wi = d === 1 ? 1 : -1
    }

    if (a) this.#rS(q)
    _v[q] = b
    _s[q] = cc
    this.scaleOmega(wr, wi)
  }

  // ── Readout ─────────────────────────────────────────────────────────────────

  /**
   * Exact amplitude ⟨x|φ⟩, global phase included.
   *
   * `x` is indexed by qubit: `x[j]` is the value of qubit j. A string is read the
   * same way, so `'01'` means qubit 0 = 0, qubit 1 = 1. O(n²) via Eq. (55).
   */
  amplitude(x: string | ArrayLike<number>): Complex {
    const { n, W, _F, _M, _g, _v, _s } = this
    if (x.length !== n) throw new RangeError(`expected ${n} bits, got ${x.length}`)
    const bit = typeof x === 'string'
      ? (j: number) => (x.charCodeAt(j) === 49 ? 1 : 0)
      : (j: number) => (x[j] ? 1 : 0)

    // Q = ∏_{p: x_p=1} U_C⁻¹ X_p U_C, accumulated as i^μ X(au) Z(at).
    const au = new Uint32Array(W), at = new Uint32Array(W)
    let mu = 0
    for (let p = 0; p < n; p++) {
      if (!bit(p)) continue
      const base = p * W
      // Z(at)·X(F_p) = (-1)^{at·F_p} X(F_p)·Z(at)
      let acc = 0
      for (let w = 0; w < W; w++) acc ^= (at[w] ?? 0) & (_F[base + w] ?? 0)
      mu = (mu + (_g[p] ?? 0) + 2 * parity(acc)) & 3
      for (let w = 0; w < W; w++) {
        au[w] = (au[w] ?? 0) ^ (_F[base + w] ?? 0)
        at[w] = (at[w] ?? 0) ^ (_M[base + w] ?? 0)
      }
    }

    // ⟨0ⁿ| i^μ X(au) Z(at) U_H |s⟩ = i^μ ⟨au| Z(at) U_H |s⟩, factorised per qubit.
    let hcount = 0, sign = 0
    for (let j = 0; j < n; j++) {
      const w = j >>> 5, sh = j & 31
      const vj = _v[j] ?? 0, sj = _s[j] ?? 0
      const uj = ((au[w] ?? 0) >>> sh) & 1, tj = ((at[w] ?? 0) >>> sh) & 1
      if (vj === 0) {
        if (uj !== sj) return ZERO
        sign ^= tj & sj
      } else {
        hcount++
        sign ^= uj & (sj ^ tj)
      }
    }

    const [ir, ii] = POW_I[mu]!
    const k = 2 ** (-hcount / 2) * (sign ? -1 : 1)
    return c((this._wRe * ir - this._wIm * ii) * k, (this._wRe * ii + this._wIm * ir) * k)
  }

  /**
   * Incremental amplitude evaluator for single-bit walks (Eq. 57).
   *
   * `amplitude(x)` rebuilds Q_x = ∏_{p: x_p=1} U_C⁻¹X_pU_C from scratch, costing
   * O(n·W). Along a Metropolis chain successive points differ in one bit, and
   * Q_{x⊕e_j} = (U_C⁻¹X_jU_C)·Q_x is a single Pauli multiply — O(W). Evaluating
   * the result is also O(W) here rather than the per-qubit loop `amplitude` uses,
   * so a step costs O(W) instead of O(n·W): a factor of n.
   *
   * The walker is a snapshot. Applying gates to this state afterwards invalidates
   * it; make a new one.
   */
  walker(x: ArrayLike<number>): { flip(j: number): void; value(): Complex } {
    const { n, W, _F, _M, _g, _v, _s } = this
    if (x.length !== n) throw new RangeError(`expected ${n} bits, got ${x.length}`)

    // Pack s and v once; the walker then never touches per-qubit arrays.
    const sP = new Uint32Array(W), vP = new Uint32Array(W)
    let hcount = 0
    for (let j = 0; j < n; j++) {
      const w = j >>> 5, bit = 1 << (j & 31)
      if (_s[j]) sP[w] = (sP[w] ?? 0) | bit
      if (_v[j]) { vP[w] = (vP[w] ?? 0) | bit; hcount++ }
    }

    // Q_x = i^mu·X(au)·Z(at), built once for the starting point.
    const au = new Uint32Array(W), at = new Uint32Array(W)
    let mu = 0
    for (let p = 0; p < n; p++) {
      if (!x[p]) continue
      const base = p * W
      let acc = 0
      for (let w = 0; w < W; w++) acc ^= (at[w] ?? 0) & (_F[base + w] ?? 0)
      mu = (mu + (_g[p] ?? 0) + 2 * parity(acc)) & 3
      for (let w = 0; w < W; w++) {
        au[w] = (au[w] ?? 0) ^ (_F[base + w] ?? 0)
        at[w] = (at[w] ?? 0) ^ (_M[base + w] ?? 0)
      }
    }

    const scale = 2 ** (-hcount / 2)
    const wRe = this._wRe, wIm = this._wIm

    return {
      // Q ← (i^{γ_j}X(F_j)Z(M_j))·Q, using Z(M_j)X(au) = (-1)^{M_j·au}X(au)Z(M_j).
      // Applying it twice restores Q, so a flip works in both directions.
      flip(j: number): void {
        const base = j * W
        let acc = 0
        for (let w = 0; w < W; w++) acc ^= (_M[base + w] ?? 0) & (au[w] ?? 0)
        mu = (mu + (_g[j] ?? 0) + 2 * parity(acc)) & 3
        for (let w = 0; w < W; w++) {
          au[w] = (au[w] ?? 0) ^ (_F[base + w] ?? 0)
          at[w] = (at[w] ?? 0) ^ (_M[base + w] ?? 0)
        }
      },
      value(): Complex {
        let miss = 0, signAcc = 0
        for (let w = 0; w < W; w++) {
          const auw = au[w] ?? 0, atw = at[w] ?? 0, sw = sP[w] ?? 0, vw = vP[w] ?? 0
          miss |= (auw ^ sw) & ~vw                       // off-layer disagreement
          signAcc ^= (atw & sw & ~vw) ^ (auw & (sw ^ atw) & vw)
        }
        if (miss !== 0) return ZERO
        const [ir, ii] = POW_I[mu]!
        const k = scale * (parity(signAcc) ? -1 : 1)
        return c((wRe * ir - wIm * ii) * k, (wRe * ii + wIm * ir) * k)
      },
    }
  }

  /**
   * Inner product ⟨φ|φ_A⟩ with the equatorial stabilizer state
   * φ_A = 2^{-n/2} Σ_x i^{xAxᵀ}|x⟩ (Lemma 3).
   *
   * `A` is symmetric and packed n×n with diagonal entries in Z₄ and off-diagonal
   * entries in {0,1} — exactly the matrices `randomEquatorial` draws from.
   *
   * Conjugating the C-layer turns φ_A into another quadratic-form superposition
   * (Eq. 66) whose overlap with U_H|s⟩ collapses to a single exponential sum over
   * the H-layer support, so the whole thing costs O(n³) instead of O(2ⁿ). This is
   * the primitive behind norm estimation.
   */
  innerProductEquatorial(A: Uint8Array): Complex {
    const { n, W, _F, _G, _M, _g, _v, _s } = this
    if (A.length !== n * n) throw new RangeError(`expected a ${n}×${n} matrix, got ${A.length} entries`)

    // J: diagonal γ, off-diagonal (M Fᵀ) mod 2. Then fold A into it.
    const AJ = new Uint8Array(n * n)
    for (let a = 0; a < n; a++) {
      AJ[a * n + a] = ((A[a * n + a] ?? 0) + (_g[a] ?? 0)) & 3
      for (let b = a + 1; b < n; b++) {
        let acc = 0
        for (let w = 0; w < W; w++) acc ^= (_M[a * W + w] ?? 0) & (_F[b * W + w] ?? 0)
        const e = ((A[a * n + b] ?? 0) ^ parity(acc)) & 1
        AJ[a * n + b] = e
        AJ[b * n + a] = e
      }
    }

    // K = Gᵀ·AJ·G, kept mod 4. Two O(n³) passes rather than one O(n⁴) triple sum.
    // Off-diagonal shifts of 2 in AJ cancel on the diagonal of K because AJ is
    // symmetric, so reducing them mod 2 above is safe at mod-4 precision here.
    const T = new Uint8Array(n * n)
    for (let cRow = 0; cRow < n; cRow++) {
      for (let b = 0; b < n; b++) {
        let acc = 0
        const wb = b >>> 5, bb = 1 << (b & 31)
        for (let d = 0; d < n; d++) if ((_G[d * W + wb] ?? 0) & bb) acc += AJ[cRow * n + d] ?? 0
        T[cRow * n + b] = acc & 3
      }
    }
    const K = new Uint8Array(n * n)
    for (let a = 0; a < n; a++) {
      const wa = a >>> 5, ba = 1 << (a & 31)
      for (let b = 0; b < n; b++) {
        let acc = 0
        for (let cRow = 0; cRow < n; cRow++) if ((_G[cRow * W + wa] ?? 0) & ba) acc += T[cRow * n + b] ?? 0
        K[a * n + b] = acc & 3
      }
    }

    // u = s + sK (mod 2); it shifts the diagonal of the restricted form by 2u.
    const u = new Uint8Array(n)
    for (let b = 0; b < n; b++) {
      let acc = 0
      for (let a = 0; a < n; a++) if (_s[a]) acc += K[a * n + b] ?? 0
      u[b] = ((_s[b] ?? 0) + acc) & 1
    }

    // Restrict K + 2·diag(u) to the H-layer support.
    const idx: number[] = []
    for (let j = 0; j < n; j++) if (_v[j]) idx.push(j)
    const m = idx.length
    const B = new Uint8Array(m * m)
    for (let i = 0; i < m; i++) {
      const a = idx[i]!
      B[i * m + i] = ((K[a * n + a] ?? 0) + 2 * (u[a] ?? 0)) & 3
      for (let j = i + 1; j < m; j++) {
        const e = (K[a * n + idx[j]!] ?? 0) & 1
        B[i * m + j] = e
        B[j * m + i] = e
      }
    }

    const z = expSum(B, m)
    if (z.re.sign === 0 && z.im.sign === 0) return ZERO

    // i^{sKsᵀ}·(-1)^{s·v}, then the shared 2^{-(n+|v|)/2} scale.
    let e = 0, sv = 0
    for (let a = 0; a < n; a++) {
      if (!_s[a]) continue
      sv ^= _v[a] ?? 0
      e += K[a * n + a] ?? 0
      for (let b = a + 1; b < n; b++) if (_s[b]) e += 2 * (K[a * n + b] ?? 0)
    }
    const [pr, pi] = POW_I[e & 3]!
    const scale = (sv ? -1 : 1) * 2 ** (-(n + m) / 2)

    // (re + i·im)·(pr + i·pi)·scale, then multiply by conj(ω).
    const zr = dyadic(z.re), zi = dyadic(z.im)
    const re = (zr * pr - zi * pi) * scale
    const im = (zr * pi + zi * pr) * scale
    return c(re * this._wRe + im * this._wIm, im * this._wRe - re * this._wIm)
  }

  /**
   * Draw one basis state from |⟨x|φ⟩|², returned as a per-qubit bit array.
   *
   * The distribution is uniform over an affine subspace: pick w agreeing with s
   * off the H-layer and uniformly random on it, then x = wGᵀ. O(n²).
   */
  sample(rand: () => number): Uint8Array {
    const { n, W, _G, _v, _s } = this
    const packed = new Uint32Array(W)
    for (let j = 0; j < n; j++) {
      const flip = (_v[j] ?? 0) === 1 && rand() < 0.5 ? 1 : 0
      if (((_s[j] ?? 0) ^ flip) === 1) {
        const w = j >>> 5
        packed[w] = (packed[w] ?? 0) | (1 << (j & 31))
      }
    }
    const x = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      const base = i * W
      let acc = 0
      for (let w = 0; w < W; w++) acc ^= (packed[w] ?? 0) & (_G[base + w] ?? 0)
      x[i] = parity(acc)
    }
    return x
  }

  /** @see densify */
  toStatevector(): Complex[] { return densify(this.n, b => this.amplitude(b)) }
}

/**
 * @internal
 * Dense statevector from an amplitude function, indexed with qubit 0 as the
 * least significant bit — the same convention as `Circuit.statevector()`.
 * Exponential in n; for tests and small-scale inspection only.
 */
export function densify(n: number, amplitude: (bits: Uint8Array) => Complex): Complex[] {
  if (n > 24) throw new RangeError(`refusing to densify ${n} qubits`)
  const out: Complex[] = new Array(1 << n)
  const bits = new Uint8Array(n)
  for (let i = 0; i < 1 << n; i++) {
    for (let j = 0; j < n; j++) bits[j] = (i >>> j) & 1
    out[i] = amplitude(bits)
  }
  return out
}

/**
 * Draw A uniformly from M_n — symmetric, off-diagonal entries in {0,1} and
 * diagonal entries in Z₄ — which specifies the equatorial stabilizer state
 * φ_A = 2^{-n/2} Σ_x i^{xAxᵀ}|x⟩.
 *
 * Equatorial states have equal weight on every basis vector, so sampling one is
 * just O(n²) coin flips. Lemma 2 uses them as the probe states for norm
 * estimation, where the uniform distribution over M_n is what makes
 * 2ⁿ|⟨φ_A|ψ⟩|² an unbiased estimator of ‖ψ‖².
 */
export function randomEquatorial(n: number, rand: () => number): Uint8Array {
  const A = new Uint8Array(n * n)
  for (let i = 0; i < n; i++) {
    A[i * n + i] = Math.floor(rand() * 4) & 3
    for (let j = i + 1; j < n; j++) {
      const bit = rand() < 0.5 ? 0 : 1
      A[i * n + j] = bit
      A[j * n + i] = bit
    }
  }
  return A
}
