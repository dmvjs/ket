/**
 * Clifford+T simulation by low-rank stabilizer decomposition (sum over Cliffords).
 *
 * A state is carried as a linear combination of stabilizer states in CH-form,
 *
 *   |ψ⟩ = Σ_α c_α |φ_α⟩,
 *
 * Clifford gates act on every term and leave the term count alone. A non-Clifford
 * diagonal gate splits each term in two, so cost is exponential in the count of
 * non-Clifford gates and only *polynomial in qubit count* — the opposite trade to
 * a statevector, and the reason this reaches circuit widths a statevector cannot.
 *
 * Any diag(1, e^{iθ}) is written over the two diagonal Cliffords bracketing e^{iθ}
 * on the unit circle, {I, S, Z, S†} = diag(1, i^k). With a + b = 1 and
 * a·i^k + b·i^{k+1} = e^{iθ} this is the ℓ₁-optimal two-term choice. For T (θ=π/4)
 * it gives ‖c‖₁ = 1.0824, i.e. a stabilizer extent of 1/cos²(π/8) ≈ 2^0.228 per T
 * gate — the scaling quoted for sum-over-Cliffords in the reference below.
 *
 * Exact by default: `maxTerms` is a ceiling, not an allocation, and reaching it
 * turns on sparsification (the Sparsification Lemma), which is unbiased but
 * randomised. `sparsified` reports whether that happened, so an exact run is
 * always distinguishable from an approximate one.
 *
 * Like `StabilizerCH` and unlike `Circuit`, gate methods mutate and return
 * `this`. Copying a decomposition means copying every term, so the immutable
 * style the rest of the library uses would dominate the cost here.
 *
 * Reference: Bravyi, Browne, Calpin, Campbell, Gosset, Howard, "Simulation of
 * quantum circuits by low-rank stabilizer decompositions", Quantum 3, 181 (2019).
 */
import { StabilizerCH, densify, randomEquatorial } from './stabilizer-ch.js'
import { c, type Complex } from './complex.js'
import { makePrng } from './prng.js'
import { sampleFromOracle, type AmplitudeOracle, type WalkOracle, type SampleOptions } from './stabilizer-sampling.js'

const HALF_PI = Math.PI / 2
const TAU = 2 * Math.PI
/** Coefficients below this are dropped as exactly zero. */
const EPS = 1e-12

/**
 * @internal
 * Serializable gate for slice replay and worker transport.
 *
 * `StabilizerCH` objects cannot cross a worker boundary, but the *recipe* for a
 * term can: the decomposition tree is deterministic, so term α is reproducible
 * from the op list plus the bits of α alone. That is what makes the term
 * dimension embarrassingly parallel — no state is shared or communicated.
 */
export type SrOp =
  | { g: 'h' | 's' | 'sdg' | 'x' | 'y' | 'z'; q: number }
  | { g: 'cx' | 'cz' | 'swap'; a: number; b: number }
  | { g: 'phase' | 'rz'; q: number; theta: number }

/** @internal One term of a slice: a coefficient and the state it multiplies. */
export interface SliceTerm { re: number; im: number; state: StabilizerCH }

/** Two-term Clifford decomposition of diag(1, e^{iθ}). */
interface Split {
  /** Diagonal Clifford index for the first branch, or the only one if `!splits`. */
  k: number
  splits: boolean
  aRe: number; aIm: number; bRe: number; bIm: number
}

/**
 * Write diag(1, e^{iθ}) over the two diagonal Cliffords bracketing e^{iθ}.
 * Shared by the sequential and sliced builders so the two can never drift.
 */
function decompose(theta: number): Split {
  let th = theta % TAU
  if (th < 0) th += TAU
  let k = Math.floor(th / HALF_PI)
  let phi = th - k * HALF_PI
  if (phi > HALF_PI - 1e-12) { k += 1; phi = 0 }   // snap to the Clifford angle
  k &= 3

  // a + b = 1 and a·i^k + b·i^{k+1} = e^{iθ}  ⟹  b = (e^{iφ} − 1)/(i − 1)
  const cosP = Math.cos(phi), sinP = Math.sin(phi)
  const bRe = (1 - cosP + sinP) / 2, bIm = (1 - cosP - sinP) / 2
  const aRe = 1 - bRe, aIm = -bIm

  if (Math.hypot(bRe, bIm) < EPS) return { k, splits: false, aRe, aIm, bRe, bIm }
  if (Math.hypot(aRe, aIm) < EPS) return { k: (k + 1) & 3, splits: false, aRe, aIm, bRe, bIm }
  return { k, splits: true, aRe, aIm, bRe, bIm }
}

/** Apply diag(1, i^k) — the diagonal Clifford for k mod 4. */
function applyDiag(st: StabilizerCH, k: number, q: number): StabilizerCH {
  switch (k & 3) {
    case 1: return st.s(q)
    case 2: return st.z(q)
    case 3: return st.sdg(q)
    default: return st
  }
}

/**
 * @internal
 * Number of non-Clifford splits in an op list; the decomposition has 2^t terms.
 * Gates at Clifford angles cost nothing and are not counted.
 */
export function countSplits(ops: readonly SrOp[]): number {
  let t = 0
  for (const op of ops) if ((op.g === 'phase' || op.g === 'rz') && decompose(op.theta).splits) t++
  return t
}

/**
 * Stabilizer extent ξ = ‖c‖₁² of the decomposition `ops` produces.
 *
 * Multiplicative over the splitting gates, so it is known before any term is
 * built. Each T gate contributes 1/cos²(π/8) ≈ 2^0.228, which is where the
 * quoted sum-over-Cliffords exponent comes from.
 */
export function extent(ops: readonly SrOp[]): number {
  let l1 = 1
  for (const op of ops) {
    if (op.g !== 'phase' && op.g !== 'rz') continue
    const d = decompose(op.theta)
    if (!d.splits) continue
    l1 *= Math.sqrt(d.aRe ** 2 + d.aIm ** 2) + Math.sqrt(d.bRe ** 2 + d.bIm ** 2)
  }
  return l1 ** 2
}

/**
 * Terms needed to approximate the output of `ops` to ℓ₂ error `delta`:
 * k = ⌈ξ/δ²⌉, from the Sparsification Lemma bound χ_δ ≤ 1 + ξ/δ².
 *
 * This is what makes large T-counts reachable at all. Exact simulation costs 2^t
 * terms; this costs ξ/δ² = 2^{0.228t}/δ², so t=50 at δ=0.2 needs roughly 69k
 * terms — about what an *exact* t=16 run already costs.
 */
export function termBudget(ops: readonly SrOp[], delta: number): number {
  if (!(delta > 0)) throw new RangeError(`targetError must be positive, got ${delta}`)
  return Math.max(1, Math.ceil(extent(ops) / delta ** 2))
}

/**
 * @internal
 * Build terms `[lo, hi)` of the 2^t decomposition by replaying `ops` with the
 * branch at split i fixed to bit i of the term index.
 *
 * Independent of every other slice, so slices can be built concurrently and
 * their partial amplitude sums added. Term ordering matches the sequential
 * builder exactly, which is what the slice-union tests pin.
 */
export function buildSlice(n: number, ops: readonly SrOp[], lo: number, hi: number): SliceTerm[] {
  const out: SliceTerm[] = []
  for (let alpha = lo; alpha < hi; alpha++) {
    const st = new StabilizerCH(n)
    let re = 1, im = 0
    let split = 0
    for (const op of ops) {
      switch (op.g) {
        case 'h': case 's': case 'sdg': case 'x': case 'y': case 'z': st[op.g](op.q); break
        case 'cx': case 'cz': case 'swap': st[op.g](op.a, op.b); break
        default: {
          const d = decompose(op.theta)
          if (!d.splits) applyDiag(st, d.k, op.q)
          else {
            const bit = (alpha >>> split) & 1
            split++
            const cr = bit ? d.bRe : d.aRe, ci = bit ? d.bIm : d.aIm
            const r = re
            re = r * cr - im * ci
            im = r * ci + im * cr
            applyDiag(st, d.k + bit, op.q)
          }
          if (op.g === 'rz') {
            const g = -op.theta / 2, gr = Math.cos(g), gi = Math.sin(g)
            const r = re
            re = r * gr - im * gi
            im = r * gi + im * gr
          }
        }
      }
    }
    out.push({ re, im, state: st })
  }
  return out
}

export interface StabilizerRankOptions {
  /**
   * Ceiling on the number of stabilizer terms. Exceeding it triggers
   * sparsification down to this many terms. Default `Infinity` (exact).
   */
  maxTerms?: number
  /** Seed for the sparsification RNG. */
  seed?: number
}

export interface NormEstimateOptions {
  /** Target relative error. Cost scales as ε⁻². Default 0.1. */
  epsilon?: number
  /** Failure probability. Cost scales as log δ⁻¹. Default 0.05. */
  delta?: number
  /** RNG for drawing equatorial probe states. Defaults to this state's own. */
  rand?: () => number
}

export class StabilizerRank {
  readonly n: number
  #re: number[]
  #im: number[]
  #st: StabilizerCH[]
  readonly #maxTerms: number
  #rand: () => number
  #sparsifications = 0

  constructor(n: number, { maxTerms = Infinity, seed = 0x5eed }: StabilizerRankOptions = {}) {
    if (maxTerms < 1) throw new RangeError(`maxTerms must be at least 1, got ${maxTerms}`)
    this.n = n
    this.#re = [1]
    this.#im = [0]
    this.#st = [new StabilizerCH(n)]
    this.#maxTerms = maxTerms
    this.#rand = makePrng(seed)
  }

  /** Number of stabilizer terms currently held. */
  get termCount(): number { return this.#st.length }

  /**
   * `true` if sparsification has run, i.e. results are approximate. Always
   * `false` for a run that stayed under `maxTerms`.
   */
  get sparsified(): boolean { return this.#sparsifications > 0 }

  /**
   * How many times sparsification fired.
   *
   * The Sparsification Lemma bounds the error of a *single* application. A
   * streaming run cannot hold 2^t terms to sparsify once at the end, so it
   * sparsifies repeatedly and the errors compound — the per-application bound no
   * longer certifies the total. Use this to see how far from the single-shot
   * regime a run drifted, and `estimateNorm` to measure what it actually cost.
   */
  get sparsifications(): number { return this.#sparsifications }

  /** ℓ₁ norm of the coefficient vector; its square bounds the sampling cost. */
  get l1(): number {
    let acc = 0
    for (let i = 0; i < this.#re.length; i++) acc += Math.sqrt(this.#re[i]! ** 2 + this.#im[i]! ** 2)
    return acc
  }

  // ── Clifford layer: applied to every term, term count unchanged ─────────────

  #each(f: (s: StabilizerCH) => void): this {
    for (const s of this.#st) f(s)
    return this
  }

  h(q: number): this { return this.#each(s => { s.h(q) }) }
  s(q: number): this { return this.#each(s => { s.s(q) }) }
  sdg(q: number): this { return this.#each(s => { s.sdg(q) }) }
  x(q: number): this { return this.#each(s => { s.x(q) }) }
  y(q: number): this { return this.#each(s => { s.y(q) }) }
  z(q: number): this { return this.#each(s => { s.z(q) }) }
  cx(a: number, b: number): this { return this.#each(s => { s.cx(a, b) }) }
  cz(a: number, b: number): this { return this.#each(s => { s.cz(a, b) }) }
  swap(a: number, b: number): this { return this.#each(s => { s.swap(a, b) }) }

  // ── Non-Clifford diagonal layer: doubles the term count ─────────────────────

  /**
   * @internal Replace the decomposition with prebuilt terms, reassembling slices
   * from {@link buildSlice}. Paired with internal machinery; not public API.
   */
  setTerms(terms: readonly SliceTerm[]): this {
    if (terms.length === 0) throw new RangeError('a decomposition needs at least one term')
    this.#re = terms.map(t => t.re)
    this.#im = terms.map(t => t.im)
    this.#st = terms.map(t => t.state)
    return this
  }

  /**
   * Phase gate diag(1, e^{iθ}) on qubit q.
   *
   * Splits every term over the two diagonal Cliffords bracketing e^{iθ}. Exact
   * multiples of π/2 are Clifford and cost nothing — no split is emitted.
   */
  phase(theta: number, q: number): this {
    const { k, splits, aRe, aIm, bRe, bIm } = decompose(theta)
    if (!splits) return this.#each(s => { applyDiag(s, k, q) })

    const m = this.#st.length
    const re = new Array<number>(2 * m), im = new Array<number>(2 * m)
    const st = new Array<StabilizerCH>(2 * m)
    for (let i = 0; i < m; i++) {
      const cr = this.#re[i]!, ci = this.#im[i]!, s0 = this.#st[i]!
      const s1 = s0.clone()
      re[i] = cr * aRe - ci * aIm
      im[i] = cr * aIm + ci * aRe
      st[i] = applyDiag(s0, k, q)
      re[m + i] = cr * bRe - ci * bIm
      im[m + i] = cr * bIm + ci * bRe
      st[m + i] = applyDiag(s1, k + 1, q)
    }
    this.#re = re; this.#im = im; this.#st = st
    if (this.#st.length > this.#maxTerms) this.sparsify(this.#maxTerms)
    return this
  }

  /** T = diag(1, e^{iπ/4}). */
  t(q: number): this { return this.phase(Math.PI / 4, q) }

  /** T† = diag(1, e^{-iπ/4}). */
  tdg(q: number): this { return this.phase(-Math.PI / 4, q) }

  /** Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}); equals phase(θ) up to a global phase. */
  rz(theta: number, q: number): this {
    this.phase(theta, q)
    const g = -theta / 2
    const gr = Math.cos(g), gi = Math.sin(g)
    for (let i = 0; i < this.#re.length; i++) {
      const r = this.#re[i]!, m = this.#im[i]!
      this.#re[i] = r * gr - m * gi
      this.#im[i] = r * gi + m * gr
    }
    return this
  }

  // ── Sparsification ──────────────────────────────────────────────────────────

  /**
   * Replace the decomposition with `k` terms drawn i.i.d. with probability
   * |c_α|/‖c‖₁, each re-weighted to (‖c‖₁/k)·(c_α/|c_α|).
   *
   * Unbiased — the expectation is the original state — with squared error of
   * order ξ/k, where ξ = ‖c‖₁² is the stabilizer extent. Sampling is with
   * replacement, so repeats are expected and correct.
   */
  sparsify(k: number, rand: () => number = this.#rand): this {
    const m = this.#st.length
    if (k >= m) return this
    const abs = new Float64Array(m)
    const cum = new Float64Array(m)
    let l1 = 0
    for (let i = 0; i < m; i++) {
      abs[i] = Math.sqrt(this.#re[i]! ** 2 + this.#im[i]! ** 2)
      l1 += abs[i]!
      cum[i] = l1
    }
    if (l1 < EPS) return this

    const re = new Array<number>(k), im = new Array<number>(k), st = new Array<StabilizerCH>(k)
    const w = l1 / k
    for (let j = 0; j < k; j++) {
      const target = rand() * l1
      let lo = 0, hi = m - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (cum[mid]! < target) lo = mid + 1; else hi = mid
      }
      const a = abs[lo]!
      re[j] = (w * this.#re[lo]!) / a
      im[j] = (w * this.#im[lo]!) / a
      st[j] = this.#st[lo]!.clone()
    }
    this.#re = re; this.#im = im; this.#st = st
    this.#sparsifications++
    return this
  }

  // ── Readout ─────────────────────────────────────────────────────────────────

  /** Exact amplitude ⟨x|ψ⟩ = Σ_α c_α⟨x|φ_α⟩. O(k·n²). */
  amplitude(x: string | ArrayLike<number>): Complex {
    let re = 0, im = 0
    for (let i = 0; i < this.#st.length; i++) {
      const z = this.#st[i]!.amplitude(x)
      const cr = this.#re[i]!, ci = this.#im[i]!
      re += cr * z.re - ci * z.im
      im += cr * z.im + ci * z.re
    }
    return c(re, im)
  }

  /** |⟨x|ψ⟩|². */
  probability(x: string | ArrayLike<number>): number {
    const z = this.amplitude(x)
    return z.re * z.re + z.im * z.im
  }

  /** @see densify */
  toStatevector(): Complex[] { return densify(this.n, b => this.amplitude(b)) }

  /**
   * Estimate ‖ψ‖² to relative error `epsilon` with confidence `1 - delta`
   * (Lemma 2), without touching the 2ⁿ amplitudes.
   *
   * η_A = 2ⁿ|⟨φ_A|ψ⟩|² over a uniformly random equatorial φ_A is unbiased for
   * ‖ψ‖² with variance at most ‖ψ‖⁴, so Chebyshev gives a (1±ε) estimate from
   * 4/ε² samples with probability 3/4, and a median over O(log 1/δ) such means
   * lifts that to 1-δ.
   *
   * Cost is O(k·n³·ε⁻²·log δ⁻¹) — polynomial in qubit count, linear in terms.
   * This is the only way to check a sparsified run: an exact decomposition of a
   * unitary circuit has ‖ψ‖² = 1 by construction, so drift away from 1 measures
   * the damage sparsification actually did rather than the bound it promised.
   */
  estimateNorm({ epsilon = 0.1, delta = 0.05, rand = this.#rand }: NormEstimateOptions = {}): number {
    if (!(epsilon > 0)) throw new RangeError(`epsilon must be positive, got ${epsilon}`)
    if (!(delta > 0 && delta < 1)) throw new RangeError(`delta must be in (0,1), got ${delta}`)
    const { n } = this
    const perMean = Math.ceil(4 / epsilon ** 2)
    const means = Math.max(1, Math.ceil(8 * Math.log(1 / delta)))
    // Fold 2^{n/2} into the amplitude so η stays O(1); 2ⁿ alone would overflow
    // for wide circuits even though the product is order unity.
    const half = 2 ** (n / 2)

    const estimates: number[] = []
    for (let m = 0; m < means; m++) {
      let acc = 0
      for (let i = 0; i < perMean; i++) {
        const A = randomEquatorial(n, rand)
        let re = 0, im = 0
        for (let a = 0; a < this.#st.length; a++) {
          // ⟨φ_A|ψ⟩ = Σ c_α·conj(⟨φ_α|φ_A⟩)
          const z = this.#st[a]!.innerProductEquatorial(A)
          const cr = this.#re[a]!, ci = this.#im[a]!
          re += cr * z.re + ci * z.im
          im += ci * z.re - cr * z.im
        }
        acc += (re * half) ** 2 + (im * half) ** 2
      }
      estimates.push(acc / perMean)
    }
    estimates.sort((p, q) => p - q)
    return estimates[estimates.length >> 1]!
  }

  /**
   * Draw `shots` basis states from P(x) ∝ |⟨x|ψ⟩|².
   *
   * Delegates to {@link sampleFromOracle} over this state's own amplitudes; the
   * worker-backed backend passes a distributed oracle to the same function.
   */
  sample(shots: number, rand: () => number, opts: SampleOptions = {}): Uint8Array[] {
    return sampleFromOracle(this.n, shots, rand, this.#oracle(), {
      ...opts,
      terms: this.#st.length,
      start: r => this.#st[Math.floor(r() * this.#st.length)]!.sample(r),
      walk: x => this.#walkOracle(x),
    })
  }

  /**
   * Incremental probability evaluator over all terms, seeded at `x` (Eq. 57).
   *
   * One walker per term; amplitudes stay linear so they sum, and the modulus is
   * taken only after. Snapshots the decomposition — applying gates afterwards
   * invalidates it.
   */
  #walkOracle(x: Uint8Array): WalkOracle {
    const walkers = this.#st.map(st => st.walker(x))
    const re = this.#re, im = this.#im
    return {
      flip(j: number): void { for (const w of walkers) w.flip(j) },
      probability(): number {
        let ar = 0, ai = 0
        for (let a = 0; a < walkers.length; a++) {
          const z = walkers[a]!.value()
          const cr = re[a]!, ci = im[a]!
          ar += cr * z.re - ci * z.im
          ai += cr * z.im + ci * z.re
        }
        return ar * ar + ai * ai
      },
    }
  }

  /** Batched amplitude evaluator over this state's terms. */
  #oracle(): AmplitudeOracle {
    const { n } = this
    return (basis, count) => {
      const re = new Float64Array(count), im = new Float64Array(count)
      const bits = new Uint8Array(n)
      for (let i = 0; i < count; i++) {
        bits.set(basis.subarray(i * n, (i + 1) * n))
        const z = this.amplitude(bits)
        re[i] = z.re
        im[i] = z.im
      }
      return { re, im }
    }
  }
}
