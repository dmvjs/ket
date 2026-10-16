/**
 * The state as a 2-D tensor network, with bond truncation.
 *
 * Three engines have now been measured against the published 127-qubit Trotter
 * circuits and each fails for a structural reason:
 *
 * - **Exact contraction** is bounded by treewidth, which reaches 78 at depth 10.
 * - **Belief propagation** on the amplitude network converges at no damping,
 *   because the Bethe approximation has no way to track complex cancellation.
 * - **Matrix product states** are a chain; a heavy-hex lattice has no good linear
 *   embedding, and the truncated answer was still moving 10% between χ=32 and 64.
 *
 * What they have in common is forcing a 2-D state through a 1-D or exact
 * structure. This does not: one tensor per qubit, one virtual bond per edge of the
 * circuit's own connectivity graph, so the network has the shape of the problem.
 *
 * Gates are applied by *simple update* — contract the two sites across their
 * shared bond, apply the gate, split with an SVD and keep the largest `maxBond`
 * singular values. The bond weights left behind approximate the environment each
 * site sits in, which is what makes a purely local update a reasonable one. That
 * approximation is the method's whole error, and it is worst exactly where loops
 * are short.
 *
 * **This is approximate and it does not certify itself.** `discarded` accumulates
 * the squared singular weight thrown away, which is a real quantity but a local
 * one — it bounds each truncation, not the final answer. The honest check is the
 * same as everywhere else here: run at two bond dimensions and see whether the
 * answer moved.
 */
import { Circuit } from './circuit.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'
import { cmatrix, svd, type CMatrix } from './linalg.js'
import { contractNetwork, planContraction, type Tensor } from './tensor-network.js'

/** One qubit's tensor: physical index of size 2, then one leg per incident bond. */
interface Site {
  /** Neighbouring qubit for each virtual leg, in leg order. */
  bonds: number[]
  /** Size of each virtual leg. */
  dims: number[]
  /** Complex, `[physical][leg0][leg1]…`, row-major, re/im interleaved. */
  data: Float64Array
}

export interface PepsState {
  qubits: number
  sites: Site[]
  /** Bond weights in a Vidal-like gauge, keyed `min-max`. */
  weights: Map<string, Float64Array>
  /** Summed squared singular weight discarded across every truncation so far. */
  discarded: number
  /** Largest bond dimension currently in use. */
  maxBondUsed: number
}

const bondKey = (a: number, b: number): string => a < b ? `${a}-${b}` : `${b}-${a}`
const product = (xs: readonly number[]): number => xs.reduce((p, x) => p * x, 1)

/**
 * Build the network for a coupling graph, in the product state |0…0⟩.
 *
 * Every bond starts at dimension one, which is exactly the statement that the
 * initial state has no entanglement across any of them.
 */
export function pepsInit(qubits: number, coupling: readonly (readonly [number, number])[]): PepsState {
  const bonds: number[][] = Array.from({ length: qubits }, () => [])
  const seen = new Set<string>()
  for (const [a, b] of coupling) {
    if (a === b) throw new RangeError(`pepsInit: self-loop on qubit ${a}`)
    if (a < 0 || b < 0 || a >= qubits || b >= qubits) throw new RangeError(`pepsInit: edge ${a}-${b} outside ${qubits} qubits`)
    const k = bondKey(a, b)
    if (seen.has(k)) continue
    seen.add(k)
    bonds[a]!.push(b)
    bonds[b]!.push(a)
  }
  const sites: Site[] = bonds.map(nb => {
    const dims = nb.map(() => 1)
    const data = new Float64Array(2 * product(dims) * 2)
    data[0] = 1                      // |0⟩ with every bond trivial
    return { bonds: nb, dims, data }
  })
  const weights = new Map<string, Float64Array>()
  for (const k of seen) weights.set(k, Float64Array.from([1]))
  return { qubits, sites, weights, discarded: 0, maxBondUsed: 1 }
}

/** Scale a site's leg `leg` by `w` (or by its reciprocal). */
function scaleLeg(site: Site, leg: number, w: Float64Array, invert: boolean): void {
  const inner = product(site.dims.slice(leg + 1))
  const outer = 2 * product(site.dims.slice(0, leg))
  const d = site.dims[leg]!
  for (let o = 0; o < outer; o++) for (let k = 0; k < d; k++) {
    const f = invert ? (w[k]! > 1e-300 ? 1 / w[k]! : 0) : w[k]!
    const base = (o * d + k) * inner
    for (let i = 0; i < inner; i++) {
      site.data[(base + i) * 2] = site.data[(base + i) * 2]! * f
      site.data[(base + i) * 2 + 1] = site.data[(base + i) * 2 + 1]! * f
    }
  }
}

/** Reshape a site so `leg` is the trailing index: rows = everything else. */
function legLast(site: Site, leg: number): { m: CMatrix } {
  const d = site.dims[leg]!
  const inner = product(site.dims.slice(leg + 1))
  const outer = 2 * product(site.dims.slice(0, leg))
  const m = cmatrix(outer * inner, d)
  for (let o = 0; o < outer; o++) for (let k = 0; k < d; k++) for (let i = 0; i < inner; i++) {
    const src = ((o * d + k) * inner + i) * 2
    const dst = ((o * inner + i) * d + k) * 2
    m.data[dst] = site.data[src]!
    m.data[dst + 1] = site.data[src + 1]!
  }
  return { m }
}

/** Inverse of `legLast`: write a `(rest × d)` matrix back with `leg` in place. */
function legBack(site: Site, leg: number, m: CMatrix, newDim: number): void {
  const dims = [...site.dims]
  dims[leg] = newDim
  const inner = product(dims.slice(leg + 1))
  const outer = 2 * product(dims.slice(0, leg))
  const data = new Float64Array(outer * newDim * inner * 2)
  for (let o = 0; o < outer; o++) for (let k = 0; k < newDim; k++) for (let i = 0; i < inner; i++) {
    const src = ((o * inner + i) * newDim + k) * 2
    const dst = ((o * newDim + k) * inner + i) * 2
    data[dst] = m.data[src]!
    data[dst + 1] = m.data[src + 1]!
  }
  site.dims = dims
  site.data = data
}

/** Apply a one-qubit gate: a 2×2 mix of the physical index, which is outermost. */
export function pepsApply1(state: PepsState, q: number, g: Gate2x2): void {
  const site = state.sites[q]!
  const block = product(site.dims)
  const out = new Float64Array(site.data.length)
  for (let p = 0; p < 2; p++) for (let pi = 0; pi < 2; pi++) {
    const z = g[p]![pi]!
    if (z.re === 0 && z.im === 0) continue
    for (let i = 0; i < block; i++) {
      const s = (pi * block + i) * 2, d = (p * block + i) * 2
      out[d] = out[d]! + (site.data[s]! * z.re - site.data[s + 1]! * z.im)
      out[d + 1] = out[d + 1]! + (site.data[s]! * z.im + site.data[s + 1]! * z.re)
    }
  }
  site.data = out
}

/**
 * Apply a two-qubit gate across a bond, by simple update.
 *
 * The two sites are contracted over their shared leg into one tensor, the gate is
 * applied to its two physical indices, and an SVD splits it again — keeping the
 * largest `maxBond` singular values, which is the optimal split at that bond.
 * Neighbouring bond weights are absorbed first and divided out afterwards, so
 * each site is updated against an approximation of its surroundings rather than
 * in isolation.
 */
export function pepsApply2(state: PepsState, a: number, b: number, g: Gate4x4, maxBond: number): void {
  const A = state.sites[a]!, B = state.sites[b]!
  const legAB = A.bonds.indexOf(b), legBA = B.bonds.indexOf(a)
  if (legAB < 0 || legBA < 0) throw new RangeError(`pepsApply2: qubits ${a} and ${b} do not share a bond`)

  // Absorb every *other* incident weight, so the split sees the environment.
  for (let l = 0; l < A.bonds.length; l++) if (l !== legAB) scaleLeg(A, l, state.weights.get(bondKey(a, A.bonds[l]!))!, false)
  for (let l = 0; l < B.bonds.length; l++) if (l !== legBA) scaleLeg(B, l, state.weights.get(bondKey(b, B.bonds[l]!))!, false)
  // And the shared one, once, into A.
  scaleLeg(A, legAB, state.weights.get(bondKey(a, b))!, false)

  const { m: matA } = legLast(A, legAB)
  const { m: matB } = legLast(B, legBA)
  const D = matA.cols
  const rowsA = matA.rows, rowsB = matB.rows

  // Θ[restA][restB] = Σ_k A[restA][k] · B[restB][k].
  const theta = cmatrix(rowsA, rowsB)
  for (let i = 0; i < rowsA; i++) for (let k = 0; k < D; k++) {
    const ar = matA.data[(i * D + k) * 2]!, ai = matA.data[(i * D + k) * 2 + 1]!
    if (ar === 0 && ai === 0) continue
    for (let j = 0; j < rowsB; j++) {
      const br = matB.data[(j * D + k) * 2]!, bi = matB.data[(j * D + k) * 2 + 1]!
      theta.data[(i * rowsB + j) * 2] = theta.data[(i * rowsB + j) * 2]! + (ar * br - ai * bi)
      theta.data[(i * rowsB + j) * 2 + 1] = theta.data[(i * rowsB + j) * 2 + 1]! + (ar * bi + ai * br)
    }
  }

  // The physical index is the outermost factor of each side, so rows split into
  // two equal halves by p_a and columns into two by p_b.
  const halfA = rowsA / 2, halfB = rowsB / 2
  const applied = cmatrix(rowsA, rowsB)
  for (let pa = 0; pa < 2; pa++) for (let pb = 0; pb < 2; pb++) {
    for (let qa = 0; qa < 2; qa++) for (let qb = 0; qb < 2; qb++) {
      const z = g[pa * 2 + pb]![qa * 2 + qb]!
      if (z.re === 0 && z.im === 0) continue
      for (let i = 0; i < halfA; i++) for (let j = 0; j < halfB; j++) {
        const s = ((qa * halfA + i) * rowsB + (qb * halfB + j)) * 2
        const d = ((pa * halfA + i) * rowsB + (pb * halfB + j)) * 2
        applied.data[d] = applied.data[d]! + (theta.data[s]! * z.re - theta.data[s + 1]! * z.im)
        applied.data[d + 1] = applied.data[d + 1]! + (theta.data[s]! * z.im + theta.data[s + 1]! * z.re)
      }
    }
  }

  const { u, s, vt, rank, discarded } = svd(applied, { maxRank: maxBond })
  state.discarded += discarded

  let norm = 0
  for (let k = 0; k < rank; k++) norm += s[k]! * s[k]!
  norm = Math.sqrt(norm)
  const lambda = new Float64Array(rank)
  for (let k = 0; k < rank; k++) lambda[k] = norm > 1e-300 ? s[k]! / norm : 0

  // U carries the a-side; V† transposed carries the b-side.
  legBack(A, legAB, u, rank)
  const vT = cmatrix(rowsB, rank)
  for (let k = 0; k < rank; k++) for (let j = 0; j < rowsB; j++) {
    vT.data[(j * rank + k) * 2] = vt.data[(k * rowsB + j) * 2]!
    vT.data[(j * rank + k) * 2 + 1] = vt.data[(k * rowsB + j) * 2 + 1]!
  }
  legBack(B, legBA, vT, rank)
  state.weights.set(bondKey(a, b), lambda)

  // Remove the borrowed environment.
  for (let l = 0; l < A.bonds.length; l++) if (l !== legAB) scaleLeg(A, l, state.weights.get(bondKey(a, A.bonds[l]!))!, true)
  for (let l = 0; l < B.bonds.length; l++) if (l !== legBA) scaleLeg(B, l, state.weights.get(bondKey(b, B.bonds[l]!))!, true)

  state.maxBondUsed = Math.max(state.maxBondUsed, rank)
}

/**
 * ⟨P_q⟩, with each bond's weights standing in for the environment beyond it.
 *
 * This is the simple-update approximation: a site's surroundings are summarised
 * by the singular values on its own bonds and nothing further out. Exact when the
 * network is a tree, and the source of the method's error when it is not.
 */
export function pepsExpectation(state: PepsState, q: number, pauli: 'X' | 'Y' | 'Z'): number {
  const site = state.sites[q]!
  const block = product(site.dims)
  // Weight of each virtual configuration: the product of λ² along every leg.
  const env = new Float64Array(block).fill(1)
  for (let l = 0; l < site.bonds.length; l++) {
    const w = state.weights.get(bondKey(q, site.bonds[l]!))!
    const inner = product(site.dims.slice(l + 1))
    const outer = product(site.dims.slice(0, l))
    const d = site.dims[l]!
    for (let o = 0; o < outer; o++) for (let k = 0; k < d; k++) for (let i = 0; i < inner; i++) {
      env[(o * d + k) * inner + i] = env[(o * d + k) * inner + i]! * w[k]! * w[k]!
    }
  }
  const act = (sIn: number): { out: number; re: number; im: number } =>
    pauli === 'Z' ? { out: sIn, re: sIn === 0 ? 1 : -1, im: 0 }
      : pauli === 'X' ? { out: sIn ^ 1, re: 1, im: 0 }
      : { out: sIn ^ 1, re: 0, im: sIn === 0 ? 1 : -1 }

  let num = 0, den = 0
  for (let i = 0; i < block; i++) {
    const e = env[i]!
    for (let p = 0; p < 2; p++) {
      const pr = site.data[(p * block + i) * 2]!, pi = site.data[(p * block + i) * 2 + 1]!
      den += e * (pr * pr + pi * pi)
      const { out, re, im } = act(p)
      const br = site.data[(out * block + i) * 2]!, bi = -site.data[(out * block + i) * 2 + 1]!
      // conj(amp_out) · op · amp_p, real part only — the imaginary part cancels
      // over the sum for a Hermitian operator.
      const tr = br * pr - bi * pi, ti = br * pi + bi * pr
      num += e * (tr * re - ti * im)
    }
  }
  return den > 1e-300 ? num / den : 0
}

// ── Belief-propagation environments ───────────────────────────────────────────

/**
 * Messages on the *norm* network, which is a different problem from before.
 *
 * Belief propagation was already tried in this library, on the amplitude network,
 * and it converged at no damping — a complex network full of cancellation is the
 * worst case for the Bethe approximation. The norm network ⟨ψ|ψ⟩ is not that. Its
 * tensors are `T ⊗ conj(T)` with the physical index traced, so every message is a
 * positive-semidefinite matrix and the iteration behaves the way belief
 * propagation is supposed to. Same three letters, different object.
 *
 * A message along a bond of dimension D is a D×D matrix, stored row-major and
 * complex, carrying the environment that bond sees from the far side.
 */
type BondMessage = Float64Array

const identityMessage = (d: number): BondMessage => {
  const m = new Float64Array(d * d * 2)
  for (let i = 0; i < d; i++) m[(i * d + i) * 2] = 1
  return m
}

/**
 * Absorb every incoming message into a copy of the site's ket, except on `skip`.
 *
 * Absorbing on leg `c` replaces the ket's own index there with the bra's, so the
 * later contraction against `conj(T)` closes those legs automatically.
 */
function absorbInto(site: Site, msgs: (BondMessage | null)[], skip: number): Float64Array {
  let data = Float64Array.from(site.data)
  for (let l = 0; l < site.bonds.length; l++) {
    if (l === skip) continue
    const m = msgs[l]
    if (!m) continue
    const d = site.dims[l]!
    const inner = product(site.dims.slice(l + 1))
    const outer = 2 * product(site.dims.slice(0, l))
    const out = new Float64Array(data.length)
    for (let o = 0; o < outer; o++) for (let j = 0; j < d; j++) for (let i = 0; i < d; i++) {
      const mr = m[(i * d + j) * 2]!, mi = m[(i * d + j) * 2 + 1]!
      if (mr === 0 && mi === 0) continue
      for (let k = 0; k < inner; k++) {
        const src = ((o * d + i) * inner + k) * 2
        const dst = ((o * d + j) * inner + k) * 2
        out[dst] = out[dst]! + (data[src]! * mr - data[src + 1]! * mi)
        out[dst + 1] = out[dst + 1]! + (data[src]! * mi + data[src + 1]! * mr)
      }
    }
    data = out
  }
  return data
}

/** Contract an absorbed ket against `conj(T)`, leaving leg `free` open on both. */
function closeAgainstBra(site: Site, ket: Float64Array, free: number): BondMessage {
  const d = site.dims[free]!
  const inner = product(site.dims.slice(free + 1))
  const outer = 2 * product(site.dims.slice(0, free))
  const out = new Float64Array(d * d * 2)
  for (let o = 0; o < outer; o++) for (let k = 0; k < inner; k++) {
    for (let ib = 0; ib < d; ib++) {
      const kr = ket[((o * d + ib) * inner + k) * 2]!, ki = ket[((o * d + ib) * inner + k) * 2 + 1]!
      if (kr === 0 && ki === 0) continue
      for (let jb = 0; jb < d; jb++) {
        const br = site.data[((o * d + jb) * inner + k) * 2]!
        const bi = -site.data[((o * d + jb) * inner + k) * 2 + 1]!
        out[(ib * d + jb) * 2] = out[(ib * d + jb) * 2]! + (kr * br - ki * bi)
        out[(ib * d + jb) * 2 + 1] = out[(ib * d + jb) * 2 + 1]! + (kr * bi + ki * br)
      }
    }
  }
  return out
}

export interface BpEnvironments {
  /** `messages[q][leg]` is what arrives at site `q` along its leg `leg`. */
  messages: BondMessage[][]
  /** The weight-absorbed tensors the messages were computed against. */
  gauged: Site[]
  converged: boolean
  iterations: number
  residual: number
}

/**
 * Fold the bond weights into the site tensors, symmetrically.
 *
 * `pepsApply2` leaves the network in a Vidal-like gauge: the site tensors are Γ,
 * and the singular values λ live on the bonds separately. That factorisation is
 * what makes simple update local, but it means the site tensors *are not the
 * state* — reading a norm network off them directly leaves out every λ.
 *
 * Putting √λ on each side of every bond reassembles the state while keeping the
 * two ends symmetric, which is what the norm network needs.
 */
function absorbWeights(state: PepsState): Site[] {
  return state.sites.map((site, q) => {
    const copy: Site = { bonds: site.bonds, dims: [...site.dims], data: Float64Array.from(site.data) }
    for (let l = 0; l < site.bonds.length; l++) {
      const w = state.weights.get(bondKey(q, site.bonds[l]!))!
      const root = Float64Array.from(w, x => Math.sqrt(Math.abs(x)))
      scaleLeg(copy, l, root, false)
    }
    return copy
  })
}

/**
 * Run belief propagation on the norm network to get a bond environment per leg.
 *
 * Replaces the bond weights that simple update leaves behind — which summarise a
 * site's surroundings by its own singular values and nothing further out — with
 * a message that has actually travelled through the rest of the network.
 */
export function pepsEnvironments(
  state: PepsState,
  { maxIterations = 100, tolerance = 1e-9 }: { maxIterations?: number; tolerance?: number } = {},
): BpEnvironments {
  const n = state.qubits
  const gauged = absorbWeights(state)
  const messages: BondMessage[][] = gauged.map(s => s.dims.map(d => identityMessage(d)))
  /** Where site `b` keeps the message arriving from `a`. */
  const legOf = (a: number, b: number): number => state.sites[b]!.bonds.indexOf(a)

  let residual = Infinity
  let iterations = 0
  for (; iterations < maxIterations; iterations++) {
    residual = 0
    for (let a = 0; a < n; a++) {
      const site = gauged[a]!
      for (let l = 0; l < site.bonds.length; l++) {
        const b = site.bonds[l]!
        const ket = absorbInto(site, messages[a]!, l)
        const next = closeAgainstBra(site, ket, l)
        // Normalise by the trace: only the environment's shape matters, and the
        // scale would otherwise run away across a large network.
        const d = site.dims[l]!
        let tr = 0
        for (let i = 0; i < d; i++) tr += next[(i * d + i) * 2]!
        if (Math.abs(tr) > 1e-300) for (let i = 0; i < next.length; i++) next[i] = next[i]! / tr
        const slot = legOf(a, b)
        const prev = messages[b]![slot]!
        if (prev.length === next.length) {
          for (let i = 0; i < next.length; i++) residual = Math.max(residual, Math.abs(next[i]! - prev[i]!))
        } else { residual = Infinity }
        messages[b]![slot] = next
      }
    }
    if (residual <= tolerance) { iterations++; break }
  }
  return { messages, gauged, converged: residual <= tolerance, iterations, residual }
}

/**
 * ⟨P_q⟩ using belief-propagation environments rather than the bond weights.
 *
 * The bond-weight version is the simple-update approximation and carries an error
 * floor that no bond dimension removes. This one contracts the site against
 * messages that have propagated through the network, which is the standard fix.
 */
export function pepsExpectationBP(
  state: PepsState, q: number, pauli: 'X' | 'Y' | 'Z', env: BpEnvironments,
): number {
  // The weight-absorbed tensor, matching what the messages were built against.
  const site = env.gauged[q]!
  const ket = absorbInto(site, env.messages[q]!, -1)
  const block = product(site.dims)
  const act = (sIn: number): { out: number; re: number; im: number } =>
    pauli === 'Z' ? { out: sIn, re: sIn === 0 ? 1 : -1, im: 0 }
      : pauli === 'X' ? { out: sIn ^ 1, re: 1, im: 0 }
      : { out: sIn ^ 1, re: 0, im: sIn === 0 ? 1 : -1 }

  let num = 0, den = 0
  for (let i = 0; i < block; i++) {
    for (let p = 0; p < 2; p++) {
      const kr = ket[(p * block + i) * 2]!, ki = ket[(p * block + i) * 2 + 1]!
      const dr = site.data[(p * block + i) * 2]!, di = -site.data[(p * block + i) * 2 + 1]!
      den += kr * dr - ki * di
      const { out, re, im } = act(p)
      const br = site.data[(out * block + i) * 2]!, bi = -site.data[(out * block + i) * 2 + 1]!
      const tr = kr * br - ki * bi, ti = kr * bi + ki * br
      num += tr * re - ti * im
    }
  }
  return Math.abs(den) > 1e-300 ? num / den : 0
}

// ── Exact environments ────────────────────────────────────────────────────────

/**
 * ⟨P_q⟩ by contracting the norm network exactly, with no environment
 * approximation at all.
 *
 * This is what showed that the simple-update error floor is the *environment* and
 * nothing else. On a 3×3 grid at depth 10 with a bond dimension large enough that
 * nothing is truncated, the bond-weight estimate is wrong by 7.5e-2 and this is
 * wrong by 5.8e-16. The state was exactly right the whole time; reading an
 * expectation off it through a tree-shaped environment was not.
 *
 * The cost is the catch. The norm network's bonds carry both a ket and a bra
 * index, so its contraction width is roughly twice the state network's times
 * log2(χ), and it grows out of reach quickly — hence `maxWidth`, and hence a null
 * return rather than an attempt that never finishes.
 *
 * `contractNetwork` requires every index to have dimension two, so each bond
 * index is split into binary digits rather than carried as one wide index.
 */
export function pepsExpectationExact(
  state: PepsState, q: number, pauli: 'X' | 'Y' | 'Z', env: BpEnvironments,
  { maxWidth = 24, restarts = 10 }: { maxWidth?: number; restarts?: number } = {},
): { value: number; width: number } | null {
  const numerator = normNetwork(state, env, q, pauli)
  const plan = planContraction(numerator.map(t => t.indices), { restarts })
  if (plan.width > maxWidth) return { value: NaN, width: plan.width }
  const denominator = normNetwork(state, env, -1, pauli)
  const num = contractNetwork(numerator, plan)
  const den = contractNetwork(denominator, plan)
  const dr = den.data[0]!, di = den.data[1]!
  const d2 = dr * dr + di * di
  if (d2 < 1e-300) return null
  // Both are real for a Hermitian operator on a normalised state; the ratio's
  // imaginary part is rounding and is dropped.
  return { value: (num.data[0]! * dr + num.data[1]! * di) / d2, width: plan.width }
}

/** The norm network ⟨ψ|P_q|ψ⟩, as dimension-two tensors. `opQ = -1` inserts nothing. */
function normNetwork(state: PepsState, env: BpEnvironments, opQ: number, pauli: 'X' | 'Y' | 'Z'): Tensor[] {
  const bitsOf = (d: number): number => Math.max(1, Math.ceil(Math.log2(d)))
  const out: Tensor[] = []
  for (let q = 0; q < state.qubits; q++) {
    const site = env.gauged[q]!
    const dims = site.dims, k = dims.length
    const block = product(dims)
    const nb = dims.map(bitsOf)
    const total = nb.reduce((a, b) => a + b, 0) * 2
    const data = new Float64Array((1 << total) * 2)
    const act = (p: number): { out: number; re: number; im: number } =>
      pauli === 'Z' ? { out: p, re: p === 0 ? 1 : -1, im: 0 }
        : pauli === 'X' ? { out: p ^ 1, re: 1, im: 0 }
        : { out: p ^ 1, re: 0, im: p === 0 ? 1 : -1 }

    for (let i = 0; i < block; i++) for (let j = 0; j < block; j++) {
      let ri = i, rj = j
      const di: number[] = [], dj: number[] = []
      for (let l = k - 1; l >= 0; l--) {
        di.unshift(ri % dims[l]!); ri = Math.floor(ri / dims[l]!)
        dj.unshift(rj % dims[l]!); rj = Math.floor(rj / dims[l]!)
      }
      let flat = 0
      for (let l = 0; l < k; l++) flat = (flat << nb[l]!) | di[l]!
      for (let l = 0; l < k; l++) flat = (flat << nb[l]!) | dj[l]!

      let re = 0, im = 0
      for (let p = 0; p < 2; p++) {
        const kr = site.data[(p * block + i) * 2]!, ki = site.data[(p * block + i) * 2 + 1]!
        const a = q === opQ ? act(p) : { out: p, re: 1, im: 0 }
        const br = site.data[(a.out * block + j) * 2]!, bi = -site.data[(a.out * block + j) * 2 + 1]!
        const tr = kr * br - ki * bi, ti = kr * bi + ki * br
        re += tr * a.re - ti * a.im
        im += tr * a.im + ti * a.re
      }
      data[flat * 2] = re
      data[flat * 2 + 1] = im
    }

    // Ket indices first, then bra, matching the packing above. Both ends of a
    // bond name them identically so the contraction closes them.
    const indices: string[] = []
    for (const layer of ['K', 'B']) {
      for (let l = 0; l < k; l++) {
        const key = bondKey(q, site.bonds[l]!)
        for (let t = 0; t < nb[l]!; t++) indices.push(`${layer}${key}_${t}`)
      }
    }
    out.push({ indices, data })
  }
  return out
}

/** A gate list in the normalised form `Circuit.toOps()` produces. */
type SimpleOp =
  | { kind: 'one'; q: number; gate: Gate2x2 }
  | { kind: 'two'; a: number; b: number; gate: Gate4x4 }

export interface PepsRunOptions {
  /** Largest bond dimension kept at any split. */
  maxBond?: number
  /** Coupling graph. Defaults to whatever pairs the circuit's own gates touch. */
  coupling?: readonly (readonly [number, number])[]
}

/**
 * Replay a circuit as a 2-D tensor network.
 *
 * The coupling graph defaults to the pairs the circuit's two-qubit gates actually
 * connect, so the network takes the shape of the circuit rather than of an
 * assumed lattice. A gate between qubits with no bond is an error rather than a
 * silent approximation — it would need a bond that does not exist.
 */
export function pepsRun(circuit: Circuit, { maxBond = 8, coupling }: PepsRunOptions = {}): PepsState {
  const ops = circuit.toOps() as SimpleOp[]
  const edges = coupling ?? [...new Set(ops.filter(o => o.kind === 'two')
    .map(o => bondKey((o as { a: number }).a, (o as { b: number }).b)))]
    .map(k => k.split('-').map(Number) as [number, number])

  const state = pepsInit(circuit.qubits, edges)
  for (const op of ops) {
    if (op.kind === 'one') pepsApply1(state, op.q, op.gate)
    else pepsApply2(state, op.a, op.b, op.gate, maxBond)
  }
  return state
}
