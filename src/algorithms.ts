/**
 * Standard quantum algorithm library.
 *
 * All functions return immutable Circuit instances (or numbers for VQE).
 * Qubit 0 is the least-significant bit throughout.
 */

import { Circuit } from './circuit.js'

// ── Quantum Fourier Transform ─────────────────────────────────────────────────

/**
 * n-qubit Quantum Fourier Transform.
 * Output convention: qubit 0 = LSB (IonQ / ket standard).
 */
export function qft(n: number): Circuit {
  let c = new Circuit(n)
  for (let j = n - 1; j >= 0; j--) {
    c = c.h(j)
    for (let k = j - 1; k >= 0; k--) {
      c = c.cu1(Math.PI / 2 ** (j - k), k, j)
    }
  }
  for (let i = 0; i < Math.floor(n / 2); i++) c = c.swap(i, n - 1 - i)
  return c
}

/**
 * n-qubit inverse QFT (QFT†).
 */
export function iqft(n: number): Circuit {
  let c = new Circuit(n)
  for (let i = 0; i < Math.floor(n / 2); i++) c = c.swap(i, n - 1 - i)
  for (let j = 0; j < n; j++) {
    for (let k = j - 1; k >= 0; k--) c = c.cu1(-Math.PI / 2 ** (j - k), k, j)
    c = c.h(j)
  }
  return c
}

// ── Grover's search ───────────────────────────────────────────────────────────

/**
 * Number of ancilla qubits required by `grover` for an n-qubit search space.
 * The returned circuit has `n + groverAncilla(n)` total qubits;
 * qubits 0..n-1 hold the search register.
 */
export function groverAncilla(n: number): number {
  return Math.max(0, n - 3)
}

/**
 * Multi-controlled X gate (Barenco staircase decomposition).
 * Requires `controls.length - 2` clean ancilla for n ≥ 3 controls.
 */
function mcx(c: Circuit, controls: number[], target: number, ancilla: number[]): Circuit {
  const n = controls.length
  if (n === 0) return c.x(target)
  if (n === 1) return c.cnot(controls[0]!, target)
  if (n === 2) return c.ccx(controls[0]!, controls[1]!, target)
  // Forward staircase
  c = c.ccx(controls[0]!, controls[1]!, ancilla[0]!)
  for (let i = 2; i <= n - 2; i++) c = c.ccx(controls[i]!, ancilla[i - 2]!, ancilla[i - 1]!)
  c = c.ccx(controls[n - 1]!, ancilla[n - 3]!, target)
  // Uncompute ancilla
  for (let i = n - 2; i >= 2; i--) c = c.ccx(controls[i]!, ancilla[i - 2]!, ancilla[i - 1]!)
  c = c.ccx(controls[0]!, controls[1]!, ancilla[0]!)
  return c
}

/**
 * Grover diffusion operator: H^n · (2|0⟩⟨0| - I) · H^n.
 * Acts on qubits 0..n-1; ancilla[] are clean helper qubits.
 */
function groverDiffuse(c: Circuit, n: number, ancilla: number[]): Circuit {
  for (let q = 0; q < n; q++) c = c.h(q)
  for (let q = 0; q < n; q++) c = c.x(q)
  c = c.h(n - 1)
  c = mcx(c, Array.from({ length: n - 1 }, (_, i) => i), n - 1, ancilla)
  c = c.h(n - 1)
  for (let q = 0; q < n; q++) c = c.x(q)
  for (let q = 0; q < n; q++) c = c.h(q)
  return c
}

/**
 * Grover's search circuit.
 *
 * @param n         Number of search qubits (search space = 2^n).
 * @param oracle    Callback that marks target states with a phase flip on qubits 0..n-1.
 * @param iterations Grover iterations (default: optimal ≈ (π/4)√(2^n)).
 *
 * The circuit has `n + groverAncilla(n)` total qubits.
 * Ancilla qubits (n..) are initialised to |0⟩ and returned to |0⟩ after each diffusion.
 */
export function grover(
  n: number,
  oracle: (c: Circuit) => Circuit,
  iterations?: number,
): Circuit {
  const anc = groverAncilla(n)
  const ancilla = Array.from({ length: anc }, (_, i) => n + i)
  const iters = iterations ?? Math.max(1, Math.round((Math.PI / 4) * Math.sqrt(2 ** n)))
  let c = new Circuit(n + anc)
  for (let q = 0; q < n; q++) c = c.h(q)
  for (let i = 0; i < iters; i++) {
    c = oracle(c)
    c = groverDiffuse(c, n, ancilla)
  }
  return c
}

// ── Quantum Phase Estimation ──────────────────────────────────────────────────

/**
 * Append an in-place inverse QFT over qubits 0..precision-1 of an existing circuit.
 */
function appendIqft(c: Circuit, precision: number): Circuit {
  for (let i = 0; i < Math.floor(precision / 2); i++) c = c.swap(i, precision - 1 - i)
  for (let j = 0; j < precision; j++) {
    for (let k = j - 1; k >= 0; k--) c = c.cu1(-Math.PI / 2 ** (j - k), k, j)
    c = c.h(j)
  }
  return c
}

/**
 * Quantum Phase Estimation circuit.
 *
 * Estimates the phase φ of an eigenstate |u⟩ such that U|u⟩ = e^{2πiφ}|u⟩.
 *
 * @param precision    Number of counting qubits (phase resolution = 1/2^precision).
 * @param unitary      Callback applied for each counting qubit k: applies controlled-U^{2^k}
 *                     with `control` as the control qubit and `targets` as the target register.
 * @param targetQubits Number of target qubits (default 1).
 *
 * Circuit layout: qubits 0..precision-1 = counting register, qubits precision.. = target register.
 * Initialise the target register to an eigenstate |u⟩ before running.
 */
export function phaseEstimation(
  precision: number,
  unitary: (c: Circuit, control: number, power: number, targets: number[]) => Circuit,
  targetQubits = 1,
): Circuit {
  const targets = Array.from({ length: targetQubits }, (_, i) => precision + i)
  let c = new Circuit(precision + targetQubits)
  for (let k = 0; k < precision; k++) c = c.h(k)
  for (let k = 0; k < precision; k++) c = unitary(c, k, 2 ** k, targets)
  return appendIqft(c, precision)
}

// ── Trotterized Hamiltonian simulation ────────────────────────────────────────

/**
 * Apply exp(−i·theta·P) for a single Pauli string P to circuit c.
 * Basis: X→H, Y→Rx(−π/2); Rz(2θ) on the last active qubit; reverse.
 */
function pauliEvolution(c: Circuit, n: number, ops: string, theta: number): Circuit {
  const active: [q: number, p: string][] = []
  for (let q = 0; q < n; q++) {
    const p = ops[n - 1 - q] ?? 'I'
    if (p !== 'I') active.push([q, p])
  }
  if (active.length === 0) return c
  for (const [q, p] of active) {
    if (p === 'X') c = c.h(q)
    else if (p === 'Y') c = c.rx(-Math.PI / 2, q)
  }
  for (let i = 0; i < active.length - 1; i++) c = c.cnot(active[i]![0], active[i + 1]![0])
  c = c.rz(2 * theta, active[active.length - 1]![0])
  for (let i = active.length - 2; i >= 0; i--) c = c.cnot(active[i]![0], active[i + 1]![0])
  for (const [q, p] of active) {
    if (p === 'X') c = c.h(q)
    else if (p === 'Y') c = c.rx(Math.PI / 2, q)
  }
  return c
}

/**
 * Trotterized Hamiltonian simulation: e^{−iHt} ≈ (∏_j e^{−iH_j·t/r})^r.
 *
 * Implements the Lie–Trotter product formula (order=1) and the
 * symmetric Trotter–Suzuki decomposition (order=2) for a Hamiltonian
 * expressed as a sum of Pauli strings.
 *
 * @param n           Number of qubits.
 * @param hamiltonian Pauli-string Hamiltonian — same convention as {@link PauliTerm}:
 *                    `ops[0]` acts on qubit n−1 (MSB), `ops[n−1]` on qubit 0 (LSB).
 * @param t           Total evolution time.
 * @param steps       Number of Trotter steps r (default 1). Error scales as O(t²/r).
 * @param order       1 = first-order, 2 = second-order Suzuki (error O(t³/r²), default 1).
 * @returns           Circuit approximating e^{−iHt}|ψ⟩.
 */
export function trotter(
  n: number,
  hamiltonian: PauliTerm[],
  t: number,
  steps = 1,
  order: 1 | 2 = 1,
): Circuit {
  for (const { ops } of hamiltonian) {
    if (ops.length !== n) throw new TypeError(`ops '${ops}' length must equal n (${n})`)
  }
  let c = new Circuit(n)
  const dt = t / steps
  if (order === 1) {
    for (let s = 0; s < steps; s++)
      for (const { coeff, ops } of hamiltonian)
        c = pauliEvolution(c, n, ops, coeff * dt)
  } else {
    // Second-order Suzuki: forward half-step + reverse half-step
    const rev = hamiltonian.toReversed()
    for (let s = 0; s < steps; s++) {
      for (const { coeff, ops } of hamiltonian) c = pauliEvolution(c, n, ops, coeff * dt / 2)
      for (const { coeff, ops } of rev)         c = pauliEvolution(c, n, ops, coeff * dt / 2)
    }
  }
  return c
}

// ── QAOA Max-Cut ──────────────────────────────────────────────────────────────

/**
 * Max-Cut cost Hamiltonian as a Pauli-string sum.
 *
 * H_C = Σ_{(u,v)∈E} (I − Z_u·Z_v) / 2
 *
 * `vqe(qaoa(n, edges, gamma, beta), maxCutHamiltonian(n, edges))`
 * returns the expected number of cut edges (exact, no sampling).
 */
export function maxCutHamiltonian(n: number, edges: readonly [number, number][]): PauliTerm[] {
  if (edges.length === 0) return []
  const terms: PauliTerm[] = [{ coeff: edges.length / 2, ops: 'I'.repeat(n) }]
  for (const [u, v] of edges) {
    const arr = Array<string>(n).fill('I')
    arr[n - 1 - u] = 'Z'
    arr[n - 1 - v] = 'Z'
    terms.push({ coeff: -0.5, ops: arr.join('') })
  }
  return terms
}

/**
 * QAOA circuit for Max-Cut (Farhi et al. 2014).
 *
 * Circuit: H^⊗n · [U_C(γ_l) · U_B(β_l)]_{l=0..p−1}
 *
 *   U_C(γ) = ∏_{(u,v)∈E} exp(iγ Z_u Z_v / 2)   cost unitary
 *   U_B(β) = ∏_q exp(−iβ X_q)                   mixer unitary
 *
 * Optimise γ and β with `vqe` + a classical optimiser, or sweep analytically
 * for small p. Larger p → better approximation ratio; p=1 already beats
 * the Goemans–Williamson 0.878 SDP bound for some graph families.
 *
 * @param n      Number of nodes (qubits).
 * @param edges  Graph edges — [u, v] pairs, 0-indexed.
 * @param gamma  Cost angles, one per layer.
 * @param beta   Mixer angles, one per layer (must equal gamma.length).
 */
export function qaoa(
  n: number,
  edges: readonly [number, number][],
  gamma: readonly number[],
  beta: readonly number[],
): Circuit {
  if (gamma.length !== beta.length) {
    throw new TypeError(`gamma and beta must have equal length (got ${gamma.length} vs ${beta.length})`)
  }
  let c = new Circuit(n)
  for (let q = 0; q < n; q++) c = c.h(q)
  for (let l = 0; l < gamma.length; l++) {
    for (const [u, v] of edges) c = c.cnot(u, v).rz(-gamma[l]!, v).cnot(u, v)
    for (let q = 0; q < n; q++) c = c.rx(2 * beta[l]!, q)
  }
  return c
}

// ── Variational Quantum Eigensolver ───────────────────────────────────────────

/**
 * A term in a Pauli-string Hamiltonian: coeff · (P_{n-1} ⊗ … ⊗ P_0).
 *
 * `ops` is a string of length n over {I, X, Y, Z}.
 * ops[0] acts on qubit n-1 (MSB); ops[n-1] acts on qubit 0 (LSB).
 */
export interface PauliTerm {
  coeff: number
  ops: string
}

/**
 * VQE energy estimate: ⟨ansatz|H|ansatz⟩ for a Pauli-string Hamiltonian.
 *
 * Uses exact statevector probabilities (no sampling noise).
 * Basis rotations: X → H, Y → Rx(π/2), Z → identity.
 *
 * @param ansatz      Circuit that prepares the trial state |ψ⟩.
 * @param hamiltonian List of Pauli terms.
 * @returns           ⟨ψ|H|ψ⟩
 */
export function vqe(ansatz: Circuit, hamiltonian: PauliTerm[]): number {
  const n = ansatz.qubits
  let energy = 0

  for (const { coeff, ops } of hamiltonian) {
    if (ops.length !== n)
      throw new TypeError(`ops '${ops}' length must equal ansatz.qubits (${n})`)
    if (Math.abs(coeff) < 1e-15) continue
    const upper = ops.toUpperCase()
    if (!/[XYZ]/.test(upper)) { energy += coeff; continue }

    // Rotate each qubit into the Z basis for its Pauli operator
    let rot = ansatz
    for (let q = 0; q < n; q++) {
      const pauli = upper[n - 1 - q]!  // upper[n-1-q] acts on qubit q
      if (pauli === 'X') rot = rot.h(q)
      else if (pauli === 'Y') rot = rot.rx(Math.PI / 2, q)
    }

    // ⟨P⟩ = Σ_b (-1)^{parity(b)} · Pr(b)
    //   parity(b) = XOR of bits at positions where Pauli ≠ I
    const probs = rot.exactProbs()
    let exp = 0
    for (const [bs, prob] of Object.entries(probs)) {
      let parity = 0
      for (let q = 0; q < n; q++) {
        const pauli = upper[n - 1 - q]!  // upper[n-1-q] acts on qubit q
        if (pauli !== 'I' && bs[q] === '1') parity ^= 1  // bs[q] = qubit q's bit (q0 leftmost)
      }
      exp += (parity === 0 ? 1 : -1) * prob
    }
    energy += coeff * exp
  }

  return energy
}

// ── Ansatz circuits ───────────────────────────────────────────────────────────

/** An ansatz function paired with its known parameter count. */
export interface AnsatzFn {
  (params: readonly number[]): Circuit
  /** Total number of trainable parameters. */
  paramCount: number
}

function makeAnsatz(paramCount: number, fn: (params: readonly number[]) => Circuit): AnsatzFn {
  return Object.assign(fn, { paramCount })
}

/**
 * Hardware-efficient ansatz with real amplitudes.
 *
 * Alternating layers of Ry(θ) rotations and linear CNOT entanglement,
 * with a final Ry layer.  All state amplitudes remain real throughout.
 * Total parameters: `n × (reps + 1)`.
 *
 * @example
 * const ansatz = realAmplitudes(2, 2)
 * const { energy } = minimize(ansatz, hamiltonian, Array(ansatz.paramCount).fill(0))
 */
export function realAmplitudes(n: number, reps = 3): AnsatzFn {
  const paramCount = n * (reps + 1)
  return makeAnsatz(paramCount, params => {
    if (params.length !== paramCount)
      throw new RangeError(`realAmplitudes(${n}, ${reps}) needs ${paramCount} params, got ${params.length}`)
    let c = new Circuit(n), p = 0
    for (let r = 0; r < reps; r++) {
      for (let q = 0; q < n; q++) c = c.ry(params[p++]!, q)
      for (let q = 0; q < n - 1; q++) c = c.cnot(q, q + 1)
    }
    for (let q = 0; q < n; q++) c = c.ry(params[p++]!, q)
    return c
  })
}

/**
 * Hardware-efficient ansatz with full SU(2) single-qubit rotations.
 *
 * Alternating layers of Ry(θ)·Rz(φ) rotations and linear CNOT entanglement,
 * with a final Ry·Rz layer.  Total parameters: `2n × (reps + 1)`.
 *
 * Strictly more expressive than `realAmplitudes` — can reach any product state
 * and, with entanglement, arbitrary entangled states.
 */
export function efficientSU2(n: number, reps = 3): AnsatzFn {
  const paramCount = 2 * n * (reps + 1)
  return makeAnsatz(paramCount, params => {
    if (params.length !== paramCount)
      throw new RangeError(`efficientSU2(${n}, ${reps}) needs ${paramCount} params, got ${params.length}`)
    let c = new Circuit(n), p = 0
    for (let r = 0; r < reps; r++) {
      for (let q = 0; q < n; q++) { c = c.ry(params[p++]!, q); c = c.rz(params[p++]!, q) }
      for (let q = 0; q < n - 1; q++) c = c.cnot(q, q + 1)
    }
    for (let q = 0; q < n; q++) { c = c.ry(params[p++]!, q); c = c.rz(params[p++]!, q) }
    return c
  })
}

// ── Pauli operator algebra ────────────────────────────────────────────────────

// Pauli indices: I=0, X=1, Y=2, Z=3
// PAULI_PROD[a][b] = [resultIdx, phaseExp] where physical phase = i^phaseExp
const PAULI_IDX: Readonly<Record<string, number>> = { I: 0, X: 1, Y: 2, Z: 3 }
const PAULI_CHR = 'IXYZ'
const PAULI_PROD: readonly (readonly [number, number][])[] = [
  [[0,0],[1,0],[2,0],[3,0]], // I·{I,X,Y,Z}
  [[1,0],[0,0],[3,1],[2,3]], // X·{I,X,Y,Z}: XX=I, XY=iZ, XZ=-iY
  [[2,0],[3,3],[0,0],[1,1]], // Y·{I,X,Y,Z}: YX=-iZ, YY=I, YZ=iX
  [[3,0],[2,1],[1,3],[0,0]], // Z·{I,X,Y,Z}: ZX=iY, ZY=-iX, ZZ=I
]

type CCoeff = { re: number; im: number }

function addC(a: CCoeff, b: CCoeff): CCoeff { return { re: a.re + b.re, im: a.im + b.im } }
function mulC(a: CCoeff, b: CCoeff): CCoeff {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }
}
function scaleC(c: CCoeff, s: number): CCoeff { return { re: c.re * s, im: c.im * s } }
function phaseC(c: CCoeff, exp: number): CCoeff {
  switch (exp & 3) {
    case 1: return { re: -c.im, im:  c.re }
    case 2: return { re: -c.re, im: -c.im }
    case 3: return { re:  c.im, im: -c.re }
    default: return c
  }
}

interface InternalTerm { ops: string; coeff: CCoeff }

/**
 * Pauli-string operator with full complex-coefficient arithmetic.
 *
 * Use `PauliOp.from()` to build from a real Hamiltonian and `.toTerms()` to
 * convert back for `vqe()`, `gradient()`, and `minimize()`.
 *
 * @example
 * const H = PauliOp.from([{ coeff: 1, ops: 'ZI' }, { coeff: 1, ops: 'IZ' }])
 * const V = PauliOp.from([{ coeff: 0.5, ops: 'XX' }])
 * const energy = vqe(circuit, H.add(V).toTerms())
 */
export class PauliOp {
  readonly #terms: readonly InternalTerm[]

  private constructor(terms: readonly InternalTerm[]) { this.#terms = terms }

  /** Construct from a real-coefficient Pauli Hamiltonian. */
  static from(terms: PauliTerm[]): PauliOp {
    return new PauliOp(
      terms.map(({ ops, coeff }) => ({ ops: ops.toUpperCase(), coeff: { re: coeff, im: 0 } })),
    )
  }

  static #collect(terms: InternalTerm[]): PauliOp {
    const map = new Map<string, CCoeff>()
    for (const { ops, coeff } of terms) {
      const acc = map.get(ops)
      map.set(ops, acc ? addC(acc, coeff) : { ...coeff })
    }
    return new PauliOp(
      [...map.entries()]
        .filter(([, c]) => Math.abs(c.re) > 1e-15 || Math.abs(c.im) > 1e-15)
        .map(([ops, coeff]) => ({ ops, coeff })),
    )
  }

  /** A + B */
  add(other: PauliOp): PauliOp {
    return PauliOp.#collect([...this.#terms, ...other.#terms])
  }

  /** c · A */
  scale(factor: number): PauliOp {
    return new PauliOp(this.#terms.map(t => ({ ...t, coeff: scaleC(t.coeff, factor) })))
  }

  /** A · B  (Pauli string product with phase tracking) */
  mul(other: PauliOp): PauliOp {
    const result: InternalTerm[] = []
    for (const a of this.#terms) {
      for (const b of other.#terms) {
        if (a.ops.length !== b.ops.length)
          throw new TypeError(`ops length mismatch: '${a.ops}' vs '${b.ops}'`)
        let phaseExp = 0, ops = ''
        for (let i = 0; i < a.ops.length; i++) {
          const [r, pe] = PAULI_PROD[PAULI_IDX[a.ops[i]!]!]![PAULI_IDX[b.ops[i]!]!]!
          phaseExp = (phaseExp + pe) & 3
          ops += PAULI_CHR[r]
        }
        result.push({ ops, coeff: phaseC(mulC(a.coeff, b.coeff), phaseExp) })
      }
    }
    return PauliOp.#collect(result)
  }

  /** [A, B] = AB − BA */
  commutator(other: PauliOp): PauliOp {
    return this.mul(other).add(other.mul(this).scale(-1))
  }

  /**
   * Convert to `PauliTerm[]` for use with `vqe()`, `gradient()`, and `minimize()`.
   * Throws if any imaginary coefficients exceed `tol` (operator is not Hermitian).
   */
  toTerms(tol = 1e-10): PauliTerm[] {
    for (const { ops, coeff } of this.#terms)
      if (Math.abs(coeff.im) > tol)
        throw new TypeError(`PauliOp has imaginary coefficient on '${ops}' — operator is not Hermitian`)
    return this.#terms.map(({ ops, coeff }) => ({ ops, coeff: coeff.re }))
  }
}

// ── Parameter shift rule ──────────────────────────────────────────────────────

/**
 * Compute the exact analytic gradient of ⟨ψ(params)|H|ψ(params)⟩ w.r.t. each
 * parameter using the parameter shift rule:
 *
 *   ∂⟨H⟩/∂θᵢ = ½[⟨H⟩(θᵢ + π/2) − ⟨H⟩(θᵢ − π/2)]
 *
 * This is exact (not finite-difference) for any gate of the form e^{−iθP/2}
 * where P is a Pauli operator — i.e. Rx, Ry, Rz, and all standard rotation
 * gates.  It returns 2N evaluations of `vqe()` for N parameters.
 *
 * @param ansatz      Function mapping a parameter vector to a Circuit.
 * @param hamiltonian Pauli-string Hamiltonian (same format as `vqe()`).
 * @param params      Current parameter values θ.
 * @returns           Gradient vector — `result[i]` = ∂⟨H⟩/∂params[i].
 *
 * @example
 * const ansatz = (p: readonly number[]) => new Circuit(1).ry(p[0]!, 0)
 * const H = [{ coeff: 1, ops: 'Z' }]
 * gradient(ansatz, H, [Math.PI / 4])  // ≈ [-0.7071]  (= −sin(π/4))
 */
export function gradient(
  ansatz: (params: readonly number[]) => Circuit,
  hamiltonian: PauliTerm[],
  params: readonly number[],
): number[] {
  const shift = Math.PI / 2
  return Array.from({ length: params.length }, (_, i) => {
    const plus  = params.map((p, j) => j === i ? p + shift : p)
    const minus = params.map((p, j) => j === i ? p - shift : p)
    return 0.5 * (vqe(ansatz(plus), hamiltonian) - vqe(ansatz(minus), hamiltonian))
  })
}

/** Options for {@link minimize}. */
export interface MinimizeOptions {
  /** Gradient descent learning rate (default `0.1`). */
  lr?: number
  /** Maximum number of gradient steps (default `200`). */
  steps?: number
  /** Convergence threshold: stops when gradient L2 norm < tol (default `1e-6`). */
  tol?: number
}

/** Result returned by {@link minimize}. */
export interface MinimizeResult {
  /** Optimal parameters found. */
  params: number[]
  /** ⟨ψ(params)|H|ψ(params)⟩ at those parameters. */
  energy: number
  /** Number of gradient steps taken. */
  steps: number
  /** True if the gradient norm dropped below `tol` before exhausting `steps`. */
  converged: boolean
}

/**
 * Minimize ⟨ψ(params)|H|ψ(params)⟩ with gradient descent + parameter shift.
 *
 * Runs until the gradient L2 norm drops below `tol` or the step budget is
 * exhausted.  Use `gradient()` directly if you need a different optimizer
 * (Adam, L-BFGS, etc.).
 *
 * @example
 * const ansatz = (p: readonly number[]) =>
 *   new Circuit(2).ry(p[0]!, 0).cnot(0, 1).ry(p[1]!, 1)
 * const { energy, params, converged } = minimize(ansatz, H, [0, 0])
 */
export function minimize(
  ansatz: (params: readonly number[]) => Circuit,
  hamiltonian: PauliTerm[],
  initialParams: readonly number[],
  { lr = 0.1, steps = 200, tol = 1e-6 }: MinimizeOptions = {},
): MinimizeResult {
  let params = [...initialParams]

  for (let step = 0; step < steps; step++) {
    const grad     = gradient(ansatz, hamiltonian, params)
    const gradNorm = Math.sqrt(grad.reduce((s, g) => s + g * g, 0))
    if (gradNorm < tol) {
      return { params, energy: vqe(ansatz(params), hamiltonian), steps: step, converged: true }
    }
    params = params.map((p, i) => p - lr * grad[i]!)
  }

  return { params, energy: vqe(ansatz(params), hamiltonian), steps, converged: false }
}
