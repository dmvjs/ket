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
