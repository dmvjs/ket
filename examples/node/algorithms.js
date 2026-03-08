/**
 * Quantum algorithms — QFT, phase estimation, and VQE.
 *
 * Run:  node examples/node/algorithms.js
 */

// npm users: import { Circuit, qft, iqft, vqe } from 'ket'
import { Circuit, qft, iqft, vqe } from '../../dist/ket.js'

// ─── Quantum Fourier Transform ─────────────────────────────────────────────────
console.log('QFT(4) circuit:')
console.log(qft(4).draw())

// Round-trip: QFT then IQFT on any state returns the original probabilities.
const prep = new Circuit(3).h(0).x(2)           // an arbitrary initial state

const roundTrip = new Circuit(3)
  .defineGate('prep',  prep)
  .defineGate('qft',   qft(3))
  .defineGate('iqft',  iqft(3))
  .gate('prep',  0, 1, 2)
  .gate('qft',   0, 1, 2)
  .gate('iqft',  0, 1, 2)

console.log('\nQFT → IQFT round-trip (original vs restored probabilities):')
const orig    = prep.exactProbs()
const restored = roundTrip.exactProbs()
for (const [bs, p] of Object.entries(orig)) {
  const r    = restored[bs] ?? 0
  const diff = Math.abs(r - p)
  console.log(`  |${bs}⟩  orig=${p.toFixed(4)}  restored=${r.toFixed(4)}  Δ=${diff.toExponential(1)}`)
}

// ─── Quantum Phase Estimation (T gate, φ = 1/8) ──────────────────────────────
//
// T|1⟩ = e^{iπ/4}|1⟩ → eigenphase φ = 1/8.
// 3 counting qubits (q0..q2) + 1 target qubit (q3, prepared in |1⟩).
// After QPE the counting register reads |001⟩, encoding φ = 1/8.

let qpe = new Circuit(4).x(3)                                             // |1⟩ on target
for (let k = 0; k < 3; k++) qpe = qpe.h(k)                               // superpose counting
for (let k = 0; k < 3; k++) qpe = qpe.crz(Math.PI / 2 * (2 ** k), k, 3) // CU^{2^k}
qpe = qpe.swap(0, 2)
for (let j = 0; j < 3; j++) {
  for (let k = j - 1; k >= 0; k--) qpe = qpe.cu1(-Math.PI / 2 ** (j - k), k, j)
  qpe = qpe.h(j)
}

const qpeProbs  = qpe.exactProbs()
const [top, p]  = Object.entries(qpeProbs).sort((a, b) => b[1] - a[1])[0]
console.log('\nQuantum Phase Estimation — T gate (eigenphase = 1/8):')
console.log(`  Top bitstring: |${top}⟩  probability: ${p.toFixed(6)}`)
console.log(`  Counting bits q0..q2: ${top.slice(1).split('').reverse().join('')}  → phase = 1/8 ✓`)

// ─── Variational Quantum Eigensolver ──────────────────────────────────────────
//
// Hamiltonian H = 0.5·Z + 0.5·X.
// Ground state energy = −1/√2 ≈ −0.7071, achieved at θ ≈ 5π/4 for Ry(θ)|0⟩.

const H = [{ coeff: 0.5, ops: 'Z' }, { coeff: 0.5, ops: 'X' }]

console.log('\nVQE sweep — Ry(θ) ansatz for H = 0.5·Z + 0.5·X:')
let best = { theta: 0, energy: Infinity }
for (let i = 0; i <= 16; i++) {
  const theta  = (i / 16) * 2 * Math.PI
  const energy = vqe(new Circuit(1).ry(theta, 0), H)
  const bar    = energy < 0 ? '█'.repeat(Math.round(-energy * 20)) : '·'
  console.log(`  θ = ${(theta / Math.PI).toFixed(2).padStart(4)}π   E = ${energy.toFixed(4).padStart(8)}   ${bar}`)
  if (energy < best.energy) best = { theta, energy }
}
console.log(`\n  Minimum energy: ${best.energy.toFixed(6)}   (exact: ${(-Math.SQRT2 / 2).toFixed(6)})`)
