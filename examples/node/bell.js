/**
 * Bell state — entanglement, state inspection, and sampling.
 *
 * Run:  node examples/node/bell.js
 */

// npm users: import { Circuit } from 'ket'
import { Circuit } from '../../dist/ket.js'

// ─── Build the circuit ─────────────────────────────────────────────────────────
// Immutable: every gate method returns a new Circuit.
const bell = new Circuit(2).h(0).cnot(0, 1)

// ─── Visualize ────────────────────────────────────────────────────────────────
console.log('Circuit:')
console.log(bell.draw())

// ─── State inspection ─────────────────────────────────────────────────────────
console.log('State:  ', bell.stateAsString())
// 0.7071|00⟩ + 0.7071|11⟩

console.log('Probs:  ', bell.exactProbs())
// { '00': 0.5, '11': 0.5 }

console.log('|11⟩ amplitude:', bell.amplitude('11'))
// { re: 0.7071..., im: 0 }

// Bloch sphere for each qubit.
// A Bell state is maximally entangled: each qubit is individually maximally mixed.
// The Bloch vector has length 0, giving theta = π/2, phi = 0 (equatorial, undefined φ).
console.log('Bloch q0:', bell.blochAngles(0))
console.log('Bloch q1:', bell.blochAngles(1))

// ─── Shot-based sampling ──────────────────────────────────────────────────────
const result = bell
  .creg('out', 2)
  .measure(0, 'out', 0)
  .measure(1, 'out', 1)
  .run({ shots: 1000, seed: 42 })

console.log('\nDistribution (1000 shots, seed 42):')
console.log(result.render())
