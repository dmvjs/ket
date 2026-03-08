/**
 * Noise models and density matrix simulation.
 *
 * Shows: named device profiles, depolarizing channels, purity decay, von Neumann
 * entropy, and exact mixed-state probabilities without Monte Carlo sampling.
 *
 * Run:  node examples/node/noise.js
 */

// npm users: import { Circuit } from 'ket'
import { Circuit } from '../../dist/ket.js'

const bell = new Circuit(2).h(0).cnot(0, 1)

// ─── Pure state baseline ───────────────────────────────────────────────────────
const pure = bell.dm()
console.log('Pure Bell state:')
console.log('  purity   =', pure.purity().toFixed(6), '(1.0 = perfectly pure)')
console.log('  entropy  =', pure.entropy().toFixed(6), 'bits (0 = pure state)')
console.log('  probs    =', pure.probabilities())

// ─── Named device noise profiles ──────────────────────────────────────────────
console.log('\nPurity under named device noise (Bell state: H + CNOT):')
for (const profile of ['aria-1', 'forte-1', 'harmony']) {
  const dm = bell.dm({ noise: profile })
  console.log(`  ${profile.padEnd(10)}  purity = ${dm.purity().toFixed(6)}  entropy = ${dm.entropy().toFixed(4)} bits`)
}

// ─── Custom noise parameters ───────────────────────────────────────────────────
console.log('\nPurity vs. two-qubit gate error rate p2:')
for (const p2 of [0, 0.01, 0.05, 0.10, 0.20]) {
  const dm = bell.dm({ noise: { p2 } })
  const bar = '█'.repeat(Math.round(dm.purity() * 20))
  console.log(`  p2=${String(p2).padEnd(5)}  ${bar.padEnd(20)}  purity = ${dm.purity().toFixed(4)}`)
}

// ─── Single-qubit depolarizing channel ────────────────────────────────────────
// After H then depolarize(p1=0.75): off-diagonal → 0, diagonal → 0.5 each.
// This is the maximally mixed state: entropy = 1 bit.
const maxMixed = new Circuit(1).h(0).dm({ noise: { p1: 0.75 } })
console.log('\nMaximally mixed state (H + p1=0.75):')
console.log('  purity  =', maxMixed.purity().toFixed(6), '(0.5 = maximally mixed for 1 qubit)')
console.log('  entropy =', maxMixed.entropy().toFixed(6), 'bits (1.0 = max entropy for 1 qubit)')
console.log('  probs   =', maxMixed.probabilities())

// ─── Bloch sphere from density matrix ─────────────────────────────────────────
console.log('\nBloch sphere coordinates for qubit 0 of a noisy Bell state:')
const noisyBell = bell.dm({ noise: 'aria-1' })
console.log('  theta =', noisyBell.blochAngles(0).theta.toFixed(6), '(π/2 = equatorial = maximally mixed)')
console.log('  phi   =', noisyBell.blochAngles(0).phi.toFixed(6))
