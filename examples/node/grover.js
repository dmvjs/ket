/**
 * Grover's search — amplitude amplification on 2 and 3 qubits.
 *
 * Run:  node examples/node/grover.js
 */

// npm users: import { Circuit, grover } from 'ket'
import { Circuit, grover } from '../../dist/ket.js'

// ─── 2-qubit search: mark |11⟩ ────────────────────────────────────────────────
// The oracle applies a phase flip to the target state.
// CZ flips the phase of |11⟩ (both controls in |1⟩).
// For N=4 (2 qubits), 1 Grover iteration gives the exact maximum.
const search2 = grover(2, c => c.cz(0, 1), 1)

console.log('2-qubit Grover — oracle marks |11⟩')
console.log(search2.draw())

const probs2 = search2.exactProbs()
console.log('Exact probabilities:')
for (const [bs, p] of Object.entries(probs2).sort()) {
  const bar = '█'.repeat(Math.round(p * 30))
  console.log(`  |${bs}⟩  ${bar.padEnd(30)}  ${(p * 100).toFixed(1)}%`)
}

// ─── 3-qubit search: mark |111⟩ ───────────────────────────────────────────────
// CCZ = H·CCX·H applies a phase flip when all three qubits are |1⟩.
const search3 = grover(3, c => c.h(2).ccx(0, 1, 2).h(2))

console.log('\n3-qubit Grover — oracle marks |111⟩')
const top3 = search3.run({ shots: 2048, seed: 1 }).most
console.log('Most probable outcome:', top3)
// → '111'
