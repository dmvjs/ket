/**
 * Quantum teleportation — transfer an arbitrary qubit state over a Bell channel.
 *
 * Protocol (unitary version — no mid-circuit classical control needed):
 *   q0 = payload state (Alice's qubit)
 *   q1 = Alice's half of the Bell pair
 *   q2 = Bob's half of the Bell pair
 *
 *   1. Create Bell pair on q1/q2
 *   2. Alice: CNOT(q0 → q1), H(q0)
 *   3. Bob: CX(q1, q2), CZ(q0, q2)   ← equivalent to measuring q0/q1 and correcting
 *   4. Trace out q0/q1 — q2 is left in the original payload state.
 *
 * Run:  node examples/node/teleportation.js
 */

// npm users: import { Circuit } from 'ket'
import { Circuit } from '../../dist/ket.js'

function teleport(prep) {
  let c = new Circuit(3)
  c = prep(c)                    // prepare payload on q0
  c = c.h(1).cnot(1, 2)         // Bell pair on q1/q2
  c = c.cnot(0, 1).h(0)         // Alice's Bell measurement
  c = c.cx(1, 2).cz(0, 2)       // Bob's corrections
  return c
}

// Helper: probability of qubit 2 measuring |0⟩ across all bitstrings
function pQubit2(probs) {
  let p0 = 0
  for (const [bs, p] of Object.entries(probs)) {
    if (bs[0] === '0') p0 += p   // qubit 2 is the MSB in a 3-qubit string
  }
  return p0
}

const cases = [
  { label: '|0⟩',         prep: c => c,                           expected: 1.0 },
  { label: '|1⟩',         prep: c => c.x(0),                     expected: 0.0 },
  { label: '|+⟩',         prep: c => c.h(0),                     expected: 0.5 },
  { label: 'Rx(π/3)|0⟩',  prep: c => c.rx(Math.PI / 3, 0),       expected: Math.cos(Math.PI / 6) ** 2 },
]

console.log('Quantum teleportation — P(q2 = |0⟩) after teleporting each state:\n')

for (const { label, prep, expected } of cases) {
  const circuit  = teleport(prep)
  const probs    = circuit.exactProbs()
  const p0       = pQubit2(probs)
  const match    = Math.abs(p0 - expected) < 1e-9 ? '✓' : '✗'
  console.log(`  ${label.padEnd(14)}  P(q2=0) = ${p0.toFixed(6)}  expected ${expected.toFixed(6)}  ${match}`)
}
