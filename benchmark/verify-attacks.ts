/**
 * Does the attack the adversary recommends actually give the right answer?
 *
 * Every verdict so far has been a *prediction* — "tensor-network at width 20 fits
 * your budget" — and none of them was ever cashed. A recommendation that is
 * affordable and wrong is worse than no recommendation, so this runs the winning
 * attack and checks it against ground truth.
 *
 * Ground truth is the exact statevector, which caps the sizes here. That is the
 * point rather than a limitation: the attacks are the same code at any width, so
 * validating them where an oracle exists is what earns trust where none does. The
 * published-scale runs in `published-circuits.ts` inherit exactly this much
 * confidence and no more.
 *
 * Circuits are real fragments of the published families: the light cone of a
 * single-site observable on IBM's actual heavy-hex lattice, and small instances of
 * Google's coupler-pattern family. The transverse-field angle sweeps across the
 * published range, including the two endpoints where the circuit is entirely
 * Clifford and the answer is known exactly.
 *
 * Run with: node --max-old-space-size=8192 benchmark/verify-attacks.ts
 */
import {
  Circuit, amplitudeByContraction, heavyHex127, lightCone, pauliPathExpectation,
  rcsCircuit, searchHardness, trotterIsing, type Attack,
} from '@kirkelliott/ket'

const say = (s: string): void => { process.stdout.write(`${s}\n`) }
const BUDGET = { memoryBytes: 16e9, seconds: 3600, probeChi: 8, probeTerms: 4096, restarts: 8 }

/** Largest deviation between two distributions, over the union of their support. */
function maxDeviation(a: Record<string, number>, b: Record<string, number>): number {
  let worst = 0
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    worst = Math.max(worst, Math.abs((a[k] ?? 0) - (b[k] ?? 0)))
  }
  return worst
}

/**
 * Run the recommended attack and measure how far off it is.
 *
 * Returns null for an attack with nothing to cash: a report where nothing landed
 * has no recommendation to test.
 */
function cash(circuit: Circuit, attack: Attack): { error: number; what: string } | null {
  const exact = circuit.exactProbs()
  const bits = '0'.repeat(circuit.qubits)
  switch (attack.name) {
    case 'clifford':
      return { error: maxDeviation(exact, circuit.runClifford({ shots: 200_000, seed: 1 }).probs), what: 'distribution' }
    case 'mps':
      return { error: maxDeviation(exact, circuit.runMps({ shots: 200_000, seed: 1 }).probs), what: 'distribution' }
    case 'statevector':
      return { error: maxDeviation(exact, circuit.run({ shots: 200_000, seed: 1 }).probs), what: 'distribution' }
    case 'stabilizer-rank':
      return { error: maxDeviation(exact, circuit.runStabilizerRank({ shots: 200_000, seed: 1 }).probs), what: 'distribution' }
    case 'tensor-network': {
      const want = circuit.amplitude(bits)
      const got = amplitudeByContraction(circuit, bits, { restarts: 8 })
      return { error: Math.max(Math.abs(got.re - want.re), Math.abs(got.im - want.im)), what: 'amplitude ⟨0…0|ψ⟩' }
    }
    case 'pauli-path': case 'noisy-pauli-path': {
      const n = circuit.qubits
      const term = { coeff: 1, ops: Array.from({ length: n }, (_, k) => k === n - 1 ? 'Z' : 'I').join('') }
      // Ground truth for ⟨Z_{n-1}⟩ straight off the exact distribution.
      let truth = 0
      for (const [b, p] of Object.entries(exact)) truth += (b[n - 1] === '1' ? -1 : 1) * p
      const got = pauliPathExpectation(circuit, [term], { threshold: 1e-12, maxTerms: 1 << 18 })
      return { error: Math.abs(got.value - truth), what: '⟨Z⟩' }
    }
    default: return null
  }
}

/** Sampling error at 200k shots swamps anything smaller, so judge against that. */
const SAMPLING_FLOOR = 6 * Math.sqrt(0.25 / 200_000)

interface Row { label: string; qubits: number; attack: string; what: string; error: number; ok: boolean }
const rows: Row[] = []

function check(label: string, circuit: Circuit, about: number[]): void {
  const s = searchHardness(circuit, { ...BUDGET, about })
  const reduced = s.circuit
  if (reduced.qubits > 20) { say(`${label}: cone left ${reduced.qubits} qubits — no oracle, skipped`); return }
  const best = s.after.best
  if (!best) { rows.push({ label, qubits: reduced.qubits, attack: 'none landed', what: '—', error: NaN, ok: true }); return }
  const got = cash(reduced, best)
  if (!got) { say(`${label}: ${best.name} has nothing to cash here`); return }
  rows.push({
    label, qubits: reduced.qubits, attack: best.name, what: got.what, error: got.error,
    ok: got.error < SAMPLING_FLOOR + 1e-6,
  })
}

const THETAS: [string, number][] = [
  ['0 (Clifford)', 0],
  ['π/8', Math.PI / 8],
  ['π/4', Math.PI / 4],
  ['1.2', 1.2],
  ['π/2 (Clifford)', Math.PI / 2],
]

say('Cashing the adversary\'s recommendation against the exact statevector.')
say(`A row passes when the error is below the 200k-shot sampling floor (${SAMPLING_FLOOR.toExponential(1)}).\n`)

// IBM's real lattice, reduced by the light cone of one observable to a size an
// oracle can reach. Depth 1 keeps the cone small enough on a degree-3 graph.
for (const [name, theta] of THETAS) {
  for (const depth of [1, 2]) {
    const c = trotterIsing({ qubits: 127, coupling: heavyHex127(), depth, thetaH: theta })
    const cone = lightCone(c, [62])
    check(`IBM heavy-hex d=${depth}, θ=${name}`, cone.circuit, [cone.mapping![62]!])
  }
}

// Google's coupler-pattern family, small enough to hold whole.
for (const [rows_, cols, depth] of [[2, 4, 4], [3, 4, 6], [2, 5, 8]] as const) {
  const c = rcsCircuit({ rows: rows_, cols, depth, seed: 3 })
  check(`Google RCS ${rows_}x${cols} d=${depth}`, c, [0])
}

const w = [
  Math.max(...rows.map(r => r.label.length), 5),
  6, 16, 18, 11,
]
say(['circuit'.padEnd(w[0]!), 'qubits', 'attack'.padEnd(w[2]!), 'answers'.padEnd(w[3]!), 'error'].join('  '))
say(w.map(x => '─'.repeat(x)).join('  '))
for (const r of rows) {
  say([
    r.label.padEnd(w[0]!), String(r.qubits).padStart(6), r.attack.padEnd(w[2]!),
    r.what.padEnd(w[3]!), Number.isNaN(r.error) ? '—' : `${r.error.toExponential(1)} ${r.ok ? '✓' : '✗ WRONG'}`,
  ].join('  '))
}
const bad = rows.filter(r => !r.ok)
say('')
say(bad.length === 0
  ? `all ${rows.length} recommendations cashed correctly`
  : `${bad.length} of ${rows.length} recommendations returned a WRONG answer: ${bad.map(r => r.label).join(', ')}`)
if (bad.length > 0) process.exitCode = 1
