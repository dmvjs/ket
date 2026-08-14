/**
 * Submit a Shor period-finding circuit to IonQ.
 *
 * Run:  IONQ_API_KEY=... node examples/node/ionq-shor.js
 *       IONQ_API_KEY=... SHOR_PRECISION=5 node examples/node/ionq-shor.js
 *
 * Circuit size is set by the counting-register precision, not by N. The
 * textbook default of 2n+1 counting qubits is far more than N=15 requires:
 * every base has order 2 or 4, both of which divide 2^3, so three counting
 * qubits resolve the phase exactly. That is a third of the gates.
 *
 * The circuit is verified locally before submission — an expensive job that
 * computes the wrong thing is worse than no job at all.
 */
import { shorCircuit, submitIonQ } from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
if (!apiKey) {
  console.error('Set IONQ_API_KEY. Get one at https://cloud.ionq.com/settings/keys')
  process.exit(1)
}
const N = BigInt(process.env.SHOR_N ?? 15)
const A = BigInt(process.env.SHOR_A ?? 7)
const PRECISION = Number(process.env.SHOR_PRECISION ?? 3)
const SHOTS = Number(process.env.IONQ_SHOTS ?? 1024)
const TARGET = process.env.IONQ_TARGET ?? 'simulator'

const c = shorCircuit(N, A, PRECISION)

// ─── Local check ──────────────────────────────────────────────────────────────
// The counting register should concentrate on multiples of 2^precision / r.
const dist = c.run({ shots: 2000, seed: 11 })
const peaks = new Map()
for (const [bits, p] of Object.entries(dist.probs)) {
  let v = 0
  for (let k = 0; k < PRECISION; k++) if (bits[k] === '1') v |= 1 << k
  peaks.set(v, (peaks.get(v) ?? 0) + p)
}
const top = [...peaks].filter(([, w]) => w > 0.02).sort((a, b) => a[0] - b[0])
console.log(`N = ${N}, a = ${A}, precision = ${PRECISION}`)
console.log('local peaks    :', top.map(([v, w]) => `${v}:${(w * 100).toFixed(1)}%`).join('  '))

// Counting peaks is not enough: a uniform distribution has the most peaks of all.
// Nor is peak height — with r peaks each carries only 1/r, so a large r is
// legitimately close to uniform (N=15, a=7 gives four peaks at 25% against 12.5%).
//
// The invariant that actually holds: when the order of `a` divides 2^precision,
// phase estimation is exact and all but r of the 2^precision outcomes are
// identically zero. A spread spectrum leaves none of them empty. So measure
// sparsity, which is scale-free in r.
const Q = 2 ** PRECISION
const uniform = 1 / Q
let empty = 0
for (let v = 0; v < Q; v++) if ((peaks.get(v) ?? 0) < 0.1 * uniform) empty++
if (empty < Q / 4) {
  console.error(`\nNo periodic structure: ${empty} of ${Q} outcomes are empty; exact phase ` +
                'estimation leaves all but the peaks at zero.')
  console.error(`The order of a=${A} mod ${N} does not divide ${Q}. Either raise SHOR_PRECISION,`)
  console.error('or pick a base whose order is a power of two.')
  process.exit(1)
}

// ─── Expand and size ──────────────────────────────────────────────────────────
// toIonQ() is a serializer and rejects cu1/cswap; toIonQBasis() expands them.
const native = c.toIonQBasis()
const payload = native.toIonQ()
const mb = JSON.stringify(payload).length / 1e6
console.log(`qubits         : ${c.qubits}`)
console.log(`gates          : ${c.gateCounts().total} source -> ${payload.circuit.length} expanded`)
console.log(`payload        : ${mb.toFixed(2)} MB`)
if (TARGET !== 'simulator') native.checkDevice(TARGET.replace('qpu.', ''))

// ─── Submit ───────────────────────────────────────────────────────────────────
// Deliberately does not block: large circuits can sit for a long time, and the
// job id is enough to recover the result later.
const { id } = await submitIonQ(payload, {
  apiKey, target: TARGET, shots: SHOTS, name: `ket Shor N=${N}, a=${A}, p=${PRECISION}`,
  // Stated rather than inherited: at this depth a device noise model flattens
  // the distribution completely, which looks identical to a wrong answer.
  ...(TARGET === 'simulator' ? { noise: { model: process.env.IONQ_NOISE ?? 'ideal' } } : {}),
})
console.log(`\nsubmitted      : ${id}`)
console.log(`check with     : node examples/node/ionq-job.js ${id}`)
console.log(`cancel with    : node examples/node/ionq-cancel.js ${id}`)
