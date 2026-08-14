/**
 * Submit Shor's two-register discrete-logarithm circuit to IonQ.
 *
 * Run:  IONQ_API_KEY=... node examples/node/ionq-dlog.js
 *       IONQ_API_KEY=... DLOG_P=17 DLOG_G=3 DLOG_X=5 node examples/node/ionq-dlog.js
 *
 * Defaults to p=5, which expands to 8,907 gates. The p=17 instance in
 * dlog-shor.js expands to 54,031 — larger than circuits IonQ's simulator has
 * been observed to reject at compile time, so it is not the default here.
 *
 * Companion to ionq-shor.js, exercising a different circuit shape: two exponent
 * registers and two inverse QFTs rather than one. See examples/node/dlog-shor.js
 * for the mathematics and the local-only version.
 *
 * g must have order a power of two so the transform is exact; dlog-shor.js
 * reports a suitable generator when the supplied one does not qualify.
 */
import { Circuit, submitIonQ, modPow, modInverse, beauregardU, applyIqft } from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
if (!apiKey) {
  console.error('Set IONQ_API_KEY. Get one at https://cloud.ionq.com/settings/keys')
  process.exit(1)
}
const P = BigInt(process.env.DLOG_P ?? 5)
const G = BigInt(process.env.DLOG_G ?? 2)
const X = BigInt(process.env.DLOG_X ?? 3)
const SHOTS = Number(process.env.IONQ_SHOTS ?? 1024)
const TARGET = process.env.IONQ_TARGET ?? 'simulator'

let r = 0n
for (let k = 1n; k < P; k++) if (modPow(G, k, P) === 1n) { r = k; break }
if (r === 0n || (r & (r - 1n)) !== 0n) {
  console.error(`g=${G} has order ${r}, which is not a power of two.`)
  console.error('Run examples/node/dlog-shor.js — it reports a suitable generator.')
  process.exit(1)
}

const h = modPow(G, X, P)
const t = r.toString(2).length - 1
const n = P.toString(2).length
const A = 0, B = A + t, XR = B + t, ACC = XR + n, ANC = ACC + n + 1

let c = new Circuit(ANC + 1)
for (let j = 0; j < t; j++) c = c.h(A + j).h(B + j)
c = c.x(XR)
for (const [base, reg] of [[G, A], [h, B]])
  for (let j = 0; j < t; j++) {
    const a = modPow(base, 1n << BigInt(j), P)
    c = beauregardU(c, n, a, modInverse(a, P), P, reg + j, XR, ACC, ANC)
  }
c = applyIqft(applyIqft(c, t, A), t, B)

const native = c.toIonQBasis()
const payload = native.toIonQ()
console.log(`p = ${P}, g = ${G}, order r = ${r} = 2^${t}, h = ${h} (x = ${X} withheld)`)
console.log(`qubits         : ${c.qubits}`)
console.log(`gates          : ${c.gateCounts().total} source -> ${payload.circuit.length} expanded`)
console.log(`payload        : ${(JSON.stringify(payload).length / 1e6).toFixed(2)} MB`)
if (TARGET !== 'simulator') native.checkDevice(TARGET.replace('qpu.', ''))

const { id } = await submitIonQ(payload, {
  apiKey, target: TARGET, shots: SHOTS, name: `ket dlog p=${P}, g=${G}`,
  ...(TARGET === 'simulator' ? { noise: { model: process.env.IONQ_NOISE ?? 'ideal' } } : {}),
})
console.log(`\nsubmitted      : ${id}`)
console.log(`check with     : node examples/node/ionq-job.js ${id}`)
console.log(`\nOutcomes should satisfy beta = x*alpha (mod ${r}), with alpha in qubits`)
console.log(`0-${t - 1} and beta in qubits ${t}-${2 * t - 1}; then x = beta * alpha^-1.`)
