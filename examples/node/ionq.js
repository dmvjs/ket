/**
 * IonQ hardware submission — simulate locally, then run the same circuit remotely.
 *
 * Run:  IONQ_API_KEY=... node examples/node/ionq.js
 *
 * Defaults to IonQ's cloud `simulator` target: free, no queue. Use it to confirm
 * your key and this client work before spending QPU time. For real hardware:
 *
 *   IONQ_API_KEY=... IONQ_TARGET=qpu.forte-1 node examples/node/ionq.js
 */

// npm users: import { ... } from '@kirkelliott/ket'
import { Circuit, runIonQ, countsToProbs, IONQ_DEVICES } from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
if (!apiKey) {
  console.error('Set IONQ_API_KEY. Get one at https://cloud.ionq.com/settings/keys')
  process.exit(1)
}
const target = process.env.IONQ_TARGET ?? 'simulator'
const shots = Number(process.env.IONQ_SHOTS ?? 1024)

// ─── Build and check locally — free, instant ──────────────────────────────────
const bell = new Circuit(2).h(0).cnot(0, 1)

console.log(bell.draw())
console.log('local simulation :', bell.exactProbs())

// Fail before spending queue time if the circuit cannot run on the target.
// ket ships profiles for a subset of IonQ's fleet; targets it does not know
// (aria-2, forte-enterprise-1) are submitted without a local pre-flight rather
// than rejected here.
const device = target.replace('qpu.', '')
if (device in IONQ_DEVICES) bell.checkDevice(device)
else if (target !== 'simulator') console.log(`  (no local profile for ${target} — skipping pre-flight)`)

// ─── Submit, wait, fetch results ──────────────────────────────────────────────
console.log(`\nsubmitting to ${target} (${shots} shots)…`)
const { job, counts } = await runIonQ(bell.toIonQ(), {
  apiKey, target, shots, name: 'ket bell',
  onPoll: j => process.stdout.write(`\r  status: ${j.status}          `),
})
process.stdout.write('\n')
console.log('job id           :', job.id)
console.log('hardware result  :', countsToProbs(counts, bell.qubits, shots))
console.log(
  '\n|00> and |11> should dominate; |01>/|10> weight is noise' +
  (target === 'simulator' ? ' (none expected on the ideal simulator).' : '.'),
)
