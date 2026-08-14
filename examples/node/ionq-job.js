/**
 * Inspect an IonQ job by id, and decode its results if it has finished.
 *
 * Run:  IONQ_API_KEY=... node examples/node/ionq-job.js <job-id>
 *       IONQ_API_KEY=... node examples/node/ionq-job.js <job-id> --watch
 *
 * Jobs live on IonQ's side, so interrupting whatever submitted them loses
 * nothing — the id is enough to recover the work. `ready` means accepted and
 * queued, not failed.
 */
import { getIonQJob, awaitIonQJob, getIonQResults } from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
const id = process.argv[2]
if (!apiKey || !id) {
  console.error('usage: IONQ_API_KEY=... node examples/node/ionq-job.js <job-id> [--watch]')
  process.exit(1)
}

const job = process.argv.includes('--watch')
  ? await awaitIonQJob(id, {
      apiKey, timeoutSeconds: 7200, intervalSeconds: 10,
      onPoll: j => process.stdout.write(`\r${j.status}   `),
    })
  : await getIonQJob(id, { apiKey })
process.stdout.write('\r')

console.log(`id        ${job.id}`)
console.log(`status    ${job.status}`)
console.log(`target    ${job.target ?? '?'}`)
console.log(`qubits    ${job.qubits ?? '?'}`)
if (job.gate_counts)    console.log(`gates     ${JSON.stringify(job.gate_counts)}`)
// The server picks a noise model when the request omits one, so report what was
// actually used rather than what was intended.
console.log(`noise     ${job.noise ? JSON.stringify(job.noise) : '(not reported — server default)'}`)
if (job.shots)          console.log(`shots     ${job.shots}`)
if (job.execution_time) console.log(`exec      ${job.execution_time} ms`)
if (job.cost_usd !== undefined) console.log(`cost      $${job.cost_usd}`)
if (job.failure)        console.log(`failure   ${JSON.stringify(job.failure)}`)

if (job.status !== 'completed') {
  console.log('\nNot finished. `ready` means queued and accepted, not failed.')
  console.log('Re-run with --watch to block until it lands.')
  console.log('Jobs are serialized per account, so a long one blocks everything')
  console.log(`behind it: node examples/node/ionq-cancel.js ${job.id} releases the slot.`)
  process.exit(0)
}

const hist = await getIonQResults(job, { apiKey })
const top = Object.entries(hist).sort((a, b) => b[1] - a[1])
console.log(`\ndistinct outcomes: ${top.length}`)
for (const [k, p] of top.slice(0, 10)) {
  console.log(`  ${String(k).padStart(10)}  ${(p * 100).toFixed(2)} %`)
}
