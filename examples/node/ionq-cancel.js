/**
 * Cancel a queued or running IonQ job.
 *
 * Run:  IONQ_API_KEY=... node examples/node/ionq-cancel.js <job-id>
 *
 * Jobs are serialized per account: one long submission holds up everything
 * behind it. If a small circuit that used to run in seconds is sitting at
 * `ready`, cancelling whatever is ahead of it releases the slot.
 */
import { cancelIonQJob } from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
const id = process.argv[2]
if (!apiKey || !id) {
  console.error('usage: IONQ_API_KEY=... node examples/node/ionq-cancel.js <job-id>')
  process.exit(1)
}

const job = await cancelIonQJob(id, { apiKey })
console.log(`id      ${job.id ?? id}`)
console.log(`status  ${job.status ?? 'canceled'}`)
console.log('\nSlot released. Anything queued behind it should start shortly.')
