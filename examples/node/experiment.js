/**
 * An experiment end to end: define a circuit family, predict it on the
 * simulator under the device's own published noise, run it on IonQ, compare
 * with error bars, keep every record.
 *
 * Run:  IONQ_API_KEY=... node examples/node/experiment.js
 *
 * Defaults to IonQ's cloud `simulator` (free, finishes in seconds) under its
 * `forte-1` noise model, so there is something to measure: the ideal model
 * returns the exact distribution, not even shot-sampled. Choose the model with
 * IONQ_NOISE (`ideal`, `aria-1`, `forte-1`, ...).
 *
 * On a QPU the queue can be hours, so the run is resumable: the first
 * invocation submits and writes `experiment.pending.json`; run it again later
 * and it collects whatever has finished, finishing the record when everything
 * is in.
 *
 *   IONQ_API_KEY=... IONQ_TARGET=qpu.forte-1 node examples/node/experiment.js
 */

// npm users: import { ... } from '@kirkelliott/ket'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import {
  Circuit, DEVICES,
  sweep, submitSweep, collectSweep, finishSweep, parsePendingExperiment, parseExperiment,
  compareExperiments,
  submitIonQ, getIonQJob, getIonQResults, ionqHistogramToCounts, countsToProbs, getIonQNoise,
} from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
if (!apiKey) {
  console.error('Set IONQ_API_KEY. Get one at https://cloud.ionq.com/settings/keys')
  process.exit(1)
}
const target = process.env.IONQ_TARGET ?? 'simulator'
const onQpu = target.startsWith('qpu.')
const simNoise = onQpu ? undefined : (process.env.IONQ_NOISE ?? 'forte-1')
const shots = Number(process.env.IONQ_SHOTS ?? 1000)
const PENDING = 'experiment.pending.json'
const DONE = 'experiment.json'

// ─── 1. The experiment: a Ry angle scan through a CNOT ────────────────────────
// P(11) should follow sin²(θ/2); every deviation from that is the device.
const grid = { theta: [0, 0.5, 1.0, 1.5708, 2.1, 2.6, 3.1416] }
const build = ({ theta }) => new Circuit(2).ry(theta, 0).cnot(0, 1)

// ─── 2. Reference and prediction, locally — free, instant ─────────────────────
const reference = await sweep(grid, build, c => c.exactProbs(), { name: 'ry-cnot exact' })

// Noise model: on a QPU, the device's latest published characterisation if we
// can get it, else the profile frozen in DEVICES. On the simulator, the DEVICES
// profile of whatever model IonQ is emulating; none if that is `ideal`.
let noise, noiseSource = 'none'
const fromDevices = name => {
  const profile = DEVICES[name]
  if (profile) { noise = profile.noise; noiseSource = `DEVICES profile ${name}` }
}
if (onQpu) {
  try {
    const r = await getIonQNoise(target, { apiKey })
    noise = r.noise
    noiseSource = `IonQ characterisation of ${new Date((r.characterization.date ?? 0) * 1000).toISOString().slice(0, 10)}`
  } catch (e) {
    fromDevices(target.slice(4))
    console.warn(`characterisation unavailable (${e.message.split('\n')[0]}); using ${noiseSource}`)
  }
} else if (simNoise !== 'ideal') fromDevices(simNoise)
const predicted = await sweep(grid, build, (c, i) => c.run({ shots, seed: i + 1, ...(noise && { noise }) }),
  { name: `ry-cnot predicted (${noiseSource})`, meta: { noise, noiseSource } })

// ─── 3. Submit, or resume ─────────────────────────────────────────────────────
let pending
if (existsSync(PENDING)) {
  pending = parsePendingExperiment(readFileSync(PENDING, 'utf8'))
  console.log(`resuming ${pending.points.length} jobs from ${PENDING}`)
} else {
  pending = await submitSweep(grid, build, async c => {
    const { id } = await submitIonQ(c.toIonQ(), { apiKey, target, shots, ...(simNoise && { noise: { model: simNoise } }) })
    return id
  }, { name: `ry-cnot on ${target}`, meta: { target, shots, ...(simNoise && { simulatorNoise: simNoise }) }, concurrency: 4 })
  writeFileSync(PENDING, JSON.stringify(pending, null, 2))
  console.log(`submitted ${pending.points.length} jobs → ${PENDING}`)
}

// ─── 4. Collect what has finished ─────────────────────────────────────────────
const collect = async (id, c) => {
  const job = await getIonQJob(id, { apiKey })
  if (job.status === 'failed' || job.status === 'canceled') throw new Error(`job ${id} ${job.status}: ${job.failure?.error ?? ''}`)
  if (job.status !== 'completed') return undefined
  const counts = ionqHistogramToCounts(await getIonQResults(job, { apiKey }), shots)
  return { outcomes: countsToProbs(counts, c.qubits, shots), shots, backend: `ionq:${target}`, jobId: id }
}
const remaining = p => p.points.filter(pt => pt.result === undefined).length

// The simulator finishes in seconds, so wait for it; a QPU can take hours, so
// collect once and come back later.
const patience = target === 'simulator' ? 30 : 1
for (let i = 0; i < patience && remaining(pending) > 0; i++) {
  if (i > 0) await new Promise(r => setTimeout(r, 2000))
  pending = await collectSweep(pending, collect, { concurrency: 4 })
  writeFileSync(PENDING, JSON.stringify(pending, null, 2))
}
if (remaining(pending) > 0) {
  console.log(`${remaining(pending)} of ${pending.points.length} jobs still queued on ${target}. Run again later to collect.`)
  process.exit(0)
}

// ─── 5. Compare: device against ideal, and against the noise-model prediction ──
const measured = finishSweep(pending)
writeFileSync(DONE, JSON.stringify(measured, null, 2))

const vsIdeal = compareExperiments(measured, reference, { seed: 1 })
const vsModel = compareExperiments(measured, predicted, { seed: 1 })

console.log(`\n${target}${simNoise ? ` (${simNoise} noise model)` : ''}, ${shots} shots per point — P(11) against sin²(θ/2)\n`)
console.log('   θ      exact   model   device   TVD vs exact   σ    TVD vs model   σ')
for (const [i, p] of measured.points.entries()) {
  const theta = p.params.theta
  const f = x => (x ?? 0).toFixed(3)
  const sig = x => x === undefined ? '  — ' : x.toFixed(1).padStart(4)
  console.log(
    `  ${theta.toFixed(2).padStart(4)}   ${f(reference.points[i].probs['11'])}   ${f(predicted.points[i].probs['11'])}   ${f(p.probs['11'])}    ` +
    `${vsIdeal.points[i].tvd.toFixed(3)}        ${sig(vsIdeal.points[i].sigma)}     ${vsModel.points[i].tvd.toFixed(3)}        ${sig(vsModel.points[i].sigma)}`)
}
console.log(`
mean TVD vs exact ${vsIdeal.meanTvd.toFixed(4)}, beyond shot noise at ${vsIdeal.beyondNoise.length}/${measured.points.length} points
mean TVD vs model ${vsModel.meanTvd.toFixed(4)}, beyond shot noise at ${vsModel.beyondNoise.length}/${measured.points.length} points
noise model: ${noiseSource}${noise ? ' ' + JSON.stringify(noise) : ''}

records: ${DONE} (measured), ${PENDING} (job ids). Reload with parseExperiment / parsePendingExperiment.`)

// Both records round-trip; this is the shape anything downstream can rely on.
parseExperiment(readFileSync(DONE, 'utf8'))
