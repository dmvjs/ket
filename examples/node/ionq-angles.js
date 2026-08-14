/**
 * Verify IonQ's rotation-angle convention against measured probabilities.
 *
 * Run:  IONQ_API_KEY=... node examples/node/ionq-angles.js
 *
 * H · Rz(theta) · H on one qubit gives p(0) = cos^2(theta/2). That is a curve,
 * not a single value, so it pins the angle scale rather than merely agreeing
 * with it: a convention off by a factor of pi still produces valid output, just
 * for a different theta, and the shape of the disagreement identifies the factor.
 *
 * This is the check that would have caught ket sending `rotation` in pi-radians
 * where IonQ's QIS gateset expects radians. A Bell pair cannot catch it — it has
 * no rotation fields at all.
 */
import { Circuit, submitIonQ, awaitIonQJob, getIonQResults } from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
if (!apiKey) {
  console.error('Set IONQ_API_KEY. Get one at https://cloud.ionq.com/settings/keys')
  process.exit(1)
}

const ANGLES = [0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4, Math.PI]
const results = []

for (const theta of ANGLES) {
  const c = new Circuit(1).h(0).rz(theta, 0).h(0)
  const { id } = await submitIonQ(c.toIonQ(), {
    apiKey, target: 'simulator', shots: 1024, noise: { model: 'ideal' },
    name: `ket angle check theta=${theta.toFixed(4)}`,
  })
  const job = await awaitIonQJob(id, { apiKey, timeoutSeconds: 600 })
  const hist = await getIonQResults(job, { apiKey })
  // Read the shot count back off the job: the server does not necessarily honour
  // what was requested, and the tolerance below depends on it.
  const shots = Number(job.shots) || 100
  results.push({ theta, shots, measured: hist['0'] ?? 0, expected: Math.cos(theta / 2) ** 2 })
  process.stdout.write('.')
}
process.stdout.write('\n\n')

// Results are sampled, so the tolerance is the binomial standard error, not a
// fixed epsilon: at 100 shots one sigma is ~0.035, and a constant threshold
// either rejects correct data or accepts a convention error.
console.log('theta      expected p(0)   IonQ p(0)   error      sigma   within')
let worstSigma = 0
for (const { theta, shots, measured, expected } of results) {
  const err = Math.abs(measured - expected)
  const sigma = Math.sqrt(Math.max(expected * (1 - expected), 1e-12) / shots)
  const nSigma = err / sigma
  worstSigma = Math.max(worstSigma, nSigma)
  console.log(`${theta.toFixed(4)}     ${expected.toFixed(6)}        ${measured.toFixed(6)}    ` +
              `${err.toExponential(1)}   ${sigma.toFixed(4)}  ${nSigma.toFixed(1)}σ`)
}
console.log(`\nlargest deviation ${worstSigma.toFixed(1)}σ (shots: ${results[0]?.shots})`)
console.log(worstSigma < 4
  ? 'Angle convention confirmed: rotation is in radians.'
  : 'MISMATCH — ket and IonQ disagree on the rotation scale.')

// A wrong scale does not produce a near miss, so print the alternatives. This is
// what makes a sweep conclusive where a single angle is not: at theta=0 every
// convention agrees, and only the shape of the curve separates them.
const TH = Math.PI / 4
console.log(`\nat theta=${TH.toFixed(4)}, p(0) under each reading of the rotation field:`)
console.log(`  radians (correct)  ${(Math.cos(TH / 2) ** 2).toFixed(6)}`)
console.log(`  pi-radians         ${(Math.cos((TH * Math.PI) / 2) ** 2).toFixed(6)}`)
console.log(`  turns              ${(Math.cos(TH * Math.PI) ** 2).toFixed(6)}`)
