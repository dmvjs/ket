/**
 * Measure peak MPS bond dimension (χ) for the full Beauregard QPE circuit.
 * Run with: npx tsx scripts/measure-chi.ts
 *
 * Known results (exact, no χ cap):
 *   n=4 (N=15,  a=7):  χ =   4   (~0.4s)
 *   n=5 (N=21,  a=2):  χ =  27   (~2.7s)
 *   n=6 (N=35,  a=3):  χ =  44   (~29s)
 *   n=7 (N=77,  a=2):  χ = 143   (~907s)
 *
 * The χ growth is super-linear in n.  Controlled-U_a gates in QPE generate
 * intermediate entanglement that the modular multiplier does not fully
 * disentangle across shots, causing χ to grow with both n and the period r.
 * Exact MPS simulation is not efficient for large N.
 */
import { Circuit } from '../src/circuit.js'
import { beauregardU, modInverse, applyIqft } from '../src/beauregard.js'

function measurePeakChi(N: bigint, a: bigint, maxBond = 512): { chi: number; ms: number; capped: boolean } {
  const n         = Math.ceil(Math.log2(Number(N)))
  const precision = 2 * n + 1
  const xOff      = precision
  const bOff      = precision + n
  const anc       = precision + 2 * n + 1
  const totalQ    = anc + 1

  let c = new Circuit(totalQ)
  for (let k = 0; k < precision; k++) c = c.h(k)
  c = c.x(xOff)

  let ak = a % N
  for (let k = 0; k < precision; k++) {
    const akInv = modInverse(ak, N)
    c = beauregardU(c, n, ak, akInv, N, k, xOff, bOff, anc)
    ak = ak * ak % N
  }
  c = applyIqft(c, precision, 0)

  const t0     = Date.now()
  const dist   = c.runMps({ shots: 1, seed: 1, maxBond })
  const capped = dist.peakChi! >= maxBond
  return { chi: dist.peakChi!, ms: Date.now() - t0, capped }
}

const cases: Array<[bigint, bigint, string]> = [
  [15n,  7n, '4-bit'],
  [21n,  2n, '5-bit'],
  [35n,  3n, '6-bit'],
  [77n,  2n, '7-bit'],
  [143n, 2n, '8-bit'],
  [221n, 2n, '8-bit'],
]

for (const [N, a, label] of cases) {
  process.stdout.write(`N=${N} (${label}, a=${a}): computing...`)
  try {
    const { chi, ms, capped } = measurePeakChi(N, a)
    console.log(` χ=${chi}${capped ? ' (HIT CAP — approx)' : ''}  (${(ms / 1000).toFixed(1)}s)`)
  } catch (e) {
    console.log(` ERROR: ${e}`)
  }
}
