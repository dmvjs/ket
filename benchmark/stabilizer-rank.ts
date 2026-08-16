/**
 * Stabilizer-rank scaling — how far Clifford+T simulation reaches on this machine.
 *
 * Measures exactly one T-count per invocation. Loop in the shell:
 *
 *   for t in 50 55 60 65 70; do
 *     T=$t node --max-old-space-size=32768 benchmark/stabilizer-rank.ts
 *   done
 *
 * One point per process is not fussiness — measuring a sweep inside a single
 * process corrupts both numbers it reports:
 *
 *   - `process.resourceUsage().maxRSS` is a process-lifetime high-water mark, so
 *     later points inherit earlier peaks.
 *   - Timings inherit heap growth and GC pressure from earlier points. Measured:
 *     t=65 took 228 s in a fresh process and 532 s as the fourth point of a
 *     sweep — a 2.3× inflation that is pure measurement artefact.
 *
 * maxRSS is used rather than `process.memoryUsage().rss`, which reports post-GC
 * residual after the call returns and understates large runs. An interval sampler
 * cannot substitute: `runStabilizerRank` is synchronous and never yields.
 *
 * Not wired into .github/workflows/benchmark.yml — high-T points take minutes to
 * hours and would stall CI.
 */
import { Circuit, termBudget, maxTGates, bytesPerTerm } from '@kirkelliott/ket'

const T = Number(process.env.T ?? 50)
const QUBITS = Number(process.env.QUBITS ?? 100)
const DELTA = Number(process.env.DELTA ?? 0.3)

/**
 * H wall, `t` T gates, then a CNOT layer — entangled across the full width and
 * cheap to build. Term count comes from the T gates alone, so this costs exactly
 * what a GHZ chain of the same t would.
 *
 * Not a GHZ chain, deliberately. That support is {0…0, 1…1}, which single-bit
 * flips cannot cross, so the Metropolis sampler freezes on whichever end it seeds
 * and reports one bitstring for every shot. The decomposition timings this
 * benchmark exists to measure were unaffected, but the shots it drew alongside
 * them were degenerate, and the sampler now rejects that circuit outright.
 */
function circuit(n: number, t: number): Circuit {
  let c = new Circuit(n)
  for (let i = 0; i < n; i++) c = c.h(i)
  for (let i = 0; i < t; i++) c = c.t((i * 7) % n)
  for (let i = 0; i + 1 < n; i += 2) c = c.cnot(i, i + 1)
  return c
}

const phaseOps = Array.from({ length: T }, (_, i) => ({
  g: 'phase' as const, q: i % QUBITS, theta: Math.PI / 4,
}))
const budget = termBudget(phaseOps, DELTA)
const resident = (budget * bytesPerTerm(QUBITS)) / 1e9

const started = performance.now()
const d = circuit(QUBITS, T).runStabilizerRank({
  shots: 5, seed: 1, targetError: DELTA, burnIn: 4, thin: 1,
})
const secs = (performance.now() - started) / 1000
const maxRss = (process.resourceUsage().maxRSS * 1024) / 1e9

console.log(
  `n=${QUBITS} δ=${DELTA} t=${T}  ` +
  `budget=${budget.toLocaleString()} terms  ` +
  `${secs.toFixed(1)}s  maxRSS=${maxRss.toFixed(2)}GB  ` +
  `resident=${resident.toFixed(2)}GB  peakFactor=${(maxRss / resident).toFixed(2)}  ` +
  `approx=${d.truncated}`)
console.log(
  `  model: memory ceiling at 64 GB is t=` +
  `${maxTGates({ qubits: QUBITS, targetError: DELTA, memoryBytes: 64e9 })} ` +
  `(memory only — runtime binds first; see maxTGates docs)`)
