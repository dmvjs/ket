/**
 * Stabilizer-rank scaling — how far Clifford+T simulation reaches on this machine.
 *
 * Run with: node --max-old-space-size=32768 benchmark/stabilizer-rank.ts
 * Optionally: T_MIN=40 T_MAX=76 QUBITS=100 DELTA=0.3 node ... benchmark/stabilizer-rank.ts
 *
 * Not wired into .github/workflows/benchmark.yml — the high-T points take minutes
 * to hours and would stall CI. This is a machine-characterisation tool: it prints
 * measured time and peak RSS against the `maxTGates` model so the two can be
 * compared directly.
 *
 * Memory is read from `process.resourceUsage().maxRSS`, the OS high-water mark.
 * `process.memoryUsage().rss` sampled after the call returns is post-GC residual,
 * not peak, and understates a run that has already released its decomposition —
 * and an interval sampler cannot help because `runStabilizerRank` is synchronous
 * and never yields the event loop. maxRSS is process-lifetime and therefore
 * monotonic across the sweep, which is fine for an increasing T schedule.
 */
import { Circuit, termBudget, maxTGates, bytesPerTerm } from '@kirkelliott/ket'

const QUBITS = Number(process.env.QUBITS ?? 100)
const DELTA = Number(process.env.DELTA ?? 0.3)
const T_MIN = Number(process.env.T_MIN ?? 40)
const T_MAX = Number(process.env.T_MAX ?? 60)
const STEP = Number(process.env.STEP ?? 5)

/** GHZ chain plus `t` T gates — entangled across the full width, cheap to build. */
function circuit(n: number, t: number): Circuit {
  let c = new Circuit(n).h(0)
  for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
  for (let i = 0; i < t; i++) c = c.t((i * 7) % n)
  return c
}

const phaseOps = (t: number, n: number) =>
  Array.from({ length: t }, (_, i) => ({ g: 'phase' as const, q: i % n, theta: Math.PI / 4 }))

console.log(`n=${QUBITS}  targetError=${DELTA}  ${(bytesPerTerm(QUBITS) / 1024).toFixed(2)} KB/term`)
for (const gb of [16, 64]) {
  console.log(`  model ceiling at ${gb} GB: t=${maxTGates({ qubits: QUBITS, targetError: DELTA, memoryBytes: gb * 1e9 })}`)
}
console.log()

for (let t = T_MIN; t <= T_MAX; t += STEP) {
  const budget = termBudget(phaseOps(t, QUBITS), DELTA)
  const started = performance.now()
  try {
    const d = circuit(QUBITS, t).runStabilizerRank({
      shots: 5, seed: 1, targetError: DELTA, burnIn: 4, thin: 1,
    })
    const secs = (performance.now() - started) / 1000
    const maxRss = (process.resourceUsage().maxRSS * 1024) / 1e9
    console.log(
      `t=${String(t).padStart(2)}  budget=${budget.toLocaleString().padStart(10)} terms  ` +
      `${secs.toFixed(1).padStart(7)}s  maxRSS=${maxRss.toFixed(2)}GB  ` +
      `approx=${d.truncated}  outcomes=${Object.keys(d.probs).length}`)
  } catch (err) {
    const secs = (performance.now() - started) / 1000
    console.log(`t=${t}  budget=${budget.toLocaleString()}  FAILED after ${secs.toFixed(1)}s: ${(err as Error).message}`)
    break
  }
}
