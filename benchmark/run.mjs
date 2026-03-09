/**
 * ket benchmark — outputs a JSON object with median times (ms) per scenario.
 * Run with: node benchmark/run.mjs
 * Used by .github/workflows/benchmark.yml to update the README.
 */

import { Circuit, qft, DEVICES } from '../dist/ket.js'

const RUNS = 5  // take median of N runs

function median(arr) {
  const s = [...arr].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]
}

function bench(fn) {
  fn() // warmup
  const times = []
  for (let i = 0; i < RUNS; i++) {
    const t = performance.now()
    fn()
    times.push(performance.now() - t)
  }
  return median(times)
}

/** Single timed run — used for chart data after JIT is warm from bench() above. */
function point(fn) {
  const t = performance.now()
  const out = fn()
  const ms = performance.now() - t
  // Estimate memory from statevector entry count: ~96 bytes per Map entry (BigInt key + Complex value + overhead)
  const memMB = out instanceof Map ? out.size * 96 / (1024 * 1024) : 0
  return { ms, memMB }
}

function randomCircuit(n, depth) {
  let c = new Circuit(n)
  for (let d = 0; d < depth; d++) {
    for (let q = 0; q < n; q++) c = c.h(q)
    for (let q = 0; q < n - 1; q += 2) c = c.cnot(q, q + 1)
  }
  return c
}

function ghz(n) {
  let c = new Circuit(n).h(0)
  for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
  return c
}

const results = {}

// Statevector: random depth-4 circuits (caps out ~20q; beyond that takes minutes on CI)
for (const n of [8, 12, 16, 20]) {
  const c = randomCircuit(n, 4)
  results[`sv_random_${n}q`] = bench(() => c.statevector())
}

// QFT
for (const n of [8, 12, 16, 20]) {
  const c = qft(n)
  results[`sv_qft_${n}q`] = bench(() => c.statevector())
}

// MPS: GHZ (low entanglement, χ=2 is exact)
for (const n of [20, 50, 100]) {
  const c = ghz(n)
  results[`mps_ghz_${n}q`] = bench(() => c.runMps({ shots: 1000, maxBond: 2 }))
}

// MPS: random circuits with χ=8
for (const n of [20, 30, 50]) {
  const c = randomCircuit(n, 4)
  results[`mps_random_${n}q_chi8`] = bench(() => c.runMps({ shots: 1000, maxBond: 8 }))
}

// compose(): measure pure concatenation overhead vs building equivalent circuit directly
{
  const half = randomCircuit(16, 2)
  const full = randomCircuit(16, 4)
  results['compose_16q_d4']       = bench(() => half.compose(half))
  results['compose_vs_direct_16q'] = bench(() => full.statevector())
  // compose() cost should be negligible — just array spread; statevector() dominates
  const composed = half.compose(half)
  results['compose_sv_16q'] = bench(() => composed.statevector())
}

// bind(): parametric ansatz substitution overhead
{
  // 10-parameter ansatz — realistic for VQE
  let ansatz = new Circuit(5)
  const names = ['t0','t1','t2','t3','t4','t5','t6','t7','t8','t9']
  for (let i = 0; i < 5; i++) ansatz = ansatz.ry(names[i*2], i).rz(names[i*2+1], i)
  for (let i = 0; i < 4; i++) ansatz = ansatz.cnot(i, i + 1)
  const vals = Object.fromEntries(names.map((n, i) => [n, i * 0.1]))
  results['bind_10param_5q']    = bench(() => ansatz.bind(vals))
  const bound = ansatz.bind(vals)
  results['bind_then_sv_5q']    = bench(() => ansatz.bind(vals).statevector())
  results['bind_baseline_sv_5q'] = bench(() => bound.statevector())
  // bind + sv overhead = bind_then_sv - bind_baseline_sv; should be < 0.1ms
}

// stateAsArray(): overhead vs raw statevector()
{
  const c = randomCircuit(12, 4)
  results['stateAsArray_12q']   = bench(() => c.stateAsArray())
  results['statevector_12q']    = bench(() => c.statevector())
  // stateAsArray overhead = one extra pass over amplitudes; should be < 5% of total
}

// Noise simulation with device profiles
{
  const bell = new Circuit(2).h(0).cnot(0, 1).creg('c', 2).measure(0, 'c', 0).measure(1, 'c', 1)
  results['noise_ibm_sherbrooke_1k'] = bench(() => bell.run({ shots: 1000, noise: 'ibm_sherbrooke' }))
  results['noise_h1_1_1k']           = bench(() => bell.run({ shots: 1000, noise: 'h1-1' }))
  results['noise_forte1_1k']         = bench(() => bell.run({ shots: 1000, noise: 'forte-1' }))
  results['noise_clean_1k']          = bench(() => bell.run({ shots: 1000 }))
}

// Chart data: per-qubit time + memory for n=2..20 (JIT is warm from above runs)
results.charts = {}

// Bell state: H(0) + CNOT(0, n-1) — only 2 amplitudes regardless of n (sparse)
results.charts.bell = []
for (let n = 2; n <= 20; n++) {
  const c = new Circuit(n).h(0).cnot(0, n - 1)
  results.charts.bell.push({ n, ...point(() => c.statevector()) })
}

// Uniform superposition: H on all qubits — all 2^n amplitudes (dense)
results.charts.uniform = []
for (let n = 2; n <= 20; n++) {
  let c = new Circuit(n)
  for (let q = 0; q < n; q++) c = c.h(q)
  results.charts.uniform.push({ n, ...point(() => c.statevector()) })
}

// QFT: all 2^n amplitudes with quadratic gate depth
results.charts.qft = []
for (let n = 2; n <= 20; n++) {
  const c = qft(n)
  results.charts.qft.push({ n, ...point(() => c.statevector()) })
}

console.log(JSON.stringify(results, null, 2))
