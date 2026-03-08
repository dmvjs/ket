/**
 * ket benchmark — outputs a JSON object with median times (ms) per scenario.
 * Run with: node benchmark/run.mjs
 * Used by .github/workflows/benchmark.yml to update the README.
 */

import { Circuit, qft } from '../dist/ket.js'

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
