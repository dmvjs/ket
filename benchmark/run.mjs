/**
 * ket benchmark — outputs a JSON object with median times (ms) per scenario.
 * Run with: node benchmark/run.mjs
 * Used by .github/workflows/benchmark.yml to update the README.
 */

import { Circuit, qft } from '../dist/ket.js'

const RUNS = 7  // take median of N runs

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

// Statevector: random depth-4 circuits (statevector caps out ~20q; 24q takes minutes)
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

console.log(JSON.stringify(results, null, 2))
