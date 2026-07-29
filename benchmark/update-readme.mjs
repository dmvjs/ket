/**
 * Reads benchmark JSON from stdin, writes SVG charts, builds a markdown table,
 * and splices everything into docs/REFERENCE.md between the benchmark markers.
 *
 * The markers must exist — a missing pair is a hard error rather than a silent
 * no-op, so a docs restructure can never quietly stop updating the numbers.
 *
 * Usage: node benchmark/run.mjs | node benchmark/update-readme.mjs
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { renderChart } from './chart.mjs'

const root      = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const docsPath  = resolve(root, 'docs', 'REFERENCE.md')
const chartsDir = resolve(root, 'benchmark', 'charts')

const START = '<!-- benchmark:start -->'
const END   = '<!-- benchmark:end -->'

const chunks = []
process.stdin.on('data', d => chunks.push(d))
process.stdin.on('end', () => {
  const r = JSON.parse(Buffer.concat(chunks).toString())
  const fmt = ms => ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : ms < 1000 ? `${ms.toFixed(1)}ms` : `${(ms / 1000).toFixed(2)}s`

  // Write SVG charts
  mkdirSync(chartsDir, { recursive: true })
  if (r.charts) {
    const chartDefs = [
      { key: 'bell',    title: 'Bell state — H(0) + CNOT(0, n−1)' },
      { key: 'uniform', title: 'Uniform superposition — H on all n qubits' },
      { key: 'qft',     title: 'Quantum Fourier Transform (QFT)' },
    ]
    for (const { key, title } of chartDefs) {
      if (r.charts[key]) {
        writeFileSync(resolve(chartsDir, `${key}.svg`), renderChart(title, r.charts[key]))
      }
    }
    console.log('SVG charts written to benchmark/charts/')
  }

  const table = [
    '| Circuit | Backend | Qubits | Time |',
    '|---|---|---|---|',
    `| Random depth-4 | Statevector | 8  | ${fmt(r.sv_random_8q)}  |`,
    `| Random depth-4 | Statevector | 12 | ${fmt(r.sv_random_12q)} |`,
    `| Random depth-4 | Statevector | 16 | ${fmt(r.sv_random_16q)} |`,
    `| Random depth-4 | Statevector | 20 | ${fmt(r.sv_random_20q)} |`,
    `| QFT            | Statevector | 8  | ${fmt(r.sv_qft_8q)}     |`,
    `| QFT            | Statevector | 12 | ${fmt(r.sv_qft_12q)}    |`,
    `| QFT            | Statevector | 16 | ${fmt(r.sv_qft_16q)}    |`,
    `| QFT            | Statevector | 20 | ${fmt(r.sv_qft_20q)}    |`,
    `| GHZ            | MPS χ=2     | 20 | ${fmt(r.mps_ghz_20q)}   |`,
    `| GHZ            | MPS χ=2     | 50 | ${fmt(r.mps_ghz_50q)}   |`,
    `| GHZ            | MPS χ=2     | 100| ${fmt(r.mps_ghz_100q)}  |`,
    `| Random depth-4 | MPS χ=8     | 20 | ${fmt(r.mps_random_20q_chi8)} |`,
    `| Random depth-4 | MPS χ=8     | 30 | ${fmt(r.mps_random_30q_chi8)} |`,
    `| Random depth-4 | MPS χ=8     | 50 | ${fmt(r.mps_random_50q_chi8)} |`,
  ].join('\n')

  // Paths are relative to docs/REFERENCE.md, hence the ../ prefix.
  const chartsBlock = r.charts ? [
    '![Bell state benchmark](../benchmark/charts/bell.svg)',
    '![Uniform superposition benchmark](../benchmark/charts/uniform.svg)',
    '![QFT benchmark](../benchmark/charts/qft.svg)',
    '',
  ].join('\n') : ''

  const section = [
    START,
    '',
    'Measured on GitHub Actions `ubuntu-latest` (2-core, Node.js 22). Median of 5 runs.',
    'Regenerated on every push to main — edits between these markers are overwritten.',
    '',
    'Statevector is exact but O(2ⁿ) — time and memory grow with the number of non-zero amplitudes, not just qubit count. Sparse circuits like Bell maintain two amplitudes at any width and run in near-constant time. Dense circuits (uniform superposition, QFT) fill all 2ⁿ entries and hit the exponential wall around 20 qubits. The MPS backend removes that ceiling for circuits with bounded entanglement.',
    '',
    chartsBlock,
    table,
    '',
    END,
  ].join('\n')

  const doc   = readFileSync(docsPath, 'utf8')
  const start = doc.indexOf(START)
  const end   = doc.indexOf(END)

  if (start === -1 || end <= start) {
    console.error(`error: could not find ${START} … ${END} in docs/REFERENCE.md`)
    process.exit(1)
  }

  writeFileSync(docsPath, doc.slice(0, start) + section + doc.slice(end + END.length))
  console.log('docs/REFERENCE.md updated with benchmark results.')
})
