/**
 * Reads benchmark JSON from stdin, writes SVG charts, builds a markdown table,
 * and splices everything into README.md between the benchmark marker comments.
 *
 * Usage: node benchmark/run.mjs | node benchmark/update-readme.mjs
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { renderChart } from './chart.mjs'

const root      = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const readmePath = resolve(root, 'README.md')
const chartsDir  = resolve(root, 'benchmark', 'charts')

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

  const chartsBlock = r.charts ? [
    '![Bell state benchmark](benchmark/charts/bell.svg)',
    '![Uniform superposition benchmark](benchmark/charts/uniform.svg)',
    '![QFT benchmark](benchmark/charts/qft.svg)',
    '',
  ].join('\n') : ''

  const section = [
    '<!-- benchmark:start -->',
    '',
    'Measured on GitHub Actions `ubuntu-latest` (2-core, Node.js 22). Median of 5 runs.',
    '',
    'Statevector is exact but O(2ⁿ) — time and memory grow with the number of non-zero amplitudes, not just qubit count. Sparse circuits like Bell maintain two amplitudes at any width and run in near-constant time. Dense circuits (uniform superposition, QFT) fill all 2ⁿ entries and hit the exponential wall around 20 qubits. The MPS backend removes that ceiling for circuits with bounded entanglement.',
    '',
    chartsBlock,
    table,
    '',
    '<!-- benchmark:end -->',
  ].join('\n')

  const readme = readFileSync(readmePath, 'utf8')
  const start = readme.indexOf('<!-- benchmark:start -->')
  const end   = readme.indexOf('<!-- benchmark:end -->') + '<!-- benchmark:end -->'.length

  let updated
  if (start !== -1 && end > start) {
    updated = readme.slice(0, start) + section + readme.slice(end)
  } else {
    updated = readme.replace('## Testing\n', `## Performance\n\n${section}\n\n## Testing\n`)
  }

  writeFileSync(readmePath, updated)
  console.log('README.md updated with benchmark results.')
})
