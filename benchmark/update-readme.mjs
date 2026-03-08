/**
 * Reads benchmark JSON from stdin, builds a markdown table,
 * and splices it into README.md between the benchmark marker comments.
 *
 * Usage: node benchmark/run.mjs | node benchmark/update-readme.mjs
 */

import { readFileSync, writeFileSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const readmePath = resolve(root, 'README.md')

const chunks = []
process.stdin.on('data', d => chunks.push(d))
process.stdin.on('end', () => {
  const r = JSON.parse(Buffer.concat(chunks).toString())
  const fmt = ms => ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : ms < 1000 ? `${ms.toFixed(1)}ms` : `${(ms / 1000).toFixed(2)}s`

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

  const section = [
    '<!-- benchmark:start -->',
    '',
    `Measured on GitHub Actions \`ubuntu-latest\` (2-core, Node.js 22). Median of 7 runs.`,
    '',
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
    // Append before the Testing section
    updated = readme.replace('## Testing\n', `## Performance\n\n${section}\n\n## Testing\n`)
  }

  writeFileSync(readmePath, updated)
  console.log('README.md updated with benchmark results.')
})
