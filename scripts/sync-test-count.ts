/**
 * Rewrite every quoted test count to what the suite actually has.
 *
 * Run:  npm run sync:counts
 *
 * The count appears in README, the reference and the site (three places,
 * deliberately few). check-site-version.ts fails deploy if they disagree; this
 * is the one command that makes them agree.
 */
import { execSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const root = new URL('..', import.meta.url).pathname
const out = execSync('npx vitest run --reporter=json', { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], maxBuffer: 64 * 1024 * 1024 })
const report = JSON.parse(out.slice(out.indexOf('{'))) as { numTotalTests: number; numFailedTests: number }
if (report.numFailedTests > 0) { console.error(`${report.numFailedTests} tests failing; not updating counts`); process.exit(1) }
const n = report.numTotalTests.toLocaleString('en-US')

const FILES = ['README.md', 'docs/REFERENCE.md', 'site/index.html', 'site/docs.html']
for (const f of FILES) {
  const path = join(root, f)
  const before = readFileSync(path, 'utf8')
  const after = before.replace(/\b[\d,]+ tests\b/g, `${n} tests`)
  if (after !== before) { writeFileSync(path, after); console.log(`${f}: → ${n} tests`) }
}
console.log(`${n} tests`)
