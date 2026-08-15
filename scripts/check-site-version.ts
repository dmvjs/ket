/**
 * Fail if the site's version badge disagrees with package.json.
 *
 * Run:  node scripts/check-site-version.ts
 *
 * Wired into `deploy:site` so a stale badge cannot reach production. The site is
 * hand-authored, so nothing else couples it to the package version — docs.html
 * sat two minor versions behind until someone happened to read it.
 */
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

const root = new URL('..', import.meta.url).pathname
const { version } = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')) as { version: string }

const BADGE = /<span class="ver">v([\d.]+)<\/span>/g
const issues: string[] = []

for (const file of readdirSync(join(root, 'site')).filter(f => f.endsWith('.html'))) {
  const html = readFileSync(join(root, 'site', file), 'utf8')
  for (const [, found] of html.matchAll(BADGE))
    if (found !== version) issues.push(`site/${file}: badge says v${found}, package.json says v${version}`)
  // The subtitle carries the version inline rather than in a badge.
  for (const [, found] of html.matchAll(/· v([\d.]+)</g))
    if (found !== version) issues.push(`site/${file}: subtitle says v${found}, package.json says v${version}`)
}

// Test counts are quoted in several places across README and the site, and drift
// independently of the version. Rather than pin a number here (which would go
// stale in turn), require that every quoted count agrees with every other — a
// footer left at an old value then fails against the header beside it.
const COUNT = /([\d,]+) tests/g
const counts = new Map<string, string[]>()
for (const file of ['README.md', ...readdirSync(join(root, 'site')).filter(f => f.endsWith('.html')).map(f => `site/${f}`)]) {
  for (const [, n] of readFileSync(join(root, file), 'utf8').matchAll(COUNT)) {
    if (!counts.has(n)) counts.set(n, [])
    counts.get(n)!.push(file)
  }
}
if (counts.size > 1) {
  const seen = [...counts].map(([n, files]) => `${n} in ${[...new Set(files)].join(', ')}`)
  issues.push(`test counts disagree: ${seen.join('; ')}`)
}

// Backend counts drift the same way: the landing page claimed four for two
// releases after the fifth backend shipped, while its own subtitle said five.
// Require every count that describes the library to agree.
const WORDS: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6 }
const BACKENDS = /(one|two|three|four|five|six|\d+) (?:exact )?backends/gi
const backendCounts = new Map<number, string[]>()
for (const file of ['README.md', 'docs/REFERENCE.md',
                    ...readdirSync(join(root, 'site')).filter(f => f.endsWith('.html')).map(f => `site/${f}`)]) {
  for (const [, raw] of readFileSync(join(root, file), 'utf8').matchAll(BACKENDS)) {
    const n = WORDS[raw.toLowerCase()] ?? Number(raw)
    if (!backendCounts.has(n)) backendCounts.set(n, [])
    backendCounts.get(n)!.push(file)
  }
}
// The Grover demo on the landing page counts device profiles, not backends, so
// six is excluded rather than treated as a disagreement.
backendCounts.delete(6)
if (backendCounts.size > 1) {
  issues.push(`backend counts disagree: ${[...backendCounts]
    .map(([n, f]) => `${n} in ${[...new Set(f)].join(', ')}`).join('; ')}`)
}

if (issues.length) {
  console.error(`Site is stale:\n  - ${issues.join('\n  - ')}`)
  process.exit(1)
}
console.log(`site version v${version} and test count ${[...counts.keys()][0] ?? 'n/a'} consistent`)
