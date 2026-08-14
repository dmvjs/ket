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

if (issues.length) {
  console.error(`Site version is stale:\n  - ${issues.join('\n  - ')}`)
  process.exit(1)
}
console.log(`site version matches package.json (v${version})`)
