/**
 * Validate every checkable factual claim in the docs against the built library.
 *
 * Run:  npm run build && node scripts/check-docs.ts
 *
 * Wired into `deploy:site` alongside check-site-version.ts. Written after seven
 * rounds of spot-checking each found something the previous round had declared
 * clean: version badges two releases stale, a whole release's API missing from
 * the site, a footer quoting a test count from 1,301 tests ago, a payload sample
 * showing a serialization bug the library had already fixed, retired hardware
 * presented as current. Every one of those was mechanically checkable.
 *
 * The rule: if a doc states something the library can be asked about, ask it.
 */
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { Circuit, DEVICES, IONQ_DEVICES, shorCircuit, shorBeauregard } from '../dist/ket.js'

const root = new URL('..', import.meta.url).pathname
const read = (f: string) => readFileSync(join(root, f), 'utf8')
const SURFACES = ['README.md', 'docs/REFERENCE.md',
  ...readdirSync(join(root, 'site')).filter(f => f.endsWith('.html')).map(f => `site/${f}`)]

const issues: string[] = []
const fail = (f: string, msg: string) => issues.push(`${f}: ${msg}`)

for (const f of SURFACES) {
  const t = read(f)

  // 1. No retired device may be presented as available.
  for (const [name, d] of Object.entries(DEVICES)) {
    if (d.status === 'available') continue
    // A retired name is fine in prose that says so; flag it only in table rows.
    if (new RegExp(`<b>${name}</b>`).test(t)) fail(f, `table row for retired device '${name}'`)
  }

  // 2. Any IonQ rotation value in a sample must match what toIonQ emits.
  //    A stale sample taught users the exact bug 0.8.0 fixed.
  for (const [, v] of t.matchAll(/"gate":\s*"r[xyz]"[^}]*?"rotation":\s*(-?[\d.]+)/g)) {
    const val = Math.abs(Number(v))
    if (val > 0 && val < 1.001 && Math.abs(val - 0.5) < 1e-9)
      fail(f, `rotation ${v} looks like the pre-0.8.0 pi-radian convention (expected radians)`)
  }

  // 3. Native gate lists must match IONQ_DEVICES.
  for (const [name, d] of Object.entries(IONQ_DEVICES)) {
    const row = t.match(new RegExp(`<b>${name}</b>[\\s\\S]{0,400}?</tr>`))
    if (!row) continue
    for (const g of d.nativeGates)
      if (!row[0].includes(g)) fail(f, `${name} row omits native gate '${g}'`)
  }
}

// 4. Quoted gate counts for the documented circuits must be reproducible.
const claims: [string, number][] = [
  ['5,798', shorCircuit(15n, 7n, 3).toIonQBasis().toIonQ().circuit.length],
  ['17,447', shorCircuit(15n, 7n, 9).toIonQBasis().toIonQ().circuit.length],
  ['31,260', shorCircuit(33n, 5n).gateCounts().twoQubit],
]
for (const f of SURFACES) {
  const t = read(f)
  for (const [quoted, actual] of claims) {
    if (!t.includes(quoted)) continue
    if (Number(quoted.replace(/,/g, '')) !== actual)
      fail(f, `quotes ${quoted} gates but the library produces ${actual.toLocaleString()}`)
  }
}

// 5. Shor's headline result must still hold.
const r = shorBeauregard(15n, { a: 7n })
if (r.method !== 'quantum' || r.qubits !== 19 || String(r.period) !== '4')
  issues.push(`shorBeauregard(15n,{a:7n}) changed: ${JSON.stringify({ m: r.method, q: r.qubits, p: String(r.period) })}`)

// 6. Every public export should appear somewhere a reader can find it.
const api = Object.keys(await import('../dist/ket.js'))
const ref = read('docs/REFERENCE.md'), site = read('site/docs.html')
const undocumented = api.filter(n => !ref.includes(n) && !site.includes(n))
if (undocumented.length) issues.push(`exports documented nowhere: ${undocumented.join(' ')}`)

if (issues.length) {
  console.error(`Docs disagree with the library:\n  - ${issues.join('\n  - ')}`)
  process.exit(1)
}
console.log(`docs verified against the library (${SURFACES.length} surfaces, ${api.length} exports)`)
