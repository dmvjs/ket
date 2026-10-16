/**
 * Point index.html at the exact bundle it was built against.
 *
 * The homepage runs the library, so a browser holding a stale `ket.global.js`
 * shows old numbers with nothing to indicate anything is wrong — which is how a
 * corrected figure kept reappearing as the old one. A content hash in the query
 * string means a rebuilt bundle is a different URL and cannot be served from
 * cache.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'

const bundle = readFileSync('site/ket.global.js')
const hash = createHash('md5').update(bundle).digest('hex').slice(0, 8)
const page = 'site/index.html'
const before = readFileSync(page, 'utf8')
const after = before.replace(
  /<script src="ket\.global\.js(\?v=[a-f0-9]+)?"><\/script>/,
  `<script src="ket.global.js?v=${hash}"></script>`,
)
if (before === after && !after.includes(`?v=${hash}`)) {
  throw new Error('stamp-bundle: could not find the ket.global.js script tag in index.html')
}
writeFileSync(page, after)
console.log(`stamped site/index.html with bundle hash ${hash}`)
