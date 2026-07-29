/**
 * Build script — replaces the inline `build` shell command in package.json.
 *
 * Build order:
 *   1. Worker standalone  → dist/mps.worker.js   (used when running from dist/ directly)
 *   2. Worker minified    → captured in memory, URL-encoded → __MPS_WORKER_DATA_URL__
 *   3. Main bundles       → dist/ket.js + dist/ket.min.js
 *                           __MPS_WORKER_DATA_URL__ is --define'd into both so the worker
 *                           code is self-contained even after downstream re-bundling
 *   4. TypeScript declarations → dist/*.d.ts via tsc --emitDeclarationOnly
 *
 * Why the data URL matters:
 *   new URL('./mps.worker.js', import.meta.url) resolves relative to the *current* module.
 *   When a user bundles ket into their own app, that module lives in their output directory
 *   and mps.worker.js is absent. Embedding the worker as a data: URL removes the file
 *   dependency entirely — the worker code travels inside the bundle as a string constant.
 */

import { build }    from 'esbuild'
import { execSync } from 'child_process'

// ── 1. Worker standalone ──────────────────────────────────────────────────────

await build({
  entryPoints: ['src/mps.worker.ts'],
  bundle:      true,
  format:      'esm',
  platform:    'node',
  outfile:     'dist/mps.worker.js',
})

// ── 2. Worker → data URL ──────────────────────────────────────────────────────

const { outputFiles: [workerFile] } = await build({
  entryPoints: ['src/mps.worker.ts'],
  bundle:      true,
  format:      'esm',
  platform:    'node',
  minify:      true,
  write:       false,
})

// data: URL works in Node.js worker_threads (≥12.17) and is safe to stringify
// into downstream bundles — a plain string constant with no file-system dependency.
const workerDataUrl = 'data:text/javascript,' + encodeURIComponent(workerFile!.text)

// ── 3. Main bundles ───────────────────────────────────────────────────────────

const shared = {
  entryPoints: ['src/index.ts'],
  bundle:      true,
  format:      'esm',
  external:    ['node:worker_threads'],
  define:      { __MPS_WORKER_DATA_URL__: JSON.stringify(workerDataUrl) },
} as const

await build({ ...shared,               outfile: 'dist/ket.js'     })
await build({ ...shared, minify: true, outfile: 'dist/ket.min.js' })

// ── 4. TypeScript declarations ────────────────────────────────────────────────

execSync('tsc --emitDeclarationOnly', { stdio: 'inherit' })
