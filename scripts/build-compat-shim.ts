/**
 * Write `dist/compat.js`, the entry point behind `@kirkelliott/ket/compat`.
 *
 * It is a re-export of the main bundle rather than a bundle of its own: two
 * bundles would each carry a copy of `Circuit`, and a `Circuit` produced by one
 * copy fails `instanceof` against the other and cannot reach its private fields.
 * `QuantumCircuit` is exported from `src/index.ts` so this alias can be thin.
 */

import { writeFileSync } from 'node:fs'

const shim = `export { QuantumCircuit, QuantumCircuit as default } from './ket.js'\n`

writeFileSync(new URL('../dist/compat.js', import.meta.url), shim)
console.log('wrote dist/compat.js (re-export shim)')
