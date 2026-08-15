import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import * as source from './index.js'

/**
 * Tests for the built artifacts rather than the source.
 *
 * The rest of the suite imports `src/*.ts` directly, which cannot see anything
 * that only goes wrong once the code is bundled: a format that will not build, a
 * global that is never defined, an entry point missing from the exports map, two
 * bundles each carrying their own copy of `Circuit`. Every bug of that shape in
 * this package so far has been invisible to the source suite.
 *
 * The full suite cannot simply be re-run against the bundle — test files import
 * internal modules (`prng.js`, `exp-sum.js`, `dense.js`, …) that the public
 * bundle deliberately does not export. What is checked here is the shipped
 * surface: what a consumer can actually reach, in each format it ships in.
 *
 * Skipped wholesale when `dist/` is absent, since `npm test` does not build. But
 * once a build exists, every expected artifact must — a half-built dist fails
 * rather than quietly skipping.
 */
const dist = (name: string): string => fileURLToPath(new URL(`../dist/${name}`, import.meta.url))
const built = existsSync(dist('ket.js'))

describe.runIf(built)('built artifacts', () => {
  const ARTIFACTS = ['ket.js', 'ket.min.js', 'ket.global.js', 'compat.js', 'index.d.ts', 'compat.d.ts']

  it('ships every artifact the exports map and CDN fields point at', () => {
    for (const name of ARTIFACTS) {
      expect(existsSync(dist(name)), `${name} missing from dist/`).toBe(true)
    }
  })

  it('declares those artifacts in package.json', () => {
    const pkg = JSON.parse(readFileSync(fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8'))
    expect(pkg.exports['.'].import).toBe('./dist/ket.js')
    expect(pkg.exports['./compat'].import).toBe('./dist/compat.js')
    expect(pkg.unpkg).toBe('./dist/ket.global.js')
    expect(pkg.jsdelivr).toBe('./dist/ket.global.js')
  })

  describe('ESM bundle', () => {
    it('exports exactly the public surface of src/index.ts', async () => {
      const bundle = await import(dist('ket.js'))
      expect(Object.keys(bundle).sort()).toEqual(Object.keys(source).sort())
    })

    it('computes the same result as the source build', async () => {
      const { Circuit } = await import(dist('ket.js'))
      const bundled = new Circuit(2).h(0).cnot(0, 1).exactProbs()
      const direct  = new source.Circuit(2).h(0).cnot(0, 1).exactProbs()
      expect(bundled).toEqual(direct)
    })
  })

  describe('global (IIFE) bundle', () => {
    // Loaded the way a <script src> tag would: evaluated as a classic script,
    // assigning one global. A top-level await anywhere in the graph makes this
    // format impossible to build, so this is also the guard against reintroducing one.
    const evalGlobal = (): Record<string, unknown> => {
      const src = readFileSync(dist('ket.global.js'), 'utf8')
      return new Function(`${src}\n;return ket`)() as Record<string, unknown>
    }

    it('evaluates as a classic script and defines one global', () => {
      expect(() => evalGlobal()).not.toThrow()
      expect(typeof evalGlobal()['Circuit']).toBe('function')
    })

    it('exposes the same names as the ESM bundle', () => {
      expect(Object.keys(evalGlobal()).sort()).toEqual(Object.keys(source).sort())
    })

    it('runs a circuit, including the MPS path that reads import.meta', () => {
      const ket = evalGlobal() as unknown as typeof source
      const bell = new ket.Circuit(2).h(0).cnot(0, 1)
      expect(bell.exactProbs()).toEqual(new source.Circuit(2).h(0).cnot(0, 1).exactProbs())
      // `import.meta` is empty in this format; an unguarded read would throw here.
      expect(bell.runMps({ shots: 64, seed: 1, noise: { p1: 0.001, p2: 0.002 } }).shots).toBe(64)
    })

    it('never reaches for a node: builtin at module scope', () => {
      // A browser rejects `node:` specifiers at the network layer and logs an
      // error even when the failure is caught, so the bundle must not ask.
      const src = readFileSync(dist('ket.global.js'), 'utf8')
      expect(src).not.toMatch(/\bimport\s*\(\s*["']node:/)
      expect(src).not.toMatch(/\bfrom\s*["']node:/)
    })
  })

  describe('compat entry point', () => {
    it('re-exports rather than bundling a second copy of the engine', async () => {
      // Two bundles would each carry their own Circuit, and a Circuit from one
      // fails `instanceof` against the other.
      const shim = readFileSync(dist('compat.js'), 'utf8')
      expect(shim).toContain("from './ket.js'")
      expect(shim.length).toBeLessThan(200)

      const { QuantumCircuit } = await import(dist('compat.js'))
      const { Circuit }        = await import(dist('ket.js'))
      const qc = new QuantumCircuit(2)
      qc.addGate('h', 0, 0)
      expect(qc.toKet()).toBeInstanceOf(Circuit)
    })

    it('offers the class as both a named and a default export', async () => {
      const mod = await import(dist('compat.js'))
      expect(typeof mod.QuantumCircuit).toBe('function')
      expect(mod.default).toBe(mod.QuantumCircuit)
    })
  })
})
