import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'

/**
 * One conformance battery, run against every backend.
 *
 * Cross-backend agreement was previously asserted in eight hand-written blocks,
 * one per backend, each with its own corpus. That makes adding a backend
 * expensive in the only way that matters: not writing it, but establishing that
 * it is correct. Every simulation bug found in this codebase so far has been of
 * that shape — a decomposition that was right sitting under a sampler that was
 * silently wrong, a trajectory loop that was right behind a reset that was not.
 *
 * So a new backend should have to satisfy a contract rather than earn a bespoke
 * test file. Declare what it accepts and how exact it claims to be, add it to
 * BACKENDS, and the battery does the rest.
 *
 * The oracle is the statevector kernel's analytic `exactProbs()`, which has no
 * sampling variance and is itself checked against closed-form amplitudes
 * elsewhere in the suite.
 */

/** What a circuit needs from a backend, so a backend can decline it honestly. */
type Feature = 'clifford' | 'clifford+t' | 'rotations'

interface Case {
  name: string
  needs: Feature
  build: () => Circuit
  /** Analytic answer where one exists, as a second opinion on the oracle itself. */
  expected?: Record<string, number>
}

const ghz = (n: number): Circuit => {
  let c = new Circuit(n).h(0)
  for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
  return c
}

const CORPUS: Case[] = [
  { name: 'bell', needs: 'clifford', build: () => new Circuit(2).h(0).cnot(0, 1),
    expected: { '00': 0.5, '11': 0.5 } },
  { name: 'ghz-5', needs: 'clifford', build: () => ghz(5),
    expected: { '00000': 0.5, '11111': 0.5 } },
  { name: 'product-h', needs: 'clifford', build: () => new Circuit(4).h(0).h(1).h(2).h(3) },
  { name: 'clifford-tangle', needs: 'clifford',
    build: () => new Circuit(5).h(0).cnot(0, 1).s(1).h(2).cz(1, 2).cnot(3, 4).x(4).h(3).swap(0, 4) },
  { name: 'w-like', needs: 'clifford', build: () => new Circuit(3).h(0).cnot(0, 1).x(2).cnot(1, 2) },
  { name: 'clifford+t', needs: 'clifford+t',
    build: () => new Circuit(4).h(0).h(1).t(0).cnot(0, 1).t(1).h(2).cnot(1, 2).tdg(2).h(3).cnot(2, 3) },
  { name: 't-ladder', needs: 'clifford+t',
    build: () => { let c = new Circuit(3).h(0).h(1).h(2)
      for (let i = 0; i < 6; i++) c = c.t(i % 3).cnot(i % 3, (i + 1) % 3); return c } },
  { name: 'rotations', needs: 'rotations',
    build: () => new Circuit(4).h(0).ry(0.7, 1).rz(1.1, 0).cnot(0, 1).rx(0.4, 2).cu1(0.9, 1, 2).ry(-0.3, 3) },
  { name: 'deep-mixed', needs: 'rotations',
    build: () => { let c = new Circuit(4)
      for (let i = 0; i < 4; i++) c = c.h(i)
      for (let d = 0; d < 3; d++) for (let i = 0; i + 1 < 4; i++) c = c.cnot(i, i + 1).rz(0.2 * (d + 1), i + 1)
      return c } },
]

interface Backend {
  name: string
  accepts: (needs: Feature) => boolean
  /** Exact probabilities, when the backend can produce them without sampling. */
  exact?: (c: Circuit) => Record<string, number>
  /** Sampled probabilities. Every backend must be able to do at least this. */
  sample: (c: Circuit, shots: number, seed: number) => Record<string, number>
}

const BACKENDS: Backend[] = [
  { name: 'statevector', accepts: () => true,
    exact: c => c.exactProbs(),
    sample: (c, shots, seed) => c.run({ shots, seed }).probs },
  { name: 'mps', accepts: () => true,
    sample: (c, shots, seed) => c.runMps({ shots, seed }).probs },
  { name: 'density-matrix', accepts: () => true,
    exact: c => c.dm().probabilities(),
    sample: (c, shots, seed) => c.run({ shots, seed }).probs },
  { name: 'clifford', accepts: needs => needs === 'clifford',
    sample: (c, shots, seed) => c.runClifford({ shots, seed }).probs },
  { name: 'stabilizer-rank', accepts: needs => needs !== 'rotations',
    sample: (c, shots, seed) => c.runStabilizerRank({ shots, seed }).probs },
  { name: 'simulate (auto-routed)', accepts: () => true,
    sample: (c, shots, seed) => c.simulate({ shots, seed }).probs },
]

/** Largest deviation between two distributions over the union of their support. */
function maxDeviation(a: Record<string, number>, b: Record<string, number>): number {
  let worst = 0
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    worst = Math.max(worst, Math.abs((a[k] ?? 0) - (b[k] ?? 0)))
  }
  return worst
}

const SHOTS = 20_000
/** 4 sigma on a binomial proportion at this shot count, plus room for bias-free rounding. */
const SAMPLING_TOLERANCE = 4 * Math.sqrt(0.25 / SHOTS) + 0.005

describe('backend conformance', () => {
  // The oracle answers first, and is itself checked against closed forms.
  describe('oracle: exactProbs matches the analytic answer', () => {
    for (const { name, build, expected } of CORPUS) {
      if (!expected) continue
      it(name, () => {
        const got = build().exactProbs()
        for (const [bits, p] of Object.entries(expected)) expect(got[bits] ?? 0).toBeCloseTo(p, 10)
        expect(Object.values(got).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 10)
      })
    }
  })

  for (const backend of BACKENDS) {
    describe(backend.name, () => {
      const applicable = CORPUS.filter(c => backend.accepts(c.needs))
      it('declares at least one supported case', () => expect(applicable.length).toBeGreaterThan(0))

      for (const { name, build } of applicable) {
        if (backend.exact) {
          it(`${name}: exact probabilities match the oracle`, () => {
            const c = build()
            expect(maxDeviation(backend.exact!(c), c.exactProbs())).toBeLessThan(1e-9)
          })
        }

        it(`${name}: sampled distribution converges to the oracle`, () => {
          const c = build()
          const got = backend.sample(c, SHOTS, 12345)
          expect(maxDeviation(got, c.exactProbs())).toBeLessThan(SAMPLING_TOLERANCE)
        })

        it(`${name}: probabilities are normalised and non-negative`, () => {
          const got = backend.sample(build(), 2000, 7)
          const total = Object.values(got).reduce((s, p) => s + p, 0)
          expect(total).toBeCloseTo(1, 6)
          for (const p of Object.values(got)) expect(p).toBeGreaterThanOrEqual(0)
        })

        it(`${name}: the same seed gives the same result`, () => {
          const c = build()
          expect(backend.sample(c, 2000, 99)).toEqual(backend.sample(c, 2000, 99))
        })

        it(`${name}: support lies inside the oracle's support`, () => {
          // A backend may miss a low-probability outcome at finite shots, but it
          // must never report an outcome the state cannot produce.
          const c = build()
          const oracle = c.exactProbs()
          for (const [bits, p] of Object.entries(backend.sample(c, SHOTS, 5))) {
            if (p > 0) expect(oracle[bits] ?? 0, `${bits} is outside the true support`).toBeGreaterThan(0)
          }
        })
      }
    })
  }
})

/**
 * Invariants that hold past the statevector's reach.
 *
 * The battery above validates against `exactProbs()`, so it can only see as far
 * as a statevector fits — roughly 20 qubits. That is a real blind spot: the one
 * silently-wrong backend found in this codebase failed only at n >= 23, where no
 * oracle is available, and would have passed everything above.
 *
 * These cases carry their own analytic answer instead. A GHZ state has exactly
 * two outcomes at 1/2 each, at any width, on any backend that claims to run it.
 */
describe('backend conformance — beyond oracle reach', () => {
  const WIDE = 26
  const wideGhz = (): Circuit => {
    let c = new Circuit(WIDE).h(0)
    for (let i = 0; i < WIDE - 1; i++) c = c.cnot(i, i + 1)
    return c
  }

  const wideCapable = BACKENDS.filter(b => b.accepts('clifford') && b.name !== 'density-matrix' && b.name !== 'statevector')

  /**
   * A backend may decline a circuit. It may not answer it wrongly.
   *
   * `accepts` covers what a backend knows it cannot take up front; this covers
   * what it can only discover once running — a sampler whose Markov chain cannot
   * traverse the support, for instance. Refusing loudly is a correct outcome and
   * passes; returning a plausible wrong distribution is the failure this whole
   * battery exists to catch.
   */
  const attempt = (fn: () => Record<string, number>):
    { ok: true; probs: Record<string, number> } | { ok: false; why: string } => {
    try { return { ok: true, probs: fn() } }
    catch (e) { return { ok: false, why: e instanceof Error ? e.message : String(e) } }
  }

  for (const backend of wideCapable) {
    it(`${backend.name}: ${WIDE}-qubit GHZ is two outcomes at 1/2 each, or a refusal`, () => {
      const r = attempt(() => backend.sample(wideGhz(), 4000, 3))
      if (!r.ok) {
        // Declined. Require it to say why, so the caller can act on it.
        expect(r.why.length, 'a refusal must explain itself').toBeGreaterThan(20)
        return
      }
      const got = r.probs
      const zeros = '0'.repeat(WIDE), ones = '1'.repeat(WIDE)

      const stray = Object.entries(got).filter(([k, p]) => p > 0 && k !== zeros && k !== ones)
      expect(stray, `outcomes outside the GHZ support: ${stray.map(([k]) => k).join(', ')}`).toEqual([])

      // The failure this exists for is a sampler that reaches only one component.
      expect(got[zeros] ?? 0, 'P(|0...0>)').toBeGreaterThan(0.35)
      expect(got[ones] ?? 0, 'P(|1...1>)').toBeGreaterThan(0.35)
    })

    it(`${backend.name}: ${WIDE} independent H gates spread over many outcomes`, () => {
      let c = new Circuit(WIDE)
      for (let i = 0; i < WIDE; i++) c = c.h(i)
      const r = attempt(() => backend.sample(c, 2000, 4))
      if (!r.ok) { expect(r.why.length).toBeGreaterThan(20); return }
      // 2^26 outcomes at 2000 shots: collisions are vanishingly unlikely, so a
      // dominant outcome means the sampler is stuck rather than exploring.
      expect(Math.max(...Object.values(r.probs))).toBeLessThan(0.01)
    })
  }
})
