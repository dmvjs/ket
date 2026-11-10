import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { sweep } from './experiment.js'
import {
  bootstrapOutcomes, compareExperiments, compareOutcomes, hellingerFidelity, klDivergence,
  linearXeb, probInterval, shotNoiseTvd, tvd, wilson, zExpectation,
} from './stats.js'

const bell = new Circuit(2).h(0).cnot(0, 1)
const BELL = bell.exactProbs()
const UNIFORM2 = { '00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25 }

describe('distances', () => {
  it('tvd: 0 for identical, 1 for disjoint, symmetric, ½ between Bell and uniform', () => {
    expect(tvd(BELL, BELL)).toBe(0)
    expect(tvd({ '0': 1 }, { '1': 1 })).toBe(1)
    expect(tvd(BELL, UNIFORM2)).toBeCloseTo(0.5, 12)
    expect(tvd(UNIFORM2, BELL)).toBeCloseTo(0.5, 12)
  })

  it('hellingerFidelity: 1 identical, 0 disjoint, ½ between Bell and uniform', () => {
    expect(hellingerFidelity(BELL, BELL)).toBeCloseTo(1, 12)
    expect(hellingerFidelity({ '0': 1 }, { '1': 1 })).toBe(0)
    expect(hellingerFidelity(BELL, UNIFORM2)).toBeCloseTo(0.5, 12)
  })

  it('klDivergence: 0 identical, ln 2 Bell‖uniform, ∞ where the reference has no support', () => {
    expect(klDivergence(BELL, BELL)).toBe(0)
    expect(klDivergence(BELL, UNIFORM2)).toBeCloseTo(Math.LN2, 12)
    expect(klDivergence(UNIFORM2, BELL)).toBe(Infinity)
  })

  it('linearXeb: 1 on the ideal distribution itself for a bit-flipped state, 0 against uniform', () => {
    const peaked = { '101': 1 }
    expect(linearXeb(peaked, peaked)).toBeCloseTo(7, 12)      // 2³·1 − 1: a delta is maximally non-uniform
    expect(linearXeb(BELL, UNIFORM2)).toBeCloseTo(0, 12)
    expect(linearXeb(BELL, BELL)).toBeCloseTo(1, 12)          // 4·(½·½ + ½·½) − 1
    const rnd = new Circuit(3).h(0).ry(0.7, 1).cnot(0, 1).rx(1.1, 2).cz(1, 2).h(2).exactProbs()
    const self = linearXeb(rnd, rnd)
    expect(self).toBeGreaterThan(0)
    expect(linearXeb(rnd, rnd)).toBeCloseTo(8 * Object.values(rnd).reduce((s, p) => s + p * p, 0) - 1, 12)
    expect(linearXeb(rnd, { '000': 0.125, '001': 0.125, '010': 0.125, '011': 0.125, '100': 0.125, '101': 0.125, '110': 0.125, '111': 0.125 })).toBeCloseTo(0, 12)
  })

  it('accepts Distributions and raw counts, normalising', () => {
    const d = bell.run({ shots: 1000, seed: 1 })
    expect(tvd(d, d.probs)).toBe(0)
    expect(tvd({ '00': 500, '11': 500 }, BELL)).toBe(0)
    expect(hellingerFidelity({ '00': 3, '11': 1 }, { '00': 0.75, '11': 0.25 })).toBeCloseTo(1, 12)
  })

  it('rejects mixed widths, non-bitstrings, negative weights, and empty maps', () => {
    expect(() => tvd({ '0': 1 }, { '00': 1 })).toThrow(/widths differ/)
    expect(() => tvd({ '0': 0.5, '01': 0.5 }, { '0': 1 })).toThrow(/mixed width/)
    expect(() => tvd({ 'x': 1 }, { '0': 1 })).toThrow(/not a bitstring/)
    expect(() => tvd({ '0': -1 }, { '0': 1 })).toThrow(/weight/)
    expect(() => tvd({}, { '0': 1 })).toThrow(/empty/)
  })
})

describe('zExpectation', () => {
  it('reads parities', () => {
    expect(zExpectation(BELL)).toBeCloseTo(1, 12)                 // ⟨ZZ⟩
    expect(zExpectation(BELL, [0])).toBeCloseTo(0, 12)            // ⟨ZI⟩
    expect(zExpectation(new Circuit(1).h(0).exactProbs())).toBeCloseTo(0, 12)
    expect(zExpectation(new Circuit(1).x(0).exactProbs())).toBe(-1)
    expect(zExpectation({ '011': 1 }, [1, 2])).toBe(1)
    expect(zExpectation({ '011': 1 }, [0, 1])).toBe(-1)
    expect(() => zExpectation(BELL, [2])).toThrow(RangeError)
  })
})

describe('intervals', () => {
  it('wilson behaves at the edges and matches the normal approximation in the middle', () => {
    const edge = wilson(100, 100)
    expect(edge.high).toBe(1)
    expect(edge.low).toBeGreaterThan(0.95)
    expect(edge.low).toBeLessThan(1)
    expect(wilson(0, 50).low).toBe(0)
    expect(wilson(0, 50).high).toBeGreaterThan(0)
    const mid = wilson(500, 1000)
    expect((mid.high - mid.low) / 2).toBeCloseTo(1.96 * Math.sqrt(0.25 / 1000), 3)
    expect(() => wilson(5, 0)).toThrow(RangeError)
    expect(() => wilson(11, 10)).toThrow(RangeError)
  })

  it('probInterval takes shots from a Distribution or the options', () => {
    const d = bell.run({ shots: 400, seed: 3 })
    const iv = probInterval(d, '00')
    expect(iv.low).toBeLessThan(0.5)
    expect(iv.high).toBeGreaterThan(0.5)
    expect(probInterval(d.probs, '00', { shots: 400 })).toEqual(iv)
    expect(probInterval(d, '01').low).toBe(0)
    expect(() => probInterval(d.probs, '00')).toThrow(/shots/)
  })
})

describe('bootstrap', () => {
  it('estimates the standard error of a 50/50 probability', () => {
    const d = bell.run({ shots: 1000, seed: 4 })
    const b = bootstrapOutcomes(d, s => s['00'] ?? 0, { seed: 1 })
    const want = Math.sqrt(0.25 / 1000)
    expect(Math.abs(b.stderr - want) / want).toBeLessThan(0.25)
    expect(b.low).toBeLessThan(0.5); expect(b.high).toBeGreaterThan(0.5)
    expect(b.estimate).toBe(d.probs['00'])
    expect(Math.abs(b.mean - b.estimate)).toBeLessThan(3 * b.stderr)
  })

  it('is reproducible by seed, and finds zero variance where there is none', () => {
    const d = bell.run({ shots: 500, seed: 5 })
    const a = bootstrapOutcomes(d, s => zExpectation(s), { seed: 9 })
    const b = bootstrapOutcomes(d, s => zExpectation(s), { seed: 9 })
    expect(a).toEqual(b)
    expect(a.stderr).toBe(0)       // Bell parity is always even
    expect(a.low).toBe(1); expect(a.high).toBe(1)
  })

  it('validates its inputs', () => {
    expect(() => bootstrapOutcomes(BELL, () => 0)).toThrow(/shot count/)
    expect(() => bootstrapOutcomes(BELL, () => 0, { shots: 10, resamples: 1 })).toThrow(RangeError)
    expect(() => bootstrapOutcomes(BELL, () => 0, { shots: 10, confidence: 1 })).toThrow(RangeError)
    expect(() => bootstrapOutcomes(BELL, () => 0, { shots: 10, resample: 5 } as never)).toThrow(/did you mean 'resamples'/)
  })

  it('shotNoiseTvd matches the analytic expectation for a uniform reference', () => {
    // E|p̂ − p| ≈ √(2/π)·√(p(1−p)/n) per outcome; TVD halves the sum over 4 outcomes.
    const n = 1000
    const want = 2 * Math.sqrt(2 / Math.PI) * Math.sqrt(0.25 * 0.75 / n)
    const got = shotNoiseTvd(UNIFORM2, n, { seed: 2, resamples: 2000 })
    expect(Math.abs(got.mean - want) / want).toBeLessThan(0.1)
    expect(got.estimate).toBe(0)
  })
})

describe('compareOutcomes', () => {
  it('flags a noisy device beyond shot noise and a clean sample within it', () => {
    const shots = 4000
    const noisy = bell.run({ shots, seed: 6, noise: { p2: 0.1 } })
    const c = compareOutcomes(noisy, BELL, { seed: 1 })
    expect(c.tvd).toBeGreaterThan(0.03)
    expect(c.fidelity).toBeLessThan(1)
    expect(c.shotNoise!.mean).toBeLessThan(0.02)
    expect(c.excess!).toBeGreaterThan(0)
    expect(c.sigma!).toBeGreaterThan(3)

    const clean = bell.run({ shots, seed: 7 })
    const k = compareOutcomes(clean, BELL, { seed: 1 })
    expect(k.sigma!).toBeLessThan(3)
  })

  it('stays finite against a single-bitstring reference, where the shot-noise floor is exactly zero', () => {
    const ideal = { '00': 1 }
    const slightlyOff = { '00': 0.986, '01': 0.006, '10': 0.004, '11': 0.004 }
    const c = compareOutcomes(slightlyOff, ideal, { shots: 1000, seed: 1 })
    expect(c.shotNoise!.mean).toBe(0)
    expect(c.shotNoise!.stderr).toBe(0)
    expect(c.tvdStderr!).toBeGreaterThan(0)
    expect(Number.isFinite(c.sigma!)).toBe(true)
    expect(c.sigma!).toBeGreaterThan(3)
    const exact = compareOutcomes(ideal, ideal, { shots: 1000, seed: 1 })
    expect(exact.sigma).toBe(0)
    expect(exact.tvdStderr).toBe(0)
  })

  it('leaves the noise floor undefined without a shot count, and takes shots from options', () => {
    const c = compareOutcomes(BELL, UNIFORM2)
    expect(c.tvd).toBeCloseTo(0.5, 12)
    expect(c.shotNoise).toBeUndefined(); expect(c.excess).toBeUndefined(); expect(c.sigma).toBeUndefined(); expect(c.tvdStderr).toBeUndefined()
    const d = compareOutcomes({ '00': 0.5, '11': 0.5 }, BELL, { shots: 100, seed: 1 })
    expect(d.shotNoise!.mean).toBeGreaterThan(0)
    expect(d.sigma!).toBeLessThan(0)   // exactly on the reference, below the typical shot-noise distance
  })
})

describe('compareExperiments', () => {
  const grid = { theta: [0.3, 0.9, 1.5, 2.1] }
  const build = ({ theta }: { theta: number }) => new Circuit(2).ry(theta, 0).cnot(0, 1)

  it('compares a noisy sweep to its exact reference point by point', async () => {
    const reference = await sweep(grid, build, c => c.exactProbs())
    const noisy = await sweep(grid, build, (c, i) => c.run({ shots: 3000, seed: 10 + i, noise: { p2: 0.15 } }))
    const clean = await sweep(grid, build, (c, i) => c.run({ shots: 3000, seed: 20 + i }))

    const bad = compareExperiments(noisy, reference, { seed: 1 })
    expect(bad.points).toHaveLength(4)
    expect(bad.points.map(p => p.params.theta)).toEqual(grid.theta)
    expect(bad.beyondNoise.length).toBeGreaterThanOrEqual(3)
    expect(bad.maxTvd).toBeGreaterThanOrEqual(bad.meanTvd)
    expect(bad.meanFidelity).toBeLessThan(1)

    const good = compareExperiments(clean, reference, { seed: 1 })
    expect(good.beyondNoise).toEqual([])
    expect(good.meanTvd).toBeLessThan(bad.meanTvd)
  })

  it('requires aligned points', async () => {
    const a = await sweep({ theta: [0.3, 0.9] }, build, c => c.exactProbs())
    const b = await sweep({ theta: [0.3] }, build, c => c.exactProbs())
    const c = await sweep({ theta: [0.3, 1.0] }, build, c => c.exactProbs())
    expect(() => compareExperiments(a, b)).toThrow(/2 observed points vs 1/)
    expect(() => compareExperiments(a, c)).toThrow(/differ at index 1/)
    expect(compareExperiments(b, b).points[0]!.shotNoise).toBeUndefined()
  })
})
