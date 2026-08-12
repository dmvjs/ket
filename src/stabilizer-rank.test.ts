import { describe, it, expect } from 'vitest'
import { StabilizerRank, buildSlice, countSplits, extent, termBudget, type SrOp } from './stabilizer-rank.js'
import { sampleFromOracle, type AmplitudeOracle } from './stabilizer-sampling.js'
import { maxTGates, bytesPerTerm } from './stabilizer-capacity.js'
import { Circuit } from './circuit.js'
import { makePrng } from './prng.js'
import type { Complex } from './complex.js'

const S2 = Math.SQRT1_2

function dense(c: Circuit): Complex[] {
  const sv = c.statevector()
  const out: Complex[] = Array.from({ length: 2 ** c.qubits }, () => ({ re: 0, im: 0 }))
  for (const [idx, z] of sv) out[Number(idx)] = z
  return out
}

function expectVec(got: Complex[], want: Complex[], label: string, dp = 10): void {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) {
    expect(got[i]!.re, `${label} re[${i}]`).toBeCloseTo(want[i]!.re, dp)
    expect(got[i]!.im, `${label} im[${i}]`).toBeCloseTo(want[i]!.im, dp)
  }
}

type Gate =
  | { g: 'h' | 's' | 'sdg' | 'x' | 'y' | 'z' | 't' | 'tdg'; a: number }
  | { g: 'cx' | 'cz' | 'swap'; a: number; b: number }

/** Build a random Clifford+T circuit with a bounded number of T gates. */
function randomCliffordT(n: number, len: number, maxT: number, rand: () => number): Gate[] {
  const ONE = ['h', 's', 'sdg', 'x', 'y', 'z'] as const
  const TWO = ['cx', 'cz', 'swap'] as const
  const ops: Gate[] = []
  let used = 0
  for (let i = 0; i < len; i++) {
    const r = rand()
    if (used < maxT && r < 0.2) {
      ops.push({ g: rand() < 0.5 ? 't' : 'tdg', a: Math.floor(rand() * n) })
      used++
    } else if (n > 1 && r < 0.55) {
      const a = Math.floor(rand() * n)
      let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
      ops.push({ g: TWO[Math.floor(rand() * TWO.length)]!, a, b })
    } else {
      ops.push({ g: ONE[Math.floor(rand() * ONE.length)]!, a: Math.floor(rand() * n) })
    }
  }
  return ops
}

function onCircuit(n: number, ops: Gate[]): Circuit {
  let c = new Circuit(n)
  for (const op of ops) {
    if (op.g === 'cx') c = c.cnot(op.a, op.b)
    else if (op.g === 'cz') c = c.cz(op.a, op.b)
    else if (op.g === 'swap') c = c.swap(op.a, op.b)
    else c = c[op.g](op.a)
  }
  return c
}

function onRank(n: number, ops: Gate[]): StabilizerRank {
  const sr = new StabilizerRank(n)
  for (const op of ops) {
    if (op.g === 'cx' || op.g === 'cz' || op.g === 'swap') sr[op.g](op.a, op.b)
    else sr[op.g](op.a)
  }
  return sr
}

describe('StabilizerRank — exact single-gate behaviour', () => {
  it('starts as |0…0⟩ with one term', () => {
    const sr = new StabilizerRank(3)
    expect(sr.termCount).toBe(1)
    expect(sr.amplitude('000').re).toBeCloseTo(1, 12)
  })

  it('T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2', () => {
    const sr = new StabilizerRank(1).h(0).t(0)
    expect(sr.amplitude('0')).toMatchObject({ re: expect.closeTo(S2, 10), im: expect.closeTo(0, 10) })
    expect(sr.amplitude('1')).toMatchObject({ re: expect.closeTo(0.5, 10), im: expect.closeTo(0.5, 10) })
  })

  it('T splits into two terms; Clifford angles do not', () => {
    const sr = new StabilizerRank(2)
    expect(sr.termCount).toBe(1)
    sr.t(0); expect(sr.termCount).toBe(2)
    sr.t(1); expect(sr.termCount).toBe(4)
    sr.phase(Math.PI / 2, 0); expect(sr.termCount).toBe(4)   // S
    sr.phase(Math.PI, 0); expect(sr.termCount).toBe(4)       // Z
    sr.phase(0, 1); expect(sr.termCount).toBe(4)             // I
    sr.phase(-Math.PI / 2, 1); expect(sr.termCount).toBe(4)  // S†
  })

  it('the T decomposition is ℓ₁-optimal: ‖c‖₁ = 2^{0.228} per T gate', () => {
    const sr = new StabilizerRank(1).h(0)
    expect(sr.l1).toBeCloseTo(1, 12)
    sr.t(0)
    // ξ = ‖c‖₁² = 1/cos²(π/8)
    expect(sr.l1 ** 2).toBeCloseTo(1 / Math.cos(Math.PI / 8) ** 2, 10)
    expect(Math.log2(sr.l1 ** 2)).toBeCloseTo(0.2284, 3)
  })

  it('T·T† = I exactly', () => {
    const sr = new StabilizerRank(1).h(0).t(0).tdg(0)
    expect(sr.amplitude('0').re).toBeCloseTo(S2, 10)
    expect(sr.amplitude('1')).toMatchObject({ re: expect.closeTo(S2, 10), im: expect.closeTo(0, 10) })
  })

  it('T⁸ = I', () => {
    const sr = new StabilizerRank(1).h(0)
    for (let i = 0; i < 8; i++) sr.t(0)
    expect(sr.amplitude('1')).toMatchObject({ re: expect.closeTo(S2, 9), im: expect.closeTo(0, 9) })
  })

  it('is exact by default and reports it', () => {
    const sr = new StabilizerRank(2).h(0).t(0).t(1)
    expect(sr.sparsified).toBe(false)
  })
})

describe('StabilizerRank — differential vs statevector backend', () => {
  for (const n of [1, 2, 3, 4, 5]) {
    it(`random Clifford+T circuits, n=${n}`, () => {
      for (let seed = 0; seed < 25; seed++) {
        const rand = makePrng(seed * 7919 + n * 131 + 5)
        const ops = randomCliffordT(n, 5 * n + 12, 8, rand)
        expectVec(onRank(n, ops).toStatevector(), dense(onCircuit(n, ops)), `n=${n} seed=${seed}`)
      }
    })
  }

  it('matches u1(θ) at arbitrary non-Clifford angles', () => {
    const rand = makePrng(31337)
    for (let seed = 0; seed < 30; seed++) {
      const n = 3
      let circ = new Circuit(n).h(0).h(1).h(2).cnot(0, 1).cnot(1, 2)
      const sr = new StabilizerRank(n).h(0).h(1).h(2).cx(0, 1).cx(1, 2)
      for (let i = 0; i < 4; i++) {
        const q = Math.floor(rand() * n)
        const th = (rand() * 4 - 2) * Math.PI
        circ = circ.u1(th, q)
        sr.phase(th, q)
      }
      expectVec(sr.toStatevector(), dense(circ), `u1 seed=${seed}`)
    }
  })

  it('rz(θ) carries the e^{-iθ/2} global phase', () => {
    const rand = makePrng(6060)
    for (let seed = 0; seed < 20; seed++) {
      const th = (rand() * 4 - 2) * Math.PI
      const q = Math.floor(rand() * 2)
      const circ = new Circuit(2).h(0).cnot(0, 1).rz(th, q)
      const sr = new StabilizerRank(2).h(0).cx(0, 1).rz(th, q)
      expectVec(sr.toStatevector(), dense(circ), `rz seed=${seed}`)
    }
  })

  it('term count is 2^t, independent of qubit count', () => {
    for (const n of [2, 5, 40, 200]) {
      const sr = new StabilizerRank(n)
      for (let q = 0; q < n; q++) sr.h(q)
      for (let i = 0; i < 6; i++) sr.t(i % n)
      expect(sr.termCount).toBe(64)
    }
  })
})

describe('StabilizerRank — sparsification', () => {
  it('caps the term count and flags the run as approximate', () => {
    const sr = new StabilizerRank(4, { maxTerms: 16, seed: 7 })
    for (let q = 0; q < 4; q++) sr.h(q)
    for (let i = 0; i < 10; i++) sr.t(i % 4)
    expect(sr.termCount).toBeLessThanOrEqual(16)
    expect(sr.sparsified).toBe(true)
  })

  it('is unbiased — averaging independent sparsifications converges to exact', () => {
    const n = 3
    const build = (maxTerms: number, seed: number): StabilizerRank => {
      const sr = new StabilizerRank(n, { maxTerms, seed })
      sr.h(0).h(1).h(2).cx(0, 1).t(0).t(1).cx(1, 2).t(2).t(0)
      return sr
    }
    const exact = build(Infinity, 0).toStatevector()
    const dim = 1 << n
    const acc = Array.from({ length: dim }, () => ({ re: 0, im: 0 }))
    const runs = 4000
    for (let r = 0; r < runs; r++) {
      const sv = build(4, r * 2654435761 + 1).toStatevector()
      for (let i = 0; i < dim; i++) { acc[i]!.re += sv[i]!.re; acc[i]!.im += sv[i]!.im }
    }
    for (let i = 0; i < dim; i++) {
      expect(acc[i]!.re / runs, `mean re[${i}]`).toBeCloseTo(exact[i]!.re, 1)
      expect(acc[i]!.im / runs, `mean im[${i}]`).toBeCloseTo(exact[i]!.im, 1)
    }
  })

  it('error shrinks as the term budget grows', () => {
    const n = 3
    const err = (maxTerms: number): number => {
      const exact = (() => {
        const s = new StabilizerRank(n)
        s.h(0).h(1).h(2).t(0).t(1).t(2).cx(0, 1).t(0).t(2)
        return s.toStatevector()
      })()
      let acc = 0
      const trials = 60
      for (let r = 0; r < trials; r++) {
        const s = new StabilizerRank(n, { maxTerms, seed: r * 7717 + 3 })
        s.h(0).h(1).h(2).t(0).t(1).t(2).cx(0, 1).t(0).t(2)
        const sv = s.toStatevector()
        let d = 0
        for (let i = 0; i < 1 << n; i++) {
          d += (sv[i]!.re - exact[i]!.re) ** 2 + (sv[i]!.im - exact[i]!.im) ** 2
        }
        acc += d
      }
      return acc / trials
    }
    expect(err(16)).toBeLessThan(err(2))
    expect(err(4)).toBeLessThan(err(2))
  })
})

describe('StabilizerRank — sampling', () => {
  it('reproduces |⟨x|ψ⟩|² for a Clifford+T state', () => {
    const n = 3
    const sr = new StabilizerRank(n)
    sr.h(0).h(1).h(2).t(0).cx(0, 1).t(1).cx(1, 2).t(2).h(1)
    const want = sr.toStatevector().map(z => z.re * z.re + z.im * z.im)

    const shots = 20000
    const counts = new Float64Array(1 << n)
    for (const x of sr.sample(shots, makePrng(4242), { burnIn: 500, thin: 12 })) {
      let idx = 0
      for (let j = 0; j < n; j++) idx |= (x[j] ?? 0) << j
      counts[idx]!++
    }
    for (let i = 0; i < 1 << n; i++) {
      expect(counts[i]! / shots, `outcome ${i}`).toBeCloseTo(want[i]!, 1)
    }
  })
})

describe('Circuit.runStabilizerRank', () => {
  it('reports the backend and that an unbounded run stayed exact', () => {
    const d = new Circuit(3).h(0).t(0).cnot(0, 1).t(1).runStabilizerRank({ shots: 200, seed: 1 })
    expect(d.backend).toBe('stabilizer-rank')
    expect(d.truncated).toBe(false)
  })

  it('flags a sparsified run as approximate', () => {
    let c = new Circuit(4).h(0).h(1).h(2).h(3)
    for (let i = 0; i < 10; i++) c = c.t(i % 4)
    expect(c.runStabilizerRank({ shots: 100, seed: 2, maxTerms: 8 }).truncated).toBe(true)
  })

  it('decomposes Toffoli exactly on every basis input', () => {
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let t = 0; t < 2; t++) {
      let c = new Circuit(3)
      if (a) c = c.x(0)
      if (b) c = c.x(1)
      if (t) c = c.x(2)
      c = c.ccx(0, 1, 2)
      const want = `${a}${b}${a && b ? t ^ 1 : t}`
      const d = c.runStabilizerRank({ shots: 64, seed: 5, burnIn: 30, thin: 4 })
      expect(d.probs[want], `ccx(${a},${b},${t})`).toBeCloseTo(1, 6)
    }
  })

  it('matches exactProbs on a Toffoli in superposition', () => {
    const c = new Circuit(3).h(0).h(1).ccx(0, 1, 2)
    const want = c.exactProbs()
    const got = c.runStabilizerRank({ shots: 20000, seed: 8, burnIn: 500, thin: 12 }).probs
    for (const [bits, p] of Object.entries(want)) {
      expect(got[bits] ?? 0, `outcome ${bits}`).toBeCloseTo(p, 1)
    }
  })

  it('matches exactProbs on a Clifford+T circuit', () => {
    const c = new Circuit(3).h(0).h(1).h(2).t(0).cnot(0, 1).t(1).cnot(1, 2).t(2).h(1)
    const want = c.exactProbs()
    const got = c.runStabilizerRank({ shots: 20000, seed: 3, burnIn: 500, thin: 12 }).probs
    for (const [bits, p] of Object.entries(want)) {
      expect(got[bits] ?? 0, `outcome ${bits}`).toBeCloseTo(p, 1)
    }
  })

  it('rejects gates outside the Clifford + diagonal set', () => {
    expect(() => new Circuit(2).rx(0.3, 0).runStabilizerRank()).toThrow(TypeError)
    expect(() => new Circuit(2).ry(0.3, 0).runStabilizerRank()).toThrow(/not supported/)
    expect(() => new Circuit(2).creg('m', 1).measure(0, 'm', 0).runStabilizerRank())
      .toThrow(/mid-circuit measurement/)
  })

  it('accepts arbitrary diagonal rotations', () => {
    const c = new Circuit(2).h(0).h(1).u1(0.371, 0).rz(1.113, 1).cz(0, 1)
    const want = c.exactProbs()
    const got = c.runStabilizerRank({ shots: 8000, seed: 4, burnIn: 400, thin: 10 }).probs
    for (const [bits, p] of Object.entries(want)) {
      expect(got[bits] ?? 0, `outcome ${bits}`).toBeCloseTo(p, 1)
    }
  })

  it('runs at 60 qubits, far past statevector reach', () => {
    let c = new Circuit(60).h(0)
    for (let q = 0; q < 59; q++) c = c.cnot(q, q + 1)
    for (let i = 0; i < 8; i++) c = c.t(i * 7 % 60)
    const d = c.runStabilizerRank({ shots: 64, seed: 6, maxTerms: 64, burnIn: 40, thin: 4 })
    expect(d.backend).toBe('stabilizer-rank')
    expect(d.shots).toBe(64)
    expect(Object.keys(d.probs).length).toBeGreaterThan(0)
  })
})

describe('StabilizerRank — slice replay (the basis for worker fan-out)', () => {
  const OPS: SrOp[] = [
    { g: 'h', q: 0 }, { g: 'h', q: 1 }, { g: 'h', q: 2 },
    { g: 'phase', q: 0, theta: Math.PI / 4 },
    { g: 'cx', a: 0, b: 1 },
    { g: 'rz', q: 1, theta: 0.7391 },
    { g: 'phase', q: 2, theta: Math.PI / 2 },      // Clifford — must not split
    { g: 'cz', a: 1, b: 2 },
    { g: 'phase', q: 2, theta: -Math.PI / 4 },
    { g: 'y', q: 0 },
    { g: 'phase', q: 1, theta: 1.234 },
  ]

  const sequential = (): StabilizerRank => {
    const sr = new StabilizerRank(3)
    for (const op of OPS) {
      if (op.g === 'cx' || op.g === 'cz' || op.g === 'swap') sr[op.g](op.a, op.b)
      else if (op.g === 'phase') sr.phase(op.theta, op.q)
      else if (op.g === 'rz') sr.rz(op.theta, op.q)
      else sr[op.g](op.q)
    }
    return sr
  }

  it('counts only genuinely non-Clifford splits', () => {
    expect(countSplits(OPS)).toBe(4)                                  // the π/2 gate is free
    expect(countSplits([{ g: 'phase', q: 0, theta: Math.PI }])).toBe(0)
    expect(countSplits([{ g: 'rz', q: 0, theta: Math.PI / 4 }])).toBe(1)
  })

  it('a full single slice reproduces the sequential build exactly', () => {
    const total = 2 ** countSplits(OPS)
    const sliced = new StabilizerRank(3).setTerms(buildSlice(3, OPS, 0, total))
    expect(sliced.termCount).toBe(sequential().termCount)
    expectVec(sliced.toStatevector(), sequential().toStatevector(), 'one slice')
  })

  it('disjoint slices partition the decomposition and sum to the whole', () => {
    const total = 2 ** countSplits(OPS)
    for (const w of [2, 3, 4, 5, 8, 16]) {
      const terms = []
      for (let i = 0; i < w; i++) {
        const lo = Math.floor((i * total) / w), hi = Math.floor(((i + 1) * total) / w)
        terms.push(...buildSlice(3, OPS, lo, hi))
      }
      expect(terms.length, `w=${w} term count`).toBe(total)
      const joined = new StabilizerRank(3).setTerms(terms)
      expectVec(joined.toStatevector(), sequential().toStatevector(), `w=${w}`)
    }
  })

  it('slice amplitudes are additive — the reduce step workers rely on', () => {
    const total = 2 ** countSplits(OPS)
    const whole = sequential()
    const half = total >> 1
    const a = new StabilizerRank(3).setTerms(buildSlice(3, OPS, 0, half))
    const b = new StabilizerRank(3).setTerms(buildSlice(3, OPS, half, total))
    for (const bits of ['000', '101', '011', '111']) {
      const za = a.amplitude(bits), zb = b.amplitude(bits), zw = whole.amplitude(bits)
      expect(za.re + zb.re, `re ${bits}`).toBeCloseTo(zw.re, 10)
      expect(za.im + zb.im, `im ${bits}`).toBeCloseTo(zw.im, 10)
    }
  })

  it('replay matches the statevector backend end to end', () => {
    let circ = new Circuit(3).h(0).h(1).h(2).u1(Math.PI / 4, 0).cnot(0, 1)
    circ = circ.rz(0.7391, 1).u1(Math.PI / 2, 2).cz(1, 2).u1(-Math.PI / 4, 2).y(0).u1(1.234, 1)
    const total = 2 ** countSplits(OPS)
    const sliced = new StabilizerRank(3).setTerms(buildSlice(3, OPS, 0, total))
    expectVec(sliced.toStatevector(), dense(circ), 'replay vs statevector')
  })
})

describe('StabilizerRank — Metropolis fallback', () => {
  it('covers every outcome of a uniform state — the chain must be aperiodic', () => {
    // Regression: single-bit-flip proposals flip Hamming parity on every accepted
    // move. On a uniform state everything is accepted, so a non-lazy chain paired
    // with an even `thin` samples one parity class only and silently drops half
    // the support. Laziness is what makes this pass.
    const n = 3
    const sr = new StabilizerRank(n).h(0).h(1).h(2)
    const seen = new Set<string>()
    for (const x of sr.sample(600, makePrng(17), { method: 'metropolis', burnIn: 100, thin: 10 })) {
      seen.add(Array.from(x).join(''))
    }
    expect(seen.size).toBe(1 << n)
  })

  it('exact sampling reaches support that Metropolis provably cannot', () => {
    // h,h,ccx puts weight on 000/100/010/111. |111⟩ is two flips from every other
    // supported state, so a single-flip chain cannot reach it — Section 4.2 warns
    // the chain need not be irreducible. Exact enumeration is unaffected.
    const c = new Circuit(3).h(0).h(1).ccx(0, 1, 2)
    const exact = c.runStabilizerRank({ shots: 8000, seed: 2, method: 'exact' }).probs
    for (const bits of ['000', '100', '010', '111']) {
      expect(exact[bits] ?? 0, `exact ${bits}`).toBeCloseTo(0.25, 1)
    }
    const metro = c.runStabilizerRank({ shots: 8000, seed: 2, method: 'metropolis', burnIn: 500, thin: 11 }).probs
    expect(metro['111'] ?? 0).toBe(0)
  })

  it('refuses to sample from a chain it cannot seed', () => {
    // At P(x) = 0 the acceptance test P(y) >= P(x) passes unconditionally, so a
    // chain seeded off-support degenerates into a uniform random walk that still
    // returns plausible-looking bitstrings. Failing loudly is the only safe move.
    const zero: AmplitudeOracle = (_b, count) => ({ re: new Float64Array(count), im: new Float64Array(count) })
    expect(() => sampleFromOracle(40, 8, makePrng(1), zero, { method: 'metropolis' }))
      .toThrow(/could not seed/)
  })

  it('seeds from a term, so a sparse state stays on its support', () => {
    // GHZ support is 2 of 2^40 states; a uniformly random seed would never land
    // on it. Every sample must be all-zeros or all-ones, never a random string.
    const n = 40
    const sr = new StabilizerRank(n).h(0)
    for (let q = 0; q < n - 1; q++) sr.cx(q, q + 1)
    sr.t(0).t(3)
    for (const x of sr.sample(40, makePrng(9), { method: 'metropolis', burnIn: 20, thin: 3 })) {
      const ones = x.reduce((a, b) => a + b, 0)
      expect(ones === 0 || ones === n, `sample must lie in the GHZ support, got ${ones} ones`).toBe(true)
    }
  })

  it('auto routes to exact for small circuits and Metropolis for wide ones', () => {
    const small = new StabilizerRank(3).h(0).h(1).h(2)
    const seen = new Set<string>()
    for (const x of small.sample(400, makePrng(21))) seen.add(Array.from(x).join(''))
    expect(seen.size).toBe(8)

    const wide = new StabilizerRank(40)
    for (let q = 0; q < 40; q++) wide.h(q)
    expect(() => wide.sample(4, makePrng(3), { burnIn: 10, thin: 2 })).not.toThrow()
  })
})

describe('Circuit.runStabilizerRank — worker fallback', () => {
  // Workers need the built bundle (dist/stabilizer-rank.worker.js), so from
  // source these fall back to the sequential path. The result must still be
  // correct; only the speed differs. Slice-union equivalence is covered above,
  // and the built worker path is exercised by the release build.
  it('falls back to sequential without workers and stays correct', () => {
    const c = new Circuit(3).h(0).h(1).h(2).t(0).cnot(0, 1).t(1)
    const want = c.exactProbs()
    const got = c.runStabilizerRank({ shots: 20000, seed: 5, workers: 4 }).probs
    for (const [bits, p] of Object.entries(want)) {
      expect(got[bits] ?? 0, `outcome ${bits}`).toBeCloseTo(p, 1)
    }
  })

  it('declines to parallelise a sparsified run', () => {
    let c = new Circuit(4).h(0).h(1).h(2).h(3)
    for (let i = 0; i < 8; i++) c = c.t(i % 4)
    const d = c.runStabilizerRank({ shots: 100, seed: 6, workers: 4, maxTerms: 8 })
    expect(d.truncated).toBe(true)
  })
})

describe('StabilizerRank — beyond statevector reach', () => {
  it('simulates 100 qubits with T gates', () => {
    const n = 100
    const sr = new StabilizerRank(n, { maxTerms: 64, seed: 11 })
    for (let q = 0; q < n; q++) sr.h(q)
    for (let q = 0; q < n - 1; q++) sr.cx(q, q + 1)
    for (let i = 0; i < 12; i++) sr.t(i * 7 % n)
    expect(sr.termCount).toBeLessThanOrEqual(64)

    const x = sr.sample(1, makePrng(9), { burnIn: 20, thin: 1 })[0]!
    expect(x.length).toBe(n)
    const p = sr.probability(x)
    expect(Number.isFinite(p)).toBe(true)
    expect(p).toBeGreaterThan(0)
  })
})

describe('StabilizerRank — extent and term budget', () => {
  const T: SrOp[] = [{ g: 'phase', q: 0, theta: Math.PI / 4 }]

  it('a single T gate has extent 1/cos²(π/8)', () => {
    expect(extent(T)).toBeCloseTo(1 / Math.cos(Math.PI / 8) ** 2, 12)
    expect(Math.log2(extent(T))).toBeCloseTo(0.2284, 3)
  })

  it('is multiplicative over splitting gates and ignores Clifford angles', () => {
    const xi = 1 / Math.cos(Math.PI / 8) ** 2
    const five: SrOp[] = Array.from({ length: 5 }, () => ({ g: 'phase', q: 0, theta: Math.PI / 4 }))
    expect(extent(five)).toBeCloseTo(xi ** 5, 10)
    expect(extent([...five, { g: 'phase', q: 0, theta: Math.PI / 2 }])).toBeCloseTo(xi ** 5, 10)
    expect(extent([{ g: 'h', q: 0 }, { g: 'cx', a: 0, b: 1 }])).toBe(1)
  })

  it('matches the ℓ₁ norm the decomposition actually reaches', () => {
    const sr = new StabilizerRank(1).h(0)
    for (let i = 0; i < 4; i++) sr.t(0)
    expect(sr.l1 ** 2).toBeCloseTo(extent(Array.from({ length: 4 },
      () => ({ g: 'phase', q: 0, theta: Math.PI / 4 }) as SrOp)), 9)
  })

  it('turns a 50-T circuit into a tractable budget', () => {
    const fifty: SrOp[] = Array.from({ length: 50 }, (_, i) => ({ g: 'phase', q: i % 10, theta: Math.PI / 4 }))
    expect(countSplits(fifty)).toBe(50)
    expect(2 ** 50).toBeGreaterThan(1e15)                      // exact is hopeless
    expect(termBudget(fifty, 0.2)).toBeLessThan(80_000)        // sparsified is not
    expect(termBudget(fifty, 0.1)).toBeLessThan(300_000)
  })

  it('rejects a non-positive target error', () => {
    expect(() => termBudget(T, 0)).toThrow(RangeError)
    expect(() => termBudget(T, -1)).toThrow(RangeError)
  })
})

describe('StabilizerRank — norm estimation (Lemma 2)', () => {
  const exactNorm = (sr: StabilizerRank): number =>
    sr.toStatevector().reduce((a, z) => a + z.re * z.re + z.im * z.im, 0)

  it('recovers ‖ψ‖² = 1 for exact unitary circuits', () => {
    for (const seed of [1, 2, 3, 4]) {
      const sr = new StabilizerRank(3)
      sr.h(0).h(1).h(2).t(0).cx(0, 1).t(1).cx(1, 2).t(2).h(1)
      expect(exactNorm(sr)).toBeCloseTo(1, 9)
      const got = sr.estimateNorm({ epsilon: 0.15, delta: 0.1, rand: makePrng(seed) })
      expect(got, `seed=${seed}`).toBeGreaterThan(0.7)
      expect(got, `seed=${seed}`).toBeLessThan(1.3)
    }
  })

  it('tracks a deliberately rescaled state', () => {
    const sr = new StabilizerRank(3).h(0).cx(0, 1).t(0).t(2)
    sr.sparsify(sr.termCount)                       // no-op, keeps terms intact
    const want = exactNorm(sr)
    const got = sr.estimateNorm({ epsilon: 0.15, delta: 0.1, rand: makePrng(9) })
    expect(got / want).toBeGreaterThan(0.7)
    expect(got / want).toBeLessThan(1.3)
  })

  it('agrees with the exact norm across several circuits', () => {
    const circuits: ((s: StabilizerRank) => void)[] = [
      s => { s.h(0).h(1).t(0).cz(0, 1) },
      s => { s.h(0).cx(0, 1).cx(1, 2).t(2) },
      s => { s.h(0).h(1).h(2).t(0).t(1).t(2) },
      s => { s.x(0).h(1).t(1).cx(1, 2).s(2) },
    ]
    circuits.forEach((build, i) => {
      const sr = new StabilizerRank(3)
      build(sr)
      const want = exactNorm(sr)
      const got = sr.estimateNorm({ epsilon: 0.15, delta: 0.1, rand: makePrng(100 + i) })
      expect(got / want, `circuit ${i}`).toBeGreaterThan(0.7)
      expect(got / want, `circuit ${i}`).toBeLessThan(1.3)
    })
  })

  it('rejects invalid precision parameters', () => {
    const sr = new StabilizerRank(2).h(0)
    expect(() => sr.estimateNorm({ epsilon: 0 })).toThrow(RangeError)
    expect(() => sr.estimateNorm({ delta: 0 })).toThrow(RangeError)
    expect(() => sr.estimateNorm({ delta: 1 })).toThrow(RangeError)
  })
})

describe('Circuit.runStabilizerRank — error-driven sparsification', () => {
  it('derives a term budget from targetError', () => {
    let c = new Circuit(4).h(0).h(1).h(2).h(3)
    for (let i = 0; i < 12; i++) c = c.t(i % 4)
    const d = c.runStabilizerRank({ shots: 200, seed: 3, targetError: 0.3 })
    expect(d.truncated).toBe(true)                    // 2^12 terms exceeds the budget
    expect(d.backend).toBe('stabilizer-rank')
  })

  it('leaves a small circuit exact when the budget already covers it', () => {
    const c = new Circuit(3).h(0).h(1).h(2).t(0).t(1)
    const d = c.runStabilizerRank({ shots: 200, seed: 3, targetError: 0.3 })
    expect(d.truncated).toBe(false)                   // 4 terms is under ⌈ξ/δ²⌉
  })

  it('an explicit maxTerms overrides targetError', () => {
    let c = new Circuit(3).h(0).h(1).h(2)
    for (let i = 0; i < 8; i++) c = c.t(i % 3)
    // targetError alone would allow ~19 terms; maxTerms 2 must bind instead.
    const d = c.runStabilizerRank({ shots: 100, seed: 1, targetError: 0.3, maxTerms: 2 })
    expect(d.truncated).toBe(true)
  })

  it('stays close to the exact distribution at a modest target error', () => {
    const c = new Circuit(3).h(0).h(1).h(2).t(0).cnot(0, 1).t(1).cnot(1, 2).t(2).h(1)
    const want = c.exactProbs()
    const got = c.runStabilizerRank({ shots: 40000, seed: 5, targetError: 0.15 }).probs
    for (const [bits, p] of Object.entries(want)) {
      expect(got[bits] ?? 0, `outcome ${bits}`).toBeCloseTo(p, 1)
    }
  })

  it('counts how many times sparsification compounded', () => {
    const sr = new StabilizerRank(3, { maxTerms: 4, seed: 1 })
    sr.h(0).h(1).h(2)
    expect(sr.sparsifications).toBe(0)
    for (let i = 0; i < 6; i++) sr.t(i % 3)
    expect(sr.sparsifications).toBeGreaterThan(1)     // the bound covers one, not many
    expect(sr.sparsified).toBe(true)
  })
})

describe('StabilizerRank — sparsification error is actually controlled', () => {
  it('output error falls as targetError tightens', () => {
    // n=6, t=16 keeps the exact decomposition (65536 terms) affordable, so the
    // deviation measured here is sparsification damage, not sampling noise.
    // The per-application bound does not certify a repeatedly sparsified run;
    // this is the empirical check that the knob still works in practice.
    let c = new Circuit(6).h(0).h(1).h(2)
    for (let i = 0; i < 5; i++) c = c.cnot(i, i + 1)
    for (let i = 0; i < 16; i++) c = c.t((i * 5) % 6)
    const exact = c.exactProbs()

    const tv = (probs: Record<string, number>): number => {
      let d = 0
      for (const k of new Set([...Object.keys(exact), ...Object.keys(probs)])) {
        d += Math.abs((exact[k] ?? 0) - (probs[k] ?? 0))
      }
      return d / 2
    }
    const meanTv = (targetError?: number): number => {
      let acc = 0
      const seeds = 5
      for (let s = 0; s < seeds; s++) {
        acc += tv(c.runStabilizerRank(
          targetError === undefined
            ? { shots: 20000, seed: s + 1 }
            : { shots: 20000, seed: s + 1, targetError }).probs)
      }
      return acc / seeds
    }

    const loose = meanTv(0.5), tight = meanTv(0.1), exactRun = meanTv(undefined)
    expect(tight).toBeLessThan(loose)
    expect(exactRun).toBeLessThan(tight)
    expect(exactRun).toBeLessThan(0.02)          // exact run: sampling noise only
    expect(tight).toBeLessThan(0.15)
  })

  it('norm estimation detects sparsification damage', () => {
    // A unitary circuit has ‖ψ‖² = 1 exactly. Heavy sparsification perturbs it,
    // and estimateNorm sees that without enumerating 2ⁿ amplitudes.
    const build = (maxTerms: number): StabilizerRank => {
      const sr = new StabilizerRank(4, { maxTerms, seed: 3 })
      sr.h(0).h(1).h(2).h(3)
      for (let i = 0; i < 10; i++) sr.t(i % 4)
      return sr
    }
    const exact = build(Infinity)
    expect(exact.sparsified).toBe(false)
    expect(exact.estimateNorm({ epsilon: 0.15, delta: 0.1, rand: makePrng(2) })).toBeGreaterThan(0.7)

    const crushed = build(2)
    expect(crushed.sparsifications).toBeGreaterThan(1)
    expect(Number.isFinite(crushed.estimateNorm({ epsilon: 0.2, delta: 0.2, rand: makePrng(2) }))).toBe(true)
  })
})

describe('StabilizerRank — the T-count ceiling is a system property', () => {
  it('models measured per-term memory', () => {
    // Measured with packed tableaus: 1.58 / 2.72 / 6.31 / 18.32 / 63.44 KB.
    // Assert relative agreement — the constant term is an allocator detail, so
    // demanding absolute KB precision would be pinning noise.
    for (const [n, kb] of [[10, 1.58], [50, 2.72], [100, 6.31], [200, 18.32], [400, 63.44]] as const) {
      expect(Math.abs(bytesPerTerm(n) / 1024 - kb) / kb, `n=${n}`).toBeLessThan(0.02)
    }
  })

  it('reproduces the measured ceiling at n=100', () => {
    // Peak RSS re-measured with packed tableaus at n=100, δ=0.3. The model is
    // accurate to ±1 T gate, which is the useful resolution: each extra gate
    // costs another factor of ζ in memory.
    for (const [gb, t] of [[1.02, 50], [2.05, 55], [4.70, 60]] as const) {
      expect(Math.abs(maxTGates({ qubits: 100, targetError: 0.3, memoryBytes: gb * 1e9 }) - t),
        `${gb} GB should allow about t=${t}`).toBeLessThanOrEqual(1)
    }
  })

  it('moves with width, tolerance and memory', () => {
    const base = maxTGates({ qubits: 100, targetError: 0.3, memoryBytes: 16e9 })
    expect(maxTGates({ qubits: 20, targetError: 0.3, memoryBytes: 16e9 })).toBeGreaterThan(base)
    expect(maxTGates({ qubits: 400, targetError: 0.3, memoryBytes: 16e9 })).toBeLessThan(base)
    expect(maxTGates({ qubits: 100, targetError: 0.5, memoryBytes: 16e9 })).toBeGreaterThan(base)
    expect(maxTGates({ qubits: 100, targetError: 0.1, memoryBytes: 16e9 })).toBeLessThan(base)
    expect(maxTGates({ qubits: 100, targetError: 0.3, memoryBytes: 64e9 })).toBeGreaterThan(base)
  })

  it('agrees with termBudget at the ceiling', () => {
    const qubits = 100, targetError = 0.3, memoryBytes = 16e9
    const t = maxTGates({ qubits, targetError, memoryBytes })
    const ops: SrOp[] = Array.from({ length: t }, (_, i) => ({ g: 'phase', q: i % qubits, theta: Math.PI / 4 }))
    // 5 mirrors the module's PEAK_FACTOR: peak RSS over resident size.
    const need = termBudget(ops, targetError) * bytesPerTerm(qubits) * 5
    expect(need).toBeLessThanOrEqual(memoryBytes)
  })

  it('returns 0 when nothing fits and rejects bad inputs', () => {
    expect(maxTGates({ qubits: 1000, targetError: 0.01, memoryBytes: 1e5 })).toBe(0)
    expect(() => maxTGates({ qubits: 10, targetError: 0, memoryBytes: 1e9 })).toThrow(RangeError)
    expect(() => maxTGates({ qubits: 10, targetError: 0.1, memoryBytes: 0 })).toThrow(RangeError)
  })
})
