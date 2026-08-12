import { describe, it, expect } from 'vitest'
import { StabilizerCH, randomEquatorial } from './stabilizer-ch.js'
import { Circuit } from './circuit.js'
import { makePrng } from './prng.js'
import type { Complex } from './complex.js'

const S2 = Math.SQRT1_2

/** Densify a Circuit's sparse statevector; qubit 0 is the least significant bit. */
function dense(c: Circuit): Complex[] {
  const sv = c.statevector()
  const out: Complex[] = Array.from({ length: 2 ** c.qubits }, () => ({ re: 0, im: 0 }))
  for (const [idx, z] of sv) out[Number(idx)] = z
  return out
}

function expectVec(got: Complex[], want: Complex[], label: string): void {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) {
    expect(got[i]!.re, `${label} re[${i}]`).toBeCloseTo(want[i]!.re, 10)
    expect(got[i]!.im, `${label} im[${i}]`).toBeCloseTo(want[i]!.im, 10)
  }
}

type Gate =
  | { g: 'h' | 's' | 'sdg' | 'x' | 'y' | 'z'; a: number }
  | { g: 'cx' | 'cz' | 'swap'; a: number; b: number }

const ONE_Q = ['h', 's', 'sdg', 'x', 'y', 'z'] as const
const TWO_Q = ['cx', 'cz', 'swap'] as const

function randomClifford(n: number, len: number, rand: () => number): Gate[] {
  const ops: Gate[] = []
  for (let i = 0; i < len; i++) {
    if (n > 1 && rand() < 0.4) {
      const a = Math.floor(rand() * n)
      let b = Math.floor(rand() * (n - 1))
      if (b >= a) b++
      ops.push({ g: TWO_Q[Math.floor(rand() * TWO_Q.length)]!, a, b })
    } else {
      ops.push({ g: ONE_Q[Math.floor(rand() * ONE_Q.length)]!, a: Math.floor(rand() * n) })
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

function onCH(n: number, ops: Gate[]): StabilizerCH {
  const st = new StabilizerCH(n)
  for (const op of ops) {
    if (op.g === 'cx' || op.g === 'cz' || op.g === 'swap') st[op.g](op.a, op.b)
    else st[op.g](op.a)
  }
  return st
}

describe('StabilizerCH — construction', () => {
  it('initialises to |0…0⟩', () => {
    const st = new StabilizerCH(3)
    expect(st.amplitude('000')).toEqual({ re: 1, im: 0 })
    for (const x of ['001', '010', '100', '111']) {
      expect(st.amplitude(x)).toEqual({ re: 0, im: 0 })
    }
  })

  it('rejects invalid sizes and qubit indices', () => {
    expect(() => new StabilizerCH(0)).toThrow(RangeError)
    expect(() => new StabilizerCH(2.5)).toThrow(RangeError)
    expect(() => new StabilizerCH(2).h(2)).toThrow(RangeError)
    expect(() => new StabilizerCH(2).h(-1)).toThrow(RangeError)
    expect(() => new StabilizerCH(2).cx(1, 1)).toThrow(RangeError)
    expect(() => new StabilizerCH(2).amplitude('0')).toThrow(RangeError)
  })
})

describe('StabilizerCH — exact single-qubit amplitudes', () => {
  it('H|0⟩ = (|0⟩+|1⟩)/√2', () => {
    const st = new StabilizerCH(1).h(0)
    expect(st.amplitude('0').re).toBeCloseTo(S2, 12)
    expect(st.amplitude('1').re).toBeCloseTo(S2, 12)
  })

  it('S·H|0⟩ = (|0⟩+i|1⟩)/√2 — phase on the excited branch only', () => {
    const st = new StabilizerCH(1).h(0).s(0)
    expect(st.amplitude('0')).toMatchObject({ re: expect.closeTo(S2, 12), im: expect.closeTo(0, 12) })
    expect(st.amplitude('1')).toMatchObject({ re: expect.closeTo(0, 12), im: expect.closeTo(S2, 12) })
  })

  it('S†·H|0⟩ = (|0⟩-i|1⟩)/√2', () => {
    const st = new StabilizerCH(1).h(0).sdg(0)
    expect(st.amplitude('1').im).toBeCloseTo(-S2, 12)
  })

  it('X|0⟩ = |1⟩ with no spurious phase', () => {
    const st = new StabilizerCH(1).x(0)
    expect(st.amplitude('0')).toMatchObject({ re: expect.closeTo(0, 12), im: expect.closeTo(0, 12) })
    expect(st.amplitude('1')).toMatchObject({ re: expect.closeTo(1, 12), im: expect.closeTo(0, 12) })
  })

  it('Y|0⟩ = i|1⟩ — global phase is tracked, not discarded', () => {
    const st = new StabilizerCH(1).y(0)
    expect(st.amplitude('1')).toMatchObject({ re: expect.closeTo(0, 12), im: expect.closeTo(1, 12) })
  })

  it('Z·H|0⟩ = (|0⟩-|1⟩)/√2', () => {
    const st = new StabilizerCH(1).h(0).z(0)
    expect(st.amplitude('1').re).toBeCloseTo(-S2, 12)
  })

  it('H·H = I exactly', () => {
    const st = new StabilizerCH(1).h(0).h(0)
    expect(st.amplitude('0')).toMatchObject({ re: expect.closeTo(1, 12), im: expect.closeTo(0, 12) })
  })

  it('S⁴ = I exactly, including phase', () => {
    const st = new StabilizerCH(1).h(0).s(0).s(0).s(0).s(0)
    expect(st.amplitude('1')).toMatchObject({ re: expect.closeTo(S2, 12), im: expect.closeTo(0, 12) })
  })
})

describe('StabilizerCH — entangled states', () => {
  it('Bell state (|00⟩+|11⟩)/√2', () => {
    const st = new StabilizerCH(2).h(0).cx(0, 1)
    expect(st.amplitude('00').re).toBeCloseTo(S2, 12)
    expect(st.amplitude('11').re).toBeCloseTo(S2, 12)
    expect(st.amplitude('10').re).toBeCloseTo(0, 12)
    expect(st.amplitude('01').re).toBeCloseTo(0, 12)
  })

  it('GHZ-6 has exactly two non-zero amplitudes', () => {
    const st = new StabilizerCH(6).h(0)
    for (let q = 0; q < 5; q++) st.cx(q, q + 1)
    const sv = st.toStatevector()
    const nz = sv.filter(z => Math.hypot(z.re, z.im) > 1e-12)
    expect(nz.length).toBe(2)
    expect(sv[0]!.re).toBeCloseTo(S2, 12)
    expect(sv[63]!.re).toBeCloseTo(S2, 12)
  })

  it('CZ matches the Circuit backend on a superposition', () => {
    const ops: Gate[] = [{ g: 'h', a: 0 }, { g: 'h', a: 1 }, { g: 'cz', a: 0, b: 1 }]
    expectVec(onCH(2, ops).toStatevector(), dense(onCircuit(2, ops)), 'cz')
  })
})

describe('StabilizerCH — differential vs statevector backend', () => {
  // The acceptance criterion for CH-form: amplitudes must agree with the
  // statevector kernel exactly, global phase included. Agreement up to phase
  // would be satisfied by the existing CHP tableau and is not the claim here.
  for (const n of [1, 2, 3, 4, 5, 6]) {
    it(`random Clifford circuits, n=${n}`, () => {
      for (let seed = 0; seed < 40; seed++) {
        const rand = makePrng(seed * 6151 + n * 97 + 1)
        const ops = randomClifford(n, 6 * n + 10, rand)
        expectVec(onCH(n, ops).toStatevector(), dense(onCircuit(n, ops)), `n=${n} seed=${seed}`)
      }
    })
  }

  it('Hadamard-heavy circuits exercise both Proposition 4 branches', () => {
    // V0 = ∅ (every disagreeing qubit already in the H-layer) is only reached
    // when Hadamards dominate, so bias the mix hard toward h.
    for (let seed = 0; seed < 60; seed++) {
      const rand = makePrng(seed * 7919 + 3)
      const n = 4
      const ops: Gate[] = []
      for (let i = 0; i < 30; i++) {
        if (rand() < 0.65) ops.push({ g: 'h', a: Math.floor(rand() * n) })
        else if (rand() < 0.5) ops.push({ g: 's', a: Math.floor(rand() * n) })
        else {
          const a = Math.floor(rand() * n)
          let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
          ops.push({ g: 'cx', a, b })
        }
      }
      expectVec(onCH(n, ops).toStatevector(), dense(onCircuit(n, ops)), `hh seed=${seed}`)
    }
  })

  it('preserves norm under random Clifford evolution', () => {
    for (let seed = 0; seed < 25; seed++) {
      const rand = makePrng(seed * 104729 + 11)
      const st = onCH(5, randomClifford(5, 60, rand))
      const norm = st.toStatevector().reduce((acc, z) => acc + z.re * z.re + z.im * z.im, 0)
      expect(norm).toBeCloseTo(1, 10)
    }
  })
})

describe('StabilizerCH — clone', () => {
  it('is independent of its source', () => {
    const a = new StabilizerCH(3).h(0).cx(0, 1)
    const b = a.clone()
    b.h(2).s(2)
    expectVec(a.toStatevector(), onCH(3, [{ g: 'h', a: 0 }, { g: 'cx', a: 0, b: 1 }]).toStatevector(), 'clone src')
    expect(b.amplitude('001')).not.toEqual(a.amplitude('001'))
  })
})

describe('StabilizerCH — sampling', () => {
  it('reproduces |⟨x|φ⟩|² for a random Clifford state', () => {
    const rand = makePrng(20260811)
    const n = 4
    const st = onCH(n, randomClifford(n, 40, makePrng(5)))
    const want = st.toStatevector().map(z => z.re * z.re + z.im * z.im)

    const shots = 40000
    const counts = new Float64Array(1 << n)
    for (let i = 0; i < shots; i++) {
      const x = st.sample(rand)
      let idx = 0
      for (let j = 0; j < n; j++) idx |= (x[j] ?? 0) << j
      counts[idx]!++
    }
    for (let i = 0; i < 1 << n; i++) {
      expect(counts[i]! / shots, `outcome ${i}`).toBeCloseTo(want[i]!, 1)
    }
  })

  it('agrees with amplitude() at scale, beyond any statevector', () => {
    // On its support a stabilizer state is flat: P(x) = 2^-|v| for every x with
    // non-zero amplitude (Section 4.1). That pins sample() against amplitude()
    // at sizes where no dense vector exists, so it is the only cross-check left.
    for (const n of [40, 60, 120]) {
      const rand = makePrng(n * 31 + 99)
      const st = new StabilizerCH(n)
      for (let i = 0; i < 12 * n; i++) {
        const r = rand()
        if (r < 0.34) st.h(Math.floor(rand() * n))
        else if (r < 0.6) st.s(Math.floor(rand() * n))
        else {
          const a = Math.floor(rand() * n)
          let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
          st.cx(a, b)
        }
      }
      const seen = new Set<number>()
      for (let i = 0; i < 200; i++) {
        const z = st.amplitude(st.sample(rand))
        const p = z.re * z.re + z.im * z.im
        expect(p, `n=${n} sampled state must be in the support`).toBeGreaterThan(0)
        seen.add(Math.round(Math.log2(p)))
      }
      expect(seen.size, `n=${n} support must be flat`).toBe(1)
      const k = [...seen][0]!
      expect(Number.isInteger(k) && k < 0 && k >= -n).toBe(true)
    }
  })

  it('only ever returns states in the support', () => {
    const rand = makePrng(7)
    const st = new StabilizerCH(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    for (let i = 0; i < 500; i++) {
      const x = st.sample(rand)
      const all = x[0]! + x[1]! + x[2]! + x[3]!
      expect(all === 0 || all === 4).toBe(true)
    }
  })
})

describe('StabilizerCH — equatorial inner product (Lemma 3)', () => {
  /** ⟨φ|φ_A⟩ by definition: Σ_x conj(⟨x|φ⟩)·2^{-n/2}·i^{xAxᵀ}. */
  function brute(st: StabilizerCH, A: Uint8Array, n: number): Complex {
    const POW = [[1, 0], [0, 1], [-1, 0], [0, -1]] as const
    const sv = st.toStatevector()
    let re = 0, im = 0
    for (let mask = 0; mask < 1 << n; mask++) {
      let e = 0
      for (let a = 0; a < n; a++) {
        if (!((mask >> a) & 1)) continue
        e += A[a * n + a] ?? 0
        for (let b = a + 1; b < n; b++) if ((mask >> b) & 1) e += 2 * (A[a * n + b] ?? 0)
      }
      const [pr, pi] = POW[((e % 4) + 4) % 4]!
      const z = sv[mask]!
      // conj(z) · (pr + i·pi)
      re += z.re * pr + z.im * pi
      im += z.re * pi - z.im * pr
    }
    const k = 2 ** (-n / 2)
    return { re: re * k, im: im * k }
  }

  it('matches brute force on random Clifford states', () => {
    for (const n of [1, 2, 3, 4, 5]) {
      for (let seed = 0; seed < 30; seed++) {
        const rand = makePrng(seed * 6151 + n * 733 + 7)
        const st = onCH(n, randomClifford(n, 5 * n + 8, rand))
        const A = randomEquatorial(n, rand)
        const got = st.innerProductEquatorial(A)
        const want = brute(st, A, n)
        expect(got.re, `n=${n} seed=${seed} re`).toBeCloseTo(want.re, 9)
        expect(got.im, `n=${n} seed=${seed} im`).toBeCloseTo(want.im, 9)
      }
    }
  })

  it('tracks the global phase', () => {
    const rand = makePrng(4242)
    const n = 3
    const A = randomEquatorial(n, rand)
    const base = new StabilizerCH(n).h(0).cx(0, 1)
    const rotated = base.clone().scaleOmega(0, 1)          // multiply by i
    const a = base.innerProductEquatorial(A)
    const b = rotated.innerProductEquatorial(A)
    // ⟨i·φ|φ_A⟩ = -i·⟨φ|φ_A⟩
    expect(b.re).toBeCloseTo(a.im, 10)
    expect(b.im).toBeCloseTo(-a.re, 10)
  })

  it('rejects a wrongly sized matrix', () => {
    expect(() => new StabilizerCH(3).innerProductEquatorial(new Uint8Array(4))).toThrow(RangeError)
  })

  it('is polynomial — 200 qubits is instant', () => {
    const rand = makePrng(11)
    const n = 200
    const st = new StabilizerCH(n).h(0)
    for (let q = 0; q < n - 1; q++) st.cx(q, q + 1)
    const t0 = performance.now()
    const z = st.innerProductEquatorial(randomEquatorial(n, rand))
    expect(Number.isFinite(z.re) && Number.isFinite(z.im)).toBe(true)
    expect(performance.now() - t0).toBeLessThan(20_000)
  })
})

describe('StabilizerCH — incremental walker (Eq. 57)', () => {
  it('agrees with amplitude() at the seed point', () => {
    for (const n of [1, 2, 3, 4, 5]) {
      for (let seed = 0; seed < 20; seed++) {
        const rand = makePrng(seed * 977 + n * 61 + 3)
        const st = onCH(n, randomClifford(n, 5 * n + 8, rand))
        for (let mask = 0; mask < 1 << n; mask++) {
          const x = Uint8Array.from({ length: n }, (_, j) => (mask >> j) & 1)
          const w = st.walker(x)
          const want = st.amplitude(x), got = w.value()
          expect(got.re, `n=${n} seed=${seed} x=${mask} re`).toBeCloseTo(want.re, 10)
          expect(got.im, `n=${n} seed=${seed} x=${mask} im`).toBeCloseTo(want.im, 10)
        }
      }
    }
  })

  it('tracks a random single-bit walk exactly', () => {
    for (const n of [2, 3, 4, 5, 6]) {
      for (let seed = 0; seed < 20; seed++) {
        const rand = makePrng(seed * 3571 + n * 131 + 11)
        const st = onCH(n, randomClifford(n, 6 * n + 10, rand))
        const x = Uint8Array.from({ length: n }, () => (rand() < 0.5 ? 0 : 1))
        const w = st.walker(x)
        for (let step = 0; step < 60; step++) {
          const j = Math.floor(rand() * n)
          x[j] = (x[j]! ^ 1) as 0 | 1
          w.flip(j)
          const want = st.amplitude(x), got = w.value()
          expect(got.re, `n=${n} seed=${seed} step=${step} re`).toBeCloseTo(want.re, 10)
          expect(got.im, `n=${n} seed=${seed} step=${step} im`).toBeCloseTo(want.im, 10)
        }
      }
    }
  })

  it('flipping the same bit twice is the identity', () => {
    const rand = makePrng(808)
    const n = 5
    const st = onCH(n, randomClifford(n, 40, rand))
    const x = new Uint8Array(n)
    const w = st.walker(x)
    const before = w.value()
    w.flip(2); w.flip(2)
    const after = w.value()
    expect(after.re).toBeCloseTo(before.re, 12)
    expect(after.im).toBeCloseTo(before.im, 12)
  })

  it('rejects a wrongly sized seed', () => {
    expect(() => new StabilizerCH(3).walker(new Uint8Array(2))).toThrow(RangeError)
  })
})
