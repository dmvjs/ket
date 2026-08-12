import { describe, it, expect } from 'vitest'
import { expSum, dyadic } from './exp-sum.js'
import { makePrng } from './prng.js'

/** Z(B) by definition: Σ_x i^{xBxᵀ}, enumerating all 2^m assignments. */
function brute(B: Uint8Array, m: number): { re: number; im: number } {
  const POW = [[1, 0], [0, 1], [-1, 0], [0, -1]] as const
  let re = 0, im = 0
  for (let mask = 0; mask < 1 << m; mask++) {
    let e = 0
    for (let a = 0; a < m; a++) {
      if (!((mask >> a) & 1)) continue
      e += B[a * m + a] ?? 0
      for (let b = a + 1; b < m; b++) if ((mask >> b) & 1) e += 2 * (B[a * m + b] ?? 0)
    }
    const [pr, pi] = POW[((e % 4) + 4) % 4]!
    re += pr; im += pi
  }
  return { re, im }
}

/** Random symmetric B: diagonal in Z₄, off-diagonal in {0,1}. */
function randomB(m: number, rand: () => number): Uint8Array {
  const B = new Uint8Array(m * m)
  for (let i = 0; i < m; i++) {
    B[i * m + i] = Math.floor(rand() * 4)
    for (let j = i + 1; j < m; j++) {
      const v = rand() < 0.5 ? 0 : 1
      B[i * m + j] = v
      B[j * m + i] = v
    }
  }
  return B
}

const val = (B: Uint8Array, m: number): { re: number; im: number } => {
  const z = expSum(B, m)
  return { re: dyadic(z.re), im: dyadic(z.im) }
}

describe('expSum — against the definition', () => {
  it('handles the empty form', () => {
    expect(val(new Uint8Array(0), 0)).toEqual({ re: 1, im: 0 })
  })

  it('matches brute force on every single-variable form', () => {
    // Z = 1 + i^d for d ∈ Z₄: 2, 1+i, 0, 1-i.
    for (const [d, want] of [[0, { re: 2, im: 0 }], [1, { re: 1, im: 1 }],
                             [2, { re: 0, im: 0 }], [3, { re: 1, im: -1 }]] as const) {
      expect(val(Uint8Array.from([d]), 1), `d=${d}`).toEqual(want)
    }
  })

  it('matches brute force on random forms, m = 1…9', () => {
    for (let m = 1; m <= 9; m++) {
      for (let seed = 0; seed < 60; seed++) {
        const rand = makePrng(m * 7919 + seed * 31 + 1)
        const B = randomB(m, rand)
        const got = val(B, m), want = brute(B, m)
        expect(got.re, `m=${m} seed=${seed} re`).toBeCloseTo(want.re, 9)
        expect(got.im, `m=${m} seed=${seed} im`).toBeCloseTo(want.im, 9)
      }
    }
  })

  it('matches brute force on purely diagonal forms', () => {
    for (let m = 1; m <= 8; m++) {
      for (let seed = 0; seed < 20; seed++) {
        const rand = makePrng(seed * 104729 + m)
        const B = new Uint8Array(m * m)
        for (let i = 0; i < m; i++) B[i * m + i] = Math.floor(rand() * 4)
        const got = val(B, m), want = brute(B, m)
        expect(got.re, `diag m=${m} seed=${seed} re`).toBeCloseTo(want.re, 9)
        expect(got.im, `diag m=${m} seed=${seed} im`).toBeCloseTo(want.im, 9)
      }
    }
  })

  it('matches brute force on purely off-diagonal forms', () => {
    for (let m = 2; m <= 8; m++) {
      for (let seed = 0; seed < 20; seed++) {
        const rand = makePrng(seed * 2654435761 + m * 17)
        const B = randomB(m, rand)
        for (let i = 0; i < m; i++) B[i * m + i] = 0
        const got = val(B, m), want = brute(B, m)
        expect(got.re, `offdiag m=${m} seed=${seed} re`).toBeCloseTo(want.re, 9)
        expect(got.im, `offdiag m=${m} seed=${seed} im`).toBeCloseTo(want.im, 9)
      }
    }
  })

  it('stays exact where a float64 would overflow', () => {
    // Z(0) = 2^m. At m = 200 that is far past MAX_SAFE_INTEGER, so the dyadic
    // exponent must survive symbolically rather than be collapsed.
    const m = 200
    const z = expSum(new Uint8Array(m * m), m)
    expect(z.re).toEqual({ sign: 1, exp: m })
    expect(z.im).toEqual({ sign: 0, exp: 0 })
  })

  it('is O(m³), not O(2^m)', () => {
    const rand = makePrng(5)
    const m = 300
    const t0 = performance.now()
    expSum(randomB(m, rand), m)
    expect(performance.now() - t0).toBeLessThan(10_000)
  })
})
