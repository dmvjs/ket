import { describe, expect, it } from 'vitest'
import { cmatrix, conjTranspose, matmul, svd, type CMatrix } from './linalg.js'

const rng = (seed: number): (() => number) => {
  let s = (seed >>> 0) || 1
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x100000000 - 0.5 }
}

const random = (rows: number, cols: number, seed: number): CMatrix => {
  const m = cmatrix(rows, cols)
  const r = rng(seed)
  for (let i = 0; i < rows * cols * 2; i++) m.data[i] = r()
  return m
}

/** Rebuild U·diag(s)·V† and compare entry by entry. */
const reconstruct = (u: CMatrix, s: Float64Array, vt: CMatrix): CMatrix => {
  const us = cmatrix(u.rows, u.cols)
  for (let i = 0; i < u.rows; i++) for (let k = 0; k < u.cols; k++) {
    us.data[(i * u.cols + k) * 2] = u.data[(i * u.cols + k) * 2]! * s[k]!
    us.data[(i * u.cols + k) * 2 + 1] = u.data[(i * u.cols + k) * 2 + 1]! * s[k]!
  }
  return matmul(us, vt)
}

const maxDiff = (a: CMatrix, b: CMatrix): number => {
  let w = 0
  for (let i = 0; i < a.data.length; i++) w = Math.max(w, Math.abs(a.data[i]! - b.data[i]!))
  return w
}

describe('svd — reconstructs what it decomposed', () => {
  it('at full rank, for every shape including non-square', () => {
    for (const [rows, cols] of [[4, 4], [8, 3], [3, 8], [1, 5], [6, 1], [12, 9]] as const) {
      const m = random(rows, cols, rows * 31 + cols)
      const r = svd(m)
      expect(maxDiff(reconstruct(r.u, r.s, r.vt), m), `${rows}x${cols}`).toBeLessThan(1e-10)
    }
  })

  it('singular values come back descending and non-negative', () => {
    const r = svd(random(9, 6, 7))
    for (let k = 1; k < r.rank; k++) expect(r.s[k]!).toBeLessThanOrEqual(r.s[k - 1]!)
    for (let k = 0; k < r.rank; k++) expect(r.s[k]!).toBeGreaterThanOrEqual(0)
  })

  it('the singular vectors are orthonormal', () => {
    for (const [rows, cols] of [[8, 5], [5, 8]] as const) {
      const r = svd(random(rows, cols, rows + cols))
      const uu = matmul(conjTranspose(r.u), r.u)
      const vv = matmul(r.vt, conjTranspose(r.vt))
      for (const g of [uu, vv]) {
        for (let i = 0; i < g.rows; i++) for (let j = 0; j < g.cols; j++) {
          const re = g.data[(i * g.cols + j) * 2]!, im = g.data[(i * g.cols + j) * 2 + 1]!
          expect(re, `${rows}x${cols} [${i}][${j}] re`).toBeCloseTo(i === j ? 1 : 0, 9)
          expect(im, `${rows}x${cols} [${i}][${j}] im`).toBeCloseTo(0, 9)
        }
      }
    }
  })

  it('handles a rank-deficient matrix without inventing rank', () => {
    // Two identical columns: the true rank is below the column count.
    const m = random(6, 4, 21)
    for (let i = 0; i < 6; i++) {
      m.data[(i * 4 + 3) * 2] = m.data[(i * 4 + 1) * 2]!
      m.data[(i * 4 + 3) * 2 + 1] = m.data[(i * 4 + 1) * 2 + 1]!
    }
    const r = svd(m)
    expect(r.rank).toBeLessThan(4)
    expect(maxDiff(reconstruct(r.u, r.s, r.vt), m)).toBeLessThan(1e-9)
  })
})

describe('svd — truncation is optimal and reports its cost', () => {
  it('discarded equals the dropped singular values squared', () => {
    const m = random(10, 7, 99)
    const full = svd(m)
    for (const k of [1, 3, 5]) {
      const cut = svd(m, { maxRank: k })
      let expected = 0
      for (let i = k; i < full.rank; i++) expected += full.s[i]! ** 2
      expect(cut.rank).toBe(k)
      expect(cut.discarded).toBeCloseTo(expected, 8)
    }
  })

  it('the truncation error in the matrix matches the discarded weight', () => {
    // Eckart–Young: the Frobenius error of the best rank-k approximation is
    // exactly the root of the summed squares of the dropped singular values.
    const m = random(9, 6, 5)
    for (const k of [1, 2, 4]) {
      const cut = svd(m, { maxRank: k })
      const approx = reconstruct(cut.u, cut.s, cut.vt)
      let err = 0
      for (let i = 0; i < m.data.length; i += 2) {
        err += (m.data[i]! - approx.data[i]!) ** 2 + (m.data[i + 1]! - approx.data[i + 1]!) ** 2
      }
      expect(Math.sqrt(err), `k=${k}`).toBeCloseTo(Math.sqrt(cut.discarded), 8)
    }
  })

  it('keeping every value discards nothing', () => {
    const r = svd(random(7, 5, 3), { maxRank: 100 })
    expect(r.discarded).toBeLessThan(1e-20)
  })
})

describe('matmul and conjTranspose', () => {
  it('conjTranspose twice is the identity', () => {
    const m = random(5, 3, 11)
    expect(maxDiff(conjTranspose(conjTranspose(m)), m)).toBe(0)
  })

  it('matmul agrees with a direct sum', () => {
    const a = random(4, 3, 1), b = random(3, 5, 2)
    const c = matmul(a, b)
    for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) {
      let re = 0, im = 0
      for (let k = 0; k < 3; k++) {
        const ar = a.data[(i * 3 + k) * 2]!, ai = a.data[(i * 3 + k) * 2 + 1]!
        const br = b.data[(k * 5 + j) * 2]!, bi = b.data[(k * 5 + j) * 2 + 1]!
        re += ar * br - ai * bi
        im += ar * bi + ai * br
      }
      expect(c.data[(i * 5 + j) * 2]!).toBeCloseTo(re, 12)
      expect(c.data[(i * 5 + j) * 2 + 1]!).toBeCloseTo(im, 12)
    }
  })

  it('rejects a shape mismatch', () => {
    expect(() => matmul(random(3, 4, 1), random(5, 2, 2))).toThrow(RangeError)
  })
})
