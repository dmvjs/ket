import { describe, expect, it } from 'vitest'
import {
  denseCNOT, denseControlled, denseCsrSwap, denseCSwap, denseSingle, denseSWAP,
  denseToffoli, denseTwo, denseUnitary, denseWork, denseZero,
  DENSE_QUBIT_LIMIT, type DenseState,
} from './dense.js'
import { resolveWorkerCount } from './dense-parallel.js'
import { sliceOf } from './dense-protocol.js'
import * as G from './gates.js'
import type { Complex } from './complex.js'

/** Deterministic non-trivial state — every amplitude distinct and non-zero. */
function fixture(n: number, seed = 1): DenseState {
  const d = denseZero(n)
  let s = seed >>> 0 || 1
  const r = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 0x100000000 }
  let acc = 0
  for (let i = 0; i < (1 << n); i++) {
    const re = r() * 2 - 1, im = r() * 2 - 1
    d.data[i << 1] = re; d.data[(i << 1) | 1] = im
    acc += re * re + im * im
  }
  const f = 1 / Math.sqrt(acc)
  for (let i = 0; i < d.data.length; i++) d.data[i]! *= f
  return d
}

const clone = (d: DenseState): DenseState => ({ n: d.n, data: d.data.slice() })

/** Every slot identical — not merely close. Parallelism must not perturb a bit. */
function expectBitIdentical(a: DenseState, b: DenseState, what: string): void {
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) {
      expect.fail(`${what}: slot ${i} differs — ${a.data[i]} vs ${b.data[i]}`)
    }
  }
}

/**
 * Splitting a kernel's work space and applying the pieces must equal applying
 * the whole range.
 *
 * This is the property the worker pool rests on. Threads write the same buffer
 * with no locking, so if two work items could ever touch the same amplitude the
 * result would be a data race — silent, load-dependent, and invisible to a
 * tolerance-based comparison. Checking uneven splits in a shuffled order is what
 * makes a missed or double-applied amplitude fail here rather than in
 * production, and bit-identity is the right bar: floating-point addition is not
 * associative, so anything short of identical would mean the work items overlap.
 */
function expectRangeAdditive(
  n: number, k: number, apply: (d: DenseState, lo: number, hi: number) => void, what: string,
): void {
  const total = denseWork(n, k)
  const base = fixture(n, n * 7 + k)

  const whole = clone(base)
  apply(whole, 0, total)

  for (const parts of [2, 3, 5, 8, total + 2]) {
    const split = clone(base)
    // Apply the slices out of order: concurrent threads finish in any order, so
    // the result must not depend on sequence.
    const order = Array.from({ length: parts }, (_, i) => i).reverse()
    for (const i of order) {
      const [lo, hi] = sliceOf(total, i, parts)
      if (lo < hi) apply(split, lo, hi)
    }
    expectBitIdentical(whole, split, `${what} split into ${parts}`)
  }
}

describe('dense-parallel — work-space decomposition', () => {
  describe('sliceOf tiles the work space exactly', () => {
    it('covers [0, total) with no gap and no overlap', () => {
      for (const total of [0, 1, 2, 3, 7, 16, 100, 1023, 4096]) {
        for (const count of [1, 2, 3, 4, 5, 8, 13, 16]) {
          let cursor = 0
          for (let i = 0; i < count; i++) {
            const [lo, hi] = sliceOf(total, i, count)
            expect(lo, `total=${total} count=${count} slice=${i} start`).toBe(cursor)
            expect(hi, `total=${total} count=${count} slice=${i} end`).toBeGreaterThanOrEqual(lo)
            cursor = hi
          }
          expect(cursor, `total=${total} count=${count} covers everything`).toBe(total)
        }
      }
    })

    it('gives every item to exactly one slice', () => {
      const total = 257, count = 7
      const owner = new Int32Array(total).fill(-1)
      for (let i = 0; i < count; i++) {
        const [lo, hi] = sliceOf(total, i, count)
        for (let p = lo; p < hi; p++) {
          expect(owner[p], `item ${p} claimed twice`).toBe(-1)
          owner[p] = i
        }
      }
      expect([...owner].every(o => o >= 0), 'every item claimed').toBe(true)
    })
  })

  describe('kernels are range-additive', () => {
    const n = 7
    const U3 = Array.from({ length: 8 }, (_, r) =>
      Array.from({ length: 8 }, (_, c): Complex => {
        // A permutation-with-phases matrix: unitary by construction.
        const th = (r * 13 + c * 7) * 0.3
        return (r === (c * 3 + 1) % 8) ? { re: Math.cos(th), im: Math.sin(th) } : { re: 0, im: 0 }
      }))

    it('denseSingle', () => {
      for (let q = 0; q < n; q++) {
        expectRangeAdditive(n, 1, (d, lo, hi) => denseSingle(d, q, G.H, lo, hi), `single q${q}`)
        expectRangeAdditive(n, 1, (d, lo, hi) => denseSingle(d, q, G.Rx(1.1), lo, hi), `single rx q${q}`)
      }
    })

    it('denseCNOT / denseSWAP', () => {
      for (const [a, b] of [[0, 1], [1, 0], [0, 6], [6, 0], [2, 5]]) {
        expectRangeAdditive(n, 2, (d, lo, hi) => denseCNOT(d, a!, b!, lo, hi), `cnot ${a}->${b}`)
        expectRangeAdditive(n, 2, (d, lo, hi) => denseSWAP(d, a!, b!, lo, hi), `swap ${a},${b}`)
      }
    })

    it('denseControlled / denseTwo', () => {
      for (const [a, b] of [[0, 1], [3, 0], [1, 6]]) {
        expectRangeAdditive(n, 2, (d, lo, hi) => denseControlled(d, a!, b!, G.T, lo, hi), `cu ${a}->${b}`)
        expectRangeAdditive(n, 2, (d, lo, hi) => denseTwo(d, a!, b!, G.Xx(0.8), lo, hi), `two ${a},${b}`)
      }
    })

    it('denseToffoli / denseCSwap / denseCsrSwap', () => {
      for (const [a, b, c] of [[0, 1, 2], [2, 0, 5], [6, 3, 1]]) {
        expectRangeAdditive(n, 3, (d, lo, hi) => denseToffoli(d, a!, b!, c!, lo, hi), `ccx ${a},${b},${c}`)
        expectRangeAdditive(n, 3, (d, lo, hi) => denseCSwap(d, a!, b!, c!, lo, hi), `cswap ${a},${b},${c}`)
        expectRangeAdditive(n, 3, (d, lo, hi) => denseCsrSwap(d, a!, b!, c!, lo, hi), `csrswap ${a},${b},${c}`)
      }
    })

    it('denseUnitary', () => {
      for (const qs of [[0, 1, 2], [4, 1, 6], [6, 5, 0]]) {
        expectRangeAdditive(n, 3, (d, lo, hi) => denseUnitary(d, qs, U3, lo, hi), `unitary ${qs}`)
      }
    })
  })

  describe('work-space sizing', () => {
    it('halves per acted-on qubit and never goes negative', () => {
      expect(denseWork(10, 1)).toBe(512)
      expect(denseWork(10, 2)).toBe(256)
      expect(denseWork(10, 3)).toBe(128)
      // A state narrower than the gate yields no work rather than a negative
      // shift count, which `1 << (n - k)` would produce.
      expect(denseWork(2, 3)).toBe(0)
      expect(denseWork(1, 3)).toBe(0)
      expect(denseWork(0, 1)).toBe(0)
    })

    it('stays positive at the architectural ceiling', () => {
      expect(denseWork(DENSE_QUBIT_LIMIT, 1)).toBeGreaterThan(0)
      expect(denseWork(DENSE_QUBIT_LIMIT, 2)).toBeGreaterThan(0)
    })
  })

  describe('resolveWorkerCount', () => {
    it('reserves one slice for the calling thread', () => {
      expect(resolveWorkerCount(8, 16)).toBe(7)
      expect(resolveWorkerCount(1, 16)).toBe(0)
    })

    it('clamps to the core count', () => {
      expect(resolveWorkerCount(64, 8)).toBe(7)
    })

    it('rejects nonsense', () => {
      expect(() => resolveWorkerCount(0, 8)).toThrow(/positive integer/)
      expect(() => resolveWorkerCount(-2, 8)).toThrow(/positive integer/)
      expect(() => resolveWorkerCount(2.5, 8)).toThrow(/positive integer/)
    })
  })
})
