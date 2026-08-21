/**
 * Integration coverage for the worker path.
 *
 * The unit tests in `dense-parallel.test.ts` cover the property that makes
 * threading safe — that a kernel's work space can be split arbitrarily — but
 * they never start a thread. Nothing there would catch a broken barrier, a
 * mis-encoded op in the control block, or a worker that reads its slice bounds
 * wrong, and those are exactly the failures that produce a hang or a race rather
 * than an obviously wrong number.
 *
 * `Circuit.run({ workers })` cannot be exercised from here: it resolves the
 * worker relative to `import.meta.url`, which under vitest ends in `.ts`, so the
 * option is deliberately ignored and the serial path runs. These tests therefore
 * drive `DensePool` directly against the built worker, and skip when `dist` has
 * not been built — `npm run build` before `npm test` to include them, which is
 * the order `prepublishOnly` already uses.
 */
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it, afterAll } from 'vitest'
import {
  acquireDensePool, DensePool, PARALLEL_MIN_WORK, releaseDensePool, workerFileExists,
} from './dense-parallel.js'
import {
  denseSingle, denseCNOT, denseControlled, denseCsrSwap, denseCSwap, denseSWAP,
  denseToffoli, denseTwo, denseUnitary, denseZero, type DenseState,
} from './dense.js'
import { wt } from './worker-shim.js'
import * as G from './gates.js'
import type { Complex } from './complex.js'

const WORKER_URL = new URL('../dist/dense.worker.js', import.meta.url)
const BUILT = existsSync(fileURLToPath(WORKER_URL))
const CAN_RUN = BUILT && wt !== null

/** Wide enough that gates clear PARALLEL_MIN_WORK and actually reach the pool. */
const N = 18

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const WorkerClass = wt?.Worker as any

/**
 * Barrier timeout for tests. Far below the 5-minute production default: a bug
 * here shows up as a thread that never reports, and a test suite that hangs for
 * five minutes per case tells you far less than one that fails in fifteen
 * seconds with the diagnostic attached.
 */
const TEST_TIMEOUT_MS = 15_000

const pools: DensePool[] = []
function pool(count: number, timeoutMs = TEST_TIMEOUT_MS, url = WORKER_URL): DensePool {
  const p = new DensePool(count, url, WorkerClass, timeoutMs)
  pools.push(p)
  return p
}
afterAll(() => { for (const p of pools) { try { p.terminate() } catch { /* already down */ } } })

/** An n-qubit state on a SharedArrayBuffer, filled deterministically. */
function shared(n: number, seed = 3): DenseState {
  const d = denseZero(n, true)
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

const copyUnshared = (d: DenseState): DenseState => ({ n: d.n, data: d.data.slice() })

function expectBitIdentical(a: DenseState, b: DenseState, what: string): void {
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) {
      expect.fail(`${what}: slot ${i} differs — serial ${a.data[i]} vs parallel ${b.data[i]}`)
    }
  }
}

const U3 = Array.from({ length: 8 }, (_, r) =>
  Array.from({ length: 8 }, (_, c): Complex => {
    const th = (r * 13 + c * 7) * 0.3
    return (r === (c * 3 + 1) % 8) ? { re: Math.cos(th), im: Math.sin(th) } : { re: 0, im: 0 }
  }))

describe.skipIf(!CAN_RUN)('dense worker pool — integration (requires npm run build)', () => {
  /**
   * Every op the pool can dispatch, run on real threads and compared against the
   * serial kernel on an identical state. Bit-identity is the right bar: if two
   * work items ever touched the same amplitude, the threads would race and the
   * result would drift rather than merely round differently.
   */
  it.each([2, 3, 5])('applies every op kind identically across %i threads', (threads) => {
    const p = pool(threads)
    const ops: [string, (d: DenseState, exec: DensePool | null) => void][] = [
      ['single H',    (d, e) => e ? e.single(d, 4, G.H) : denseSingle(d, 4, G.H)],
      ['single Rx',   (d, e) => e ? e.single(d, 0, G.Rx(1.1)) : denseSingle(d, 0, G.Rx(1.1))],
      ['single hi-q', (d, e) => e ? e.single(d, N - 1, G.T) : denseSingle(d, N - 1, G.T)],
      ['cnot',        (d, e) => e ? e.cnot(d, 2, 9) : denseCNOT(d, 2, 9)],
      ['cnot rev',    (d, e) => e ? e.cnot(d, N - 1, 0) : denseCNOT(d, N - 1, 0)],
      ['swap',        (d, e) => e ? e.swap(d, 1, 7) : denseSWAP(d, 1, 7)],
      ['controlled',  (d, e) => e ? e.controlled(d, 3, 8, G.T) : denseControlled(d, 3, 8, G.T)],
      ['two',         (d, e) => e ? e.two(d, 5, 6, G.Xx(0.8)) : denseTwo(d, 5, 6, G.Xx(0.8))],
      ['two far',     (d, e) => e ? e.two(d, 0, N - 1, G.Ms(0.3, 1.1)) : denseTwo(d, 0, N - 1, G.Ms(0.3, 1.1))],
      ['toffoli',     (d, e) => e ? e.toffoli(d, 1, 4, 11) : denseToffoli(d, 1, 4, 11)],
      ['cswap',       (d, e) => e ? e.cswap(d, 2, 5, 12) : denseCSwap(d, 2, 5, 12)],
      ['csrswap',     (d, e) => e ? e.csrswap(d, 0, 3, 6) : denseCsrSwap(d, 0, 3, 6)],
      ['unitary x3',  (d, e) => e ? e.unitary(d, [2, 7, 13], U3) : denseUnitary(d, [2, 7, 13], U3)],
    ]
    for (const [name, apply] of ops) {
      const par = shared(N, name.length + threads)
      const ser = copyUnshared(par)
      apply(par, p)
      apply(ser, null)
      expectBitIdentical(ser, par, `${name} on ${threads} threads`)
    }
  })

  it('matches serial over a full multi-gate circuit', () => {
    const p = pool(4)
    const par = shared(N, 99)
    const ser = copyUnshared(par)
    const run = (d: DenseState, e: DensePool | null) => {
      for (let layer = 0; layer < 3; layer++) {
        for (let q = 0; q < N; q++) {
          const g = [G.H, G.T, G.Rz(0.4), G.Ry(-1.2)][(q + layer) % 4]!
          if (e) e.single(d, q, g); else denseSingle(d, q, g)
        }
        for (let q = 0; q + 1 < N; q++) {
          if (e) e.cnot(d, q, q + 1); else denseCNOT(d, q, q + 1)
        }
        for (let q = 0; q + 1 < N; q += 2) {
          if (e) e.two(d, q, q + 1, G.Zz(0.3)); else denseTwo(d, q, q + 1, G.Zz(0.3))
        }
      }
    }
    run(par, p)
    run(ser, null)
    expectBitIdentical(ser, par, 'multi-gate circuit')
  })

  it('follows a state across re-attach', () => {
    // The pool binds lazily and must notice when it is handed a different buffer,
    // which is what happens when a second run promotes its own state.
    const p = pool(3)
    for (const seed of [11, 22, 33]) {
      const par = shared(N, seed)
      const ser = copyUnshared(par)
      p.single(par, 6, G.H)
      denseSingle(ser, 6, G.H)
      expectBitIdentical(ser, par, `re-attach seed ${seed}`)
    }
  })

  it('survives detach and reuse', () => {
    const p = pool(3)
    const d1 = shared(N, 44)
    p.single(d1, 2, G.H)
    p.detach()
    p.detach()                                   // idempotent
    const d2 = shared(N, 55)
    const ref = copyUnshared(d2)
    p.single(d2, 2, G.H)
    denseSingle(ref, 2, G.H)
    expectBitIdentical(ref, d2, 'after detach/reattach')
  })

  it('falls back to serial for a state on an ordinary ArrayBuffer', () => {
    // A run that did not ask for workers can still meet a pool-backed executor;
    // an unshared buffer must run in-thread rather than throwing.
    const p = pool(2)
    const plain = denseZero(N)                   // not shared
    for (let q = 0; q < N; q++) denseSingle(plain, q, G.H)
    const ref = copyUnshared(plain)
    expect(() => p.single(plain, 3, G.T)).not.toThrow()
    denseSingle(ref, 3, G.T)
    expectBitIdentical(ref, plain, 'unshared fallback')
  })

  it('falls back to serial below the parallel work threshold', () => {
    // Small states must bypass the barrier entirely — correctness here is the
    // same check, but the point is that it does not deadlock waiting on workers
    // that were never told to do anything.
    const small = 8
    expect((1 << small) >>> 1).toBeLessThan(PARALLEL_MIN_WORK)
    const p = pool(3)
    const d = shared(small, 66)
    const ref = copyUnshared(d)
    p.single(d, 1, G.H)
    p.cnot(d, 0, 2)
    denseSingle(ref, 1, G.H)
    denseCNOT(ref, 0, 2)
    expectBitIdentical(ref, d, 'sub-threshold fallback')
  })

  it('reports a diagnostic instead of hanging when a worker cannot start', () => {
    // A worker that dies at boot never increments the done counter, so the
    // barrier would otherwise sit for its full timeout and then say nothing
    // useful. Shortened here because that is the only way to observe it.
    const missing = new URL('../dist/dense.worker.does-not-exist.js', import.meta.url)
    const p = pool(2, 750, missing)
    const d = shared(N, 77)
    expect(() => p.single(d, 4, G.H)).toThrow(/\[ket\] dense workers/)
  }, 20_000)

  it('refuses further gates once its barrier has failed, without re-waiting', () => {
    // The threads are gone after a wedge, so a second gate can only reach the
    // same conclusion — but parking on the barrier to get there would cost
    // another full timeout, and in production that timeout is five minutes.
    const missing = new URL('../dist/dense.worker.does-not-exist.js', import.meta.url)
    const p = pool(2, 750, missing)
    const d = shared(N, 78)
    expect(() => p.single(d, 4, G.H)).toThrow(/\[ket\] dense workers/)
    expect(p.dead).toBe(true)

    const started = Date.now()
    expect(() => p.single(d, 4, G.H)).toThrow(/torn down after an earlier failure/)
    expect(Date.now() - started, 'second gate paid the barrier timeout again').toBeLessThan(250)
  }, 20_000)

  it('rebuilds rather than handing back a pool whose threads are gone', () => {
    // The process-wide cache keys on size and url alone. Without a liveness
    // test it would re-serve a wedged pool to every later run, so one failure
    // would poison `workers` for the life of the process.
    const first = acquireDensePool(3, WORKER_URL)
    expect(first).not.toBeNull()
    expect(acquireDensePool(3, WORKER_URL), 'a live pool is reused').toBe(first)

    first!.terminate()                    // what `#wedged` does after a failure
    const second = acquireDensePool(3, WORKER_URL)
    expect(second).not.toBe(first)
    expect(second!.dead).toBe(false)

    // And the replacement genuinely works, rather than merely being a new object.
    const d = shared(N, 88)
    const ref = copyUnshared(d)
    second!.single(d, 5, G.H)
    denseSingle(ref, 5, G.H)
    expectBitIdentical(ref, d, 'rebuilt pool')
    releaseDensePool()
  }, 20_000)

  it('declines a worker bundle that is not there instead of timing out on it', () => {
    // A boot failure cannot be seen from a thread parked in `Atomics.wait`, so
    // without this probe a missing side-car worker costs a full barrier timeout
    // per run. Declining up front is the quiet serial fallback instead.
    expect(workerFileExists(WORKER_URL)).toBe(true)
    const missing = new URL('../dist/dense.worker.does-not-exist.js', import.meta.url)
    expect(workerFileExists(missing)).toBe(false)

    const started = Date.now()
    expect(acquireDensePool(2, missing)).toBeNull()
    expect(Date.now() - started).toBeLessThan(250)
  })
})

