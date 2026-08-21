/**
 * Multi-threaded dense statevector execution.
 *
 * A general 2×2 butterfly is 28 flops (16 multiplies, 12 adds) against 64 bytes
 * of traffic — two amplitudes read and written. Holding the memory pattern fixed
 * and varying only the body measures where the time actually goes, and it splits
 * almost evenly: at n = 22…26, a 0-flop pair exchange takes 61% of the shipped
 * kernel's time, so ~40% is arithmetic and ~60% is memory.
 *
 * That balance is why this module exists and why nothing else worked. Sitting at
 * the ridge means neither resource has slack to trade into the other: removing
 * arithmetic caps out at 1.67x (the 0-flop floor), and *adding* it to buy back
 * memory passes — gate fusion — measured 0.36–0.54x, a 2-3x loss, because V8
 * will not keep a fused block in registers.
 *
 * Adding cores is the one move that lifts both limits at once, since each thread
 * brings its own ALUs and its own share of bandwidth. Hence near-linear scaling
 * where a purely bandwidth-bound kernel would have saturated early:
 *
 *     1 thread    36 GB/s
 *     2 threads   68 GB/s
 *     4 threads  114 GB/s
 *     8 threads  184 GB/s
 *
 * End to end on a 218-gate circuit at n = 26 that is 13.4s → 1.5s, 9.0x.
 *
 * This is safe to do without any locking because of how `dense.ts` decomposes
 * work: each kernel walks a flat work space [0, W) in which distinct items touch
 * disjoint amplitudes. Threads therefore never write the same slot, and the
 * result of a split run is bit-identical to a serial one — not merely close.
 *
 * Threads must still rendezvous *between* gates, since consecutive gates
 * partition the state differently. That barrier is the one cost this design
 * pays, and it is why small states stay serial: see {@link PARALLEL_MIN_WORK}.
 */
import {
  denseCNOT, denseControlled, denseCsrSwap, denseCSwap, denseSingle, denseSWAP,
  denseToffoli, denseTwo, denseUnitary, denseWork,
  type DenseExec, type DenseState,
} from './dense.js'
import {
  CTRL, CTRL_WORDS, gateToShared, MAX_PARALLEL_UNITARY_QUBITS, OP, OP_ARITY, PARAM_SLOTS, sliceOf,
  type AttachMessage, type FixedArityOpcode, type Opcode,
} from './dense-protocol.js'
import type { Complex } from './complex.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'
import { wt } from './worker-shim.js'

/**
 * Work items below which a gate runs serially regardless of the pool.
 *
 * A barrier round trip costs tens of microseconds. One work item moves 64 bytes,
 * so 2^16 items is ~4 MiB of traffic — around 100µs on one core, comfortably
 * above the barrier and the point where splitting starts to pay. Below it the
 * rendezvous costs more than the gate.
 *
 * In qubit terms this keeps single-qubit gates serial below n = 17 and leaves
 * them parallel above it, which matches where the measured crossover sits.
 */
export const PARALLEL_MIN_WORK = 1 << 16

/** Milliseconds to wait for a worker before declaring the pool wedged. */
export const BARRIER_TIMEOUT_MS = 300_000

/** Options accepted by `node:worker_threads`.Worker that this pool relies on. */
type WorkerCtor = new (u: URL, o?: { execArgv?: string[]; workerData?: unknown }) => unknown

/**
 * A set of worker threads sharing one statevector.
 *
 * Workers outlive the state they operate on: {@link attach} hands them a new
 * buffer, {@link detach} releases it, and the threads persist in between so a
 * second run does not pay ~20ms of spawn latency per thread.
 */
export class DensePool implements DenseExec {
  // Worker instances are `any`: the protocol they honour is enforced by the
  // control block, not by a structural type the host can check.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly #workers: any[]
  readonly #ctrl:    Int32Array
  readonly #params:  Float64Array
  readonly #ctrlSab: SharedArrayBuffer
  readonly #paramSab: SharedArrayBuffer
  /** Participants in a barrier: the workers plus the calling thread. */
  readonly #parts:   number
  /** The buffer the workers currently hold, or null when detached. */
  #bound: SharedArrayBuffer | null = null
  /**
   * Set once the threads are gone — after a failed barrier or a {@link terminate}.
   *
   * A pool that missed a rendezvous cannot be trusted again: its workers have
   * been torn down and the shared control block is left mid-generation, so every
   * later gate would park for a further {@link BARRIER_TIMEOUT_MS} waiting on
   * threads that no longer exist. Recording the death is what lets the pool say
   * so at once, and lets {@link acquireDensePool} build a replacement instead of
   * handing the next run the corpse.
   */
  #dead = false
  /**
   * How long a barrier waits before reporting the pool wedged.
   *
   * Injectable only so the failure path can be covered: a worker that dies at
   * boot never increments DONE, so the only way to observe {@link #wedged} in a
   * test is to shorten the wait from five minutes.
   */
  readonly #timeoutMs: number

  constructor(workerCount: number, workerUrl: URL, WorkerClass: WorkerCtor, timeoutMs = BARRIER_TIMEOUT_MS) {
    this.#timeoutMs = timeoutMs
    this.#ctrlSab  = new SharedArrayBuffer(CTRL_WORDS * 4)
    this.#paramSab = new SharedArrayBuffer(PARAM_SLOTS * 8)
    this.#ctrl     = new Int32Array(this.#ctrlSab)
    this.#params   = new Float64Array(this.#paramSab)
    this.#parts    = workerCount + 1
    this.#workers  = Array.from({ length: workerCount }, (_, id) => {
      // execArgv: [] for the same reason the MPS pool does it — a worker
      // inheriting the parent's CLI flags can die at boot on flags that are
      // invalid for a file-backed worker, surfacing as a silent timeout.
      const w = new WorkerClass(workerUrl, { execArgv: [], workerData: { id, count: workerCount + 1 } }) as any
      // An unreffed worker still runs and still answers; it just stops voting on
      // when the process may exit, so a persistent pool cannot strand the host.
      w.unref()
      // Without these an early exit is silent and the barrier below just waits.
      w.on('error', (e: Error) => { w.__ketError = e })
      w.on('exit',  (code: number) => { w.__ketExit = code })
      return w
    })
  }

  /**
   * Hand the workers a state to operate on. Blocks until all have it.
   *
   * Called lazily by `#bind` rather than by the caller: a state only becomes
   * dense partway through a circuit, so there is no moment beforehand at which
   * the buffer to attach exists.
   */
  attach(d: DenseState): void {
    const buffer = d.data.buffer
    if (!(buffer instanceof SharedArrayBuffer)) {
      throw new TypeError('DensePool.attach: state must be backed by a SharedArrayBuffer')
    }
    this.detach()
    Atomics.store(this.#ctrl, CTRL.DONE, 0)
    const msg: AttachMessage = {
      ctrlSab:  this.#ctrlSab,
      paramSab: this.#paramSab,
      dataSab:  buffer,
      n:        d.n,
    }
    for (const w of this.#workers) w.postMessage(msg)
    this.#awaitDone('attach')
    this.#bound = buffer
  }

  /** True once the threads are gone. A dead pool serves no further gates. */
  get dead(): boolean { return this.#dead }

  /** Release the state. Workers stay alive, ready for the next {@link attach}. */
  detach(): void {
    if (this.#dead || this.#bound === null) return
    this.#publish(OP.DETACH, [])
    this.#bound = null
    this.#awaitDone('detach')
  }

  /** Stop the threads for good. Safe on a pool that already failed. */
  terminate(): void {
    // Teardown must not throw. `detach` publishes an op and waits for an answer,
    // which is precisely what a pool with dying threads cannot give — and the
    // threads are going away either way, so a barrier failure here is moot.
    try { this.detach() } catch { /* torn down regardless */ }
    this.#dead  = true
    this.#bound = null
    for (const w of this.#workers) void w.terminate()
  }

  // ── DenseExec ───────────────────────────────────────────────────────────────
  //
  // Each method stages any gate coefficients, then hands `#dispatch` the opcode,
  // the qubits, and a closure applying the kernel over a range. That closure is
  // the whole of the serial path too: a gate that should not be split is the
  // same closure invoked over the full work space on this thread. There is no
  // second code path to keep in agreement with the first.

  single(d: DenseState, q: number, gate: Gate2x2): void {
    gateToShared(this.#params, gate, 2)
    this.#dispatch(d, OP.SINGLE, [q], (lo, hi) => denseSingle(d, q, gate, lo, hi))
  }

  controlled(d: DenseState, control: number, target: number, gate: Gate2x2): void {
    if (control === target) throw new TypeError(`control and target qubits must differ (got ${control})`)
    gateToShared(this.#params, gate, 2)
    this.#dispatch(d, OP.CONTROLLED, [control, target],
      (lo, hi) => denseControlled(d, control, target, gate, lo, hi))
  }

  cnot(d: DenseState, control: number, target: number): void {
    this.#dispatch(d, OP.CNOT, [control, target], (lo, hi) => denseCNOT(d, control, target, lo, hi))
  }

  swap(d: DenseState, a: number, b: number): void {
    this.#dispatch(d, OP.SWAP, [a, b], (lo, hi) => denseSWAP(d, a, b, lo, hi))
  }

  toffoli(d: DenseState, c1: number, c2: number, target: number): void {
    this.#dispatch(d, OP.TOFFOLI, [c1, c2, target], (lo, hi) => denseToffoli(d, c1, c2, target, lo, hi))
  }

  cswap(d: DenseState, control: number, a: number, b: number): void {
    this.#dispatch(d, OP.CSWAP, [control, a, b], (lo, hi) => denseCSwap(d, control, a, b, lo, hi))
  }

  csrswap(d: DenseState, control: number, a: number, b: number): void {
    this.#dispatch(d, OP.CSRSWAP, [control, a, b], (lo, hi) => denseCsrSwap(d, control, a, b, lo, hi))
  }

  two(d: DenseState, a: number, b: number, gate: Gate4x4): void {
    gateToShared(this.#params, gate, 4)
    this.#dispatch(d, OP.TWO, [a, b], (lo, hi) => denseTwo(d, a, b, gate, lo, hi))
  }

  unitary(d: DenseState, qs: readonly number[], matrix: readonly (readonly Complex[])[]): void {
    const k = qs.length
    // A wide unitary carries a 4^k coefficient matrix and leaves only 2^(n-k)
    // work items, so past the cap the marshalling outweighs the split — and
    // staging it would overrun the shared coefficient buffer, which is sized for
    // exactly this many qubits.
    if (k > MAX_PARALLEL_UNITARY_QUBITS) return denseUnitary(d, qs, matrix)
    gateToShared(this.#params, matrix, 1 << k)
    Atomics.store(this.#ctrl, CTRL.NQUBITS, k)
    for (let i = 0; i < k; i++) Atomics.store(this.#ctrl, CTRL.QUBITS + i, qs[i]!)
    this.#dispatch(d, OP.UNITARY, qs, (lo, hi) => denseUnitary(d, qs, matrix, lo, hi))
  }

  // ── barrier ─────────────────────────────────────────────────────────────────

  /**
   * Run one op, split across the pool when that is worth doing.
   *
   * Arity comes from {@link OP_ARITY} rather than the call site, so main and the
   * workers derive their slice bounds from the same table — see the note there
   * for what disagreeing would cost.
   *
   * Falls back to running `own` over the whole space on this thread when the
   * gate is too small to repay the barrier, or when the state sits on an
   * ordinary `ArrayBuffer` — which happens whenever a run that did not ask for
   * workers borrows a pool-backed executor.
   */
  #dispatch(
    d: DenseState, opcode: Opcode, qubits: readonly number[],
    own: (lo: number, hi: number) => void,
  ): void {
    // A pool whose threads are gone cannot serve this one either. Refusing now
    // is the whole point of tracking the death: parking on the barrier would
    // cost another full timeout to reach the same conclusion.
    if (this.#dead) {
      throw new Error(
        '[ket] dense workers: pool was torn down after an earlier failure and serves no further gates. ' +
        'Re-run without the workers option to use the single-threaded path.',
      )
    }
    const k = opcode === OP.UNITARY ? qubits.length : OP_ARITY[opcode as FixedArityOpcode]
    const total = denseWork(d.n, k)
    if (total < PARALLEL_MIN_WORK || !this.#bind(d)) {
      own(0, total)
      return
    }
    this.#publish(opcode, qubits)
    const [lo, hi] = sliceOf(total, 0, this.#parts)
    if (lo < hi) own(lo, hi)
    this.#awaitDone('gate')
  }

  /**
   * Ensure the workers hold this state's buffer, attaching if not.
   *
   * Returns false for an unshared buffer, which the workers cannot see at all.
   * Binding lazily is what lets the pool follow a state that only became dense
   * partway through a circuit — the promotion point is decided by fill, so there
   * is no moment beforehand at which the buffer to attach exists.
   */
  #bind(d: DenseState): boolean {
    const buffer = d.data.buffer
    if (!(buffer instanceof SharedArrayBuffer)) return false
    if (this.#bound !== buffer) this.attach(d)
    return true
  }

  /** Publish an op and wake the workers on it. */
  #publish(opcode: Opcode, qubits: readonly number[]): void {
    const ctrl = this.#ctrl
    // DONE must be cleared before GEN is bumped, or a worker that wakes and
    // finishes immediately would have its increment overwritten by the reset.
    Atomics.store(ctrl, CTRL.DONE, 0)
    Atomics.store(ctrl, CTRL.OPCODE, opcode)
    Atomics.store(ctrl, CTRL.ARG0, qubits[0] ?? 0)
    Atomics.store(ctrl, CTRL.ARG1, qubits[1] ?? 0)
    Atomics.store(ctrl, CTRL.ARG2, qubits[2] ?? 0)
    Atomics.add(ctrl, CTRL.GEN, 1)
    Atomics.notify(ctrl, CTRL.GEN)
  }

  /** Block until every worker has reported in for the current generation. */
  #awaitDone(what: string): void {
    const ctrl = this.#ctrl
    const target = this.#workers.length
    const deadline = Date.now() + this.#timeoutMs
    for (;;) {
      const done = Atomics.load(ctrl, CTRL.DONE)
      if (done >= target) return
      const remaining = deadline - Date.now()
      if (remaining <= 0) throw this.#wedged(what, done, target)
      // Returns 'not-equal' at once if DONE moved between the load and here, so
      // a worker finishing in that window cannot be missed.
      Atomics.wait(ctrl, CTRL.DONE, done, remaining)
    }
  }

  /** Explain a worker that never reported. Its `error` event cannot be
   *  delivered while this thread is parked in `Atomics.wait`, so read whatever
   *  the listener managed to record before falling back to the generic case. */
  #wedged(what: string, done: number, target: number): Error {
    const failed = this.#workers.find(w => w.__ketError)
    for (const w of this.#workers) void w.terminate()
    // Mark the death before returning the error. The throw unwinds past every
    // caller that could have cleaned up, so unless the pool disowns itself here
    // the process-wide cache keeps serving it — see `acquireDensePool`.
    this.#dead  = true
    this.#bound = null
    if (failed) {
      return new Error(`[ket] dense workers: ${what} failed — ${(failed.__ketError as Error).message}`)
    }
    return new Error(
      `[ket] dense workers: ${what} — only ${done} of ${target} workers responded within ${(this.#timeoutMs / 1000).toFixed(0)}s. ` +
      `Re-run without the workers option to use the single-threaded path.`,
    )
  }
}

// ── pool lifecycle ────────────────────────────────────────────────────────────

/**
 * One persistent pool, rebuilt only when the requested size changes.
 *
 * Thread spawn is ~20ms each, which would otherwise be charged to every run.
 */
let _pool: { pool: DensePool; size: number; url: string } | null = null

/**
 * Get a pool of `size` workers, or null if this host cannot run them.
 *
 * Returns null rather than throwing so callers degrade to the serial path: a
 * browser has no `worker_threads`, and running from TypeScript sources means the
 * bundled worker file does not exist yet.
 */
export function acquireDensePool(size: number, workerUrl: URL | null): DensePool | null {
  if (wt === null || workerUrl === null || size < 1) return null
  if (!workerFileExists(workerUrl)) return null
  const href = workerUrl.href
  // Evict a pool whose threads are gone before the size/url match can hand it
  // back. Reusing it would give the next run a corpse that re-pays the barrier
  // timeout on its first gate, and every run after that the same again.
  if (_pool?.pool.dead === true) _pool = null
  if (_pool !== null && _pool.size === size && _pool.url === href) return _pool.pool
  _pool?.pool.terminate()
  _pool = { pool: new DensePool(size, workerUrl, wt.Worker as unknown as WorkerCtor), size, url: href }
  return _pool.pool
}

/** Drop the pool so its threads stop. Exposed for tests and shutdown. */
export function releaseDensePool(): void {
  _pool?.pool.terminate()
  _pool = null
}

/**
 * Whether a worker bundle is actually present at `url`.
 *
 * A worker file that is not there dies at boot, and that death is invisible to a
 * thread parked in `Atomics.wait` — the event loop cannot run to deliver it — so
 * the pool learns of it only by timing out, five minutes later, on a question
 * one stat call answers now. Checking up front turns the most likely failure
 * (a bundle shipped without its side-car workers) back into the quiet serial
 * fallback the caller's warning already describes.
 *
 * Read through `getBuiltinModule` for the reasons given in `worker-shim.ts`: no
 * top-level await, and nothing requested on a host that cannot serve it. An
 * unanswerable question means yes — a false negative would disable threads on a
 * host that could have run them, which is the worse way to be wrong.
 */
export function workerFileExists(url: URL): boolean {
  if (typeof process === 'undefined') return true
  const load = (process as unknown as { getBuiltinModule?: (id: string) => unknown }).getBuiltinModule
  if (typeof load !== 'function') return true
  try {
    const fs = load.call(process, 'node:fs') as { existsSync?: (p: URL) => boolean }
    return fs.existsSync?.(url) ?? true
  } catch {
    return true
  }
}

/**
 * Cores this host will admit to, or 0 if it will not say.
 *
 * Read through `getBuiltinModule` for the reasons given in `worker-shim.ts`: no
 * top-level await, and nothing requested on a host that cannot serve it.
 */
export function availableParallelism(): number {
  if (typeof process === 'undefined') return 0
  const load = (process as unknown as { getBuiltinModule?: (id: string) => unknown }).getBuiltinModule
  if (typeof load !== 'function') return 0
  try {
    const os = load.call(process, 'node:os') as { availableParallelism?: () => number; cpus?: () => unknown[] }
    return os.availableParallelism?.() ?? os.cpus?.().length ?? 0
  } catch {
    return 0
  }
}

/**
 * Threads to use for `workers: n`, clamped to what the machine can serve.
 *
 * Returns the count of *worker* threads. The calling thread takes a slice too,
 * so `workers: 8` spawns 7 and runs 8-wide — asking for more threads than there
 * are cores only adds barrier participants that contend for the same memory
 * controllers, which measured strictly slower.
 */
export function resolveWorkerCount(requested: number, cpus = availableParallelism()): number {
  if (!Number.isInteger(requested) || requested < 1) {
    throw new RangeError(`workers must be a positive integer (got ${requested})`)
  }
  const cap = cpus > 0 ? Math.min(requested, cpus) : requested
  return Math.max(0, cap - 1)
}
