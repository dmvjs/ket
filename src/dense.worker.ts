/**
 * Dense statevector worker.
 *
 * Unlike the MPS and stabilizer-rank workers, which each run a whole independent
 * slice of shots and report a result, this worker shares one statevector with
 * every other thread and applies part of each gate to it. There is no result to
 * send back: the state *is* the shared buffer, and the work is done when the
 * buffer has been written.
 *
 * That makes the loop below a barrier rather than a job queue. Per gate:
 *
 *   1. Main writes the op into the control block and bumps GEN.
 *   2. Every worker wakes from `Atomics.wait` on GEN, applies its slice of the
 *      op's work space, and increments DONE.
 *   3. Main waits until DONE reaches the worker count, then moves to the next
 *      gate.
 *
 * Gates must be separated by a barrier because consecutive gates act on
 * different qubits and therefore partition the state differently — worker 0's
 * amplitudes for gate k are not its amplitudes for gate k+1. Within a single
 * gate no barrier is needed at all: the work spaces in `dense.ts` are built so
 * that distinct work items touch disjoint amplitudes, so the threads never race
 * and never need a lock.
 *
 * `Atomics.wait` parks the thread rather than spinning, so an idle pool costs no
 * CPU, and a worker blocked in it is still killed by `terminate()`.
 */
import { parentPort, workerData } from 'node:worker_threads'
import {
  denseCNOT, denseControlled, denseCsrSwap, denseCSwap, denseSingle, denseSWAP,
  denseToffoli, denseTwo, denseUnitary, denseWork, type DenseState,
} from './dense.js'
import {
  CTRL, OP, OP_ARITY, sliceOf, gateFromShared,
  type AttachMessage, type FixedArityOpcode,
} from './dense-protocol.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'

const { id, count } = workerData as { id: number; count: number }

parentPort!.on('message', (msg: AttachMessage) => {
  const ctrl = new Int32Array(msg.ctrlSab)
  const params = new Float64Array(msg.paramSab)
  const d: DenseState = { n: msg.n, data: new Float64Array(msg.dataSab) }

  // Adopt the generation counter as it stands *before* acknowledging.
  //
  // GEN is never reset — it keeps climbing for the life of the pool, across
  // every attach. Starting from 0 here would mean that on any attach after the
  // first, the wait below sees GEN already ahead and returns immediately, and
  // the worker would then read whatever opcode the *previous* cycle left in
  // place. That is always DETACH, since detaching is how a cycle ends, so the
  // worker would fall straight out of the loop and silently stop serving gates
  // while main waited out the barrier timeout on every one.
  //
  // Reading before the acknowledgement is what makes this safe: main does not
  // publish anything until every worker has acked, so GEN cannot move between
  // this load and the wait.
  let gen = Atomics.load(ctrl, CTRL.GEN)

  // Signal that this worker has the buffers and is entering the barrier loop.
  Atomics.add(ctrl, CTRL.DONE, 1)
  Atomics.notify(ctrl, CTRL.DONE)

  for (;;) {
    // Park until main publishes a new generation. Returns 'not-equal'
    // immediately if main already bumped GEN, so a wakeup is never lost.
    Atomics.wait(ctrl, CTRL.GEN, gen)
    const next = Atomics.load(ctrl, CTRL.GEN)
    // A wake without a new generation is not ours to act on. Re-running the last
    // op would apply it twice, which no later check could detect.
    if (next === gen) continue
    gen = next

    const opcode = Atomics.load(ctrl, CTRL.OPCODE)
    if (opcode === OP.DETACH) break

    applySlice(d, ctrl, params)

    Atomics.add(ctrl, CTRL.DONE, 1)
    Atomics.notify(ctrl, CTRL.DONE)
  }

  // Acknowledge the detach so main knows the buffers are released, then go back
  // to waiting for the next attach. The worker outlives the state it served.
  Atomics.add(ctrl, CTRL.DONE, 1)
  Atomics.notify(ctrl, CTRL.DONE)
})

/** Apply this worker's share of the op currently published in the control block. */
function applySlice(d: DenseState, ctrl: Int32Array, params: Float64Array): void {
  const opcode = Atomics.load(ctrl, CTRL.OPCODE)
  const a0 = Atomics.load(ctrl, CTRL.ARG0)
  const a1 = Atomics.load(ctrl, CTRL.ARG1)
  const a2 = Atomics.load(ctrl, CTRL.ARG2)

  // Arity from the shared table, never re-derived here — main slices the same
  // work space, and the two must agree exactly. See OP_ARITY.
  const k = opcode === OP.UNITARY
    ? Atomics.load(ctrl, CTRL.NQUBITS)
    : OP_ARITY[opcode as FixedArityOpcode]
  // Main thread takes slice 0, so worker `id` takes slice `id + 1` of `count`.
  const [lo, hi] = sliceOf(denseWork(d.n, k), id + 1, count)
  if (lo >= hi) return

  switch (opcode) {
    case OP.SINGLE:     denseSingle(d, a0, gateFromShared(params, 2) as Gate2x2, lo, hi); break
    case OP.CONTROLLED: denseControlled(d, a0, a1, gateFromShared(params, 2) as Gate2x2, lo, hi); break
    case OP.CNOT:       denseCNOT(d, a0, a1, lo, hi); break
    case OP.SWAP:       denseSWAP(d, a0, a1, lo, hi); break
    case OP.TOFFOLI:    denseToffoli(d, a0, a1, a2, lo, hi); break
    case OP.CSWAP:      denseCSwap(d, a0, a1, a2, lo, hi); break
    case OP.CSRSWAP:    denseCsrSwap(d, a0, a1, a2, lo, hi); break
    case OP.TWO:        denseTwo(d, a0, a1, gateFromShared(params, 4) as Gate4x4, lo, hi); break
    case OP.UNITARY: {
      const qs: number[] = []
      for (let i = 0; i < k; i++) qs.push(Atomics.load(ctrl, CTRL.QUBITS + i))
      denseUnitary(d, qs, gateFromShared(params, 1 << k), lo, hi)
      break
    }
    default: break
  }
}
