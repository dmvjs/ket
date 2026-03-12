/**
 * MPS trajectory worker.
 *
 * Each worker independently runs a slice of noisy trajectory shots and
 * returns the bitstring count map to the main thread.
 *
 * Protocol:
 *   1. Main posts a WorkerJob that includes a SharedArrayBuffer-backed flag.
 *   2. Worker runs all shots, posts counts via parentPort.postMessage.
 *   3. Worker atomically sets flag → 1 and notifies.
 *   4. Main wakes from Atomics.wait, dequeues result with receiveMessageOnPort.
 *
 * The postMessage-before-notify ordering guarantees the message is in the
 * port queue by the time Atomics.wait returns on the main thread.
 */
import { parentPort } from 'node:worker_threads'
import { MpsTrajectory, applyTrajOps, type TrajOp } from './mps.js'
import * as G from './gates.js'

/** Serialized job sent from the main thread to a worker. */
export interface WorkerJob {
  ops:          TrajOp[]
  n:            number
  maxBond:      number
  truncErr:     number
  p1:           number
  p2:           number
  pMeas:        number
  shots:        number
  seed:         number
  initialState: string | undefined
  flag:         Int32Array    // SharedArrayBuffer-backed; notified when done
  port:         MessagePort   // transferred MessageChannel port for sending result
}

function makePrng(seed: number): () => number {
  let s = (seed >>> 0) || 1
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x100000000 }
}

parentPort!.on('message', ({
  ops, n, maxBond, truncErr, p1, p2, pMeas, shots, seed, initialState, flag, port,
}: WorkerJob) => {
  const rng  = makePrng(seed)
  const traj = new MpsTrajectory(n, maxBond, truncErr)
  const out  = new Map<bigint, number>()

  for (let i = 0; i < shots; i++) {
    traj.reset()
    if (initialState !== undefined) {
      for (let q = 0; q < n; q++) {
        if (initialState[q] === '1') traj.apply1(q, G.X)
      }
    }
    applyTrajOps(traj, ops, p1, p2, rng)
    let idx = traj.sample(rng)
    if (pMeas) {
      for (let q = 0; q < n; q++) {
        if (rng() < pMeas) idx ^= 1n << BigInt(q)
      }
    }
    out.set(idx, (out.get(idx) ?? 0) + 1)
  }

  // Send via the dedicated MessageChannel port, then signal the main thread.
  // postMessage-before-notify ordering ensures the message is in port1's queue
  // by the time Atomics.wait returns on the main thread.
  port.postMessage({ counts: [...out] as [bigint, number][] })
  Atomics.store(flag, 0, 1)
  Atomics.notify(flag, 0)
})
