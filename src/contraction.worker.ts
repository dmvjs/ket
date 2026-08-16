/**
 * Contraction-sampling worker.
 *
 * Each worker draws a contiguous range of shots and returns the outcome counts.
 * Shots seed their own stream from a global index, so a range yields exactly the
 * shots the single-threaded run would have produced and the merged result does
 * not depend on how many workers ran.
 *
 * Protocol matches the MPS worker: post the result, then set the shared flag and
 * notify, so the message is queued before the main thread wakes from
 * `Atomics.wait` and can dequeue it with `receiveMessageOnPort`.
 */
import { parentPort } from 'node:worker_threads'
import { Circuit, type CircuitJSON } from './circuit.js'
import { sampleShotRange } from './tensor-network.js'

/** Serialized job sent from the main thread to a worker. */
export interface ContractionJob {
  circuit:   CircuitJSON
  seed:      number
  blockSize: number
  restarts:  number
  lo:        number
  hi:        number
  flag:      Int32Array
  port:      MessagePort
}

parentPort!.on('message', ({ circuit, seed, blockSize, restarts, lo, hi, flag, port }: ContractionJob) => {
  const sampled = sampleShotRange(Circuit.fromJSON(circuit), { seed, blockSize, restarts }, lo, hi)
  port.postMessage({
    counts: [...sampled.counts],
    width: sampled.width,
    contractions: sampled.contractions,
  })
  Atomics.store(flag, 0, 1)
  Atomics.notify(flag, 0)
})
