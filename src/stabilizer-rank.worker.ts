/**
 * Stabilizer-rank worker.
 *
 * Each worker owns a contiguous slice of the 2^t branch-index space. Because the
 * decomposition tree is deterministic, a slice is rebuilt from the op list and
 * its index bounds alone — no stabilizer state ever crosses the boundary, and
 * workers never talk to each other.
 *
 * Amplitudes are linear in the terms, so the main thread reduces by adding the
 * per-slice partial sums. Probabilities are *not* linear, so the reduce has to
 * happen on amplitudes before any modulus is taken.
 *
 * Protocol, mirroring `mps.worker.ts`:
 *   1. Main posts a job carrying a SharedArrayBuffer-backed flag and a port.
 *   2. Worker does the work and posts its result through the port.
 *   3. Worker sets flag → 1 and notifies.
 *   4. Main wakes from Atomics.wait and dequeues with receiveMessageOnPort.
 * Posting before notifying is what guarantees the message is already queued when
 * the main thread wakes.
 */
import { parentPort } from 'node:worker_threads'
import { buildSlice, type SrOp, type SliceTerm } from './stabilizer-rank.js'

/** Build this worker's slice of the decomposition. */
export interface SrBuildJob {
  kind: 'build'
  n:    number
  ops:  SrOp[]
  lo:   number
  hi:   number
  flag: Int32Array
  port: MessagePort
}

/**
 * Evaluate partial amplitudes for a batch of basis states, packed n bits each.
 * Carries no port: the one transferred with the build job is retained, since a
 * transferred port is detached on the sender and cannot be sent twice.
 */
export interface SrAmpJob {
  kind:  'amp'
  basis: Uint8Array
  count: number
  flag:  Int32Array
}

/**
 * Release the retained port and slice.
 *
 * Required, not optional: a MessagePort held in module scope keeps the worker's
 * event loop alive, which keeps the whole process alive. Without this a script
 * that uses `workers` never exits.
 */
export interface SrCloseJob { kind: 'close' }

export type SrJob = SrBuildJob | SrAmpJob | SrCloseJob

let terms: SliceTerm[] = []
let nQubits = 0
let out: MessagePort | null = null

parentPort!.on('message', (job: SrJob) => {
  if (job.kind === 'close') {
    out?.close()
    out = null
    terms = []
    return
  }
  if (job.kind === 'build') {
    nQubits = job.n
    out = job.port
    terms = buildSlice(job.n, job.ops, job.lo, job.hi)
    out.postMessage({ built: terms.length })
  } else {
    const { basis, count } = job
    const re = new Float64Array(count)
    const im = new Float64Array(count)
    const bits = new Uint8Array(nQubits)
    for (let i = 0; i < count; i++) {
      bits.set(basis.subarray(i * nQubits, (i + 1) * nQubits))
      let ar = 0, ai = 0
      for (const term of terms) {
        const z = term.state.amplitude(bits)
        ar += term.re * z.re - term.im * z.im
        ai += term.re * z.im + term.im * z.re
      }
      re[i] = ar
      im[i] = ai
    }
    out!.postMessage({ re, im }, [re.buffer, im.buffer])
  }
  Atomics.store(job.flag, 0, 1)
  Atomics.notify(job.flag, 0)
})
