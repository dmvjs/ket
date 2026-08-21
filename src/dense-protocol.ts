/**
 * Wire format shared by the dense worker pool and its workers.
 *
 * Kept in its own module because both sides import it and neither may drag the
 * other in: the worker must not pull in `node:worker_threads` consumers from the
 * main-thread pool, and the pool must not pull in the worker entry point.
 *
 * Everything crossing the boundary per gate lives in two `SharedArrayBuffer`s —
 * a small `Int32Array` of control words and a `Float64Array` of gate
 * coefficients — so a gate dispatch costs a handful of atomic stores rather than
 * a structured clone. `postMessage` is used exactly once per state, to hand over
 * the buffers.
 */

/** Index of each control word in the shared `Int32Array`. */
export const CTRL = {
  /** Bumped by main to publish a new op; workers park on this. */
  GEN: 0,
  /** Workers increment on completion; main waits for it to reach the count. */
  DONE: 1,
  /** Which {@link OP} is published. */
  OPCODE: 2,
  ARG0: 3,
  ARG1: 4,
  ARG2: 5,
  /** Qubit count for {@link OP.UNITARY}. */
  NQUBITS: 6,
  /** Start of the qubit list for {@link OP.UNITARY}. */
  QUBITS: 7,
} as const

/**
 * Largest `unitary` this pool will split across threads.
 *
 * A k-qubit unitary has 2^(n−k) work items and a 4^k coefficient matrix, so the
 * arithmetic per amplitude grows as 2^k while the parallel work shrinks. Past 5
 * qubits the matrix alone is 1024 complex entries and the op is better run on
 * one thread than marshalled to many; `dense-parallel.ts` falls back to serial.
 */
export const MAX_PARALLEL_UNITARY_QUBITS = 5

/** Control words to allocate: the fixed header plus room for a qubit list. */
export const CTRL_WORDS = CTRL.QUBITS + MAX_PARALLEL_UNITARY_QUBITS

/** Coefficient slots to allocate: a 2^5 × 2^5 complex matrix, as re/im pairs. */
export const PARAM_SLOTS = (1 << MAX_PARALLEL_UNITARY_QUBITS) ** 2 * 2

/** Operation codes published in {@link CTRL.OPCODE}. */
export const OP = {
  /** Leave the barrier loop and release the buffers. */
  DETACH: 0,
  SINGLE: 1,
  CONTROLLED: 2,
  CNOT: 3,
  SWAP: 4,
  TOFFOLI: 5,
  CSWAP: 6,
  CSRSWAP: 7,
  TWO: 8,
  UNITARY: 9,
} as const

/** Any published operation code. */
export type Opcode = (typeof OP)[keyof typeof OP]

/**
 * Ops whose arity is fixed by the opcode alone.
 *
 * `DETACH` carries no work and `UNITARY` is sized per call, so both are excluded
 * — which is what makes {@link OP_ARITY} exhaustively checkable.
 */
export type FixedArityOpcode = Exclude<Opcode, typeof OP.DETACH | typeof OP.UNITARY>

/**
 * Qubits each op acts on — the divisor for its work space, `2ⁿ >>> k`.
 *
 * Single source of truth, and it has to be: both sides compute their slice from
 * this, so a disagreement would have them dividing work spaces of *different
 * sizes*. The slices would still tile something, just not the same thing, and
 * the result is amplitudes silently skipped or gates applied twice — on one op,
 * with no error and nothing downstream able to notice.
 *
 * Typed over {@link FixedArityOpcode} rather than `number` so that adding an
 * opcode without adding its arity is a compile error. Left as `Record<number,
 * number>` it would be a lookup returning `undefined`, and `2ⁿ >>> undefined`
 * is `2ⁿ` — the whole space, silently unsliced.
 */
export const OP_ARITY: Readonly<Record<FixedArityOpcode, number>> = {
  [OP.SINGLE]:     1,
  [OP.CONTROLLED]: 2,
  [OP.CNOT]:       2,
  [OP.SWAP]:       2,
  [OP.TWO]:        2,
  [OP.TOFFOLI]:    3,
  [OP.CSWAP]:      3,
  [OP.CSRSWAP]:    3,
}

/** Handed to a worker once per state, carrying the buffers it will share. */
export interface AttachMessage {
  ctrlSab:  SharedArrayBuffer
  paramSab: SharedArrayBuffer
  dataSab:  SharedArrayBuffer
  n:        number
}

/**
 * Slice `index` of `count` over a work space of `total` items.
 *
 * Uses the exact `floor(total · i / count)` boundary rather than a rounded chunk
 * size so the slices tile [0, total) with no gap and no overlap for every
 * (total, count) pair, including counts that do not divide the total and totals
 * smaller than the count. Any amplitude missed here would silently not have its
 * gate applied.
 */
export function sliceOf(total: number, index: number, count: number): [number, number] {
  const lo = Math.floor((total * index) / count)
  const hi = Math.floor((total * (index + 1)) / count)
  return [lo, hi]
}

/** Complex matrix stored as flat re/im pairs, row-major — the shared layout. */
type FlatComplex = { re: number; im: number }

/** Write a dim × dim complex matrix into the shared coefficient buffer. */
export function gateToShared(
  params: Float64Array, matrix: readonly (readonly FlatComplex[])[], dim: number,
): void {
  for (let r = 0; r < dim; r++) {
    const row = matrix[r]!
    for (let c = 0; c < dim; c++) {
      const v = row[c]!
      params[(r * dim + c) * 2]     = v.re
      params[(r * dim + c) * 2 + 1] = v.im
    }
  }
}

/** Read back a dim × dim complex matrix from the shared coefficient buffer. */
export function gateFromShared(params: Float64Array, dim: number): FlatComplex[][] {
  const out: FlatComplex[][] = []
  for (let r = 0; r < dim; r++) {
    const row: FlatComplex[] = []
    for (let c = 0; c < dim; c++) {
      row.push({ re: params[(r * dim + c) * 2]!, im: params[(r * dim + c) * 2 + 1]! })
    }
    out.push(row)
  }
  return out
}
