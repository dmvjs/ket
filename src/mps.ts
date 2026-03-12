/**
 * Matrix Product State (MPS) tensor network simulator — Float64Array edition.
 *
 * Tensors are stored as flat Float64Arrays with interleaved re/im:
 *   T[l][p][r] at element index ((l * 2 + p) * chiR + r)
 *   real part:  data[((l * 2 + p) * chiR + r) * 2]
 *   imag part:  data[((l * 2 + p) * chiR + r) * 2 + 1]
 *
 * This eliminates all Complex object allocations in hot loops, reducing
 * GC pressure and improving cache locality by 3-8× over the previous
 * Complex[][][] layout.
 *
 * Handles 50+ qubit circuits with bounded entanglement (GHZ, BV, shallow QFT)
 * in O(n · chi² · 4) bytes — vs O(2ⁿ) for full statevector.
 */

import { type Complex, ONE, ZERO } from './complex.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * Rank-3 tensor stored as a flat Float64Array.
 * T[l][p][r] = data[((l*2+p)*chiR+r)*2] (re) and +1 (im).
 */
export type Tensor = {
  readonly data: Float64Array
  readonly chiL: number
  readonly chiR: number
}

/** Matrix Product State: one rank-3 tensor per qubit site. */
export type MPS = Tensor[]

// ── Constant gate matrices ────────────────────────────────────────────────────

export const CNOT4: Gate4x4 = [
  [ONE, ZERO, ZERO, ZERO],
  [ZERO, ONE, ZERO, ZERO],
  [ZERO, ZERO, ZERO, ONE],
  [ZERO, ZERO, ONE, ZERO],
]

export const SWAP4: Gate4x4 = [
  [ONE, ZERO, ZERO, ZERO],
  [ZERO, ZERO, ONE, ZERO],
  [ZERO, ONE, ZERO, ZERO],
  [ZERO, ZERO, ZERO, ONE],
]

/** Build a controlled-U gate4x4 (MSB = control). */
export function controlledGate([[a, b], [cc, d]]: Gate2x2): Gate4x4 {
  return [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, a, b],
    [ZERO, ZERO, cc, d],
  ]
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/** Create a new zero-filled tensor. */
function makeTensor(chiL: number, chiR: number): Tensor {
  return { data: new Float64Array(chiL * 2 * chiR * 2), chiL, chiR }
}

// ── MPS initialisation ────────────────────────────────────────────────────────

/** Return the |0...0⟩ MPS with bond dimension 1. */
export function mpsInit(n: number): MPS {
  return Array.from({ length: n }, () => {
    const t = makeTensor(1, 1)
    t.data[0] = 1 // T[0][0][0].re = 1 → |0⟩
    return t
  })
}

// ── Single-qubit gate ─────────────────────────────────────────────────────────

/** Apply a single-qubit gate to site q. O(chiL · chiR). */
export function mpsApply1(mps: MPS, q: number, [[a, b], [cc, d]]: Gate2x2): MPS {
  const { data, chiL, chiR } = mps[q]!
  const newData = new Float64Array(data.length)
  const are = a.re, aim = a.im
  const bre = b.re, bim = b.im
  const cre = cc.re, cim = cc.im
  const dre = d.re, dim = d.im

  for (let l = 0; l < chiL; l++) {
    for (let r = 0; r < chiR; r++) {
      const i0 = ((l * 2 + 0) * chiR + r) * 2
      const i1 = ((l * 2 + 1) * chiR + r) * 2
      const t0re = data[i0]!, t0im = data[i0 + 1]!
      const t1re = data[i1]!, t1im = data[i1 + 1]!

      // new[l][0][r] = a · T[l][0][r] + b · T[l][1][r]
      newData[i0]     = are * t0re - aim * t0im + bre * t1re - bim * t1im
      newData[i0 + 1] = are * t0im + aim * t0re + bre * t1im + bim * t1re

      // new[l][1][r] = c · T[l][0][r] + d · T[l][1][r]
      newData[i1]     = cre * t0re - cim * t0im + dre * t1re - dim * t1im
      newData[i1 + 1] = cre * t0im + cim * t0re + dre * t1im + dim * t1re
    }
  }

  return mps.with(q, { data: newData, chiL, chiR })
}

// ── QR decomposition (Modified Gram-Schmidt on Float64Array) ─────────────────

/**
 * QR decompose rows×cols complex matrix M (row-major Float64Array, interleaved re/im).
 * Returns Q (rows×bond row-major) and R (bond×cols row-major), both Float64Array,
 * where bond ≤ maxBond. The column-major Gram-Schmidt scans all cols for non-zero pivots.
 */
function qrMGS(
  mData: Float64Array,
  rows: number,
  cols: number,
  maxBond: number,
): { qData: Float64Array; rData: Float64Array; bond: number } {
  // Column-major working copy: aData[(j * rows + i) * 2 + (0|1)] = A[j][i]
  const aData = new Float64Array(cols * rows * 2)
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const src = (i * cols + j) * 2
      const dst = (j * rows + i) * 2
      aData[dst]     = mData[src]!
      aData[dst + 1] = mData[src + 1]!
    }
  }

  // Q column vectors (each length = rows, stored as Float64Array of rows*2 re/im pairs)
  const qCols: Float64Array[] = []

  for (let j = 0; j < cols && qCols.length < maxBond; j++) {
    // ‖A[j]‖²
    let n2 = 0
    for (let i = 0; i < rows; i++) {
      const re = aData[(j * rows + i) * 2]!, im = aData[(j * rows + i) * 2 + 1]!
      n2 += re * re + im * im
    }
    if (n2 < 1e-28) continue  // linearly dependent or zero — skip

    const inv = 1 / Math.sqrt(n2)
    const qj = new Float64Array(rows * 2)
    for (let i = 0; i < rows; i++) {
      qj[i * 2]     = aData[(j * rows + i) * 2]! * inv
      qj[i * 2 + 1] = aData[(j * rows + i) * 2 + 1]! * inv
    }
    qCols.push(qj)

    // Orthogonalise remaining columns against qj: A[l] -= (conj(qj)·A[l]) · qj
    for (let l = j + 1; l < cols; l++) {
      let dre = 0, dim = 0
      for (let i = 0; i < rows; i++) {
        const qre = qj[i * 2]!, qim = qj[i * 2 + 1]!
        const are = aData[(l * rows + i) * 2]!, aim = aData[(l * rows + i) * 2 + 1]!
        dre += qre * are + qim * aim   // Re(conj(q) · a)
        dim += qre * aim - qim * are   // Im(conj(q) · a)
      }
      for (let i = 0; i < rows; i++) {
        const qre = qj[i * 2]!, qim = qj[i * 2 + 1]!
        const idx = (l * rows + i) * 2
        // TypeScript strict: read then write (no compound assignment on indexed access)
        aData[idx]     = (aData[idx]     ?? 0) - (dre * qre - dim * qim)
        aData[idx + 1] = (aData[idx + 1] ?? 0) - (dre * qim + dim * qre)
      }
    }
  }

  const bond = qCols.length

  // Build Q in row-major layout: Q[i][k] = qCols[k][i]
  // qData[(i * bond + k) * 2] = re, +1 = im
  const qData = new Float64Array(rows * bond * 2)
  for (let k = 0; k < bond; k++) {
    const col = qCols[k]!
    for (let i = 0; i < rows; i++) {
      qData[(i * bond + k) * 2]     = col[i * 2]!
      qData[(i * bond + k) * 2 + 1] = col[i * 2 + 1]!
    }
  }

  // R = Q† · M: rData[(k * cols + j) * 2] = re, +1 = im
  const rData = new Float64Array(bond * cols * 2)
  for (let k = 0; k < bond; k++) {
    const qj = qCols[k]!
    for (let j = 0; j < cols; j++) {
      let rre = 0, rim = 0
      for (let i = 0; i < rows; i++) {
        const qre = qj[i * 2]!, qim = qj[i * 2 + 1]!
        const mre = mData[(i * cols + j) * 2]!, mim = mData[(i * cols + j) * 2 + 1]!
        rre += qre * mre + qim * mim   // Re(conj(q) · m)
        rim += qre * mim - qim * mre   // Im(conj(q) · m)
      }
      rData[(k * cols + j) * 2]     = rre
      rData[(k * cols + j) * 2 + 1] = rim
    }
  }

  return { qData, rData, bond }
}

// ── Two-qubit gate (adjacent) ─────────────────────────────────────────────────

/**
 * Apply a two-qubit gate to adjacent sites a and a+1.
 * Contracts A⊗B, applies gate, then QR-decomposes back to A'⊗B'.
 * Bond dimension is capped at maxBond.
 *
 * O(chiL · chiR · chiM) for contraction + O(chiL · chiR · bond) for QR.
 * All arithmetic runs inline on Float64Arrays — no Complex object allocations.
 */
function mpsApply2Adjacent(mps: MPS, a: number, gate: Gate4x4, maxBond: number): MPS {
  const b = a + 1
  const A = mps[a]!, B = mps[b]!
  const chiL = A.chiL, chiM = B.chiL, chiR = B.chiR

  const rows = chiL * 2, cols = 2 * chiR
  const mData = new Float64Array(rows * cols * 2)

  // Contract A⊗B and scatter through gate into M
  // M[la*2+pa2][pb2*chiR+rb] += gate[pa2*2+pb2][pa*2+pb] * sum_m A[la][pa][m]*B[m][pb][rb]
  for (let la = 0; la < chiL; la++) {
    for (let pa = 0; pa < 2; pa++) {
      for (let pb = 0; pb < 2; pb++) {
        for (let rb = 0; rb < chiR; rb++) {
          // theta = sum_m A[la][pa][m] * B[m][pb][rb]
          let tre = 0, tim = 0
          for (let m = 0; m < chiM; m++) {
            const ai = ((la * 2 + pa) * chiM + m) * 2
            const bi = ((m * 2 + pb) * chiR + rb) * 2
            const are = A.data[ai]!, aim = A.data[ai + 1]!
            const bre = B.data[bi]!, bim = B.data[bi + 1]!
            tre += are * bre - aim * bim
            tim += are * bim + aim * bre
          }
          // Scatter into M via gate
          for (let pa2 = 0; pa2 < 2; pa2++) {
            for (let pb2 = 0; pb2 < 2; pb2++) {
              const g = gate[pa2 * 2 + pb2]![pa * 2 + pb]!
              const mi = ((la * 2 + pa2) * cols + pb2 * chiR + rb) * 2
              mData[mi]     = (mData[mi]     ?? 0) + g.re * tre - g.im * tim
              mData[mi + 1] = (mData[mi + 1] ?? 0) + g.re * tim + g.im * tre
            }
          }
        }
      }
    }
  }

  const { qData, rData, bond } = qrMGS(mData, rows, cols, maxBond)

  // Q is rows×bond = (chiL*2)×bond → newA: chiL × 2 × bond
  // qData[(row * bond + k) * 2] where row = la*2+pa
  // matches newA.data[((la*2+pa) * bond + k) * 2] exactly → zero-copy reshape
  const newA: Tensor = { data: qData, chiL, chiR: bond }

  // R is bond×cols = bond×(2*chiR) → newB: bond × 2 × chiR
  // rData[(k * cols + pb*chiR + rb) * 2] where cols = 2*chiR
  // = rData[(k * 2*chiR + pb*chiR + rb) * 2]
  // matches newB.data[((k*2+pb)*chiR+rb)*2] = newB.data[(k*2*chiR+pb*chiR+rb)*2] exactly
  const newB: Tensor = { data: rData, chiL: bond, chiR }

  return mps.with(a, newA).with(b, newB)
}

// ── Two-qubit gate (arbitrary) ────────────────────────────────────────────────

/**
 * Apply a two-qubit gate to sites a and b (a < b).
 * Non-adjacent pairs are handled by a SWAP network.
 */
export function mpsApply2(mps: MPS, a: number, b: number, gate: Gate4x4, maxBond: number): MPS {
  if (b === a + 1) return mpsApply2Adjacent(mps, a, gate, maxBond)
  for (let q = b - 1; q > a; q--) mps = mpsApply2Adjacent(mps, q, SWAP4, maxBond)
  mps = mpsApply2Adjacent(mps, a, gate, maxBond)
  for (let q = a + 1; q < b; q++) mps = mpsApply2Adjacent(mps, q, SWAP4, maxBond)
  return mps
}

// ── Sampling ──────────────────────────────────────────────────────────────────

/**
 * Draw one sample from the MPS by sequential left-to-right marginal collapse.
 * Returns the sampled bitstring as a BigInt (qubit 0 = LSB).
 */
export function mpsSample(mps: MPS, rand: () => number): bigint {
  const n = mps.length
  // Left-boundary vector: starts as [1+0i] (scalar)
  let stateRe = new Float64Array([1])
  let stateIm = new Float64Array([1])  // will be set to [0] right away
  stateIm[0] = 0
  let result = 0n

  for (let q = 0; q < n; q++) {
    const { data, chiL, chiR } = mps[q]!

    // v[p][r] = sum_l state[l] * T[l][p][r]
    const v0re = new Float64Array(chiR)  // p=0 real
    const v0im = new Float64Array(chiR)  // p=0 imag
    const v1re = new Float64Array(chiR)  // p=1 real
    const v1im = new Float64Array(chiR)  // p=1 imag

    for (let l = 0; l < chiL; l++) {
      const sre = stateRe[l]!, sim = stateIm[l]!
      for (let r = 0; r < chiR; r++) {
        const i0 = ((l * 2 + 0) * chiR + r) * 2
        const i1 = ((l * 2 + 1) * chiR + r) * 2
        const t0re = data[i0]!, t0im = data[i0 + 1]!
        const t1re = data[i1]!, t1im = data[i1 + 1]!
        v0re[r] = (v0re[r] ?? 0) + sre * t0re - sim * t0im
        v0im[r] = (v0im[r] ?? 0) + sre * t0im + sim * t0re
        v1re[r] = (v1re[r] ?? 0) + sre * t1re - sim * t1im
        v1im[r] = (v1im[r] ?? 0) + sre * t1im + sim * t1re
      }
    }

    let prob0 = 0
    for (let r = 0; r < chiR; r++) prob0 += v0re[r]! * v0re[r]! + v0im[r]! * v0im[r]!
    let prob1 = 0
    for (let r = 0; r < chiR; r++) prob1 += v1re[r]! * v1re[r]! + v1im[r]! * v1im[r]!

    const total = prob0 + prob1
    const bit = total > 0 ? (rand() < prob0 / total ? 0 : 1) : 0
    if (bit === 1) result |= 1n << BigInt(q)

    const chosen = bit === 0 ? prob0 : prob1
    const inv = chosen > 0 ? 1 / Math.sqrt(chosen) : 0

    if (bit === 0) {
      stateRe = new Float64Array(chiR)
      stateIm = new Float64Array(chiR)
      for (let r = 0; r < chiR; r++) { stateRe[r] = v0re[r]! * inv; stateIm[r] = v0im[r]! * inv }
    } else {
      stateRe = new Float64Array(chiR)
      stateIm = new Float64Array(chiR)
      for (let r = 0; r < chiR; r++) { stateRe[r] = v1re[r]! * inv; stateIm[r] = v1im[r]! * inv }
    }
  }

  return result
}

// ── Mutable trajectory runner ─────────────────────────────────────────────────

/**
 * Mutable, pre-allocated MPS for quantum trajectory simulation.
 *
 * Allocates all workspace at construction time. apply1, apply2, reset,
 * and sample are zero-allocation in the hot path — no Float64Array is
 * created after the constructor returns.
 *
 * Usage: construct once, then call reset() + circuit ops + sample() per shot.
 */
export class MpsTrajectory {
  readonly n:      number
  readonly maxChi: number

  // Per-site tensor storage. data[q] holds up to maxChi × 2 × maxChi complex values.
  // Layout: data[q][((l*2+p)*chiR+r)*2] = re, +1 = im, where chiR = this.chiR[q].
  readonly data: Float64Array[]
  readonly chiL: Int32Array
  readonly chiR: Int32Array

  // Workspace for apply2Adjacent — sized for worst-case maxChi bonds.
  private readonly mBuf:    Float64Array  // contracted matrix, rows × cols
  private readonly aBuf:    Float64Array  // column-major copy of mBuf for MGS
  private readonly qColBuf: Float64Array  // MGS Q columns, packed: col k at k*rows
  private readonly qBuf:    Float64Array  // Q output, row-major: rows × bond
  private readonly rBuf:    Float64Array  // R output, row-major: bond × cols

  // Workspace for SVD (one-sided complex Jacobi).
  // vBuf: V matrix (cols × cols, column-major), cols = maxChi * 2.
  // sigmaBuf: singular values per column, length maxChi * 2.
  // orderBuf: column index permutation sorted by descending sigma, length maxChi * 2.
  private readonly vBuf:     Float64Array
  private readonly sigmaBuf: Float64Array
  private readonly orderBuf: Int32Array

  // Workspace for sample — one allocation per maxChi, reused across all qubits.
  private readonly sv0re: Float64Array
  private readonly sv0im: Float64Array
  private readonly sv1re: Float64Array
  private readonly sv1im: Float64Array
  private readonly stRe:  Float64Array
  private readonly stIm:  Float64Array

  constructor(n: number, maxChi: number) {
    this.n      = n
    this.maxChi = maxChi

    const maxRows = maxChi * 2
    const maxCols = maxChi * 2

    this.data = Array.from({ length: n }, () => new Float64Array(maxRows * maxCols * 2))
    this.chiL = new Int32Array(n)
    this.chiR = new Int32Array(n)

    this.mBuf    = new Float64Array(maxRows * maxCols * 2)
    this.aBuf    = new Float64Array(maxCols * maxRows * 2)
    this.qColBuf = new Float64Array(maxChi  * maxRows * 2)
    this.qBuf    = new Float64Array(maxRows * maxChi  * 2)
    this.rBuf    = new Float64Array(maxChi  * maxCols * 2)

    this.vBuf     = new Float64Array(maxCols * maxCols * 2)
    this.sigmaBuf = new Float64Array(maxCols)
    this.orderBuf = new Int32Array(maxCols)

    this.sv0re = new Float64Array(maxChi)
    this.sv0im = new Float64Array(maxChi)
    this.sv1re = new Float64Array(maxChi)
    this.sv1im = new Float64Array(maxChi)
    this.stRe  = new Float64Array(maxChi)
    this.stIm  = new Float64Array(maxChi)

    this.reset()
  }

  /** Reset to |0...0⟩. Sets all tensors to T[0][0][0]=1 with chiL=chiR=1. */
  reset(): void {
    for (let q = 0; q < this.n; q++) {
      this.data[q]!.fill(0)
      this.data[q]![0] = 1
      this.chiL[q] = 1
      this.chiR[q] = 1
    }
  }

  /** Apply a single-qubit gate in place. O(chiL · chiR). */
  apply1(q: number, [[a, b], [c, d]]: Gate2x2): void {
    const are = a.re, aim = a.im
    const bre = b.re, bim = b.im
    const cre = c.re, cim = c.im
    const dre = d.re, dim = d.im
    const data = this.data[q]!
    const chiL = this.chiL[q]!, chiR = this.chiR[q]!

    for (let l = 0; l < chiL; l++) {
      for (let r = 0; r < chiR; r++) {
        const i0 = ((l * 2 + 0) * chiR + r) * 2
        const i1 = ((l * 2 + 1) * chiR + r) * 2
        const t0re = data[i0]!, t0im = data[i0 + 1]!
        const t1re = data[i1]!, t1im = data[i1 + 1]!
        data[i0]     = are * t0re - aim * t0im + bre * t1re - bim * t1im
        data[i0 + 1] = are * t0im + aim * t0re + bre * t1im + bim * t1re
        data[i1]     = cre * t0re - cim * t0im + dre * t1re - dim * t1im
        data[i1 + 1] = cre * t0im + cim * t0re + dre * t1im + dim * t1re
      }
    }
  }

  /** Apply a two-qubit gate in place. Non-adjacent pairs handled via SWAP network. */
  apply2(a: number, b: number, gate: Gate4x4): void {
    if (b === a + 1) { this.apply2Adjacent(a, gate); return }
    for (let q = b - 1; q > a; q--) this.apply2Adjacent(q, SWAP4)
    this.apply2Adjacent(a, gate)
    for (let q = a + 1; q < b; q++) this.apply2Adjacent(q, SWAP4)
  }

  private apply2Adjacent(a: number, gate: Gate4x4): void {
    const b    = a + 1
    const chiL = this.chiL[a]!
    const chiM = this.chiR[a]!   // == this.chiL[b]
    const chiR = this.chiR[b]!
    const rows = chiL * 2
    const cols = chiR * 2
    const dA   = this.data[a]!
    const dB   = this.data[b]!
    const mBuf = this.mBuf

    // Clear the used region of mBuf before accumulating.
    const mSize = rows * cols * 2
    for (let i = 0; i < mSize; i++) mBuf[i] = 0

    // Contract dA ⊗ dB through gate into mBuf.
    // mBuf[(la*2+pa2) * cols + pb2*chiR + rb)[*2] += gate[pa2*2+pb2][pa*2+pb] * theta
    // where theta = sum_m dA[la][pa][m] * dB[m][pb][rb]
    for (let la = 0; la < chiL; la++) {
      for (let pa = 0; pa < 2; pa++) {
        for (let pb = 0; pb < 2; pb++) {
          for (let rb = 0; rb < chiR; rb++) {
            let tre = 0, tim = 0
            for (let m = 0; m < chiM; m++) {
              const ai = ((la * 2 + pa) * chiM + m) * 2
              const bi = ((m  * 2 + pb) * chiR + rb) * 2
              tre += dA[ai]! * dB[bi]! - dA[ai + 1]! * dB[bi + 1]!
              tim += dA[ai]! * dB[bi + 1]! + dA[ai + 1]! * dB[bi]!
            }
            for (let pa2 = 0; pa2 < 2; pa2++) {
              for (let pb2 = 0; pb2 < 2; pb2++) {
                const g  = gate[pa2 * 2 + pb2]![pa * 2 + pb]!
                const mi = ((la * 2 + pa2) * cols + pb2 * chiR + rb) * 2
                mBuf[mi]     = (mBuf[mi]     ?? 0) + g.re * tre - g.im * tim
                mBuf[mi + 1] = (mBuf[mi + 1] ?? 0) + g.re * tim + g.im * tre
              }
            }
          }
        }
      }
    }

    // SVD-decompose mBuf and write results directly into dA and dB.
    // SVD gives the optimal Schmidt truncation at each bond cut.
    const bond  = this.svdDecompose(rows, cols)
    const qSize = rows * bond * 2
    const rSize = bond * cols * 2
    for (let i = 0; i < qSize; i++) dA[i] = this.qBuf[i]!
    for (let i = 0; i < rSize; i++) dB[i] = this.rBuf[i]!
    this.chiR[a] = bond
    this.chiL[b] = bond
  }

  /**
   * MGS QR decomposition of this.mBuf (rows × cols, row-major) into this.qBuf
   * (rows × bond) and this.rBuf (bond × cols). Returns bond dimension.
   */
  private qrDecompose(rows: number, cols: number): number {
    const mBuf    = this.mBuf
    const aBuf    = this.aBuf
    const qColBuf = this.qColBuf
    const maxChi  = this.maxChi

    // Transpose mBuf into aBuf (column-major) for column-by-column iteration.
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        const src = (i * cols + j) * 2
        const dst = (j * rows + i) * 2
        aBuf[dst]     = mBuf[src]!
        aBuf[dst + 1] = mBuf[src + 1]!
      }
    }

    let bond = 0
    for (let j = 0; j < cols && bond < maxChi; j++) {
      let n2 = 0
      for (let i = 0; i < rows; i++) {
        const re = aBuf[(j * rows + i) * 2]!, im = aBuf[(j * rows + i) * 2 + 1]!
        n2 += re * re + im * im
      }
      if (n2 < 1e-28) continue

      const inv   = 1 / Math.sqrt(n2)
      const qBase = bond * rows
      for (let i = 0; i < rows; i++) {
        qColBuf[(qBase + i) * 2]     = aBuf[(j * rows + i) * 2]!     * inv
        qColBuf[(qBase + i) * 2 + 1] = aBuf[(j * rows + i) * 2 + 1]! * inv
      }

      // Project out the new Q column from all remaining columns.
      for (let l = j + 1; l < cols; l++) {
        let dre = 0, dim = 0
        for (let i = 0; i < rows; i++) {
          const qre = qColBuf[(qBase + i) * 2]!, qim = qColBuf[(qBase + i) * 2 + 1]!
          const are = aBuf[(l * rows + i) * 2]!, aim = aBuf[(l * rows + i) * 2 + 1]!
          dre += qre * are + qim * aim
          dim += qre * aim - qim * are
        }
        for (let i = 0; i < rows; i++) {
          const qre = qColBuf[(qBase + i) * 2]!, qim = qColBuf[(qBase + i) * 2 + 1]!
          const idx = (l * rows + i) * 2
          aBuf[idx]     = (aBuf[idx]     ?? 0) - (dre * qre - dim * qim)
          aBuf[idx + 1] = (aBuf[idx + 1] ?? 0) - (dre * qim + dim * qre)
        }
      }
      bond++
    }

    // Build qBuf (row-major) from packed Q columns.
    const qBuf = this.qBuf
    for (let k = 0; k < bond; k++) {
      const qBase = k * rows
      for (let i = 0; i < rows; i++) {
        qBuf[(i * bond + k) * 2]     = qColBuf[(qBase + i) * 2]!
        qBuf[(i * bond + k) * 2 + 1] = qColBuf[(qBase + i) * 2 + 1]!
      }
    }

    // Build rBuf (row-major): R = Q† · M.
    const rBuf = this.rBuf
    const rSize = bond * cols * 2
    for (let i = 0; i < rSize; i++) rBuf[i] = 0
    for (let k = 0; k < bond; k++) {
      const qBase = k * rows
      for (let j = 0; j < cols; j++) {
        let rre = 0, rim = 0
        for (let i = 0; i < rows; i++) {
          const qre = qColBuf[(qBase + i) * 2]!, qim = qColBuf[(qBase + i) * 2 + 1]!
          const mre = mBuf[(i * cols + j) * 2]!, mim = mBuf[(i * cols + j) * 2 + 1]!
          rre += qre * mre + qim * mim
          rim += qre * mim - qim * mre
        }
        rBuf[(k * cols + j) * 2]     = rre
        rBuf[(k * cols + j) * 2 + 1] = rim
      }
    }

    return bond
  }

  /**
   * One-sided complex Jacobi SVD on this.mBuf (rows × cols, row-major).
   *
   * Decomposes M ≈ U · diag(σ) · V† where:
   *   qBuf  — U   (rows × bond, row-major): left singular vectors
   *   rBuf  — σ·V† (bond × cols, row-major): scaled right singular vectors
   *
   * Algorithm: sweep Jacobi rotations over column pairs until orthogonal,
   * then sort by descending singular value and truncate to maxChi.
   * Gives the optimal low-rank approximation at each MPS bond cut (Schmidt decomp).
   *
   * All buffers pre-allocated; zero heap allocation in this method.
   */
  private svdDecompose(rows: number, cols: number): number {
    const mBuf    = this.mBuf
    const aBuf    = this.aBuf    // column-major working copy of M
    const vBuf    = this.vBuf    // V matrix, column-major (cols × cols)
    const maxChi  = this.maxChi

    // Copy mBuf (row-major) → aBuf (column-major).
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        const src = (i * cols + j) * 2
        const dst = (j * rows + i) * 2
        aBuf[dst]     = mBuf[src]!
        aBuf[dst + 1] = mBuf[src + 1]!
      }
    }

    // Initialize V = I (column-major, cols × cols).
    const vSize = cols * cols * 2
    for (let k = 0; k < vSize; k++) vBuf[k] = 0
    for (let k = 0; k < cols; k++) vBuf[(k * cols + k) * 2] = 1

    // Jacobi sweeps: for each column pair (p,q), find and apply the rotation
    // that zeros the (p,q) off-diagonal of the Gram matrix A†A.
    //
    // Rotation (right-multiply A and V by [[c, -s], [s·e^{-iα}, c·e^{-iα}]]):
    //   new_col_p = c·col_p + s·e^{-iα}·col_q
    //   new_col_q = -s·col_p + c·e^{-iα}·col_q
    // where α = arg(G_pq), tan(2θ) = 2|G_pq| / (G_pp - G_qq).
    const maxSweeps = 20
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let maxOff = 0

      for (let p = 0; p < cols - 1; p++) {
        for (let q = p + 1; q < cols; q++) {
          // Gram entries: G_pp, G_qq ∈ ℝ; G_pq = col_p† · col_q ∈ ℂ.
          let Gpp = 0, Gqq = 0, Gpqre = 0, Gpqim = 0
          for (let i = 0; i < rows; i++) {
            const Apre = aBuf[(p * rows + i) * 2]!, Apim = aBuf[(p * rows + i) * 2 + 1]!
            const Aqre = aBuf[(q * rows + i) * 2]!, Aqim = aBuf[(q * rows + i) * 2 + 1]!
            Gpp   += Apre * Apre + Apim * Apim
            Gqq   += Aqre * Aqre + Aqim * Aqim
            Gpqre += Apre * Aqre + Apim * Aqim   // Re(col_p† · col_q)
            Gpqim += Apre * Aqim - Apim * Aqre   // Im(col_p† · col_q)
          }

          const r = Math.sqrt(Gpqre * Gpqre + Gpqim * Gpqim)
          if (r > maxOff) maxOff = r
          if (r < 1e-14 * Math.sqrt(Gpp * Gqq) + 1e-28) continue

          // e^{-iα} where α = arg(G_pq): epre = Re, epim = Im.
          const epre = Gpqre / r, epim = -Gpqim / r

          // Jacobi angle from real-symmetric off-diagonal |G_pq|.
          // G'_pq = 0 requires t satisfying t² - 2τt - 1 = 0; small root: t = τ - √(τ²+1).
          const tau = (Gqq - Gpp) / (2 * r)
          const t   = tau >= 0
            ? -1 / ( tau + Math.sqrt(1 + tau * tau))
            :  1 / (-tau + Math.sqrt(1 + tau * tau))
          const c = 1 / Math.sqrt(1 + t * t)
          const s = t * c

          // Update columns p, q of aBuf.
          for (let i = 0; i < rows; i++) {
            const pi = (p * rows + i) * 2, qi = (q * rows + i) * 2
            const Apr = aBuf[pi]!, Api = aBuf[pi + 1]!
            const Aqr = aBuf[qi]!, Aqi = aBuf[qi + 1]!
            // s · e^{-iα} · A[q]
            const seqr = s * (epre * Aqr - epim * Aqi)
            const seqi = s * (epre * Aqi + epim * Aqr)
            // c · e^{-iα} · A[q]
            const ceqr = c * (epre * Aqr - epim * Aqi)
            const ceqi = c * (epre * Aqi + epim * Aqr)
            aBuf[pi]     = c * Apr + seqr
            aBuf[pi + 1] = c * Api + seqi
            aBuf[qi]     = -s * Apr + ceqr
            aBuf[qi + 1] = -s * Api + ceqi
          }

          // Apply same rotation to V columns p, q.
          for (let i = 0; i < cols; i++) {
            const pi = (p * cols + i) * 2, qi = (q * cols + i) * 2
            const Vpr = vBuf[pi]!, Vpi = vBuf[pi + 1]!
            const Vqr = vBuf[qi]!, Vqi = vBuf[qi + 1]!
            const seVr = s * (epre * Vqr - epim * Vqi)
            const seVi = s * (epre * Vqi + epim * Vqr)
            const ceVr = c * (epre * Vqr - epim * Vqi)
            const ceVi = c * (epre * Vqi + epim * Vqr)
            vBuf[pi]     = c * Vpr + seVr
            vBuf[pi + 1] = c * Vpi + seVi
            vBuf[qi]     = -s * Vpr + ceVr
            vBuf[qi + 1] = -s * Vpi + ceVi
          }
        }
      }

      if (maxOff < 1e-14) break
    }

    // Singular values = column norms of aBuf.
    const sigmaBuf = this.sigmaBuf
    for (let k = 0; k < cols; k++) {
      let s2 = 0
      for (let i = 0; i < rows; i++) {
        const re = aBuf[(k * rows + i) * 2]!, im = aBuf[(k * rows + i) * 2 + 1]!
        s2 += re * re + im * im
      }
      sigmaBuf[k] = Math.sqrt(s2)
    }

    // Insertion-sort orderBuf by descending sigma (zero-alloc, n ≤ 2·maxChi).
    const orderBuf = this.orderBuf
    for (let k = 0; k < cols; k++) orderBuf[k] = k
    for (let i = 1; i < cols; i++) {
      let j = i
      while (j > 0 && sigmaBuf[orderBuf[j - 1]!]! < sigmaBuf[orderBuf[j]!]!) {
        const tmp = orderBuf[j - 1]!; orderBuf[j - 1] = orderBuf[j]!; orderBuf[j] = tmp; j--
      }
    }

    // Bond = number of non-negligible singular values, capped at maxChi.
    let bond = 0
    while (bond < cols && bond < maxChi && sigmaBuf[orderBuf[bond]!]! > 1e-14) bond++
    if (bond === 0) bond = 1  // always keep at least one component

    // Build qBuf: U·Σ (rows × bond, row-major).
    // After Jacobi convergence aBuf[:,col_k] = sigma_k * U[:,col_k], so we copy
    // directly without normalising.  Storing U·Σ in the left tensor and V† in the
    // right tensor ensures sample() computes correct marginal probabilities via the
    // Schmidt weights: P(phys=p) = ||ΣU[p,:]||² = Σ_k σ_k² |U[p][k]|².
    const qBuf = this.qBuf
    for (let k = 0; k < bond; k++) {
      const col = orderBuf[k]!
      for (let i = 0; i < rows; i++) {
        const src = (col * rows + i) * 2
        qBuf[(i * bond + k) * 2]     = aBuf[src]!
        qBuf[(i * bond + k) * 2 + 1] = aBuf[src + 1]!
      }
    }

    // Build rBuf: V† (bond × cols, row-major).
    // rBuf[k][j] = conj(V[j][col_k]) — no sigma factor here; weights live in qBuf.
    //   V[j][col_k] = vBuf[(col_k * cols + j) * 2] (column-major V)
    const rBuf = this.rBuf
    for (let k = 0; k < bond; k++) {
      const col = orderBuf[k]!
      for (let j = 0; j < cols; j++) {
        const vi = (col * cols + j) * 2
        rBuf[(k * cols + j) * 2]     =  vBuf[vi]!
        rBuf[(k * cols + j) * 2 + 1] = -vBuf[vi + 1]!  // conj for V†
      }
    }

    return bond
  }

  /**
   * Sample a basis state by sequential left-to-right marginal collapse.
   * No allocations — uses pre-allocated sv0/sv1/st workspace.
   */
  sample(rand: () => number): bigint {
    const stRe = this.stRe, stIm = this.stIm
    stRe[0] = 1; stIm[0] = 0
    let result = 0n

    for (let q = 0; q < this.n; q++) {
      const data = this.data[q]!
      const chiL = this.chiL[q]!, chiR = this.chiR[q]!
      const v0re = this.sv0re, v0im = this.sv0im
      const v1re = this.sv1re, v1im = this.sv1im

      for (let r = 0; r < chiR; r++) { v0re[r] = 0; v0im[r] = 0; v1re[r] = 0; v1im[r] = 0 }

      for (let l = 0; l < chiL; l++) {
        const sre = stRe[l]!, sim = stIm[l]!
        for (let r = 0; r < chiR; r++) {
          const i0 = ((l * 2 + 0) * chiR + r) * 2
          const i1 = ((l * 2 + 1) * chiR + r) * 2
          v0re[r] = (v0re[r] ?? 0) + sre * data[i0]!     - sim * data[i0 + 1]!
          v0im[r] = (v0im[r] ?? 0) + sre * data[i0 + 1]! + sim * data[i0]!
          v1re[r] = (v1re[r] ?? 0) + sre * data[i1]!     - sim * data[i1 + 1]!
          v1im[r] = (v1im[r] ?? 0) + sre * data[i1 + 1]! + sim * data[i1]!
        }
      }

      let p0 = 0, p1 = 0
      for (let r = 0; r < chiR; r++) {
        p0 += v0re[r]! * v0re[r]! + v0im[r]! * v0im[r]!
        p1 += v1re[r]! * v1re[r]! + v1im[r]! * v1im[r]!
      }

      const total = p0 + p1
      const bit   = total > 0 && rand() >= p0 / total ? 1 : 0
      if (bit === 1) result |= 1n << BigInt(q)

      const chosen = bit === 0 ? p0 : p1
      const inv    = chosen > 0 ? 1 / Math.sqrt(chosen) : 0
      const vRe    = bit === 0 ? v0re : v1re
      const vIm    = bit === 0 ? v0im : v1im
      for (let r = 0; r < chiR; r++) { stRe[r] = vRe[r]! * inv; stIm[r] = vIm[r]! * inv }
    }

    return result
  }

  /** Maximum bond dimension currently in use across all sites. */
  maxBondUsed(): number {
    let max = 1
    for (let q = 0; q < this.n; q++) {
      if (this.chiL[q]! > max) max = this.chiL[q]!
      if (this.chiR[q]! > max) max = this.chiR[q]!
    }
    return max
  }
}

// ── Debug / test utilities ────────────────────────────────────────────────────

/**
 * @internal — exposed for testing only.
 * Contract the full MPS into a dense 2^n amplitude array.
 * Returns Float64Array of length 2^n * 2 (interleaved re/im).
 * Only practical for n ≤ 20.
 */
export function mpsContract(mps: MPS): Float64Array {
  const n = mps.length
  const dim = 1 << n
  const result = new Float64Array(dim * 2)

  for (let idx = 0; idx < dim; idx++) {
    // Contract qubit by qubit for this bitstring
    let vecRe = new Float64Array([1])
    let vecIm = new Float64Array([1])
    vecIm[0] = 0

    for (let q = 0; q < n; q++) {
      const { data, chiL, chiR } = mps[q]!
      const bit = (idx >> q) & 1  // physical index for qubit q (LSB = qubit 0)
      const newRe = new Float64Array(chiR)
      const newIm = new Float64Array(chiR)

      for (let l = 0; l < chiL; l++) {
        const vRe = vecRe[l]!, vIm = vecIm[l]!
        for (let r = 0; r < chiR; r++) {
          const i = ((l * 2 + bit) * chiR + r) * 2
          const tRe = data[i]!, tIm = data[i + 1]!
          newRe[r] = (newRe[r] ?? 0) + vRe * tRe - vIm * tIm
          newIm[r] = (newIm[r] ?? 0) + vRe * tIm + vIm * tRe
        }
      }

      vecRe = newRe
      vecIm = newIm
    }

    // At the end chiR = 1, so vecRe[0], vecIm[0] is the amplitude
    result[idx * 2]     = vecRe[0]!
    result[idx * 2 + 1] = vecIm[0]!
  }

  return result
}

/**
 * @internal — exposed for testing only.
 * Return max bond dimension currently in the MPS.
 */
export function mpsMaxBond(mps: MPS): number {
  return Math.max(...mps.map(t => Math.max(t.chiL, t.chiR)))
}

/**
 * @internal — exposed for testing only.
 * Return the tensor at site q (for inspection).
 */
export function mpsTensor(mps: MPS, q: number): Tensor {
  return mps[q]!
}
