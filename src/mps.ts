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
import * as G from './gates.js'

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
  readonly n:        number
  readonly maxChi:   number
  /**
   * Relative Schmidt truncation threshold. Singular values σ_k < truncErr · σ_max
   * are discarded in addition to the hard `maxChi` cap.
   *
   * 0 (default) means truncate by absolute tolerance only (1e-14).
   * Typical values: 1e-8 for chemistry/VQE circuits with rapidly decaying Schmidt spectra.
   */
  readonly truncErr: number

  // Per-site tensor storage. data[q] holds up to maxChi × 2 × maxChi complex values.
  // Layout: data[q][((l*2+p)*chiR+r)*2] = re, +1 = im, where chiR = this.chiR[q].
  //
  // Vidal canonical (Γ-Λ) form:
  //   data[q]       = Γ[q]      — left-isometric in the Vidal sense: Γ†Γ = I after
  //                               weighting by the boundary lambdas (not plain U†U=I)
  //   bondLambda[b] = Λ[b]      — Schmidt values at bond (b, b+1), sorted descending
  //
  // The state amplitude is: ψ = Γ[0] · Λ[0] · Γ[1] · Λ[1] · ... · Γ[n-1]
  // (boundary lambdas Λ[-1] = Λ[n] = 1 are implicit).
  //
  // This form ensures:
  //   • apply2Adjacent includes all three boundary lambdas in theta → optimal truncation
  //   • bondLambda holds the true Schmidt spectrum at every bond → correct bondEntropies()
  //   • sample() weighted by bondLambda[q] gives exact marginals (right environment = Λ²)
  readonly data:       Float64Array[]
  readonly chiL:       Int32Array
  readonly chiR:       Int32Array
  readonly bondLambda: Float64Array[]  // n-1 entries; bondLambda[b][k] = σ_k at bond b

  // Workspace for apply2Adjacent — sized for worst-case maxChi bonds.
  private readonly mBuf: Float64Array  // contracted θ matrix, rows × cols
  private readonly aBuf: Float64Array  // column-major working copy for Jacobi
  private readonly qBuf: Float64Array  // U·Σ output, row-major: rows × bond
  private readonly rBuf: Float64Array  // V†   output, row-major: bond × cols

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

  constructor(n: number, maxChi: number, truncErr = 0) {
    this.n        = n
    this.maxChi   = maxChi
    this.truncErr = truncErr

    const maxRows = maxChi * 2
    const maxCols = maxChi * 2

    this.data       = Array.from({ length: n }, () => new Float64Array(maxRows * maxCols * 2))
    this.chiL       = new Int32Array(n)
    this.chiR       = new Int32Array(n)
    this.bondLambda = Array.from({ length: Math.max(n - 1, 0) }, () => new Float64Array(maxChi))

    this.mBuf = new Float64Array(maxRows * maxCols * 2)
    this.aBuf = new Float64Array(maxCols * maxRows * 2)
    this.qBuf = new Float64Array(maxRows * maxChi  * 2)
    this.rBuf = new Float64Array(maxChi  * maxCols * 2)

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
    for (let b = 0; b < this.n - 1; b++) {
      this.bondLambda[b]!.fill(0)
      this.bondLambda[b]![0] = 1
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

    // Vidal canonical boundary lambdas:
    //   lambdaL = Λ[a-1] — left  boundary of Γ[a]   (null → trivial scalar 1)
    //   lambdaM = Λ[a]   — bond  shared by Γ[a],Γ[b] (always present)
    //   lambdaR = Λ[a+1] — right boundary of Γ[b]    (null → trivial scalar 1)
    // Theta = Λ[a-1] · Γ[a] · Λ[a] · Γ[b] · Λ[a+1]
    const lambdaL: Float64Array | null = a     > 0         ? this.bondLambda[a - 1]! : null
    const lambdaM: Float64Array        =                      this.bondLambda[a]!
    const lambdaR: Float64Array | null = b     < this.n - 1 ? this.bondLambda[b]!    : null

    // Clear the used region of mBuf before accumulating.
    const mSize = rows * cols * 2
    for (let i = 0; i < mSize; i++) mBuf[i] = 0

    // Contract Γ[a] ⊗ Γ[b] through (lambdas + gate) into mBuf.
    // theta = Λ_L[la] · (Σ_m Γ[a][la,pa,m] · Λ_M[m] · Γ[b][m,pb,rb]) · Λ_R[rb]
    for (let la = 0; la < chiL; la++) {
      const scaleL = lambdaL ? lambdaL[la]! : 1
      for (let pa = 0; pa < 2; pa++) {
        for (let pb = 0; pb < 2; pb++) {
          for (let rb = 0; rb < chiR; rb++) {
            const scaleR = lambdaR ? lambdaR[rb]! : 1
            let tre = 0, tim = 0
            for (let m = 0; m < chiM; m++) {
              const scaleM = lambdaM[m]!
              const ai = ((la * 2 + pa) * chiM + m) * 2
              const bi = ((m  * 2 + pb) * chiR + rb) * 2
              tre += scaleM * (dA[ai]! * dB[bi]! - dA[ai + 1]! * dB[bi + 1]!)
              tim += scaleM * (dA[ai]! * dB[bi + 1]! + dA[ai + 1]! * dB[bi]!)
            }
            const scale = scaleL * scaleR
            tre *= scale; tim *= scale
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

    // SVD: theta = U · Σ · V†. Vidal canonical tensors:
    //   Γ[a]  = U / Λ[a-1]   (divide each row la by lambdaL[la])
    //   Λ[a]  = Σ             (stored in bondLambda[a])
    //   Γ[b]  = V† / Λ[a+1]  (divide each col rb by lambdaR[rb])
    const bond = this.svdDecompose(rows, cols)

    // Extract Γ[a] from qBuf (= U after normalization in svdDecompose).
    // Divide row la by lambdaL[la] to remove the left boundary lambda absorbed into theta.
    const qBuf = this.qBuf
    for (let la = 0; la < chiL; la++) {
      const invL = lambdaL ? (lambdaL[la]! > 1e-14 ? 1 / lambdaL[la]! : 0) : 1
      for (let p = 0; p < 2; p++) {
        for (let k = 0; k < bond; k++) {
          const i = ((la * 2 + p) * bond + k) * 2
          dA[i]     = qBuf[i]!     * invL
          dA[i + 1] = qBuf[i + 1]! * invL
        }
      }
    }

    // Extract Γ[b] from rBuf (= V†). Divide col rb by lambdaR[rb].
    const rBuf = this.rBuf
    for (let k = 0; k < bond; k++) {
      for (let p = 0; p < 2; p++) {
        for (let rb = 0; rb < chiR; rb++) {
          const invR = lambdaR ? (lambdaR[rb]! > 1e-14 ? 1 / lambdaR[rb]! : 0) : 1
          const i = (k * cols + p * chiR + rb) * 2
          dB[i]     = rBuf[i]!     * invR
          dB[i + 1] = rBuf[i + 1]! * invR
        }
      }
    }

    this.chiR[a] = bond
    this.chiL[b] = bond

    // Store true Schmidt values Λ[a] = Σ in bondLambda[a].
    const lambda = this.bondLambda[a]!
    for (let k = 0; k < bond; k++)           lambda[k] = this.sigmaBuf[this.orderBuf[k]!]!
    for (let k = bond; k < this.maxChi; k++) lambda[k] = 0
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
    const mBuf   = this.mBuf
    const aBuf   = this.aBuf    // column-major working copy of M
    const vBuf   = this.vBuf    // V matrix, column-major (cols × cols)
    const maxChi = this.maxChi

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

    // Frobenius norm² of M = trace of G = A†A, invariant under Jacobi rotations.
    // Used for a scale-relative convergence test so the threshold holds for any
    // bond matrix magnitude (product-state tensors vs deep-circuit tensors).
    let frobSq = 0
    for (let k = 0; k < cols; k++) {
      for (let i = 0; i < rows; i++) {
        const re = aBuf[(k * rows + i) * 2]!, im = aBuf[(k * rows + i) * 2 + 1]!
        frobSq += re * re + im * im
      }
    }

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
            const seqr = s * (epre * Aqr - epim * Aqi)
            const seqi = s * (epre * Aqi + epim * Aqr)
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

      // Relative convergence: off-diagonal coupling negligible vs total energy.
      // Guard frobSq=0 (zero input matrix) so the condition doesn't become maxOff<0.
      if (frobSq < 1e-28 || maxOff < 1e-14 * frobSq) break
    }

    // Singular values = column norms of aBuf after Jacobi convergence.
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

    // Bond dimension: keep singular values above the cutoff, capped at maxChi.
    // cutoff = max(absolute floor, relative fraction of σ_max).
    const sigma0 = sigmaBuf[orderBuf[0]!]!
    const cutoff = sigma0 > 0
      ? Math.max(1e-14, this.truncErr * sigma0)
      : 1e-14
    let bond = 0
    while (bond < cols && bond < maxChi && sigmaBuf[orderBuf[bond]!]! > cutoff) bond++
    if (bond === 0) bond = 1  // always keep at least one component

    // Build qBuf: U (rows × bond, row-major) — normalised left singular vectors.
    // aBuf[:,col_k] = σ_k · U[:,col_k] after Jacobi convergence; divide by σ_k to get U.
    // apply2Adjacent then divides further by Λ[a-1] to produce Γ[a] = U/Λ[a-1].
    const qBuf = this.qBuf
    for (let k = 0; k < bond; k++) {
      const col   = orderBuf[k]!
      const sigma = sigmaBuf[col]!
      const inv   = sigma > 1e-14 ? 1 / sigma : 0
      for (let i = 0; i < rows; i++) {
        const src = (col * rows + i) * 2
        qBuf[(i * bond + k) * 2]     = aBuf[src]!     * inv
        qBuf[(i * bond + k) * 2 + 1] = aBuf[src + 1]! * inv
      }
    }

    // Build rBuf: V† (bond × cols, row-major).
    // rBuf[k][j] = conj(V[j][col_k]);  V is column-major: V[j][col_k] = vBuf[(col_k*cols+j)*2].
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

      // Vidal canonical: data[q] = Γ[q] (isometry); right environment = Λ[q]².
      // Weight v_p by bondLambda[q] so that P(p) = ||v_p · Λ[q]||² gives the correct marginal.
      // For q = n-1 (last site) there is no right bond lambda — no weighting needed.
      if (q < this.n - 1) {
        const lambda = this.bondLambda[q]!
        for (let r = 0; r < chiR; r++) {
          const s = lambda[r]!
          v0re[r] = (v0re[r] ?? 0) * s
          v0im[r] = (v0im[r] ?? 0) * s
          v1re[r] = (v1re[r] ?? 0) * s
          v1im[r] = (v1im[r] ?? 0) * s
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

  /**
   * Von Neumann entanglement entropies S_b = -Σ_k σ_k² log₂(σ_k²) at each bond.
   * Returns n-1 values. Uses the stored bondLambda Schmidt values (left-canonical).
   * O(n · χ) — suitable for circuit monitoring and truncation diagnostics.
   */
  bondEntropies(): number[] {
    const result: number[] = new Array(this.n - 1)
    for (let b = 0; b < this.n - 1; b++) {
      const lambda = this.bondLambda[b]!
      const chi    = this.chiR[b]!
      let S = 0
      for (let k = 0; k < chi; k++) {
        const s = lambda[k]!
        if (s > 1e-14) S -= s * s * Math.log2(s * s)
      }
      result[b] = S
    }
    return result
  }
}

// ── Trajectory depolarizing channels ─────────────────────────────────────────

const TWO_PAULI_TRAJ: readonly (readonly [Gate2x2 | null, Gate2x2 | null])[] = [
  [null, G.X], [null, G.Y], [null, G.Z],
  [G.X, null], [G.X, G.X], [G.X, G.Y], [G.X, G.Z],
  [G.Y, null], [G.Y, G.X], [G.Y, G.Y], [G.Y, G.Z],
  [G.Z, null], [G.Z, G.X], [G.Z, G.Y], [G.Z, G.Z],
]

/** Apply single-qubit depolarizing channel to a trajectory MPS in place. */
export function dep1Traj(traj: MpsTrajectory, q: number, p: number, rand: number): void {
  if (rand >= p) return
  const r = rand / p
  if      (r < 1/3) traj.apply1(q, G.X)
  else if (r < 2/3) traj.apply1(q, G.Y)
  else               traj.apply1(q, G.Z)
}

/** Apply two-qubit depolarizing channel to a trajectory MPS in place. */
export function dep2Traj(traj: MpsTrajectory, a: number, b: number, p: number, rand: number): void {
  if (rand >= p) return
  const [pa, pb] = TWO_PAULI_TRAJ[Math.min(Math.floor(rand / p * 15), 14)]!
  if (pa) traj.apply1(a, pa)
  if (pb) traj.apply1(b, pb)
}

// ── Serializable trajectory ops ───────────────────────────────────────────────

/**
 * Normalized, serializable gate operation for trajectory execution.
 *
 * `controlled` and multi-qubit `unitary` ops from Circuit are pre-expanded
 * into `single` / `two` before being stored as TrajOp, so the worker dispatch
 * loop stays a flat switch with no helper calls.
 */
export type TrajOp =
  | { kind: 'single'; q: number;              gate: Gate2x2     }
  | { kind: 'cnot';   control: number; target: number           }
  | { kind: 'swap';   a: number;       b: number                }
  | { kind: 'two';    a: number;       b: number; gate: Gate4x4 }
  | { kind: 'barrier'                                           }

/**
 * Execute one trajectory: apply all ops with optional depolarizing noise.
 * rng() is only called when p1 or p2 is non-zero.
 */
export function applyTrajOps(
  traj: MpsTrajectory,
  ops: readonly TrajOp[],
  p1: number,
  p2: number,
  rng: () => number,
): void {
  for (const op of ops) {
    switch (op.kind) {
      case 'single':
        traj.apply1(op.q, op.gate)
        if (p1) dep1Traj(traj, op.q, p1, rng())
        break
      case 'cnot':
        traj.apply2(op.control, op.target, CNOT4)
        if (p2) dep2Traj(traj, op.control, op.target, p2, rng())
        break
      case 'swap':
        traj.apply2(op.a, op.b, SWAP4)
        if (p2) dep2Traj(traj, op.a, op.b, p2, rng())
        break
      case 'two':
        traj.apply2(op.a, op.b, op.gate)
        if (p2) dep2Traj(traj, op.a, op.b, p2, rng())
        break
      case 'barrier': break
      default: { const _exhaustive: never = op; break }
    }
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
