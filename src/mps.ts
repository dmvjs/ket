/**
 * Matrix Product State (MPS) tensor network simulator.
 *
 * Represents an n-qubit pure state as a chain of rank-3 tensors:
 *   mps[q] : Complex[chiL][2][chiR]
 *
 * where chiL / chiR are the left / right bond dimensions.
 * Site 0 has chiL = 1; site n-1 has chiR = 1.
 *
 * Handles 50+ qubit circuits with bounded entanglement (GHZ, BV, shallow QFT)
 * in O(n · chi² · 2) memory — vs O(2ⁿ) for full statevector.
 */

import { add, c, mul, norm2, scale, type Complex, ZERO, ONE } from './complex.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'

// ── Types ────────────────────────────────────────────────────────────────────

/** Rank-3 tensor indexed [leftBond][physical][rightBond]. */
type Tensor = Complex[][][]

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

// ── MPS initialisation ───────────────────────────────────────────────────────

/** Return the |0...0⟩ MPS with bond dimension 1. */
export function mpsInit(n: number): MPS {
  return Array.from({ length: n }, () => [[[ONE], [ZERO]]])
}

// ── Single-qubit gate ─────────────────────────────────────────────────────────

/** Apply a single-qubit gate to site q. O(chi · 2). */
export function mpsApply1(mps: MPS, q: number, [[a, b], [cc, d]]: Gate2x2): MPS {
  const T = mps[q]!
  const chiL = T.length, chiR = T[0]![0]!.length
  const newT: Tensor = Array.from({ length: chiL }, (_, l) => [
    Array.from({ length: chiR }, (_, r) => add(mul(a, T[l]![0]![r]!), mul(b, T[l]![1]![r]!))),
    Array.from({ length: chiR }, (_, r) => add(mul(cc, T[l]![0]![r]!), mul(d, T[l]![1]![r]!))),
  ])
  return mps.with(q, newT)
}

// ── QR decomposition (Modified Gram-Schmidt) ──────────────────────────────────

/**
 * QR decompose M (rows × cols) with bond truncation.
 * Returns Q (rows × bond) with orthonormal columns and R = Q†M (bond × cols),
 * where bond ≤ maxBond. Scans all columns for non-zero pivots (implicit column pivoting).
 */
function qrMGS(M: Complex[][], maxBond: number): { Q: Complex[][]; R: Complex[][]; bond: number } {
  const rows = M.length, cols = M[0]!.length

  // Column-major working copy for Gram-Schmidt
  const A: Complex[][] = Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => M[i]![j]!)
  )

  const Qcols: Complex[][] = []  // orthonormal basis vectors (each length = rows)

  // Scan all columns; collect up to maxBond non-zero orthonormal vectors
  for (let j = 0; j < cols && Qcols.length < maxBond; j++) {
    let n2 = 0
    for (const v of A[j]!) n2 += norm2(v)
    if (n2 < 1e-28) continue  // linearly dependent or zero — skip

    const inv = 1 / Math.sqrt(n2)
    const qj = A[j]!.map(v => scale(inv, v))
    Qcols.push(qj)

    // Orthogonalise remaining columns against qj
    for (let l = j + 1; l < cols; l++) {
      let re = 0, im = 0
      for (let i = 0; i < rows; i++) {
        const qi = qj[i]!, al = A[l]![i]!
        re += qi.re * al.re + qi.im * al.im   // conj(qi) · al
        im += qi.re * al.im - qi.im * al.re
      }
      for (let i = 0; i < rows; i++) {
        const qi = qj[i]!
        A[l]![i] = add(A[l]![i]!, c(
          -(re * qi.re - im * qi.im),
          -(re * qi.im + im * qi.re),
        ))
      }
    }
  }

  const bond = Qcols.length
  const Q: Complex[][] = Array.from({ length: rows }, (_, i) =>
    Array.from({ length: bond }, (_, k) => Qcols[k]![i]!)
  )

  // R = Q† · M  (exact: Q has orthonormal columns so Q·R = projection of M)
  const R: Complex[][] = Array.from({ length: bond }, (_, k) =>
    Array.from({ length: cols }, (_, j) => {
      let re = 0, im = 0
      for (let i = 0; i < rows; i++) {
        const qi = Qcols[k]![i]!, mi = M[i]![j]!
        re += qi.re * mi.re + qi.im * mi.im
        im += qi.re * mi.im - qi.im * mi.re
      }
      return c(re, im)
    })
  )

  return { Q, R, bond }
}

// ── Two-qubit gate (adjacent) ─────────────────────────────────────────────────

/**
 * Apply a two-qubit gate to adjacent sites a and a+1.
 * Contracts A⊗B, applies gate, then QR-decomposes back to A'⊗B'.
 * Bond dimension is capped at maxBond.
 */
function mpsApply2Adjacent(mps: MPS, a: number, gate: Gate4x4, maxBond: number): MPS {
  const b = a + 1
  const A = mps[a]!, B = mps[b]!
  const chiL = A.length, chiM = B.length, chiR = B[0]![0]!.length

  // Build M[la*2+pa'][pb'*chiR+rb] by contracting and applying gate inline
  const rows = chiL * 2, cols = 2 * chiR
  const M: Complex[][] = Array.from({ length: rows }, () =>
    Array.from<Complex>({ length: cols }).fill(ZERO)
  )

  for (let la = 0; la < chiL; la++) {
    for (let pa = 0; pa < 2; pa++) {
      for (let pb = 0; pb < 2; pb++) {
        for (let rb = 0; rb < chiR; rb++) {
          // theta = sum_m A[la][pa][m] * B[m][pb][rb]
          let re = 0, im = 0
          for (let m = 0; m < chiM; m++) {
            const av = A[la]![pa]![m]!, bv = B[m]![pb]![rb]!
            re += av.re * bv.re - av.im * bv.im
            im += av.re * bv.im + av.im * bv.re
          }
          // Scatter into M via gate rows
          for (let pa2 = 0; pa2 < 2; pa2++) {
            for (let pb2 = 0; pb2 < 2; pb2++) {
              const g = gate[pa2 * 2 + pb2]![pa * 2 + pb]!
              const row = la * 2 + pa2, col = pb2 * chiR + rb
              const cur = M[row]![col]!
              M[row]![col] = c(
                cur.re + g.re * re - g.im * im,
                cur.im + g.re * im + g.im * re,
              )
            }
          }
        }
      }
    }
  }

  const { Q, R, bond } = qrMGS(M, maxBond)

  // Reshape Q → newA[la][pa'][bond]
  const newA: Tensor = Array.from({ length: chiL }, (_, la) => [
    Array.from({ length: bond }, (_, j) => Q[la * 2 + 0]![j]!),
    Array.from({ length: bond }, (_, j) => Q[la * 2 + 1]![j]!),
  ])

  // Reshape R → newB[bond][pb'][rb]
  const newB: Tensor = Array.from({ length: bond }, (_, j) => [
    Array.from({ length: chiR }, (_, rb) => R[j]![0 * chiR + rb]!),
    Array.from({ length: chiR }, (_, rb) => R[j]![1 * chiR + rb]!),
  ])

  return mps.with(a, newA).with(b, newB)
}

// ── Two-qubit gate (arbitrary) ────────────────────────────────────────────────

/**
 * Apply a two-qubit gate to sites a and b (a < b).
 * Non-adjacent pairs are handled by a SWAP network: bubble b down to a+1,
 * apply the gate, then restore. The net effect on the state is exact.
 */
export function mpsApply2(mps: MPS, a: number, b: number, gate: Gate4x4, maxBond: number): MPS {
  if (b === a + 1) return mpsApply2Adjacent(mps, a, gate, maxBond)
  // Bubble qubit-b leftward to position a+1
  for (let q = b - 1; q > a; q--) mps = mpsApply2Adjacent(mps, q, SWAP4, maxBond)
  mps = mpsApply2Adjacent(mps, a, gate, maxBond)
  // Restore
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
  let state: Complex[] = [ONE]  // left-boundary vector, starts as scalar 1
  let result = 0n

  for (let q = 0; q < n; q++) {
    const T = mps[q]!, chiR = T[0]![0]!.length

    // v[p][r] = sum_l state[l] * T[l][p][r]
    const v: [Complex[], Complex[]] = [
      new Array<Complex>(chiR).fill(ZERO),
      new Array<Complex>(chiR).fill(ZERO),
    ]
    for (let l = 0; l < state.length; l++) {
      const sl = state[l]!
      for (let p = 0; p < 2; p++) {
        for (let r = 0; r < chiR; r++) {
          v[p]![r] = add(v[p]![r]!, mul(sl, T[l]![p]![r]!))
        }
      }
    }

    const prob0 = v[0]!.reduce((s, amp) => s + norm2(amp), 0)
    const prob1 = v[1]!.reduce((s, amp) => s + norm2(amp), 0)
    const bit = rand() < prob0 / (prob0 + prob1) ? 0 : 1

    if (bit === 1) result |= 1n << BigInt(q)
    const inv = 1 / Math.sqrt(bit === 0 ? prob0 : prob1)
    state = v[bit]!.map(amp => scale(inv, amp))
  }

  return result
}
