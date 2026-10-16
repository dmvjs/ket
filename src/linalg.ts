/**
 * Dense complex linear algebra, for the tensor-network engines.
 *
 * ket had a singular value decomposition already, but welded into the MPS
 * trajectory class: a private method over preallocated buffers, sized for one
 * bond cut, with the truncation policy baked in. Nothing else could call it. A
 * 2-D tensor network needs the same decomposition on matrices of arbitrary shape,
 * so this is that algorithm as a function.
 *
 * One-sided Jacobi throughout. It is slower than a bidiagonalising SVD on large
 * matrices and much easier to get right on complex input, and the matrices here
 * are bond-sized rather than large.
 *
 * Complex matrices are `Float64Array`s in row-major order with real and imaginary
 * parts interleaved, matching the convention in `tensor-network.ts`.
 */

/** A complex matrix: `rows × cols`, row-major, re/im interleaved. */
export interface CMatrix {
  data: Float64Array
  rows: number
  cols: number
}

export const cmatrix = (rows: number, cols: number): CMatrix =>
  ({ data: new Float64Array(rows * cols * 2), rows, cols })

export interface Svd {
  /** `rows × rank`, the left singular vectors. */
  u: CMatrix
  /** `rank` singular values, descending. */
  s: Float64Array
  /** `rank × cols`, already conjugate-transposed. */
  vt: CMatrix
  rank: number
  /** Sum of the squares of the singular values that were dropped. */
  discarded: number
}

/**
 * Singular value decomposition, `M = U · diag(s) · V†`.
 *
 * `maxRank` keeps only the largest singular values, which is the optimal
 * low-rank approximation in the Frobenius norm — the truncation every tensor
 * network approximation is built on. `discarded` reports what that cost, so a
 * caller can bound its own error rather than assume it away.
 *
 * `cutoff` drops singular values below that fraction of the largest, which
 * removes numerical dust that would otherwise become spurious bond dimension.
 */
export function svd(
  m: CMatrix,
  { maxRank = Infinity, cutoff = 1e-14, sweeps = 30 }: { maxRank?: number; cutoff?: number; sweeps?: number } = {},
): Svd {
  // One-sided Jacobi orthogonalises *columns*, so a wide matrix is transposed and
  // the factors swapped back at the end.
  if (m.rows < m.cols) {
    const t = conjTranspose(m)
    const r = svd(t, { maxRank, cutoff, sweeps })
    return { u: conjTranspose(r.vt), s: r.s, vt: conjTranspose(r.u), rank: r.rank, discarded: r.discarded }
  }

  const { rows, cols } = m
  // Column-major working copy: A starts as M and ends as U·diag(s).
  const a = new Float64Array(rows * cols * 2)
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    a[(j * rows + i) * 2] = m.data[(i * cols + j) * 2]!
    a[(j * rows + i) * 2 + 1] = m.data[(i * cols + j) * 2 + 1]!
  }
  // V accumulates the rotations, column-major, cols × cols, starting at identity.
  const v = new Float64Array(cols * cols * 2)
  for (let k = 0; k < cols; k++) v[(k * cols + k) * 2] = 1

  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0
    for (let p = 0; p < cols - 1; p++) {
      for (let q = p + 1; q < cols; q++) {
        // Gram entries for the column pair: G_pp, G_qq real, G_pq complex.
        let gpp = 0, gqq = 0, gre = 0, gim = 0
        for (let i = 0; i < rows; i++) {
          const pr = a[(p * rows + i) * 2]!, pi = a[(p * rows + i) * 2 + 1]!
          const qr = a[(q * rows + i) * 2]!, qi = a[(q * rows + i) * 2 + 1]!
          gpp += pr * pr + pi * pi
          gqq += qr * qr + qi * qi
          gre += pr * qr + pi * qi
          gim += pr * qi - pi * qr
        }
        const r = Math.hypot(gre, gim)
        if (r > off) off = r
        if (r < 1e-15 * Math.sqrt(gpp * gqq) + 1e-300) continue

        // Rotate the phase out first, then apply the real Jacobi rotation that
        // zeros the off-diagonal.
        const er = gre / r, ei = -gim / r
        const tau = (gqq - gpp) / (2 * r)
        const t = tau >= 0 ? -1 / (tau + Math.sqrt(1 + tau * tau)) : 1 / (-tau + Math.sqrt(1 + tau * tau))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = t * c

        const rotate = (buf: Float64Array, len: number): void => {
          for (let i = 0; i < len; i++) {
            const pIdx = (p * len + i) * 2, qIdx = (q * len + i) * 2
            const pr = buf[pIdx]!, pi = buf[pIdx + 1]!
            const qr = buf[qIdx]!, qi = buf[qIdx + 1]!
            // e = er + i·ei applied to column q before mixing.
            const eqr = qr * er - qi * ei, eqi = qr * ei + qi * er
            buf[pIdx] = c * pr + s * eqr
            buf[pIdx + 1] = c * pi + s * eqi
            buf[qIdx] = -s * pr + c * eqr
            buf[qIdx + 1] = -s * pi + c * eqi
          }
        }
        rotate(a, rows)
        rotate(v, cols)
      }
    }
    if (off < 1e-15) break
  }

  // Column norms are the singular values; sort descending.
  const norms = new Float64Array(cols)
  for (let j = 0; j < cols; j++) {
    let n2 = 0
    for (let i = 0; i < rows; i++) {
      const re = a[(j * rows + i) * 2]!, im = a[(j * rows + i) * 2 + 1]!
      n2 += re * re + im * im
    }
    norms[j] = Math.sqrt(n2)
  }
  const order = Array.from({ length: cols }, (_, j) => j).sort((x, y) => norms[y]! - norms[x]!)

  const largest = norms[order[0]!] ?? 0
  let rank = 0
  for (const j of order) {
    if (rank >= maxRank) break
    if (norms[j]! <= cutoff * largest) break
    rank++
  }
  rank = Math.max(rank, 1)

  let discarded = 0
  for (let k = rank; k < cols; k++) discarded += norms[order[k]!]! ** 2

  const u = cmatrix(rows, rank)
  const s = new Float64Array(rank)
  const vt = cmatrix(rank, cols)
  for (let k = 0; k < rank; k++) {
    const j = order[k]!
    const sigma = norms[j]!
    s[k] = sigma
    const inv = sigma > 1e-300 ? 1 / sigma : 0
    for (let i = 0; i < rows; i++) {
      u.data[(i * rank + k) * 2] = a[(j * rows + i) * 2]! * inv
      u.data[(i * rank + k) * 2 + 1] = a[(j * rows + i) * 2 + 1]! * inv
    }
    // Row k of V† is the conjugate of column j of V.
    for (let i = 0; i < cols; i++) {
      vt.data[(k * cols + i) * 2] = v[(j * cols + i) * 2]!
      vt.data[(k * cols + i) * 2 + 1] = -v[(j * cols + i) * 2 + 1]!
    }
  }
  return { u, s, vt, rank, discarded }
}

/** `M†`. */
export function conjTranspose(m: CMatrix): CMatrix {
  const out = cmatrix(m.cols, m.rows)
  for (let i = 0; i < m.rows; i++) for (let j = 0; j < m.cols; j++) {
    out.data[(j * m.rows + i) * 2] = m.data[(i * m.cols + j) * 2]!
    out.data[(j * m.rows + i) * 2 + 1] = -m.data[(i * m.cols + j) * 2 + 1]!
  }
  return out
}

/** `A · B`. */
export function matmul(a: CMatrix, b: CMatrix): CMatrix {
  if (a.cols !== b.rows) throw new RangeError(`matmul: ${a.rows}x${a.cols} times ${b.rows}x${b.cols}`)
  const out = cmatrix(a.rows, b.cols)
  for (let i = 0; i < a.rows; i++) for (let k = 0; k < a.cols; k++) {
    const ar = a.data[(i * a.cols + k) * 2]!, ai = a.data[(i * a.cols + k) * 2 + 1]!
    if (ar === 0 && ai === 0) continue
    for (let j = 0; j < b.cols; j++) {
      const br = b.data[(k * b.cols + j) * 2]!, bi = b.data[(k * b.cols + j) * 2 + 1]!
      out.data[(i * b.cols + j) * 2] = out.data[(i * b.cols + j) * 2]! + (ar * br - ai * bi)
      out.data[(i * b.cols + j) * 2 + 1] = out.data[(i * b.cols + j) * 2 + 1]! + (ar * bi + ai * br)
    }
  }
  return out
}
