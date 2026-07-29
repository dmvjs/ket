/**
 * Dense density-matrix backend.
 *
 * The sparse `Map<bigint, Complex>` in `density.ts` is the right representation
 * for a near-pure state under light noise, where ρ has few non-zero entries. It
 * fails badly once ρ fills in: at n=12 the map needs 4¹² = 16.7M boxed entries
 * with BigInt keys, which exhausts a 4 GB heap before finishing.
 *
 * Here ρ lives in one contiguous `Float64Array`, real and imaginary interleaved.
 * Entry (r, c) occupies `data[2·(r·dim + c)]` and the slot after it. The same
 * n=12 matrix is then 268 MB of unboxed f64 and every channel is a flat loop.
 *
 * Row/column index convention matches the sparse backend: qubit q is bit q of
 * both the row and column index.
 */

import type { Complex } from './complex.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'

/** Dense ρ over n qubits. `data.length === 2·dim²` with `dim = 2ⁿ`. */
export interface DenseDM {
  readonly n: number
  readonly dim: number
  readonly data: Float64Array
}

/**
 * Largest n this backend will allocate by default: 4¹² entries × 16 bytes =
 * 256 MiB. n=13 would need 1 GiB, which is not a reasonable default. Override
 * per call with `DenseOptions`.
 */
export const MAX_DENSE_DM_QUBITS = 12

/** Flat offset of entry (r, c). */
const at = (dim: number, r: number, c: number): number => ((r * dim + c) << 1)

/** Allocate a zeroed ρ of the same shape. */
const blank = (d: DenseDM): Float64Array => new Float64Array(d.data.length)

/** |0…0⟩⟨0…0| over n qubits. */
export function denseDmZero(n: number): DenseDM {
  const dim = 1 << n
  const data = new Float64Array(2 * dim * dim)
  data[0] = 1
  return { n, dim, data }
}

/** Materialise a sparse ρ (key = (row << n) | col) as dense. */
export function dmFromSparse(dm: Map<bigint, Complex>, n: number): DenseDM {
  const dim = 1 << n
  const shift = BigInt(n)
  const mask = (1n << shift) - 1n
  const out: DenseDM = { n, dim, data: new Float64Array(2 * dim * dim) }
  for (const [k, v] of dm) {
    const r = Number(k >> shift), c = Number(k & mask)
    const p = at(dim, r, c)
    out.data[p] = v.re
    out.data[p + 1] = v.im
  }
  return out
}

/** Convert back to the sparse map, dropping negligible entries as the sparse backend does. */
export function dmToSparse(d: DenseDM): Map<bigint, Complex> {
  const out = new Map<bigint, Complex>()
  const { dim, data, n } = d
  const shift = BigInt(n)
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      const p = at(dim, r, c)
      const re = data[p]!, im = data[p + 1]!
      if (re * re + im * im >= 1e-14) out.set((BigInt(r) << shift) | BigInt(c), { re, im })
    }
  }
  return out
}

/** Number of non-negligible entries. */
export function denseDmNnz(d: DenseDM): number {
  const { data } = d
  let count = 0
  for (let i = 0; i < data.length; i += 2) {
    const re = data[i]!, im = data[i + 1]!
    if (re * re + im * im >= 1e-14) count++
  }
  return count
}

/** Read entry (r, c). */
export function denseDmGet(d: DenseDM, r: number, c: number): Complex {
  const p = at(d.dim, r, c)
  return { re: d.data[p]!, im: d.data[p + 1]! }
}

// ── Unitary evolution ─────────────────────────────────────────────────────────

/**
 * ρ → UρU† for a single-qubit gate on q.
 *
 * Visits each 2×2 block once, from the (row bit = 0, col bit = 0) corner:
 * left-multiply the row pair by U, then right-multiply the column pair by U†.
 */
export function ddSingle(d: DenseDM, q: number, [[a, b], [c, e]]: Gate2x2): void {
  const { dim, data } = d
  const m = 1 << q
  const ar = a.re, ai = a.im, br = b.re, bi = b.im
  const cr = c.re, ci = c.im, dr = e.re, di = e.im

  for (let r0 = 0; r0 < dim; r0++) {
    if ((r0 & m) !== 0) continue
    const r1 = r0 | m
    for (let c0 = 0; c0 < dim; c0++) {
      if ((c0 & m) !== 0) continue
      const c1 = c0 | m
      const p00 = at(dim, r0, c0), p01 = at(dim, r0, c1)
      const p10 = at(dim, r1, c0), p11 = at(dim, r1, c1)

      const x00r = data[p00]!, x00i = data[p00 + 1]!
      const x01r = data[p01]!, x01i = data[p01 + 1]!
      const x10r = data[p10]!, x10i = data[p10 + 1]!
      const x11r = data[p11]!, x11i = data[p11 + 1]!

      // t = U·ρ  (mix rows)
      const t00r = ar * x00r - ai * x00i + br * x10r - bi * x10i
      const t00i = ar * x00i + ai * x00r + br * x10i + bi * x10r
      const t01r = ar * x01r - ai * x01i + br * x11r - bi * x11i
      const t01i = ar * x01i + ai * x01r + br * x11i + bi * x11r
      const t10r = cr * x00r - ci * x00i + dr * x10r - di * x10i
      const t10i = cr * x00i + ci * x00r + dr * x10i + di * x10r
      const t11r = cr * x01r - ci * x01i + dr * x11r - di * x11i
      const t11i = cr * x01i + ci * x01r + dr * x11i + di * x11r

      // t·U†  (mix columns; U†[j][k] = conj(U[k][j]))
      data[p00]     = ar * t00r + ai * t00i + br * t01r + bi * t01i
      data[p00 + 1] = ar * t00i - ai * t00r + br * t01i - bi * t01r
      data[p01]     = cr * t00r + ci * t00i + dr * t01r + di * t01i
      data[p01 + 1] = cr * t00i - ci * t00r + dr * t01i - di * t01r
      data[p10]     = ar * t10r + ai * t10i + br * t11r + bi * t11i
      data[p10 + 1] = ar * t10i - ai * t10r + br * t11i - bi * t11r
      data[p11]     = cr * t10r + ci * t10i + dr * t11r + di * t11i
      data[p11 + 1] = cr * t10i - ci * t10r + dr * t11i - di * t11r
    }
  }
}

/** ρ → GρG† for a 4×4 gate on (a, b), with `a` the MSB of the local index. */
export function ddTwo(d: DenseDM, qa: number, qb: number, gate: Gate4x4): void {
  ddUnitaryN(d, [qa, qb], gate as unknown as readonly (readonly Complex[])[])
}

/** ρ → PρPᵀ for a pure index permutation (CNOT, SWAP, Toffoli, CSWAP). */
export function ddPerm(d: DenseDM, f: (i: number) => number): void {
  const { dim, data } = d
  const out = blank(d)
  // Precompute the permutation once — f is called 2·dim times, not 2·dim².
  const map = new Int32Array(dim)
  for (let i = 0; i < dim; i++) map[i] = f(i)
  for (let r = 0; r < dim; r++) {
    const fr = map[r]!
    for (let c = 0; c < dim; c++) {
      const src = at(dim, r, c), dst = at(dim, fr, map[c]!)
      out[dst] = data[src]!
      out[dst + 1] = data[src + 1]!
    }
  }
  data.set(out)
}

/**
 * ρ → UρU† for a 2^k × 2^k unitary on `qs` (qs[0] = MSB of the local index).
 *
 * Gathers each local block, applies U on the left and U† on the right, and
 * writes it back. Scratch buffers are allocated once and reused per block.
 */
export function ddUnitaryN(d: DenseDM, qs: readonly number[], matrix: readonly (readonly Complex[])[]): void {
  const { dim, data } = d
  const k = qs.length
  const sz = 1 << k
  const masks = qs.map(q => 1 << q)
  const all = masks.reduce((acc, m) => acc | m, 0)

  // local index -> index offset contributed by the target qubits (qs[0] = MSB)
  const offset = new Int32Array(sz)
  for (let i = 0; i < sz; i++) {
    let o = 0
    for (let bit = 0; bit < k; bit++) if ((i >> (k - 1 - bit)) & 1) o |= masks[bit]!
    offset[i] = o
  }

  const gr = new Float64Array(sz * sz), gi = new Float64Array(sz * sz)
  for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) {
    const v = matrix[i]![j]!
    gr[i * sz + j] = v.re; gi[i * sz + j] = v.im
  }

  const bR = new Float64Array(sz * sz), bI = new Float64Array(sz * sz)
  const tR = new Float64Array(sz * sz), tI = new Float64Array(sz * sz)

  for (let rBase = 0; rBase < dim; rBase++) {
    if ((rBase & all) !== 0) continue
    for (let cBase = 0; cBase < dim; cBase++) {
      if ((cBase & all) !== 0) continue

      for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) {
        const p = at(dim, rBase | offset[i]!, cBase | offset[j]!)
        bR[i * sz + j] = data[p]!; bI[i * sz + j] = data[p + 1]!
      }

      // t = U·ρ
      for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) {
        let sr = 0, si = 0
        for (let m = 0; m < sz; m++) {
          const ur = gr[i * sz + m]!, ui = gi[i * sz + m]!
          const xr = bR[m * sz + j]!, xi = bI[m * sz + j]!
          sr += ur * xr - ui * xi
          si += ur * xi + ui * xr
        }
        tR[i * sz + j] = sr; tI[i * sz + j] = si
      }

      // ρ' = t·U†  where U†[m][j] = conj(U[j][m])
      for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) {
        let sr = 0, si = 0
        for (let m = 0; m < sz; m++) {
          const ur = gr[j * sz + m]!, ui = gi[j * sz + m]!   // conj applied inline
          const xr = tR[i * sz + m]!, xi = tI[i * sz + m]!
          sr += xr * ur + xi * ui
          si += xi * ur - xr * ui
        }
        const p = at(dim, rBase | offset[i]!, cBase | offset[j]!)
        data[p] = sr; data[p + 1] = si
      }
    }
  }
}

// ── Noise channels ────────────────────────────────────────────────────────────

/**
 * Single-qubit depolarizing channel, closed form per 2×2 block:
 *   same parity  → (1−2p/3)·ρ[r][c] + (2p/3)·ρ[r^m][c^m]
 *   cross parity → (1−4p/3)·ρ[r][c]
 */
export function ddDepolarize1(d: DenseDM, q: number, p: number): void {
  if (p <= 0) return
  const { dim, data } = d
  const m = 1 << q
  const sa = 1 - 2 * p / 3, sb = 2 * p / 3
  const cf = 1 - 4 * p / 3

  for (let r0 = 0; r0 < dim; r0++) {
    if ((r0 & m) !== 0) continue
    const r1 = r0 | m
    for (let c0 = 0; c0 < dim; c0++) {
      if ((c0 & m) !== 0) continue
      const c1 = c0 | m
      const p00 = at(dim, r0, c0), p11 = at(dim, r1, c1)
      const p01 = at(dim, r0, c1), p10 = at(dim, r1, c0)

      const a0r = data[p00]!, a0i = data[p00 + 1]!
      const a1r = data[p11]!, a1i = data[p11 + 1]!
      data[p00]     = sa * a0r + sb * a1r
      data[p00 + 1] = sa * a0i + sb * a1i
      data[p11]     = sa * a1r + sb * a0r
      data[p11 + 1] = sa * a1i + sb * a0i

      data[p01]! *= cf; data[p01 + 1]! *= cf
      data[p10]! *= cf; data[p10 + 1]! *= cf
    }
  }
}

/**
 * Two-qubit depolarizing channel: (1−p)ρ + (p/15)·Σ_{P≠II} PρP†.
 *
 * `pauli15` carries [flipA, flipB, zA, zB] per non-identity Pauli, matching the
 * table in `density.ts`. Accumulates into a fresh buffer because each source
 * entry scatters to sixteen destinations.
 */
export function ddDepolarize2(
  d: DenseDM, qa: number, qb: number, p: number,
  pauli15: readonly (readonly [0|1, 0|1, 0|1, 0|1])[],
): void {
  if (p <= 0) return
  const { dim, data } = d
  const out = blank(d)
  const ma = 1 << qa, mb = 1 << qb
  const w1 = 1 - p, w2 = p / 15

  for (let r = 0; r < dim; r++) {
    const bar = (r >> qa) & 1, bbr = (r >> qb) & 1
    for (let c = 0; c < dim; c++) {
      const src = at(dim, r, c)
      const vr = data[src]!, vi = data[src + 1]!
      if (vr === 0 && vi === 0) continue

      out[src]! += w1 * vr
      out[src + 1]! += w1 * vi

      const bac = (c >> qa) & 1, bbc = (c >> qb) & 1
      for (const [fa, fb, za, zb] of pauli15) {
        const perm = (fa ? ma : 0) | (fb ? mb : 0)
        const parity = (za * (bar ^ bac)) ^ (zb * (bbr ^ bbc))
        const s = parity ? -w2 : w2
        const dst = at(dim, r ^ perm, c ^ perm)
        out[dst]! += s * vr
        out[dst + 1]! += s * vi
      }
    }
  }
  data.set(out)
}

/**
 * Amplitude damping on q.
 *   (0,0) → ρ[0][0] + γ·ρ[1][1]
 *   (1,1) → (1−γ)·ρ[1][1]
 *   off-diagonal in q → √(1−γ)·ρ
 */
export function ddAmplitudeDamping1(d: DenseDM, q: number, gamma: number): void {
  if (gamma <= 0) return
  const { dim, data } = d
  const m = 1 << q
  const sqG = Math.sqrt(1 - gamma)

  for (let r0 = 0; r0 < dim; r0++) {
    if ((r0 & m) !== 0) continue
    const r1 = r0 | m
    for (let c0 = 0; c0 < dim; c0++) {
      if ((c0 & m) !== 0) continue
      const c1 = c0 | m
      const p00 = at(dim, r0, c0), p11 = at(dim, r1, c1)
      const p01 = at(dim, r0, c1), p10 = at(dim, r1, c0)

      const v11r = data[p11]!, v11i = data[p11 + 1]!
      data[p00]! += gamma * v11r
      data[p00 + 1]! += gamma * v11i
      data[p11]     = (1 - gamma) * v11r
      data[p11 + 1] = (1 - gamma) * v11i

      data[p01]! *= sqG; data[p01 + 1]! *= sqG
      data[p10]! *= sqG; data[p10 + 1]! *= sqG
    }
  }
}

/** Pure dephasing on q: diagonal-in-q entries unchanged, off-diagonal scaled by √(1−λ). */
export function ddPhaseDamping1(d: DenseDM, q: number, lambda: number): void {
  if (lambda <= 0) return
  const { dim, data } = d
  const sqL = Math.sqrt(1 - lambda)
  for (let r = 0; r < dim; r++) {
    const rq = (r >> q) & 1
    for (let c = 0; c < dim; c++) {
      if (rq === ((c >> q) & 1)) continue
      const p = at(dim, r, c)
      data[p]! *= sqL; data[p + 1]! *= sqL
    }
  }
}

/** Single-qubit Kraus channel ε(ρ) = Σ_k K_k ρ K_k† on q. */
export function ddKraus1(d: DenseDM, q: number, kraus: readonly Gate2x2[]): void {
  const { dim, data } = d
  const m = 1 << q

  for (let r0 = 0; r0 < dim; r0++) {
    if ((r0 & m) !== 0) continue
    const r1 = r0 | m
    for (let c0 = 0; c0 < dim; c0++) {
      if ((c0 & m) !== 0) continue
      const c1 = c0 | m
      const pos = [at(dim, r0, c0), at(dim, r0, c1), at(dim, r1, c0), at(dim, r1, c1)]
      const xr = [data[pos[0]!]!, data[pos[1]!]!, data[pos[2]!]!, data[pos[3]!]!]
      const xi = [data[pos[0]! + 1]!, data[pos[1]! + 1]!, data[pos[2]! + 1]!, data[pos[3]! + 1]!]

      for (let iq = 0; iq < 2; iq++) {
        for (let jq = 0; jq < 2; jq++) {
          let re = 0, im = 0
          for (const K of kraus) {
            for (let ip = 0; ip < 2; ip++) {
              for (let jp = 0; jp < 2; jp++) {
                const kip = K[iq]![ip]!, kjp = K[jq]![jp]!
                const kRe = kip.re * kjp.re + kip.im * kjp.im
                const kIm = kip.im * kjp.re - kip.re * kjp.im
                const idx = ip * 2 + jp
                re += kRe * xr[idx]! - kIm * xi[idx]!
                im += kRe * xi[idx]! + kIm * xr[idx]!
              }
            }
          }
          const p = pos[iq * 2 + jq]!
          data[p] = re; data[p + 1] = im
        }
      }
    }
  }
}

/** Two-qubit Kraus channel ε(ρ) = Σ_k K_k ρ K_k† on (a, b). */
export function ddKraus2(d: DenseDM, qa: number, qb: number, kraus: readonly Gate4x4[]): void {
  const { dim, data } = d
  const ma = 1 << qa, mb = 1 << qb
  const all = ma | mb
  const off = [0, mb, ma, ma | mb]   // local index -> offset, qa is MSB

  const xr = new Float64Array(16), xi = new Float64Array(16)

  for (let rBase = 0; rBase < dim; rBase++) {
    if ((rBase & all) !== 0) continue
    for (let cBase = 0; cBase < dim; cBase++) {
      if ((cBase & all) !== 0) continue

      for (let ri = 0; ri < 4; ri++) for (let ci = 0; ci < 4; ci++) {
        const p = at(dim, rBase | off[ri]!, cBase | off[ci]!)
        xr[ri * 4 + ci] = data[p]!; xi[ri * 4 + ci] = data[p + 1]!
      }

      for (let ri = 0; ri < 4; ri++) {
        for (let ci = 0; ci < 4; ci++) {
          let re = 0, im = 0
          for (const K of kraus) {
            for (let rp = 0; rp < 4; rp++) {
              for (let cp = 0; cp < 4; cp++) {
                const kri = K[ri]![rp]!, kci = K[ci]![cp]!
                const kRe = kri.re * kci.re + kri.im * kci.im
                const kIm = kri.im * kci.re - kri.re * kci.im
                const idx = rp * 4 + cp
                re += kRe * xr[idx]! - kIm * xi[idx]!
                im += kRe * xi[idx]! + kIm * xr[idx]!
              }
            }
          }
          const p = at(dim, rBase | off[ri]!, cBase | off[ci]!)
          data[p] = re; data[p + 1] = im
        }
      }
    }
  }
}
