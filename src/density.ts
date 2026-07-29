/**
 * Exact density matrix simulation for mixed-state and noise research.
 *
 * Representation: sparse Map<bigint, Complex> where key = (row << n) | col,
 * with n = qubits.  Only entries with |ρ[r][c]|² > 1e-14 are stored.
 *
 * Complexity: O(4ⁿ) in the worst case — practical up to ~12 qubits.
 */

import { add, Complex, conj, isNegligible, mul, ZERO } from './complex.js'
import { Gate2x2, Gate4x4 } from './statevector.js'
import {
  ddAmplitudeDamping1, ddDepolarize1, ddDepolarize2, ddKraus1, ddKraus2, ddPerm,
  ddPhaseDamping1, ddSingle, ddTwo, ddUnitaryN, denseDmGet, dmFromSparse,
  MAX_DENSE_DM_QUBITS, type DenseDM,
} from './density-dense.js'
import { controlledGate } from './mps.js'

// ─── Sparse DM type ────────────────────────────────────────────────────────

/** Sparse density matrix.  Key = (row << n) | col. */
type DM = Map<bigint, Complex>

function dmGet(dm: DM, shift: bigint, r: bigint, c: bigint): Complex {
  return dm.get((r << shift) | c) ?? ZERO
}

function dmSet(dm: DM, shift: bigint, r: bigint, c: bigint, v: Complex): void {
  const k = (r << shift) | c
  if (!isNegligible(v)) dm.set(k, v)
  else dm.delete(k)
}

function dmAcc(dm: DM, shift: bigint, r: bigint, c: bigint, v: Complex): void {
  const k = (r << shift) | c
  const ex = dm.get(k)
  const nx = ex ? add(ex, v) : v
  if (!isNegligible(nx)) dm.set(k, nx)
  else dm.delete(k)
}

// ─── Unitary evolution ─────────────────────────────────────────────────────

/**
 * ρ → U_q ρ U_q†  — single-qubit gate on qubit q.
 *
 * Two-step per context (r_base, c_base):
 *   1. Left-multiply row part by U
 *   2. Right-multiply col part by U†
 */
function spSingle(dm: DM, n: number, q: number, [[a, b], [c, d]]: Gate2x2): DM {
  const next: DM = new Map()
  const shift    = BigInt(n)
  const dimMask  = (1n << shift) - 1n
  const qMask    = 1n << BigInt(q)
  const keyRMask = qMask << shift
  const keyCMask = qMask
  const seen     = new Set<bigint>()
  const ca = conj(a), cb = conj(b), cc = conj(c), cd = conj(d)

  for (const k of dm.keys()) {
    const ctx = k & ~keyRMask & ~keyCMask
    if (seen.has(ctx)) continue
    seen.add(ctx)

    const r0 = ctx >> shift,   c0 = ctx & dimMask
    const r1 = r0 | qMask,     c1 = c0 | qMask

    const p00 = dmGet(dm, shift, r0, c0), p01 = dmGet(dm, shift, r0, c1)
    const p10 = dmGet(dm, shift, r1, c0), p11 = dmGet(dm, shift, r1, c1)

    // Step 1: left-multiply rows by U
    const t00 = add(mul(a, p00), mul(b, p10)), t01 = add(mul(a, p01), mul(b, p11))
    const t10 = add(mul(c, p00), mul(d, p10)), t11 = add(mul(c, p01), mul(d, p11))

    // Step 2: right-multiply cols by U†  (U†[0][0]=a*, U†[1][0]=b*, U†[0][1]=c*, U†[1][1]=d*)
    dmSet(next, shift, r0, c0, add(mul(ca, t00), mul(cb, t01)))
    dmSet(next, shift, r0, c1, add(mul(cc, t00), mul(cd, t01)))
    dmSet(next, shift, r1, c0, add(mul(ca, t10), mul(cb, t11)))
    dmSet(next, shift, r1, c1, add(mul(cc, t10), mul(cd, t11)))
  }
  return next
}

/** ρ → G_{ab} ρ G_{ab}†  — 4×4 two-qubit gate on qubits a (MSB), b. */
function spTwo(dm: DM, n: number, a: number, b: number, gate: Gate4x4): DM {
  const next: DM = new Map()
  const shift   = BigInt(n)
  const dimMask = (1n << shift) - 1n
  const ma = 1n << BigInt(a), mb = 1n << BigInt(b)
  const seen = new Set<bigint>()

  for (const k of dm.keys()) {
    const ctx = k & ~(ma << shift) & ~(mb << shift) & ~ma & ~mb
    if (seen.has(ctx)) continue
    seen.add(ctx)

    const rb = ctx >> shift, cb = ctx & dimMask
    const rowB = [rb, rb | mb, rb | ma, rb | ma | mb]   // |00⟩,|01⟩,|10⟩,|11⟩ (a=MSB)
    const colB = [cb, cb | mb, cb | ma, cb | ma | mb]

    // Read 4×4 block
    const p: Complex[][] = Array.from({ length: 4 }, (_, ri) =>
      Array.from({ length: 4 }, (_, ci) => dmGet(dm, shift, rowB[ri]!, colB[ci]!))
    )
    // Left multiply: t[ri][ci] = Σ_k gate[ri][k] · p[k][ci]
    const t: Complex[][] = Array.from({ length: 4 }, (_, ri) =>
      Array.from({ length: 4 }, (_, ci) => {
        let v = ZERO
        for (let k2 = 0; k2 < 4; k2++) v = add(v, mul(gate[ri]![k2]!, p[k2]![ci]!))
        return v
      })
    )
    // Right multiply by gate†: new[ri][ci] = Σ_k t[ri][k] · conj(gate[ci][k])
    for (let ri = 0; ri < 4; ri++) {
      for (let ci = 0; ci < 4; ci++) {
        let v = ZERO
        for (let k2 = 0; k2 < 4; k2++) v = add(v, mul(t[ri]![k2]!, conj(gate[ci]![k2]!)))
        dmSet(next, shift, rowB[ri]!, colB[ci]!, v)
      }
    }
  }
  return next
}

/** ρ → perm(ρ)  — pure-permutation unitary (CNOT, SWAP, Toffoli, CSwap). */
function spPerm(dm: DM, n: number, f: (i: bigint) => bigint): DM {
  const shift = BigInt(n), dimMask = (1n << shift) - 1n
  const next: DM = new Map()
  for (const [k, v] of dm) {
    const r = k >> shift, c = k & dimMask
    next.set((f(r) << shift) | f(c), v)
  }
  return next
}

/** ρ → U ρ U†  — N-qubit unitary on qubits `qs` (qs[0] = MSB of local index). */
function spUnitaryN(dm: DM, n: number, qs: readonly number[], matrix: readonly (readonly Complex[])[]): DM {
  const next: DM = new Map()
  const shift    = BigInt(n)
  const dimMask  = (1n << shift) - 1n
  const localDim = 1 << qs.length
  const masks    = qs.map(q => 1n << BigInt(q))
  const allMask  = masks.reduce((a, b) => a | b, 0n)
  const seen     = new Set<bigint>()

  // Expand a base index (with qubit bits zeroed) into all 2^N local variants.
  // qs[0] is MSB of the local index i, matching the matrix convention.
  const buildBases = (base: bigint): bigint[] =>
    Array.from({ length: localDim }, (_, i) => {
      let b = base
      for (let bit = 0; bit < qs.length; bit++) {
        if ((i >> (qs.length - 1 - bit)) & 1) b |= masks[bit]!
      }
      return b
    })

  for (const key of dm.keys()) {
    const ctx = key & ~(allMask << shift) & ~allMask
    if (seen.has(ctx)) continue
    seen.add(ctx)

    const rowBases = buildBases(ctx >> shift)
    const colBases = buildBases(ctx & dimMask)

    const p: Complex[][] = Array.from({ length: localDim }, (_, ri) =>
      Array.from({ length: localDim }, (_, ci) => dmGet(dm, shift, rowBases[ri]!, colBases[ci]!))
    )
    // Left multiply: t = U · p
    const t: Complex[][] = Array.from({ length: localDim }, (_, ri) =>
      Array.from({ length: localDim }, (_, ci) => {
        let v = ZERO
        for (let j = 0; j < localDim; j++) v = add(v, mul(matrix[ri]![j]!, p[j]![ci]!))
        return v
      })
    )
    // Right multiply by U†: result = t · U†
    for (let ri = 0; ri < localDim; ri++) {
      for (let ci = 0; ci < localDim; ci++) {
        let v = ZERO
        for (let j = 0; j < localDim; j++) v = add(v, mul(t[ri]![j]!, conj(matrix[ci]![j]!)))
        dmSet(next, shift, rowBases[ri]!, colBases[ci]!, v)
      }
    }
  }
  return next
}

// ─── Noise channels ────────────────────────────────────────────────────────

/**
 * Single-qubit depolarizing channel: ε(ρ) = (1−p)ρ + (p/3)(XρX + YρY + ZρZ).
 *
 * Efficient closed-form per context (no matrix multiply):
 *   - same parity  (bit_q(r) == bit_q(c)):  (1−2p/3)·ρ[r][c] + (2p/3)·ρ[r^m][c^m]
 *   - cross parity (bit_q(r) != bit_q(c)):  (1−4p/3)·ρ[r][c]
 */
function spDepolarize1(dm: DM, n: number, q: number, p: number): DM {
  if (p <= 0) return dm
  const next: DM = new Map()
  const shift    = BigInt(n)
  const dimMask  = (1n << shift) - 1n
  const qMask    = 1n << BigInt(q)
  const keyRMask = qMask << shift
  const keyCMask = qMask
  const seen     = new Set<bigint>()
  const sa = 1 - 2 * p / 3, sb = 2 * p / 3   // same-parity coefficients
  const cf = 1 - 4 * p / 3                     // cross-parity coefficient

  for (const k of dm.keys()) {
    const ctx = k & ~keyRMask & ~keyCMask
    if (seen.has(ctx)) continue
    seen.add(ctx)

    const r0 = ctx >> shift, c0 = ctx & dimMask
    const r1 = r0 | qMask,   c1 = c0 | qMask

    const p00 = dmGet(dm, shift, r0, c0), p11 = dmGet(dm, shift, r1, c1)
    dmSet(next, shift, r0, c0, { re: sa * p00.re + sb * p11.re, im: sa * p00.im + sb * p11.im })
    dmSet(next, shift, r1, c1, { re: sa * p11.re + sb * p00.re, im: sa * p11.im + sb * p00.im })

    const p01 = dmGet(dm, shift, r0, c1), p10 = dmGet(dm, shift, r1, c0)
    dmSet(next, shift, r0, c1, { re: cf * p01.re, im: cf * p01.im })
    dmSet(next, shift, r1, c0, { re: cf * p10.re, im: cf * p10.im })
  }
  return next
}

// Precomputed: [flip_a, flip_b, z_a, z_b] for each of the 15 non-identity 2-qubit Paulis.
// flip=1 → X or Y (bit permutation); z=1 → Y or Z (sign flip for same bit value).
const PAULI15: readonly [0|1, 0|1, 0|1, 0|1][] = [
  [0,1,0,0],[0,1,0,1],[0,0,0,1],              // I⊗X  I⊗Y  I⊗Z
  [1,0,0,0],[1,1,0,0],[1,1,0,1],[1,0,0,1],    // X⊗I  X⊗X  X⊗Y  X⊗Z
  [1,0,1,0],[1,1,1,0],[1,1,1,1],[1,0,1,1],    // Y⊗I  Y⊗X  Y⊗Y  Y⊗Z
  [0,0,1,0],[0,1,1,0],[0,1,1,1],[0,0,1,1],    // Z⊗I  Z⊗X  Z⊗Y  Z⊗Z
]

/**
 * Two-qubit depolarizing channel: ε(ρ) = (1−p)ρ + (p/15) Σ_{P≠II} PρP†.
 *
 * For each entry ρ[r][c], each Pauli Pa⊗Pb maps it to a phase · ρ[r^perm][c^perm].
 * The phase is (−1) raised to the XOR of the z-components and the bit values.
 */
function spDepolarize2(dm: DM, n: number, a: number, b: number, p: number): DM {
  if (p <= 0) return dm
  const next: DM = new Map()
  const shift   = BigInt(n)
  const dimMask = (1n << shift) - 1n
  const ma = 1n << BigInt(a), mb = 1n << BigInt(b)
  const w1 = 1 - p, w2 = p / 15

  for (const [k, v] of dm) {
    const r = k >> shift, c = k & dimMask
    dmAcc(next, shift, r, c, { re: w1 * v.re, im: w1 * v.im })

    const ba_r = Number((r >> BigInt(a)) & 1n), bb_r = Number((r >> BigInt(b)) & 1n)
    const ba_c = Number((c >> BigInt(a)) & 1n), bb_c = Number((c >> BigInt(b)) & 1n)

    for (const [fa, fb, za, zb] of PAULI15) {
      const perm  = (fa ? ma : 0n) | (fb ? mb : 0n)
      const parity = (za * (ba_r ^ ba_c)) ^ (zb * (bb_r ^ bb_c))
      const scale  = parity ? -w2 : w2
      dmAcc(next, shift, r ^ perm, c ^ perm, { re: scale * v.re, im: scale * v.im })
    }
  }
  return next
}

/**
 * Amplitude damping channel on qubit q: ε(ρ) = K0 ρ K0† + K1 ρ K1†
 * K0 = diag(1, √(1−γ)),  K1 = [[0,√γ],[0,0]].
 *
 * Effect on density matrix elements (r_q, c_q = bit q of row/col index):
 *   r_q=0, c_q=0: ρ[r][c] + γ·ρ[r|q][c|q]   (gain from |1⟩→|0⟩ decay)
 *   r_q=1, c_q=1: (1−γ)·ρ[r][c]
 *   r_q≠c_q:       √(1−γ)·ρ[r][c]             (damp off-diagonal)
 */
function spAmplitudeDamping1(dm: DM, n: number, q: number, gamma: number): DM {
  if (gamma <= 0) return dm
  const next: DM = new Map()
  const shift  = BigInt(n)
  const dimMsk = (1n << shift) - 1n
  const qMask  = 1n << BigInt(q)
  const sqG    = Math.sqrt(1 - gamma)
  const seen   = new Set<bigint>()

  for (const k of dm.keys()) {
    const ctx = k & ~(qMask << shift) & ~qMask
    if (seen.has(ctx)) continue
    seen.add(ctx)

    const r0 = ctx >> shift, c0 = ctx & dimMsk
    const r1 = r0 | qMask,   c1 = c0 | qMask

    // r_q=0, c_q=0: contribution from r_q=1,c_q=1 term via K1
    const v00 = dmGet(dm, shift, r0, c0), v11 = dmGet(dm, shift, r1, c1)
    dmSet(next, shift, r0, c0, { re: v00.re + gamma * v11.re, im: v00.im + gamma * v11.im })
    // r_q=1, c_q=1: scaled by (1-gamma)
    dmSet(next, shift, r1, c1, { re: (1 - gamma) * v11.re, im: (1 - gamma) * v11.im })
    // r_q=0, c_q=1 and r_q=1, c_q=0: damp by sqrt(1-gamma)
    const v01 = dmGet(dm, shift, r0, c1), v10 = dmGet(dm, shift, r1, c0)
    dmSet(next, shift, r0, c1, { re: sqG * v01.re, im: sqG * v01.im })
    dmSet(next, shift, r1, c0, { re: sqG * v10.re, im: sqG * v10.im })
  }
  return next
}

/**
 * Pure dephasing channel on qubit q: ε(ρ) = K0 ρ K0† + K1 ρ K1†
 * K0 = diag(1, √(1−λ)),  K1 = diag(0, √λ).
 *
 * Effect: diagonal elements unchanged; off-diagonal elements (r_q ≠ c_q) scaled by √(1−λ).
 */
function spPhaseDamping1(dm: DM, n: number, q: number, lambda: number): DM {
  if (lambda <= 0) return dm
  const sqL  = Math.sqrt(1 - lambda)
  const next: DM = new Map()
  const shift  = BigInt(n)
  const dimMsk = (1n << shift) - 1n

  for (const [k, v] of dm) {
    const r = k >> shift, c = k & dimMsk
    const rq = (r >> BigInt(q)) & 1n
    const cq = (c >> BigInt(q)) & 1n
    const s  = rq === cq ? 1 : sqL
    next.set(k, { re: s * v.re, im: s * v.im })
  }
  return next
}

/**
 * Apply a single-qubit Kraus channel ε(ρ) = Σ_k K_k ρ K_k† on qubit q.
 */
function spKraus1(dm: DM, n: number, q: number, kraus: readonly Gate2x2[]): DM {
  const shift  = BigInt(n)
  const dimMsk = (1n << shift) - 1n
  const qMask  = 1n << BigInt(q)
  const next: DM = new Map()
  const seen   = new Set<bigint>()

  for (const k of dm.keys()) {
    const ctx = k & ~(qMask << shift) & ~qMask
    if (seen.has(ctx)) continue
    seen.add(ctx)

    const r0 = ctx >> shift, c0 = ctx & dimMsk
    const r1 = r0 | qMask,   c1 = c0 | qMask

    // The four density matrix elements for this context
    const v00 = dmGet(dm, shift, r0, c0), v01 = dmGet(dm, shift, r0, c1)
    const v10 = dmGet(dm, shift, r1, c0), v11 = dmGet(dm, shift, r1, c1)
    const rho = [[v00, v01], [v10, v11]] as const

    // ε(ρ)[i_q,j_q] = Σ_k Σ_{i',j'} K_k[i_q,i'] ρ[i'][j'] K_k*[j_q,j']
    for (let iq = 0; iq < 2; iq++) {
      for (let jq = 0; jq < 2; jq++) {
        let re = 0, im = 0
        for (const K of kraus) {
          for (let ip = 0; ip < 2; ip++) {
            for (let jp = 0; jp < 2; jp++) {
              const kip = K[iq]![ip]!, kjp = K[jq]![jp]!
              const kRe = kip.re * kjp.re + kip.im * kjp.im  // K[iq,ip]·K*[jq,jp]
              const kIm = kip.im * kjp.re - kip.re * kjp.im
              const rij = rho[ip]![jp]!
              re += kRe * rij.re - kIm * rij.im
              im += kRe * rij.im + kIm * rij.re
            }
          }
        }
        const rOut = iq === 0 ? r0 : r1
        const cOut = jq === 0 ? c0 : c1
        dmAcc(next, shift, rOut, cOut, { re, im })
      }
    }
  }
  return next
}

/**
 * Apply a two-qubit Kraus channel ε(ρ) = Σ_k K_k ρ K_k† on qubits a and b.
 */
function spKraus2(dm: DM, n: number, a: number, b: number, kraus: readonly Gate4x4[]): DM {
  const shift  = BigInt(n)
  const dimMsk = (1n << shift) - 1n
  const ma = 1n << BigInt(a), mb = 1n << BigInt(b)
  const contexts = new Set<bigint>()
  const next: DM = new Map()

  for (const k of dm.keys()) {
    const r = k >> shift, c = k & dimMsk
    const ctx = (r & ~ma & ~mb) << shift | (c & ~ma & ~mb)
    contexts.add(ctx)
  }

  for (const ctx of contexts) {
    const rBase = ctx >> shift, cBase = ctx & dimMsk

    // Read existing 4×4 block
    const rho: number[][] = []   // re,im interleaved: rho[rowIdx*2], rho[rowIdx*2+1]
    for (let ri = 0; ri < 4; ri++) {
      const ra = (ri >> 1) & 1, rb = ri & 1
      const rIdx = rBase | (ra ? ma : 0n) | (rb ? mb : 0n)
      for (let ci = 0; ci < 4; ci++) {
        const ca = (ci >> 1) & 1, cb = ci & 1
        const cIdx = cBase | (ca ? ma : 0n) | (cb ? mb : 0n)
        const v = dmGet(dm, shift, rIdx, cIdx)
        if (!rho[ri]) rho[ri] = []
        rho[ri]![ci * 2]     = v.re
        rho[ri]![ci * 2 + 1] = v.im
      }
    }

    // ε(ρ)[ri,ci] = Σ_k Σ_{ri',ci'} K_k[ri,ri'] ρ[ri'][ci'] K_k*[ci,ci']
    for (let ri = 0; ri < 4; ri++) {
      const ra = (ri >> 1) & 1, rb = ri & 1
      const rIdx = rBase | (ra ? ma : 0n) | (rb ? mb : 0n)
      for (let ci = 0; ci < 4; ci++) {
        const ca = (ci >> 1) & 1, cb = ci & 1
        const cIdx = cBase | (ca ? ma : 0n) | (cb ? mb : 0n)
        let re = 0, im = 0
        for (const K of kraus) {
          for (let rp = 0; rp < 4; rp++) {
            for (let cp = 0; cp < 4; cp++) {
              const kri = K[ri]![rp]!, kci = K[ci]![cp]!
              const kRe = kri.re * kci.re + kri.im * kci.im
              const kIm = kri.im * kci.re - kri.re * kci.im
              const rhoRe = rho[rp]![cp * 2]!, rhoIm = rho[rp]![cp * 2 + 1]!
              re += kRe * rhoRe - kIm * rhoIm
              im += kRe * rhoIm + kIm * rhoRe
            }
          }
        }
        dmAcc(next, shift, rIdx, cIdx, { re, im })
      }
    }
  }
  return next
}

// ─── Hybrid dispatch ────────────────────────────────────────────────────────
//
// ρ starts sparse and is promoted once it fills in. The sparse map is right for
// a near-pure state under light noise; it collapses under a genuinely mixed one,
// where at n=12 it needs 4¹² boxed entries and exhausts the heap. Dense costs a
// flat 16·4ⁿ bytes — 256 MiB at n=12 — and every channel becomes a tight loop.

/** ρ in whichever representation currently suits it. */
type DmState =
  | { readonly kind: 'sparse'; readonly dm: DM }
  | { readonly kind: 'dense';  readonly d: DenseDM }

/**
 * Promote at 1/32 fill.
 *
 * Sparse costs roughly 100 bytes per entry against dense's flat 16·4ⁿ, and each
 * sparse operation carries BigInt keys and boxed values through its inner loop.
 * A thirty-second full is early enough to catch a densifying state well before
 * the map becomes the larger of the two, and late enough to leave genuinely
 * sparse states alone.
 */
const DM_PROMOTE_FILL = 32

function dmSettle(dm: DM, n: number): DmState {
  if (n <= MAX_DENSE_DM_QUBITS && dm.size * DM_PROMOTE_FILL > 4 ** n) {
    return { kind: 'dense', d: dmFromSparse(dm, n) }
  }
  return { kind: 'sparse', dm }
}

const dmZeroState = (): DmState => ({ kind: 'sparse', dm: new Map([[0n, { re: 1, im: 0 }]]) })

/**
 * Force the dense representation regardless of fill.
 * @internal — exported so tests can run the same circuit down both paths.
 */
export function dmPromote(s: DmState, n: number): DmState {
  return s.kind === 'dense' ? s : { kind: 'dense', d: dmFromSparse(s.dm, n) }
}

// Dense states mutate in place and return the same object; sparse ones rebuild
// and re-test for promotion.

function applySingle(s: DmState, n: number, q: number, g: Gate2x2): DmState {
  if (s.kind === 'dense') { ddSingle(s.d, q, g); return s }
  return dmSettle(spSingle(s.dm, n, q, g), n)
}

function applyTwo(s: DmState, n: number, a: number, b: number, g: Gate4x4): DmState {
  if (s.kind === 'dense') { ddTwo(s.d, a, b, g); return s }
  return dmSettle(spTwo(s.dm, n, a, b, g), n)
}

function applyPerm(s: DmState, n: number, f: (i: bigint) => bigint): DmState {
  // A permutation cannot change the number of non-zero entries, so no promotion
  // test is needed. The dense path converts the index map once per call, not per
  // entry — ddPerm evaluates f exactly `dim` times.
  if (s.kind === 'dense') { ddPerm(s.d, i => Number(f(BigInt(i)))); return s }
  return { kind: 'sparse', dm: spPerm(s.dm, n, f) }
}

function applyUnitaryN(s: DmState, n: number, qs: readonly number[], m: readonly (readonly Complex[])[]): DmState {
  if (s.kind === 'dense') { ddUnitaryN(s.d, qs, m); return s }
  return dmSettle(spUnitaryN(s.dm, n, qs, m), n)
}

function depolarize1(s: DmState, n: number, q: number, p: number): DmState {
  if (s.kind === 'dense') { ddDepolarize1(s.d, q, p); return s }
  return dmSettle(spDepolarize1(s.dm, n, q, p), n)
}

function depolarize2(s: DmState, n: number, a: number, b: number, p: number): DmState {
  if (s.kind === 'dense') { ddDepolarize2(s.d, a, b, p, PAULI15); return s }
  return dmSettle(spDepolarize2(s.dm, n, a, b, p), n)
}

function amplitudeDamping1(s: DmState, n: number, q: number, gamma: number): DmState {
  if (s.kind === 'dense') { ddAmplitudeDamping1(s.d, q, gamma); return s }
  return dmSettle(spAmplitudeDamping1(s.dm, n, q, gamma), n)
}

function phaseDamping1(s: DmState, n: number, q: number, lambda: number): DmState {
  if (s.kind === 'dense') { ddPhaseDamping1(s.d, q, lambda); return s }
  return dmSettle(spPhaseDamping1(s.dm, n, q, lambda), n)
}

function applyKraus1DM(s: DmState, n: number, q: number, kraus: readonly Gate2x2[]): DmState {
  if (s.kind === 'dense') { ddKraus1(s.d, q, kraus); return s }
  return dmSettle(spKraus1(s.dm, n, q, kraus), n)
}

function applyKraus2DM(s: DmState, n: number, a: number, b: number, kraus: readonly Gate4x4[]): DmState {
  if (s.kind === 'dense') { ddKraus2(s.d, a, b, kraus); return s }
  return dmSettle(spKraus2(s.dm, n, a, b, kraus), n)
}

// ─── DensityMatrix class ────────────────────────────────────────────────────

/**
 * Exact mixed-state density matrix returned by `circuit.dm()`.
 *
 * For n qubits the state space has 4ⁿ entries in the worst case.
 * In practice, near-pure states and modest noise keep the DM very sparse.
 *
 * All bitstring keys follow the standard convention: qubit 0 is the leftmost character.
 */
export class DensityMatrix {
  readonly qubits: number
  readonly #state: DmState
  readonly #shift: bigint
  readonly #dimMask: bigint

  /** @internal */
  constructor(qubits: number, state: DmState) {
    this.qubits   = qubits
    this.#state   = state
    this.#shift   = BigInt(qubits)
    this.#dimMask = (1n << this.#shift) - 1n
  }

  /**
   * Iterate the non-zero entries as (row, col, value), whichever representation
   * is live. Lets the accessors below share one traversal instead of each
   * branching on the representation.
   */
  * #entries(): Generator<[bigint, bigint, Complex]> {
    if (this.#state.kind === 'sparse') {
      for (const [k, v] of this.#state.dm) yield [k >> this.#shift, k & this.#dimMask, v]
      return
    }
    const { dim, data } = this.#state.d
    for (let r = 0; r < dim; r++) {
      for (let c = 0; c < dim; c++) {
        const p = ((r * dim + c) << 1)
        const re = data[p]!, im = data[p + 1]!
        if (re * re + im * im >= 1e-14) yield [BigInt(r), BigInt(c), { re, im }]
      }
    }
  }

  /** ρ[row][col]. */
  get(row: bigint, col: bigint): Complex {
    return this.#state.kind === 'sparse'
      ? dmGet(this.#state.dm, this.#shift, row, col)
      : denseDmGet(this.#state.d, Number(row), Number(col))
  }

  /**
   * Diagonal probabilities: P(bitstring) = ρ[bs][bs].
   *
   * Keys are standard bitstrings (q0 leftmost).  Only non-negligible values
   * (> 1e-14) are included.
   */
  probabilities(): Readonly<Record<string, number>> {
    const out: Record<string, number> = {}
    for (const [r, c, v] of this.#entries()) {
      if (r === c && v.re > 1e-14) out[r.toString(2).padStart(this.qubits, '0').split('').reverse().join('')] = v.re
    }
    return Object.freeze(out)
  }

  /**
   * Purity Tr(ρ²) = Σ_{r,c} |ρ[r][c]|².
   *
   * Equals 1 for a pure state; equals 1/2ⁿ for the maximally mixed state.
   * Values below 1 indicate entanglement-induced or noise-induced mixing.
   */
  purity(): number {
    let p = 0
    for (const [, , v] of this.#entries()) p += v.re * v.re + v.im * v.im
    return p
  }

  /**
   * Von Neumann entropy S = −Tr(ρ log₂ ρ) in bits.
   *
   * Computed by diagonalising the full 2ⁿ × 2ⁿ density matrix via Jacobi
   * iteration.  Practical for n ≤ 8 (matrix size ≤ 256 × 256).
   *
   * @throws RangeError for n > 12 (4096 × 4096 matrix — too expensive).
   */
  entropy(): number {
    const dim = 1 << this.qubits
    if (dim > 4096) throw new RangeError(`entropy(): circuit too large (n=${this.qubits}, dim=${dim})`)

    // Build dense Hermitian matrix
    const re = new Float64Array(dim * dim)
    const im = new Float64Array(dim * dim)
    for (const [rb, cb, v] of this.#entries()) {
      const r = Number(rb), c = Number(cb)
      re[r * dim + c] = v.re
      im[r * dim + c] = v.im
    }

    const λ = jacobiEigenvalues(re, im, dim)
    let S = 0
    for (const lam of λ) { if (lam > 1e-14) S -= lam * Math.log2(lam) }
    return S
  }

  /**
   * Bloch sphere coordinates (θ, φ) for qubit q from the reduced density matrix.
   *
   * Computes ρ_q = Tr_{others}(ρ) then extracts:
   *   rx = 2·Re(ρ_q[0][1]),  ry = −2·Im(ρ_q[0][1]),  rz = ρ_q[0][0] − ρ_q[1][1]
   *
   * - θ = arccos(rz)  ∈ [0, π]
   * - φ = atan2(ry, rx)  ∈ (−π, π]
   */
  blochAngles(q: number): { theta: number; phi: number } {
    const qMask = 1n << BigInt(q)
    let rho00 = 0, rho11 = 0, rho01re = 0, rho01im = 0

    for (const [r, c, v] of this.#entries()) {
      if (r === c) {
        if ((r & qMask) === 0n) rho00 += v.re   // Tr_{others}(Π₀ ρ)
        else                    rho11 += v.re   // Tr_{others}(Π₁ ρ)
      } else if ((r & qMask) === 0n && c === (r | qMask)) {
        // ρ_q[0][1] = Σ_{ctx} ρ[ctx][ctx|mask]
        rho01re += v.re; rho01im += v.im
      }
    }

    const rz = rho00 - rho11, rx = 2 * rho01re, ry = -2 * rho01im
    return { theta: Math.acos(Math.max(-1, Math.min(1, rz))), phi: Math.atan2(ry, rx) }
  }
}

// ─── Jacobi eigenvalue solver (Hermitian matrix) ──────────────────────────

/**
 * Compute real eigenvalues of an n×n Hermitian matrix stored as flat row-major
 * Float64Arrays `re` and `im`.  Uses cyclic Jacobi sweeps until convergence.
 */
function jacobiEigenvalues(re: Float64Array, im: Float64Array, n: number): number[] {
  // Work on copies; accumulate unitary transform in R (re), I (im)
  const R = new Float64Array(re), Im = new Float64Array(im)

  for (let sweep = 0; sweep < 30 * n; sweep++) {
    let maxOff = 0
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const oRe = R[p * n + q]!, oIm = Im[p * n + q]!
        const off2 = oRe * oRe + oIm * oIm
        if (off2 > maxOff) maxOff = off2
      }
    }
    if (maxOff < 1e-28) break

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const oRe = R[p * n + q]!, oIm = Im[p * n + q]!
        const off2 = oRe * oRe + oIm * oIm
        if (off2 < 1e-28) continue

        // Annihilate the (p,q) entry with a complex Givens rotation
        //   G = [[cg, w], [-conj(w), cg]],  w = sg·e^{iφ},  φ = arg(ρ[p,q])
        // applied as A ← G A G†. G is unitary because its lower-left entry is
        // −conj(w), not −w; getting that wrong leaves G non-unitary, which is
        // invisible for a single rotation but compounds badly over a sweep.
        const phi = Math.atan2(oIm, oRe)  // angle of R[p,q]
        const mag = Math.sqrt(off2)

        // A'[p][q] ∝ cg·sg·(b − a) + (cg² − sg²)·mag, so annihilation needs
        // (cg² − sg²)/(cg·sg) = (a − b)/mag with a = A[p][p], b = A[q][q].
        // Taking tau = (a − b)/2 and the smaller root keeps |t| ≤ 1.
        const tau = (R[p * n + p]! - R[q * n + q]!) / 2
        const t   = mag / (Math.abs(tau) + Math.sqrt(tau * tau + mag * mag)) * (tau < 0 ? -1 : 1)
        const cg  = 1 / Math.sqrt(1 + t * t)
        const sg  = t * cg

        const sre = sg * Math.cos(phi), sim = sg * Math.sin(phi)   // w = sre + i·sim

        for (let k = 0; k < n; k++) {
          const xRe = R[p * n + k]!, xIm = Im[p * n + k]!
          const yRe = R[q * n + k]!, yIm = Im[q * n + k]!
          // Row p: cg·x + w·y
          R[p * n + k]  =  cg * xRe + sre * yRe - sim * yIm
          Im[p * n + k] =  cg * xIm + sre * yIm + sim * yRe
          // Row q: −conj(w)·x + cg·y = (−sre + i·sim)·x + cg·y
          R[q * n + k]  = -sre * xRe - sim * xIm + cg * yRe
          Im[q * n + k] = -sre * xIm + sim * xRe + cg * yIm
        }
        for (let k = 0; k < n; k++) {
          const xRe = R[k * n + p]!, xIm = Im[k * n + p]!
          const yRe = R[k * n + q]!, yIm = Im[k * n + q]!
          // Col p: right-multiply by G[:,p] = (cg, sre−i·sim)ᵀ
          R[k * n + p]  =  cg * xRe + sre * yRe + sim * yIm
          Im[k * n + p] =  cg * xIm + sre * yIm - sim * yRe
          // Col q: right-multiply by G[:,q] = (−sre−i·sim, cg)ᵀ
          R[k * n + q]  = -sre * xRe + sim * xIm + cg * yRe
          Im[k * n + q] = -sre * xIm - sim * xRe + cg * yIm
        }
      }
    }
  }

  return Array.from({ length: n }, (_, i) => R[i * n + i]!)
}

// ─── Exported types ────────────────────────────────────────────────────────

/** Noise parameters for density matrix simulation. */
export interface DmNoiseParams {
  /** Single-qubit depolarizing probability per gate. */
  p1?: number
  /** Two-qubit depolarizing probability per gate. */
  p2?: number
  /**
   * Amplitude damping (T1 relaxation) probability per single-qubit gate.
   * Applied as the exact Kraus channel ε(ρ) = K0 ρ K0† + K1 ρ K1† after each 1Q gate.
   */
  gamma?: number
  /**
   * Pure dephasing (T2 beyond T1) probability per single-qubit gate.
   * Kills off-diagonal coherences without changing populations.
   */
  lambda?: number
  /**
   * Custom Kraus operators applied after each **single-qubit** gate only.
   * Not applied after two-qubit gates — use `kraus2` for those.
   * Must satisfy Σ_k K_k† K_k = I.
   */
  kraus1?: readonly Gate2x2[]
  /**
   * Custom Kraus operators applied after each **two-qubit** gate only.
   * Not applied after single-qubit gates — use `kraus1` for those.
   * Must satisfy Σ_k K_k† K_k = I.
   */
  kraus2?: readonly Gate4x4[]
}

/** Published IonQ device profiles — mirrored from circuit.ts. */
export const DM_DEVICE_NOISE: Readonly<Record<string, DmNoiseParams>> = {
  'aria-1':  { p1: 0.0003, p2: 0.005 },
  'forte-1': { p1: 0.0001, p2: 0.002 },
  'harmony': { p1: 0.001,  p2: 0.015 },
}

/** Op types that runDM can simulate (classical ops excluded — use pure circuits). */
export type DmOp =
  | { kind: 'single';     q: number;                      gate: Gate2x2 }
  | { kind: 'cnot';       control: number; target: number }
  | { kind: 'swap';       a: number;       b: number }
  | { kind: 'two';        a: number;       b: number;     gate: Gate4x4 }
  | { kind: 'controlled'; control: number; target: number; gate: Gate2x2 }
  | { kind: 'toffoli';    c1: number;      c2: number;    target: number }
  | { kind: 'cswap';      control: number; a: number;     b: number }
  | { kind: 'csrswap';    control: number; a: number;     b: number }
  | { kind: 'unitary';    qubits: readonly number[];      matrix: readonly (readonly Complex[])[] }

/**
 * Simulate `ops` on the |0…0⟩⟨0…0| initial state and return the exact
 * density matrix, optionally with per-gate depolarizing noise.
 */
export function runDM(ops: readonly DmOp[], qubits: number, noise?: DmNoiseParams): DensityMatrix {
  const p1     = noise?.p1     ?? 0
  const p2     = noise?.p2     ?? 0
  const gamma  = noise?.gamma  ?? 0
  const lambda = noise?.lambda ?? 0
  const kraus1 = noise?.kraus1
  const kraus2 = noise?.kraus2
  let dm: DmState = dmZeroState()
  const n = qubits

  const sq2 = 1 / Math.sqrt(2)
  // √iSWAP 4×4 in {|00⟩,|01⟩,|10⟩,|11⟩} (a=MSB) — used for csrswap
  const SRISW: Gate4x4 = [
    [{ re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: sq2, im: 0 }, { re: 0, im: sq2 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: 0, im: sq2 }, { re: sq2, im: 0 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }],
  ]

  for (const op of ops) {
    switch (op.kind) {
      case 'single':
        dm = applySingle(dm, n, op.q, op.gate)
        if (p1)     dm = depolarize1(dm, n, op.q, p1)
        if (gamma)  dm = amplitudeDamping1(dm, n, op.q, gamma)
        if (lambda) dm = phaseDamping1(dm, n, op.q, lambda)
        if (kraus1) dm = applyKraus1DM(dm, n, op.q, kraus1)
        break
      case 'cnot': {
        const cm = 1n << BigInt(op.control), tm = 1n << BigInt(op.target)
        dm = applyPerm(dm, n, i => (i & cm) !== 0n ? i ^ tm : i)
        if (p2)     dm = depolarize2(dm, n, op.control, op.target, p2)
        if (gamma)  { dm = amplitudeDamping1(dm, n, op.control, gamma); dm = amplitudeDamping1(dm, n, op.target, gamma) }
        if (lambda) { dm = phaseDamping1(dm, n, op.control, lambda);    dm = phaseDamping1(dm, n, op.target, lambda) }
        if (kraus2) dm = applyKraus2DM(dm, n, op.control, op.target, kraus2)
        break
      }
      case 'swap': {
        const am = 1n << BigInt(op.a), bm = 1n << BigInt(op.b)
        dm = applyPerm(dm, n, i => {
          const ba = (i & am) !== 0n, bb = (i & bm) !== 0n
          return ba === bb ? i : i ^ am ^ bm
        })
        if (p2)     dm = depolarize2(dm, n, op.a, op.b, p2)
        if (gamma)  { dm = amplitudeDamping1(dm, n, op.a, gamma); dm = amplitudeDamping1(dm, n, op.b, gamma) }
        if (lambda) { dm = phaseDamping1(dm, n, op.a, lambda);    dm = phaseDamping1(dm, n, op.b, lambda) }
        if (kraus2) dm = applyKraus2DM(dm, n, op.a, op.b, kraus2)
        break
      }
      case 'two':
        dm = applyTwo(dm, n, op.a, op.b, op.gate)
        if (p2)     dm = depolarize2(dm, n, op.a, op.b, p2)
        if (gamma)  { dm = amplitudeDamping1(dm, n, op.a, gamma); dm = amplitudeDamping1(dm, n, op.b, gamma) }
        if (lambda) { dm = phaseDamping1(dm, n, op.a, lambda);    dm = phaseDamping1(dm, n, op.b, lambda) }
        if (kraus2) dm = applyKraus2DM(dm, n, op.a, op.b, kraus2)
        break
      case 'controlled':
        dm = applyTwo(dm, n, op.control, op.target, controlledGate(op.gate))
        if (p2)     dm = depolarize2(dm, n, op.control, op.target, p2)
        if (gamma)  { dm = amplitudeDamping1(dm, n, op.control, gamma); dm = amplitudeDamping1(dm, n, op.target, gamma) }
        if (lambda) { dm = phaseDamping1(dm, n, op.control, lambda);    dm = phaseDamping1(dm, n, op.target, lambda) }
        if (kraus2) dm = applyKraus2DM(dm, n, op.control, op.target, kraus2)
        break
      case 'toffoli': {
        const c1m = 1n << BigInt(op.c1), c2m = 1n << BigInt(op.c2), tm = 1n << BigInt(op.target)
        dm = applyPerm(dm, n, i => (i & c1m) !== 0n && (i & c2m) !== 0n ? i ^ tm : i)
        break
      }
      case 'cswap': {
        const cm = 1n << BigInt(op.control), am = 1n << BigInt(op.a), bm = 1n << BigInt(op.b)
        dm = applyPerm(dm, n, i => {
          if ((i & cm) === 0n) return i
          const ba = (i & am) !== 0n, bb = (i & bm) !== 0n
          return ba === bb ? i : i ^ am ^ bm
        })
        break
      }
      case 'csrswap': {
        // Controlled-√iSWAP as a single 3-qubit unitary: block-diag(I₄, √iSWAP)
        // with the control as the MSB of the local index. Expressing it this way
        // rather than splitting ρ into control-parity blocks keeps it on the
        // shared conjugation path, so it works on either representation.
        const C_SRISW: Complex[][] = Array.from({ length: 8 }, (_, i) =>
          Array.from({ length: 8 }, (_, j) => {
            if (i < 4 || j < 4) return i === j ? { re: 1, im: 0 } : { re: 0, im: 0 }
            return SRISW[i - 4]![j - 4]!
          })
        )
        dm = applyUnitaryN(dm, n, [op.control, op.a, op.b], C_SRISW)
        break
      }
      case 'unitary':
        dm = applyUnitaryN(dm, n, op.qubits, op.matrix)
        break
      default: { const _exhaustive: never = op; void _exhaustive }
    }
  }

  return new DensityMatrix(qubits, dm)
}
