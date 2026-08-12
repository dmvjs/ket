/**
 * Quadratic-form exponential sums over F₂ and Z₄ (Lemma 4).
 *
 * Evaluates Z(B) = Σ_{x ∈ {0,1}^m} i^{xBxᵀ} in O(m³) rather than the 2^m the
 * definition suggests. This is the kernel that makes stabilizer inner products —
 * and therefore norm estimation — polynomial.
 *
 * Results are returned in dyadic form (sign·2^exp) rather than as plain numbers:
 * Z can reach 2^m, which overflows a float64 well before m = 100, while the
 * quantities built from it are O(1) once the 2^{-(n+|v|)/2} prefactor is folded
 * in. Keeping the exponent symbolic defers that cancellation instead of losing
 * it to overflow.
 *
 * Reference: Bravyi, Browne, Calpin, Campbell, Gosset, Howard, Quantum 3, 181
 * (2019), Lemma 4 and Proposition 6.
 *
 * @internal Kernel for `StabilizerCH.innerProductEquatorial`; not public API.
 */

/** The value sign·2^exp; `sign === 0` denotes exactly zero. */
export interface Dyadic { sign: number; exp: number }

/** Collapse a dyadic to a float. Only safe once the exponent is small. */
export function dyadic(d: Dyadic): number {
  return d.sign === 0 ? 0 : d.sign * 2 ** d.exp
}

/**
 * Σ_{x ∈ {0,1}^m} (-1)^{xMxᵀ + Lxᵀ}, for arbitrary binary M — not necessarily
 * symmetric, diagonal allowed.
 *
 * When M is symmetric every quadratic term cancels mod 2 and the form collapses
 * to the linear one with coefficients L + diag(M), giving 2^m or 0. Otherwise a
 * pair of variables whose coefficients disagree can be summed out in closed form
 * (Eq. 73), eliminating two variables per step at O(m²) each.
 */
function sumF2(matIn: Uint8Array, vecIn: Uint8Array, mIn: number): Dyadic {
  let mat = matIn, vec = vecIn, m = mIn
  let sign = 1, exp = 0

  for (;;) {
    let a = -1, b = -1
    search: for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < m; j++) {
        if (mat[i * m + j] !== mat[j * m + i]) { a = i; b = j; break search }
      }
    }

    if (a < 0) {
      // Symmetric: Q(x) = Σ (L_i + M_ii)·x_i. Non-zero only if that vanishes.
      for (let i = 0; i < m; i++) {
        if (((vec[i] ?? 0) ^ (mat[i * m + i] ?? 0)) & 1) return { sign: 0, exp: 0 }
      }
      return { sign, exp: exp + m }
    }

    const c1 = ((vec[a] ?? 0) ^ (mat[a * m + a] ?? 0)) & 1
    const c2 = ((vec[b] ?? 0) ^ (mat[b * m + b] ?? 0)) & 1
    if (c1 & c2) sign = -sign
    exp += 1

    const rest: number[] = []
    for (let i = 0; i < m; i++) if (i !== a && i !== b) rest.push(i)
    const r = rest.length

    // m1, m2 are the couplings of the eliminated pair to everything remaining.
    const m1 = new Uint8Array(r), m2 = new Uint8Array(r)
    for (let k = 0; k < r; k++) {
      const i = rest[k]!
      m1[k] = ((mat[a * m + i] ?? 0) ^ (mat[i * m + a] ?? 0)) & 1
      m2[k] = ((mat[b * m + i] ?? 0) ^ (mat[i * m + b] ?? 0)) & 1
    }

    const nextMat = new Uint8Array(r * r), nextVec = new Uint8Array(r)
    for (let k = 0; k < r; k++) {
      const i = rest[k]!
      nextVec[k] = ((vec[i] ?? 0) ^ (c1 & (m2[k] ?? 0)) ^ (c2 & (m1[k] ?? 0))) & 1
      for (let l = 0; l < r; l++) {
        nextMat[k * r + l] = ((mat[i * m + rest[l]!] ?? 0) ^ ((m1[k] ?? 0) & (m2[l] ?? 0))) & 1
      }
    }
    mat = nextMat; vec = nextVec; m = r
  }
}

/**
 * Z(B) = Σ_{x ∈ {0,1}^m} i^{xBxᵀ}, returned as separate dyadic real and
 * imaginary parts.
 *
 * `B` is symmetric and packed m×m: diagonal entries live in Z₄, off-diagonal
 * entries in {0,1} — the only residues the sum depends on. Proposition 6 lifts
 * the Z₄ form to an F₂ form in one extra variable, whose two linear variants
 * give Re and Im.
 */
export function expSum(B: Uint8Array, m: number): { re: Dyadic; im: Dyadic } {
  // B_aa = 2·L_a + K_a splits the diagonal into its Z₂ and Z₄ halves.
  const K = new Uint8Array(m), L = new Uint8Array(m)
  for (let i = 0; i < m; i++) {
    const d = (B[i * m + i] ?? 0) & 3
    K[i] = d & 1
    L[i] = d >> 1
  }

  // Q over m+1 variables; index m is Proposition 6's auxiliary variable.
  const w = m + 1
  const mat = new Uint8Array(w * w), vec = new Uint8Array(w)
  for (let i = 0; i < m; i++) {
    vec[i] = L[i] ?? 0
    mat[i * w + m] = K[i] ?? 0                                   // K_a·x_a·x_{m+1}
    for (let j = i + 1; j < m; j++) {
      mat[i * w + j] = ((B[i * m + j] ?? 0) ^ ((K[i] ?? 0) & (K[j] ?? 0))) & 1
    }
  }

  const re = sumF2(mat, vec, w)
  vec[m] = 1                                                      // Q(x) + x_{m+1}
  const im = sumF2(mat, vec, w)
  return {
    re: re.sign === 0 ? re : { sign: re.sign, exp: re.exp - 1 },
    im: im.sign === 0 ? im : { sign: im.sign, exp: im.exp - 1 },
  }
}
