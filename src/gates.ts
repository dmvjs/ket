/**
 * IonQ QIS gate matrices.
 *
 * All gates are defined as 2×2 unitaries on the computational basis {|0⟩, |1⟩}.
 * Names match IonQ's native gate set (h, x, y, z, s, si, t, ti, v, vi, rx, ry, rz).
 */

import { c, Complex, ONE, ZERO } from './complex.js'
import { Gate2x2, Gate4x4 } from './statevector.js'

const sq2 = 1 / Math.sqrt(2) // 1/√2

export const Id: Gate2x2 = [[ONE,  ZERO],        [ZERO,   ONE]]   // identity
export const H:  Gate2x2 = [[c(sq2), c(sq2)],  [c(sq2), c(-sq2)]]
export const X:  Gate2x2 = [[ZERO, ONE],        [ONE,    ZERO]]
export const Y:  Gate2x2 = [[ZERO, c(0,-1)],    [c(0,1), ZERO]]
export const Z:  Gate2x2 = [[ONE,  ZERO],        [ZERO,   c(-1)]]
export const S:  Gate2x2 = [[ONE,  ZERO],        [ZERO,   c(0, 1)]]   // √Z
export const Si: Gate2x2 = [[ONE,  ZERO],        [ZERO,   c(0,-1)]]   // S†
export const T:  Gate2x2 = [[ONE,  ZERO],        [ZERO,   c(sq2, sq2)]] // ⁴√Z
export const Ti: Gate2x2 = [[ONE,  ZERO],        [ZERO,   c(sq2,-sq2)]] // T†
export const V:  Gate2x2 = [[c(.5, .5),  c(.5,-.5)],  [c(.5,-.5), c(.5, .5)]]  // √X
export const Vi: Gate2x2 = [[c(.5,-.5),  c(.5, .5)],  [c(.5, .5), c(.5,-.5)]]  // √X†

/** Rotation around X axis by angle θ (radians). */
export const Rx = (theta: number): Gate2x2 => {
  const cos = Math.cos(theta / 2)
  const sin = Math.sin(theta / 2)
  return [[c(cos), c(0, -sin)], [c(0, -sin), c(cos)]]
}

/** Rotation around Y axis by angle θ (radians). */
export const Ry = (theta: number): Gate2x2 => {
  const cos = Math.cos(theta / 2)
  const sin = Math.sin(theta / 2)
  return [[c(cos, 0), c(-sin)], [c(sin), c(cos)]]
}

/** Rotation around Z axis by angle θ (radians). */
export const Rz = (theta: number): Gate2x2 => {
  const cos = Math.cos(theta / 2)
  const sin = Math.sin(theta / 2)
  return [[c(cos, -sin), ZERO], [ZERO, c(cos, sin)]]
}

// ── Named phase rotation gates ───────────────────────────────────────────────
// Rz(π/2ⁿ) aliases — identical to S/T up to global phase; distinct gate names
// for IonQ JSON serialisation and readability.

/** Rz(π/2) — phase rotation by a half-turn. Equal to S up to global phase. */
export const R2: Gate2x2 = Rz(Math.PI / 2)

/** Rz(π/4) — phase rotation by a quarter-turn. Equal to T up to global phase. */
export const R4: Gate2x2 = Rz(Math.PI / 4)

/** Rz(π/8) — phase rotation by an eighth-turn. */
export const R8: Gate2x2 = Rz(Math.PI / 8)

// ── OpenQASM basis gates (U1 / U2 / U3) ─────────────────────────────────────

/**
 * U3(θ, φ, λ) — general single-qubit unitary; the basis gate of OpenQASM 2.0.
 *
 *   ┌  cos(θ/2)           −e^(iλ)·sin(θ/2)      ┐
 *   └  e^(iφ)·sin(θ/2)    e^(i(φ+λ))·cos(θ/2)   ┘
 *
 * Key identities:
 *   U3(π, 0, π)   = X           U3(π/2, 0, π) = H
 *   U3(θ, 0, 0)   = Ry(θ)       U3(0, 0, λ)   = U1(λ)
 */
export const U3 = (theta: number, phi: number, lambda: number): Gate2x2 => {
  const cos = Math.cos(theta / 2)
  const sin = Math.sin(theta / 2)
  return [
    [c(cos),                                       c(-sin * Math.cos(lambda), -sin * Math.sin(lambda))],
    [c(sin * Math.cos(phi), sin * Math.sin(phi)),  c(cos * Math.cos(phi + lambda), cos * Math.sin(phi + lambda))],
  ]
}

/**
 * U2(φ, λ) = U3(π/2, φ, λ) — equatorial gate. U2(0, π) = H.
 */
export const U2 = (phi: number, lambda: number): Gate2x2 => U3(Math.PI / 2, phi, lambda)

/**
 * U1(λ) = diag(1, e^(iλ)) — phase gate; equal to Rz(λ) up to global phase.
 * U1(π/2) = S,  U1(π/4) = T,  U1(π) = Z.
 */
export const U1 = (lambda: number): Gate2x2 =>
  [[ONE, ZERO], [ZERO, c(Math.cos(lambda), Math.sin(lambda))]]

// ── Two-qubit interaction gates ──────────────────────────────────────────────
// Column/row order: |00⟩, |01⟩, |10⟩, |11⟩ with the first qubit argument as MSB.

/**
 * XX(θ) = exp(−iθ/2 · X⊗X) — Ising-XX interaction; IonQ native gate.
 * XX(π/2) is maximally entangling. XX(0) = I.
 */
export const Xx = (theta: number): Gate4x4 => {
  const co = c(Math.cos(theta / 2)), ni = c(0, -Math.sin(theta / 2))
  return [
    [co, ZERO, ZERO,   ni],
    [ZERO,  co,   ni, ZERO],
    [ZERO,  ni,   co, ZERO],
    [ni, ZERO, ZERO,   co],
  ]
}

/**
 * YY(θ) = exp(−iθ/2 · Y⊗Y) — Ising-YY interaction; IonQ native gate.
 */
export const Yy = (theta: number): Gate4x4 => {
  const co = c(Math.cos(theta / 2))
  const ni = c(0, -Math.sin(theta / 2))
  const pi = c(0,  Math.sin(theta / 2))
  return [
    [co, ZERO, ZERO,   pi],
    [ZERO,  co,   ni, ZERO],
    [ZERO,  ni,   co, ZERO],
    [pi, ZERO, ZERO,   co],
  ]
}

/**
 * ZZ(θ) = exp(−iθ/2 · Z⊗Z) — Ising-ZZ interaction; IonQ native gate.
 * Diagonal: diag(e^(−iθ/2), e^(iθ/2), e^(iθ/2), e^(−iθ/2)).
 * Identity: ZZ(θ) = CNOT · (I⊗Rz(θ)) · CNOT.
 */
export const Zz = (theta: number): Gate4x4 => {
  const cos = Math.cos(theta / 2), sin = Math.sin(theta / 2)
  const m = c(cos, -sin), p = c(cos, sin)
  return [
    [m, ZERO, ZERO, ZERO],
    [ZERO,    p, ZERO, ZERO],
    [ZERO, ZERO,    p, ZERO],
    [ZERO, ZERO, ZERO,    m],
  ]
}

/**
 * XY(θ) — XY interaction gate.
 * Acts as exp(iθ/2 · (X⊗X + Y⊗Y)/2) in the |01⟩/|10⟩ subspace.
 * XY(0) = I,  XY(π) = iSWAP,  XY(π/2) = √iSWAP.
 */
export const Xy = (theta: number): Gate4x4 => {
  const co = c(Math.cos(theta / 2)), is = c(0, Math.sin(theta / 2))
  return [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO,   co,   is, ZERO],
    [ZERO,   is,   co, ZERO],
    [ZERO, ZERO, ZERO,  ONE],
  ]
}

/** iSWAP = XY(π): swaps qubits and multiplies by i. */
export const ISwap: Gate4x4 = Xy(Math.PI)

/** √iSWAP = XY(π/2): square root of iSWAP. */
export const SrSwap: Gate4x4 = Xy(Math.PI / 2)

// ── IonQ hardware-native gates ───────────────────────────────────────────────

/**
 * GPI(φ) = [[0, e^{−iφ}], [e^{iφ}, 0]] — IonQ hardware-native single-qubit gate.
 *
 * Key identities:
 *   GPI(0)   = X        GPI(π/2) = Y
 *   GPI(φ)†  = GPI(φ)  (Hermitian: self-adjoint and self-inverse, GPI² = I)
 */
export const Gpi = (phi: number): Gate2x2 => {
  const cos = Math.cos(phi), sin = Math.sin(phi)
  return [[ZERO, c(cos, -sin)], [c(cos, sin), ZERO]]
}

/**
 * GPI2(φ) = (1/√2)[[1, −ie^{−iφ}], [−ie^{iφ}, 1]] — IonQ hardware-native half-rotation.
 *
 * Key identities:
 *   GPI2(0)     = Rx(π/2)    GPI2(π/2)  = Ry(π/2)
 *   GPI2(φ)⁻¹  = GPI2(φ+π)  GPI2(0)²   = X  (up to global phase)
 */
export const Gpi2 = (phi: number): Gate2x2 => {
  const cos = Math.cos(phi), sin = Math.sin(phi)
  // −i·e^{−iφ} = −i(cosφ − i·sinφ) = −sinφ − i·cosφ
  // −i·e^{ iφ} = −i(cosφ + i·sinφ) =  sinφ − i·cosφ
  return [[c(sq2), c(-sq2 * sin, -sq2 * cos)], [c(sq2 * sin, -sq2 * cos), c(sq2)]]
}

/**
 * MS(φ₀, φ₁) — Mølmer-Sørensen entangling gate; IonQ's native two-qubit operation.
 *
 * MS(0, 0)       = XX(π/2)   (maximally entangling)
 * MS(π/2, π/2)   = YY(π/2)
 * MS(φ₀, φ₁)|00⟩ always produces a superposition of |00⟩ and |11⟩ with equal probability.
 */
export const Ms = (phi0: number, phi1: number): Gate4x4 => {
  const sp = Math.sin(phi0 + phi1), cp = Math.cos(phi0 + phi1)
  const sd = Math.sin(phi0 - phi1), cd = Math.cos(phi0 - phi1)
  // Entries = (1/√2) × { diagonal: 1,
  //   [0][3]: −i·e^{−i(φ₀+φ₁)},  [1][2]: −i·e^{−i(φ₀−φ₁)},
  //   [2][1]: −i·e^{ i(φ₀−φ₁)},  [3][0]: −i·e^{ i(φ₀+φ₁)} }
  return [
    [c(sq2),                  ZERO,                    ZERO,                    c(-sp * sq2, -cp * sq2)],
    [ZERO,                    c(sq2),                  c(-sd * sq2, -cd * sq2), ZERO                  ],
    [ZERO,                    c( sd * sq2, -cd * sq2), c(sq2),                  ZERO                  ],
    [c( sp * sq2, -cp * sq2), ZERO,                    ZERO,                    c(sq2)                ],
  ]
}
