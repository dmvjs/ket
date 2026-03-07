/**
 * IonQ QIS gate matrices.
 *
 * All gates are defined as 2×2 unitaries on the computational basis {|0⟩, |1⟩}.
 * Names match IonQ's native gate set (h, x, y, z, s, si, t, ti, v, vi, rx, ry, rz).
 */

import { c, Complex, ONE, ZERO } from './complex.js'
import { Gate2x2 } from './statevector.js'

const sq2 = 1 / Math.sqrt(2) // 1/√2

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
