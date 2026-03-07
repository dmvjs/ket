/**
 * IonQ QIS gate matrices.
 *
 * All gates are defined as 2×2 unitaries on the computational basis {|0⟩, |1⟩}.
 * Names match IonQ's native gate set (h, x, y, z, s, si, t, ti, v, vi, rx, ry, rz).
 */

import { c, Complex, ONE, ZERO } from './complex.js'
import { Gate2x2 } from './statevector.js'

const R2 = 1 / Math.sqrt(2) // 1/√2

export const H:  Gate2x2 = [[c(R2), c(R2)],  [c(R2), c(-R2)]]
export const X:  Gate2x2 = [[ZERO, ONE],      [ONE,   ZERO]]
export const Y:  Gate2x2 = [[ZERO, c(0,-1)],  [c(0,1), ZERO]]
export const Z:  Gate2x2 = [[ONE,  ZERO],     [ZERO,  c(-1)]]
export const S:  Gate2x2 = [[ONE,  ZERO],     [ZERO,  c(0, 1)]]  // √Z
export const Si: Gate2x2 = [[ONE,  ZERO],     [ZERO,  c(0,-1)]]  // S†
export const T:  Gate2x2 = [[ONE,  ZERO],     [ZERO,  c(R2, R2)]] // ⁴√Z
export const Ti: Gate2x2 = [[ONE,  ZERO],     [ZERO,  c(R2,-R2)]] // T†
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
