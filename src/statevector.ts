/**
 * Sparse statevector over BigInt indices.
 *
 * State index bit layout: qubit 0 is bit 0 (LSB), matching IonQ's bitstring
 * convention where q0 is the rightmost character.
 *
 * Using BigInt eliminates the 32-bit overflow that breaks quantum-circuit above
 * qubit 30 (e.g. `1 << 31` silently wraps in JavaScript).
 */

import { add, Complex, isNegligible, mul, ZERO } from './complex.js'

export type StateVector = Map<bigint, Complex>

/** |0...0⟩ for n qubits. */
export const zero = (n: number): StateVector => new Map([[0n, { re: 1, im: 0 }]])

/** Accumulate amplitude into a map, creating entry if needed. */
function accumulate(sv: StateVector, idx: bigint, amp: Complex): void {
  const existing = sv.get(idx)
  const next = existing ? add(existing, amp) : amp
  if (!isNegligible(next)) {
    sv.set(idx, next)
  } else {
    sv.delete(idx)
  }
}

/** 2×2 unitary gate matrix: [[a, b], [c, d]]. */
export type Gate2x2 = [[Complex, Complex], [Complex, Complex]]

/** 4×4 unitary gate matrix. Row/column order: |00⟩, |01⟩, |10⟩, |11⟩ (qubit a = MSB). */
export type Gate4x4 = [
  [Complex, Complex, Complex, Complex],
  [Complex, Complex, Complex, Complex],
  [Complex, Complex, Complex, Complex],
  [Complex, Complex, Complex, Complex],
]

/**
 * Apply a single-qubit gate to qubit q.
 *
 * For each pair of basis states |x₀⟩ (bit q = 0) and |x₁⟩ (bit q = 1):
 *   new[x₀] = a·old[x₀] + b·old[x₁]
 *   new[x₁] = c·old[x₀] + d·old[x₁]
 */
export function applySingle(sv: StateVector, q: number, [[a, b], [c, d]]: Gate2x2): StateVector {
  const next: StateVector = new Map()
  const mask = 1n << BigInt(q)
  const seen = new Set<bigint>()

  for (const idx of sv.keys()) {
    const base = idx & ~mask // idx with bit q = 0
    if (seen.has(base)) continue
    seen.add(base)

    const amp0 = sv.get(base) ?? ZERO
    const amp1 = sv.get(base | mask) ?? ZERO

    accumulate(next, base,        add(mul(a, amp0), mul(b, amp1)))
    accumulate(next, base | mask, add(mul(c, amp0), mul(d, amp1)))
  }

  return next
}

/**
 * Apply a controlled-NOT gate: if control = |1⟩, flip target.
 */
export function applyCNOT(sv: StateVector, control: number, target: number): StateVector {
  const next: StateVector = new Map()
  const cmask = 1n << BigInt(control)
  const tmask = 1n << BigInt(target)

  for (const [idx, amp] of sv) {
    next.set((idx & cmask) !== 0n ? idx ^ tmask : idx, amp)
  }

  return next
}

/**
 * Apply a SWAP gate: exchange qubit a and qubit b.
 */
export function applySWAP(sv: StateVector, a: number, b: number): StateVector {
  const next: StateVector = new Map()
  const amask = 1n << BigInt(a)
  const bmask = 1n << BigInt(b)

  for (const [idx, amp] of sv) {
    const bitA = (idx & amask) !== 0n
    const bitB = (idx & bmask) !== 0n
    if (bitA === bitB) {
      next.set(idx, amp)
    } else {
      // Swap the two bits
      next.set(idx ^ amask ^ bmask, amp)
    }
  }

  return next
}

/**
 * Apply a two-qubit gate to qubits a and b.
 *
 * Qubit a is the MSB of the 2-bit local index so the column ordering matches
 * the standard ket notation: |00⟩, |01⟩, |10⟩, |11⟩ where the first digit is a.
 */
export function applyTwo(sv: StateVector, a: number, b: number, gate: Gate4x4): StateVector {
  const next: StateVector = new Map()
  const ma = 1n << BigInt(a)
  const mb = 1n << BigInt(b)
  const seen = new Set<bigint>()

  for (const idx of sv.keys()) {
    const ctx = idx & ~(ma | mb)
    if (seen.has(ctx)) continue
    seen.add(ctx)

    const bases = [ctx, ctx | mb, ctx | ma, ctx | ma | mb]
    const amps  = bases.map(i => sv.get(i) ?? ZERO)

    for (let r = 0; r < 4; r++) {
      let out = ZERO
      for (let c = 0; c < 4; c++) out = add(out, mul(gate[r]![c]!, amps[c]!))
      accumulate(next, bases[r]!, out)
    }
  }

  return next
}

/**
 * Apply a Toffoli (CCX) gate: flip target if both c1 = |1⟩ and c2 = |1⟩.
 * Pure permutation — no amplitude arithmetic.
 */
export function applyToffoli(sv: StateVector, c1: number, c2: number, target: number): StateVector {
  const next: StateVector = new Map()
  const c1mask = 1n << BigInt(c1)
  const c2mask = 1n << BigInt(c2)
  const tmask  = 1n << BigInt(target)
  for (const [idx, amp] of sv) {
    next.set((idx & c1mask) !== 0n && (idx & c2mask) !== 0n ? idx ^ tmask : idx, amp)
  }
  return next
}

/**
 * Apply a Fredkin (CSWAP) gate: swap qubits a and b if control = |1⟩.
 * Pure permutation — no amplitude arithmetic.
 */
export function applyCSwap(sv: StateVector, control: number, a: number, b: number): StateVector {
  const next: StateVector = new Map()
  const cmask = 1n << BigInt(control)
  const amask = 1n << BigInt(a)
  const bmask = 1n << BigInt(b)
  for (const [idx, amp] of sv) {
    if ((idx & cmask) === 0n) {
      next.set(idx, amp)
    } else {
      const bitA = (idx & amask) !== 0n
      const bitB = (idx & bmask) !== 0n
      next.set(bitA === bitB ? idx : idx ^ amask ^ bmask, amp)
    }
  }
  return next
}

/**
 * Apply a controlled single-qubit gate: if control = |1⟩, apply gate to target.
 */
export function applyControlled(sv: StateVector, control: number, target: number, [[a,b],[c,d]]: Gate2x2): StateVector {
  const next: StateVector = new Map()
  const cmask = 1n << BigInt(control)
  const tmask = 1n << BigInt(target)
  const seen  = new Set<bigint>()

  for (const [idx, amp] of sv) {
    if ((idx & cmask) === 0n) {
      accumulate(next, idx, amp)
      continue
    }
    const base = idx & ~tmask
    if (seen.has(base)) continue
    seen.add(base)
    const amp0 = sv.get(base)         ?? ZERO
    const amp1 = sv.get(base | tmask) ?? ZERO
    accumulate(next, base,         add(mul(a, amp0), mul(b, amp1)))
    accumulate(next, base | tmask, add(mul(c, amp0), mul(d, amp1)))
  }

  return next
}

/** Return probability of each basis state as a plain Record<decimal-string, number>. */
export function probabilities(sv: StateVector): Map<bigint, number> {
  const probs = new Map<bigint, number>()
  for (const [idx, amp] of sv) {
    const p = amp.re * amp.re + amp.im * amp.im
    if (p > 1e-14) probs.set(idx, p)
  }
  return probs
}
