/** Immutable complex number. */
export type Complex = { readonly re: number; readonly im: number }

export const ZERO: Complex = { re: 0, im: 0 }
export const ONE: Complex  = { re: 1, im: 0 }
export const I: Complex    = { re: 0, im: 1 }

export const c = (re: number, im = 0): Complex => ({ re, im })

export const add = (a: Complex, b: Complex): Complex =>
  ({ re: a.re + b.re, im: a.im + b.im })

export const mul = (a: Complex, b: Complex): Complex =>
  ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re })

export const scale = (s: number, a: Complex): Complex =>
  ({ re: s * a.re, im: s * a.im })

export const conj = (a: Complex): Complex => ({ re: a.re, im: -a.im })

/** |a|² — probability weight of an amplitude. */
export const norm2 = (a: Complex): number => a.re * a.re + a.im * a.im

/**
 * Amplitude magnitude below which a value is rounding dust rather than physics.
 *
 * A statevector is normalised, so this absolute bound is also a relative one.
 * Exact cancellations — the |1⟩ branch of `H·H`, the unselected half of a Grover
 * reflection — land within a few ulps of zero, so 1e-15 removes them while
 * keeping every amplitude that carries real weight.
 *
 * It must not be confused with the 1e-14 filters applied to *probabilities* when
 * results are reported. Those act on |amplitude|², where 1e-14 is a genuinely
 * meaningless probability. Applying the same number to an amplitude discards
 * everything below 1e-7, which is eight orders of magnitude of real physics: a
 * 12-qubit product state can hold hundreds of amplitudes under that bound, and
 * zeroing them perturbs the state far beyond their own size, because subsequent
 * entangling gates spread the loss across the whole register.
 */
export const AMP_EPSILON = 1e-15

/** True when an amplitude is indistinguishable from zero. See {@link AMP_EPSILON}. */
export const isNegligible = (a: Complex): boolean => norm2(a) < AMP_EPSILON * AMP_EPSILON
