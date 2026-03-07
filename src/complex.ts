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

export const isNegligible = (a: Complex): boolean => norm2(a) < 1e-14
