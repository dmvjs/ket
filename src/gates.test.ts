/**
 * Direct matrix-level tests for gates.ts.
 *
 * Every test here operates on the raw Gate2x2 / Gate4x4 numbers — no Circuit,
 * no simulation backend, no measurement sampling.  This separates gate-matrix
 * correctness from circuit-engine correctness.
 *
 * Things NOT tested here (covered by circuit.test.ts via Circuit.run()):
 *   behavioural outcomes, measurement distributions, Ramsey fringes, Bell-state
 *   production, controlled-gate firing, algorithm-level identities.
 */

import { describe, expect, it } from 'vitest'
import * as G from './gates.js'
import { add, mul, conj, c, ZERO, ONE } from './complex.js'
import type { Complex } from './complex.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'

// ─── 2×2 helpers ──────────────────────────────────────────────────────────────

function mul2(A: Gate2x2, B: Gate2x2): Gate2x2 {
  const r = (i: 0|1, j: 0|1): Complex =>
    add(mul(A[i][0], B[0][j]), mul(A[i][1], B[1][j]))
  return [[r(0,0), r(0,1)], [r(1,0), r(1,1)]]
}

function adj2(A: Gate2x2): Gate2x2 {
  return [[conj(A[0][0]), conj(A[1][0])],
          [conj(A[0][1]), conj(A[1][1])]]
}

// ─── 4×4 helpers ──────────────────────────────────────────────────────────────

type Row4 = [Complex, Complex, Complex, Complex]

function mul4(A: Gate4x4, B: Gate4x4): Gate4x4 {
  const r = (i: number, j: number): Complex =>
    [0,1,2,3].reduce<Complex>((s,k) => add(s, mul(A[i as 0][k as 0], B[k as 0][j as 0])), c(0))
  return [
    [r(0,0),r(0,1),r(0,2),r(0,3)] as Row4,
    [r(1,0),r(1,1),r(1,2),r(1,3)] as Row4,
    [r(2,0),r(2,1),r(2,2),r(2,3)] as Row4,
    [r(3,0),r(3,1),r(3,2),r(3,3)] as Row4,
  ]
}

function adj4(A: Gate4x4): Gate4x4 {
  const e = (r: number, c: number): Complex => conj(A[c as 0][r as 0])
  return [
    [e(0,0),e(0,1),e(0,2),e(0,3)] as Row4,
    [e(1,0),e(1,1),e(1,2),e(1,3)] as Row4,
    [e(2,0),e(2,1),e(2,2),e(2,3)] as Row4,
    [e(3,0),e(3,1),e(3,2),e(3,3)] as Row4,
  ]
}

// ─── Assertion helpers ────────────────────────────────────────────────────────

const TOL = 11 // decimal places for toBeCloseTo

function expectI2(M: Gate2x2): void {
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) {
      expect(M[i as 0][j as 0].re).toBeCloseTo(i === j ? 1 : 0, TOL)
      expect(M[i as 0][j as 0].im).toBeCloseTo(0, TOL)
    }
}

function expectI4(M: Gate4x4): void {
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      expect(M[i as 0][j as 0].re).toBeCloseTo(i === j ? 1 : 0, TOL)
      expect(M[i as 0][j as 0].im).toBeCloseTo(0, TOL)
    }
}

function eq2(A: Gate2x2, B: Gate2x2): void {
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) {
      expect(A[i as 0][j as 0].re).toBeCloseTo(B[i as 0][j as 0].re, TOL)
      expect(A[i as 0][j as 0].im).toBeCloseTo(B[i as 0][j as 0].im, TOL)
    }
}

function eq4(A: Gate4x4, B: Gate4x4): void {
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      expect(A[i as 0][j as 0].re).toBeCloseTo(B[i as 0][j as 0].re, TOL)
      expect(A[i as 0][j as 0].im).toBeCloseTo(B[i as 0][j as 0].im, TOL)
    }
}

// Convenient constants
const sq2 = 1 / Math.sqrt(2)
const I2: Gate2x2 = [[ONE, ZERO], [ZERO, ONE]]
const I4: Gate4x4 = [
  [ONE, ZERO, ZERO, ZERO],
  [ZERO, ONE, ZERO, ZERO],
  [ZERO, ZERO, ONE, ZERO],
  [ZERO, ZERO, ZERO, ONE],
]

// ─── Unitarity: U†U = I ───────────────────────────────────────────────────────

describe('static 1-qubit gates — U†U = I₂', () => {
  const gates: [string, Gate2x2][] = [
    ['Id', G.Id], ['H', G.H],
    ['X',  G.X],  ['Y',  G.Y], ['Z', G.Z],
    ['S',  G.S],  ['Si', G.Si],
    ['T',  G.T],  ['Ti', G.Ti],
    ['V',  G.V],  ['Vi', G.Vi],
  ]
  for (const [name, gate] of gates) {
    it(`${name}†·${name} = I₂`, () => expectI2(mul2(adj2(gate), gate)))
  }
})

describe('parameterized 1-qubit gates — U†U = I₂ for arbitrary θ', () => {
  const θ = 1.23
  it('Rx(θ)†·Rx(θ) = I₂', () => expectI2(mul2(adj2(G.Rx(θ)), G.Rx(θ))))
  it('Ry(θ)†·Ry(θ) = I₂', () => expectI2(mul2(adj2(G.Ry(θ)), G.Ry(θ))))
  it('Rz(θ)†·Rz(θ) = I₂', () => expectI2(mul2(adj2(G.Rz(θ)), G.Rz(θ))))
  it('U3(θ,φ,λ)†·U3(θ,φ,λ) = I₂', () => {
    const u = G.U3(1.1, 0.7, 0.4)
    expectI2(mul2(adj2(u), u))
  })
  it('U2(φ,λ)†·U2(φ,λ) = I₂', () => {
    const u = G.U2(0.5, 1.2)
    expectI2(mul2(adj2(u), u))
  })
  it('U1(λ)†·U1(λ) = I₂', () => {
    const u = G.U1(0.9)
    expectI2(mul2(adj2(u), u))
  })
  it('Gpi(φ)†·Gpi(φ) = I₂', () => {
    const u = G.Gpi(0.6)
    expectI2(mul2(adj2(u), u))
  })
  it('Gpi2(φ)†·Gpi2(φ) = I₂', () => {
    const u = G.Gpi2(0.6)
    expectI2(mul2(adj2(u), u))
  })
  it('R2†·R2 = I₂', () => expectI2(mul2(adj2(G.R2), G.R2)))
  it('R4†·R4 = I₂', () => expectI2(mul2(adj2(G.R4), G.R4)))
  it('R8†·R8 = I₂', () => expectI2(mul2(adj2(G.R8), G.R8)))
})

describe('2-qubit gates — U†U = I₄', () => {
  const θ = 0.8
  it('ISwap†·ISwap = I₄',  () => expectI4(mul4(adj4(G.ISwap),  G.ISwap)))
  it('SrSwap†·SrSwap = I₄', () => expectI4(mul4(adj4(G.SrSwap), G.SrSwap)))
  it('Xx(θ)†·Xx(θ) = I₄',  () => expectI4(mul4(adj4(G.Xx(θ)),  G.Xx(θ))))
  it('Yy(θ)†·Yy(θ) = I₄',  () => expectI4(mul4(adj4(G.Yy(θ)),  G.Yy(θ))))
  it('Zz(θ)†·Zz(θ) = I₄',  () => expectI4(mul4(adj4(G.Zz(θ)),  G.Zz(θ))))
  it('Xy(θ)†·Xy(θ) = I₄',  () => expectI4(mul4(adj4(G.Xy(θ)),  G.Xy(θ))))
  it('Ms(φ₀,φ₁)†·Ms(φ₀,φ₁) = I₄', () => {
    const m = G.Ms(0.9, 0.4)
    expectI4(mul4(adj4(m), m))
  })
})

// ─── Exact matrix entries ─────────────────────────────────────────────────────

describe('H — exact entries', () => {
  it('all four entries have magnitude 1/√2', () => {
    expect(G.H[0][0].re).toBeCloseTo( sq2, TOL)
    expect(G.H[0][1].re).toBeCloseTo( sq2, TOL)
    expect(G.H[1][0].re).toBeCloseTo( sq2, TOL)
    expect(G.H[1][1].re).toBeCloseTo(-sq2, TOL)
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++)
        expect(G.H[i as 0][j as 0].im).toBeCloseTo(0, TOL)
  })
  it('is real-symmetric', () => {
    expect(G.H[0][1].re).toBeCloseTo(G.H[1][0].re, TOL)
    expect(G.H[0][1].im).toBeCloseTo(G.H[1][0].im, TOL)
  })
})

describe('S — exact entries', () => {
  it('S[0][0] = 1, S[1][1] = i, off-diagonal = 0', () => {
    expect(G.S[0][0].re).toBe(1); expect(G.S[0][0].im).toBe(0)
    expect(G.S[1][1].re).toBeCloseTo(0, TOL); expect(G.S[1][1].im).toBeCloseTo(1, TOL)
    expect(G.S[0][1].re).toBe(0); expect(G.S[0][1].im).toBe(0)
    expect(G.S[1][0].re).toBe(0); expect(G.S[1][0].im).toBe(0)
  })
})

describe('T — exact entries', () => {
  it('T[0][0] = 1, T[1][1] = e^(iπ/4) = (1+i)/√2, off-diagonal = 0', () => {
    expect(G.T[0][0].re).toBe(1); expect(G.T[0][0].im).toBe(0)
    expect(G.T[1][1].re).toBeCloseTo(sq2, TOL)
    expect(G.T[1][1].im).toBeCloseTo(sq2, TOL)
    expect(G.T[0][1].re).toBe(0); expect(G.T[0][1].im).toBe(0)
    expect(G.T[1][0].re).toBe(0); expect(G.T[1][0].im).toBe(0)
  })
  it('Ti[1][1] = e^(−iπ/4) = (1−i)/√2 — conjugate of T[1][1]', () => {
    expect(G.Ti[1][1].re).toBeCloseTo( sq2, TOL)
    expect(G.Ti[1][1].im).toBeCloseTo(-sq2, TOL)
  })
})

describe('Y — exact entries', () => {
  it('Y = [[0, −i], [i, 0]] exactly', () => {
    expect(G.Y[0][0].re).toBe(0); expect(G.Y[0][0].im).toBe(0)
    expect(G.Y[0][1].re).toBe(0); expect(G.Y[0][1].im).toBe(-1)
    expect(G.Y[1][0].re).toBe(0); expect(G.Y[1][0].im).toBe(1)
    expect(G.Y[1][1].re).toBe(0); expect(G.Y[1][1].im).toBe(0)
  })
})

// ─── Algebraic identities at matrix level ─────────────────────────────────────

describe('matrix-level squaring identities', () => {
  it('H² = I₂', () => expectI2(mul2(G.H, G.H)))
  it('X² = I₂', () => expectI2(mul2(G.X, G.X)))
  it('Y² = I₂', () => expectI2(mul2(G.Y, G.Y)))  // Y² = -(-I) = ... actually Y² = I
  it('Z² = I₂', () => expectI2(mul2(G.Z, G.Z)))
  it('S² = Z  (exact matrix equality)', () => eq2(mul2(G.S, G.S), G.Z))
  it('T² = S  (exact matrix equality)', () => eq2(mul2(G.T, G.T), G.S))
  it('T⁴ = Z  (T⁴ = (T²)² = S² = Z)', () => eq2(mul2(mul2(G.T, G.T), mul2(G.T, G.T)), G.Z))
  it('V² = X  (exact matrix equality)', () => eq2(mul2(G.V, G.V), G.X))
})

describe('inverse pairs — A·A† = I₂', () => {
  it('S·Si = I₂', () => expectI2(mul2(G.S,  G.Si)))
  it('T·Ti = I₂', () => expectI2(mul2(G.T,  G.Ti)))
  it('V·Vi = I₂', () => expectI2(mul2(G.V,  G.Vi)))
  it('Si = S†',   () => eq2(G.Si, adj2(G.S)))
  it('Ti = T†',   () => eq2(G.Ti, adj2(G.T)))
  it('Vi = V†',   () => eq2(G.Vi, adj2(G.V)))
})

// ─── Parameterized 1-qubit gate special values ────────────────────────────────

describe('Rx special values', () => {
  it('Rx(0) = I₂', () => eq2(G.Rx(0), I2))
  it('Rx(2π) = −I₂', () => {
    const M = G.Rx(2 * Math.PI)
    expect(M[0][0].re).toBeCloseTo(-1, TOL); expect(M[0][0].im).toBeCloseTo(0, TOL)
    expect(M[1][1].re).toBeCloseTo(-1, TOL); expect(M[1][1].im).toBeCloseTo(0, TOL)
    expect(M[0][1].re).toBeCloseTo( 0, TOL); expect(M[0][1].im).toBeCloseTo(0, TOL)
  })
  it('Rx(π) = −i·X: off-diagonal entries are −i', () => {
    const M = G.Rx(Math.PI)
    expect(M[0][0].re).toBeCloseTo(0,  TOL); expect(M[0][0].im).toBeCloseTo(0,  TOL)
    expect(M[0][1].re).toBeCloseTo(0,  TOL); expect(M[0][1].im).toBeCloseTo(-1, TOL)
    expect(M[1][0].re).toBeCloseTo(0,  TOL); expect(M[1][0].im).toBeCloseTo(-1, TOL)
    expect(M[1][1].re).toBeCloseTo(0,  TOL); expect(M[1][1].im).toBeCloseTo(0,  TOL)
  })
  it('Rx(θ)·Rx(−θ) = I₂', () => {
    const θ = 0.7
    expectI2(mul2(G.Rx(θ), G.Rx(-θ)))
  })
})

describe('Ry special values', () => {
  it('Ry(0) = I₂', () => eq2(G.Ry(0), I2))
  it('Ry(2π) = −I₂', () => {
    const M = G.Ry(2 * Math.PI)
    expect(M[0][0].re).toBeCloseTo(-1, TOL)
    expect(M[1][1].re).toBeCloseTo(-1, TOL)
  })
  it('Ry(π) is real: [[0,−1],[1,0]]', () => {
    const M = G.Ry(Math.PI)
    expect(M[0][0].re).toBeCloseTo( 0, TOL); expect(M[0][0].im).toBeCloseTo(0, TOL)
    expect(M[0][1].re).toBeCloseTo(-1, TOL); expect(M[0][1].im).toBeCloseTo(0, TOL)
    expect(M[1][0].re).toBeCloseTo( 1, TOL); expect(M[1][0].im).toBeCloseTo(0, TOL)
    expect(M[1][1].re).toBeCloseTo( 0, TOL); expect(M[1][1].im).toBeCloseTo(0, TOL)
  })
  it('Ry(θ)·Ry(−θ) = I₂', () => {
    const θ = 0.7
    expectI2(mul2(G.Ry(θ), G.Ry(-θ)))
  })
})

describe('Rz special values', () => {
  it('Rz(0) = I₂', () => eq2(G.Rz(0), I2))
  it('Rz(2π) = −I₂', () => {
    const M = G.Rz(2 * Math.PI)
    expect(M[0][0].re).toBeCloseTo(-1, TOL)
    expect(M[1][1].re).toBeCloseTo(-1, TOL)
  })
  it('Rz(π) = −i·Z: diagonal entries are −i and +i', () => {
    const M = G.Rz(Math.PI)
    expect(M[0][0].re).toBeCloseTo( 0, TOL); expect(M[0][0].im).toBeCloseTo(-1, TOL)
    expect(M[1][1].re).toBeCloseTo( 0, TOL); expect(M[1][1].im).toBeCloseTo( 1, TOL)
    expect(M[0][1].re).toBeCloseTo( 0, TOL); expect(M[0][1].im).toBeCloseTo( 0, TOL)
    expect(M[1][0].re).toBeCloseTo( 0, TOL); expect(M[1][0].im).toBeCloseTo( 0, TOL)
  })
  it('Rz(θ)·Rz(−θ) = I₂', () => {
    const θ = 0.7
    expectI2(mul2(G.Rz(θ), G.Rz(-θ)))
  })
})

// ─── U1 / U2 / U3 matrix identities ──────────────────────────────────────────

describe('U1 matches named gates at matrix level', () => {
  it('U1(π) = Z', () => eq2(G.U1(Math.PI), G.Z))
  it('U1(π/2) = S', () => eq2(G.U1(Math.PI / 2), G.S))
  it('U1(π/4) = T', () => eq2(G.U1(Math.PI / 4), G.T))
  it('U1(0) = I₂', () => eq2(G.U1(0), I2))
})

describe('U3 special cases match named gates at matrix level', () => {
  it('U3(π, 0, π) = X', () => eq2(G.U3(Math.PI, 0, Math.PI), G.X))
  it('U3(π/2, 0, π) = H', () => eq2(G.U3(Math.PI / 2, 0, Math.PI), G.H))
  it('U2(0, π) = H',      () => eq2(G.U2(0, Math.PI), G.H))
  it('U3(0, 0, λ) = U1(λ) for arbitrary λ', () => {
    const λ = 1.3
    eq2(G.U3(0, 0, λ), G.U1(λ))
  })
  it('U3(θ, 0, 0) = Ry(θ) for arbitrary θ', () => {
    const θ = 0.9
    eq2(G.U3(θ, 0, 0), G.Ry(θ))
  })
})

// ─── R2 / R4 / R8 — defined as Rz aliases ────────────────────────────────────

describe('R2 / R4 / R8 match Rz at matrix level', () => {
  it('R2 = Rz(π/2)',   () => eq2(G.R2, G.Rz(Math.PI / 2)))
  it('R4 = Rz(π/4)',   () => eq2(G.R4, G.Rz(Math.PI / 4)))
  it('R8 = Rz(π/8)',   () => eq2(G.R8, G.Rz(Math.PI / 8)))
})

// ─── GPI / GPI2 matrix-level identities ──────────────────────────────────────

describe('Gpi exact entries and identities', () => {
  it('Gpi(0) = X at matrix level', () => eq2(G.Gpi(0), G.X))
  it('Gpi(π/2) = Y at matrix level', () => eq2(G.Gpi(Math.PI / 2), G.Y))
  it('Gpi is Hermitian: Gpi† = Gpi', () => {
    const φ = 0.8
    eq2(adj2(G.Gpi(φ)), G.Gpi(φ))
  })
  it('Gpi² = I₂ for arbitrary φ (self-inverse)', () => {
    const φ = 1.1
    expectI2(mul2(G.Gpi(φ), G.Gpi(φ)))
  })
  it('Gpi(φ)[0][0] = 0 and Gpi(φ)[1][1] = 0 for all φ', () => {
    const φ = 0.5
    expect(G.Gpi(φ)[0][0].re).toBeCloseTo(0, TOL)
    expect(G.Gpi(φ)[0][0].im).toBeCloseTo(0, TOL)
    expect(G.Gpi(φ)[1][1].re).toBeCloseTo(0, TOL)
    expect(G.Gpi(φ)[1][1].im).toBeCloseTo(0, TOL)
  })
  it('off-diagonal entries are complex conjugates: Gpi[0][1] = conj(Gpi[1][0])', () => {
    const φ = 1.2
    const M = G.Gpi(φ)
    expect(M[0][1].re).toBeCloseTo( M[1][0].re, TOL)
    expect(M[0][1].im).toBeCloseTo(-M[1][0].im, TOL)
  })
})

describe('Gpi2 matrix-level identities', () => {
  it('Gpi2(0) top-left = 1/√2, no imaginary part', () => {
    expect(G.Gpi2(0)[0][0].re).toBeCloseTo(sq2, TOL)
    expect(G.Gpi2(0)[0][0].im).toBeCloseTo(0, TOL)
  })
  it('Gpi2(0)² = −i·X: entries [[0,−i],[−i,0]]', () => {
    const M = mul2(G.Gpi2(0), G.Gpi2(0))
    expect(M[0][0].re).toBeCloseTo( 0, TOL); expect(M[0][0].im).toBeCloseTo( 0, TOL)
    expect(M[0][1].re).toBeCloseTo( 0, TOL); expect(M[0][1].im).toBeCloseTo(-1, TOL)
    expect(M[1][0].re).toBeCloseTo( 0, TOL); expect(M[1][0].im).toBeCloseTo(-1, TOL)
    expect(M[1][1].re).toBeCloseTo( 0, TOL); expect(M[1][1].im).toBeCloseTo( 0, TOL)
  })
  it('Gpi2(φ)·Gpi2(φ+π) = I₂ for arbitrary φ', () => {
    const φ = 0.7
    expectI2(mul2(G.Gpi2(φ), G.Gpi2(φ + Math.PI)))
  })
  it('Gpi2(0) entries: [[sq2, −i·sq2],[−i·sq2, sq2]]', () => {
    const M = G.Gpi2(0)
    expect(M[0][0].re).toBeCloseTo( sq2, TOL); expect(M[0][0].im).toBeCloseTo(   0, TOL)
    expect(M[0][1].re).toBeCloseTo(   0, TOL); expect(M[0][1].im).toBeCloseTo(-sq2, TOL)
    expect(M[1][0].re).toBeCloseTo(   0, TOL); expect(M[1][0].im).toBeCloseTo(-sq2, TOL)
    expect(M[1][1].re).toBeCloseTo( sq2, TOL); expect(M[1][1].im).toBeCloseTo(   0, TOL)
  })
})

// ─── Two-qubit gate special values ───────────────────────────────────────────

describe('2-qubit gates — θ=0 gives I₄', () => {
  it('Xx(0) = I₄', () => eq4(G.Xx(0), I4))
  it('Yy(0) = I₄', () => eq4(G.Yy(0), I4))
  it('Zz(0) = I₄', () => eq4(G.Zz(0), I4))
  it('Xy(0) = I₄', () => eq4(G.Xy(0), I4))
})

describe('2-qubit gate inverse: gate(θ)·gate(−θ) = I₄', () => {
  const θ = 0.9
  it('Xx(θ)·Xx(−θ) = I₄', () => eq4(mul4(G.Xx( θ), G.Xx(-θ)), I4))
  it('Yy(θ)·Yy(−θ) = I₄', () => eq4(mul4(G.Yy( θ), G.Yy(-θ)), I4))
  it('Zz(θ)·Zz(−θ) = I₄', () => eq4(mul4(G.Zz( θ), G.Zz(-θ)), I4))
  it('Xy(θ)·Xy(−θ) = I₄', () => eq4(mul4(G.Xy( θ), G.Xy(-θ)), I4))
})

describe('Xy special values match named gates exactly', () => {
  it('Xy(π) = ISwap', () => eq4(G.Xy(Math.PI), G.ISwap))
  it('Xy(π/2) = SrSwap', () => eq4(G.Xy(Math.PI / 2), G.SrSwap))
})

describe('ISwap² = diag(1,−1,−1,1)', () => {
  it('ISwap² diagonal is (1, −1, −1, 1)', () => {
    const M = mul4(G.ISwap, G.ISwap)
    expect(M[0][0].re).toBeCloseTo( 1, TOL); expect(M[0][0].im).toBeCloseTo(0, TOL)
    expect(M[1][1].re).toBeCloseTo(-1, TOL); expect(M[1][1].im).toBeCloseTo(0, TOL)
    expect(M[2][2].re).toBeCloseTo(-1, TOL); expect(M[2][2].im).toBeCloseTo(0, TOL)
    expect(M[3][3].re).toBeCloseTo( 1, TOL); expect(M[3][3].im).toBeCloseTo(0, TOL)
  })
  it('ISwap² off-diagonal entries are all zero', () => {
    const M = mul4(G.ISwap, G.ISwap)
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++)
        if (i !== j) {
          expect(M[i as 0][j as 0].re).toBeCloseTo(0, TOL)
          expect(M[i as 0][j as 0].im).toBeCloseTo(0, TOL)
        }
  })
})

// ─── MS gate identities at matrix level ──────────────────────────────────────

describe('Ms matrix-level identities', () => {
  it('Ms(0, 0) = Xx(π/2) exactly', () => eq4(G.Ms(0, 0), G.Xx(Math.PI / 2)))
  it('Ms(π/2, π/2) = Yy(π/2) exactly', () => eq4(G.Ms(Math.PI / 2, Math.PI / 2), G.Yy(Math.PI / 2)))
  it('Ms(φ₀, φ₁)·Ms(φ₀, φ₁)† = I₄ for arbitrary angles', () => {
    const m = G.Ms(1.1, 0.7)
    expectI4(mul4(m, adj4(m)))
  })
})

// ─── Zz decomposition: Zz diagonal entries ───────────────────────────────────

describe('Zz(θ) diagonal structure', () => {
  it('Zz(θ) is diagonal', () => {
    const M = G.Zz(0.6)
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++)
        if (i !== j) {
          expect(M[i as 0][j as 0].re).toBeCloseTo(0, TOL)
          expect(M[i as 0][j as 0].im).toBeCloseTo(0, TOL)
        }
  })
  it('Zz(θ): |00⟩ and |11⟩ get phase e^(−iθ/2), |01⟩ and |10⟩ get e^(iθ/2)', () => {
    const θ = 0.8
    const M = G.Zz(θ)
    const expMinus: Complex = c(Math.cos(θ/2), -Math.sin(θ/2))
    const expPlus:  Complex = c(Math.cos(θ/2),  Math.sin(θ/2))
    expect(M[0][0].re).toBeCloseTo(expMinus.re, TOL); expect(M[0][0].im).toBeCloseTo(expMinus.im, TOL)
    expect(M[1][1].re).toBeCloseTo(expPlus.re,  TOL); expect(M[1][1].im).toBeCloseTo(expPlus.im,  TOL)
    expect(M[2][2].re).toBeCloseTo(expPlus.re,  TOL); expect(M[2][2].im).toBeCloseTo(expPlus.im,  TOL)
    expect(M[3][3].re).toBeCloseTo(expMinus.re, TOL); expect(M[3][3].im).toBeCloseTo(expMinus.im, TOL)
  })
})
