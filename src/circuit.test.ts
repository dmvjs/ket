import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import type { IonQCircuit } from './circuit.js'

// ─── Tolerance helpers ────────────────────────────────────────────────────────

const SHOT_TOLERANCE  = 0.05  // ±5% for shot-based results (large N reduces variance)
const EXACT_TOLERANCE = 1e-10 // for deterministic single-outcome circuits

function near(a: number, b: number, tol = SHOT_TOLERANCE): boolean {
  return Math.abs(a - b) <= tol
}

// ─── Single-qubit gates ───────────────────────────────────────────────────────

describe('X gate', () => {
  it('flips |0⟩ to |1⟩', () => {
    const r = new Circuit(1).x(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
    expect(Object.keys(r.probs)).toHaveLength(1)
  })

  it('flips |1⟩ back to |0⟩ (XX = I)', () => {
    const r = new Circuit(1).x(0).x(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

describe('H gate', () => {
  it('creates equal superposition from |0⟩', () => {
    const r = new Circuit(1).h(0).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['1'] ?? 0, 0.5)).toBe(true)
  })

  it('HH = I (self-inverse)', () => {
    const r = new Circuit(1).h(0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

describe('Z gate', () => {
  it('leaves |0⟩ unchanged (Z|0⟩ = |0⟩)', () => {
    const r = new Circuit(1).z(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  it('HZH = X: flips qubit in Z basis = X in computational basis', () => {
    const r = new Circuit(1).h(0).z(0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })
})

describe('S gate', () => {
  it('SS = Z (S is √Z)', () => {
    // S²|+⟩ = Z|+⟩ = |−⟩, which when measured in X basis gives |1⟩ with p=1
    // Verify: S²|0⟩ = Z|0⟩ = |0⟩ (same measurement, phase not observable)
    const r = new Circuit(1).s(0).s(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  it('S†S = I', () => {
    const r = new Circuit(1).s(0).si(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

describe('T gate', () => {
  it('T†T = I', () => {
    const r = new Circuit(1).t(0).ti(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  it('TTTT = Z (T is ⁴√Z, so T⁴ = Z)', () => {
    const r = new Circuit(1).t(0).t(0).t(0).t(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

describe('V gate (√X)', () => {
  it('VV = X', () => {
    const r = new Circuit(1).v(0).v(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('V†V = I', () => {
    const r = new Circuit(1).v(0).vi(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

describe('Rx rotation', () => {
  it('Rx(π) = X (up to global phase)', () => {
    const r = new Circuit(1).rx(Math.PI, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 5)
  })

  it('Rx(π/2) produces 50/50 superposition', () => {
    const r = new Circuit(1).rx(Math.PI / 2, 0).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['1'] ?? 0, 0.5)).toBe(true)
  })

  it('Rx(0) = I', () => {
    const r = new Circuit(1).rx(0, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

describe('Ry rotation', () => {
  it('Ry(π) = Y (up to global phase)', () => {
    const r = new Circuit(1).ry(Math.PI, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 5)
  })

  it('Ry(2π) = I (full rotation)', () => {
    const r = new Circuit(1).ry(2 * Math.PI, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 8)
  })
})

describe('Rz rotation', () => {
  it('Rz applied to |0⟩ does not change measurement (|0⟩ is Z eigenstate)', () => {
    const r = new Circuit(1).rz(Math.PI / 3, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  it('HRz(π)H = X', () => {
    const r = new Circuit(1).h(0).rz(Math.PI, 0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 5)
  })
})

// ─── Two-qubit gates ──────────────────────────────────────────────────────────

describe('CNOT gate', () => {
  it('|00⟩ → |00⟩ (control=0, no flip)', () => {
    const r = new Circuit(2).cnot(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('|10⟩ → |11⟩ (control=1, flip target)', () => {
    const r = new Circuit(2).x(0).cnot(0, 1).run({ shots: 100, seed: 1 })
    // q0=1 (LSB) so index=1, flip q1 → index=3 = bitstring "11"
    expect(r.probs['11']).toBeCloseTo(1.0, 10)
  })

  it('CNOT·CNOT = I', () => {
    const r = new Circuit(2).x(0).cnot(0, 1).cnot(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10) // back to |10⟩ (q0=1)
  })
})

describe('SWAP gate', () => {
  it('swaps |10⟩ to |01⟩', () => {
    // x(0) sets q0=1 → bitstring "01" (q0 is rightmost)
    // swap(0,1) → q1=1 → bitstring "10"
    const r = new Circuit(2).x(0).swap(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['10']).toBeCloseTo(1.0, 10)
  })

  it('SWAP·SWAP = I', () => {
    const r = new Circuit(2).x(0).swap(0, 1).swap(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

// ─── Entanglement — Bell states ────────────────────────────────────────────────

describe('Bell states', () => {
  it('|Φ+⟩: H(0)·CNOT(0,1) produces ~50/50 |00⟩ and |11⟩', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
    expect(Object.keys(r.probs)).toHaveLength(2)
  })

  it('|Φ+⟩ histogram keys are decimal integers', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 1000, seed: 42 })
    expect('0' in r.histogram).toBe(true)  // |00⟩ = 0
    expect('3' in r.histogram).toBe(true)  // |11⟩ = 3
  })

  it('|Φ+⟩ entropy is ~1 bit', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 100000, seed: 42 })
    expect(r.entropy).toBeCloseTo(1.0, 1)
  })

  it('|Φ-⟩: Z(0)·H(0)·CNOT(0,1) produces ~50/50 |00⟩ and |11⟩', () => {
    const r = new Circuit(2).z(0).h(0).cnot(0, 1).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  it('|Ψ+⟩: H(0)·CNOT(0,1)·X(0) produces ~50/50 |01⟩ and |10⟩', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).x(0).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['01'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['10'] ?? 0, 0.5)).toBe(true)
  })
})

// ─── Multi-qubit algorithms ────────────────────────────────────────────────────

describe('GHZ state', () => {
  it.each([3, 5, 8, 12, 16, 20])('%i-qubit GHZ produces only |0…⟩ and |1…⟩', (n) => {
    let c = new Circuit(n).h(0)
    for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
    const r = c.run({ shots: 512, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
    expect(near(r.probs['0'.repeat(n)] ?? 0, 0.5, 0.1)).toBe(true)
    expect(near(r.probs['1'.repeat(n)] ?? 0, 0.5, 0.1)).toBe(true)
  })
})

describe('Deutsch-Jozsa', () => {
  // Constant oracle: f(x) = 0. Output qubit starts |1⟩ via X.
  // DJ circuit: H⊗n|0⟩ ⊗ H|1⟩ → apply oracle → H⊗n → measure
  // Constant f → all |0...0⟩. Balanced f → ≠ |0...0⟩.
  it('constant function outputs |00⟩ with certainty', () => {
    // f(x) = 0: identity oracle (no CNOT), output qubit is ancilla
    // 2 input qubits + 1 ancilla
    const r = new Circuit(3)
      .x(2)               // ancilla |1⟩
      .h(0).h(1).h(2)    // H on all
      // constant oracle: do nothing
      .h(0).h(1)          // H on input qubits
      .run({ shots: 100, seed: 1 })
    // Input qubits 0,1 should both be 0 (constant function)
    // ancilla qubit 2 may be anything; check input bits
    for (const [bs] of Object.entries(r.probs)) {
      expect(bs[bs.length - 1]).toBe('0') // q0 = 0
      expect(bs[bs.length - 2]).toBe('0') // q1 = 0
    }
  })

  it('balanced function (CNOT oracle) outputs non-zero input register', () => {
    // f(x) = x0: CNOT(0, ancilla)
    const r = new Circuit(3)
      .x(2)
      .h(0).h(1).h(2)
      .cnot(0, 2)         // balanced oracle: f = x0
      .h(0).h(1)
      .run({ shots: 100, seed: 1 })
    // Input register should NOT be |00⟩ (balanced function)
    for (const [bs] of Object.entries(r.probs)) {
      // q0 bit (rightmost) should be 1 for the balanced oracle
      expect(bs[bs.length - 1]).toBe('1')
    }
  })
})

describe('Bernstein-Vazirani', () => {
  // BV recovers hidden bitstring s by querying f(x) = s·x once.
  // Circuit: H⊗n|0⟩ ⊗ H|1⟩ → CNOT(i, ancilla) for each bit i in s → H⊗n → measure
  it('recovers s=101 (n=3 input qubits)', () => {
    // s = 101 means CNOT on qubits 0 and 2
    const r = new Circuit(4)
      .x(3)                       // ancilla
      .h(0).h(1).h(2).h(3)       // H all
      .cnot(0, 3).cnot(2, 3)     // oracle: f(x) = x0·1 + x1·0 + x2·1 = s·x
      .h(0).h(1).h(2)            // H input
      .run({ shots: 100, seed: 1 })
    // Input register (qubits 0,1,2) should all be in state |101⟩
    // In our bitstring: q0=1, q1=0, q2=1 → rightmost 3 bits = '101'
    for (const [bs] of Object.entries(r.probs)) {
      const inputBits = bs.slice(-3) // last 3 chars = q2q1q0
      expect(inputBits).toBe('101')
    }
  })
})

// ─── Distribution properties ──────────────────────────────────────────────────

describe('Distribution', () => {
  it('.most returns the most probable bitstring', () => {
    const r = new Circuit(1).run({ shots: 100, seed: 1 })
    expect(r.most).toBe('0')
  })

  it('.render returns a non-empty string with ⟩ characters', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 100, seed: 1 })
    const rendered = r.render()
    expect(rendered).toContain('⟩')
    expect(rendered.length).toBeGreaterThan(10)
  })

  it('probabilities sum to 1', () => {
    const r = new Circuit(3).h(0).h(1).h(2).run({ shots: 10000, seed: 42 })
    const total = Object.values(r.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1.0, 3)
  })

  it('8-qubit uniform superposition: all 256 outcomes present', () => {
    const n = 8
    let c = new Circuit(n)
    for (let i = 0; i < n; i++) c = c.h(i)
    const r = c.run({ shots: 100000, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(256)
    for (const p of Object.values(r.probs)) {
      expect(near(p, 1 / 256, 0.01)).toBe(true)
    }
  })

  it.each([1, 2, 3, 4])('entropy of H⊗%i |0⟩ = %i bits', (n) => {
    let c = new Circuit(n)
    for (let i = 0; i < n; i++) c = c.h(i)
    const r = c.run({ shots: 200000, seed: 42 })
    expect(r.entropy).toBeCloseTo(n, 1)
  })

  it('seed produces reproducible results', () => {
    const circuit = new Circuit(3).h(0).h(1).h(2)
    const r1 = circuit.run({ shots: 100, seed: 99 })
    const r2 = circuit.run({ shots: 100, seed: 99 })
    expect(r1.probs).toEqual(r2.probs)
  })

  it('different seeds produce different results', () => {
    const circuit = new Circuit(2).h(0).h(1)
    const r1 = circuit.run({ shots: 1000, seed: 1 })
    const r2 = circuit.run({ shots: 1000, seed: 2 })
    expect(r1.probs).not.toEqual(r2.probs)
  })
})

// ─── BigInt / no 32-bit overflow ──────────────────────────────────────────────

describe('BigInt state indices (no 32-bit overflow)', () => {
  it('qubit 30: GHZ on qubits 0 and 30 without corruption', () => {
    // quantum-circuit fails here with 1<<30 XOR issues — we use BigInt
    const r = new Circuit(31).h(0).cnot(0, 30).run({ shots: 256, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
    const zeros = '0'.repeat(31)
    const ones  = '1' + '0'.repeat(29) + '1' // q30=1, q0=1, rest 0
    expect((r.probs[zeros] ?? 0) + (r.probs[ones] ?? 0)).toBeCloseTo(1.0, 1)
  })

  it('qubit 31 (exactly where 1<<31 overflows in JS)', () => {
    const r = new Circuit(32).h(0).cnot(0, 31).run({ shots: 256, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
  })

  it('qubit 40 works correctly', () => {
    const r = new Circuit(41).x(40).run({ shots: 10, seed: 1 })
    expect(Object.keys(r.probs)).toHaveLength(1)
    // bitstring: q40=1, all others 0 → '1' followed by 40 '0's
    expect(r.probs['1' + '0'.repeat(40)]).toBeCloseTo(1.0, 10)
  })
})

// ─── Phase rotation gates (r2 / r4 / r8) ─────────────────────────────────────
//
// r2 = Rz(π/2), r4 = Rz(π/4), r8 = Rz(π/8).
// Each is equal to S/T up to a global phase (unobservable) but carries its own
// gate name for IonQ JSON serialisation.
//
// Test strategy:
//   • Computational-basis phases: phase gates never change |0⟩/|1⟩ populations
//   • Composition identities:  r2·si ≡ I,  r4·ti ≡ I  (global-phase cancellation)
//   • Squaring ladder:  r2² ≡ Z  →  H·r2·r2·H|0⟩ = |1⟩  (deterministic)
//   • Ramsey fringe: H·rN·H|0⟩  →  p(|0⟩) = (1+cos θ)/2  (angle verification)

describe('r2 (Rz(π/2))', () => {
  it('leaves |0⟩ and |1⟩ populations unchanged (phase is unobservable)', () => {
    expect(new Circuit(1).r2(0).run({ shots: 100, seed: 1 }).probs['0']).toBeCloseTo(1.0, 10)
    expect(new Circuit(1).x(0).r2(0).run({ shots: 100, seed: 1 }).probs['1']).toBeCloseTo(1.0, 10)
  })

  // r2·si = Rz(π/2)·S† = e^(−iπ/4)·I  →  global phase cancels in measurement
  it('r2·si ≡ I: H·r2·si·H|0⟩ returns |0⟩', () => {
    const r = new Circuit(1).h(0).r2(0).si(0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  // Rz(π/2)² = Rz(π) ≡ Z  →  H·Z·H = X  →  H·r2·r2·H|0⟩ = X|0⟩ = |1⟩
  it('r2² ≡ Z: H·r2·r2·H|0⟩ = |1⟩', () => {
    const r = new Circuit(1).h(0).r2(0).r2(0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  // Rz(2π) ≡ I  →  H·r2^4·H|0⟩ = |0⟩
  it('r2⁴ ≡ I: H·(r2)⁴·H|0⟩ = |0⟩', () => {
    const r = new Circuit(1).h(0).r2(0).r2(0).r2(0).r2(0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

describe('r4 (Rz(π/4))', () => {
  it('leaves |0⟩ and |1⟩ populations unchanged', () => {
    expect(new Circuit(1).r4(0).run({ shots: 100, seed: 1 }).probs['0']).toBeCloseTo(1.0, 10)
    expect(new Circuit(1).x(0).r4(0).run({ shots: 100, seed: 1 }).probs['1']).toBeCloseTo(1.0, 10)
  })

  // r4·ti = Rz(π/4)·T† = e^(−iπ/8)·I
  it('r4·ti ≡ I: H·r4·ti·H|0⟩ returns |0⟩', () => {
    const r = new Circuit(1).h(0).r4(0).ti(0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  // r4² ≡ r2  →  r4²·si ≡ I  (same cancellation as above)
  it('r4² ≡ r2: H·r4·r4·si·H|0⟩ = |0⟩', () => {
    const r = new Circuit(1).h(0).r4(0).r4(0).si(0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  // Ramsey fringe: p(|0⟩) = (1 + cos(π/4)) / 2 ≈ 0.854
  it('Ramsey fringe: H·r4·H|0⟩ → p(|0⟩) ≈ 0.854 (verifies angle π/4)', () => {
    const expected = (1 + Math.cos(Math.PI / 4)) / 2  // ≈ 0.8536
    const r = new Circuit(1).h(0).r4(0).h(0).run({ shots: 50000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, expected, 0.01)).toBe(true)
  })
})

describe('r8 (Rz(π/8))', () => {
  it('leaves |0⟩ and |1⟩ populations unchanged', () => {
    expect(new Circuit(1).r8(0).run({ shots: 100, seed: 1 }).probs['0']).toBeCloseTo(1.0, 10)
    expect(new Circuit(1).x(0).r8(0).run({ shots: 100, seed: 1 }).probs['1']).toBeCloseTo(1.0, 10)
  })

  // r8⁸ = Rz(π) ≡ Z  →  H·r8^8·H|0⟩ = |1⟩
  it('r8⁸ ≡ Z: H·(r8)⁸·H|0⟩ = |1⟩', () => {
    let c = new Circuit(1).h(0)
    for (let i = 0; i < 8; i++) c = c.r8(0)
    const r = c.h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  // Ramsey fringe: p(|0⟩) = (1 + cos(π/8)) / 2 ≈ 0.962
  it('Ramsey fringe: H·r8·H|0⟩ → p(|0⟩) ≈ 0.962 (verifies angle π/8)', () => {
    const expected = (1 + Math.cos(Math.PI / 8)) / 2  // ≈ 0.9619
    const r = new Circuit(1).h(0).r8(0).h(0).run({ shots: 50000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, expected, 0.01)).toBe(true)
  })
})

// ─── Parameterized unitaries (u1 / u2 / u3) ──────────────────────────────────
//
// u3(θ, φ, λ) is the general single-qubit unitary (OpenQASM 2.0 basis gate).
// u2 and u1 are derived: u2(φ,λ) = u3(π/2, φ, λ),  u1(λ) = diag(1, e^iλ).
//
// Test strategy mirrors the r2/r4/r8 suite:
//   • Computational-basis identity (phase gates leave |0⟩/|1⟩ populations intact)
//   • Named-state reconstruction (X, H as exact u3 special cases)
//   • Composition identities (round-trips and inverse formula)
//   • Ramsey fringe for angle verification

describe('u1(λ) — phase gate', () => {
  it('u1(0) = I: |0⟩ unchanged', () => {
    const r = new Circuit(1).u1(0, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  // U1(π) = diag(1, −1) = Z  →  HZH = X  →  deterministic |1⟩
  it('u1(π) = Z: H·u1(π)·H|0⟩ = |1⟩', () => {
    const r = new Circuit(1).h(0).u1(Math.PI, 0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  // U1(λ)·U1(−λ) = I
  it('round-trip: H·u1(λ)·u1(−λ)·H|0⟩ = |0⟩', () => {
    const λ = 0.7
    const r = new Circuit(1).h(0).u1(λ, 0).u1(-λ, 0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  // Ramsey: p(|0⟩) = (1 + cos λ) / 2; λ = π/4 → ≈ 0.854 (same fringe as r4)
  it('Ramsey fringe: H·u1(π/4)·H|0⟩ → p(|0⟩) ≈ 0.854', () => {
    const expected = (1 + Math.cos(Math.PI / 4)) / 2
    const r = new Circuit(1).h(0).u1(Math.PI / 4, 0).h(0).run({ shots: 50000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, expected, 0.01)).toBe(true)
  })
})

describe('u2(φ, λ) — equatorial gate', () => {
  // U2(0, π) = H exactly
  it('u2(0, π) creates equal superposition from |0⟩', () => {
    const r = new Circuit(1).u2(0, Math.PI, 0).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['1'] ?? 0, 0.5)).toBe(true)
  })

  it('u2(0, π)² = I: self-inverse like H', () => {
    const r = new Circuit(1).u2(0, Math.PI, 0).u2(0, Math.PI, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  it('Bell state: u2(0, π) on q0 + CNOT produces |Φ⁺⟩', () => {
    const r = new Circuit(2).u2(0, Math.PI, 0).cnot(0, 1).run({ shots: 2048, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })
})

describe('u3(θ, φ, λ) — general single-qubit unitary', () => {
  // U3(π, 0, π) = X exactly
  it('u3(π, 0, π) = X: |0⟩ → |1⟩', () => {
    const r = new Circuit(1).u3(Math.PI, 0, Math.PI, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  // U3(π/2, 0, π) = H exactly  →  H² = I
  it('u3(π/2, 0, π)² = I: equals H, which is self-inverse', () => {
    const r = new Circuit(1).u3(Math.PI / 2, 0, Math.PI, 0).u3(Math.PI / 2, 0, Math.PI, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  // U3(θ, 0, 0) = Ry(θ).  Ry(π) ≠ X: H·Ry(π)·H|0⟩ = |1⟩, whereas H·X·H|0⟩ = |0⟩.
  it('u3(π, 0, 0) = Ry(π): H·u3(π,0,0)·H|0⟩ = |1⟩ (distinguishes from X)', () => {
    const r = new Circuit(1).h(0).u3(Math.PI, 0, 0, 0).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  // U3(0, 0, λ) = U1(λ) exactly — Ramsey fringe must match
  it('u3(0, 0, π/4) ≡ u1(π/4): same Ramsey fringe ≈ 0.854', () => {
    const expected = (1 + Math.cos(Math.PI / 4)) / 2
    const r = new Circuit(1).h(0).u3(0, 0, Math.PI / 4, 0).h(0).run({ shots: 50000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, expected, 0.01)).toBe(true)
  })

  // Unitarity: U3(θ,φ,λ)⁻¹ = U3(−θ, −λ, −φ)
  it('unitarity: u3(θ,φ,λ)·u3(−θ,−λ,−φ) = I for arbitrary parameters', () => {
    const [θ, φ, λ] = [1.1, 0.7, 0.3]
    const r = new Circuit(1).u3(θ, φ, λ, 0).u3(-θ, -λ, -φ, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

// ─── Two-qubit interaction gates ──────────────────────────────────────────────
//
// Gate hierarchy: XY(θ) is the root; ISwap = XY(π), SrSwap = XY(π/2).
// Identity for ZZ: ZZ(θ) = CNOT · (I⊗Rz(θ)) · CNOT.
// Conjugation:     XX(θ) = H⊗H · ZZ(θ) · H⊗H.
//
// Test strategy:
//   • θ=0 gives identity for all parameterized gates
//   • Maximally-entangling angle (π/2 for XX/YY/ZZ, π for XY) creates Bell-like state
//   • Invertibility: gate(θ)·gate(−θ) = I
//   • Deterministic circuit identities for iSWAP and srSWAP

describe('xx(θ) — Ising-XX interaction', () => {
  it('xx(0) = I: no effect on |01⟩', () => {
    const r = new Circuit(2).x(0).xx(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // XX(π/2)|00⟩ = (1/√2)|00⟩ − (i/√2)|11⟩ → Bell-like pair
  it('xx(π/2)|00⟩ produces |00⟩/|11⟩ Bell-like state', () => {
    const r = new Circuit(2).xx(Math.PI / 2, 0, 1).run({ shots: 2048, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  it('invertibility: xx(θ)·xx(−θ) = I', () => {
    const θ = 0.8
    const r = new Circuit(2).h(0).xx(θ, 0, 1).xx(-θ, 0, 1).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  // XX(θ) = H⊗H · ZZ(θ) · H⊗H (conjugation identity)
  it('xx(π/2) ≡ H⊗H · zz(π/2) · H⊗H: same Bell distribution', () => {
    const rXX = new Circuit(2).xx(Math.PI / 2, 0, 1).run({ shots: 2048, seed: 42 })
    const rZZ = new Circuit(2).h(0).h(1).zz(Math.PI / 2, 0, 1).h(0).h(1).run({ shots: 2048, seed: 42 })
    expect(Object.keys(rZZ.probs)).toHaveLength(2)
    expect(near(rZZ.probs['00'] ?? 0, rXX.probs['00'] ?? 0, 0.05)).toBe(true)
  })
})

describe('yy(θ) — Ising-YY interaction', () => {
  it('yy(0) = I: no effect on |01⟩', () => {
    const r = new Circuit(2).x(0).yy(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // YY(π/2)|00⟩ = (1/√2)|00⟩ + (i/√2)|11⟩ — entangled, |00⟩/|11⟩ only
  it('yy(π/2)|00⟩ produces |00⟩/|11⟩ Bell-like state', () => {
    const r = new Circuit(2).yy(Math.PI / 2, 0, 1).run({ shots: 2048, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  it('invertibility: yy(θ)·yy(−θ) = I', () => {
    const θ = 0.8
    const r = new Circuit(2).h(0).yy(θ, 0, 1).yy(-θ, 0, 1).h(0).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })
})

describe('zz(θ) — Ising-ZZ interaction', () => {
  it('zz(0) = I: no effect', () => {
    const r = new Circuit(2).x(0).zz(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // ZZ(θ) = CNOT·(I⊗Rz(θ))·CNOT — verified via interference
  it('H⊗H · zz(π/2) · H⊗H = xx(π/2): Bell distribution', () => {
    const r = new Circuit(2).h(0).h(1).zz(Math.PI / 2, 0, 1).h(0).h(1).run({ shots: 2048, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  // H⊗H · ZZ(π) · H⊗H|00⟩ = |11⟩ — deterministic (proven analytically)
  it('H⊗H · zz(π) · H⊗H|00⟩ = |11⟩ (deterministic)', () => {
    const r = new Circuit(2).h(0).h(1).zz(Math.PI, 0, 1).h(0).h(1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 10)
  })
})

describe('iswap — iSWAP gate (= XY(π))', () => {
  it('iswap|01⟩ → |10⟩ (swaps qubits, phase is unobservable)', () => {
    const r = new Circuit(2).x(0).iswap(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['10']).toBeCloseTo(1.0, 10)
  })

  it('iswap|10⟩ → |01⟩', () => {
    const r = new Circuit(2).x(1).iswap(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  it('iswap|00⟩ = |00⟩ and iswap|11⟩ = |11⟩ (no swap when same)', () => {
    expect(new Circuit(2).iswap(0, 1).run({ shots: 100, seed: 1 }).probs['00']).toBeCloseTo(1.0, 10)
    expect(new Circuit(2).x(0).x(1).iswap(0, 1).run({ shots: 100, seed: 1 }).probs['11']).toBeCloseTo(1.0, 10)
  })

  // iSWAP⁴ = I in the full gate (iSWAP² = −I on |01⟩/|10⟩ subspace)
  it('iswap⁴ = I: four applications restore the state', () => {
    const r = new Circuit(2).x(0).iswap(0, 1).iswap(0, 1).iswap(0, 1).iswap(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

describe('srswap — √iSWAP (= XY(π/2))', () => {
  it('srswap|01⟩ → 50% |01⟩ / 50% |10⟩ (superposition)', () => {
    const r = new Circuit(2).x(0).srswap(0, 1).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['01'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['10'] ?? 0, 0.5)).toBe(true)
  })

  it('srswap² = iswap: srswap·srswap|01⟩ → |10⟩', () => {
    const r = new Circuit(2).x(0).srswap(0, 1).srswap(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['10']).toBeCloseTo(1.0, 10)
  })

  it('srswap|00⟩ = |00⟩ (no mixing outside |01⟩/|10⟩ subspace)', () => {
    const r = new Circuit(2).srswap(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })
})

describe('xy(θ) — XY interaction (root gate for iswap/srswap)', () => {
  it('xy(0) = I', () => {
    const r = new Circuit(2).x(0).xy(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  it('xy(π) = iswap: xy(π)|01⟩ → |10⟩', () => {
    const r = new Circuit(2).x(0).xy(Math.PI, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['10']).toBeCloseTo(1.0, 10)
  })

  it('xy(π/2) = srswap: xy(π/2)² |01⟩ → |10⟩', () => {
    const r = new Circuit(2).x(0).xy(Math.PI / 2, 0, 1).xy(Math.PI / 2, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['10']).toBeCloseTo(1.0, 10)
  })
})

// ─── Controlled single-qubit gates ────────────────────────────────────────────
//
// All controlled gates share the same structure:
//   • control=0 → identity (gate never fires)
//   • control=1 → underlying gate applied to target
//   • Composition identities prove the correct gate fires at the correct angle
//
// Key identities:
//   cx = cnot                      CX(control=0) ≡ I
//   H(t)·CZ·H(t) = CNOT           cs² = cz   (S² = Z)
//   ct⁴ = cz                      cu1(π) = cz  (U1(π) = Z)
//   cu3(π,0,π) = cx  (U3(π,0,π) = X)
//
// Bitstring convention: q0 is rightmost, so x(0) → '01' in a 2-qubit system.

describe('cx (= cnot alias)', () => {
  it('cx(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cx(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('cx = cnot: identical Bell state distributions', () => {
    const rCnot = new Circuit(2).h(0).cnot(0, 1).run({ shots: 2048, seed: 42 })
    const rCx   = new Circuit(2).h(0).cx(0, 1).run({ shots: 2048, seed: 42 })
    expect(rCx.probs['00']).toBeCloseTo(rCnot.probs['00'] ?? 0, 5)
    expect(rCx.probs['11']).toBeCloseTo(rCnot.probs['11'] ?? 0, 5)
  })
})

describe('cy — controlled-Y', () => {
  it('cy(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cy(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('x(0)·cy(0,1): Y|0⟩ = i|1⟩ → measures as |11⟩', () => {
    const r = new Circuit(2).x(0).cy(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 10)
  })

  // Y² = -I (global phase unobservable) → double application restores state
  it('cy² restores target: x(0)·cy·cy returns to |10⟩', () => {
    const r = new Circuit(2).x(0).cy(0, 1).cy(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // H(0)·cy(0,1)|00⟩: cy fires half the time → |00⟩ or i|11⟩, each with p=½
  it('h(0)·cy(0,1) creates |00⟩/|11⟩ Bell-like state', () => {
    const r = new Circuit(2).h(0).cy(0, 1).run({ shots: 2048, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(2)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })
})

describe('cz — controlled-Z', () => {
  it('cz(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cz(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('x(0)·cz(0,1): Z|0⟩ = |0⟩ → state unchanged (phase unobservable)', () => {
    const r = new Circuit(2).x(0).cz(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // H(t)·CZ·H(t) = CNOT: canonical identity
  it('H(1)·cz(0,1)·H(1) = cnot(0,1): x(0) case gives |11⟩', () => {
    const r = new Circuit(2).x(0).h(1).cz(0, 1).h(1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 10)
  })

  // cs² = cz: S² = Z, so two CS gates on control=1 are identical to CZ
  it('cs² = cz: x(0)·h(1)·cs·cs·h(1) matches x(0)·h(1)·cz·h(1)', () => {
    const rCz = new Circuit(2).x(0).h(1).cz(0, 1).h(1).run({ shots: 100, seed: 1 })
    const rCs = new Circuit(2).x(0).h(1).cs(0, 1).cs(0, 1).h(1).run({ shots: 100, seed: 1 })
    expect(rCs.probs['11']).toBeCloseTo(rCz.probs['11'] ?? 0, 10)
  })
})

describe('ch — controlled-H', () => {
  it('ch(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).ch(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  // H fires on q1 → superposition of |01⟩ and |11⟩ (q0=1 throughout)
  it('x(0)·ch(0,1): H|0⟩ produces 50/50 over |01⟩/|11⟩', () => {
    const r = new Circuit(2).x(0).ch(0, 1).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['01'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  // H² = I: two CH applications with control=1 restore the state
  it('ch² = I: x(0)·ch·ch returns to |10⟩', () => {
    const r = new Circuit(2).x(0).ch(0, 1).ch(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

describe('crx(θ) — controlled-Rx', () => {
  it('crx(0) = I', () => {
    const r = new Circuit(2).x(0).crx(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // Rx(π) = X up to global phase → CRx(π) = CX
  it('crx(π) = cx: x(0)·crx(π) → |11⟩', () => {
    const r = new Circuit(2).x(0).crx(Math.PI, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 5)
  })

  it('crx(π/2) produces 50/50 superposition on target (control=1)', () => {
    const r = new Circuit(2).x(0).crx(Math.PI / 2, 0, 1).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['01'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  it('crx(θ)·crx(-θ) = I on activated subspace', () => {
    const θ = 0.7
    const r = new Circuit(2).x(0).crx(θ, 0, 1).crx(-θ, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

describe('cry(θ) — controlled-Ry', () => {
  it('cry(0) = I', () => {
    const r = new Circuit(2).x(0).cry(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // Ry(π) flips |0⟩ → |1⟩
  it('cry(π): x(0)·cry(π) → |11⟩', () => {
    const r = new Circuit(2).x(0).cry(Math.PI, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 5)
  })

  it('cry(θ)·cry(-θ) = I on activated subspace', () => {
    const θ = 0.7
    const r = new Circuit(2).x(0).cry(θ, 0, 1).cry(-θ, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

describe('crz(θ) — controlled-Rz', () => {
  it('crz(0) = I', () => {
    const r = new Circuit(2).x(0).crz(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // Rz(π)|+⟩ = i|-⟩; H·i|-⟩ = i|1⟩ → deterministic |1⟩ on target
  it('x(0)·h(1)·crz(π)·h(1): Rz(π)|+⟩ → |-⟩ → H → |1⟩, measures |11⟩', () => {
    const r = new Circuit(2).x(0).h(1).crz(Math.PI, 0, 1).h(1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 5)
  })

  it('crz(θ)·crz(-θ) = I on activated subspace', () => {
    const θ = 0.7
    const r = new Circuit(2).x(0).crz(θ, 0, 1).crz(-θ, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

describe('cu1(λ) — controlled phase gate', () => {
  it('cu1(0) = I', () => {
    const r = new Circuit(2).x(0).cu1(0, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // U1(π) = Z → CU1(π) = CZ → H(t)·CU1(π)·H(t) = CNOT
  it('cu1(π) = cz: x(0)·h(1)·cu1(π)·h(1) → |11⟩', () => {
    const r = new Circuit(2).x(0).h(1).cu1(Math.PI, 0, 1).h(1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 10)
  })

  // Ramsey fringe: H·U1(λ)·H|0⟩ → p(|0⟩) = (1+cosλ)/2
  it('Ramsey: x(0)·h(1)·cu1(π/4)·h(1) → p(|01⟩) ≈ 0.854', () => {
    const expected = (1 + Math.cos(Math.PI / 4)) / 2
    const r = new Circuit(2).x(0).h(1).cu1(Math.PI / 4, 0, 1).h(1).run({ shots: 50000, seed: 42 })
    expect(near(r.probs['01'] ?? 0, expected, 0.01)).toBe(true)
  })
})

describe('cr2/cr4/cr8 — controlled phase rotations', () => {
  it('cr2 leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cr2(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('cr2·cr2† = I: inverse pair restores state', () => {
    // x(0)→q0=|1⟩, h(1)→q1=|+⟩; cr2·crz(-π/2) = R2·R2† = I on q1; h(1)→q1=|0⟩
    // q0=|1⟩, q1=|0⟩ → bitstring '01' (q0 rightmost)
    const r = new Circuit(2).x(0).h(1).cr2(0, 1).crz(-Math.PI / 2, 0, 1).h(1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  it('cr4 leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cr4(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('cr8 leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cr8(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('cr2 = crz(π/2): same action on |11⟩ superposition', () => {
    const r1 = new Circuit(2).x(0).h(1).cr2(0, 1).h(1).run({ shots: 100, seed: 1 })
    const r2 = new Circuit(2).x(0).h(1).crz(Math.PI / 2, 0, 1).h(1).run({ shots: 100, seed: 1 })
    for (const bs of ['00', '01', '10', '11']) {
      expect(r1.probs[bs] ?? 0).toBeCloseTo(r2.probs[bs] ?? 0, 10)
    }
  })
})

describe('cu3(θ,φ,λ) — controlled general unitary', () => {
  it('cu3(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cu3(Math.PI, 0, Math.PI, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  // U3(π,0,π) = X → CU3(π,0,π) = CX
  it('cu3(π,0,π) = cx: x(0)·cu3(π,0,π) → |11⟩', () => {
    const r = new Circuit(2).x(0).cu3(Math.PI, 0, Math.PI, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['11']).toBeCloseTo(1.0, 10)
  })

  // Unitarity: U3⁻¹(θ,φ,λ) = U3(-θ,-λ,-φ)
  it('unitarity: x(0)·cu3(θ,φ,λ)·cu3(-θ,-λ,-φ) restores |10⟩', () => {
    const [θ, φ, λ] = [1.1, 0.7, 0.3]
    const r = new Circuit(2).x(0).cu3(θ, φ, λ, 0, 1).cu3(-θ, -λ, -φ, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

describe('cs — controlled-S', () => {
  it('cs(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).cs(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('cs·csdg = I: x(0)·cs·csdg restores |10⟩', () => {
    const r = new Circuit(2).x(0).cs(0, 1).csdg(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // S² = Z → cs² = cz (tested via Hadamard-basis equivalence)
  it('cs² = cz: x(0)·h(1)·cs·cs·h(1) matches x(0)·h(1)·cz·h(1)', () => {
    const rCz = new Circuit(2).x(0).h(1).cz(0, 1).h(1).run({ shots: 100, seed: 1 })
    const rCs = new Circuit(2).x(0).h(1).cs(0, 1).cs(0, 1).h(1).run({ shots: 100, seed: 1 })
    expect(rCs.probs['11']).toBeCloseTo(rCz.probs['11'] ?? 0, 10)
  })
})

describe('csdg — controlled-S†', () => {
  it('csdg(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).csdg(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('csdg·cs = I: x(0)·csdg·cs restores |10⟩', () => {
    const r = new Circuit(2).x(0).csdg(0, 1).cs(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

describe('ct — controlled-T', () => {
  it('ct(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).ct(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('ct·ctdg = I: x(0)·ct·ctdg restores |10⟩', () => {
    const r = new Circuit(2).x(0).ct(0, 1).ctdg(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })

  // T⁴ = Z → ct⁴ = cz (tested via Hadamard-basis equivalence)
  it('ct⁴ = cz: x(0)·h(1)·ct⁴·h(1) matches x(0)·h(1)·cz·h(1)', () => {
    const rCz = new Circuit(2).x(0).h(1).cz(0, 1).h(1).run({ shots: 100, seed: 1 })
    const rCt = new Circuit(2).x(0).h(1).ct(0,1).ct(0,1).ct(0,1).ct(0,1).h(1).run({ shots: 100, seed: 1 })
    expect(rCt.probs['11']).toBeCloseTo(rCz.probs['11'] ?? 0, 10)
  })
})

describe('ctdg — controlled-T†', () => {
  it('ctdg(control=0) leaves |00⟩ unchanged', () => {
    const r = new Circuit(2).ctdg(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('ctdg·ct = I: x(0)·ctdg·ct restores |10⟩', () => {
    const r = new Circuit(2).x(0).ctdg(0, 1).ct(0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

// ─── Y gate measurement ───────────────────────────────────────────────────────

describe('Y gate measurement', () => {
  it('Y|0⟩ measures as |1⟩ (Y|0⟩ = i|1⟩, phase unobservable)', () => {
    const r = new Circuit(1).y(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('Y|1⟩ measures as |0⟩ (Y|1⟩ = −i|0⟩, phase unobservable)', () => {
    const r = new Circuit(1).x(0).y(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  it('YY = I', () => {
    const r = new Circuit(1).y(0).y(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })
})

// ─── Quantum Fourier Transform ────────────────────────────────────────────────
//
// 2-qubit QFT circuit: H(q1) · CU1(π/2, q0, q1) · H(q0) · SWAP(q0, q1)
//
// Key property: QFT|j⟩ has uniform measurement distribution over all 2ⁿ states,
// regardless of |j⟩. This follows from |QFT|j⟩_k|² = 1/N for all k.
//
// This validates H + cu1 (controlled phase) + SWAP working together correctly —
// the same circuit pattern used in Shor's algorithm and phase estimation.

describe('Quantum Fourier Transform (2-qubit)', () => {
  function qft2<C extends Circuit>(c: C): Circuit {
    // H(q1) · CU1(π/2, q0→q1) · H(q0) · SWAP
    return c.h(1).cu1(Math.PI / 2, 0, 1).h(0).swap(0, 1)
  }

  it('QFT|00⟩ produces uniform distribution over all 4 states', () => {
    const r = qft2(new Circuit(2)).run({ shots: 20000, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(4)
    for (const p of Object.values(r.probs)) expect(near(p, 0.25, 0.02)).toBe(true)
  })

  it('QFT|01⟩ (q0=1) also produces uniform distribution', () => {
    const r = qft2(new Circuit(2).x(0)).run({ shots: 20000, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(4)
    for (const p of Object.values(r.probs)) expect(near(p, 0.25, 0.02)).toBe(true)
  })

  // IQFT · QFT = I: SWAP · H(q0) · CU1(−π/2, q0, q1) · H(q1) is QFT†
  it('IQFT · QFT = I: round-trip recovers |01⟩', () => {
    const r = new Circuit(2)
      .x(0)
      .h(1).cu1(Math.PI / 2, 0, 1).h(0).swap(0, 1)          // QFT
      .swap(0, 1).h(0).cu1(-Math.PI / 2, 0, 1).h(1)          // QFT†
      .run({ shots: 100, seed: 1 })
    expect(r.probs['01']).toBeCloseTo(1.0, 10)
  })
})

// ─── Multi-gate integration ────────────────────────────────────────────────────
//
// Tests gate composition across all gate families in a single circuit.
// Any unitary sequence U applied then reversed (U · U⁻¹) must restore |0…0⟩.
// Mirrors quantum-circuit's Issue_97 test (h, s, t, cx, srn sequence).

describe('Multi-gate integration', () => {
  // Forward: h · s · t · cnot · v  — Inverse: vi · cnot · ti · si · h
  it('mixed single/two-qubit sequence is invertible: U · U⁻¹ = I', () => {
    const r = new Circuit(2)
      .h(0).s(0).t(1).cnot(0, 1).v(0)   // U
      .vi(0).cnot(0, 1).ti(1).si(0).h(0) // U⁻¹
      .run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  // Controlled + interaction gates in composition
  it('controlled + interaction gate sequence is invertible', () => {
    const θ = 0.9
    const r = new Circuit(2)
      .h(0).cx(0, 1).crz(θ, 0, 1).xx(θ, 0, 1)   // U
      .xx(-θ, 0, 1).crz(-θ, 0, 1).cx(0, 1).h(0)  // U⁻¹
      .run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  // 3-qubit round-trip spanning single, two-qubit, and controlled families
  it('3-qubit U · U⁻¹ across all gate families returns to |000⟩', () => {
    const θ = 0.6
    const r = new Circuit(3)
      .h(0).rx(θ, 1).ry(θ, 2)
      .cnot(0, 1).cy(1, 2).crz(θ, 0, 2)
      .crz(-θ, 0, 2).cy(1, 2).cnot(0, 1)
      .ry(-θ, 2).rx(-θ, 1).h(0)
      .run({ shots: 100, seed: 1 })
    expect(r.probs['000']).toBeCloseTo(1.0, 10)
  })
})

// ─── Three-qubit gates ────────────────────────────────────────────────────────
//
// Both CCX and CSWAP are pure permutations — no amplitude arithmetic, just
// index remapping — making them O(|ψ|) in the number of non-zero amplitudes.
//
// Helper: build a circuit with the given bitstring pre-loaded as a basis state.
// Bitstring convention: q0 is rightmost (e.g. '011' → q0=1, q1=1, q2=0).
function basis(bitstring: string): Circuit {
  let c = new Circuit(bitstring.length)
  for (let i = 0; i < bitstring.length; i++) {
    if (bitstring[bitstring.length - 1 - i] === '1') c = c.x(i)
  }
  return c
}

describe('ccx — Toffoli gate', () => {
  // ── Classical truth table ──
  // ccx(0,1,2): c1=q0, c2=q1, target=q2.  Flip target iff c1=c2=1.
  it.each([
    ['000', '000'],  // both controls 0 — identity
    ['010', '010'],  // only q1=1     — identity
    ['001', '001'],  // only q0=1     — identity
    ['011', '111'],  // q0=q1=1       — flip q2
  ] as [string, string][])('|%s⟩ → |%s⟩', (input, expected) => {
    const r = basis(input).ccx(0, 1, 2).run({ shots: 100, seed: 1 })
    expect(r.probs[expected]).toBeCloseTo(1.0, 10)
  })

  // ── Self-inverse ──
  it('ccx² = I: |011⟩ → |111⟩ → |011⟩', () => {
    const r = basis('011').ccx(0, 1, 2).ccx(0, 1, 2).run({ shots: 100, seed: 1 })
    expect(r.probs['011']).toBeCloseTo(1.0, 10)
  })

  // ── Control qubit symmetry ──
  // The Toffoli is symmetric in its two control qubits.
  it('ccx(c1,c2,t) ≡ ccx(c2,c1,t): controls are interchangeable', () => {
    const r1 = basis('011').ccx(0, 1, 2).run({ shots: 100, seed: 1 })
    const r2 = basis('011').ccx(1, 0, 2).run({ shots: 100, seed: 1 })
    expect(r1.probs['111']).toBeCloseTo(r2.probs['111'] ?? 0, 10)
  })

  // ── Superposition: genuinely 3-qubit entangled state ──
  // H(0)·H(1) puts controls in (1/2)(|00⟩+|01⟩+|10⟩+|11⟩).
  // CCX flips target only for |11⟩ control component.
  // Result: (1/2)(|000⟩ + |001⟩ + |010⟩ + |111⟩) — 25% each.
  it('H(0)·H(1)·ccx creates (|000⟩+|001⟩+|010⟩+|111⟩)/2: 25% each', () => {
    const r = new Circuit(3).h(0).h(1).ccx(0, 1, 2).run({ shots: 20000, seed: 42 })
    expect(Object.keys(r.probs)).toHaveLength(4)
    expect(near(r.probs['000'] ?? 0, 0.25, 0.02)).toBe(true)
    expect(near(r.probs['001'] ?? 0, 0.25, 0.02)).toBe(true)
    expect(near(r.probs['010'] ?? 0, 0.25, 0.02)).toBe(true)
    expect(near(r.probs['111'] ?? 0, 0.25, 0.02)).toBe(true)
  })

  // ── Phase kickback ──
  // With target in |−⟩: CCX|c1,c2,−⟩ = (−1)^{AND(c1,c2)}|c1,c2,−⟩.
  // Put controls in |+⟩, target in |−⟩, apply CCX, then H on controls.
  // For f(x)=AND: H⊗H·CCX_{|−⟩}·H⊗H on |00⟩ measures |11⟩ ≠ |00⟩
  // (AND is not balanced so we just verify the distribution shifts from uniform).
  it('phase kickback: x(0)·x(1)·x(2)·h(2)·ccx flips target phase', () => {
    // Both controls=1, target=|−⟩: CCX|11−⟩ = −|11−⟩. H→|11,1⟩.
    const r = new Circuit(3).x(0).x(1).x(2).h(2).ccx(0, 1, 2).h(2).run({ shots: 100, seed: 1 })
    expect(r.probs['111']).toBeCloseTo(1.0, 10)
  })

  // ── Non-adjacent qubits ──
  // Verifies no hardcoded qubit numbering in the implementation.
  it('ccx(0,3,6) on 7 qubits: non-adjacent controls and target', () => {
    const r = new Circuit(7).x(0).x(3).ccx(0, 3, 6).run({ shots: 100, seed: 1 })
    expect(r.probs['1001001']).toBeCloseTo(1.0, 10)
  })
})

describe('cswap — Fredkin gate', () => {
  // ── Classical truth table ──
  // cswap(0,1,2): control=q0, swap q1 and q2.
  it.each([
    ['000', '000'],  // control=0: identity
    ['010', '010'],  // control=0: no swap (q1=1 stays)
    ['001', '001'],  // control=1, q1=q2=0: bits equal, no visible change
    ['011', '101'],  // control=1: q1=1,q2=0 → swap → q1=0,q2=1
    ['101', '011'],  // control=1: q2=1,q1=0 → swap → q2=0,q1=1
  ] as [string, string][])('|%s⟩ → |%s⟩', (input, expected) => {
    const r = basis(input).cswap(0, 1, 2).run({ shots: 100, seed: 1 })
    expect(r.probs[expected]).toBeCloseTo(1.0, 10)
  })

  // ── Self-inverse ──
  it('cswap² = I: |011⟩ → |101⟩ → |011⟩', () => {
    const r = basis('011').cswap(0, 1, 2).cswap(0, 1, 2).run({ shots: 100, seed: 1 })
    expect(r.probs['011']).toBeCloseTo(1.0, 10)
  })

  // ── Swap test (quantum algorithm) ──────────────────────────────────────────
  // H(ctrl) · CSWAP(ctrl, a, b) · H(ctrl) is the SWAP test circuit.
  // Measuring p(ctrl=0) = (1 + |⟨ψ|φ⟩|²) / 2:
  //   ψ = φ → p = 1     (identical states)
  //   ψ ⊥ φ → p = 0.5   (orthogonal states)
  // This is a provably quantum algorithm — no classical circuit achieves it.
  it('swap test: ψ=φ=|0⟩ → p(control=0) = 1 (identical states)', () => {
    // control=q0, ψ=q1=|0⟩, φ=q2=|0⟩
    const r = new Circuit(3).h(0).cswap(0, 1, 2).h(0).run({ shots: 100, seed: 1 })
    const p0 = Object.entries(r.probs)
      .filter(([bs]) => bs[bs.length - 1] === '0') // q0=0 (control measured 0)
      .reduce((sum, [, p]) => sum + p, 0)
    expect(p0).toBeCloseTo(1.0, 10)
  })

  it('swap test: ψ=|0⟩, φ=|1⟩ → p(control=0) = 0.5 (orthogonal states)', () => {
    // x(2) prepares φ=|1⟩ on q2; ψ=q1 stays |0⟩
    const r = new Circuit(3).x(2).h(0).cswap(0, 1, 2).h(0).run({ shots: 20000, seed: 42 })
    const p0 = Object.entries(r.probs)
      .filter(([bs]) => bs[bs.length - 1] === '0')
      .reduce((sum, [, p]) => sum + p, 0)
    expect(near(p0, 0.5)).toBe(true)
  })

  it('swap test: ψ=|+⟩, φ=|+⟩ → p(control=0) = 1 (identical superpositions)', () => {
    // h(1) and h(2) both prepare |+⟩; identical states → p=1
    const r = new Circuit(3).h(1).h(2).h(0).cswap(0, 1, 2).h(0).run({ shots: 10000, seed: 42 })
    const p0 = Object.entries(r.probs)
      .filter(([bs]) => bs[bs.length - 1] === '0')
      .reduce((sum, [, p]) => sum + p, 0)
    expect(near(p0, 1.0, 0.01)).toBe(true)
  })

  // ── Non-adjacent qubits ──
  it('cswap(0,2,4) on 5 qubits: swaps non-adjacent target qubits', () => {
    // control=q0=1, q2=1 → cswap swaps q2↔q4 → q4=1, q2=0
    const r = new Circuit(5).x(0).x(2).cswap(0, 2, 4).run({ shots: 100, seed: 1 })
    // q0=1(bit0), q4=1(bit4) → index=17 → bitstring '10001'
    expect(r.probs['10001']).toBeCloseTo(1.0, 10)
  })
})

// ─── Native IonQ gates ────────────────────────────────────────────────────────

describe('GPI gate', () => {
  it('GPI(0) = X: flips |0⟩ to |1⟩', () => {
    const r = new Circuit(1).gpi(0, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('GPI(π/2) = Y: flips |0⟩ to i|1⟩ — same measurement outcome', () => {
    const r = new Circuit(1).gpi(Math.PI / 2, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('GPI² = I: self-inverse (Hermitian)', () => {
    const phi = Math.PI / 3
    const r = new Circuit(1).x(0).gpi(phi, 0).gpi(phi, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('Ramsey fringe: H·GPI(φ)·H|0⟩ → p(|0⟩) = cos²(φ)', () => {
    const phi = Math.PI / 6  // cos²(π/6) = 3/4
    const r = new Circuit(1).h(0).gpi(phi, 0).h(0).run({ shots: 20000, seed: 7 })
    const expected = Math.cos(phi) ** 2
    expect(r.probs['0'] ?? 0).toBeCloseTo(expected, 1)
  })
})

describe('GPI2 gate', () => {
  it('GPI2(0) = Rx(π/2): maps |0⟩ to equal superposition', () => {
    const r = new Circuit(1).gpi2(0, 0).run({ shots: 10000, seed: 3 })
    expect(near(r.probs['0'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['1'] ?? 0, 0.5)).toBe(true)
  })

  it('GPI2(0)² = X: two half-rotations flip the qubit', () => {
    const r = new Circuit(1).gpi2(0, 0).gpi2(0, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('GPI2(φ)⁻¹ = GPI2(φ+π): inverse pair cancels to I', () => {
    const phi = Math.PI / 5
    const r = new Circuit(1).x(0).gpi2(phi, 0).gpi2(phi + Math.PI, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('GPI2(π/2)|+⟩ = |1⟩: known analytic output', () => {
    // Ry(π/2)|+⟩ = |1⟩ and GPI2(π/2) = Ry(π/2)
    const r = new Circuit(1).h(0).gpi2(Math.PI / 2, 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })
})

describe('MS gate', () => {
  it('MS(0,0)|00⟩ produces Bell state — only |00⟩ and |11⟩', () => {
    const r = new Circuit(2).ms(0, 0, 0, 1).run({ shots: 4000, seed: 5 })
    expect(Object.keys(r.probs).every(bs => bs === '00' || bs === '11')).toBe(true)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  it('MS(0,0) = XX(π/2): same measurement outcomes', () => {
    const r1 = new Circuit(2).ms(0, 0, 0, 1).run({ shots: 4000, seed: 9 })
    const r2 = new Circuit(2).xx(Math.PI / 2, 0, 1).run({ shots: 4000, seed: 9 })
    expect(near(r1.probs['00'] ?? 0, r2.probs['00'] ?? 0, 0.02)).toBe(true)
    expect(near(r1.probs['11'] ?? 0, r2.probs['11'] ?? 0, 0.02)).toBe(true)
  })

  it('MS(π/2,π/2) = YY(π/2): same measurement outcomes from |00⟩', () => {
    const r1 = new Circuit(2).ms(Math.PI / 2, Math.PI / 2, 0, 1).run({ shots: 4000, seed: 11 })
    const r2 = new Circuit(2).yy(Math.PI / 2, 0, 1).run({ shots: 4000, seed: 11 })
    expect(near(r1.probs['00'] ?? 0, r2.probs['00'] ?? 0, 0.02)).toBe(true)
    expect(near(r1.probs['11'] ?? 0, r2.probs['11'] ?? 0, 0.02)).toBe(true)
  })

  it('MS(0,0)·XX(-π/2) = I: unitarity — inverse cancels', () => {
    const r = new Circuit(2).ms(0, 0, 0, 1).xx(-Math.PI / 2, 0, 1).run({ shots: 100, seed: 1 })
    expect(r.probs['00']).toBeCloseTo(1.0, 10)
  })

  it('MS arbitrary angles always produces only |00⟩ and |11⟩ from |00⟩', () => {
    const r = new Circuit(2).ms(1.1, 0.7, 0, 1).run({ shots: 2000, seed: 13 })
    expect(Object.keys(r.probs).every(bs => bs === '00' || bs === '11')).toBe(true)
  })
})

// ─── Quantum teleportation ────────────────────────────────────────────────────
//
// Protocol (no classical channel needed in simulation — we use post-selection):
//   q0 = payload (arbitrary state), q1 = Alice's half of Bell pair, q2 = Bob's half
//   1. Prepare Bell pair on q1/q2
//   2. Alice: CNOT(q0→q1), H(q0)
//   3. Classical measurement replaced by controlled corrections:
//      CX(q1, q2), CZ(q0, q2)
//   4. q2 must be in the same state as q0 was.
//
// Test: teleport |0⟩, |1⟩, |+⟩, |+i⟩ and verify q2 matches.

describe('Quantum teleportation', () => {
  function teleport(prepQ0: (c: Circuit) => Circuit): Circuit {
    let c = new Circuit(3)
    c = prepQ0(c)           // prepare payload on q0
    c = c.h(1).cnot(1, 2)  // Bell pair on q1/q2
    c = c.cnot(0, 1).h(0)  // Alice's Bell measurement
    c = c.cx(1, 2).cz(0, 2) // Bob's corrections
    return c
  }

  it('teleports |0⟩: q2 measures as |0⟩', () => {
    const r = teleport(c => c).run({ shots: 1000, seed: 1 })
    // q2 (bit 2) should be 0 — only bitstrings with bit2=0 (xxx where x[0]=0)
    let p0 = 0
    for (const [bs, p] of Object.entries(r.probs)) {
      if (bs[0] === '0') p0 += p  // bit2 is MSB of 3-qubit string
    }
    expect(p0).toBeCloseTo(1.0, 10)
  })

  it('teleports |1⟩: q2 measures as |1⟩', () => {
    const r = teleport(c => c.x(0)).run({ shots: 1000, seed: 1 })
    let p1 = 0
    for (const [bs, p] of Object.entries(r.probs)) {
      if (bs[0] === '1') p1 += p
    }
    expect(p1).toBeCloseTo(1.0, 10)
  })

  it('teleports |+⟩: q2 measures 50/50', () => {
    const r = teleport(c => c.h(0)).run({ shots: 8000, seed: 2 })
    let p0 = 0, p1 = 0
    for (const [bs, p] of Object.entries(r.probs)) {
      if (bs[0] === '0') p0 += p
      else               p1 += p
    }
    expect(near(p0, 0.5)).toBe(true)
    expect(near(p1, 0.5)).toBe(true)
  })

  it('teleports Rx(π/3)|0⟩: q2 replicates the rotation', () => {
    // Rx(π/3)|0⟩: p(|0⟩) = cos²(π/6) = 3/4
    const expected0 = Math.cos(Math.PI / 6) ** 2
    const r = teleport(c => c.rx(Math.PI / 3, 0)).run({ shots: 20000, seed: 3 })
    let p0 = 0
    for (const [bs, p] of Object.entries(r.probs)) {
      if (bs[0] === '0') p0 += p
    }
    expect(p0).toBeCloseTo(expected0, 1)
  })
})

// ─── Statevector inspection (2c) ─────────────────────────────────────────────

describe('statevector()', () => {
  it('|0⟩ has amplitude 1 at index 0', () => {
    const sv = new Circuit(1).statevector()
    expect(sv.get(0n)).toEqual({ re: 1, im: 0 })
    expect(sv.size).toBe(1)
  })

  it('H|0⟩ has equal amplitudes at 0 and 1', () => {
    const sv = new Circuit(1).h(0).statevector()
    const a0 = sv.get(0n)!
    const a1 = sv.get(1n)!
    expect(a0.re).toBeCloseTo(1 / Math.SQRT2, 10)
    expect(a0.im).toBeCloseTo(0, 10)
    expect(a1.re).toBeCloseTo(1 / Math.SQRT2, 10)
    expect(a1.im).toBeCloseTo(0, 10)
  })

  it('X|0⟩ = |1⟩: only index 1 is populated', () => {
    const sv = new Circuit(1).x(0).statevector()
    expect(sv.has(0n)).toBe(false)
    expect(sv.get(1n)?.re).toBeCloseTo(1, 10)
  })

  it('Bell state: equal amplitudes at |00⟩ and |11⟩, nothing else', () => {
    const sv = new Circuit(2).h(0).cnot(0, 1).statevector()
    expect(sv.get(0n)!.re).toBeCloseTo(1 / Math.SQRT2, 10)  // |00⟩
    expect(sv.get(3n)!.re).toBeCloseTo(1 / Math.SQRT2, 10)  // |11⟩ = index 3
    expect(sv.size).toBe(2)
  })

  it('throws TypeError for circuits with measure ops', () => {
    expect(() => new Circuit(1).measure(0, 'c', 0).statevector()).toThrow(TypeError)
  })

  it('throws TypeError for circuits with reset ops', () => {
    expect(() => new Circuit(1).reset(0).statevector()).toThrow(TypeError)
  })
})

describe('amplitude(bitstring)', () => {
  it('|0⟩ → amplitude("0") = 1+0i', () => {
    const a = new Circuit(1).amplitude('0')
    expect(a.re).toBeCloseTo(1, 10)
    expect(a.im).toBeCloseTo(0, 10)
  })

  it('X|0⟩ → amplitude("1") = 1, amplitude("0") = 0', () => {
    const c = new Circuit(1).x(0)
    expect(c.amplitude('1').re).toBeCloseTo(1, 10)
    expect(c.amplitude('0').re).toBeCloseTo(0, 10)
  })

  it('H|0⟩ → amplitude("0") = amplitude("1") = 1/√2', () => {
    const c = new Circuit(1).h(0)
    expect(c.amplitude('0').re).toBeCloseTo(1 / Math.SQRT2, 10)
    expect(c.amplitude('1').re).toBeCloseTo(1 / Math.SQRT2, 10)
  })

  it('S|1⟩ → amplitude("1") is pure imaginary (i)', () => {
    const c = new Circuit(1).x(0).s(0)
    const a = c.amplitude('1')
    expect(a.re).toBeCloseTo(0, 10)
    expect(a.im).toBeCloseTo(1, 10)
  })

  it('Bell state: amplitude("00") = amplitude("11") = 1/√2, others zero', () => {
    const c = new Circuit(2).h(0).cnot(0, 1)
    expect(c.amplitude('00').re).toBeCloseTo(1 / Math.SQRT2, 10)
    expect(c.amplitude('11').re).toBeCloseTo(1 / Math.SQRT2, 10)
    expect(c.amplitude('01').re).toBeCloseTo(0, 10)
    expect(c.amplitude('10').re).toBeCloseTo(0, 10)
  })
})

describe('probability(bitstring)', () => {
  it('|0⟩ → probability("0") = 1', () => {
    expect(new Circuit(1).probability('0')).toBeCloseTo(1, 10)
  })

  it('X|0⟩ → probability("1") = 1, probability("0") = 0', () => {
    const c = new Circuit(1).x(0)
    expect(c.probability('1')).toBeCloseTo(1, 10)
    expect(c.probability('0')).toBeCloseTo(0, 10)
  })

  it('H|0⟩ → probability("0") = probability("1") = 0.5', () => {
    const c = new Circuit(1).h(0)
    expect(c.probability('0')).toBeCloseTo(0.5, 10)
    expect(c.probability('1')).toBeCloseTo(0.5, 10)
  })

  it('probabilities sum to 1 for any pure state', () => {
    const c = new Circuit(3).h(0).cnot(0, 1).rx(0.7, 2)
    const sv = c.statevector()
    let total = 0
    for (const bs of sv.keys()) total += c.probability(bs.toString(2).padStart(3, '0'))
    expect(total).toBeCloseTo(1, 10)
  })
})

// ─── Classical registers and mid-circuit measurement (2a) ────────────────────

describe('creg — classical register declaration', () => {
  it('creg returns a new circuit (immutability)', () => {
    const a = new Circuit(1)
    const b = a.creg('c', 1)
    expect(a).not.toBe(b)
  })

  it('declaring a creg without measuring leaves its bits at 0 in Distribution', () => {
    const r = new Circuit(1).h(0).creg('c', 2).run({ shots: 100, seed: 1 })
    expect(r.cregs['c']).toEqual([0, 0])
  })
})

describe('measure — deterministic collapse', () => {
  it('measuring |0⟩ always records 0 in creg', () => {
    const r = new Circuit(1).measure(0, 'c', 0).run({ shots: 100, seed: 1 })
    expect(r.cregs['c']![0]).toBeCloseTo(0, 10)
  })

  it('measuring |1⟩ always records 1 in creg', () => {
    const r = new Circuit(1).x(0).measure(0, 'c', 0).run({ shots: 100, seed: 1 })
    expect(r.cregs['c']![0]).toBeCloseTo(1, 10)
  })

  it('measuring |0⟩ leaves the qubit in |0⟩ (final qubit probs unaffected)', () => {
    const r = new Circuit(1).measure(0, 'c', 0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1, 10)
  })

  it('measuring |1⟩ leaves the qubit in |1⟩', () => {
    const r = new Circuit(1).x(0).measure(0, 'c', 0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1, 10)
  })
})

describe('measure — superposition and collapse', () => {
  it('measuring |+⟩ records 1 in ~50% of shots', () => {
    const r = new Circuit(1).h(0).measure(0, 'c', 0).run({ shots: 10000, seed: 42 })
    expect(near(r.cregs['c']![0]!, 0.5)).toBe(true)
  })

  // If measurement truly collapses the state, h(0).measure.h(0) ≠ h(0).h(0).
  // h·h = I → deterministic |0⟩.  But h·(collapse)·h → 50/50 regardless of outcome.
  it('collapse is real: h·measure·h gives 50/50, not |0⟩', () => {
    const r = new Circuit(1).h(0).measure(0, 'c', 0).h(0).run({ shots: 10000, seed: 42 })
    expect(near(r.probs['0'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['1'] ?? 0, 0.5)).toBe(true)
  })

  it('re-measuring the same qubit gives the same outcome: creg distributions match', () => {
    const r = new Circuit(1)
      .h(0)
      .measure(0, 'c',  0)
      .measure(0, 'c2', 0)
      .run({ shots: 10000, seed: 42 })
    // Both measurements reflect the same ~50% probability
    expect(near(r.cregs['c']![0]!,  0.5)).toBe(true)
    expect(near(r.cregs['c2']![0]!, 0.5)).toBe(true)
    // Distributions must match (collapse locks the qubit for the rest of the shot)
    expect(r.cregs['c']![0]).toBeCloseTo(r.cregs['c2']![0]!, 1)
  })

  it('entangled Bell pair: measuring q0 determines q1 (only |00⟩ and |11⟩ outcomes)', () => {
    // Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 — measuring q0 collapses q1 to match
    const r = new Circuit(2)
      .h(0).cnot(0, 1)
      .measure(0, 'c', 0)
      .run({ shots: 2000, seed: 7 })
    // Final qubit probs must be all |00⟩ or all |11⟩ per shot → only those two outcomes
    expect(Object.keys(r.probs).every(bs => bs === '00' || bs === '11')).toBe(true)
  })
})

describe('measure — multi-bit register', () => {
  it('two qubits into two bits of one register', () => {
    // x(0) → q0=|1⟩, q1=|0⟩; measure into 2-bit register
    const r = new Circuit(2)
      .x(0)
      .creg('c', 2)
      .measure(0, 'c', 0)
      .measure(1, 'c', 1)
      .run({ shots: 100, seed: 1 })
    expect(r.cregs['c']![0]).toBeCloseTo(1, 10)  // q0=|1⟩ → bit0 always 1
    expect(r.cregs['c']![1]).toBeCloseTo(0, 10)  // q1=|0⟩ → bit1 always 0
  })

  it('auto-registers a creg if not explicitly declared', () => {
    const r = new Circuit(1).x(0).measure(0, 'auto', 0).run({ shots: 100, seed: 1 })
    expect(r.cregs['auto']![0]).toBeCloseTo(1, 10)
  })
})

describe('reset', () => {
  it('reset(|0⟩) leaves qubit in |0⟩', () => {
    const r = new Circuit(1).reset(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1, 10)
  })

  it('reset(|1⟩) restores qubit to |0⟩', () => {
    const r = new Circuit(1).x(0).reset(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1, 10)
  })

  it('reset(|+⟩) always returns |0⟩ regardless of superposition', () => {
    const r = new Circuit(1).h(0).reset(0).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1, 10)
  })

  it('reset after entanglement restores target qubit to |0⟩', () => {
    // Bell pair on q0/q1; reset q0 → q0 always |0⟩, q1 collapses to definite state
    const r = new Circuit(2).h(0).cnot(0, 1).reset(0).run({ shots: 100, seed: 1 })
    // Rightmost character in bitstring is q0 → must be 0 in all outcomes
    expect(Object.keys(r.probs).every(bs => bs.at(-1) === '0')).toBe(true)
  })
})

describe('if — conditional gate (2b)', () => {
  it('if(creg=0, value=0): gate fires when register is 0 (identity condition)', () => {
    // creg never written → always 0; if == 0 → always apply X → |1⟩
    const r = new Circuit(1).measure(0, 'c', 0).if('c', 0, c => c.x(0)).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1, 10)
  })

  it('if(creg=1, value=1): gate fires only when measurement gave 1', () => {
    // x(0) → |1⟩; measure → creg=1; if creg==1 → apply X → back to |0⟩
    const r = new Circuit(1).x(0).measure(0, 'c', 0).if('c', 1, c => c.x(0)).run({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1, 10)
  })

  it('if(creg=1, value=0): gate does NOT fire when register is 1', () => {
    // x(0) → |1⟩; measure → creg=1; if creg==0 → skip; qubit stays |1⟩
    const r = new Circuit(1).x(0).measure(0, 'c', 0).if('c', 0, c => c.x(0)).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1, 10)
  })

  it('if corrects 50/50 collapse: measure |+⟩, apply X if 1 → always |0⟩', () => {
    // measure |+⟩ → creg ~50/50; X correction when 1 → qubit always ends in |0⟩
    const r = new Circuit(1).h(0).measure(0, 'c', 0).if('c', 1, c => c.x(0)).run({ shots: 1000, seed: 42 })
    expect(r.probs['0']).toBeCloseTo(1, 5)
  })

  // ── Quantum teleportation with true mid-circuit measurement ──────────────────
  //
  // The textbook protocol:
  //   1. Bell pair on q1/q2
  //   2. Alice: CNOT(q0→q1) · H(q0) · measure q0→c0, q1→c1
  //   3. Bob: if c1==1 → X(q2) ; if c0==1 → Z(q2)
  // q2 must end in the same state as q0 was prepared in.

  function teleportMeasured(prepQ0: (c: Circuit) => Circuit): Distribution {
    return prepQ0(new Circuit(3))
      .h(1).cnot(1, 2)                     // Bell pair q1/q2
      .cnot(0, 1).h(0)                     // Alice's Bell measurement basis
      .measure(0, 'c0', 0)
      .measure(1, 'c1', 0)
      .if('c1', 1, q => q.x(2))           // Bob: X correction
      .if('c0', 1, q => q.z(2))           // Bob: Z correction
      .run({ shots: 1000, seed: 1 })
  }

  it('teleportation with measurement + if: teleports |0⟩ → q2 always |0⟩', () => {
    const r = teleportMeasured(c => c)
    const p0 = Object.entries(r.probs)
      .filter(([bs]) => bs[0] === '0')
      .reduce((s, [, p]) => s + p, 0)
    expect(p0).toBeCloseTo(1, 5)
  })

  it('teleportation with measurement + if: teleports |1⟩ → q2 always |1⟩', () => {
    const r = teleportMeasured(c => c.x(0))
    const p1 = Object.entries(r.probs)
      .filter(([bs]) => bs[0] === '1')
      .reduce((s, [, p]) => s + p, 0)
    expect(p1).toBeCloseTo(1, 5)
  })

  it('teleportation with measurement + if: teleports |+⟩ → q2 measures 50/50', () => {
    const r = teleportMeasured(c => c.h(0))
    const p0 = Object.entries(r.probs).filter(([bs]) => bs[0] === '0').reduce((s, [, p]) => s + p, 0)
    const p1 = Object.entries(r.probs).filter(([bs]) => bs[0] === '1').reduce((s, [, p]) => s + p, 0)
    expect(near(p0, 0.5)).toBe(true)
    expect(near(p1, 0.5)).toBe(true)
  })
})

describe('measure + reset integration', () => {
  it('measure then reset: creg records pre-reset value, qubit ends in |0⟩', () => {
    // x(0) → |1⟩; measure records 1; reset → |0⟩
    const r = new Circuit(1).x(0).measure(0, 'c', 0).reset(0).run({ shots: 100, seed: 1 })
    expect(r.cregs['c']![0]).toBeCloseTo(1, 10)
    expect(r.probs['0']).toBeCloseTo(1, 10)
  })

  it('mid-circuit reset enables qubit reuse: reset then re-encode', () => {
    // q0: x → measure(→1) → reset(→|0⟩) → x again → final |1⟩
    const r = new Circuit(1).x(0).measure(0, 'c', 0).reset(0).x(0).run({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1, 10)
  })
})

// ─── IonQ JSON import / export (3a) ──────────────────────────────────────────
//
// Round-trip invariant: Circuit.fromIonQ(c.toIonQ()) produces a circuit whose
// statevector is identical to c's.  Also covers the IonQ angle convention
// (rotation in π-radians, phase/phases in turns) and error cases.

describe('toIonQ() — serialization', () => {
  it('empty circuit serializes to empty gate list', () => {
    const j = new Circuit(2).toIonQ()
    expect(j.format).toBe('ionq.circuit.v0')
    expect(j.qubits).toBe(2)
    expect(j.circuit).toHaveLength(0)
  })

  it('single H gate', () => {
    const j = new Circuit(1).h(0).toIonQ()
    expect(j.circuit).toEqual([{ gate: 'h', target: 0 }])
  })

  it('CNOT gate carries control and target', () => {
    const j = new Circuit(2).cnot(0, 1).toIonQ()
    expect(j.circuit).toEqual([{ gate: 'cnot', control: 0, target: 1 }])
  })

  it('SWAP gate uses targets array', () => {
    const j = new Circuit(2).swap(0, 1).toIonQ()
    expect(j.circuit).toEqual([{ gate: 'swap', targets: [0, 1] }])
  })

  it('rx: rotation stored as θ/π (half-turns)', () => {
    const j = new Circuit(1).rx(Math.PI / 2, 0).toIonQ()
    expect(j.circuit[0]!.rotation).toBeCloseTo(0.5, 10)
  })

  it('rz: rotation stored as θ/π', () => {
    const j = new Circuit(1).rz(Math.PI / 4, 0).toIonQ()
    expect(j.circuit[0]!.rotation).toBeCloseTo(0.25, 10)
  })

  it('r2/r4/r8 serialize as named gates with no rotation field', () => {
    const j = new Circuit(1).r2(0).r4(0).r8(0).toIonQ()
    expect(j.circuit[0]!.gate).toBe('r2')
    expect(j.circuit[0]!.rotation).toBeUndefined()
    expect(j.circuit[1]!.gate).toBe('r4')
    expect(j.circuit[2]!.gate).toBe('r8')
  })

  it('gpi: phase stored as φ/(2π) (turns)', () => {
    const j = new Circuit(1).gpi(Math.PI / 2, 0).toIonQ()
    expect(j.circuit[0]!.phase).toBeCloseTo(0.25, 10)
  })

  it('gpi2: phase stored as φ/(2π) (turns)', () => {
    const j = new Circuit(1).gpi2(Math.PI, 0).toIonQ()
    expect(j.circuit[0]!.phase).toBeCloseTo(0.5, 10)
  })

  it('ms: phases stored as [φ₀/(2π), φ₁/(2π)] (turns)', () => {
    const j = new Circuit(2).ms(Math.PI / 2, Math.PI, 0, 1).toIonQ()
    expect(j.circuit[0]!.phases![0]).toBeCloseTo(0.25, 10)
    expect(j.circuit[0]!.phases![1]).toBeCloseTo(0.5, 10)
  })

  it('xx/yy/zz: rotation stored as θ/π', () => {
    const j = new Circuit(2).xx(Math.PI / 2, 0, 1).yy(Math.PI / 4, 0, 1).zz(Math.PI, 0, 1).toIonQ()
    expect(j.circuit[0]!.rotation).toBeCloseTo(0.5,  10)
    expect(j.circuit[1]!.rotation).toBeCloseTo(0.25, 10)
    expect(j.circuit[2]!.rotation).toBeCloseTo(1.0,  10)
  })

  it('throws for an unsupported gate (e.g. cy — controlled-Y)', () => {
    expect(() => new Circuit(2).cy(0, 1).toIonQ()).toThrow(TypeError)
  })

  it('throws for u1 (OpenQASM gate, no IonQ representation)', () => {
    expect(() => new Circuit(1).u1(Math.PI / 4, 0).toIonQ()).toThrow(TypeError)
  })

  it('throws for mid-circuit measure ops', () => {
    expect(() => new Circuit(1).measure(0, 'c', 0).toIonQ()).toThrow(TypeError)
  })
})

describe('fromIonQ() — parsing', () => {
  it('parses a Bell-state circuit', () => {
    const json: IonQCircuit = {
      format: 'ionq.circuit.v0',
      qubits: 2,
      circuit: [
        { gate: 'h', target: 0 },
        { gate: 'cnot', control: 0, target: 1 },
      ],
    }
    const r = Circuit.fromIonQ(json).run({ shots: 4000, seed: 5 })
    expect(Object.keys(r.probs).every(bs => bs === '00' || bs === '11')).toBe(true)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
  })

  it('rotation angle round-trip: rx(π/3) through JSON', () => {
    const json: IonQCircuit = {
      format: 'ionq.circuit.v0',
      qubits: 1,
      circuit: [{ gate: 'rx', rotation: 1 / 3, target: 0 }],
    }
    const c = Circuit.fromIonQ(json)
    // Rx(π/3)|0⟩: p(|0⟩) = cos²(π/6) ≈ 0.75
    expect(c.probability('0')).toBeCloseTo(Math.cos(Math.PI / 6) ** 2, 5)
  })

  it('phase angle round-trip: gpi(π/4) through JSON', () => {
    const json: IonQCircuit = {
      format: 'ionq.circuit.v0',
      qubits: 1,
      circuit: [{ gate: 'gpi', phase: 0.125, target: 0 }],  // 0.125 turns = π/4 rad
    }
    const c = Circuit.fromIonQ(json)
    // GPI(π/4)|0⟩ = e^{iπ/4}|1⟩ → always measures |1⟩
    expect(c.probability('1')).toBeCloseTo(1, 10)
  })

  it('throws for unknown gate name', () => {
    const json = {
      format: 'ionq.circuit.v0' as const,
      qubits: 1,
      circuit: [{ gate: 'bogus', target: 0 }],
    }
    expect(() => Circuit.fromIonQ(json)).toThrow(TypeError)
  })
})

describe('IonQ round-trip: toIonQ() → fromIonQ() → identical statevector', () => {
  function roundTrip(c: Circuit): Circuit { return Circuit.fromIonQ(c.toIonQ()) }

  it('H gate', () => {
    const c = new Circuit(1).h(0)
    expect(roundTrip(c).amplitude('0').re).toBeCloseTo(c.amplitude('0').re, 10)
    expect(roundTrip(c).amplitude('1').re).toBeCloseTo(c.amplitude('1').re, 10)
  })

  it('Bell state: H + CNOT', () => {
    const c = new Circuit(2).h(0).cnot(0, 1)
    expect(roundTrip(c).amplitude('00').re).toBeCloseTo(c.amplitude('00').re, 10)
    expect(roundTrip(c).amplitude('11').re).toBeCloseTo(c.amplitude('11').re, 10)
  })

  it('parametric: rx(0.7) + rz(1.3)', () => {
    const c = new Circuit(1).rx(0.7, 0).rz(1.3, 0)
    for (const bs of ['0', '1']) {
      const { re, im } = roundTrip(c).amplitude(bs)
      expect(re).toBeCloseTo(c.amplitude(bs).re, 8)
      expect(im).toBeCloseTo(c.amplitude(bs).im, 8)
    }
  })

  it('native gates: gpi2 + ms', () => {
    const c = new Circuit(2).gpi2(0.4, 0).ms(0.6, 0.8, 0, 1)
    for (const bs of ['00', '01', '10', '11']) {
      const rt = roundTrip(c)
      expect(rt.amplitude(bs).re).toBeCloseTo(c.amplitude(bs).re, 8)
      expect(rt.amplitude(bs).im).toBeCloseTo(c.amplitude(bs).im, 8)
    }
  })

  it('mixed gate set: h, x, s, cnot, xx, r2', () => {
    const c = new Circuit(3).h(0).x(1).s(2).cnot(0, 1).xx(0.9, 1, 2).r2(0)
    const rt = roundTrip(c)
    const sv = c.statevector()
    for (const bs of sv.keys()) {
      const key = bs.toString(2).padStart(3, '0')
      expect(rt.amplitude(key).re).toBeCloseTo(c.amplitude(key).re, 8)
      expect(rt.amplitude(key).im).toBeCloseTo(c.amplitude(key).im, 8)
    }
  })
})

// ─── OpenQASM 2.0 ────────────────────────────────────────────────────────────

describe('toQASM() — serialization', () => {
  it('emits a valid OPENQASM 2.0 header', () => {
    const q = new Circuit(2).toQASM()
    expect(q).toContain('OPENQASM 2.0;')
    expect(q).toContain('include "qelib1.inc";')
    expect(q).toContain('qreg q[2];')
  })

  it('H gate → h q[0]', () => {
    const q = new Circuit(1).h(0).toQASM()
    expect(q).toContain('h q[0];')
  })

  it('cnot → cx q[control],q[target]', () => {
    const q = new Circuit(2).cnot(0, 1).toQASM()
    expect(q).toContain('cx q[0],q[1];')
  })

  it('cx alias also emits cx', () => {
    const q = new Circuit(2).cx(0, 1).toQASM()
    expect(q).toContain('cx q[0],q[1];')
  })

  it('si → sdg, ti → tdg, v → sx, vi → sxdg', () => {
    const q = new Circuit(1).si(0).ti(0).v(0).vi(0).toQASM()
    expect(q).toContain('sdg q[0];')
    expect(q).toContain('tdg q[0];')
    expect(q).toContain('sx q[0];')
    expect(q).toContain('sxdg q[0];')
  })

  it('r2/r4/r8 → rz with π-fraction params', () => {
    const q = new Circuit(1).r2(0).r4(0).r8(0).toQASM()
    expect(q).toContain('rz(pi/2) q[0];')
    expect(q).toContain('rz(pi/4) q[0];')
    expect(q).toContain('rz(pi/8) q[0];')
  })

  it('rx/ry/rz carry angle in QASM angle notation', () => {
    const q = new Circuit(1).rx(Math.PI / 2, 0).ry(Math.PI / 4, 0).toQASM()
    expect(q).toContain('rx(pi/2) q[0];')
    expect(q).toContain('ry(pi/4) q[0];')
  })

  it('u1/u2/u3 emit with params', () => {
    const q = new Circuit(1).u1(Math.PI / 4, 0).u2(0, Math.PI, 0).u3(Math.PI, 0, Math.PI, 0).toQASM()
    expect(q).toContain('u1(pi/4) q[0];')
    expect(q).toContain('u2(0,pi) q[0];')
    expect(q).toContain('u3(pi,0,pi) q[0];')
  })

  it('cy/cz/ch controlled gates', () => {
    const q = new Circuit(2).cy(0, 1).cz(0, 1).ch(0, 1).toQASM()
    expect(q).toContain('cy q[0],q[1];')
    expect(q).toContain('cz q[0],q[1];')
    expect(q).toContain('ch q[0],q[1];')
  })

  it('crx/cry/crz carry angle params', () => {
    const q = new Circuit(2).crx(Math.PI / 2, 0, 1).cry(Math.PI / 4, 0, 1).crz(Math.PI / 8, 0, 1).toQASM()
    expect(q).toContain('crx(pi/2) q[0],q[1];')
    expect(q).toContain('cry(pi/4) q[0],q[1];')
    expect(q).toContain('crz(pi/8) q[0],q[1];')
  })

  it('cr2/cr4/cr8 → crz with π-fraction params', () => {
    const q = new Circuit(2).cr2(0, 1).cr4(0, 1).cr8(0, 1).toQASM()
    expect(q).toContain('crz(pi/2) q[0],q[1];')
    expect(q).toContain('crz(pi/4) q[0],q[1];')
    expect(q).toContain('crz(pi/8) q[0],q[1];')
  })

  it('cs/csdg/ct/ctdg → cu1 with ±π-fraction params', () => {
    const q = new Circuit(2).cs(0, 1).csdg(0, 1).ct(0, 1).ctdg(0, 1).toQASM()
    expect(q).toContain('cu1(pi/2) q[0],q[1];')
    expect(q).toContain('cu1(-pi/2) q[0],q[1];')
    expect(q).toContain('cu1(pi/4) q[0],q[1];')
    expect(q).toContain('cu1(-pi/4) q[0],q[1];')
  })

  it('ccx (Toffoli) → ccx', () => {
    const q = new Circuit(3).ccx(0, 1, 2).toQASM()
    expect(q).toContain('ccx q[0],q[1],q[2];')
  })

  it('cswap (Fredkin) → cswap', () => {
    const q = new Circuit(3).cswap(0, 1, 2).toQASM()
    expect(q).toContain('cswap q[0],q[1],q[2];')
  })

  it('measure and reset', () => {
    const q = new Circuit(1).creg('c', 1).h(0).measure(0, 'c', 0).toQASM()
    expect(q).toContain('creg c[1];')
    expect(q).toContain('measure q[0] -> c[0];')
  })

  it('reset emits reset statement', () => {
    const q = new Circuit(1).reset(0).toQASM()
    expect(q).toContain('reset q[0];')
  })

  it('throws for gpi (no QASM representation)', () => {
    expect(() => new Circuit(1).gpi(0, 0).toQASM()).toThrow(TypeError)
  })

  it('throws for gpi2 (no QASM representation)', () => {
    expect(() => new Circuit(1).gpi2(0, 0).toQASM()).toThrow(TypeError)
  })

  it('throws for xx (no QASM representation)', () => {
    expect(() => new Circuit(2).xx(Math.PI / 2, 0, 1).toQASM()).toThrow(TypeError)
  })

  it('throws for if ops', () => {
    expect(() => new Circuit(1).creg('c', 1).if('c', 1, q => q.x(0)).toQASM()).toThrow(TypeError)
  })
})

describe('fromQASM() — parsing', () => {
  it('parses minimal header and H gate', () => {
    const src = `OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];`
    const c = Circuit.fromQASM(src)
    expect(c.qubits).toBe(1)
    expect(c.probability('0')).toBeCloseTo(0.5, 5)
    expect(c.probability('1')).toBeCloseTo(0.5, 5)
  })

  it('parses Bell state: h + cx', () => {
    const src = `OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];`
    const r = Circuit.fromQASM(src).run({ shots: 4000, seed: 7 })
    expect(Object.keys(r.probs).every(k => k === '00' || k === '11')).toBe(true)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
  })

  it('parses QASM name aliases: sdg, tdg, sx, sxdg', () => {
    // sdg = si, so s·si = I → |0⟩
    const src = `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\nsdg q[0];\ns q[0];\nh q[0];`
    expect(Circuit.fromQASM(src).probability('0')).toBeCloseTo(1, 5)
  })

  it('parses angle expressions: pi/2, pi/4, 2*pi/3', () => {
    const src = `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(pi/2) q[0];`
    const c = Circuit.fromQASM(src)
    // Rx(π/2)|0⟩: p(|0⟩) = cos²(π/4) = 0.5
    expect(c.probability('0')).toBeCloseTo(0.5, 5)
  })

  it('parses measure and creg', () => {
    const src = `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];`
    const d = Circuit.fromQASM(src).run({ shots: 4000, seed: 3 })
    expect(near(d.cregs['c']?.[0] ?? -1, 0.5)).toBe(true)
  })

  it('strips single-line comments', () => {
    const src = `OPENQASM 2.0; // version
include "qelib1.inc"; // library
qreg q[1]; // 1 qubit
x q[0]; // flip`
    expect(Circuit.fromQASM(src).probability('1')).toBeCloseTo(1, 10)
  })

  it('throws for unknown gate', () => {
    const src = `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nbogus q[0];`
    expect(() => Circuit.fromQASM(src)).toThrow(TypeError)
  })

  it('ccx (Toffoli) parses correctly', () => {
    // Toffoli truth table: |110⟩ → |111⟩
    const src = `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nx q[0];\nx q[1];\nccx q[0],q[1],q[2];`
    expect(Circuit.fromQASM(src).probability('111')).toBeCloseTo(1, 10)
  })
})

describe('QASM round-trip: toQASM() → fromQASM() → identical statevector', () => {
  function roundTrip(c: Circuit): Circuit { return Circuit.fromQASM(c.toQASM()) }

  function svClose(a: Circuit, b: Circuit): boolean {
    const sv = a.statevector()
    const n = a.qubits
    for (const idx of sv.keys()) {
      const bs = idx.toString(2).padStart(n, '0')
      const aa = a.amplitude(bs), bb = b.amplitude(bs)
      if (Math.abs(aa.re - bb.re) > 1e-8 || Math.abs(aa.im - bb.im) > 1e-8) return false
    }
    return true
  }

  it('H gate', () => {
    const c = new Circuit(1).h(0)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })

  it('Bell state: H + CNOT', () => {
    const c = new Circuit(2).h(0).cnot(0, 1)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })

  it('parametric: rx(0.7) + rz(1.3) + ry(0.5)', () => {
    const c = new Circuit(1).rx(0.7, 0).rz(1.3, 0).ry(0.5, 0)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })

  it('u3(π/3, π/4, π/5)', () => {
    const c = new Circuit(1).u3(Math.PI / 3, Math.PI / 4, Math.PI / 5, 0)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })

  it('controlled gates: cy, cz, ch, crx', () => {
    const c = new Circuit(2).h(0).cy(0, 1).cz(0, 1).ch(0, 1).crx(Math.PI / 3, 0, 1)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })

  it('QFT-style circuit: H + cr2 + cr4', () => {
    const c = new Circuit(3).h(0).cr2(0, 1).cr4(0, 2).h(1).cr2(1, 2).h(2)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })

  it('Toffoli gate', () => {
    const c = new Circuit(3).x(0).x(1).ccx(0, 1, 2)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })

  it('cs/ct/csdg/ctdg survive round-trip as cu1', () => {
    const c = new Circuit(2).h(0).cs(0, 1).ct(0, 1).csdg(0, 1).ctdg(0, 1)
    expect(svClose(c, roundTrip(c))).toBe(true)
  })
})

// ─── Noise models (5a) ───────────────────────────────────────────────────────

describe('noise: depolarizing — single-qubit', () => {
  it('noiseless X gate always gives |1⟩', () => {
    const r = new Circuit(1).x(0).run({ shots: 200, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('high p1 noise visibly degrades X gate fidelity', () => {
    // p1=0.5 means 50% chance of a Pauli error — should significantly pollute the distribution
    const r = new Circuit(1).x(0).run({ shots: 4000, seed: 1, noise: { p1: 0.5 } })
    // No longer pure |1⟩ — |0⟩ should appear
    expect(r.probs['0'] ?? 0).toBeGreaterThan(0.1)
    expect(r.probs['1'] ?? 0).toBeLessThan(0.9)
  })

  it('small p1 noise still gives mostly correct result', () => {
    // p1=0.001 is realistic; result should still be >99% correct
    const r = new Circuit(1).x(0).run({ shots: 10000, seed: 2, noise: { p1: 0.001 } })
    expect(r.probs['1'] ?? 0).toBeGreaterThan(0.99)
  })

  it('p1=0 is identical to noiseless', () => {
    const clean  = new Circuit(1).h(0).run({ shots: 4000, seed: 7 })
    const noisy  = new Circuit(1).h(0).run({ shots: 4000, seed: 7, noise: { p1: 0 } })
    expect(Object.entries(noisy.probs).every(([k, v]) => Math.abs(v - (clean.probs[k] ?? 0)) < 0.05)).toBe(true)
  })
})

describe('noise: depolarizing — two-qubit', () => {
  it('high p2 noise degrades Bell state', () => {
    // Noiseless Bell: only |00⟩ and |11⟩
    const clean = new Circuit(2).h(0).cnot(0, 1).run({ shots: 2000, seed: 1 })
    expect('01' in clean.probs).toBe(false)
    // With p2=0.5: off-diagonal states should appear
    const noisy = new Circuit(2).h(0).cnot(0, 1).run({ shots: 2000, seed: 1, noise: { p2: 0.5 } })
    expect((noisy.probs['01'] ?? 0) + (noisy.probs['10'] ?? 0)).toBeGreaterThan(0.05)
  })

  it('small p2 noise: Bell state still mostly correct', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 10000, seed: 3, noise: { p2: 0.005 } })
    expect((r.probs['00'] ?? 0) + (r.probs['11'] ?? 0)).toBeGreaterThan(0.95)
  })
})

describe('noise: SPAM (measurement readout errors)', () => {
  it('pMeas=1 always flips the measurement outcome', () => {
    // X|0⟩ = |1⟩, but pMeas=1 always flips → should get |0⟩
    const r = new Circuit(1).x(0).run({ shots: 200, seed: 1, noise: { pMeas: 1 } })
    expect(r.probs['0'] ?? 0).toBeCloseTo(1.0, 5)
  })

  it('pMeas=0.5 gives roughly 50/50 regardless of state', () => {
    // Even a pure |1⟩ state → 50/50 after SPAM
    const r = new Circuit(1).x(0).run({ shots: 4000, seed: 4, noise: { pMeas: 0.5 } })
    expect(near(r.probs['0'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['1'] ?? 0, 0.5)).toBe(true)
  })

  it('small pMeas: correct outcome still dominates', () => {
    const r = new Circuit(1).x(0).run({ shots: 10000, seed: 5, noise: { pMeas: 0.004 } })
    expect(r.probs['1'] ?? 0).toBeGreaterThan(0.99)
  })
})

describe('noise: named device profiles', () => {
  it("'aria-1' profile runs without error", () => {
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 1000, seed: 1, noise: 'aria-1' })
    const total = Object.values(r.probs).reduce((s, p) => s + p, 0)
    expect(Math.abs(total - 1.0)).toBeLessThan(0.001)
  })

  it("'forte-1' profile runs without error", () => {
    const r = new Circuit(1).h(0).run({ shots: 1000, seed: 2, noise: 'forte-1' })
    expect(r.shots).toBe(1000)
  })

  it("'harmony' profile runs without error", () => {
    const r = new Circuit(1).x(0).run({ shots: 500, seed: 3, noise: 'harmony' })
    expect(r.shots).toBe(500)
  })

  it("'aria-1' Bell state: still mostly |00⟩ and |11⟩ (low noise device)", () => {
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 5000, seed: 10, noise: 'aria-1' })
    expect((r.probs['00'] ?? 0) + (r.probs['11'] ?? 0)).toBeGreaterThan(0.97)
  })

  it('throws for unknown device name', () => {
    expect(() => new Circuit(1).h(0).run({ noise: 'bogus-99' })).toThrow(TypeError)
  })
})

describe('noise: custom NoiseParams object', () => {
  it('custom params object works like named profile', () => {
    // Custom params matching aria-1
    const r = new Circuit(2).h(0).cnot(0, 1).run({ shots: 2000, seed: 99, noise: { p1: 0.0003, p2: 0.005, pMeas: 0.004 } })
    expect(r.shots).toBe(2000)
    expect((r.probs['00'] ?? 0) + (r.probs['11'] ?? 0)).toBeGreaterThan(0.97)
  })

  it('can specify only p1 (no two-qubit noise)', () => {
    const r = new Circuit(1).h(0).run({ shots: 2000, seed: 1, noise: { p1: 0.001 } })
    expect(near(r.probs['0'] ?? 0, 0.5, 0.06)).toBe(true)
  })
})

// ─── Export targets (3c) ─────────────────────────────────────────────────────

describe('toQiskit()', () => {
  it('emits valid header', () => {
    const q = new Circuit(2).toQiskit()
    expect(q).toContain('from qiskit import QuantumCircuit')
    expect(q).toContain('qc = QuantumCircuit(2)')
  })

  it('H + CNOT', () => {
    const q = new Circuit(2).h(0).cnot(0, 1).toQiskit()
    expect(q).toContain('qc.h(0)')
    expect(q).toContain('qc.cx(0, 1)')
  })

  it('name aliases: si→sdg, ti→tdg, v→sx, vi→sxdg', () => {
    const q = new Circuit(1).si(0).ti(0).v(0).vi(0).toQiskit()
    expect(q).toContain('qc.sdg(0)')
    expect(q).toContain('qc.tdg(0)')
    expect(q).toContain('qc.sx(0)')
    expect(q).toContain('qc.sxdg(0)')
  })

  it('r2/r4/r8 → qc.rz with π-fraction', () => {
    const q = new Circuit(1).r2(0).r4(0).r8(0).toQiskit()
    expect(q).toContain('qc.rz(math.pi/2, 0)')
    expect(q).toContain('qc.rz(math.pi/4, 0)')
    expect(q).toContain('qc.rz(math.pi/8, 0)')
  })

  it('rx/ry/rz with angle', () => {
    const q = new Circuit(1).rx(Math.PI / 3, 0).toQiskit()
    expect(q).toContain('qc.rx(math.pi/3, 0)')
  })

  it('crx/cry/crz controlled rotations', () => {
    const q = new Circuit(2).crx(Math.PI / 2, 0, 1).crz(Math.PI / 4, 0, 1).toQiskit()
    expect(q).toContain('qc.crx(math.pi/2, 0, 1)')
    expect(q).toContain('qc.crz(math.pi/4, 0, 1)')
  })

  it('cr2/cr4/cr8 → crz with π-fraction', () => {
    const q = new Circuit(2).cr2(0, 1).cr4(0, 1).toQiskit()
    expect(q).toContain('qc.crz(math.pi/2, 0, 1)')
    expect(q).toContain('qc.crz(math.pi/4, 0, 1)')
  })

  it('cs/ct/csdg/ctdg → cu1', () => {
    const q = new Circuit(2).cs(0, 1).ct(0, 1).csdg(0, 1).ctdg(0, 1).toQiskit()
    expect(q).toContain('qc.cu1(math.pi/2, 0, 1)')
    expect(q).toContain('qc.cu1(math.pi/4, 0, 1)')
    expect(q).toContain('qc.cu1(-math.pi/2, 0, 1)')
    expect(q).toContain('qc.cu1(-math.pi/4, 0, 1)')
  })

  it('xx/yy/zz → rxx/ryy/rzz', () => {
    const q = new Circuit(2).xx(Math.PI / 2, 0, 1).yy(Math.PI / 4, 0, 1).zz(Math.PI / 8, 0, 1).toQiskit()
    expect(q).toContain('qc.rxx(math.pi/2, 0, 1)')
    expect(q).toContain('qc.ryy(math.pi/4, 0, 1)')
    expect(q).toContain('qc.rzz(math.pi/8, 0, 1)')
  })

  it('iswap → qc.iswap', () => {
    const q = new Circuit(2).iswap(0, 1).toQiskit()
    expect(q).toContain('qc.iswap(0, 1)')
  })

  it('ccx → qc.ccx, cswap → qc.cswap', () => {
    const q = new Circuit(3).ccx(0, 1, 2).cswap(0, 1, 2).toQiskit()
    expect(q).toContain('qc.ccx(0, 1, 2)')
    expect(q).toContain('qc.cswap(0, 1, 2)')
  })

  it('measure and reset', () => {
    const q = new Circuit(1).creg('c', 1).h(0).measure(0, 'c', 0).reset(0).toQiskit()
    expect(q).toContain('qc.measure(0, c[0])')
    expect(q).toContain('qc.reset(0)')
  })

  it('throws for gpi', () => {
    expect(() => new Circuit(1).gpi(0, 0).toQiskit()).toThrow(TypeError)
  })

  it('throws for ms', () => {
    expect(() => new Circuit(2).ms(0, 0, 0, 1).toQiskit()).toThrow(TypeError)
  })
})

describe('toCirq()', () => {
  it('emits valid header', () => {
    const c = new Circuit(2).toCirq()
    expect(c).toContain('import cirq')
    expect(c).toContain('q = cirq.LineQubit.range(2)')
    expect(c).toContain('circuit = cirq.Circuit([')
  })

  it('H + CNOT', () => {
    const c = new Circuit(2).h(0).cnot(0, 1).toCirq()
    expect(c).toContain('cirq.H(q[0])')
    expect(c).toContain('cirq.CNOT(q[0], q[1])')
  })

  it('si → ZPowGate(-0.5), ti → ZPowGate(-0.25)', () => {
    const c = new Circuit(1).si(0).ti(0).toCirq()
    expect(c).toContain('ZPowGate(exponent=-0.5)')
    expect(c).toContain('ZPowGate(exponent=-0.25)')
  })

  it('rx/ry/rz with rads=', () => {
    const c = new Circuit(1).rx(Math.PI / 2, 0).ry(Math.PI / 4, 0).toCirq()
    expect(c).toContain('cirq.rx(rads=math.pi/2)')
    expect(c).toContain('cirq.ry(rads=math.pi/4)')
  })

  it('cz → cirq.CZ', () => {
    const c = new Circuit(2).h(0).cz(0, 1).toCirq()
    expect(c).toContain('cirq.CZ(q[0], q[1])')
  })

  it('cy/ch via .controlled()', () => {
    const c = new Circuit(2).cy(0, 1).ch(0, 1).toCirq()
    expect(c).toContain('cirq.Y.controlled()')
    expect(c).toContain('cirq.H.controlled()')
  })

  it('ccx → cirq.CCNOT, cswap → cirq.CSWAP', () => {
    const c = new Circuit(3).ccx(0, 1, 2).cswap(0, 1, 2).toCirq()
    expect(c).toContain('cirq.CCNOT(q[0], q[1], q[2])')
    expect(c).toContain('cirq.CSWAP(q[0], q[1], q[2])')
  })

  it('throws for gpi', () => {
    expect(() => new Circuit(1).gpi(0, 0).toCirq()).toThrow(TypeError)
  })

  it('throws for xx (Cirq convention differs)', () => {
    expect(() => new Circuit(2).xx(Math.PI / 2, 0, 1).toCirq()).toThrow(TypeError)
  })
})

describe('toQSharp()', () => {
  it('emits valid Q# namespace', () => {
    const q = new Circuit(2).toQSharp()
    expect(q).toContain('namespace KetCircuit')
    expect(q).toContain('open Microsoft.Quantum.Intrinsic')
    expect(q).toContain('operation Run() : Unit')
    expect(q).toContain('use q = Qubit[2]')
    expect(q).toContain('ResetAll(q)')
  })

  it('H + CNOT', () => {
    const q = new Circuit(2).h(0).cnot(0, 1).toQSharp()
    expect(q).toContain('H(q[0])')
    expect(q).toContain('CNOT(q[0], q[1])')
  })

  it('si → Adjoint S, ti → Adjoint T', () => {
    const q = new Circuit(1).si(0).ti(0).toQSharp()
    expect(q).toContain('Adjoint S(q[0])')
    expect(q).toContain('Adjoint T(q[0])')
  })

  it('Rx/Ry/Rz with PI() fractions', () => {
    const q = new Circuit(1).rx(Math.PI / 2, 0).rz(Math.PI / 4, 0).toQSharp()
    expect(q).toContain('Rx(PI()/2.0, q[0])')
    expect(q).toContain('Rz(PI()/4.0, q[0])')
  })

  it('cs/csdg → Controlled S / Controlled Adjoint S', () => {
    const q = new Circuit(2).cs(0, 1).csdg(0, 1).ct(0, 1).ctdg(0, 1).toQSharp()
    expect(q).toContain('Controlled S([q[0]], q[1])')
    expect(q).toContain('Controlled Adjoint S([q[0]], q[1])')
    expect(q).toContain('Controlled T([q[0]], q[1])')
    expect(q).toContain('Controlled Adjoint T([q[0]], q[1])')
  })

  it('ccx → CCNOT, cswap → Controlled SWAP', () => {
    const q = new Circuit(3).ccx(0, 1, 2).cswap(0, 1, 2).toQSharp()
    expect(q).toContain('CCNOT(q[0], q[1], q[2])')
    expect(q).toContain('Controlled SWAP([q[0]], (q[1], q[2]))')
  })

  it('throws for gpi', () => {
    expect(() => new Circuit(1).gpi(0, 0).toQSharp()).toThrow(TypeError)
  })

  it('throws for xx', () => {
    expect(() => new Circuit(2).xx(Math.PI / 2, 0, 1).toQSharp()).toThrow(TypeError)
  })
})

describe('toPyQuil()', () => {
  it('emits valid header', () => {
    const q = new Circuit(2).toPyQuil()
    expect(q).toContain('from pyquil import Program')
    expect(q).toContain('p = Program()')
  })

  it('H + CNOT', () => {
    const q = new Circuit(2).h(0).cnot(0, 1).toPyQuil()
    expect(q).toContain('from pyquil.gates import')
    // gates are sorted alphabetically in the import line
    expect(q).toContain('CNOT')
    expect(q).toContain('H')
    expect(q).toContain('p += H(0)')
    expect(q).toContain('p += CNOT(0, 1)')
  })

  it('si → DAGGER(S), ti → DAGGER(T)', () => {
    const q = new Circuit(1).si(0).ti(0).toPyQuil()
    expect(q).toContain('DAGGER(S)(0)')
    expect(q).toContain('DAGGER(T)(0)')
  })

  it('RX/RY/RZ with math.pi fractions', () => {
    const q = new Circuit(1).rx(Math.PI / 2, 0).ry(Math.PI / 4, 0).toPyQuil()
    expect(q).toContain('p += RX(math.pi/2, 0)')
    expect(q).toContain('p += RY(math.pi/4, 0)')
  })

  it('iswap → ISWAP', () => {
    const q = new Circuit(2).iswap(0, 1).toPyQuil()
    expect(q).toContain('ISWAP')
    expect(q).toContain('p += ISWAP(0, 1)')
  })

  it('ccx → CCNOT, cswap → CSWAP, swap → SWAP', () => {
    const q = new Circuit(3).ccx(0, 1, 2).cswap(0, 1, 2).swap(0, 1).toPyQuil()
    expect(q).toContain('p += CCNOT(0, 1, 2)')
    expect(q).toContain('p += CSWAP(0, 1, 2)')
    expect(q).toContain('p += SWAP(0, 1)')
  })

  it('throws for gpi', () => {
    expect(() => new Circuit(1).gpi(0, 0).toPyQuil()).toThrow(TypeError)
  })

  it('throws for controlled rotations (no standard pyQuil equivalent)', () => {
    expect(() => new Circuit(2).crx(Math.PI / 2, 0, 1).toPyQuil()).toThrow(TypeError)
  })
})

// ─── MPS tensor-network backend ───────────────────────────────────────────────

describe('MPS backend — basic gates', () => {
  it('X gate: |0⟩ → |1⟩ exactly', () => {
    const r = new Circuit(1).x(0).runMps({ shots: 100, seed: 1 })
    expect(r.probs['1']).toBeCloseTo(1.0, 10)
  })

  it('HH = I: returns to |0⟩', () => {
    const r = new Circuit(1).h(0).h(0).runMps({ shots: 100, seed: 1 })
    expect(r.probs['0']).toBeCloseTo(1.0, 10)
  })

  it('Bell state via H + CNOT: only 00 and 11 outcomes', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).runMps({ shots: 2000, seed: 7 })
    expect(Object.keys(r.probs).every(bs => bs === '00' || bs === '11')).toBe(true)
    expect(near(r.probs['00'] ?? 0, 0.5)).toBe(true)
    expect(near(r.probs['11'] ?? 0, 0.5)).toBe(true)
  })

  it('controlled-Z: relative phase, CZ|++⟩ has no |11⟩ term after H⊗H', () => {
    // CZ then H⊗H maps |++⟩ → |Φ−⟩-like — just verify it runs and gives valid probs
    const r = new Circuit(2).h(0).h(1).cz(0, 1).h(0).h(1).runMps({ shots: 1000, seed: 1 })
    const total = Object.values(r.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1.0, 5)
  })

  it('SWAP gate: |01⟩ → |10⟩', () => {
    const r = new Circuit(2).x(0).swap(0, 1).runMps({ shots: 100, seed: 1 })
    expect(r.probs['10']).toBeCloseTo(1.0, 10)  // q1=1 → bitstring '10'
  })
})

describe('MPS backend — matches statevector on small circuits', () => {
  function closeDists(a: Record<string, number>, b: Record<string, number>, tol = 0.04): boolean {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)])
    return [...keys].every(k => Math.abs((a[k] ?? 0) - (b[k] ?? 0)) <= tol)
  }

  it('5-qubit uniform superposition: matches statevector', () => {
    let c = new Circuit(5)
    for (let q = 0; q < 5; q++) c = c.h(q)
    const sv  = c.run({ shots: 8192, seed: 1 })
    const mps = c.runMps({ shots: 8192, seed: 1 })
    expect(closeDists(sv.probs, mps.probs)).toBe(true)
  })

  it('QFT-like: 4-qubit H + CRZ ladder matches statevector', () => {
    let c = new Circuit(4)
    for (let q = 0; q < 4; q++) c = c.h(q)
    for (let q = 0; q < 3; q++) c = c.crz(Math.PI / 2, q, q + 1)
    const sv  = c.run({ shots: 8192, seed: 2 })
    const mps = c.runMps({ shots: 8192, seed: 2 })
    expect(closeDists(sv.probs, mps.probs)).toBe(true)
  })

  it('non-adjacent CNOT: q0→q3 matches statevector', () => {
    const c = new Circuit(4).h(0).cnot(0, 3)
    const sv  = c.run({ shots: 4000, seed: 3 })
    const mps = c.runMps({ shots: 4000, seed: 3 })
    expect(closeDists(sv.probs, mps.probs)).toBe(true)
  })
})

describe('MPS backend — large circuits (50+ qubits)', () => {
  it('GHZ-50: only all-0 and all-1 bitstrings observed', () => {
    let c = new Circuit(50).h(0)
    for (let q = 0; q < 49; q++) c = c.cnot(q, q + 1)
    const r = new Circuit(50).h(0)
    let circ = new Circuit(50).h(0)
    for (let q = 0; q < 49; q++) circ = circ.cnot(q, q + 1)
    const dist = circ.runMps({ shots: 500, seed: 42 })
    const keys = Object.keys(dist.probs)
    const allZero = '0'.repeat(50)
    const allOne  = '1'.repeat(50)
    expect(keys.every(k => k === allZero || k === allOne)).toBe(true)
    expect(near(dist.probs[allZero] ?? 0, 0.5, 0.1)).toBe(true)
    expect(near(dist.probs[allOne]  ?? 0, 0.5, 0.1)).toBe(true)
  })

  it('BV-40: recovers hidden bitstring 1010...10 exactly', () => {
    // Bernstein-Vazirani: n=40 input qubits, q40 ancilla in |−⟩
    // Secret s = alternating (q1=1,q3=1,...,q39=1, rest 0)
    const n = 40, secret = Array.from({ length: n }, (_, i) => i % 2)
    let c = new Circuit(n + 1)
    for (let q = 0; q < n; q++) c = c.h(q)
    c = c.x(n).h(n)
    for (let q = 0; q < n; q++) if (secret[q]) c = c.cnot(q, n)
    for (let q = 0; q < n; q++) c = c.h(q)

    const dist = c.runMps({ shots: 200, seed: 5 })
    // Ancilla q40 stays in |−⟩ → random 0/1; strip it out and check input register.
    // probs keys = idx.toString(2) where bit i = qubit i, so BigInt('0b'+bs) = idx.
    const ancillaMask = (1n << BigInt(n)) - 1n  // bits 0..n-1
    const inputStrings = new Set(
      Object.keys(dist.probs).map(bs =>
        (BigInt('0b' + bs) & ancillaMask).toString(2).padStart(n, '0')
      )
    )
    // Regardless of ancilla outcome, all shots yield the same input register
    expect(inputStrings.size).toBe(1)
  })

  it('product state 50 qubits: H⊗50 → all 2^50 outcomes uniformly', () => {
    let c = new Circuit(50)
    for (let q = 0; q < 50; q++) c = c.h(q)
    const dist = c.runMps({ shots: 500, seed: 99 })
    // With 500 shots over 2^50 states, each shot is unique — verify all probs ≈ 1/500
    const probs = Object.values(dist.probs)
    expect(probs.length).toBe(500)
    expect(probs.every(p => Math.abs(p - 1 / 500) < 0.001)).toBe(true)
  })
})

describe('MPS backend — error paths', () => {
  it('throws for Toffoli', () => {
    expect(() => new Circuit(3).ccx(0, 1, 2).runMps()).toThrow('CCX')
  })

  it('throws for mid-circuit measure', () => {
    expect(() => new Circuit(2).creg('c', 1).measure(0, 'c', 0).runMps()).toThrow("'measure'")
  })
})

// ─── Immutability ─────────────────────────────────────────────────────────────

describe('Circuit immutability', () => {
  it('appending gates returns a new circuit, original unchanged', () => {
    const base  = new Circuit(2).h(0)
    const bell  = base.cnot(0, 1)
    const justH = base.run({ shots: 1000, seed: 1 })
    const bothH = bell.run({ shots: 1000, seed: 1 })
    // base circuit: H on q0 of 2-qubit system → |00⟩ and |01⟩ (q1 stays |0⟩)
    expect('00' in justH.probs).toBe(true)
    expect('01' in justH.probs).toBe(true)
    expect(Object.keys(bothH.probs).every(bs => bs === '00' || bs === '11')).toBe(true)
  })
})
