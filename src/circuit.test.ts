import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'

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
