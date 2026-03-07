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
  for (const n of [3, 5, 8, 12, 16, 20]) {
    it(`${n}-qubit GHZ produces only |${'0'.repeat(n)}⟩ and |${'1'.repeat(n)}⟩`, () => {
      let c = new Circuit(n).h(0)
      for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
      const r = c.run({ shots: 512, seed: 42 })
      expect(Object.keys(r.probs)).toHaveLength(2)
      expect(near(r.probs['0'.repeat(n)] ?? 0, 0.5, 0.1)).toBe(true)
      expect(near(r.probs['1'.repeat(n)] ?? 0, 0.5, 0.1)).toBe(true)
    })
  }
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

  it('entropy of uniform superposition over 2^n outcomes = n bits', () => {
    for (const n of [1, 2, 3, 4]) {
      let c = new Circuit(n)
      for (let i = 0; i < n; i++) c = c.h(i)
      const r = c.run({ shots: 200000, seed: 42 })
      expect(r.entropy).toBeCloseTo(n, 1)
    }
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
