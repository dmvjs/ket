/**
 * Targeted tests for coverage gaps identified in circuit.ts, algorithms.ts,
 * statevector.ts, mps.ts, and density.ts.  Each suite documents either a
 * confirmed bug fix or a behavioural contract not covered by circuit.test.ts.
 */

import { describe, expect, it } from 'vitest'
import { Circuit, DEVICES, IONQ_DEVICES } from './circuit.js'
import { CliffordSim } from './clifford.js'
import { trotter, iqft, qft, grover, vqe } from './algorithms.js'
import type { PauliTerm } from './algorithms.js'
import { c, add, mul, conj, norm2, ZERO, ONE, I } from './complex.js'

// ─── pauliEvolution — Y-term sign ────────────────────────────────────────────
//
// Bug fix: exp(−iθY) requires Rx(+π/2) before Rz(2θ) and Rx(−π/2) after.
// The previous code had these swapped, implementing exp(+iθY) instead.
// Distinguishable via ⟨X⟩: correct gives sin(2θ), wrong gives −sin(2θ).

describe('pauliEvolution — Y-term sign', () => {
  it('single Y: ⟨X⟩ = sin(2θ)', () => {
    const theta = Math.PI / 4
    expect(vqe(trotter(1, [{ coeff: 1, ops: 'Y' }], theta), [{ coeff: 1, ops: 'X' }]))
      .toBeCloseTo(Math.sin(2 * theta), 8)
  })

  it('Y evolution matches Ry(2θ) amplitudes', () => {
    const theta = Math.PI / 5
    const sv = trotter(1, [{ coeff: 1, ops: 'Y' }], theta).statevector()
    const ref = new Circuit(1).ry(2 * theta, 0).statevector()
    expect(sv.get(0n)!.re).toBeCloseTo(ref.get(0n)!.re, 8)
    expect(sv.get(1n)!.re).toBeCloseTo(ref.get(1n)!.re, 8)
  })

  it('Trotter order=2 more accurate than order=1 for X+Y Hamiltonian', () => {
    const ham: PauliTerm[] = [{ coeff: 1, ops: 'X' }, { coeff: 1, ops: 'Y' }]
    const exact   = trotter(1, ham, 0.5, 100, 1).exactProbs()['0'] ?? 0
    const err1    = Math.abs((trotter(1, ham, 0.5, 1, 1).exactProbs()['0'] ?? 0) - exact)
    const err2    = Math.abs((trotter(1, ham, 0.5, 1, 2).exactProbs()['0'] ?? 0) - exact)
    expect(err2).toBeLessThan(err1)
  })
})

// ─── parseAngle — rejects malformed input ────────────────────────────────────
//
// Bug fix: parseAngle() returned NaN silently for malformed expressions, which
// propagated into gate rotation parameters.  Now throws TypeError when the full
// expression string is not consumed or the result is not finite.

describe('parseAngle — rejects malformed input', () => {
  it('garbage identifier throws', () => {
    expect(() => Circuit.fromQASM('OPENQASM 2.0;\nqreg q[1];\nrz(garbage) q[0];')).toThrow(TypeError)
  })

  it('trailing operator "pi+" throws', () => {
    expect(() => Circuit.fromQASM('OPENQASM 2.0;\nqreg q[1];\nrz(pi+) q[0];')).toThrow(TypeError)
  })

  it('valid expressions parse without error', () => {
    expect(() => Circuit.fromQASM('OPENQASM 2.0;\nqreg q[1];\nrx(-pi/2) q[0];')).not.toThrow()
    expect(() => Circuit.fromQASM('OPENQASM 2.0;\nqreg q[1];\nrz(pi/4*2) q[0];')).not.toThrow()
  })
})

// ─── controlled gate — self-target throws ────────────────────────────────────
//
// Bug fix: cnot/cy/cz/cu1 with control === target produced non-unitary output
// (norm² ≠ 1) without throwing.  Now caught at circuit-build time in #checkOp
// and at execution time in applyControlled.

describe('controlled gate — self-target throws', () => {
  it('cnot(q, q) throws TypeError', () => {
    expect(() => new Circuit(1).cnot(0, 0)).toThrow(TypeError)
    expect(() => new Circuit(2).cnot(1, 1)).toThrow(TypeError)
  })

  it('cy / cz / cu1 with equal qubits throw', () => {
    expect(() => new Circuit(1).cy(0, 0)).toThrow(TypeError)
    expect(() => new Circuit(2).cz(0, 0)).toThrow(TypeError)
    expect(() => new Circuit(2).cu1(Math.PI / 2, 0, 0)).toThrow(TypeError)
  })

  it('distinct qubits do not throw', () => {
    expect(() => new Circuit(2).cnot(0, 1)).not.toThrow()
  })
})

// ─── mpsSample — bond truncation safety ──────────────────────────────────────
//
// Defensive fix: mpsSample line 250 had 1/sqrt(0) when aggressive truncation
// zeroed the marginal probability.  Now guards against total=0 and chosen=0.

describe('mpsSample — bond truncation', () => {
  it('maxBond=1 on Bell state: probabilities finite and non-NaN', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).runMps({ shots: 100, seed: 1, maxBond: 1 })
    for (const p of Object.values(r.probs)) {
      expect(isFinite(p)).toBe(true)
      expect(isNaN(p)).toBe(false)
    }
  })

  it('maxBond=1 on 4-qubit GHZ: probabilities finite', () => {
    const r = new Circuit(4).h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)
      .runMps({ shots: 50, seed: 1, maxBond: 1 })
    for (const p of Object.values(r.probs)) expect(isFinite(p)).toBe(true)
  })

  it('maxBond=2 on Bell state: correct 50/50 distribution', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).runMps({ shots: 10000, seed: 42, maxBond: 2 })
    expect(r.probs['00'] ?? 0).toBeGreaterThan(0.47)
    expect(r.probs['11'] ?? 0).toBeGreaterThan(0.47)
    expect(r.probs['01'] ?? 0).toBeLessThan(0.01)
  })

  it('probabilities sum ≤ 1 under truncation', () => {
    const r = new Circuit(4).h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)
      .runMps({ shots: 200, seed: 1, maxBond: 1 })
    const total = Object.values(r.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeGreaterThan(0)
    expect(total).toBeLessThanOrEqual(1.001)
  })
})

// ─── qrMGS — rank-deficient contraction paths ────────────────────────────────
//
// qrMGS skips linearly-dependent columns (norm² < 1e-28) after Gram-Schmidt.
// Tested via runMps: product states and double-gate roundtrips exercise rank-1
// and rank-0-after-projection code paths.

describe('qrMGS — rank-1 and roundtrip via runMps', () => {
  it('CNOT no-op on |00⟩: 100% "00"', () => {
    expect(new Circuit(2).cnot(0, 1).runMps({ shots: 100, seed: 1 }).probs['00']).toBeCloseTo(1, 10)
  })

  it('X·X = I: recovers |0⟩', () => {
    expect(new Circuit(1).x(0).x(0).runMps({ shots: 100, seed: 1 }).probs['0']).toBeCloseTo(1, 10)
  })

  it('CNOT applied twice is identity', () => {
    const r = new Circuit(2).h(0).cnot(0, 1).cnot(0, 1).runMps({ shots: 10000, seed: 42 })
    expect(r.probs['00'] ?? 0).toBeGreaterThan(0.47)
    expect(r.probs['10'] ?? 0).toBeGreaterThan(0.47)
    expect(r.probs['01'] ?? 0).toBeLessThan(0.02)
  })

  it('product state: runMps exact for all maxBond values', () => {
    for (const maxBond of [1, 2, 4]) {
      const r = new Circuit(3).h(0).h(1).h(2).runMps({ shots: 20000, seed: 1, maxBond })
      for (const bs of ['000','001','010','011','100','101','110','111'])
        expect(r.probs[bs] ?? 0).toBeGreaterThan(0.10)
    }
  })
})

// ─── depolarize1 — critical points and trace preservation ────────────────────
//
// cf = 1 − 4p/3 reaches 0 at p = 0.75 (off-diagonal terms vanish → maximally
// mixed) and goes negative for p > 0.75 (coherences flip sign — valid CPTP).
// Purity is non-monotone: minimum at p = 0.75, rises back toward p = 1.

describe('depolarize1 — trace and purity', () => {
  it('trace = 1 for all p ∈ {0, 0.75, 0.9}', () => {
    for (const p1 of [0, 0.75, 0.9]) {
      const prob = new Circuit(1).h(0).dm({ noise: { p1 } }).probabilities()
      expect((prob['0'] ?? 0) + (prob['1'] ?? 0)).toBeCloseTo(1, 10)
    }
  })

  it('p=0.75 is the maximally mixed point: probs = 0.5/0.5', () => {
    const prob = new Circuit(1).h(0).dm({ noise: { p1: 0.75 } }).probabilities()
    expect(prob['0']).toBeCloseTo(0.5, 10)
    expect(prob['1']).toBeCloseTo(0.5, 10)
  })

  it('purity non-monotone: minimum 0.5 at p=0.75, rises to 0.5+1/18 at p=1', () => {
    const pur = (p1: number) => new Circuit(1).h(0).dm({ noise: { p1 } }).purity()
    expect(pur(0)).toBeCloseTo(1, 10)
    expect(pur(0.75)).toBeCloseTo(0.5, 10)
    expect(pur(1.0)).toBeCloseTo(0.5 + 1 / 18, 10)
    expect(pur(0.75)).toBeLessThan(pur(0))
    expect(pur(1.0)).toBeGreaterThan(pur(0.75))
  })

  it('depolarize2 trace preserved at p=0.5', () => {
    const prob = new Circuit(2).h(0).cnot(0, 1).dm({ noise: { p2: 0.5 } }).probabilities()
    expect(Object.values(prob).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('depolarize2 at p=15/16 approaches maximally mixed', () => {
    const prob = new Circuit(2).h(0).cnot(0, 1).dm({ noise: { p2: 15 / 16 } }).probabilities()
    for (const bs of ['00', '01', '10', '11'])
      expect(prob[bs] ?? 0).toBeGreaterThan(0.15)
  })
})

// ─── DensityMatrix.get — untested public API ─────────────────────────────────

describe('DensityMatrix.get', () => {
  it('diagonal |0⟩: ρ[0][0]=1, ρ[1][1]=0', () => {
    const dm = new Circuit(1).dm()
    expect(dm.get(0n, 0n).re).toBeCloseTo(1, 10)
    expect(dm.get(1n, 1n).re).toBeCloseTo(0, 10)
  })

  it('|+⟩ off-diagonal: ρ[0][1] = 0.5', () => {
    const dm = new Circuit(1).h(0).dm()
    expect(dm.get(0n, 1n).re).toBeCloseTo(0.5, 10)
    expect(dm.get(0n, 1n).im).toBeCloseTo(0, 10)
  })

  it('absent entry returns zero', () => {
    const v = new Circuit(1).dm().get(1n, 0n)
    expect(v.re).toBeCloseTo(0, 10)
    expect(v.im).toBeCloseTo(0, 10)
  })

  it('Hermitian: ρ[r][c] = conj(ρ[c][r])', () => {
    const dm = new Circuit(2).h(0).cnot(0, 1).dm()
    const a = dm.get(0n, 3n), b = dm.get(3n, 0n)
    expect(a.re).toBeCloseTo(b.re, 10)
    expect(a.im).toBeCloseTo(-b.im, 10)
  })
})

// ─── DensityMatrix.entropy — jacobiEigenvalues convergence ───────────────────
//
// entropy() diagonalises the full 2ⁿ×2ⁿ DM via cyclic Jacobi sweeps (≤ 30n).
// Tests verify convergence for pure states, known 1-bit mixtures, degenerate
// spectra (all-equal eigenvalues), and monotonicity under noise.

describe('DensityMatrix.entropy', () => {
  it('pure state: S = 0', () => {
    expect(new Circuit(2).h(0).cnot(0, 1).dm().entropy()).toBeCloseTo(0, 10)
  })

  it('maximally mixed 1-qubit: S = 1', () => {
    expect(new Circuit(1).h(0).dm({ noise: { p1: 0.75 } }).entropy()).toBeCloseTo(1, 8)
  })

  it('entropy is non-negative for all noise levels', () => {
    for (const p1 of [0, 0.1, 0.5, 0.75, 1.0])
      expect(new Circuit(1).h(0).dm({ noise: { p1 } }).entropy()).toBeGreaterThanOrEqual(0)
  })

  it('degenerate spectrum (near I/4) converges: S ≈ 2', () => {
    const dm = new Circuit(2).h(0).cnot(0, 1).dm({ noise: { p1: 0.75, p2: 0.9375 } })
    expect(dm.entropy()).toBeGreaterThan(1.5)
    expect(dm.entropy()).toBeLessThanOrEqual(2.01)
  })
})

// ─── CliffordSim — n=33 word-boundary gates ──────────────────────────────────
//
// Packed tableau stores qubits in 32-bit words; qubit 32 crosses the boundary.
// Exercises Y, S, Si, CY on the second word.

describe('CliffordSim — n=33 word-boundary gates', () => {
  it('Y|0⟩ → 1, Y|1⟩ → 0', () => {
    const s0 = new CliffordSim(33); s0.y(32)
    expect(s0.measure(32, 0.1)).toBe(1)
    const s1 = new CliffordSim(33); s1.x(32); s1.y(32)
    expect(s1.measure(32, 0.1)).toBe(0)
  })

  it('H·S·S·H = X on qubit 32', () => {
    const sim = new CliffordSim(33)
    sim.h(32); sim.s(32); sim.s(32); sim.h(32)
    expect(sim.measure(32, 0.1)).toBe(1)
  })

  it('S·S† = I on qubit 32', () => {
    const sim = new CliffordSim(33); sim.x(32); sim.s(32); sim.si(32)
    expect(sim.measure(32, 0.1)).toBe(1)
  })

  it('CY across word boundary flips target when control is |1⟩', () => {
    const sim = new CliffordSim(33); sim.x(0); sim.cy(0, 32)
    expect(sim.measure(32, 0.1)).toBe(1)
  })
})

// ─── runMps — unsupported op error paths ─────────────────────────────────────

describe('runMps — unsupported ops throw', () => {
  it('csrswap throws TypeError', () => {
    expect(() => new Circuit(3).csrswap(0, 1, 2).runMps()).toThrow(TypeError)
  })

  it('cswap runs via Toffoli decomposition in MPS mode', () => {
    // cswap no longer throws — it decomposes into CX+T gates automatically
    const d = new Circuit(3).x(0).cswap(0, 1, 2).runMps({ shots: 64, seed: 1 })
    expect(d.backend).toBe('mps')
  })

})

// ─── DEVICES — structural contract ───────────────────────────────────────────

describe('DEVICES export', () => {
  it('DEVICES is a strict superset of IONQ_DEVICES', () => {
    for (const name of Object.keys(IONQ_DEVICES))
      expect(Object.keys(DEVICES)).toContain(name)
    expect(Object.keys(DEVICES).length).toBeGreaterThan(Object.keys(IONQ_DEVICES).length)
  })

  it('all entries have numeric noise fields and qubit count', () => {
    for (const dev of Object.values(DEVICES)) {
      expect(typeof dev.noise.p1).toBe('number')
      expect(typeof dev.noise.p2).toBe('number')
      expect(dev.qubits).toBeGreaterThan(0)
    }
  })

  it('any DEVICES key accepted as noise string by run()', () => {
    for (const name of Object.keys(DEVICES))
      expect(() => new Circuit(1).h(0).run({ shots: 10, seed: 1, noise: name })).not.toThrow()
  })
})
