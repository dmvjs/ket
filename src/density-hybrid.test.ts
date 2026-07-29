import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { c, type Complex } from './complex.js'
import { H, X } from './gates.js'
import type { Gate2x2 } from './statevector.js'
import {
  ddAmplitudeDamping1, ddDepolarize1, ddKraus1, ddPerm, ddPhaseDamping1, ddSingle,
  ddUnitaryN, denseDmGet, denseDmNnz, denseDmZero, dmFromSparse, dmToSparse,
  MAX_DENSE_DM_QUBITS,
} from './density-dense.js'

/**
 * ρ is promoted to dense once it passes 1/32 fill. These tests pin both sides of
 * that boundary and the kernel underneath it.
 *
 * A circuit of X gates leaves ρ with a single non-zero entry, so it stays sparse.
 * H on every qubit fills ρ completely — 4ⁿ entries against a 4ⁿ/32 threshold — so
 * it promotes. Running the same assertions across both shapes covers each path.
 */

/** ρ for |0…0⟩ evolved by H on every qubit — fully dense. */
const densifying = (n: number): Circuit => {
  let k = new Circuit(n)
  for (let q = 0; q < n; q++) k = k.h(q)
  return k
}

describe('dense density-matrix kernel', () => {
  it('|0⟩⟨0| starts with a single unit entry', () => {
    const d = denseDmZero(3)
    expect(denseDmGet(d, 0, 0)).toEqual({ re: 1, im: 0 })
    expect(denseDmNnz(d)).toBe(1)
  })

  it('round-trips through the sparse representation', () => {
    const sparse = new Map<bigint, Complex>([[0n, c(0.5)], [5n, c(0, 0.5)], [10n, c(-0.5)]])
    const back = dmToSparse(dmFromSparse(sparse, 2))
    expect(back.size).toBe(3)
    for (const [k, v] of sparse) {
      expect(back.get(k)!.re).toBeCloseTo(v.re, 15)
      expect(back.get(k)!.im).toBeCloseTo(v.im, 15)
    }
  })

  it('X conjugation flips |0⟩⟨0| to |1⟩⟨1|', () => {
    const d = denseDmZero(1)
    ddSingle(d, 0, X)
    expect(denseDmGet(d, 1, 1).re).toBeCloseTo(1, 12)
    expect(denseDmGet(d, 0, 0).re).toBeCloseTo(0, 12)
  })

  it('H conjugation gives the |+⟩⟨+| block', () => {
    const d = denseDmZero(1)
    ddSingle(d, 0, H)
    for (const [r, col] of [[0, 0], [0, 1], [1, 0], [1, 1]] as const) {
      expect(denseDmGet(d, r, col).re).toBeCloseTo(0.5, 12)
    }
  })

  it('conjugation preserves trace and Hermiticity', () => {
    const d = denseDmZero(3)
    ddSingle(d, 0, H); ddSingle(d, 1, H); ddSingle(d, 2, H)
    let trace = 0
    for (let i = 0; i < 8; i++) trace += denseDmGet(d, i, i).re
    expect(trace).toBeCloseTo(1, 12)
    for (let r = 0; r < 8; r++) for (let col = 0; col < 8; col++) {
      const a = denseDmGet(d, r, col), b = denseDmGet(d, col, r)
      expect(a.re).toBeCloseTo(b.re, 12)
      expect(a.im).toBeCloseTo(-b.im, 12)
    }
  })

  it('ddPerm applies a CNOT permutation to both indices', () => {
    const d = denseDmZero(2)
    ddSingle(d, 0, X)                       // |01⟩⟨01| in (q1 q0) order: index 1
    ddPerm(d, i => (i & 1) !== 0 ? i ^ 2 : i)  // CNOT(0 -> 1)
    expect(denseDmGet(d, 3, 3).re).toBeCloseTo(1, 12)
  })

  it('ddUnitaryN on one qubit matches ddSingle', () => {
    const a = denseDmZero(3), b = denseDmZero(3)
    ddSingle(a, 1, H)
    ddUnitaryN(b, [1], H as unknown as readonly (readonly Complex[])[])
    for (let r = 0; r < 8; r++) for (let col = 0; col < 8; col++) {
      expect(denseDmGet(b, r, col).re).toBeCloseTo(denseDmGet(a, r, col).re, 12)
      expect(denseDmGet(b, r, col).im).toBeCloseTo(denseDmGet(a, r, col).im, 12)
    }
  })

  it('full depolarizing drives one qubit to the maximally mixed state', () => {
    const d = denseDmZero(1)
    ddDepolarize1(d, 0, 0.75)   // p=3/4 is complete depolarization
    expect(denseDmGet(d, 0, 0).re).toBeCloseTo(0.5, 12)
    expect(denseDmGet(d, 1, 1).re).toBeCloseTo(0.5, 12)
    expect(denseDmGet(d, 0, 1).re).toBeCloseTo(0, 12)
  })

  it('full amplitude damping returns |1⟩⟨1| to |0⟩⟨0|', () => {
    const d = denseDmZero(1)
    ddSingle(d, 0, X)
    ddAmplitudeDamping1(d, 0, 1)
    expect(denseDmGet(d, 0, 0).re).toBeCloseTo(1, 12)
    expect(denseDmGet(d, 1, 1).re).toBeCloseTo(0, 12)
  })

  it('full dephasing kills coherence but keeps populations', () => {
    const d = denseDmZero(1)
    ddSingle(d, 0, H)
    ddPhaseDamping1(d, 0, 1)
    expect(denseDmGet(d, 0, 0).re).toBeCloseTo(0.5, 12)
    expect(denseDmGet(d, 1, 1).re).toBeCloseTo(0.5, 12)
    expect(denseDmGet(d, 0, 1).re).toBeCloseTo(0, 12)   // off-diagonal gone
  })

  it('a Kraus bit-flip channel matches its closed form', () => {
    const p = 0.25
    const k0: Gate2x2 = [[c(Math.sqrt(1 - p)), c(0)], [c(0), c(Math.sqrt(1 - p))]]
    const k1: Gate2x2 = [[c(0), c(Math.sqrt(p))], [c(Math.sqrt(p)), c(0)]]
    const d = denseDmZero(1)
    ddKraus1(d, 0, [k0, k1])
    // (1−p)|0⟩⟨0| + p|1⟩⟨1|
    expect(denseDmGet(d, 0, 0).re).toBeCloseTo(1 - p, 12)
    expect(denseDmGet(d, 1, 1).re).toBeCloseTo(p, 12)
  })
})

describe('density matrix — sparse and dense paths agree with the statevector backend', () => {
  // exactProbs() comes from an entirely separate implementation, so matching it
  // is a genuine cross-check rather than a restatement of the same code.

  it('a sparse ρ (X gates only) matches exactProbs', () => {
    const k = new Circuit(6).x(0).x(3).x(5)
    const dmProbs = k.dm().probabilities()
    expect(dmProbs).toEqual(k.exactProbs())
  })

  it('a fully dense ρ matches exactProbs', () => {
    for (const n of [4, 6]) {
      const k = densifying(n)
      const exact = k.exactProbs()
      const probs = k.dm().probabilities()
      expect(Object.keys(probs)).toHaveLength(2 ** n)
      for (const [bits, p] of Object.entries(exact)) {
        expect(Math.abs(probs[bits]! - p), `n=${n} outcome ${bits}`).toBeLessThan(1e-12)
      }
    }
  })

  it('a mixed gate set matches exactProbs across the promotion boundary', () => {
    for (const n of [3, 5, 7]) {
      let k = new Circuit(n)
      for (let q = 0; q < n; q++) k = k.h(q).t(q)
      for (let q = 0; q < n - 1; q++) k = k.cnot(q, q + 1)
      k = k.ry(0.7, 0).cz(0, n - 1)
      if (n >= 3) k = k.ccx(0, 1, 2).cswap(2, 0, 1).csrn(0, 1)
      const exact = k.exactProbs()
      const probs = k.dm().probabilities()
      for (const [bits, p] of Object.entries(exact)) {
        if (p < 1e-12) continue
        expect(Math.abs(probs[bits]! - p), `n=${n} outcome ${bits}`).toBeLessThan(1e-11)
      }
    }
  })

  it('a pure state has purity 1 on both paths', () => {
    expect(new Circuit(6).x(0).x(3).dm().purity()).toBeCloseTo(1, 12)   // sparse
    expect(densifying(6).dm().purity()).toBeCloseTo(1, 12)              // dense
  })
})

describe('density matrix — noise physics on the dense path', () => {
  it('depolarizing reduces purity below 1', () => {
    const pure = densifying(5).dm().purity()
    const noisy = densifying(5).dm({ noise: { p1: 0.05 } }).purity()
    expect(pure).toBeCloseTo(1, 12)
    expect(noisy).toBeLessThan(0.99)
    expect(noisy).toBeGreaterThan(0)
  })

  it('probabilities still sum to 1 under noise', () => {
    for (const noise of [{ p1: 0.02 }, { p2: 0.05 }, { gamma: 0.1 }, { lambda: 0.1 }]) {
      let k = new Circuit(5)
      for (let q = 0; q < 5; q++) k = k.h(q).t(q)
      for (let q = 0; q < 4; q++) k = k.cnot(q, q + 1)
      const total = Object.values(k.dm({ noise }).probabilities()).reduce((a, b) => a + b, 0)
      expect(Math.abs(total - 1), JSON.stringify(noise)).toBeLessThan(1e-10)
    }
  })

  it('full amplitude damping collapses every qubit to |0⟩', () => {
    let k = new Circuit(4)
    for (let q = 0; q < 4; q++) k = k.h(q)
    const probs = k.dm({ noise: { gamma: 1 } }).probabilities()
    expect(probs['0000']).toBeCloseTo(1, 10)
  })

  it('entropy is exact for a diagonal rho read off the dense path', () => {
    // p1=0.75 fully depolarizes each qubit, so rho is diagonal and maximally
    // mixed: S = n bits. A diagonal matrix needs no Jacobi rotations, so this
    // isolates the dense read-out from the eigensolver.
    expect(densifying(4).dm({ noise: { p1: 0.75 } }).entropy()).toBeCloseTo(4, 8)
  })

  it('blochAngles reads the dense representation correctly', () => {
    // H on every qubit puts each on the equator: theta = pi/2, phi = 0.
    const d = densifying(5).dm()
    for (let q = 0; q < 5; q++) {
      const { theta, phi } = d.blochAngles(q)
      expect(theta).toBeCloseTo(Math.PI / 2, 8)
      expect(Math.abs(phi)).toBeLessThan(1e-8)
    }
  })
})

describe('density matrix — scale', () => {
  it('n=12 completes instead of exhausting the heap', () => {
    // Previously this densified a Map<bigint, Complex> to 4^12 boxed entries and
    // died at the 4 GB heap limit. Dense it is 16·4^12 = 256 MiB of f64.
    expect(MAX_DENSE_DM_QUBITS).toBeGreaterThanOrEqual(12)
    const probs = densifying(12).dm().probabilities()
    expect(Object.keys(probs)).toHaveLength(4096)
    for (const p of Object.values(probs)) expect(Math.abs(p - 1 / 4096)).toBeLessThan(1e-12)
  }, 120_000)
})

describe('von Neumann entropy — Jacobi eigensolver', () => {
  /**
   * Regression cover for a broken complex Givens rotation.
   *
   * The rotation used −w in its lower-left entry where unitarity requires
   * −conj(w). A single rotation still annihilates its target, so a diagonal rho
   * (no rotations) and a Bell rho (one rotation) both came out exact — but the
   * error compounded across a sweep, and any state whose rho is fully dense
   * returned nonsense: |+⟩⊗⁴ gave S = 2.5676 instead of 0.
   */

  it('a pure state has zero entropy however dense its rho is', () => {
    for (let n = 1; n <= 6; n++) {
      let k = new Circuit(n)
      for (let q = 0; q < n; q++) k = k.h(q)   // rho is fully dense: all 4^n entries non-zero
      expect(k.dm().entropy(), `n=${n}`).toBeCloseTo(0, 9)
    }
  })

  it('a maximally mixed state has entropy n', () => {
    for (let n = 1; n <= 6; n++) {
      let k = new Circuit(n)
      for (let q = 0; q < n; q++) k = k.h(q)
      expect(k.dm({ noise: { p1: 0.75 } }).entropy(), `n=${n}`).toBeCloseTo(n, 8)
    }
  })

  it('entropy is additive over independent subsystems', () => {
    // S(rho_A ⊗ rho_B) = S(rho_A) + S(rho_B). Independent per-qubit depolarizing
    // makes rho a product, so n identical qubits must give exactly n·S(single).
    for (const p of [0.1, 0.3, 0.6]) {
      const single = new Circuit(1).h(0).dm({ noise: { p1: p } }).entropy()
      for (const n of [2, 3, 4]) {
        let k = new Circuit(n)
        for (let q = 0; q < n; q++) k = k.h(q)
        expect(k.dm({ noise: { p1: p } }).entropy(), `p=${p} n=${n}`).toBeCloseTo(n * single, 9)
      }
    }
  })

  it('entropy stays within [0, n] and rises with the noise rate', () => {
    let prev = -1
    for (const p1 of [0, 0.05, 0.2, 0.5, 0.75]) {
      const s = densifying(4).dm({ noise: { p1 } }).entropy()
      expect(s, `p1=${p1}`).toBeGreaterThanOrEqual(-1e-9)
      expect(s, `p1=${p1}`).toBeLessThanOrEqual(4 + 1e-9)
      expect(s, `p1=${p1} should exceed p1 below it`).toBeGreaterThan(prev)
      prev = s
    }
  })

  it('entropy is 0 after full amplitude damping — the state is pure again', () => {
    expect(densifying(4).dm({ noise: { gamma: 1 } }).entropy()).toBeCloseTo(0, 9)
  })

  it('a Bell state is pure, and entangled states still read 0', () => {
    expect(new Circuit(2).h(0).cnot(0, 1).dm().entropy()).toBeCloseTo(0, 10)
    let ghz = new Circuit(5).h(0)
    for (let i = 0; i < 4; i++) ghz = ghz.cnot(i, i + 1)
    expect(ghz.dm().entropy()).toBeCloseTo(0, 9)
  })

  it('entropy is consistent with purity for a single noisy qubit', () => {
    // For a qubit, purity Tr(rho^2) fixes the eigenvalues: lambda = (1 +/- sqrt(2P-1))/2.
    for (const p1 of [0.1, 0.4, 0.75]) {
      const d = new Circuit(1).h(0).dm({ noise: { p1 } })
      const P = d.purity()
      const r = Math.sqrt(Math.max(0, 2 * P - 1))
      const l0 = (1 + r) / 2, l1 = (1 - r) / 2
      const want = -(l0 > 0 ? l0 * Math.log2(l0) : 0) - (l1 > 0 ? l1 * Math.log2(l1) : 0)
      expect(d.entropy(), `p1=${p1}`).toBeCloseTo(want, 9)
    }
  })
})

describe('von Neumann entropy — larger systems', () => {
  // Householder + QL replaced cyclic Jacobi here. Jacobi cost O(dim³) per sweep
  // and needed roughly ten of them, which put n=9 at 9.4s and n=10 out of reach;
  // these sizes are only testable because the reduction now runs once.

  it('stays additive out to n=8', () => {
    const single = new Circuit(1).h(0).dm({ noise: { p1: 0.2 } }).entropy()
    for (const n of [6, 7, 8]) {
      let k = new Circuit(n)
      for (let q = 0; q < n; q++) k = k.h(q)
      expect(k.dm({ noise: { p1: 0.2 } }).entropy(), `n=${n}`).toBeCloseTo(n * single, 8)
    }
  }, 60_000)

  it('eigenvalue-derived quantities stay consistent with purity at n=7', () => {
    // Sum of eigenvalues is Tr(rho) = 1 and sum of squares is Tr(rho^2) = purity.
    // Entropy is bounded below by the Renyi-2 entropy -log2(purity), and above by
    // log2(dim); both are independent of the eigensolver's internals.
    for (const p1 of [0.05, 0.25]) {
      let k = new Circuit(7)
      for (let q = 0; q < 7; q++) k = k.h(q).t(q)
      for (let q = 0; q < 6; q++) k = k.cnot(q, q + 1)
      const d = k.dm({ noise: { p1 } })
      const S = d.entropy()
      const renyi2 = -Math.log2(d.purity())
      expect(S, `p1=${p1} vs Renyi-2`).toBeGreaterThanOrEqual(renyi2 - 1e-9)
      expect(S, `p1=${p1} vs log2(dim)`).toBeLessThanOrEqual(7 + 1e-9)
    }
  }, 60_000)

  it('a pure entangled state at n=8 still reads 0', () => {
    let ghz = new Circuit(8).h(0)
    for (let i = 0; i < 7; i++) ghz = ghz.cnot(i, i + 1)
    expect(ghz.dm().entropy()).toBeCloseTo(0, 8)
  }, 60_000)
})
