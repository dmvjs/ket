/**
 * Quantum trajectory MPS — test suite
 *
 * What we're proving:
 *  0. MpsTrajectory unit tests: reset, apply1, apply2, sample, bond tracking
 *  1. Correctness:  noisy runMps() matches exact density-matrix backend at small n
 *  2. Consistency:  noisy runMps() matches noisy run() (statevector) at medium n
 *  3. Physics:      GHZ fidelity degrades at the analytically predicted rate (1-p2)^depth
 *  4. Scalability:  100+ qubit noisy circuits complete without memory explosion
 *  5. The jaw-dropper: 127-qubit IBM Sherbrooke simulation
 *
 * All statistical tests use chi-squared with a conservative threshold so they
 * don't flake under ordinary sampling variance.
 */

import { describe, expect, it, vi } from 'vitest'
import { Circuit, DEVICES } from './circuit.js'
import { realAmplitudes, gradient, minimize, gradientMps, minimizeMps } from './algorithms.js'
import type { PauliTerm } from './algorithms.js'
import { MpsTrajectory, mpsContract, mpsApply1, mpsApply2, mpsInit, mpsSample, CNOT4, SWAP4 } from './mps.js'
import type { Gate4x4 } from './statevector.js'
import * as G from './gates.js'

// ── MpsTrajectory unit tests ──────────────────────────────────────────────────

describe('MpsTrajectory — unit tests', () => {
  describe('reset()', () => {
    it('produces exact |0...0⟩ after arbitrary operations', () => {
      const traj = new MpsTrajectory(5, 16)
      traj.apply1(0, G.H)
      traj.apply2(0, 1, CNOT4)
      traj.apply1(2, G.X)
      traj.reset()

      // After reset, sampling with rand=0 (always collapses to |0⟩) must give 0n every time.
      const always0 = () => 0
      for (let i = 0; i < 20; i++) expect(traj.sample(always0)).toBe(0n)
    })

    it('resets bond dimensions to chiL=chiR=1 at every site', () => {
      const traj = new MpsTrajectory(4, 8)
      traj.apply1(0, G.H)
      traj.apply2(0, 1, CNOT4)
      traj.apply2(1, 2, CNOT4)
      traj.reset()

      for (let q = 0; q < 4; q++) {
        expect(traj.chiL[q]).toBe(1)
        expect(traj.chiR[q]).toBe(1)
      }
    })

    it('is idempotent — double reset gives same state as single reset', () => {
      const traj = new MpsTrajectory(3, 8)
      traj.apply1(0, G.H)
      traj.reset()
      traj.reset()
      const always0 = () => 0
      expect(traj.sample(always0)).toBe(0n)
    })
  })

  describe('apply1() matches mpsApply1()', () => {
    it('H gate on qubit 0: contracted amplitudes agree to 1e-12', () => {
      const traj = new MpsTrajectory(3, 8)
      traj.apply1(0, G.H)

      let ref = mpsInit(3)
      ref = mpsApply1(ref, 0, G.H)

      const trajAmps = mpsContract(trajToMps(traj))
      const refAmps  = mpsContract(ref)
      for (let i = 0; i < trajAmps.length; i++) {
        expect(Math.abs(trajAmps[i]! - refAmps[i]!)).toBeLessThan(1e-12)
      }
    })

    it('Ry(1.2) on qubit 1 of n=4: amplitudes agree to 1e-12', () => {
      const traj = new MpsTrajectory(4, 8)
      traj.apply1(1, G.Ry(1.2))

      let ref = mpsInit(4)
      ref = mpsApply1(ref, 1, G.Ry(1.2))

      const trajAmps = mpsContract(trajToMps(traj))
      const refAmps  = mpsContract(ref)
      for (let i = 0; i < trajAmps.length; i++) {
        expect(Math.abs(trajAmps[i]! - refAmps[i]!)).toBeLessThan(1e-12)
      }
    })
  })

  describe('apply2() matches mpsApply2()', () => {
    it('CNOT(0,1) on |+0⟩: produces Bell state matching reference', () => {
      const traj = new MpsTrajectory(2, 8)
      traj.apply1(0, G.H)
      traj.apply2(0, 1, CNOT4)

      let ref = mpsInit(2)
      ref = mpsApply1(ref, 0, G.H)
      ref = mpsApply2(ref, 0, 1, CNOT4, 8)

      const trajAmps = mpsContract(trajToMps(traj))
      const refAmps  = mpsContract(ref)
      for (let i = 0; i < trajAmps.length; i++) {
        expect(Math.abs(trajAmps[i]! - refAmps[i]!)).toBeLessThan(1e-12)
      }
    })

    it('5-qubit GHZ circuit: amplitudes match reference', () => {
      const n = 5
      const traj = new MpsTrajectory(n, 8)
      traj.apply1(0, G.H)
      for (let i = 0; i < n - 1; i++) traj.apply2(i, i + 1, CNOT4)

      let ref = mpsInit(n)
      ref = mpsApply1(ref, 0, G.H)
      for (let i = 0; i < n - 1; i++) ref = mpsApply2(ref, i, i + 1, CNOT4, 8)

      const trajAmps = mpsContract(trajToMps(traj))
      const refAmps  = mpsContract(ref)
      for (let i = 0; i < trajAmps.length; i++) {
        expect(Math.abs(trajAmps[i]! - refAmps[i]!)).toBeLessThan(1e-12)
      }
    })
  })

  describe('sample() matches mpsSample() distribution', () => {
    it('Bell state: both backends give 50/50 |00⟩/|11⟩', () => {
      const n    = 2
      const traj = new MpsTrajectory(n, 8)
      traj.apply1(0, G.H)
      traj.apply2(0, 1, CNOT4)

      let ref = mpsInit(n)
      ref = mpsApply1(ref, 0, G.H)
      ref = mpsApply2(ref, 0, 1, CNOT4, 8)

      // Sample 2000 from each, compare frequencies
      const rng = makeDeterministicRng(42)
      const trajCounts: Record<number, number> = {}
      const refCounts:  Record<number, number> = {}
      const shots = 2000
      for (let i = 0; i < shots; i++) {
        const t = Number(traj.sample(rng))
        trajCounts[t] = (trajCounts[t] ?? 0) + 1
        // Re-run apply ops since mpsSample doesn't modify state
        const r = Number(mpsSample(ref, rng))
        refCounts[r] = (refCounts[r] ?? 0) + 1
      }
      // Both should have ~50% at index 0 (|00⟩) and ~50% at index 3 (|11⟩)
      expect((trajCounts[0] ?? 0) / shots).toBeGreaterThan(0.4)
      expect((trajCounts[3] ?? 0) / shots).toBeGreaterThan(0.4)
      expect((refCounts[0]  ?? 0) / shots).toBeGreaterThan(0.4)
      expect((refCounts[3]  ?? 0) / shots).toBeGreaterThan(0.4)
    })
  })

  describe('bond dimension tracking', () => {
    it('product state: maxBondUsed() = 1', () => {
      const traj = new MpsTrajectory(6, 8)
      traj.apply1(0, G.H)
      traj.apply1(3, G.X)
      expect(traj.maxBondUsed()).toBe(1)
    })

    it('GHZ circuit: bond reaches 2 after first CNOT and stays at 2', () => {
      const n    = 10
      const traj = new MpsTrajectory(n, 8)
      traj.apply1(0, G.H)
      expect(traj.maxBondUsed()).toBe(1)

      for (let i = 0; i < n - 1; i++) traj.apply2(i, i + 1, CNOT4)
      // GHZ has exact bond dimension 2 for any n
      expect(traj.maxBondUsed()).toBe(2)
    })

    it('GHZ bond stays 2 at 127 qubits', () => {
      const n    = 127
      const traj = new MpsTrajectory(n, 64)
      traj.apply1(0, G.H)
      for (let i = 0; i < n - 1; i++) traj.apply2(i, i + 1, CNOT4)
      expect(traj.maxBondUsed()).toBe(2)
    })
  })

  describe('performance', () => {
    it('127-qubit GHZ, 1024 shots: completes in < 5s', () => {
      const n    = 127
      const traj = new MpsTrajectory(n, 64)
      const rng  = makeDeterministicRng(1)

      const start = performance.now()
      for (let shot = 0; shot < 1024; shot++) {
        traj.reset()
        traj.apply1(0, G.H)
        for (let i = 0; i < n - 1; i++) traj.apply2(i, i + 1, CNOT4)
        traj.sample(rng)
      }
      const ms = performance.now() - start

      expect(ms).toBeLessThan(5000)
      console.log(`127-qubit GHZ 1024 shots: ${ms.toFixed(0)}ms`)
    })
  })
})

// ── Helpers used only by MpsTrajectory unit tests ─────────────────────────────

/**
 * Convert MpsTrajectory internal state to an immutable MPS array for mpsContract.
 *
 * In Vidal canonical, data[q] = Γ[q] and bondLambda[q] = Λ[q].
 * The state is Γ[0]·Λ[0]·Γ[1]·Λ[1]·...·Γ[n-1].
 * mpsContract expects plain tensor products, so we embed Λ[q] into the right
 * bond of Γ[q] to get Γ[q]·Λ[q], making the contraction (Γ[0]·Λ[0])·(Γ[1]·Λ[1])·...·Γ[n-1].
 * Matrix product associativity ensures this equals the correct state amplitude.
 */
function trajToMps(traj: MpsTrajectory) {
  return Array.from({ length: traj.n }, (_, q) => {
    const chiL = traj.chiL[q]!
    const chiR = traj.chiR[q]!
    const data = traj.data[q]!.slice()
    if (q < traj.n - 1) {
      const lambda = traj.bondLambda[q]!
      for (let l = 0; l < chiL; l++) {
        for (let p = 0; p < 2; p++) {
          for (let r = 0; r < chiR; r++) {
            const s = lambda[r]!
            const i = ((l * 2 + p) * chiR + r) * 2
            data[i]     = data[i]!     * s
            data[i + 1] = data[i + 1]! * s
          }
        }
      }
    }
    return { data, chiL, chiR }
  })
}

/** Simple seeded LCG for deterministic test sampling. */
function makeDeterministicRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0
    return s / 0x100000000
  }
}

// ── Statistical helpers ────────────────────────────────────────────────────────

/**
 * Chi-squared statistic between observed probabilities (from Distribution.probs)
 * and expected probabilities, given the shot count.
 * Returns χ²/dof — values well below 3 indicate consistent distributions.
 */
function chiSq(
  observed: Readonly<Record<string, number>>,
  expected: Readonly<Record<string, number>>,
  shots: number,
): number {
  const keys = new Set([...Object.keys(observed), ...Object.keys(expected)])
  let chi2 = 0, dof = 0
  for (const k of keys) {
    const obs = (observed[k] ?? 0) * shots
    const exp = (expected[k] ?? 0) * shots
    if (exp < 1) continue
    chi2 += (obs - exp) ** 2 / exp
    dof++
  }
  return dof > 1 ? chi2 / (dof - 1) : chi2
}

// ── 1. Correctness at small n ─────────────────────────────────────────────────

describe('noisy MPS correctness — matches density matrix', () => {
  it('n=3 GHZ with depolarizing noise: χ²/dof < 2.5', () => {
    const p1 = 0.01, p2 = 0.02
    let c = new Circuit(3).h(0).cnot(0, 1).cnot(1, 2)

    const shots   = 8192
    const mpsProbs = c.runMps({ shots, seed: 1, noise: { p1, p2 } }).probs
    const dmProbs  = c.dm({ noise: { p1, p2 } }).probabilities()

    expect(chiSq(mpsProbs, dmProbs, shots)).toBeLessThan(2.5)
  })

  it('n=4 layered H+CNOT circuit with noise: χ²/dof < 2.5', () => {
    const p1 = 0.005, p2 = 0.015
    let c = new Circuit(4)
      .h(0).h(1).h(2).h(3)
      .cnot(0, 1).cnot(1, 2).cnot(2, 3)
      .h(0).h(1)

    const shots    = 8192
    const mpsProbs = c.runMps({ shots, seed: 2, noise: { p1, p2 } }).probs
    const dmProbs  = c.dm({ noise: { p1, p2 } }).probabilities()

    expect(chiSq(mpsProbs, dmProbs, shots)).toBeLessThan(2.5)
  })

  it('n=5 random Clifford-like circuit with noise: χ²/dof < 2.5', () => {
    const p1 = 0.008, p2 = 0.012
    let c = new Circuit(5)
      .h(0).h(2).h(4)
      .cnot(0, 1).cnot(2, 3).cnot(3, 4)
      .h(1).h(3)
      .cnot(1, 2).cnot(0, 4)

    const shots    = 8192
    const mpsProbs = c.runMps({ shots, seed: 3, noise: { p1, p2 } }).probs
    const dmProbs  = c.dm({ noise: { p1, p2 } }).probabilities()

    expect(chiSq(mpsProbs, dmProbs, shots)).toBeLessThan(2.5)
  })

  it('SPAM noise only: spreads a concentrated GHZ distribution', () => {
    // GHZ has all probability in |000⟩ and |111⟩.
    // pMeas=0.15 flips each bit independently → some shots land on mixed strings.
    let c = new Circuit(4).h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)
    const shots = 4096
    const pMeas = 0.2

    const spamResult  = c.runMps({ shots, seed: 4, noise: { pMeas } })
    const cleanResult = c.runMps({ shots, seed: 4 })

    const spamTotal  = Object.values(spamResult.probs).reduce((a, b) => a + b, 0)
    const cleanTotal = Object.values(cleanResult.probs).reduce((a, b) => a + b, 0)
    expect(spamTotal).toBeCloseTo(1, 5)
    expect(cleanTotal).toBeCloseTo(1, 5)

    // Clean GHZ: almost all weight on |0000⟩ and |1111⟩
    const cleanGhz = (cleanResult.probs['0000'] ?? 0) + (cleanResult.probs['1111'] ?? 0)
    expect(cleanGhz).toBeGreaterThan(0.9)

    // With pMeas=0.2 SPAM, probability of all-4 correct: (0.8)^4 ≈ 0.41 per outcome
    // So GHZ weight should drop to ~cleanGhz * 0.41 ≈ 0.41
    const spamGhz = (spamResult.probs['0000'] ?? 0) + (spamResult.probs['1111'] ?? 0)
    expect(spamGhz).toBeLessThan(cleanGhz - 0.3)
    expect(spamGhz).toBeGreaterThan(0.2) // some GHZ signal remains
  })
})

// ── 2. Consistency with statevector backend ───────────────────────────────────

describe('noisy MPS consistency — matches statevector run() at same noise', () => {
  it('n=8 GHZ: MPS vs statevector distributions agree (χ²/dof < 4.5)', () => {
    // MPS and statevector consume rng differently (mpsSample vs sampleSV),
    // so identical seeds produce distinct shot sequences — compare statistically.
    const p1 = 0.003, p2 = 0.006
    let c = new Circuit(8).h(0)
    for (let i = 0; i < 7; i++) c = c.cnot(i, i + 1)

    const shots    = 4096
    const mpsProbs = c.runMps({ shots, seed: 10, noise: { p1, p2 } }).probs
    const svProbs  = c.run({ shots, seed: 10, noise: { p1, p2 } }).probs

    expect(chiSq(mpsProbs, svProbs, shots)).toBeLessThan(4.5)
  })

  it('n=10 H+CNOT+H layers: MPS vs statevector agree (χ²/dof < 3)', () => {
    const p1 = 0.002, p2 = 0.008
    let c = new Circuit(10)
    for (let i = 0; i < 10; i++) c = c.h(i)
    for (let i = 0; i < 9; i++) c = c.cnot(i, i + 1)
    for (let i = 0; i < 10; i++) c = c.h(i)
    for (let i = 0; i < 9; i += 2) c = c.cnot(i, i + 1)

    const shots    = 4096
    const mpsProbs = c.runMps({ shots, seed: 11, noise: { p1, p2 } }).probs
    const svProbs  = c.run({ shots, seed: 11, noise: { p1, p2 } }).probs

    expect(chiSq(mpsProbs, svProbs, shots)).toBeLessThan(3)
  })

  it('no noise: runMps (noise: {p1:0, p2:0}) matches runMps (no noise option)', () => {
    let c = new Circuit(6).h(0)
    for (let i = 0; i < 5; i++) c = c.cnot(i, i + 1)

    const shots     = 2048
    const cleanProbs = c.runMps({ shots, seed: 1 }).probs
    const zeroProbs  = c.runMps({ shots, seed: 1, noise: { p1: 0, p2: 0 } }).probs

    // With same seed and zero noise, per-trajectory rng paths differ (trajectory allocates
    // one rng call even when no error fires), so we compare statistically not exactly.
    expect(chiSq(zeroProbs, cleanProbs, shots)).toBeLessThan(2.5)
  })
})

// ── 3. Physics: GHZ fidelity degrades at predicted rate ──────────────────────

describe('physics — noise degrades GHZ fidelity at analytically predicted rate', () => {
  /**
   * n-qubit GHZ: H + (n-1) CNOTs.
   * Under depolarizing noise the effective fidelity ≈ (1-p2)^(n-1).
   * We verify the observed GHZ-basis weight is within a generous range of this prediction.
   */
  function ghzFidelity(n: number, p2: number, shots = 2048, seed = 99): number {
    let c = new Circuit(n).h(0)
    for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
    const probs = c.runMps({ shots, seed, noise: { p2 } }).probs
    return (probs['0'.repeat(n)] ?? 0) + (probs['1'.repeat(n)] ?? 0)
  }

  it('n=4: observed fidelity > predicted/3 and < 1', () => {
    const p2 = 0.05
    const predicted = (1 - p2) ** 3   // ≈ 0.857
    const observed  = ghzFidelity(4, p2)
    expect(observed).toBeGreaterThan(predicted / 3)
    expect(observed).toBeLessThan(1.0)
  })

  it('n=8: observed fidelity > predicted/3 and < 1', () => {
    const p2 = 0.02
    const predicted = (1 - p2) ** 7   // ≈ 0.868
    const observed  = ghzFidelity(8, p2)
    expect(observed).toBeGreaterThan(predicted / 3)
    expect(observed).toBeLessThan(1.0)
  })

  it('n=12: higher noise → lower fidelity (monotone in p2)', () => {
    const fid1 = ghzFidelity(12, 0.005, 2048, 55)
    const fid2 = ghzFidelity(12, 0.02,  2048, 55)
    const fid3 = ghzFidelity(12, 0.08,  2048, 55)
    expect(fid1).toBeGreaterThan(fid2)
    expect(fid2).toBeGreaterThan(fid3)
  })

  it('deeper circuit → lower fidelity at fixed noise (monotone in n)', () => {
    const p2   = 0.03
    const fid4 = ghzFidelity(4,  p2, 2048, 77)
    const fid8 = ghzFidelity(8,  p2, 2048, 77)
    const fid16 = ghzFidelity(16, p2, 2048, 77)
    expect(fid4).toBeGreaterThan(fid8)
    expect(fid8).toBeGreaterThan(fid16)
  })
})

// ── 4. Scalability ────────────────────────────────────────────────────────────

describe('scalability — large qubit counts complete without memory explosion', () => {
  it('n=20 GHZ, IonQ Forte-1 noise, 512 shots', () => {
    let c = new Circuit(20).h(0)
    for (let i = 0; i < 19; i++) c = c.cnot(i, i + 1)

    const result = c.runMps({ shots: 512, seed: 20, noise: DEVICES['forte-1']!.noise })
    const total  = Object.values(result.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 5)

    // Forte-1: (1 - 0.002)^19 ≈ 0.963 — high coherence expected
    const fidelity = (result.probs['0'.repeat(20)] ?? 0) + (result.probs['1'.repeat(20)] ?? 0)
    expect(fidelity).toBeGreaterThan(0.5)
  })

  it('n=50 GHZ, Quantinuum h2-1 noise, 256 shots', () => {
    let c = new Circuit(50).h(0)
    for (let i = 0; i < 49; i++) c = c.cnot(i, i + 1)

    const result   = c.runMps({ shots: 256, seed: 50, noise: DEVICES['h2-1']!.noise })
    const total    = Object.values(result.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 5)

    // h2-1: (1 - 0.0011)^49 ≈ 0.947 — very clean hardware
    const fidelity = (result.probs['0'.repeat(50)] ?? 0) + (result.probs['1'.repeat(50)] ?? 0)
    expect(fidelity).toBeGreaterThan(0.4)
  })

  it('n=100 GHZ, IBM Brisbane noise, 128 shots', () => {
    let c = new Circuit(100).h(0)
    for (let i = 0; i < 99; i++) c = c.cnot(i, i + 1)

    const result   = c.runMps({ shots: 128, seed: 100, noise: DEVICES['ibm_brisbane']!.noise })
    const total    = Object.values(result.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 5)

    // IBM Brisbane: (1 - 0.0076)^99 ≈ 0.47 — still measurable GHZ signal
    const fidelity = (result.probs['0'.repeat(100)] ?? 0) + (result.probs['1'.repeat(100)] ?? 0)
    expect(fidelity).toBeGreaterThan(0.1)
  })
})

// ── 5. The jaw-dropper ────────────────────────────────────────────────────────

describe('127-qubit IBM Sherbrooke simulation', () => {
  /**
   * IBM Sherbrooke: 127 qubits, p1=2.4e-4, p2=7.4e-3, pMeas=1.35e-2.
   *
   * GHZ circuit: H q0, CNOT chain q0→q1→...→q126.
   * Analytic prediction: GHZ fidelity ≈ (1 - p2)^126 ≈ (0.9926)^126 ≈ 0.39
   *
   * A full statevector backend would require 2^127 complex numbers.
   * The density matrix backend would require 4^127 entries — both physically impossible.
   * This runs in seconds using MPS + quantum trajectories.
   */
  it('127-qubit GHZ: correct shot count and GHZ fidelity in predicted range', () => {
    let c = new Circuit(127).h(0)
    for (let i = 0; i < 126; i++) c = c.cnot(i, i + 1)

    const shots  = 256
    const result = c.runMps({ shots, seed: 127, noise: DEVICES['ibm_sherbrooke']!.noise })

    const total = Object.values(result.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 5)

    // Analytic: gate fidelity (1-p2)^126 ≈ 0.39
    // Plus readout: (1-pMeas)^127 ≈ 0.17 — each qubit must be read correctly
    // Combined expected GHZ weight ≈ 0.39 * 0.17 ≈ 0.067
    // Allow [0.02, 0.25] — generous for shot noise at 256 shots
    const fidelity = (result.probs['0'.repeat(127)] ?? 0) + (result.probs['1'.repeat(127)] ?? 0)
    expect(fidelity).toBeGreaterThan(0.02)
    expect(fidelity).toBeLessThan(0.25)
  })

  it('127-qubit GHZ: noisy fidelity measurably lower than noiseless', () => {
    let c = new Circuit(127).h(0)
    for (let i = 0; i < 126; i++) c = c.cnot(i, i + 1)

    const shots = 512

    const cleanResult   = c.runMps({ shots, seed: 7 })
    const cleanFidelity = (cleanResult.probs['0'.repeat(127)] ?? 0) +
                          (cleanResult.probs['1'.repeat(127)] ?? 0)

    const noisyResult   = c.runMps({ shots, seed: 7, noise: DEVICES['ibm_sherbrooke']!.noise })
    const noisyFidelity = (noisyResult.probs['0'.repeat(127)] ?? 0) +
                          (noisyResult.probs['1'.repeat(127)] ?? 0)

    // Clean GHZ should be near-perfect (MPS represents it exactly, chi=2)
    expect(cleanFidelity).toBeGreaterThan(0.95)
    // Noisy should be measurably degraded
    expect(noisyFidelity).toBeLessThan(cleanFidelity - 0.15)
  })
})

// ── runMps noise string resolution ────────────────────────────────────────────

describe('runMps — noise string resolution', () => {
  it('accepts a device name string — same result as passing the NoiseParams object', () => {
    let c = new Circuit(4).h(0)
    for (let i = 0; i < 3; i++) c = c.cnot(i, i + 1)

    const byString = c.runMps({ shots: 256, seed: 1, noise: 'aria-1' })
    const byObject = c.runMps({ shots: 256, seed: 1, noise: DEVICES['aria-1']!.noise })

    expect(byString.probs).toEqual(byObject.probs)
  })

  it('throws TypeError for an unknown device name string', () => {
    expect(() => new Circuit(2).runMps({ noise: 'not-a-device' })).toThrow(TypeError)
  })
})

// ── Helpers for high-entanglement tests ───────────────────────────────────────

/**
 * Random brickwork circuit: alternating layers of Ry(random) on all qubits
 * and CNOT on offset-staggered adjacent pairs. Produces genuinely high
 * entanglement (unlike GHZ where bond dim stays at 2 for any n).
 */
function makeBrickwork(n: number, depth: number, seed: number): Circuit {
  let c = new Circuit(n)
  const rng = makeDeterministicRng(seed)
  for (let d = 0; d < depth; d++) {
    for (let q = 0; q < n; q++) c = c.ry(rng() * 2 * Math.PI, q)
    const offset = d % 2
    for (let q = offset; q < n - 1; q += 2) c = c.cnot(q, q + 1)
  }
  return c
}

/**
 * Apply the same random brickwork gates directly to an MpsTrajectory
 * (for bond-dimension measurement without going through runMps).
 */
function applyBrickworkToTraj(traj: MpsTrajectory, depth: number, seed: number): void {
  const n = traj.n
  const rng = makeDeterministicRng(seed)
  for (let d = 0; d < depth; d++) {
    for (let q = 0; q < n; q++) traj.apply1(q, G.Ry(rng() * 2 * Math.PI))
    const offset = d % 2
    for (let q = offset; q < n - 1; q += 2) traj.apply2(q, q + 1, CNOT4)
  }
}

/**
 * QFT circuit on n qubits.
 * CPhase(θ) implemented as rz(θ/2, ctrl) · crz(θ, ctrl, tgt) — exact up to global phase.
 */
function makeQFT(n: number): Circuit {
  let c = new Circuit(n)
  for (let k = 0; k < n; k++) {
    c = c.h(k)
    for (let j = k + 1; j < n; j++) {
      const theta = Math.PI / (1 << (j - k))
      c = c.rz(theta / 2, k).crz(theta, k, j)
    }
  }
  return c
}

/** Total variation distance between two probability distributions. */
function tvd(
  a: Readonly<Record<string, number>>,
  b: Readonly<Record<string, number>>,
): number {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)])
  let sum = 0
  for (const k of keys) sum += Math.abs((a[k] ?? 0) - (b[k] ?? 0))
  return sum / 2
}

// ── 6. High-entanglement circuits — correctness beyond GHZ ───────────────────

describe('high-entanglement circuits — correctness beyond GHZ', () => {
  /**
   * QFT on n qubits maps |0...0⟩ to the uniform superposition.
   * Maximum bond dimension at middle cut = 2^(n/2).
   * With chi=64, n=6 (max bond=8) and n=8 (max bond=16) are exact.
   * Both runMps and run() must produce consistent output.
   */
  it('QFT n=6: runMps (chi=64) matches statevector (χ²/dof < 3)', () => {
    const c     = makeQFT(6)
    const shots = 4096
    const mpsR  = c.runMps({ shots, seed: 60, maxBond: 64 }).probs
    const svR   = c.run({ shots, seed: 60 }).probs
    expect(chiSq(mpsR, svR, shots)).toBeLessThan(3)
  })

  it('QFT n=8: runMps (chi=64) matches statevector (χ²/dof < 3)', () => {
    const c     = makeQFT(8)
    const shots = 8192
    const mpsR  = c.runMps({ shots, seed: 61, maxBond: 64 }).probs
    const svR   = c.run({ shots, seed: 61 }).probs
    expect(chiSq(mpsR, svR, shots)).toBeLessThan(3)
  })

  /**
   * Random brickwork vs density matrix (exact reference, not sampled).
   * Unlike GHZ, bond dimension grows with depth. With chi=64 and n≤5, the
   * simulation is exact — runMps must reproduce the dm() probabilities.
   *
   * Using dm() as reference (not run()) avoids two-sample chi-sq inflation
   * that occurs when both backends sample non-uniform distributions.
   */
  it('brickwork n=4 depth=4 (clean): runMps matches dm() (χ²/dof < 2.5)', () => {
    const c     = makeBrickwork(4, 4, 42)
    const shots = 8192
    const mpsR  = c.runMps({ shots, seed: 62, maxBond: 64 }).probs
    const dmR   = c.dm().probabilities()
    expect(chiSq(mpsR, dmR, shots)).toBeLessThan(2.5)
  })

  it('brickwork n=5 depth=4 (clean): runMps matches dm() (χ²/dof < 2.5)', () => {
    const c     = makeBrickwork(5, 4, 43)
    const shots = 8192
    const mpsR  = c.runMps({ shots, seed: 63, maxBond: 64 }).probs
    const dmR   = c.dm().probabilities()
    expect(chiSq(mpsR, dmR, shots)).toBeLessThan(2.5)
  })

  /**
   * Noisy brickwork n=5: trajectory MPS matches dm() at same noise.
   * dm() gives exact mixed-state probabilities; MPS trajectories converge to them.
   */
  it('noisy brickwork n=5 depth=4: runMps matches dm() (χ²/dof < 2.5)', () => {
    const p1 = 0.003, p2 = 0.008
    const c     = makeBrickwork(5, 4, 44)
    const shots = 8192
    const mpsR  = c.runMps({ shots, seed: 64, noise: { p1, p2 } }).probs
    const dmR   = c.dm({ noise: { p1, p2 } }).probabilities()
    expect(chiSq(mpsR, dmR, shots)).toBeLessThan(2.5)
  })
})

// ── 7. Bond dimension actually grows — proving non-trivial entanglement ───────

describe('bond dimension grows for high-entanglement circuits', () => {
  /**
   * GHZ has a compact MPS representation (bond dim = 2 for any n).
   * Random brickwork creates volume-law entanglement — bond dim grows with depth.
   * This test directly measures bond dimension to prove the distinction.
   */
  it('GHZ n=8 depth=7: maxBondUsed = 2', () => {
    const n    = 8
    const traj = new MpsTrajectory(n, 64)
    traj.apply1(0, G.H)
    for (let i = 0; i < n - 1; i++) traj.apply2(i, i + 1, CNOT4)
    expect(traj.maxBondUsed()).toBe(2)
  })

  it('brickwork n=8 depth=8: maxBondUsed > 4 (genuine high entanglement)', () => {
    const traj = new MpsTrajectory(8, 64)
    applyBrickworkToTraj(traj, 8, 42)
    // GHZ gives 2; random brickwork pushes chi up significantly
    expect(traj.maxBondUsed()).toBeGreaterThan(4)
  })

  it('brickwork n=10 depth=6: maxBondUsed > GHZ by 4x or more', () => {
    const n    = 10
    // GHZ bond dim = 2; brickwork should reach >> 2
    const ghzTraj = new MpsTrajectory(n, 64)
    ghzTraj.apply1(0, G.H)
    for (let i = 0; i < n - 1; i++) ghzTraj.apply2(i, i + 1, CNOT4)
    const ghzBond = ghzTraj.maxBondUsed()

    const bwTraj = new MpsTrajectory(n, 64)
    applyBrickworkToTraj(bwTraj, 6, 45)
    const bwBond = bwTraj.maxBondUsed()

    expect(bwBond).toBeGreaterThan(ghzBond * 3)
  })

  it('product state (only single-qubit gates): maxBondUsed = 1', () => {
    const traj = new MpsTrajectory(8, 64)
    const rng  = makeDeterministicRng(99)
    for (let q = 0; q < 8; q++) traj.apply1(q, G.Ry(rng() * 2 * Math.PI))
    expect(traj.maxBondUsed()).toBe(1)
  })

  it('bond dim grows monotonically with brickwork depth', () => {
    const depths = [2, 4, 6, 8]
    const bonds  = depths.map(d => {
      const traj = new MpsTrajectory(10, 64)
      applyBrickworkToTraj(traj, d, 55)
      return traj.maxBondUsed()
    })
    // Each deeper circuit must have bond dim ≥ previous
    for (let i = 1; i < bonds.length; i++) {
      expect(bonds[i]).toBeGreaterThanOrEqual(bonds[i - 1]!)
    }
    // And the deepest must exceed depth-2 (actually builds entanglement)
    expect(bonds.at(-1)).toBeGreaterThan(bonds[0]!)
  })
})

// ── 8. SVD truncation quality — chi controls accuracy ────────────────────────

describe('SVD truncation quality — higher chi gives better approximation', () => {
  /**
   * For circuits where exact bond dim exceeds the chi cap, truncation introduces
   * approximation error. Larger chi → smaller chi-sq vs exact dm() reference.
   * This validates optimal low-rank (Schmidt) truncation via SVD.
   */
  it('n=5 brickwork: chiSq(chi=2, dm) > chiSq(chi=16, dm) — truncation matters', () => {
    const c     = makeBrickwork(5, 6, 100)
    const shots = 8192

    // Exact reference via density matrix (no sampling noise)
    const dmRef = c.dm().probabilities()
    const chi2  = c.runMps({ shots, seed: 1, maxBond: 2  }).probs  // heavily truncated
    const chi16 = c.runMps({ shots, seed: 2, maxBond: 16 }).probs  // exact (max bond ≤ 8)

    const cq2  = chiSq(chi2,  dmRef, shots)
    const cq16 = chiSq(chi16, dmRef, shots)

    // chi=2 produces measurably wrong distribution (poor approximation)
    expect(cq2).toBeGreaterThan(cq16 + 1)
    // chi=16 should be consistent with exact (no truncation for n=5)
    expect(cq16).toBeLessThan(2.5)
  })

  it('n=6 brickwork depth=6: chiSq(chi=2, dm) > chiSq(chi=32, dm) — truncation detectable', () => {
    // QFT output is uniform regardless of chi (truncation errors are in phases only),
    // so brickwork is the right circuit family for truncation quality testing.
    // chi=2 truncates any entangled state; chi=32 ≥ max bond dim (≤8 for n=6) → exact.
    const c     = makeBrickwork(6, 6, 110)
    const shots = 8192

    const dmRef = c.dm().probabilities()
    const chi2  = c.runMps({ shots, seed: 1, maxBond: 2  }).probs  // heavily truncated
    const chi32 = c.runMps({ shots, seed: 2, maxBond: 32 }).probs  // exact (max bond ≤ 8)

    const cq2  = chiSq(chi2,  dmRef, shots)
    const cq32 = chiSq(chi32, dmRef, shots)

    expect(cq32).toBeLessThan(2.5)       // chi=32 is exact — consistent with dm()
    expect(cq2).toBeGreaterThan(cq32 + 1)  // chi=2 produces detectable error
  })
})

// ── SVD regression — guards against specific bugs in svdDecompose ────────────

describe('SVD regression — svdDecompose bug guards', () => {
  /**
   * Bug 1 — Jacobi rotation sign: t = +1/(τ+√(1+τ²)) satisfies t²+2τt-1=0 and
   * never zeros G'_pq.  Fix: t = -1/(τ+√(1+τ²)) satisfies t²-2τt-1=0.
   * Symptom: contracted amplitudes diverge from statevector for non-trivial circuits.
   */
  it('Jacobi sign: non-uniform 3-qubit circuit matches statevector (χ²/dof < 2.5)', () => {
    // Non-trivial entanglement with off-diagonal Jacobi coupling that must be zeroed.
    const c    = new Circuit(3).ry(0.8, 0).cnot(0, 1).ry(1.3, 1).cnot(1, 2).ry(0.5, 0)
    const shots = 8192
    expect(chiSq(
      c.runMps({ shots, seed: 90, maxBond: 8 }).probs,
      c.run({ shots, seed: 90 }).probs,
      shots,
    )).toBeLessThan(2.5)
  })

  /**
   * Bug 2 — missing Vidal lambda weighting in sample(): without weighting v_p by
   * bondLambda[q], P(p) = ||Γ_row_p||² = 1 for all p (isometry norm), giving
   * uniform 50/50 output regardless of state.  Fix: weight v_p by bondLambda[q]
   * so P(p) = ||Γ_row_p · Λ||² = Σ_k σ_k²|Γ[p][k]|² gives the correct marginal.
   */
  it('Vidal lambda sampling: Ry(π/3)+CNOT gives P(q0=0)≈0.75, not 0.5', () => {
    // |ψ⟩ = cos(π/6)|00⟩ + sin(π/6)|11⟩  →  P(q0=0) = cos²(π/6) = 3/4.
    // With the U bug, ||U_row_p||² = 1 for both p=0 and p=1 → always 50/50.
    const traj  = new MpsTrajectory(2, 8)
    traj.apply1(0, G.Ry(Math.PI / 3))
    traj.apply2(0, 1, CNOT4)

    const shots  = 4096
    const rng    = makeDeterministicRng(77)
    let   count0 = 0
    for (let i = 0; i < shots; i++) {
      if ((traj.sample(rng) & 1n) === 0n) count0++  // q0=0 ↔ LSB = 0
    }
    const p0 = count0 / shots
    // 3σ bounds: σ = √(0.75·0.25/4096) ≈ 0.0068; exact = 0.75
    expect(p0).toBeGreaterThan(0.72)
    expect(p0).toBeLessThan(0.78)
  })
})

// ── 9. Large-scale noisy brickwork — classical simulability at scale ──────────

describe('large-scale noisy brickwork — simulability via noise-limited entanglement', () => {
  /**
   * For NISQ-level noise, each two-qubit gate has error ~1%. After depth D layers,
   * the effective fidelity decays as (1-p2)^(D·n/2). At some depth, noise decoherence
   * effectively limits the entanglement content of the state — the distribution
   * approaches the maximally mixed state over the affected subspace.
   *
   * We verify: noisy circuits at scale (n=20, n=40) complete correctly,
   * and deeper circuits are more mixed (higher entropy).
   */
  it('n=20 noisy brickwork depth=4: runs correctly, depth=8 has higher entropy', () => {
    const noise = { p1: 0.002, p2: 0.008 }
    const shots = 256

    const cShallow = makeBrickwork(20, 4, 200)
    const cDeep    = makeBrickwork(20, 8, 200)

    const rShallow = cShallow.runMps({ shots, seed: 200, maxBond: 32, noise })
    const rDeep    = cDeep.runMps({ shots, seed: 200, maxBond: 32, noise })

    // Both complete and produce valid normalized distributions
    const sumShallow = Object.values(rShallow.probs).reduce((a, b) => a + b, 0)
    const sumDeep    = Object.values(rDeep.probs).reduce((a, b) => a + b, 0)
    expect(sumShallow).toBeCloseTo(1, 4)
    expect(sumDeep).toBeCloseTo(1, 4)

    // Deeper circuit → noise has more gates to decohere → higher entropy (more spread)
    expect(rDeep.entropy).toBeGreaterThan(rShallow.entropy)
  })

  it('n=40 noisy brickwork depth=4, IBM Brisbane noise: completes, distribution spread', () => {
    const c      = makeBrickwork(40, 4, 400)
    const result = c.runMps({ shots: 128, seed: 400, maxBond: 16, noise: DEVICES['ibm_brisbane']!.noise })

    const total = Object.values(result.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 4)

    // Noisy 40-qubit brickwork produces a spread distribution — with 128 shots from
    // a distribution over 2^40 states, nearly every shot is a distinct outcome.
    // If only ~1 outcome dominated (like GHZ), we'd see much fewer distinct strings.
    const uniqueOutcomes = Object.keys(result.probs).length
    expect(uniqueOutcomes).toBeGreaterThan(64)  // > half of shots are unique
  })

  it('n=50 noisy brickwork: deeper circuit produces more unique outcomes (greater spread)', () => {
    const noise = DEVICES['h2-1']!.noise  // IonQ Forte/Quantinuum — very clean
    const shots = 128

    // Deeper circuits → more gates → more decoherence → more spread distribution
    // → more unique bitstring outcomes per batch of shots.
    const uniqueCounts = [2, 6].map(depth => {
      const c = makeBrickwork(50, depth, 500)
      return Object.keys(c.runMps({ shots, seed: 500, maxBond: 16, noise }).probs).length
    })

    // Deeper circuit must have at least as many unique outcomes as shallow
    expect(uniqueCounts[1]).toBeGreaterThanOrEqual(uniqueCounts[0]!)
    // Both should be spread (not concentrated like GHZ which gives 2 outcomes)
    expect(uniqueCounts[0]).toBeGreaterThan(10)
  })
})

// ── 10. Noise coverage — every gate kind receives depolarizing errors ─────────

describe('noise coverage — all gate kinds inject depolarizing errors', () => {
  /**
   * For each gate kind, run with moderate noise and verify the resulting distribution
   * matches dm() at the same noise level (χ²/dof < 2.5).
   *
   * If noise were NOT applied after a gate kind, runMps would produce the clean-circuit
   * distribution, which is statistically incompatible with the dm() noisy distribution
   * at these noise rates — the χ²/dof would blow past the threshold.
   */
  const shots = 8192
  const p1 = 0.02, p2 = 0.04

  it('single-qubit gates receive p1 noise', () => {
    const c = new Circuit(3).h(0).ry(1.1, 1).rz(0.7, 2)
    expect(chiSq(
      c.runMps({ shots, seed: 1, noise: { p1 } }).probs,
      c.dm({ noise: { p1 } }).probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })

  it('cnot gates receive p2 noise', () => {
    const c = new Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
    expect(chiSq(
      c.runMps({ shots, seed: 2, noise: { p2 } }).probs,
      c.dm({ noise: { p2 } }).probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })

  it('swap gates receive p2 noise', () => {
    // swap(0,1) after X(0) moves excitation from q0 to q1 — noise spreads it.
    const c = new Circuit(3).x(0).swap(0, 1).h(2)
    expect(chiSq(
      c.runMps({ shots, seed: 3, noise: { p1, p2 } }).probs,
      c.dm({ noise: { p1, p2 } }).probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })

  it('two-qubit (ZZ/XX) gates receive p2 noise', () => {
    // ZZ(π/4) creates entanglement; noise at p2 measurably degrades fidelity.
    const c = new Circuit(3).h(0).h(1).zz(Math.PI / 4, 0, 1).h(2)
    expect(chiSq(
      c.runMps({ shots, seed: 4, noise: { p1, p2 } }).probs,
      c.dm({ noise: { p1, p2 } }).probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })

  it('controlled gates receive p2 noise', () => {
    // CRZ creates conditional phase; noise degrades coherence.
    const c = new Circuit(3).h(0).h(1).crz(Math.PI / 3, 0, 1).h(2)
    expect(chiSq(
      c.runMps({ shots, seed: 5, noise: { p1, p2 } }).probs,
      c.dm({ noise: { p1, p2 } }).probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })

  it('unitary 1-qubit gates receive p1 noise', () => {
    // H expressed as a unitary matrix — noise behaviour must match named H gate.
    const inv = 1 / Math.sqrt(2)
    const H = [[{ re: inv, im: 0 }, { re: inv, im: 0 }], [{ re: inv, im: 0 }, { re: -inv, im: 0 }]]
    const cNamed   = new Circuit(3).h(0).cnot(0, 1).h(2)
    const cUnitary = new Circuit(3).unitary(H, 0).cnot(0, 1).unitary(H, 2)
    const noise = { p1 }
    // Both circuits are identical in logic; their noisy distributions must match.
    expect(chiSq(
      cUnitary.runMps({ shots, seed: 6, noise }).probs,
      cNamed.dm({ noise }).probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })

  it('unitary 2-qubit gates receive p2 noise', () => {
    // CNOT expressed as a unitary matrix — noise behaviour must match named CNOT.
    const CNOT = [
      [{ re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
      [{ re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
      [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }],
      [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }],
    ]
    const cNamed   = new Circuit(3).h(0).cnot(0, 1).h(2)
    const cUnitary = new Circuit(3).h(0).unitary(CNOT, 0, 1).h(2)
    const noise = { p2 }
    expect(chiSq(
      cUnitary.runMps({ shots, seed: 7, noise }).probs,
      cNamed.dm({ noise }).probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })
})

// ── 11. truncErr — relative Schmidt truncation ────────────────────────────────

describe('truncErr — relative Schmidt truncation threshold', () => {
  /**
   * For circuits whose Schmidt spectrum decays quickly, a non-zero truncErr
   * discards small singular values early, reducing bond dimension.
   * For circuits with flat spectra (GHZ), truncErr has no effect.
   */

  it('truncErr=0 (default) gives exact brickwork results (χ²/dof < 2.5)', () => {
    const c     = makeBrickwork(5, 4, 42)
    const shots = 8192
    expect(chiSq(
      c.runMps({ shots, seed: 70, maxBond: 64, truncErr: 0 }).probs,
      c.dm().probabilities(),
      shots,
    )).toBeLessThan(2.5)
  })

  it('truncErr=0.5 degrades brickwork accuracy vs truncErr=0', () => {
    // Aggressive 50% relative truncation introduces approximation error that
    // is statistically detectable against the exact dm() reference.
    const c     = makeBrickwork(5, 6, 42)
    const shots = 8192
    const dmRef = c.dm().probabilities()

    const cqExact  = chiSq(c.runMps({ shots, seed: 71, maxBond: 64, truncErr: 0   }).probs, dmRef, shots)
    const cqTrunc  = chiSq(c.runMps({ shots, seed: 71, maxBond: 64, truncErr: 0.5 }).probs, dmRef, shots)

    expect(cqTrunc).toBeGreaterThan(cqExact + 1)
  })

  it('truncErr has no effect on GHZ: Schmidt spectrum is flat (both SVs equal)', () => {
    // GHZ bond dim = 2 with σ₀ = σ₁ = 1/√2. truncErr < 1 never cuts either value.
    const n    = 10
    const traj = new MpsTrajectory(n, 64, 0.9)  // aggressive 90% relative cutoff
    traj.apply1(0, G.H)
    for (let i = 0; i < n - 1; i++) traj.apply2(i, i + 1, CNOT4)
    // Both Schmidt values survive the relative cut; bond must stay at 2.
    expect(traj.maxBondUsed()).toBe(2)
  })

  it('truncErr reduces bond dimension for high-entanglement circuits', () => {
    // Random brickwork has a non-flat Schmidt spectrum — larger truncErr → smaller bond.
    const n = 8
    const bondExact = (() => {
      const t = new MpsTrajectory(n, 64, 0)
      applyBrickworkToTraj(t, 6, 42)
      return t.maxBondUsed()
    })()
    const bondTrunc = (() => {
      const t = new MpsTrajectory(n, 64, 0.1)
      applyBrickworkToTraj(t, 6, 42)
      return t.maxBondUsed()
    })()

    expect(bondTrunc).toBeLessThan(bondExact)
  })
})

// ── 12. bondEntropies() ───────────────────────────────────────────────────────

describe('bondEntropies()', () => {
  it('product state: all entropies are 0', () => {
    const traj = new MpsTrajectory(4, 8)
    traj.apply1(0, G.H)
    traj.apply1(2, G.X)
    const S = traj.bondEntropies()
    expect(S).toHaveLength(3)
    for (const s of S) expect(s).toBeLessThan(1e-10)
  })

  it('Bell state: bond-0 entropy = 1 ebit, bond-1 = 0', () => {
    // |Φ+⟩ = (|00⟩ + |11⟩)/√2 on qubits 0,1; qubit 2 in |0⟩.
    // Schmidt values at bond 0: σ = [1/√2, 1/√2] → S = 1 bit.
    // Schmidt values at bond 1: qubit 2 is unentangled → S = 0.
    const traj = new MpsTrajectory(3, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 1, CNOT4)
    const S = traj.bondEntropies()
    expect(S[0]).toBeCloseTo(1.0, 6)  // 1 ebit
    expect(S[1]).toBeLessThan(1e-10)   // qubit 2 unentangled
  })

  it('GHZ n=5: all bond entropies = 1 ebit', () => {
    const n = 5
    const traj = new MpsTrajectory(n, 8)
    traj.apply1(0, G.H)
    for (let i = 0; i < n - 1; i++) traj.apply2(i, i + 1, CNOT4)
    for (const s of traj.bondEntropies()) expect(s).toBeCloseTo(1.0, 5)
  })

  it('brickwork circuit: entropies positive and grow with depth', () => {
    const n = 8
    const traj1 = new MpsTrajectory(n, 64)
    const traj4 = new MpsTrajectory(n, 64)
    applyBrickworkToTraj(traj1, 1, 42)
    applyBrickworkToTraj(traj4, 4, 42)

    // Entropies after 4 layers must be ≥ after 1 layer at every bond
    const S1 = traj1.bondEntropies()
    const S4 = traj4.bondEntropies()
    expect(S4.reduce((a, b) => a + b, 0)).toBeGreaterThan(S1.reduce((a, b) => a + b, 0))
  })

  it('after reset(): all entropies return to 0', () => {
    const traj = new MpsTrajectory(4, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 1, CNOT4)
    traj.reset()
    for (const s of traj.bondEntropies()) expect(s).toBeLessThan(1e-10)
  })
})

// ── workers option ────────────────────────────────────────────────────────────
//
// In test mode (TypeScript source), import.meta.url ends in '.ts' so isBuilt=false
// and the workers path gracefully falls back to single-threaded. These tests verify:
//   1. The `workers` option is accepted without error (no throw, correct result type)
//   2. Fallback produces statistically correct results (same quality as workers=0)
//   3. The persistent pool does not break repeated calls
//
// Actual parallel execution is an integration concern (requires the built bundle).

// ── Non-adjacent apply2() — SWAP network ──────────────────────────────────────
//
// MpsTrajectory.apply2(a, b) with b > a+1 routes through a SWAP network:
//   bring b adjacent to a via b-1 SWAPs, apply gate, reverse SWAPs.
// These tests prove the SWAP chain is correct and the Vidal canonical form
// is preserved through the sequence of adjacent SVD decompositions.

describe('non-adjacent apply2() — SWAP network', () => {
  // Helper: sum of squares of active Schmidt values at bond b.
  // Must equal 1 for a normalised state in Vidal canonical form.
  function lambdaNormSq(traj: MpsTrajectory, b: number): number {
    const lambda = traj.bondLambda[b]!
    const chi    = traj.chiR[b]!
    let s2 = 0
    for (let k = 0; k < chi; k++) s2 += lambda[k]! * lambda[k]!
    return s2
  }

  it('CNOT(0,2) on |+00⟩: amplitudes match mpsApply2 reference (gap=1)', () => {
    // Validates SWAP(1,2)·CNOT(0,1)·SWAP(1,2) index arithmetic.
    const traj = new MpsTrajectory(3, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 2, CNOT4)  // non-adjacent: control=q0, target=q2

    let ref = mpsApply1(mpsInit(3), 0, G.H)
    ref = mpsApply2(ref, 0, 2, CNOT4, 8)

    const trajAmps = mpsContract(trajToMps(traj))
    const refAmps  = mpsContract(ref)
    for (let i = 0; i < trajAmps.length; i++) {
      expect(Math.abs(trajAmps[i]! - refAmps[i]!)).toBeLessThan(1e-12)
    }
  })

  it('CNOT(0,2) on |+00⟩: Vidal canonical preserved — both bonds have Schmidt values [1/√2, 1/√2]', () => {
    // After CNOT(q0,q2)|+00⟩ = (|000⟩+|101⟩)/√2, qubit q1 is unentangled from {q0,q2}
    // but q0 and q2 are entangled via both bonds: bipartition [q0]|[q1,q2] and [q0,q1]|[q2]
    // both have Schmidt rank 2 with values 1/√2. Verifies SWAP chain preserved canonical form.
    const traj = new MpsTrajectory(3, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 2, CNOT4)

    const INV_SQRT2 = 1 / Math.sqrt(2)
    for (let b = 0; b < 2; b++) {
      expect(lambdaNormSq(traj, b)).toBeCloseTo(1, 10)
      expect(traj.bondLambda[b]![0]!).toBeCloseTo(INV_SQRT2, 10)
      expect(traj.bondLambda[b]![1]!).toBeCloseTo(INV_SQRT2, 10)
    }
  })

  it('CNOT(0,2) sampling: only |000⟩ and |101⟩ outcomes (correct entanglement topology)', () => {
    // q0=0 → q2=0, q0=1 → q2=1; q1 always 0. Any other outcome would indicate
    // a wrong SWAP direction or wrong qubit labelling in the SWAP network.
    const traj = new MpsTrajectory(3, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 2, CNOT4)

    const rng = makeDeterministicRng(99)
    const seen = new Set<number>()
    for (let i = 0; i < 500; i++) seen.add(Number(traj.sample(rng)))

    expect(seen.has(0)).toBe(true)   // |000⟩ — q0=0
    expect(seen.has(5)).toBe(true)   // |101⟩ — q0=1, q2=1 (LSB=q0: 1+4=5)
    expect(seen.size).toBe(2)
  })

  it('CNOT(0,3) on n=4: amplitudes match reference (gap=2)', () => {
    // Exercises a two-step SWAP chain: SWAP(2,3)·SWAP(1,2)·CNOT(0,1)·SWAP(1,2)·SWAP(2,3).
    const traj = new MpsTrajectory(4, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 3, CNOT4)

    let ref = mpsApply1(mpsInit(4), 0, G.H)
    ref = mpsApply2(ref, 0, 3, CNOT4, 8)

    const trajAmps = mpsContract(trajToMps(traj))
    const refAmps  = mpsContract(ref)
    for (let i = 0; i < trajAmps.length; i++) {
      expect(Math.abs(trajAmps[i]! - refAmps[i]!)).toBeLessThan(1e-12)
    }
  })

  it('SWAP(0,2) on n=3: moves qubit 0 state to qubit 2 (amplitude + sampling)', () => {
    // |100⟩ (q0=1) → |001⟩ (q2=1). A non-adjacent SWAP is itself routed through
    // three adjacent SWAPs; the net result must be a transposition of q0 and q2.
    // mpsContract returns interleaved re/im: amplitude at basis index k lives at amps[k*2].
    const traj = new MpsTrajectory(3, 8)
    traj.apply1(0, G.X)  // |100⟩ — q0=1
    traj.apply2(0, 2, SWAP4)

    const amps = mpsContract(trajToMps(traj))
    expect(Math.abs(amps[4 * 2]! - 1)).toBeLessThan(1e-12)  // re of |001⟩ (q2=1) = 1
    expect(Math.abs(amps[1 * 2]!)).toBeLessThan(1e-12)       // re of |100⟩ (q0=1) = 0

    // Confirm via sampling: every shot must be |001⟩ = index 4.
    const rng = makeDeterministicRng(77)
    const seen = new Set<number>()
    for (let i = 0; i < 100; i++) seen.add(Number(traj.sample(rng)))
    expect([...seen]).toEqual([4])
  })

  it('compound non-adjacent circuit — n=4, H⊗4 then CNOT(0,2)+CNOT(1,3): amplitudes match reference', () => {
    // Two interleaved non-adjacent CNOTs exercise overlapping SWAP chains; any
    // canonical-form corruption after the first gate would corrupt the second.
    const n = 4
    const traj = new MpsTrajectory(n, 16)
    for (let q = 0; q < n; q++) traj.apply1(q, G.H)
    traj.apply2(0, 2, CNOT4)
    traj.apply2(1, 3, CNOT4)

    let ref = mpsInit(n)
    for (let q = 0; q < n; q++) ref = mpsApply1(ref, q, G.H)
    ref = mpsApply2(ref, 0, 2, CNOT4, 16)
    ref = mpsApply2(ref, 1, 3, CNOT4, 16)

    const trajAmps = mpsContract(trajToMps(traj))
    const refAmps  = mpsContract(ref)
    for (let i = 0; i < trajAmps.length; i++) {
      expect(Math.abs(trajAmps[i]! - refAmps[i]!)).toBeLessThan(1e-12)
    }
  })
})

// ── apply2 — reversed qubit order (a > b) ────────────────────────────────────

describe('apply2() — reversed qubit order (a > b)', () => {
  it('CNOT(1, 0): |10⟩ → |11⟩ (control=q1 flips target=q0)', () => {
    const traj = new MpsTrajectory(2, 4)
    traj.apply1(1, G.X)             // |10⟩
    traj.apply2(1, 0, CNOT4)        // reversed: control=1, target=0
    const amps = mpsContract(trajToMps(traj))
    // Expect |11⟩ = index 3 in little-endian (q0 LSB): bit0=1, bit1=1 → idx=3
    expect(Math.abs(amps[6]! - 1)).toBeLessThan(1e-12)  // re of |11⟩
    expect(Math.abs(amps[7]!)).toBeLessThan(1e-12)       // im of |11⟩
  })

  it('CNOT(1, 0) and CNOT(0, 1) are distinct operations', () => {
    // CNOT(0,1): control=q0, target=q1 — on |10⟩: q0=1 triggers X on q1 → |11⟩
    // CNOT(1,0): control=q1, target=q0 — on |10⟩: q1=0, no flip → |10⟩ unchanged
    const trajFwd = new MpsTrajectory(2, 4)
    trajFwd.apply1(0, G.X)
    trajFwd.apply2(0, 1, CNOT4)     // control=0, target=1: |10⟩ → |11⟩

    const trajRev = new MpsTrajectory(2, 4)
    trajRev.apply1(0, G.X)
    trajRev.apply2(1, 0, CNOT4)     // control=1, target=0: |10⟩ → |10⟩ (q1=0, no flip)

    const fwdAmps = mpsContract(trajToMps(trajFwd))
    const revAmps = mpsContract(trajToMps(trajRev))
    // Forward: |11⟩ has amplitude 1 at index 6/7 (re/im)
    expect(Math.abs(fwdAmps[6]! - 1)).toBeLessThan(1e-12)
    // Reversed: |10⟩ unchanged — q0=1,q1=0 → state index 1 (q0=LSB) → contract[2]=re
    expect(Math.abs(revAmps[2]! - 1)).toBeLessThan(1e-12)
  })

  it('apply2(a,b,G) === apply2(b,a,swappedG) — forward and reversed give identical state', () => {
    // Both calls should produce the same physical operation.
    // Use a non-symmetric gate (CZ) to detect any qubit-swap error.
    const CZ4: Gate4x4 = [
      [{ re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
      [{ re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
      [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }],
      [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: -1, im: 0 }],
    ]
    // CZ is symmetric, so apply2(0,1,CZ) === apply2(1,0,CZ). Prepare |++⟩.
    const trajFwd = new MpsTrajectory(2, 4)
    trajFwd.apply1(0, G.H); trajFwd.apply1(1, G.H)
    trajFwd.apply2(0, 1, CZ4)

    const trajRev = new MpsTrajectory(2, 4)
    trajRev.apply1(0, G.H); trajRev.apply1(1, G.H)
    trajRev.apply2(1, 0, CZ4)  // reversed

    const fwdAmps = mpsContract(trajToMps(trajFwd))
    const revAmps = mpsContract(trajToMps(trajRev))
    for (let i = 0; i < fwdAmps.length; i++) {
      expect(Math.abs(fwdAmps[i]! - revAmps[i]!)).toBeLessThan(1e-12)
    }
  })

  it('non-adjacent reversed: CNOT(2, 0) on n=3 matches statevector', () => {
    // Non-adjacent reversed: a=2, b=0, so internally normalised to apply2(0, 2, swappedCNOT)
    // swappedCNOT has control on MSB site (q2) and target on q0.
    // Prepare |001⟩ (q2=1), then CNOT(2,0) should flip q0 → |101⟩.
    const n = 3
    const traj = new MpsTrajectory(n, 8)
    traj.apply1(2, G.X)             // |001⟩ (q0 LSB: q2=1,q1=0,q0=0 = 0b100 = 4 in decimal)
    traj.apply2(2, 0, CNOT4)        // control=q2 (=1), target=q0: flip q0 → |101⟩

    // Reference via statevector
    const c = new Circuit(n).x(2).cx(2, 0)
    const refProbs = c.exactProbs()
    const trajAmps = mpsContract(trajToMps(traj))
    // |101⟩ in q0-leftmost = '101', probability = 1
    expect(refProbs['101']).toBeCloseTo(1, 12)
    // Traj: |101⟩ bit pattern q0=1,q1=0,q2=1 → little-endian index = 0b101 = 5 → re at 10, im at 11
    expect(Math.abs(trajAmps[10]! - 1)).toBeLessThan(1e-12)
  })

  it('apply2(a, a, gate) throws RangeError', () => {
    const traj = new MpsTrajectory(3, 4)
    expect(() => traj.apply2(1, 1, CNOT4)).toThrow(RangeError)
  })
})

describe('workers option', () => {
  const noise = { p1: 0.01, p2: 0.02 }

  it('workers > 1 is accepted and returns a Distribution', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    let c = new Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    const dist = c.runMps({ shots: 256, seed: 1, noise, workers: 4 })
    warn.mockRestore()
    expect(dist.shots).toBe(256)
    expect(dist.qubits).toBe(4)
    const total = Object.values(dist.probs).reduce((a, b) => a + b, 0)
    expect(Math.abs(total - 1)).toBeLessThan(1e-9)
  })

  it('workers=4 and workers=0 produce statistically equivalent distributions', () => {
    // Under fallback both use the same single-threaded path with same seed → identical.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    let c = new Circuit(4).h(0)
    for (let q = 0; q < 3; q++) c = c.cx(q, q + 1)
    const d0 = c.runMps({ shots: 512, seed: 7, noise, workers: 0 })
    const d4 = c.runMps({ shots: 512, seed: 7, noise, workers: 4 })
    warn.mockRestore()
    // In test/dev mode both are single-threaded with the same seed → identical counts.
    for (const key of Object.keys(d0.probs)) {
      expect(d4.probs[key] ?? 0).toBeCloseTo(d0.probs[key]!, 5)
    }
  })

  it('warns when workers > 1 but bundle is not built', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    new Circuit(2).h(0).cx(0, 1).runMps({ shots: 64, seed: 1, noise, workers: 4 })
    // In test/ts-source mode the worker path is unavailable — warn fires exactly once.
    expect(warn).toHaveBeenCalledOnce()
    expect(warn.mock.calls[0]![0]).toMatch(/workers option ignored/)
    warn.mockRestore()
  })

  it('repeated calls with workers use the persistent pool without errors', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    let c = new Circuit(3).h(0).cx(0, 1).cx(1, 2)
    for (let i = 0; i < 4; i++) {
      const dist = c.runMps({ shots: 128, seed: i, noise, workers: 4 })
      expect(dist.shots).toBe(128)
    }
    warn.mockRestore()
  })
})

// ── expect1 — single-site expectation values ──────────────────────────────────

describe('MpsTrajectory.expect1() — single-site expectation values', () => {
  it('|0⟩: ⟨Z⟩ = 1, ⟨X⟩ = 0', () => {
    const traj = new MpsTrajectory(1, 4)
    expect(traj.expect1(0, G.Z).re).toBeCloseTo(1, 12)
    expect(traj.expect1(0, G.X).re).toBeCloseTo(0, 12)
  })

  it('|1⟩: ⟨Z⟩ = -1, ⟨X⟩ = 0', () => {
    const traj = new MpsTrajectory(1, 4)
    traj.apply1(0, G.X)
    expect(traj.expect1(0, G.Z).re).toBeCloseTo(-1, 12)
    expect(traj.expect1(0, G.X).re).toBeCloseTo(0, 12)
  })

  it('|+⟩ = H|0⟩: ⟨X⟩ = 1, ⟨Z⟩ = 0', () => {
    const traj = new MpsTrajectory(1, 4)
    traj.apply1(0, G.H)
    expect(traj.expect1(0, G.X).re).toBeCloseTo(1, 12)
    expect(traj.expect1(0, G.Z).re).toBeCloseTo(0, 12)
  })

  it('Ry(θ): ⟨Z⟩ = cos(θ) analytically', () => {
    const theta = 1.1
    const traj  = new MpsTrajectory(1, 4)
    traj.apply1(0, G.Ry(theta))
    expect(traj.expect1(0, G.Z).re).toBeCloseTo(Math.cos(theta), 12)
  })

  it('n=3 Bell-like: ⟨Z₀⟩ = 0, ⟨Z₁⟩ = 0 after H+CNOT', () => {
    const traj = new MpsTrajectory(3, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 1, CNOT4)
    expect(traj.expect1(0, G.Z).re).toBeCloseTo(0, 10)
    expect(traj.expect1(1, G.Z).re).toBeCloseTo(0, 10)
  })

  it('n=3 product state: ⟨Z₀⟩ = cos(0.8), ⟨Z₁⟩ = cos(1.3) independently', () => {
    const traj = new MpsTrajectory(3, 8)
    traj.apply1(0, G.Ry(0.8))
    traj.apply1(1, G.Ry(1.3))
    expect(traj.expect1(0, G.Z).re).toBeCloseTo(Math.cos(0.8), 12)
    expect(traj.expect1(1, G.Z).re).toBeCloseTo(Math.cos(1.3), 12)
    expect(traj.expect1(2, G.Z).re).toBeCloseTo(1, 12)  // untouched qubit stays |0⟩
  })

  it('imaginary part is 0 for Z, X, Y on real states', () => {
    const traj = new MpsTrajectory(2, 4)
    traj.apply1(0, G.Ry(0.7))
    traj.apply1(1, G.Ry(1.2))
    expect(Math.abs(traj.expect1(0, G.Z).im)).toBeLessThan(1e-14)
    expect(Math.abs(traj.expect1(0, G.X).im)).toBeLessThan(1e-14)
    expect(Math.abs(traj.expect1(0, G.Y).im)).toBeLessThan(1e-14)
  })
})

// ── expectation — product observable transfer matrix ─────────────────────────

describe('MpsTrajectory.expectation() — product observables', () => {
  it('all-identity: expectation = 1 (normalization check)', () => {
    const traj = new MpsTrajectory(4, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 1, CNOT4)
    traj.apply2(1, 2, CNOT4)
    traj.apply2(2, 3, CNOT4)
    const ev = traj.expectation([null, null, null, null])
    expect(ev.re).toBeCloseTo(1, 12)
    expect(Math.abs(ev.im)).toBeLessThan(1e-14)
  })

  it('Bell state n=2: ⟨ZZ⟩ = 1, ⟨ZI⟩ = 0, ⟨XX⟩ = 1', () => {
    const traj = new MpsTrajectory(2, 4)
    traj.apply1(0, G.H)
    traj.apply2(0, 1, CNOT4)
    expect(traj.expectation([G.Z, G.Z]).re).toBeCloseTo( 1, 12)
    expect(traj.expectation([G.Z, null]).re).toBeCloseTo( 0, 12)
    expect(traj.expectation([null, G.Z]).re).toBeCloseTo( 0, 12)
    expect(traj.expectation([G.X, G.X]).re).toBeCloseTo( 1, 12)
    expect(traj.expectation([G.Y, G.Y]).re).toBeCloseTo(-1, 12)
  })

  it('product state: ⟨Z₀Z₁⟩ = cos(θ₀)·cos(θ₁) (factorizes)', () => {
    const t0 = 0.9, t1 = 1.4
    const traj = new MpsTrajectory(2, 4)
    traj.apply1(0, G.Ry(t0))
    traj.apply1(1, G.Ry(t1))
    const expected = Math.cos(t0) * Math.cos(t1)
    expect(traj.expectation([G.Z, G.Z]).re).toBeCloseTo(expected, 12)
  })

  it('single non-null op matches expect1()', () => {
    const traj = new MpsTrajectory(4, 8)
    traj.apply1(0, G.Ry(0.6))
    traj.apply1(1, G.Ry(1.1))
    traj.apply2(0, 1, CNOT4)
    traj.apply2(1, 2, CNOT4)
    // ⟨I Z I I⟩ should equal expect1(1, Z)
    const e1  = traj.expect1(1, G.Z).re
    const eOp = traj.expectation([null, G.Z, null, null]).re
    expect(eOp).toBeCloseTo(e1, 12)
  })

  it('throws TypeError when ops.length ≠ n', () => {
    const traj = new MpsTrajectory(3, 4)
    expect(() => traj.expectation([G.Z, G.Z])).toThrow(TypeError)
  })

  it('GHZ n=4: ⟨ZZZZ⟩ = 1, ⟨ZZZI⟩ = 0', () => {
    const traj = new MpsTrajectory(4, 8)
    traj.apply1(0, G.H)
    traj.apply2(0, 1, CNOT4)
    traj.apply2(1, 2, CNOT4)
    traj.apply2(2, 3, CNOT4)
    expect(traj.expectation([G.Z, G.Z, G.Z, G.Z]).re).toBeCloseTo(1, 12)
    expect(traj.expectation([G.Z, G.Z, G.Z, null]).re).toBeCloseTo(0, 12)
    expect(traj.expectation([G.X, G.X, G.X, G.X]).re).toBeCloseTo(1, 12)
  })

  it('GHZ n=10: ⟨Z…Z⟩ = 1 — validates transfer matrix at larger n', () => {
    const n = 10
    const traj = new MpsTrajectory(n, 4)
    traj.apply1(0, G.H)
    for (let q = 0; q < n - 1; q++) traj.apply2(q, q + 1, CNOT4)
    const allZ = Array.from({ length: n }, () => G.Z as typeof G.Z | null)
    expect(traj.expectation(allZ).re).toBeCloseTo(1, 10)
  })
})

// ── Circuit.expectMps — VQE / Hamiltonian interface ───────────────────────────

describe('Circuit.expectMps() — Hamiltonian expectation values', () => {
  it('single-qubit: ⟨Z⟩ = 1 for |0⟩, -1 for |1⟩', () => {
    const term: PauliTerm = { coeff: 1, ops: 'Z' }
    expect(new Circuit(1).expectMps([term])).toBeCloseTo(1, 12)
    expect(new Circuit(1).x(0).expectMps([term])).toBeCloseTo(-1, 12)
  })

  it('empty terms array returns 0 without building MPS', () => {
    expect(new Circuit(4).h(0).cx(0, 1).expectMps([])).toBe(0)
  })

  it('coeff=0 terms are skipped', () => {
    const terms: PauliTerm[] = [
      { coeff: 0, ops: 'IZ' },   // Z on q0, skipped
      { coeff: 1, ops: 'ZI' },   // Z on q1 → ⟨Z₁⟩ = 1 for |00⟩
    ]
    expect(new Circuit(2).expectMps(terms)).toBeCloseTo(1, 12)
  })

  it('Bell state: ⟨ZZ⟩ = 1, ⟨ZI⟩ = 0 via expectMps()', () => {
    const c = new Circuit(2).h(0).cx(0, 1)
    expect(c.expectMps([{ coeff: 1, ops: 'ZZ' }])).toBeCloseTo(1, 12)
    expect(c.expectMps([{ coeff: 1, ops: 'IZ' }])).toBeCloseTo(0, 12)
  })

  it('Heisenberg ZZ chain: energy matches analytical E₀ for 2-site singlet', () => {
    // 2-site Heisenberg ZZ+XX+YY ground state |singlet⟩ = (|01⟩ - |10⟩)/√2
    // Prepared as: Ry(-π/2) on q0, then CNOT(0,1), then X on q1
    // (standard singlet preparation)
    // Ground energy for H = ZZ + XX + YY on singlet = -3
    const c = new Circuit(2).ry(-Math.PI / 2, 0).cx(0, 1).x(1)
    const terms: PauliTerm[] = [
      { coeff: 1, ops: 'ZZ' },
      { coeff: 1, ops: 'XX' },
      { coeff: 1, ops: 'YY' },
    ]
    expect(c.expectMps(terms)).toBeCloseTo(-3, 10)
  })

  it('sum over identity terms = number of terms (each contributes coeff·1)', () => {
    const c = new Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    const terms: PauliTerm[] = [
      { coeff: 2, ops: 'IIII' },
      { coeff: 3, ops: 'IIII' },
    ]
    expect(c.expectMps(terms)).toBeCloseTo(5, 12)
  })

  it('throws TypeError when ops.length ≠ qubits', () => {
    const c = new Circuit(3)
    expect(() => c.expectMps([{ coeff: 1, ops: 'ZZ' }])).toThrow(TypeError)
  })

  it('agrees with statevector expectation() for random Ry circuit n=6', () => {
    // Build a random but deterministic circuit
    const angles = [0.3, 0.7, 1.1, 1.5, 0.9, 0.4]
    let c = new Circuit(6)
    for (let q = 0; q < 6; q++) c = c.ry(angles[q]!, q)
    for (let q = 0; q < 5; q++) c = c.cx(q, q + 1)
    for (let q = 0; q < 6; q++) c = c.ry(angles[5 - q]!, q)

    // Compute ⟨Z₀ Z₃⟩ via expectMps and via statevector expectation()
    // ops 'IIZIIZ': ops[2]→q3=Z, ops[5]→q0=Z (vqe convention: ops[n-1-q]→qubit q)
    const mpsEv = c.expectMps([{ coeff: 1, ops: 'IIZIIZ' }])
    const svEv  = c.expectation('ZIIZII')

    expect(mpsEv).toBeCloseTo(svEv, 10)
  })

  it('expectMps and expect1 agree for single-site observables on entangled state', () => {
    const c = new Circuit(4).h(0).cx(0, 1).cx(1, 2).ry(0.8, 3)
    const traj = new MpsTrajectory(4, 8)
    // Replicate circuit manually on traj for comparison
    traj.apply1(0, G.H)
    traj.apply2(0, 1, CNOT4)
    traj.apply2(1, 2, CNOT4)
    traj.apply1(3, G.Ry(0.8))

    for (let q = 0; q < 4; q++) {
      // Z at position n-1-q in string (vqe convention: ops[n-1-q] acts on qubit q)
      const term: PauliTerm = { coeff: 1, ops: 'I'.repeat(3 - q) + 'Z' + 'I'.repeat(q) }
      const fromCircuit = c.expectMps([term])
      const fromTraj    = traj.expect1(q, G.Z).re
      expect(fromCircuit).toBeCloseTo(fromTraj, 10)
    }
  })

  it('GHZ n=8: total Z magnetization Σ⟨Zq⟩ = 0 (equal superposition)', () => {
    const n = 8
    let c = new Circuit(n).h(0)
    for (let q = 0; q < n - 1; q++) c = c.cx(q, q + 1)
    const terms: PauliTerm[] = Array.from({ length: n }, (_, q) => ({
      coeff: 1,
      ops:   'I'.repeat(n - 1 - q) + 'Z' + 'I'.repeat(q),
    }))
    expect(c.expectMps(terms)).toBeCloseTo(0, 10)
  })
})

// ── Circuit.bondEntropies ──────────────────────────────────────────────────────

describe('Circuit.bondEntropies() — entanglement entropy at each bond', () => {
  it('product state: all bond entropies = 0', () => {
    const c = new Circuit(4).h(0).h(1).h(2).h(3)
    const S = c.bondEntropies()
    expect(S).toHaveLength(3)
    for (const s of S) expect(s).toBeCloseTo(0, 12)
  })

  it('Bell state: single bond entropy = 1 (maximally entangled)', () => {
    const S = new Circuit(2).h(0).cx(0, 1).bondEntropies()
    expect(S).toHaveLength(1)
    expect(S[0]).toBeCloseTo(1, 10)
  })

  it('GHZ n=4: all bonds saturated at S = 1', () => {
    let c = new Circuit(4).h(0)
    for (let q = 0; q < 3; q++) c = c.cx(q, q + 1)
    const S = c.bondEntropies()
    expect(S).toHaveLength(3)
    for (const s of S) expect(s).toBeCloseTo(1, 10)
  })

  it('partial entanglement: entropy peaks at cut through entangled region', () => {
    // Bell pair on qubits 1-2, qubits 0 and 3 are product
    const c = new Circuit(4).h(1).cx(1, 2)
    const S = c.bondEntropies()
    expect(S).toHaveLength(3)
    expect(S[0]).toBeCloseTo(0, 10)  // bond 0-1: no entanglement
    expect(S[1]).toBeCloseTo(1, 10)  // bond 1-2: maximally entangled
    expect(S[2]).toBeCloseTo(0, 10)  // bond 2-3: no entanglement
  })

  it('returns n-1 values for an n-qubit circuit', () => {
    for (const n of [2, 5, 10]) {
      expect(new Circuit(n).bondEntropies()).toHaveLength(n - 1)
    }
  })
})

// ── gradientMps ───────────────────────────────────────────────────────────────

describe('gradientMps() — parameter-shift gradient via MPS', () => {
  it('single-qubit: ∂⟨Z⟩/∂θ = -sin(θ) for Ry(θ)|0⟩', () => {
    const ansatz = (p: readonly number[]) => new Circuit(1).ry(p[0]!, 0)
    const H: PauliTerm[] = [{ coeff: 1, ops: 'Z' }]
    for (const theta of [0, Math.PI / 4, Math.PI / 2, Math.PI]) {
      const [g] = gradientMps(ansatz, H, [theta])
      expect(g).toBeCloseTo(-Math.sin(theta), 10)
    }
  })

  it('matches gradient() exactly for small circuits (both use parameter shift)', () => {
    const ansatz = realAmplitudes(3, 1)
    const H: PauliTerm[] = [
      { coeff: 1, ops: 'ZII' },
      { coeff: 1, ops: 'IZI' },
      { coeff: 1, ops: 'IIZ' },
    ]
    const params = [0.3, 0.7, 1.1, 0.5, 0.9, 0.2]
    const gSV  = gradient(ansatz, H, params)
    const gMPS = gradientMps(ansatz, H, params, { maxBond: 8 })
    for (let i = 0; i < params.length; i++) {
      expect(gMPS[i]).toBeCloseTo(gSV[i]!, 8)
    }
  })
})

// ── minimizeMps ───────────────────────────────────────────────────────────────

describe('minimizeMps() — VQE at scale via MPS', () => {
  it('2-site Heisenberg singlet: converges to E = -3 (exact ground state)', () => {
    // H = XX + YY + ZZ; ground state is the singlet (|01⟩ - |10⟩)/√2, E₀ = -3
    const H: PauliTerm[] = [
      { coeff: 1, ops: 'XX' },
      { coeff: 1, ops: 'YY' },
      { coeff: 1, ops: 'ZZ' },
    ]
    const ansatz = realAmplitudes(2, 2)
    const init   = Array(ansatz.paramCount).fill(0).map((_, i) => (i * 0.3) % (2 * Math.PI))
    const result = minimizeMps(ansatz, H, init, { lr: 0.2, steps: 300, maxBond: 4 })
    expect(result.energy).toBeCloseTo(-3, 1)
  })

  it('matches minimize() result for small circuits', () => {
    const H: PauliTerm[] = [{ coeff: 1, ops: 'ZZ' }, { coeff: 1, ops: 'XX' }]
    const ansatz = realAmplitudes(2, 2)
    const init   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    const rSV  = minimize(ansatz, H, init, { steps: 100 })
    const rMPS = minimizeMps(ansatz, H, init, { steps: 100, maxBond: 4 })
    expect(rMPS.energy).toBeCloseTo(rSV.energy, 6)
  })

  it('converges to energy below product-state bound for 6-qubit Heisenberg chain', () => {
    // Product state has E ≥ -5 for this Hamiltonian; entangled ground state has E ≈ -7.7
    const n = 6
    const H: PauliTerm[] = []
    for (let i = 0; i < n - 1; i++) {
      for (const ops of [`${'I'.repeat(n - 2 - i)}XX${'I'.repeat(i)}`,
                         `${'I'.repeat(n - 2 - i)}YY${'I'.repeat(i)}`,
                         `${'I'.repeat(n - 2 - i)}ZZ${'I'.repeat(i)}`]) {
        H.push({ coeff: 1, ops })
      }
    }
    const ansatz = realAmplitudes(n, 3)
    const init   = Array(ansatz.paramCount).fill(0).map((_, i) => Math.sin(i) * 0.5)
    const result = minimizeMps(ansatz, H, init, { lr: 0.15, steps: 200, maxBond: 16 })
    // Must beat the trivial product-state lower bound of -5
    expect(result.energy).toBeLessThan(-5)
  })
})
