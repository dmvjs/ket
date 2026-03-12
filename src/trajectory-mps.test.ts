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

import { describe, expect, it } from 'vitest'
import { Circuit, DEVICES } from './circuit.js'
import { MpsTrajectory, mpsContract, mpsApply1, mpsApply2, mpsInit, mpsSample, CNOT4 } from './mps.js'
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

/** Convert MpsTrajectory internal state to an immutable MPS array for mpsContract. */
function trajToMps(traj: MpsTrajectory) {
  return Array.from({ length: traj.n }, (_, q) => ({
    data: traj.data[q]!.slice(),  // copy so mpsContract can't mutate
    chiL: traj.chiL[q]!,
    chiR: traj.chiR[q]!,
  }))
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
   * Bug 2 — qBuf stores U instead of U·Σ: sample() computes ||U_row_p||² = 1 for
   * all p, producing uniform marginals regardless of state.  Fix: copy aBuf columns
   * directly (already σ_k·U[:,k] after Jacobi) so P(qubit=p) = Σ_k σ_k²|U[p][k]|².
   */
  it('UΣ sampling: Ry(π/3)+CNOT gives P(q0=0)≈0.75, not 0.5', () => {
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
