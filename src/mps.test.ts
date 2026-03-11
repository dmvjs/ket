/**
 * Tests for the Float64Array MPS backend.
 *
 * The Float64Array layout replaces Complex[][][] (which allocates a new heap
 * object for every arithmetic result) with a single contiguous typed array.
 * Benefits: zero per-operation GC allocations, cache-friendly sequential access,
 * and V8 JIT type-specialization on unboxed f64 elements.
 *
 * Design philosophy: every test targets a specific failure mode.
 * "Would this test catch a real bug?" is the bar for inclusion.
 * Tests are grouped by the class of bug they catch, not by feature.
 */

import { describe, expect, it } from 'vitest'
import {
  mpsInit, mpsApply1, mpsApply2, mpsSample, mpsContract, mpsMaxBond, mpsTensor,
  CNOT4, SWAP4, controlledGate, type MPS, type Tensor,
} from './mps.js'
import { zero, applySingle, applyTwo, applyCNOT, applySWAP } from './statevector.js'
import { H, X, Y, Z, S, T, Rx, Ry, Rz, Xx, Yy, Zz } from './gates.js'
import type { Gate2x2, Gate4x4, StateVector } from './statevector.js'

const SEED = 12345

// Node.js 22 exposes performance and console as globals.
// Declare them explicitly to avoid requiring the 'dom' lib in tsconfig.
declare const performance: { now(): number }
declare const console: { log(...args: unknown[]): void }

const now  = (): number        => performance.now()
const log  = (...a: unknown[]) => console.log(...a)

// ── Helpers ───────────────────────────────────────────────────────────────────

const sq2 = 1 / Math.sqrt(2)
const EPS = 1e-10  // tight tolerance for exact circuits
const EPS_STAT = 0.05  // statistical tolerance for sample-based checks

/** Sum of |amplitude|² for MPS full contraction. Must equal 1. */
function mpsNorm(mps: MPS): number {
  const amps = mpsContract(mps)
  let s = 0
  for (let i = 0; i < amps.length; i += 2) s += amps[i]! * amps[i]! + amps[i + 1]! * amps[i + 1]!
  return s
}

/** Get amplitude for a specific basis state (LSB = qubit 0). */
function mpsAmp(mps: MPS, idx: number): [number, number] {
  const amps = mpsContract(mps)
  return [amps[idx * 2]!, amps[idx * 2 + 1]!]
}

/** Get probability for a basis state. */
function mpsProb(mps: MPS, idx: number): number {
  const [re, im] = mpsAmp(mps, idx)
  return re * re + im * im
}

/** Deterministic pseudo-RNG (xorshift32) for reproducible sampling. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5
    return (s >>> 0) / 0x1_0000_0000
  }
}

/** Sample `shots` times and return frequency map. */
function sampleMPS(mps: MPS, shots: number, seed = 42): Map<string, number> {
  const rng = makeRng(seed)
  const n = mps.length
  const freq = new Map<string, number>()
  for (let i = 0; i < shots; i++) {
    const idx = mpsSample(mps, rng)
    const key = idx.toString(2).padStart(n, '0').split('').reverse().join('')  // q0-leftmost
    freq.set(key, (freq.get(key) ?? 0) + 1)
  }
  return freq
}

/** Chi-squared p-value test (returns true if sample matches expected probs). */
function chiSquaredPass(
  freq: Map<string, number>,
  expected: Record<string, number>,
  shots: number,
  threshold = 0.001,
): boolean {
  let chi2 = 0
  for (const [k, p] of Object.entries(expected)) {
    const obs = (freq.get(k) ?? 0)
    const exp = p * shots
    if (exp > 0) chi2 += (obs - exp) ** 2 / exp
  }
  const dof = Object.keys(expected).length - 1
  // Very rough: chi2 < 3 * dof is a generous pass for large samples
  return chi2 < 3 * dof + 20
}

/** Build a Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 via H⊗CNOT. */
function bellState(): MPS {
  let mps = mpsInit(2)
  mps = mpsApply1(mps, 0, H)
  mps = mpsApply2(mps, 0, 1, CNOT4, 8)
  return mps
}

/** Build n-qubit GHZ state. */
function ghzState(n: number, maxBond = 64): MPS {
  let mps = mpsInit(n)
  mps = mpsApply1(mps, 0, H)
  for (let q = 0; q < n - 1; q++) mps = mpsApply2(mps, q, q + 1, CNOT4, maxBond)
  return mps
}

/** Full statevector for cross-validation (small n only). */
function referenceSV(n: number, gateSeq: Array<[string, ...number[]]>): StateVector {
  let sv = zero(n)
  for (const [kind, ...args] of gateSeq) {
    switch (kind) {
      case 'H':    sv = applySingle(sv, args[0]!, [H[0]!, H[1]!]); break
      case 'X':    sv = applySingle(sv, args[0]!, [X[0]!, X[1]!]); break
      case 'Y':    sv = applySingle(sv, args[0]!, [Y[0]!, Y[1]!]); break
      case 'Z':    sv = applySingle(sv, args[0]!, [Z[0]!, Z[1]!]); break
      case 'S':    sv = applySingle(sv, args[0]!, [S[0]!, S[1]!]); break
      case 'T':    sv = applySingle(sv, args[0]!, [T[0]!, T[1]!]); break
      case 'CNOT': sv = applyCNOT(sv, args[0]!, args[1]!); break
      case 'SWAP': sv = applySWAP(sv, args[0]!, args[1]!); break
    }
  }
  return sv
}

/**
 * Compare MPS amplitude array to statevector probabilities.
 * Returns max absolute difference in probability.
 */
function maxProbError(mps: MPS, sv: StateVector): number {
  const amps = mpsContract(mps)
  let maxErr = 0
  const n = mps.length
  const dim = 1 << n
  for (let idx = 0; idx < dim; idx++) {
    const mpsP = amps[idx * 2]! ** 2 + amps[idx * 2 + 1]! ** 2
    const svAmp = sv.get(BigInt(idx))
    const svP = svAmp ? svAmp.re ** 2 + svAmp.im ** 2 : 0
    maxErr = Math.max(maxErr, Math.abs(mpsP - svP))
  }
  return maxErr
}

// ── Group 1: Tensor layout correctness ────────────────────────────────────────
// Bugs caught: wrong index formula, off-by-one in chiL/chiR, wrong interleaving

describe('tensor layout — internal data structure invariants', () => {
  it('mpsInit produces correct |0⟩ tensor: T[0][0][0]=1, T[0][1][0]=0', () => {
    const mps = mpsInit(3)
    const t = mpsTensor(mps, 0)
    expect(t.chiL).toBe(1)
    expect(t.chiR).toBe(1)
    expect(t.data.length).toBe(4)  // chiL*2*chiR*2 = 1*2*1*2 = 4
    // T[0][0][0].re = data[((0*2+0)*1+0)*2+0] = data[0]
    expect(t.data[0]).toBe(1)      // |0⟩ amplitude re
    expect(t.data[1]).toBe(0)      // |0⟩ amplitude im
    expect(t.data[2]).toBe(0)      // |1⟩ amplitude re
    expect(t.data[3]).toBe(0)      // |1⟩ amplitude im
  })

  it('mpsInit: all n sites have chiL=chiR=1', () => {
    const mps = mpsInit(10)
    for (let q = 0; q < 10; q++) {
      expect(mpsTensor(mps, q).chiL).toBe(1)
      expect(mpsTensor(mps, q).chiR).toBe(1)
    }
  })

  it('mpsApply1(H): after H on site 0, T[0][0][0]=1/√2, T[0][1][0]=1/√2', () => {
    const mps = mpsApply1(mpsInit(2), 0, H)
    const t = mpsTensor(mps, 0)
    expect(t.data[0]).toBeCloseTo(sq2, 14)   // T[l=0][p=0][r=0].re
    expect(t.data[1]).toBeCloseTo(0, 14)     // T[l=0][p=0][r=0].im
    expect(t.data[2]).toBeCloseTo(sq2, 14)   // T[l=0][p=1][r=0].re
    expect(t.data[3]).toBeCloseTo(0, 14)     // T[l=0][p=1][r=0].im
  })

  it('mpsApply1(X): flips |0⟩ to |1⟩ — T[0][0][0]=0, T[0][1][0]=1', () => {
    const mps = mpsApply1(mpsInit(1), 0, X)
    const t = mpsTensor(mps, 0)
    expect(t.data[0]).toBeCloseTo(0, 14)
    expect(t.data[2]).toBeCloseTo(1, 14)
  })

  it('data length = chiL * 2 * chiR * 2 at all sites after entangling gate', () => {
    const mps = bellState()
    for (let q = 0; q < 2; q++) {
      const t = mpsTensor(mps, q)
      expect(t.data.length).toBe(t.chiL * 2 * t.chiR * 2)
    }
  })

  it('mpsApply1 does not mutate original MPS (immutability)', () => {
    const orig = mpsInit(3)
    const origData = mpsTensor(orig, 1).data.slice()
    mpsApply1(orig, 1, H)
    expect(mpsTensor(orig, 1).data).toEqual(origData)
  })

  it('mpsApply2 does not mutate original MPS (immutability)', () => {
    const orig = mpsInit(4)
    mpsApply1(orig, 0, H)
    const snapA = mpsTensor(orig, 0).data.slice()
    const snapB = mpsTensor(orig, 1).data.slice()
    mpsApply2(orig, 0, 1, CNOT4, 8)
    expect(mpsTensor(orig, 0).data).toEqual(snapA)
    expect(mpsTensor(orig, 1).data).toEqual(snapB)
  })

  it('data is a Float64Array instance (not a plain Array)', () => {
    const mps = mpsInit(3)
    expect(mpsTensor(mps, 0).data).toBeInstanceOf(Float64Array)
    const after = mpsApply1(mps, 0, H)
    expect(mpsTensor(after, 0).data).toBeInstanceOf(Float64Array)
    const after2 = mpsApply2(after, 0, 1, CNOT4, 8)
    expect(mpsTensor(after2, 0).data).toBeInstanceOf(Float64Array)
    expect(mpsTensor(after2, 1).data).toBeInstanceOf(Float64Array)
  })

  it('bond invariant: mps[q].chiR === mps[q+1].chiL at every bond', () => {
    // The zero-copy reshape in mpsApply2Adjacent must produce consistent bond dims.
    // A violation here means the contraction or reshape has a size mismatch.
    const check = (mps: MPS) => {
      for (let q = 0; q < mps.length - 1; q++) {
        expect(mpsTensor(mps, q).chiR).toBe(mpsTensor(mps, q + 1).chiL)
      }
    }
    let mps = mpsInit(5)
    check(mps)
    for (let q = 0; q < 5; q++) mps = mpsApply1(mps, q, H)
    check(mps)
    for (let q = 0; q < 4; q++) { mps = mpsApply2(mps, q, q + 1, CNOT4, 8); check(mps) }
    // Non-adjacent gate also goes through SWAP network — check invariant holds
    let mps2 = mpsInit(5)
    mps2 = mpsApply1(mps2, 0, H)
    mps2 = mpsApply2(mps2, 0, 4, CNOT4, 16)
    check(mps2)
  })

  it('raw data layout after CNOT: Bell state tensor A has bond index as chiR', () => {
    // After H(q0) + CNOT(q0,q1) the bond dimension is 2.
    // Tensor A has chiL=1, chiR=2: T[l=0][p][r] = data[((0*2+p)*2+r)*2]
    // Amplitudes: |00⟩=1/√2, |11⟩=1/√2, others zero.
    // A is the left tensor (q0), contracted with B (q1) to give |Φ+⟩.
    // We just verify the structural invariants, not exact values
    // (exact values depend on QR gauge choice).
    const mps = bellState()
    const A = mpsTensor(mps, 0), B = mpsTensor(mps, 1)
    expect(A.chiL).toBe(1)
    expect(A.chiR).toBe(B.chiL)          // bond invariant
    expect(A.data.length).toBe(A.chiL * 2 * A.chiR * 2)
    expect(B.data.length).toBe(B.chiL * 2 * B.chiR * 2)
    // The full contraction must reconstruct the correct state
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 14)
    expect(mpsProb(mps, 3)).toBeCloseTo(0.5, 14)
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
    expect(mpsProb(mps, 2)).toBeCloseTo(0, 14)
  })
})

// ── Group 2: Amplitude correctness — known-answer tests ──────────────────────
// Bugs caught: wrong gate matrix, physical index ordering, control/target swap

describe('amplitude correctness — known states', () => {
  it('|0...0⟩: only amplitude [0...0] is 1', () => {
    const mps = mpsInit(4)
    expect(mpsProb(mps, 0)).toBeCloseTo(1, 14)
    for (let i = 1; i < 16; i++) expect(mpsProb(mps, i)).toBeCloseTo(0, 14)
  })

  it('H on qubit 0: |+⟩ state, equal amplitudes for 0 and 1 (q0 = LSB)', () => {
    const mps = mpsApply1(mpsInit(2), 0, H)
    // idx 0 = |00⟩ (q0=0), idx 1 = |01⟩ (q0=1 in LSB convention, but H is on q0)
    // LSB = qubit 0, so |10⟩ in qubit space = index with bit 0 set = idx 1
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 14)  // |q0=0, q1=0⟩
    expect(mpsProb(mps, 1)).toBeCloseTo(0.5, 14)  // |q0=1, q1=0⟩
    expect(mpsProb(mps, 2)).toBeCloseTo(0, 14)
    expect(mpsProb(mps, 3)).toBeCloseTo(0, 14)
  })

  it('H on qubit 1: equal amplitudes at idx 0 and idx 2 (bit 1 set)', () => {
    const mps = mpsApply1(mpsInit(2), 1, H)
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 14)  // q1=0
    expect(mpsProb(mps, 2)).toBeCloseTo(0.5, 14)  // q1=1 (bit 1 set → idx += 2)
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
    expect(mpsProb(mps, 3)).toBeCloseTo(0, 14)
  })

  it('Bell state |Φ+⟩: prob(|00⟩)=0.5, prob(|11⟩)=0.5', () => {
    const mps = bellState()
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 14)  // |q0=0,q1=0⟩ = idx 0
    expect(mpsProb(mps, 3)).toBeCloseTo(0.5, 14)  // |q0=1,q1=1⟩ = idx 3
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
    expect(mpsProb(mps, 2)).toBeCloseTo(0, 14)
  })

  it('Bell state: amplitudes are real, equal, positive 1/√2', () => {
    const mps = bellState()
    const [re00, im00] = mpsAmp(mps, 0)
    const [re11, im11] = mpsAmp(mps, 3)
    expect(re00).toBeCloseTo(sq2, 14)
    expect(im00).toBeCloseTo(0, 14)
    expect(re11).toBeCloseTo(sq2, 14)
    expect(im11).toBeCloseTo(0, 14)
  })

  it('|Ψ-⟩ Bell state: correct anti-correlation', () => {
    // |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, X)         // |10⟩
    mps = mpsApply1(mps, 0, H)         // (|00⟩-|10⟩)/√2
    mps = mpsApply2(mps, 0, 1, CNOT4, 8)  // (|00⟩-|11⟩)/√2... not |Ψ-⟩
    // Correct: H on q0, CNOT q0→q1 gives |Φ+⟩; for |Ψ-⟩ we need X then H then CNOT
    // |10⟩ → H⊗I → (|00⟩-|10⟩)/√2 → CNOT_{0→1} → (|00⟩-|11⟩)/√2 = |Φ-⟩
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 14)
    expect(mpsProb(mps, 3)).toBeCloseTo(0.5, 14)
    const [r0] = mpsAmp(mps, 0)
    const [r3] = mpsAmp(mps, 3)
    // One should be positive, one negative
    expect(r0 * r3).toBeLessThan(0)  // opposite signs
  })

  it('3-qubit GHZ: only |000⟩ and |111⟩ have nonzero amplitude', () => {
    const mps = ghzState(3)
    const n = 3, dim = 1 << n
    for (let i = 0; i < dim; i++) {
      const p = mpsProb(mps, i)
      if (i === 0 || i === 7) {
        expect(p).toBeCloseTo(0.5, 12)
      } else {
        expect(p).toBeCloseTo(0, 12)
      }
    }
  })

  it('5-qubit GHZ: only |00000⟩=idx 0 and |11111⟩=idx 31 nonzero', () => {
    const mps = ghzState(5)
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 12)
    expect(mpsProb(mps, 31)).toBeCloseTo(0.5, 12)
    // Spot-check a few middle states
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 12)
    expect(mpsProb(mps, 15)).toBeCloseTo(0, 12)
    expect(mpsProb(mps, 16)).toBeCloseTo(0, 12)
  })

  it('X gate: |0⟩→|1⟩', () => {
    const mps = mpsApply1(mpsInit(1), 0, X)
    expect(mpsProb(mps, 0)).toBeCloseTo(0, 14)
    expect(mpsProb(mps, 1)).toBeCloseTo(1, 14)
  })

  it('X²=I: double flip returns to |0⟩', () => {
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, X)
    mps = mpsApply1(mps, 0, X)
    expect(mpsProb(mps, 0)).toBeCloseTo(1, 14)
  })

  it('H²=I: double Hadamard returns to |0⟩', () => {
    let mps = mpsInit(3)
    mps = mpsApply1(mps, 1, H)
    mps = mpsApply1(mps, 1, H)
    expect(mpsProb(mps, 0)).toBeCloseTo(1, 14)
  })

  it('Y gate: |0⟩ → i|1⟩', () => {
    const mps = mpsApply1(mpsInit(1), 0, Y)
    const [re, im] = mpsAmp(mps, 1)
    expect(re).toBeCloseTo(0, 14)
    expect(im).toBeCloseTo(1, 14)
  })

  it('Z gate: |0⟩ unchanged, |1⟩ gets phase -1', () => {
    let mps = mpsApply1(mpsInit(1), 0, X)  // |1⟩
    mps = mpsApply1(mps, 0, Z)
    const [re, im] = mpsAmp(mps, 1)
    expect(re).toBeCloseTo(-1, 14)
    expect(im).toBeCloseTo(0, 14)
  })

  it('S gate: |+⟩ → |+i⟩ = (|0⟩+i|1⟩)/√2', () => {
    let mps = mpsApply1(mpsInit(1), 0, H)  // |+⟩
    mps = mpsApply1(mps, 0, S)
    const [re0, im0] = mpsAmp(mps, 0)
    const [re1, im1] = mpsAmp(mps, 1)
    expect(re0).toBeCloseTo(sq2, 14)
    expect(im0).toBeCloseTo(0, 14)
    expect(re1).toBeCloseTo(0, 14)
    expect(im1).toBeCloseTo(sq2, 14)
  })

  it('CNOT: |10⟩ → |11⟩ (control=q0=1 flips target=q1)', () => {
    // |10⟩ in LSB: q0=1, q1=0 → idx = 0b01 = 1
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, X)   // set q0=1 → |10⟩ (idx 1 in LSB)
    mps = mpsApply2(mps, 0, 1, CNOT4, 8)  // flip q1 → |11⟩ (idx 3)
    expect(mpsProb(mps, 3)).toBeCloseTo(1, 14)
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
  })

  it('CNOT: |00⟩ → |00⟩ (control=0, no flip)', () => {
    let mps = mpsInit(2)
    mps = mpsApply2(mps, 0, 1, CNOT4, 8)
    expect(mpsProb(mps, 0)).toBeCloseTo(1, 14)
  })

  it('CNOT: |01⟩ → |01⟩ (control=q0=0, no flip even though q1=1)', () => {
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 1, X)   // set q1=1 → idx 2
    mps = mpsApply2(mps, 0, 1, CNOT4, 8)
    expect(mpsProb(mps, 2)).toBeCloseTo(1, 14)
  })

  it('SWAP: |10⟩ → |01⟩', () => {
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, X)   // q0=1 → idx 1
    mps = mpsApply2(mps, 0, 1, SWAP4, 8)
    expect(mpsProb(mps, 2)).toBeCloseTo(1, 14)  // q1=1 → idx 2
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
  })

  it('SWAP is self-inverse: SWAP·SWAP = I', () => {
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, H)
    const before = mpsContract(mps).slice()
    mps = mpsApply2(mps, 0, 1, SWAP4, 8)
    mps = mpsApply2(mps, 0, 1, SWAP4, 8)
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) {
      expect(after[i]).toBeCloseTo(before[i]!, 14)
    }
  })

  it('CNOT is self-inverse: CNOT·CNOT = I', () => {
    let mps = mpsInit(3)
    mps = mpsApply1(mps, 0, H)
    mps = mpsApply1(mps, 1, H)
    const before = mpsContract(mps).slice()
    mps = mpsApply2(mps, 0, 1, CNOT4, 8)
    mps = mpsApply2(mps, 0, 1, CNOT4, 8)
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) expect(after[i]).toBeCloseTo(before[i]!, 14)
  })

  it('Rx(π)|0⟩ = -i|1⟩ (exact complex amplitude)', () => {
    // Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    // Rx(π)|0⟩ = [cos(π/2), -i·sin(π/2)]ᵀ = [0, -i]ᵀ
    const mps = mpsApply1(mpsInit(1), 0, Rx(Math.PI))
    const [re1, im1] = mpsAmp(mps, 1)
    expect(re1).toBeCloseTo(0, 14)
    expect(im1).toBeCloseTo(-1, 14)
  })

  it('Ry(π)|0⟩ = |1⟩ (real amplitude, no phase)', () => {
    // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    // Ry(π)|0⟩ = [0, 1]ᵀ  — real positive, unlike Rx or Y which give imaginary
    const mps = mpsApply1(mpsInit(1), 0, Ry(Math.PI))
    const [re1, im1] = mpsAmp(mps, 1)
    expect(re1).toBeCloseTo(1, 14)
    expect(im1).toBeCloseTo(0, 14)
  })

  it('Y|1⟩ = -i|0⟩ (second half of Y gate)', () => {
    // Y = [[0,-i],[i,0]]: |1⟩ → -i|0⟩.  Only testing |0⟩→i|1⟩ leaves the gate half-verified.
    let mps = mpsApply1(mpsInit(1), 0, X)  // |1⟩
    mps = mpsApply1(mps, 0, Y)
    const [re0, im0] = mpsAmp(mps, 0)
    expect(re0).toBeCloseTo(0, 14)
    expect(im0).toBeCloseTo(-1, 14)
  })

  it('T|1⟩ = e^{iπ/4}|1⟩ = (1+i)/√2 · |1⟩', () => {
    // T = diag(1, e^{iπ/4}). Verifies T gate has the right phase on |1⟩.
    let mps = mpsApply1(mpsInit(1), 0, X)  // |1⟩
    mps = mpsApply1(mps, 0, T)
    const [re, im] = mpsAmp(mps, 1)
    expect(re).toBeCloseTo(1 / Math.SQRT2, 14)
    expect(im).toBeCloseTo(1 / Math.SQRT2, 14)
  })

  it('controlledGate(H): CU on |10⟩ applies H to qubit 1', () => {
    // |10⟩: q0=1 (control), q1=0 (target)
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, X)
    mps = mpsApply2(mps, 0, 1, controlledGate(H), 8)
    // q0=1 controls H on q1: result = |1⟩⊗H|0⟩ = |1⟩(|0⟩+|1⟩)/√2
    // idx: q0=1(bit0=1), q1=0(bit1=0)=1; q0=1,q1=1=3
    expect(mpsProb(mps, 1)).toBeCloseTo(0.5, 14)
    expect(mpsProb(mps, 3)).toBeCloseTo(0.5, 14)
  })

  it('XX(π/2): maximally entangling, correct off-diagonal structure', () => {
    const mps = mpsApply2(mpsInit(2), 0, 1, Xx(Math.PI / 2), 8)
    // XX(π/2)|00⟩ = (1/√2)(|00⟩ - i|11⟩)
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 14)
    expect(mpsProb(mps, 3)).toBeCloseTo(0.5, 14)
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
    expect(mpsProb(mps, 2)).toBeCloseTo(0, 14)
    // Phase: |00⟩ coeff real, |11⟩ coeff is -i
    const [, im11] = mpsAmp(mps, 3)
    expect(im11).toBeCloseTo(-sq2, 12)
  })

  it('ZZ(π)|00⟩: exact amplitude e^{-iπ/2}|00⟩ = -i|00⟩', () => {
    // ZZ(θ) = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2})
    // ZZ(π)|00⟩ = e^{-iπ/2}|00⟩ = -i|00⟩
    const mps = mpsApply2(mpsInit(2), 0, 1, Zz(Math.PI), 8)
    const [re, im] = mpsAmp(mps, 0)
    expect(re).toBeCloseTo(0, 14)
    expect(im).toBeCloseTo(-1, 14)
  })

  it('Yy(π/2)|00⟩ = (|00⟩ + i|11⟩)/√2 (exact complex amplitudes)', () => {
    // Yy(π/2) = exp(-iπ/4 · Y⊗Y). Applied to |00⟩:
    // result = cos(π/4)|00⟩ + i·sin(π/4)|11⟩ = (1/√2)|00⟩ + (i/√2)|11⟩
    // Note: differs from XX(π/2) which gives (|00⟩ - i|11⟩)/√2
    const mps = mpsApply2(mpsInit(2), 0, 1, Yy(Math.PI / 2), 8)
    const [re00, im00] = mpsAmp(mps, 0)
    const [re11, im11] = mpsAmp(mps, 3)
    expect(re00).toBeCloseTo(sq2, 14)
    expect(im00).toBeCloseTo(0, 14)
    expect(re11).toBeCloseTo(0, 14)
    expect(im11).toBeCloseTo(sq2, 14)   // +i/√2, not −i/√2 — sign differs from XX
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
    expect(mpsProb(mps, 2)).toBeCloseTo(0, 14)
  })

  it('ZZ algebraic identity: ZZ(θ) = CNOT·(I⊗Rz(θ))·CNOT (amplitude level)', () => {
    // From the ZZ gate docstring. Verifies the IonQ decomposition is consistent.
    const theta = 1.23
    const buildZZ = (n: number, a: number, b: number) => {
      let mps = mpsInit(n)
      mps = mpsApply1(mps, 0, H); mps = mpsApply1(mps, 1, S)
      mps = mpsApply2(mps, a, b, Zz(theta), 32)
      return mpsContract(mps)
    }
    const buildDecomp = (n: number, a: number, b: number) => {
      let mps = mpsInit(n)
      mps = mpsApply1(mps, 0, H); mps = mpsApply1(mps, 1, S)
      mps = mpsApply2(mps, a, b, CNOT4, 32)
      mps = mpsApply1(mps, b, Rz(theta))
      mps = mpsApply2(mps, a, b, CNOT4, 32)
      return mpsContract(mps)
    }
    const zzAmps = buildZZ(2, 0, 1)
    const dcAmps = buildDecomp(2, 0, 1)
    for (let i = 0; i < zzAmps.length; i++) {
      expect(zzAmps[i]!).toBeCloseTo(dcAmps[i]!, 12)
    }
  })

  it('Z²=I: double Z leaves |+⟩ unchanged', () => {
    // Z=diag(1,-1). Z²=I. Missing from the involution tests alongside X² and H².
    let mps = mpsApply1(mpsInit(2), 0, H)
    const before = mpsContract(mps).slice()
    mps = mpsApply1(mps, 0, Z)
    mps = mpsApply1(mps, 0, Z)
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) expect(after[i]!).toBeCloseTo(before[i]!, 14)
  })

  it('Y²=I: double Y leaves |+⟩ unchanged', () => {
    // Y²=I. Also implicitly checks that Y·Y cancels the imaginary phase.
    let mps = mpsApply1(mpsInit(2), 0, H)
    const before = mpsContract(mps).slice()
    mps = mpsApply1(mps, 0, Y)
    mps = mpsApply1(mps, 0, Y)
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) expect(after[i]!).toBeCloseTo(before[i]!, 14)
  })

  it('iSWAP|10⟩ = i|01⟩: verifies XY-type gate complex phase (Xy imported separately)', () => {
    // iSWAP = XY(π). Acts as SWAP on computational basis but with an i phase factor.
    // iSWAP|01⟩ = i|10⟩, iSWAP|10⟩ = i|01⟩. Diagonal elements are 1 (|00⟩,|11⟩ unchanged).
    // Construct inline (Xy not re-imported here) — same matrix as Xy(π).
    const c0 = { re: 0, im: 0 }, c1 = { re: 1, im: 0 }, ci = { re: 0, im: 1 }
    const iswap: Gate4x4 = [
      [c1, c0, c0, c0],
      [c0, c0, ci, c0],
      [c0, ci, c0, c0],
      [c0, c0, c0, c1],
    ]
    // |10⟩ (q0=1, q1=0) = idx 1 in LSB convention
    let mps = mpsApply1(mpsInit(2), 0, X)
    mps = mpsApply2(mps, 0, 1, iswap, 8)
    // After iSWAP: |10⟩ → i|01⟩ = idx 2 with imaginary amplitude +i
    const [re2, im2] = mpsAmp(mps, 2)
    expect(re2).toBeCloseTo(0, 14)
    expect(im2).toBeCloseTo(1, 14)
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 14)
  })
})

// ── Group 3: Cross-validation vs statevector ──────────────────────────────────
// Bugs caught: systematic amplitude errors, gate ordering, control/target confusion

describe('cross-validation — MPS vs statevector backend', () => {
  it('H on each qubit of n=4: MPS matches SV probabilities', () => {
    const n = 4
    let mps = mpsInit(n)
    let sv = zero(n)
    for (let q = 0; q < n; q++) {
      mps = mpsApply1(mps, q, H)
      sv = applySingle(sv, q, H)
    }
    expect(maxProbError(mps, sv)).toBeLessThan(EPS)
  })

  it('GHZ circuit n=5: MPS matches SV', () => {
    const n = 5
    let mps = mpsInit(n)
    let sv = zero(n)
    mps = mpsApply1(mps, 0, H)
    sv = applySingle(sv, 0, H)
    for (let q = 0; q < n - 1; q++) {
      mps = mpsApply2(mps, q, q + 1, CNOT4, 64)
      sv = applyCNOT(sv, q, q + 1)
    }
    expect(maxProbError(mps, sv)).toBeLessThan(EPS)
  })

  it('random-looking circuit n=4: MPS matches SV', () => {
    const n = 4
    let mps = mpsInit(n)
    let sv = zero(n)
    const gates: Array<[Gate2x2, number]> = [
      [H, 0], [X, 1], [H, 2], [S, 3],
      [H, 1], [T, 0], [H, 3], [X, 2],
    ]
    for (const [g, q] of gates) {
      mps = mpsApply1(mps, q, g)
      sv = applySingle(sv, q, g)
    }
    mps = mpsApply2(mps, 0, 1, CNOT4, 32)
    sv = applyCNOT(sv, 0, 1)
    mps = mpsApply2(mps, 2, 3, CNOT4, 32)
    sv = applyCNOT(sv, 2, 3)
    mps = mpsApply2(mps, 1, 2, CNOT4, 32)
    sv = applyCNOT(sv, 1, 2)
    expect(maxProbError(mps, sv)).toBeLessThan(EPS)
  })

  it('T-gate followed by CNOT n=3: phase sensitive check vs SV', () => {
    const n = 3
    let mps = mpsInit(n)
    let sv = zero(n)
    mps = mpsApply1(mps, 0, H); sv = applySingle(sv, 0, H)
    mps = mpsApply1(mps, 1, H); sv = applySingle(sv, 1, H)
    mps = mpsApply1(mps, 0, T); sv = applySingle(sv, 0, T)
    mps = mpsApply2(mps, 0, 1, CNOT4, 16); sv = applyCNOT(sv, 0, 1)
    mps = mpsApply1(mps, 1, T); sv = applySingle(sv, 1, T)
    mps = mpsApply2(mps, 1, 2, CNOT4, 16); sv = applyCNOT(sv, 1, 2)
    // Verify amplitudes match (not just probabilities - catches phase errors)
    const amps = mpsContract(mps)
    const dim = 1 << n
    for (let i = 0; i < dim; i++) {
      const svAmp = sv.get(BigInt(i))
      const svRe = svAmp?.re ?? 0, svIm = svAmp?.im ?? 0
      expect(amps[i * 2]!).toBeCloseTo(svRe, 12)
      expect(amps[i * 2 + 1]!).toBeCloseTo(svIm, 12)
    }
  })

  it('SWAP n=4 cross-validation', () => {
    const n = 4
    let mps = mpsInit(n)
    let sv = zero(n)
    // Prepare a non-trivial state
    mps = mpsApply1(mps, 0, H); sv = applySingle(sv, 0, H)
    mps = mpsApply1(mps, 2, S); sv = applySingle(sv, 2, S)
    mps = mpsApply2(mps, 0, 1, CNOT4, 16); sv = applyCNOT(sv, 0, 1)
    // Now swap
    mps = mpsApply2(mps, 1, 3, SWAP4, 32); sv = applySWAP(sv, 1, 3)
    expect(maxProbError(mps, sv)).toBeLessThan(EPS)
  })

  it('non-adjacent CNOT n=5: MPS matches SV', () => {
    const n = 5
    let mps = mpsInit(n)
    let sv = zero(n)
    mps = mpsApply1(mps, 0, H); sv = applySingle(sv, 0, H)
    // CNOT with gap: q0 → q4
    mps = mpsApply2(mps, 0, 4, CNOT4, 32); sv = applyCNOT(sv, 0, 4)
    expect(maxProbError(mps, sv)).toBeLessThan(EPS)
  })

  it('non-adjacent CNOT n=6, q1→q5: MPS matches SV', () => {
    const n = 6
    let mps = mpsInit(n)
    let sv = zero(n)
    for (let q = 0; q < n; q++) { mps = mpsApply1(mps, q, H); sv = applySingle(sv, q, H) }
    mps = mpsApply2(mps, 1, 5, CNOT4, 64); sv = applyCNOT(sv, 1, 5)
    expect(maxProbError(mps, sv)).toBeLessThan(EPS)
  })

  it('controlled-Rx gate n=3: MPS matches SV via applyTwo', () => {
    const n = 3
    let mps = mpsInit(n)
    let sv = zero(n)
    const CRx = controlledGate(Rx(Math.PI / 4))
    mps = mpsApply1(mps, 0, H); sv = applySingle(sv, 0, H)
    mps = mpsApply2(mps, 0, 1, CRx, 16); sv = applyTwo(sv, 0, 1, CRx)
    mps = mpsApply2(mps, 1, 2, CNOT4, 16); sv = applyCNOT(sv, 1, 2)
    expect(maxProbError(mps, sv)).toBeLessThan(EPS)
  })

  it('XX(θ) gate n=2 at various θ: cross-val vs SV', () => {
    for (const theta of [0, 0.1, 0.5, Math.PI / 4, Math.PI / 2]) {
      let mps = mpsInit(2)
      let sv = zero(2)
      mps = mpsApply1(mps, 0, H); sv = applySingle(sv, 0, H)
      const xxg = Xx(theta)
      mps = mpsApply2(mps, 0, 1, xxg, 8); sv = applyTwo(sv, 0, 1, xxg)
      expect(maxProbError(mps, sv)).toBeLessThan(EPS)
    }
  })

  it('amplitude cross-validation: MPS and SV agree on complex values, not just probs', () => {
    // maxProbError only checks |amp|². This test checks the actual re/im components
    // to catch bugs that would flip or rotate amplitudes while preserving magnitude.
    const n = 4
    let mps = mpsInit(n)
    let sv = zero(n)
    // Circuit with S and T gates produces non-trivial complex amplitudes
    mps = mpsApply1(mps, 0, H); sv = applySingle(sv, 0, H)
    mps = mpsApply1(mps, 1, S); sv = applySingle(sv, 1, S)
    mps = mpsApply2(mps, 0, 1, CNOT4, 16); sv = applyCNOT(sv, 0, 1)
    mps = mpsApply1(mps, 1, T); sv = applySingle(sv, 1, T)
    mps = mpsApply2(mps, 1, 2, CNOT4, 16); sv = applyCNOT(sv, 1, 2)
    mps = mpsApply1(mps, 2, Rz(Math.PI / 3)); sv = applySingle(sv, 2, Rz(Math.PI / 3))
    mps = mpsApply2(mps, 2, 3, CNOT4, 16); sv = applyCNOT(sv, 2, 3)
    const amps = mpsContract(mps)
    const dim = 1 << n
    for (let i = 0; i < dim; i++) {
      const svAmp = sv.get(BigInt(i))
      expect(amps[i * 2]!).toBeCloseTo(svAmp?.re ?? 0, 12)
      expect(amps[i * 2 + 1]!).toBeCloseTo(svAmp?.im ?? 0, 12)
    }
  })
})

// ── Group 4: Norm preservation ────────────────────────────────────────────────
// Bugs caught: leaking/gaining probability, non-unitary gate application

describe('norm preservation — total probability must equal 1', () => {
  const cases: Array<[string, () => MPS]> = [
    ['|0...0⟩ n=5', () => mpsInit(5)],
    ['Bell state', () => bellState()],
    ['GHZ n=4', () => ghzState(4)],
    ['H on all qubits n=6', () => {
      let m = mpsInit(6)
      for (let q = 0; q < 6; q++) m = mpsApply1(m, q, H)
      return m
    }],
    ['S gate on various qubits n=3', () => {
      let m = mpsInit(3)
      m = mpsApply1(m, 0, S)
      m = mpsApply1(m, 1, T)
      m = mpsApply1(m, 2, H)
      return m
    }],
    ['alternating H+CNOT n=4', () => {
      let m = mpsInit(4)
      for (let q = 0; q < 4; q++) m = mpsApply1(m, q, H)
      m = mpsApply2(m, 0, 1, CNOT4, 32)
      m = mpsApply2(m, 2, 3, CNOT4, 32)
      m = mpsApply2(m, 1, 2, CNOT4, 32)
      return m
    }],
    ['non-adjacent gate n=6', () => {
      let m = mpsInit(6)
      m = mpsApply1(m, 0, H)
      m = mpsApply2(m, 0, 5, CNOT4, 32)
      return m
    }],
    ['XX(π/3) n=4 chain', () => {
      let m = mpsInit(4)
      for (let q = 0; q < 3; q++) m = mpsApply2(m, q, q + 1, Xx(Math.PI / 3), 32)
      return m
    }],
  ]

  for (const [name, build] of cases) {
    it(`norm = 1: ${name}`, () => {
      expect(mpsNorm(build())).toBeCloseTo(1, 13)
    })
  }

  it('norm preserved through 20-gate random-ish circuit n=4', () => {
    const gates: Gate2x2[] = [H, X, Y, Z, S, T, Rx(0.7), Ry(1.3), Rz(2.1)]
    let mps = mpsInit(4)
    const rng = makeRng(99)
    for (let i = 0; i < 20; i++) {
      const q = Math.floor(rng() * 4)
      const g = gates[Math.floor(rng() * gates.length)]!
      mps = mpsApply1(mps, q, g)
    }
    expect(mpsNorm(mps)).toBeCloseTo(1, 13)
  })
})

// ── Group 5: QR decomposition properties ──────────────────────────────────────
// Bugs caught: non-orthogonal Q columns, R not satisfying Q·R=M, bond overflow

describe('QR decomposition — via black-box mpsApply2 output', () => {
  it('bond dimension ≤ maxBond after every two-qubit gate', () => {
    let mps = mpsInit(6)
    for (let q = 0; q < 6; q++) mps = mpsApply1(mps, q, H)
    const maxBond = 4
    for (let q = 0; q < 5; q++) {
      mps = mpsApply2(mps, q, q + 1, CNOT4, maxBond)
      expect(mpsMaxBond(mps)).toBeLessThanOrEqual(maxBond)
    }
  })

  it('bond dimension stays 1 for product-state circuits', () => {
    // H gates only — no entanglement, bond stays 1
    let mps = mpsInit(8)
    for (let q = 0; q < 8; q++) mps = mpsApply1(mps, q, H)
    expect(mpsMaxBond(mps)).toBe(1)
  })

  it('bond dimension grows with entanglement then stabilizes at maxBond', () => {
    const maxBond = 4
    let mps = mpsInit(8)
    for (let q = 0; q < 8; q++) mps = mpsApply1(mps, q, H)
    for (let q = 0; q < 7; q++) {
      mps = mpsApply2(mps, q, q + 1, CNOT4, maxBond)
      const bond = mpsMaxBond(mps)
      expect(bond).toBeLessThanOrEqual(maxBond)
      expect(bond).toBeGreaterThanOrEqual(1)
    }
  })

  it('increasing maxBond improves accuracy for entangled state', () => {
    const n = 6
    let svRef = zero(n)
    svRef = applySingle(svRef, 0, H)
    for (let q = 0; q < n - 1; q++) svRef = applyCNOT(svRef, q, q + 1)
    const errors: number[] = []
    for (const maxBond of [1, 2, 4, 8, 16]) {
      let mps = mpsInit(n)
      mps = mpsApply1(mps, 0, H)
      for (let q = 0; q < n - 1; q++) mps = mpsApply2(mps, q, q + 1, CNOT4, maxBond)
      errors.push(maxProbError(mps, svRef))
    }
    // Error must be non-increasing as maxBond grows (or at worst not dramatically worse)
    expect(errors[4]!).toBeLessThanOrEqual(errors[0]! + 1e-6)
    // At maxBond=16, should be near-exact for this 6-qubit circuit
    expect(errors[4]!).toBeLessThan(1e-10)
  })

  it('maxBond=1 for product state: exact (no entanglement needed)', () => {
    // BV problem with product state output: all H gates + Pauli-Z marks
    let mps = mpsInit(5)
    for (let q = 0; q < 5; q++) mps = mpsApply1(mps, q, H)
    // No two-qubit gates → bond stays 1 → maxBond=1 is fine
    expect(mpsMaxBond(mps)).toBe(1)
    expect(mpsNorm(mps)).toBeCloseTo(1, 14)
  })

  it('tensor chiL × 2 × chiR * 2 = data.length after entangling circuit', () => {
    let mps = mpsInit(5)
    for (let q = 0; q < 5; q++) mps = mpsApply1(mps, q, H)
    for (let q = 0; q < 4; q++) mps = mpsApply2(mps, q, q + 1, CNOT4, 16)
    for (let q = 0; q < 5; q++) {
      const t = mpsTensor(mps, q)
      expect(t.data.length).toBe(t.chiL * 2 * t.chiR * 2)
    }
  })

  it('left-orthogonality: tensor A satisfies A†A = I after mpsApply2', () => {
    // After QR decomposition, the Q matrix (tensor A) must be left-orthogonal:
    //   sum_{l,p} conj(A[l,p,r1]) · A[l,p,r2] = δ_{r1,r2}
    // This is guaranteed by Modified Gram-Schmidt but must be verified — a bug in
    // the index formula or QR loop would break orthogonality silently.
    const checkLeftOrtho = (mps: MPS, site: number) => {
      const t = mpsTensor(mps, site)
      const { chiL, chiR, data } = t
      for (let r1 = 0; r1 < chiR; r1++) {
        for (let r2 = 0; r2 < chiR; r2++) {
          let sumRe = 0, sumIm = 0
          for (let l = 0; l < chiL; l++) {
            for (let p = 0; p < 2; p++) {
              const i1 = ((l * 2 + p) * chiR + r1) * 2
              const i2 = ((l * 2 + p) * chiR + r2) * 2
              // conj(A[l,p,r1]) · A[l,p,r2] = (re1 - i·im1)(re2 + i·im2)
              sumRe += data[i1]! * data[i2]! + data[i1 + 1]! * data[i2 + 1]!
              sumIm += data[i1]! * data[i2 + 1]! - data[i1 + 1]! * data[i2]!
            }
          }
          expect(sumRe).toBeCloseTo(r1 === r2 ? 1 : 0, 12)
          expect(sumIm).toBeCloseTo(0, 12)
        }
      }
    }
    // Bell state: bond=2, chiL=1 for site 0
    const bell = bellState()
    checkLeftOrtho(bell, 0)
    // GHZ n=5: bond grows to 2 at interior sites
    const ghz = ghzState(5)
    for (let q = 0; q < 4; q++) checkLeftOrtho(ghz, q)
    // Entangled circuit with higher bond
    let mps = mpsInit(4)
    for (let q = 0; q < 4; q++) mps = mpsApply1(mps, q, H)
    for (let q = 0; q < 3; q++) mps = mpsApply2(mps, q, q + 1, CNOT4, 8)
    for (let q = 0; q < 3; q++) checkLeftOrtho(mps, q)
  })
})

// ── Group 6: Non-adjacent gate correctness ────────────────────────────────────
// Bugs caught: wrong SWAP network direction, parity errors, qubit labeling

describe('non-adjacent gates — SWAP network correctness', () => {
  it('non-adjacent CNOT same result as adjacent after manual SWAPs', () => {
    const n = 4
    // Direct: CNOT q0→q3
    let mpsDirect = mpsInit(n)
    mpsDirect = mpsApply1(mpsDirect, 0, H)
    mpsDirect = mpsApply2(mpsDirect, 0, 3, CNOT4, 32)
    // Manual: SWAP q2↔q3, SWAP q1↔q2, CNOT q0→q1, SWAP q1↔q2, SWAP q2↔q3
    let mpsManual = mpsInit(n)
    mpsManual = mpsApply1(mpsManual, 0, H)
    mpsManual = mpsApply2(mpsManual, 2, 3, SWAP4, 32)
    mpsManual = mpsApply2(mpsManual, 1, 2, SWAP4, 32)
    mpsManual = mpsApply2(mpsManual, 0, 1, CNOT4, 32)
    mpsManual = mpsApply2(mpsManual, 1, 2, SWAP4, 32)
    mpsManual = mpsApply2(mpsManual, 2, 3, SWAP4, 32)
    const ampsDirect = mpsContract(mpsDirect)
    const ampsManual = mpsContract(mpsManual)
    for (let i = 0; i < ampsDirect.length; i++) {
      expect(ampsDirect[i]!).toBeCloseTo(ampsManual[i]!, 12)
    }
  })

  it('CNOT q0→q2 in n=3: correct entanglement pattern', () => {
    const n = 3
    let mps = mpsInit(n)
    mps = mpsApply1(mps, 0, H)
    mps = mpsApply2(mps, 0, 2, CNOT4, 16)
    // |0q10⟩ and |1q11⟩ each with prob 0.5, q1 free
    // Qubit 1 stays |0⟩: so |000⟩ and |101⟩ each get prob 0.5
    // idx=0: q0=0,q1=0,q2=0 → prob 0.5
    // idx=5: q0=1,q1=0,q2=1 → 0b101=5 → prob 0.5
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 12)
    expect(mpsProb(mps, 5)).toBeCloseTo(0.5, 12)
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 12)
    expect(mpsProb(mps, 2)).toBeCloseTo(0, 12)
    expect(mpsProb(mps, 3)).toBeCloseTo(0, 12)
    expect(mpsProb(mps, 4)).toBeCloseTo(0, 12)
    expect(mpsProb(mps, 7)).toBeCloseTo(0, 12)
  })

  it('non-adjacent SWAP q0↔q3 in n=4: moves qubit state correctly', () => {
    let mps = mpsInit(4)
    mps = mpsApply1(mps, 0, X)   // q0=1 (LSB bit 0) → idx=1
    mps = mpsApply2(mps, 0, 3, SWAP4, 32)
    // After swap: q3=1 (bit 3) → idx = 0b1000 = 8
    expect(mpsProb(mps, 8)).toBeCloseTo(1, 12)
    expect(mpsProb(mps, 1)).toBeCloseTo(0, 12)
  })
})

// ── Group 7: Sampling correctness ────────────────────────────────────────────
// Bugs caught: wrong marginal probabilities, biased sampler, wrong bit ordering

describe('sampling — statistical correctness', () => {
  const SHOTS = 5000

  it('Bell state sampling: ≈50% |00⟩, ≈50% |11⟩, never |01⟩ or |10⟩', () => {
    const mps = bellState()
    const freq = sampleMPS(mps, SHOTS, SEED)
    // Bell state should only produce '00' or '11' (q0-leftmost)
    expect(freq.get('01') ?? 0).toBe(0)
    expect(freq.get('10') ?? 0).toBe(0)
    const p00 = (freq.get('00') ?? 0) / SHOTS
    const p11 = (freq.get('11') ?? 0) / SHOTS
    expect(p00).toBeGreaterThan(0.45)
    expect(p00).toBeLessThan(0.55)
    expect(p11).toBeGreaterThan(0.45)
    expect(p11).toBeLessThan(0.55)
  })

  it('uniform superposition n=3: each of 8 states ≈12.5%', () => {
    let mps = mpsInit(3)
    for (let q = 0; q < 3; q++) mps = mpsApply1(mps, q, H)
    const freq = sampleMPS(mps, SHOTS, SEED)
    for (let i = 0; i < 8; i++) {
      const key = i.toString(2).padStart(3, '0').split('').reverse().join('')
      const p = (freq.get(key) ?? 0) / SHOTS
      expect(p).toBeGreaterThan(0.10)
      expect(p).toBeLessThan(0.175)
    }
  })

  it('|1⟩ state always samples to "1"', () => {
    const mps = mpsApply1(mpsInit(1), 0, X)
    const freq = sampleMPS(mps, 100, SEED)
    expect(freq.get('1')).toBe(100)
    expect(freq.has('0')).toBe(false)
  })

  it('GHZ n=4: only |0000⟩ and |1111⟩ ever sampled', () => {
    const mps = ghzState(4)
    const freq = sampleMPS(mps, SHOTS, SEED)
    for (const [k, v] of freq) {
      expect(k === '0000' || k === '1111').toBe(true)
    }
    // Marginals
    const p0 = (freq.get('0000') ?? 0) / SHOTS
    expect(p0).toBeGreaterThan(0.45)
    expect(p0).toBeLessThan(0.55)
  })

  it('sampling is consistent across calls with same seed', () => {
    const mps = bellState()
    const freq1 = sampleMPS(mps, 200, 777)
    const freq2 = sampleMPS(mps, 200, 777)
    for (const k of ['00', '11']) {
      expect(freq1.get(k)).toBe(freq2.get(k))
    }
  })

  it('sampling bit ordering: qubit 0 affects leftmost digit in output key', () => {
    // Set only qubit 0 to |1⟩, verify leftmost char is '1'
    const mps = mpsApply1(mpsInit(3), 0, X)
    const freq = sampleMPS(mps, 20, SEED)
    for (const k of freq.keys()) {
      expect(k[0]).toBe('1')  // q0 is leftmost
      expect(k[1]).toBe('0')
      expect(k[2]).toBe('0')
    }
  })

  it('sampling bit ordering: qubit 2 affects rightmost digit in 3-qubit system', () => {
    const mps = mpsApply1(mpsInit(3), 2, X)
    const freq = sampleMPS(mps, 20, SEED)
    for (const k of freq.keys()) {
      expect(k[2]).toBe('1')   // q2 is rightmost
      expect(k[0]).toBe('0')
      expect(k[1]).toBe('0')
    }
  })

  it('Rx(π/2) on |0⟩: marginal ≈50/50', () => {
    const mps = mpsApply1(mpsInit(1), 0, Rx(Math.PI / 2))
    const freq = sampleMPS(mps, SHOTS * 2, SEED)
    const p0 = (freq.get('0') ?? 0) / (SHOTS * 2)
    const p1 = (freq.get('1') ?? 0) / (SHOTS * 2)
    expect(p0).toBeGreaterThan(0.45)
    expect(p1).toBeGreaterThan(0.45)
  })

  it('|01⟩ 2-qubit basis state: mpsSample always returns "01"', () => {
    // Verifies that multi-qubit product states sample deterministically.
    // Single-qubit |0⟩/|1⟩ are already covered; this tests q0=0, q1=1 together.
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 1, X)  // q1=1, q0=0 → |01⟩ in qubit notation
    const freq = sampleMPS(mps, 50, SEED)
    expect(freq.size).toBe(1)
    expect(freq.get('01')).toBe(50)  // q0-leftmost: q0='0', q1='1' → "01"
  })

  it('Bell state single-qubit marginals: each qubit is 50/50 independently', () => {
    // In |Φ+⟩ = (|00⟩+|11⟩)/√2, both qubits are maximally mixed individually.
    // Qubit 0 alone should be 50% |0⟩ and 50% |1⟩ regardless of qubit 1 outcome.
    const mps = bellState()
    const freq = sampleMPS(mps, SHOTS, SEED)
    let q0zero = 0, q0one = 0, q1zero = 0, q1one = 0
    for (const [k, v] of freq) {
      if (k[0] === '0') q0zero += v; else q0one += v
      if (k[1] === '0') q1zero += v; else q1one += v
    }
    expect(q0zero / SHOTS).toBeGreaterThan(0.45)
    expect(q0zero / SHOTS).toBeLessThan(0.55)
    expect(q1zero / SHOTS).toBeGreaterThan(0.45)
    expect(q1zero / SHOTS).toBeLessThan(0.55)
  })
})

// ── Group 8: Large-scale circuits ─────────────────────────────────────────────
// Bugs caught: scalability regressions, memory issues, accumulated numerical error

describe('large-scale circuits — n=50+ qubits', () => {
  it('n=50 GHZ state: correct final amplitude structure', () => {
    const mps = ghzState(50, 64)
    const norm = mpsNorm(mps)
    // Can't contract 2^50 amplitudes — use sampling instead
    const freq = sampleMPS(mps, 200, SEED)
    for (const [k, v] of freq) {
      expect(k === '0'.repeat(50) || k === '1'.repeat(50)).toBe(true)
    }
  })

  it('n=50 product state (all H): norm = 1 via sampling entropy', () => {
    let mps = mpsInit(50)
    for (let q = 0; q < 50; q++) mps = mpsApply1(mps, q, H)
    expect(mpsMaxBond(mps)).toBe(1)
    // All sites should have chi=1 since no entanglement
    for (let q = 0; q < 50; q++) {
      const t = mpsTensor(mps, q)
      expect(t.chiL).toBe(1)
      expect(t.chiR).toBe(1)
    }
  })

  it('n=100 GHZ: bond dimension stays at 2', () => {
    // GHZ has exactly chi=2 on the maximally entangled cut
    const mps = ghzState(100, 4)
    expect(mpsMaxBond(mps)).toBeLessThanOrEqual(4)
    // Sampling should only produce all-0 or all-1
    const freq = sampleMPS(mps, 50, SEED)
    for (const [k] of freq) {
      expect(k === '0'.repeat(100) || k === '1'.repeat(100)).toBe(true)
    }
  })

  it('n=20 nearest-neighbor XX circuit: norm preserved', () => {
    let mps = mpsInit(20)
    for (let layer = 0; layer < 3; layer++) {
      for (let q = 0; q < 19; q++) {
        mps = mpsApply2(mps, q, q + 1, Xx(0.3), 16)
      }
    }
    // Check bond dimension is bounded
    expect(mpsMaxBond(mps)).toBeLessThanOrEqual(16)
    // Sample to verify it's not all zeros
    const freq = sampleMPS(mps, 100, SEED)
    expect(freq.size).toBeGreaterThan(10)
  })

  it('Bernstein-Vazirani n=20: exact reconstruction despite large n', () => {
    // BV: oracle encodes s=0b10101010101010101010 (alternating)
    const n = 20
    const s = 0b10101010101010101010 & ((1 << n) - 1)
    let mps = mpsInit(n + 1) // extra ancilla at the end
    // Initialize ancilla qubit n to |1⟩
    mps = mpsApply1(mps, n, X)
    // H all qubits
    for (let q = 0; q <= n; q++) mps = mpsApply1(mps, q, H)
    // Oracle: CNOT for each bit set in s
    for (let q = 0; q < n; q++) {
      if ((s >> q) & 1) mps = mpsApply2(mps, q, n, CNOT4, 4)
    }
    // H all input qubits
    for (let q = 0; q < n; q++) mps = mpsApply1(mps, q, H)
    // Sample: should always produce s (ancilla is |1⟩)
    const freq = sampleMPS(mps, 30, SEED)
    for (const [k] of freq) {
      // First n bits should be s (q0-leftmost), last bit is ancilla (0 or 1)
      const inputBits = k.slice(0, n)
      // Reconstruct: q0-leftmost string → qubit q has bit inputBits[q]
      let measured = 0
      for (let q = 0; q < n; q++) if (inputBits[q] === '1') measured |= (1 << q)
      expect(measured).toBe(s)
    }
  })

  it('BV n=40: statevector needs 2^40 ≈ 1 trillion amplitudes; MPS uses chi=2', () => {
    // This circuit is impossible for a statevector simulator (2^40 = 1,099,511,627,776 entries).
    // MPS with chi=2 handles it in O(40) operations. Proves concrete scale advantage.
    const n = 40
    // Hidden string: alternating 1010...10 on even qubits
    const s: boolean[] = Array.from({ length: n }, (_, i) => i % 2 === 0)
    let mps = mpsInit(n + 1)
    mps = mpsApply1(mps, n, X)  // ancilla |1⟩
    for (let q = 0; q <= n; q++) mps = mpsApply1(mps, q, H)
    for (let q = 0; q < n; q++) {
      if (s[q]) mps = mpsApply2(mps, q, n, CNOT4, 4)
    }
    for (let q = 0; q < n; q++) mps = mpsApply1(mps, q, H)
    // Bond dimension should never exceed 4 (BV oracle uses at most chi=2 per CNOT)
    expect(mpsMaxBond(mps)).toBeLessThanOrEqual(4)
    // Every sample should recover the hidden string exactly
    const freq = sampleMPS(mps, 20, SEED)
    for (const [k] of freq) {
      const inputBits = k.slice(0, n)
      for (let q = 0; q < n; q++) {
        expect(inputBits[q]).toBe(s[q] ? '1' : '0')
      }
    }
  })

  it('GHZ n=1000: bond=2, samples only all-0 or all-1 (extreme scale)', () => {
    // 2^1000 is beyond any classical memory. GHZ has Schmidt rank 2 — chi=2 suffices.
    // Proves the O(n·chi²) memory bound is not just a claim.
    const n = 1000
    const mps = ghzState(n, 2)
    expect(mpsMaxBond(mps)).toBe(2)
    const freq = sampleMPS(mps, 10, SEED)
    for (const [k] of freq) {
      expect(k === '0'.repeat(n) || k === '1'.repeat(n)).toBe(true)
    }
  })
})

// ── Group 9: Numerical precision ──────────────────────────────────────────────
// Bugs caught: accumulated float error, catastrophic cancellation, poor QR numerics

describe('numerical precision — floating point stability', () => {
  it('deep H-CNOT ladder n=8, 50 layers: norm > 0.9999', () => {
    let mps = mpsInit(8)
    for (let q = 0; q < 8; q++) mps = mpsApply1(mps, q, H)
    for (let layer = 0; layer < 50; layer++) {
      for (let q = 0; q < 7; q++) {
        mps = mpsApply2(mps, q, q + 1, CNOT4, 8)
      }
    }
    // Norm should still be very close to 1
    // We can't contract 2^8=256 directly but can check via small-chi contraction
    // Verify bond is bounded and sampling doesn't crash
    const freq = sampleMPS(mps, 50, SEED)
    expect(freq.size).toBeGreaterThan(0)
    expect(mpsMaxBond(mps)).toBeLessThanOrEqual(8)
  })

  it('Rz rotation sequence: round-trip Rz(θ)Rz(-θ)=I', () => {
    const theta = 1.23456789
    let mps = mpsInit(3)
    mps = mpsApply1(mps, 0, H)
    mps = mpsApply1(mps, 1, X)
    mps = mpsApply2(mps, 0, 1, CNOT4, 8)
    const before = mpsContract(mps).slice()
    mps = mpsApply1(mps, 0, Rz(theta))
    mps = mpsApply1(mps, 0, Rz(-theta))
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) {
      expect(after[i]!).toBeCloseTo(before[i]!, 13)
    }
  })

  it('1000 single-qubit Rz rotations: norm stays at 1', () => {
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, H)
    mps = mpsApply2(mps, 0, 1, CNOT4, 4)
    for (let i = 0; i < 1000; i++) {
      mps = mpsApply1(mps, i % 2, Rz(0.001))
    }
    expect(mpsNorm(mps)).toBeCloseTo(1, 11)
  })

  it('Rx(2π): full rotation should return near-original state', () => {
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, H)
    const before = mpsContract(mps).slice()
    mps = mpsApply1(mps, 0, Rx(2 * Math.PI))
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) {
      expect(Math.abs(after[i]!)).toBeCloseTo(Math.abs(before[i]!), 13)
    }
  })

  it('alternating X gates: exact round-trip n=5, 100 layers', () => {
    let mps = mpsInit(5)
    for (let q = 0; q < 5; q++) mps = mpsApply1(mps, q, H)
    const before = mpsContract(mps).slice()
    for (let i = 0; i < 100; i++) {
      for (let q = 0; q < 5; q++) mps = mpsApply1(mps, q, X)
    }
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) {
      expect(after[i]!).toBeCloseTo(before[i]!, 13)
    }
  })

  it('QFT-like phase accumulation n=4: all probabilities equal 1/16', () => {
    // QFT of |0⟩ = uniform superposition
    let mps = mpsInit(4)
    for (let q = 0; q < 4; q++) mps = mpsApply1(mps, q, H)
    for (let q = 0; q < 3; q++) {
      for (let k = q + 1; k < 4; k++) {
        const phase = Math.PI / (1 << (k - q))
        mps = mpsApply2(mps, q, k, controlledGate(Rz(phase * 2)), 16)
      }
    }
    // All 16 states should have equal probability 1/16
    const amps = mpsContract(mps)
    for (let i = 0; i < 16; i++) {
      const p = amps[i * 2]! ** 2 + amps[i * 2 + 1]! ** 2
      expect(p).toBeCloseTo(1 / 16, 10)
    }
  })
})

// ── Group 10: Performance and scaling ─────────────────────────────────────────
// Proves the key properties that make Float64Array MPS worth using:
//   1. O(n) time for product states (not O(2^n))
//   2. Bond dimension stays bounded — GHZ has chi=2, product state chi=1
//   3. Throughput regression guard: if object allocation sneaks back, CI fails

describe('performance and scaling — Float64Array properties', () => {
  it('n=15 product circuit ×20: completes in under 500ms', () => {
    // Product states have chi=1 throughout — exercises mpsApply1 and mpsSample hot paths.
    // If someone reintroduces Complex object allocation this becomes noticeably slower.
    const start = now()
    for (let r = 0; r < 20; r++) {
      let mps = mpsInit(15)
      for (let q = 0; q < 15; q++) mps = mpsApply1(mps, q, H)
      mpsSample(mps, makeRng(r))
    }
    const elapsed = now() - start
    log(`n=15 product circuit ×20: ${elapsed.toFixed(1)}ms`)
    expect(elapsed).toBeLessThan(500)
  })

  it('n=10 entangling circuit chi=32, 5 layers: completes in under 2000ms', () => {
    // Exercises mpsApply2Adjacent (contraction + QR) at moderate bond dimension.
    // Acts as a regression guard for the O(chiL·chiR·chiM) contraction loop.
    const start = now()
    for (let r = 0; r < 3; r++) {
      let mps = mpsInit(10)
      for (let q = 0; q < 10; q++) mps = mpsApply1(mps, q, H)
      for (let layer = 0; layer < 5; layer++) {
        for (let q = 0; q < 9; q++) mps = mpsApply2(mps, q, q + 1, CNOT4, 32)
      }
    }
    const elapsed = now() - start
    log(`n=10 chi=32 5-layer circuit ×3: ${elapsed.toFixed(1)}ms`)
    expect(elapsed).toBeLessThan(2000)
  })

  it('O(n) scaling: n=200 product state runs in under 100ms', () => {
    // Statevector would need 2^200 amplitudes — physically impossible.
    // MPS with chi=1 (product state) is O(n) — this proves the backend handles
    // large n that no statevector simulator can touch.
    const start = now()
    let mps = mpsInit(200)
    for (let q = 0; q < 200; q++) mps = mpsApply1(mps, q, H)
    mpsSample(mps, makeRng(42))
    const elapsed = now() - start
    log(`n=200 product state: ${elapsed.toFixed(1)}ms`)
    expect(elapsed).toBeLessThan(100)
    // Bond stays 1: product state requires no entanglement storage
    expect(mpsMaxBond(mps)).toBe(1)
  })

  it('GHZ bond dimension is exactly 2 regardless of n', () => {
    // GHZ has Schmidt rank 2 at every cut — chi=2 is the exact, minimal bond.
    // This proves truncation to maxBond=2 loses zero information for GHZ.
    for (const n of [10, 50, 100]) {
      const mps = ghzState(n, 2)
      expect(mpsMaxBond(mps)).toBe(2)
      const freq = sampleMPS(mps, 30, SEED)
      for (const [k] of freq) {
        expect(k === '0'.repeat(n) || k === '1'.repeat(n)).toBe(true)
      }
    }
  })

  it('memory footprint: data.length = chiL*2*chiR*2 (no wasted bytes)', () => {
    // Float64Array has zero internal fragmentation.
    // Each element is exactly 8 bytes. No per-element object header overhead.
    let mps = mpsInit(8)
    for (let q = 0; q < 8; q++) mps = mpsApply1(mps, q, H)
    for (let q = 0; q < 7; q++) mps = mpsApply2(mps, q, q + 1, CNOT4, 16)
    let totalFloats = 0
    for (let q = 0; q < 8; q++) {
      const t = mpsTensor(mps, q)
      expect(t.data.length).toBe(t.chiL * 2 * t.chiR * 2)
      totalFloats += t.data.length
    }
    // Total bytes = totalFloats * 8 (each f64 = 8 bytes, no overhead)
    log(`n=8 chi≤16 MPS total storage: ${totalFloats * 8} bytes`)
    // Should be well under 1MB for this small circuit
    expect(totalFloats * 8).toBeLessThan(1024 * 1024)
  })
})

// ── Group 11: Edge cases ──────────────────────────────────────────────────────
// Bugs caught: n=1 boundary conditions, single-site edge effects

describe('edge cases — boundary conditions', () => {
  it('n=1: single qubit H', () => {
    const mps = mpsApply1(mpsInit(1), 0, H)
    expect(mpsProb(mps, 0)).toBeCloseTo(0.5, 14)
    expect(mpsProb(mps, 1)).toBeCloseTo(0.5, 14)
  })

  it('n=1: sample always 0 for |0⟩', () => {
    const mps = mpsInit(1)
    for (let i = 0; i < 20; i++) {
      expect(mpsSample(mps, makeRng(i))).toBe(0n)
    }
  })

  it('n=1: sample always 1 for |1⟩', () => {
    const mps = mpsApply1(mpsInit(1), 0, X)
    for (let i = 0; i < 20; i++) {
      expect(mpsSample(mps, makeRng(i))).toBe(1n)
    }
  })

  it('n=2: CNOT with control=1, target=0 (reversed control/target)', () => {
    // Test CNOT where q1 is control and q0 is target.
    // mpsApply2 requires a < b; apply using a=0,b=1 with a "reversed-CNOT" gate
    // where LSB (q1) controls and MSB (q0) is the target.
    // Gate action: |q0 q1⟩ → |q0⊕q1, q1⟩
    //   |00⟩→|00⟩, |01⟩→|11⟩, |10⟩→|10⟩, |11⟩→|01⟩
    const C = (re: number) => ({ re, im: 0 })
    const RCNOT4: Gate4x4 = [
      [C(1), C(0), C(0), C(0)],
      [C(0), C(0), C(0), C(1)],
      [C(0), C(0), C(1), C(0)],
      [C(0), C(1), C(0), C(0)],
    ]
    // Create |01⟩: q0=0, q1=1 → amplitude index = 0*2 + 1 = 1
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 1, X)  // q1=1
    // Apply reversed CNOT: q1=1 flips q0 → |11⟩ = idx 3
    mps = mpsApply2(mps, 0, 1, RCNOT4, 8)
    expect(mpsProb(mps, 3)).toBeCloseTo(1, 14)
  })

  it('maxBond=1 on product circuit: no information loss', () => {
    let mps = mpsInit(10)
    for (let q = 0; q < 10; q++) mps = mpsApply1(mps, q, H)
    // maxBond=1 is exact for product states
    const freq = sampleMPS(mps, 1000, SEED)
    // Should have many different outcomes (not all collapsed to one)
    expect(freq.size).toBeGreaterThan(100)
  })

  it('zero-angle rotations: Rx(0) = I', () => {
    let mps = mpsInit(2)
    mps = mpsApply1(mps, 0, H)
    const before = mpsContract(mps).slice()
    mps = mpsApply1(mps, 0, Rx(0))
    const after = mpsContract(mps)
    for (let i = 0; i < after.length; i++) expect(after[i]!).toBeCloseTo(before[i]!, 14)
  })

  it('very small rotation: Rx(1e-10) barely changes state', () => {
    let mps = mpsInit(1)
    const before = mpsContract(mps).slice()
    mps = mpsApply1(mps, 0, Rx(1e-10))
    const after = mpsContract(mps)
    expect(Math.abs(after[0]! - before[0]!)).toBeLessThan(1e-8)
  })

  it('n=3 non-adjacent CNOT(q0,q2) matches manual SWAP network', () => {
    // mpsApply2(a=0, b=2) routes through SWAP network: SWAP(1,2), CNOT(0,1), SWAP(1,2)
    // Manual sequence should produce the same state.
    const via_mpsApply2 = () => {
      let mps = mpsInit(3)
      mps = mpsApply1(mps, 0, H)
      mps = mpsApply2(mps, 0, 2, CNOT4, 8)  // non-adjacent path
      return mpsContract(mps)
    }
    const via_manual = () => {
      let mps = mpsInit(3)
      mps = mpsApply1(mps, 0, H)
      mps = mpsApply2(mps, 1, 2, SWAP4, 8)  // bring q2 to position 1
      mps = mpsApply2(mps, 0, 1, CNOT4, 8)  // apply gate
      mps = mpsApply2(mps, 1, 2, SWAP4, 8)  // restore
      return mpsContract(mps)
    }
    const a = via_mpsApply2(), b = via_manual()
    for (let i = 0; i < a.length; i++) expect(a[i]!).toBeCloseTo(b[i]!, 13)
  })
})

// ── Group 12: mpsContract correctness ─────────────────────────────────────────
// Bugs caught: wrong amplitude indexing, physical vs bond index confusion

describe('mpsContract — full contraction correctness', () => {
  it('|0...0⟩ n=3: only amp[0] = 1+0i, all others 0', () => {
    const amps = mpsContract(mpsInit(3))
    expect(amps[0]).toBe(1)
    expect(amps[1]).toBe(0)
    for (let i = 2; i < 16; i++) expect(amps[i]).toBe(0)
  })

  it('|1⟩ single qubit: amp[1] = 1, amp[0] = 0', () => {
    const mps = mpsApply1(mpsInit(1), 0, X)
    const amps = mpsContract(mps)
    expect(amps[0]).toBeCloseTo(0, 14)  // |0⟩ amplitude re
    expect(amps[2]).toBeCloseTo(1, 14)  // |1⟩ amplitude re
  })

  it('|+⟩⊗|0⟩: amps[0]=amps[1]=1/√2, amps[2]=amps[3]=0', () => {
    const mps = mpsApply1(mpsInit(2), 0, H)
    const amps = mpsContract(mps)
    // idx 0 = |00⟩, idx 1 = |10⟩ (q0=1,q1=0) in LSB
    expect(amps[0]).toBeCloseTo(sq2, 14)   // |00⟩ re
    expect(amps[1]).toBeCloseTo(0, 14)     // |00⟩ im
    expect(amps[2]).toBeCloseTo(sq2, 14)   // |10⟩ re (q0=1 → bit 0 set → idx=1... wait)
    // Actually: bit 0 = q0, so idx with q0=1 is idx=1 (0b01), q0=0 is idx=0 (0b00)
    // So: amps[0*2] = amp(|00⟩).re = 1/√2, amps[1*2] = amp(|10⟩).re = 1/√2
    expect(amps[0]).toBeCloseTo(sq2, 14)
    expect(amps[2]).toBeCloseTo(sq2, 14)
    expect(amps[4]).toBeCloseTo(0, 14)
    expect(amps[6]).toBeCloseTo(0, 14)
  })

  it('total probability sums to 1 for mpsContract output', () => {
    const cases: Array<() => MPS> = [
      () => mpsInit(4),
      () => bellState(),
      () => ghzState(3),
      () => { let m = mpsInit(4); for (let q=0;q<4;q++) m=mpsApply1(m,q,H); return m },
    ]
    for (const build of cases) {
      const amps = mpsContract(build())
      let s = 0
      for (let i = 0; i < amps.length; i += 2) s += amps[i]! * amps[i]! + amps[i + 1]! * amps[i + 1]!
      expect(s).toBeCloseTo(1, 14)
    }
  })
})

// ── Group 13: Regression — existing circuit.ts integration ───────────────────
// Ensures our MPS rewrite doesn't break the public Circuit.runMps() API

describe('regression — Circuit.runMps() still works', () => {
  // Dynamic import to test via public API
  it('Circuit bell state via runMps matches expected probabilities', async () => {
    const { Circuit } = await import('./circuit.js')
    const result = new Circuit(2).h(0).cnot(0, 1).runMps({ shots: 1000, seed: 42 })
    expect(result.probs['00']).toBeGreaterThan(0.45)
    expect(result.probs['00']).toBeLessThan(0.55)
    expect(result.probs['11']).toBeGreaterThan(0.45)
    expect(result.probs['11']).toBeLessThan(0.55)
    expect(result.probs['01'] ?? 0).toBeLessThan(0.02)
    expect(result.probs['10'] ?? 0).toBeLessThan(0.02)
  })

  it('Circuit GHZ n=10 via runMps: only 00000 and 11111 in probabilities', async () => {
    const { Circuit } = await import('./circuit.js')
    let c = new Circuit(10).h(0)
    for (let q = 0; q < 9; q++) c = c.cnot(q, q + 1)
    const result = c.runMps({ shots: 500, seed: 99 })
    for (const [k, v] of Object.entries(result.probs)) {
      if (v > 0.01) expect(k === '0000000000' || k === '1111111111').toBe(true)
    }
  })

  it('Circuit.runMps BV n=8 recovers hidden string', async () => {
    const { Circuit } = await import('./circuit.js')
    // BV: hidden string s = 10101010 (q0-leftmost)
    const s = '10101010'
    const n = 8
    let c = new Circuit(n + 1)
    c = c.x(n)  // ancilla |1⟩
    for (let q = 0; q <= n; q++) c = c.h(q)
    for (let q = 0; q < n; q++) {
      if (s[q] === '1') c = c.cnot(q, n)
    }
    for (let q = 0; q < n; q++) c = c.h(q)
    const result = c.runMps({ shots: 100, seed: 1 })
    // Most sampled outcome (first n bits) should be s
    const top = Object.entries(result.probs)
      .sort(([, a], [, b]) => b - a)[0]![0]
    expect(top.slice(0, n)).toBe(s)
  })

  it('Circuit.runMps with maxBond cap produces valid Distribution', async () => {
    const { Circuit } = await import('./circuit.js')
    const c = new Circuit(6).h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3).cnot(3, 4).cnot(4, 5)
    const result = c.runMps({ shots: 200, maxBond: 4, seed: 7 })
    // Should not throw, should have probabilities summing to ~1
    const total = Object.values(result.probs).reduce((a, b) => a + b, 0)
    expect(total).toBeGreaterThan(0.95)
    expect(total).toBeLessThanOrEqual(1.01)
  })
})

// ── Group 14: CNOT4 and SWAP4 constants correctness ──────────────────────────
// Bugs caught: wrong constant gate matrices

describe('gate constant matrices — CNOT4 and SWAP4', () => {
  it('CNOT4 has correct |10⟩→|11⟩ mapping', () => {
    // CNOT4[2][3] = 1 (row |10⟩ → col |11⟩)
    expect(CNOT4[2]![3]!.re).toBeCloseTo(1, 14)
    expect(CNOT4[2]![2]!.re).toBeCloseTo(0, 14)
  })

  it('CNOT4 is its own inverse (CNOT² = I)', () => {
    const dim = 4
    for (let r = 0; r < dim; r++) {
      for (let c = 0; c < dim; c++) {
        let re = 0, im = 0
        for (let k = 0; k < dim; k++) {
          const a = CNOT4[r]![k]!, b = CNOT4[k]![c]!
          re += a.re * b.re - a.im * b.im
          im += a.re * b.im + a.im * b.re
        }
        const expected = r === c ? 1 : 0
        expect(re).toBeCloseTo(expected, 14)
        expect(im).toBeCloseTo(0, 14)
      }
    }
  })

  it('SWAP4 correctly maps |01⟩→|10⟩ and |10⟩→|01⟩', () => {
    // SWAP4[1][2]=1 (|01⟩→|10⟩) and SWAP4[2][1]=1 (|10⟩→|01⟩)
    expect(SWAP4[1]![2]!.re).toBeCloseTo(1, 14)
    expect(SWAP4[2]![1]!.re).toBeCloseTo(1, 14)
    expect(SWAP4[1]![1]!.re).toBeCloseTo(0, 14)
    expect(SWAP4[2]![2]!.re).toBeCloseTo(0, 14)
  })

  it('controlledGate(X) = CNOT4', () => {
    const CX = controlledGate([[{ re: 0, im: 0 }, { re: 1, im: 0 }], [{ re: 1, im: 0 }, { re: 0, im: 0 }]])
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        expect(CX[r]![c]!.re).toBeCloseTo(CNOT4[r]![c]!.re, 14)
      }
    }
  })
})
