import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { CNOT4, mpsApply1, mpsApply2, mpsExpectation, mpsInit, type MPS } from './mps.js'
import * as G from './gates.js'

/**
 * Expectation values straight off the MPS, with no sampling.
 *
 * The reason this exists: judging whether a *truncated* MPS has converged means
 * comparing two runs, and comparing two sampled distributions has a noise floor
 * larger than any tolerance worth setting — a volume-law circuit whose truncation
 * is doing real damage reads as converged. A transfer-matrix contraction carries
 * no shot noise, so the comparison can actually resolve the thing it is asking
 * about.
 */

/** ⟨Z_q⟩, ⟨X_q⟩ or ⟨Y_q⟩ from the exact distribution, via a rotated readout. */
function exactExpectation(c: Circuit, q: number, pauli: 'X' | 'Y' | 'Z'): number {
  const rotated = pauli === 'Z' ? c : pauli === 'X' ? c.h(q) : c.sdg(q).h(q)
  let v = 0
  for (const [bits, p] of Object.entries(rotated.exactProbs())) v += (bits[q] === '1' ? -1 : 1) * p
  return v
}

/** Run a small circuit through the MPS kernels, exactly (bond growth uncapped). */
function toMps(build: (m: MPS) => MPS, n: number): MPS {
  return build(mpsInit(n))
}

describe('mpsExpectation — matches the statevector, without sampling', () => {
  it('on an entangled chain, for every Pauli and every site', () => {
    const n = 8
    const c = (() => {
      let k = new Circuit(n).h(0)
      for (let i = 0; i < n - 1; i++) k = k.cnot(i, i + 1)
      return k.ry(0.6, 3).rz(0.4, 5).rx(0.9, 6)
    })()
    const mps = toMps(m => {
      let s = mpsApply1(m, 0, G.H)
      for (let i = 0; i < n - 1; i++) s = mpsApply2(s, i, i + 1, CNOT4, 256)
      s = mpsApply1(s, 3, G.Ry(0.6))
      s = mpsApply1(s, 5, G.Rz(0.4))
      s = mpsApply1(s, 6, G.Rx(0.9))
      return s
    }, n)

    for (const pauli of ['X', 'Y', 'Z'] as const) {
      for (const q of [0, 3, 5, 7]) {
        expect(mpsExpectation(mps, q, pauli), `${pauli} on q${q}`)
          .toBeCloseTo(exactExpectation(c, q, pauli), 9)
      }
    }
  })

  it('on a product state, where the answers are known outright', () => {
    const n = 5
    const mps = toMps(m => {
      let s = m
      s = mpsApply1(s, 1, G.X)                 // |1⟩ ⇒ ⟨Z⟩ = −1
      s = mpsApply1(s, 2, G.H)                 // |+⟩ ⇒ ⟨X⟩ = +1, ⟨Z⟩ = 0
      return s
    }, n)
    expect(mpsExpectation(mps, 0, 'Z')).toBeCloseTo(1, 12)
    expect(mpsExpectation(mps, 1, 'Z')).toBeCloseTo(-1, 12)
    expect(mpsExpectation(mps, 2, 'Z')).toBeCloseTo(0, 12)
    expect(mpsExpectation(mps, 2, 'X')).toBeCloseTo(1, 12)
  })

  it('a GHZ chain has zero single-site magnetisation everywhere', () => {
    const n = 6
    const mps = toMps(m => {
      let s = mpsApply1(m, 0, G.H)
      for (let i = 0; i < n - 1; i++) s = mpsApply2(s, i, i + 1, CNOT4, 64)
      return s
    }, n)
    for (let q = 0; q < n; q++) expect(mpsExpectation(mps, q, 'Z'), `q${q}`).toBeCloseTo(0, 10)
  })

  it('does not need a particular canonical form — the norm is computed alongside', () => {
    // Scale one tensor: the expectation is a ratio, so it must be unchanged.
    const n = 4
    const mps = toMps(m => {
      let s = mpsApply1(m, 0, G.H)
      for (let i = 0; i < n - 1; i++) s = mpsApply2(s, i, i + 1, CNOT4, 32)
      return mpsApply1(s, 2, G.Ry(0.7))
    }, n)
    const before = mpsExpectation(mps, 2, 'Z')
    const scaled = mps.map((t, i) => i === 1
      ? { ...t, data: Float64Array.from(t.data, x => x * 3.5) }
      : t)
    expect(mpsExpectation(scaled, 2, 'Z')).toBeCloseTo(before, 10)
  })
})
