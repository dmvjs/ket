import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'

/**
 * A circuit whose measurements are all terminal is simulated once and sampled,
 * rather than re-simulated per shot. These tests pin the two things that can go
 * wrong: routing a circuit to the fast path when it is not eligible, and losing
 * classical-register bookkeeping once it is.
 *
 * The slow path is forced by appending a `reset` on a spare qubit that is
 * already |0⟩ — physically a no-op, so both paths must agree on everything.
 */

/** Bell pair with both qubits measured into a 2-bit register. */
const bell = (): Circuit =>
  new Circuit(2).h(0).cnot(0, 1).creg('out', 2).measure(0, 'out', 0).measure(1, 'out', 1)

describe('terminal measurement — fast path eligibility', () => {
  it('a gates-then-measure circuit produces correct outcomes', () => {
    const d = bell().run({ shots: 20_000, seed: 7 })
    // Bell state: only |00⟩ and |11⟩, roughly balanced.
    expect(Object.keys(d.probs).toSorted()).toEqual(['00', '11'])
    expect(Math.abs(d.probs['00']! - 0.5)).toBeLessThan(0.02)
  })

  it('classical registers are populated on the fast path', () => {
    const d = bell().run({ shots: 20_000, seed: 7 })
    // Each creg bit is 1 exactly when its qubit measured 1 — about half the time.
    expect(Math.abs(d.cregs['out']![0]! - 0.5)).toBeLessThan(0.02)
    expect(Math.abs(d.cregs['out']![1]! - 0.5)).toBeLessThan(0.02)
  })

  it('creg bits are exact for a deterministic circuit', () => {
    // |10⟩ with certainty: q0 measures 1 every shot, q1 measures 0 every shot.
    const d = new Circuit(2).x(0).creg('r', 2)
      .measure(0, 'r', 0).measure(1, 'r', 1)
      .run({ shots: 500, seed: 3 })
    expect(d.probs).toEqual({ '10': 1 })
    expect(d.cregs['r']![0]).toBe(1)
    expect(d.cregs['r']![1]).toBe(0)
  })

  it('a barrier between gates and measurement does not disqualify', () => {
    const d = new Circuit(2).h(0).cnot(0, 1).barrier().creg('out', 2)
      .measure(0, 'out', 0).measure(1, 'out', 1)
      .run({ shots: 10_000, seed: 5 })
    expect(Object.keys(d.probs).toSorted()).toEqual(['00', '11'])
  })

  it('measuring one qubit into several creg bits sets all of them', () => {
    const d = new Circuit(1).x(0).creg('r', 2)
      .measure(0, 'r', 0).measure(0, 'r', 1)
      .run({ shots: 100, seed: 1 })
    expect(d.cregs['r']!).toEqual([1, 1])
  })

  it('an unmeasured qubit still contributes to the outcome bitstring', () => {
    const d = new Circuit(2).x(1).creg('r', 1).measure(0, 'r', 0)
      .run({ shots: 100, seed: 1 })
    expect(d.probs).toEqual({ '01': 1 })
    expect(d.cregs['r']![0]).toBe(0)
  })
})

describe('terminal measurement — ineligible circuits keep the per-shot path', () => {
  it('a gate after a measurement on the same qubit is handled correctly', () => {
    // H, measure, H on |0⟩: the measurement collapses, so the second H acts on a
    // definite state and the result is 50/50 — not the |0⟩ that coherent
    // H·H would give. Routing this to the fast path would return '0' always.
    const d = new Circuit(1).h(0).creg('c', 1).measure(0, 'c', 0).h(0)
      .run({ shots: 20_000, seed: 4 })
    expect(Math.abs(d.probs['0']! - 0.5)).toBeLessThan(0.02)
    expect(Math.abs(d.probs['1']! - 0.5)).toBeLessThan(0.02)
  })

  it('a two-qubit gate touching a measured qubit disqualifies', () => {
    // Measuring q0 then entangling it with q1 must see the collapsed q0.
    const d = new Circuit(2).h(0).creg('c', 1).measure(0, 'c', 0).cnot(0, 1)
      .run({ shots: 20_000, seed: 6 })
    // Post-collapse CNOT copies q0 into q1, so only |00⟩ and |11⟩ appear.
    expect(Object.keys(d.probs).toSorted()).toEqual(['00', '11'])
  })

  it('reset disqualifies', () => {
    const d = new Circuit(1).x(0).creg('c', 1).measure(0, 'c', 0).reset(0)
      .run({ shots: 100, seed: 1 })
    // Measured 1 into the register, then reset the qubit to |0⟩.
    expect(d.cregs['c']![0]).toBe(1)
    expect(d.probs).toEqual({ '0': 1 })
  })

  it('classical feedback (if) disqualifies', () => {
    // Measure q0, then flip q1 conditioned on it. q0 is |1⟩ with certainty.
    const d = new Circuit(2).x(0).creg('c', 1).measure(0, 'c', 0)
      .if('c', 1, k => k.x(1))
      .run({ shots: 100, seed: 1 })
    expect(d.probs).toEqual({ '11': 1 })
  })
})

describe('terminal measurement — fast and per-shot paths agree', () => {
  /**
   * Same physics, different route: appending `reset` on a spare |0⟩ qubit forces
   * the per-shot path without changing the distribution over the other qubits.
   */
  it('a Bell pair agrees whichever path runs it', () => {
    const fast = new Circuit(3).h(0).cnot(0, 1).creg('out', 2)
      .measure(0, 'out', 0).measure(1, 'out', 1)
      .run({ shots: 40_000, seed: 21 })

    const slow = new Circuit(3).h(0).cnot(0, 1).creg('out', 2)
      .measure(0, 'out', 0).measure(1, 'out', 1)
      .reset(2)                                   // spare qubit, already |0⟩
      .run({ shots: 40_000, seed: 21 })

    expect(Object.keys(slow.probs).toSorted()).toEqual(Object.keys(fast.probs).toSorted())
    for (const k of Object.keys(fast.probs)) {
      expect(Math.abs(fast.probs[k]! - slow.probs[k]!), `outcome ${k}`).toBeLessThan(0.02)
    }
    for (let b = 0; b < 2; b++) {
      expect(Math.abs(fast.cregs['out']![b]! - slow.cregs['out']![b]!)).toBeLessThan(0.02)
    }
  })

  it('a densifying 8-qubit circuit agrees whichever path runs it', () => {
    const build = (n: number): Circuit => {
      let k = new Circuit(n)
      for (let q = 0; q < 8; q++) k = k.h(q).t(q)
      for (let q = 0; q < 7; q += 2) k = k.cnot(q, q + 1)
      for (let q = 0; q < 8; q++) k = k.ry(0.4 * (q + 1), q)
      k = k.creg('m', 8)
      for (let q = 0; q < 8; q++) k = k.measure(q, 'm', q)
      return k
    }
    const fast = build(9).run({ shots: 40_000, seed: 33 })
    const slow = build(9).reset(8).run({ shots: 40_000, seed: 33 })

    for (const k of Object.keys(fast.probs)) {
      if (fast.probs[k]! < 0.01) continue
      expect(Math.abs(fast.probs[k]! - (slow.probs[k] ?? 0)), `outcome ${k}`).toBeLessThan(0.015)
    }
    for (let b = 0; b < 8; b++) {
      expect(Math.abs(fast.cregs['m']![b]! - slow.cregs['m']![b]!), `bit ${b}`).toBeLessThan(0.015)
    }
  })

  it('fast-path outcome probabilities track exactProbs', () => {
    let k = new Circuit(6)
    for (let q = 0; q < 6; q++) k = k.h(q).t(q)
    k = k.cnot(0, 1).cz(2, 3).cry(0.9, 4, 5)
    const exact = k.exactProbs()

    let measured = k.creg('m', 6)
    for (let q = 0; q < 6; q++) measured = measured.measure(q, 'm', q)
    const sampled = measured.run({ shots: 100_000, seed: 12 }).probs

    for (const [bits, p] of Object.entries(exact)) {
      if (p < 0.005) continue
      expect(Math.abs((sampled[bits] ?? 0) - p), `outcome ${bits}`).toBeLessThan(0.01)
    }
  })
})
