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
    const fast = build(9).run({ shots: 15_000, seed: 33 })
    const slow = build(9).reset(8).run({ shots: 15_000, seed: 33 })

    for (const k of Object.keys(fast.probs)) {
      if (fast.probs[k]! < 0.01) continue
      expect(Math.abs(fast.probs[k]! - (slow.probs[k] ?? 0)), `outcome ${k}`).toBeLessThan(0.025)
    }
    for (let b = 0; b < 8; b++) {
      expect(Math.abs(fast.cregs['m']![b]! - slow.cregs['m']![b]!), `bit ${b}`).toBeLessThan(0.025)
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

describe('terminal measurement — MPS backend', () => {
  /**
   * `runMps()` had the same defect `run()` did: any measure op forced a full
   * re-simulation per shot, even when every measurement was terminal. On a
   * 30-qubit depth-4 circuit that was 0.18s for 1024 shots against 0.012s once
   * the state is built one time and sampled — and it scaled with shot count.
   */

  /** Brickwork of generic rotations; entanglement stays low enough for MPS. */
  const chain = (n: number, layers: number): Circuit => {
    let k = new Circuit(n)
    let a = 0.3
    for (let l = 0; l < layers; l++) {
      for (let q = 0; q < n; q++) { a += 0.31; k = k.ry(a, q) }
      for (let q = l % 2; q < n - 1; q += 2) { a += 0.17; k = k.crx(a, q, q + 1) }
    }
    return k
  }

  const measured = (k: Circuit, n: number): Circuit => {
    let m = k.creg('o', n)
    for (let q = 0; q < n; q++) m = m.measure(q, 'o', q)
    return m
  }

  it('sampled outcomes match exactProbs', () => {
    const base = chain(8, 3)
    const exact = base.exactProbs()
    const d = measured(base, 8).runMps({ shots: 200_000, seed: 5 })
    for (const [bits, p] of Object.entries(exact)) {
      if (p < 0.002) continue
      expect(Math.abs((d.probs[bits] ?? 0) - p), `outcome ${bits}`).toBeLessThan(0.005)
    }
  }, 60_000)

  it('creg bits match the exact single-qubit marginals', () => {
    const base = chain(8, 3)
    const exact = base.exactProbs()
    const d = measured(base, 8).runMps({ shots: 200_000, seed: 5 })
    for (let q = 0; q < 8; q++) {
      let p1 = 0
      for (const [bits, p] of Object.entries(exact)) if (bits[q] === '1') p1 += p
      expect(Math.abs(d.cregs['o']![q]! - p1), `qubit ${q}`).toBeLessThan(0.005)
    }
  }, 60_000)

  it('a deterministic circuit gives exact creg bits', () => {
    let k = new Circuit(4).x(0).x(2)
    k = k.creg('r', 4)
    for (let q = 0; q < 4; q++) k = k.measure(q, 'r', q)
    const d = k.runMps({ shots: 500, seed: 1 })
    expect(d.probs).toEqual({ '1010': 1 })
    expect(d.cregs['r']).toEqual([1, 0, 1, 0])
  })

  it('a wide GHZ with terminal measures stays cheap and correct', () => {
    let ghz = new Circuit(30).h(0)
    for (let i = 0; i < 29; i++) ghz = ghz.cnot(i, i + 1)
    const d = measured(ghz.t(0), 30).runMps({ shots: 4000, seed: 2 })
    expect(Object.keys(d.probs).toSorted()).toEqual(['0'.repeat(30), '1'.repeat(30)])
    expect(Math.abs(d.probs['0'.repeat(30)]! - 0.5)).toBeLessThan(0.03)
    expect(d.peakChi).toBe(2)
  })

  it('shot count no longer drives the cost', () => {
    // The old path re-simulated per shot, so 20x the shots cost 20x the time.
    const m = measured(chain(20, 3), 20)
    const t1 = performance.now(); m.runMps({ shots: 1000, seed: 1 });  const small = performance.now() - t1
    const t2 = performance.now(); m.runMps({ shots: 20000, seed: 1 }); const large = performance.now() - t2
    // Sampling is not free, but 20x the shots must cost far less than 20x the time.
    expect(large).toBeLessThan(small * 10 + 50)
  }, 60_000)

  it('non-terminal measurement still takes the per-shot path', () => {
    // A gate after the measurement means the collapse matters.
    const d = new Circuit(2).h(0).creg('c', 1).measure(0, 'c', 0).cnot(0, 1)
      .runMps({ shots: 8000, seed: 4 })
    expect(Object.keys(d.probs).toSorted()).toEqual(['00', '11'])
  }, 60_000)

  it('reset and classical feedback still route per-shot', () => {
    const r = new Circuit(1).x(0).creg('c', 1).measure(0, 'c', 0).reset(0)
      .runMps({ shots: 200, seed: 1 })
    expect(r.cregs['c']![0]).toBe(1)
    expect(r.probs).toEqual({ '0': 1 })

    const f = new Circuit(2).x(0).creg('c', 1).measure(0, 'c', 0)
      .if('c', 1, k => k.x(1))
      .runMps({ shots: 200, seed: 1 })
    expect(f.probs).toEqual({ '11': 1 })
  })
})

describe('MPS mid-circuit measurement — canonical form', () => {
  /**
   * `MpsTrajectory.measure()` projected the local tensor and rescaled it, but
   * left the neighbouring bond lambdas holding the Schmidt spectrum of the
   * *pre-measurement* state. Both the measurement marginal and `sample()` weight
   * by those lambdas, so the first qubit measured came out right and every one
   * after it drifted: on an 8-qubit circuit the sampled distribution was off by
   * 1.5e-2 against exact, forty times sampling noise. `measure()` now
   * re-canonicalises.
   *
   * `noise: { p1: 0 }` forces the per-shot path without perturbing the physics,
   * which is what isolates measurement handling from everything else.
   */

  const build = (): Circuit => {
    let k = new Circuit(8)
    let a = 0.3
    for (let l = 0; l < 3; l++) {
      for (let q = 0; q < 8; q++) { a += 0.31; k = k.ry(a, q) }
      for (let q = l % 2; q < 7; q += 2) { a += 0.17; k = k.crx(a, q, q + 1) }
    }
    return k
  }

  it('accuracy does not degrade with the number of measured qubits', () => {
    // Every shot is a fresh simulation on this path, so shot count is the whole
    // cost. 40k is chosen to sit well inside the gap between the two regimes:
    // sampling noise here is ~2e-3, while the stale-lambda bias reached 1.5e-2.
    // k=1 and k=8 are the endpoints that matter — the old code was correct for a
    // single measurement and drifted from the second one onward.
    const exact = build().exactProbs()
    for (const k of [1, 8]) {
      let m = build().creg('o', k)
      for (let q = 0; q < k; q++) m = m.measure(q, 'o', q)
      const d = m.runMps({ shots: 40_000, seed: 5, noise: { p1: 0 } })
      for (const [bits, p] of Object.entries(exact)) {
        if (p < 0.002) continue
        expect(Math.abs((d.probs[bits] ?? 0) - p), `k=${k} outcome ${bits}`).toBeLessThan(0.006)
      }
    }
  }, 60_000)

  it('measuring the same qubit twice returns the same bit', () => {
    // Projection is idempotent — a re-canonicalisation must not disturb that.
    for (const seed of [1, 2, 3, 4]) {
      const d = new Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
        .creg('a', 1).creg('b', 1)
        .measure(0, 'a', 0).measure(0, 'b', 0)
        .runMps({ shots: 2000, seed, noise: { p1: 0 } })
      expect(d.cregs['a']![0], `seed ${seed}`).toBeCloseTo(d.cregs['b']![0]!, 10)
    }
  }, 60_000)

  it('a GHZ measured qubit-by-qubit stays perfectly correlated', () => {
    // Every shot must be all-zeros or all-ones; a broken canonical form leaks
    // weight onto mixed strings.
    let ghz = new Circuit(6).h(0)
    for (let i = 0; i < 5; i++) ghz = ghz.cnot(i, i + 1)
    let m = ghz.creg('o', 6)
    for (let q = 0; q < 6; q++) m = m.measure(q, 'o', q)
    const d = m.runMps({ shots: 4000, seed: 3, noise: { p1: 0 } })
    expect(Object.keys(d.probs).toSorted()).toEqual(['000000', '111111'])
    for (let q = 0; q < 6; q++) expect(Math.abs(d.cregs['o']![q]! - 0.5)).toBeLessThan(0.03)
  }, 60_000)

  it('norm is preserved through repeated measure-and-gate rounds', () => {
    let k = new Circuit(10)
    for (let q = 0; q < 10; q++) k = k.h(q)
    k = k.creg('s', 5)
    for (let i = 0; i < 5; i++) k = k.measure(i, 's', i).h(i).crx(0.4, i, i + 1)
    const d = k.runMps({ shots: 3000, seed: 2, noise: { p1: 0 } })
    const total = Object.values(d.probs).reduce((a, b) => a + b, 0)
    expect(Math.abs(total - 1)).toBeLessThan(1e-9)
  }, 60_000)
})
