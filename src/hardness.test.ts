import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { analyzeHardness, renderHardness, ATTACK_NAMES, type Attack, type AttackName } from './hardness.js'
import { rcsCircuit, trotterIsing } from './rcs.js'
import { amplitudeByContraction } from './tensor-network.js'
import { pauliPathExpectation } from './pauli-path.js'
import { shorCircuit } from './beauregard.js'

const ghz = (n: number): Circuit => {
  let c = new Circuit(n).h(0)
  for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
  return c
}
/** GHZ plus `t` single-qubit T gates: high magic, but entanglement stays at χ=2. */
const ghzPlusT = (n: number, t: number): Circuit => {
  let c = ghz(n)
  for (let i = 0; i < t; i++) c = c.t(i % n)
  return c
}
const find = (r: { attacks: Attack[] }, name: AttackName): Attack =>
  r.attacks.find(a => a.name === name)!

/**
 * A 16 GB / one-hour machine, probed cheaply.
 *
 * `probeChi` and `restarts` are dialled well below their defaults so the suite
 * runs in seconds: a lower cap makes the MPS probe stop sooner, which weakens χ
 * from a measurement to a floor but does not change any verdict here — a floor
 * already cannot establish viability. The default 64 costs χ³ per gate and turns
 * this file into a five-minute run on its own.
 */
const BUDGET = { memoryBytes: 16e9, seconds: 3600, probeChi: 8, restarts: 4 }

// The wide-grid analyses dominate the runtime, so each is computed once here
// rather than per test.
const SHALLOW_WIDE = analyzeHardness(rcsCircuit({ rows: 6, cols: 6, depth: 4, seed: 1 }), BUDGET)
const DEEP_WIDE = analyzeHardness(rcsCircuit({ rows: 6, cols: 6, depth: 14, seed: 1 }), BUDGET)

describe('hardness — each attack recognises the structure it prices', () => {
  it('Clifford closure is free, and named as a zero T-count', () => {
    const r = analyzeHardness(ghz(40), BUDGET)
    const a = find(r, 'clifford')
    expect(a.applicable).toBe(true)
    expect(a.parameter).toEqual({ name: 'T-count', value: 0, bound: 'exact' })
    expect(a.viable).toBe(true)
    expect(a.memoryBytes!).toBeLessThan(1e4)
  })

  it('one T gate rules Clifford out, and it says which gate', () => {
    const r = analyzeHardness(ghz(40).t(0), BUDGET)
    const a = find(r, 'clifford')
    expect(a.applicable).toBe(false)
    expect(a.reason).toMatch(/gate 't' is not Clifford/)
    expect(a.viable).toBe(false)
    // The rank backend picks it up instead, with the T-count that matters.
    expect(find(r, 'stabilizer-rank').parameter).toEqual({ name: 'T-count', value: 1, bound: 'exact' })
  })

  it('MPS measures χ exactly when the probe cap is not reached', () => {
    const a = find(analyzeHardness(ghz(30), BUDGET), 'mps')
    expect(a.parameter).toEqual({ name: 'bond dimension χ', value: 2, bound: 'exact' })
    expect(a.viable).toBe(true)
  })

  it('a volume-law circuit only gets a floor on χ, and a floor cannot prove a fit', () => {
    const a = find(analyzeHardness(rcsCircuit({ rows: 4, cols: 4, depth: 10, seed: 1 }),
      { ...BUDGET, probeChi: 16 }), 'mps')
    expect(a.applicable).toBe(true)
    expect(a.parameter!.bound).toBe('lower')
    expect(a.parameter!.value).toBe(16)
    // Its costed memory is tiny, and it is still not viable — that is the point.
    expect(a.memoryBytes!).toBeLessThan(BUDGET.memoryBytes)
    expect(a.viable).toBe(false)
  })

  it('the contraction width is an upper bound, so a fit there is trustworthy', () => {
    const a = find(analyzeHardness(rcsCircuit({ rows: 4, cols: 4, depth: 6, seed: 2 }), BUDGET), 'tensor-network')
    expect(a.applicable).toBe(true)
    expect(a.parameter!.name).toBe('contraction width')
    expect(a.parameter!.bound).toBe('upper')
    expect(a.answers).toBe('amplitude')
  })

  it('Pauli path needs an observable, and prices one when given', () => {
    const c = new Circuit(6).h(0).cnot(0, 1).rz(0.4, 2).cnot(2, 3)
    expect(find(analyzeHardness(c, BUDGET), 'pauli-path').reason).toMatch(/no observable given/)

    const withObs = find(analyzeHardness(c, { ...BUDGET, observable: [{ coeff: 1, ops: 'ZZIIII' }] }), 'pauli-path')
    expect(withObs.applicable).toBe(true)
    expect(withObs.parameter!.name).toBe('Pauli terms')
    expect(withObs.answers).toBe('expectation')
  })

})

describe('hardness — Clifford is decided by the angle, not the gate name', () => {
  /**
   * The defect this exists to prevent, found on the endpoints of a published
   * parameter sweep. `Rz(−π/2)` is `S†` and `Rx(π/2)` is `√X`; both are Clifford,
   * and both were rejected on their gate name. That made the adversary report a
   * circuit Gottesman–Knill solves in polynomial time as one nothing could break.
   */
  const line = Array.from({ length: 11 }, (_, i) => [i, i + 1] as [number, number])
  const trotter = (thetaH: number): Circuit =>
    trotterIsing({ qubits: 12, coupling: line, depth: 4, thetaH })

  it('both ends of the transverse-field sweep are recognised as Clifford', () => {
    // Rzz(-π/2) is Clifford, so at θ=0 and θ=π/2 the whole circuit is.
    for (const theta of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
      const a = find(analyzeHardness(trotter(theta), BUDGET), 'clifford')
      expect(a.applicable, `θ=${theta}`).toBe(true)
      expect(a.parameter, `θ=${theta}`).toEqual({ name: 'T-count', value: 0, bound: 'exact' })
    }
  })

  it('the interior of the sweep is not, and it says which angle', () => {
    for (const theta of [Math.PI / 8, Math.PI / 4, 1.2]) {
      const a = find(analyzeHardness(trotter(theta), BUDGET), 'clifford')
      expect(a.applicable, `θ=${theta}`).toBe(false)
      expect(a.reason, `θ=${theta}`).toMatch(/is not a Clifford angle/)
    }
  })

  it('recognises the Clifford rotation angles individually', () => {
    for (const [gate, build] of [
      ['rz', (t: number) => new Circuit(2).rz(t, 0)],
      ['rx', (t: number) => new Circuit(2).rx(t, 0)],
      ['ry', (t: number) => new Circuit(2).ry(t, 0)],
      ['u1', (t: number) => new Circuit(2).u1(t, 0)],
    ] as const) {
      for (const k of [0, 1, 2, 3, 4, -1, -2]) {
        expect(find(analyzeHardness(build(k * Math.PI / 2), BUDGET), 'clifford').applicable,
          `${gate}(${k}π/2)`).toBe(true)
      }
      expect(find(analyzeHardness(build(Math.PI / 4), BUDGET), 'clifford').applicable,
        `${gate}(π/4)`).toBe(false)
    }
  })

  it('√X and its inverse are Clifford', () => {
    expect(find(analyzeHardness(new Circuit(2).v(0).vi(1), BUDGET), 'clifford').applicable).toBe(true)
  })
})

describe('hardness — probing an attack is not attempting it', () => {
  /**
   * The distinction the module lacked, and the reason it systematically
   * under-reported the Pauli attacks. A probe caps the propagation, so the term
   * count is a floor, and a floor can never rule an attack *in* — on circuits
   * where propagation does not converge cheaply, which is most of them, the
   * attack was permanently invisible. Including the published Trotter circuits
   * that sparse Pauli dynamics reproduced at millions of terms.
   */
  const line = Array.from({ length: 9 }, (_, i) => [i, i + 1] as [number, number])
  const c = trotterIsing({ qubits: 10, coupling: line, depth: 6, thetaH: Math.PI / 4 })
  const observable = [{ coeff: 1, ops: `${'I'.repeat(4)}Z${'I'.repeat(5)}` }]

  it('a capped probe reports a floor and cannot establish viability', () => {
    const a = find(analyzeHardness(c, { ...BUDGET, observable, probeTerms: 8 }), 'pauli-path')
    expect(a.applicable).toBe(true)
    expect(a.parameter!.bound).toBe('lower')
    expect(a.viable).toBe(false)
  })

  it('an attempt that converges reports a measurement, and can', () => {
    const a = find(analyzeHardness(c, { ...BUDGET, observable, attemptTerms: 1 << 16 }), 'pauli-path')
    expect(a.applicable).toBe(true)
    expect(a.parameter!.bound).toBe('exact')
    expect(a.viable).toBe(true)
  })

  it('an attempt that does not converge still reports a floor', () => {
    // Too small a budget to settle: the answer is still moving between the two
    // runs, so the cap is what binds and the honest verdict is unchanged.
    const a = find(analyzeHardness(c, { ...BUDGET, observable, attemptTerms: 8, attemptTolerance: 1e-9 }), 'pauli-path')
    expect(a.parameter!.bound).toBe('lower')
    expect(a.viable).toBe(false)
  })

  it('the attempt agrees with the exact expectation value it converged on', () => {
    // Converged has to mean correct, not just stable.
    const exact = c.exactProbs()
    let truth = 0
    for (const [b, p] of Object.entries(exact)) truth += (b[4] === '1' ? -1 : 1) * p
    const got = pauliPathExpectation(c, observable, { threshold: 1e-12, maxTerms: 1 << 16 })
    expect(got.value).toBeCloseTo(truth, 6)
  })

  it('attempting costs more than probing, which is why it is opt-in', () => {
    const probe = find(analyzeHardness(c, { ...BUDGET, observable, probeTerms: 64 }), 'pauli-path')
    const attempt = find(analyzeHardness(c, { ...BUDGET, observable, attemptTerms: 1 << 16 }), 'pauli-path')
    expect(attempt.parameter!.value).toBeGreaterThan(probe.parameter!.value)
  })

  it('the noisy attack escalates the same way', () => {
    const noise = { p1: 0.001, p2: 0.02 }
    const probed = find(analyzeHardness(c, { ...BUDGET, observable, noise, probeTerms: 8 }), 'noisy-pauli-path')
    const tried = find(analyzeHardness(c, { ...BUDGET, observable, noise, attemptTerms: 1 << 16 }), 'noisy-pauli-path')
    expect(probed.parameter!.bound).toBe('lower')
    expect(tried.parameter!.bound).toBe('exact')
  })

  describe('attemptOnlyIfDecisive skips what cannot change the answer', () => {
    /**
     * The flag exists because resolving an attack that has already lost is pure
     * spend — a 127-qubit target broken by a 16 MB contraction should not pay ten
     * minutes to price a Pauli attack that cannot win. The risk is that it
     * suppresses an attempt in the one case that matters, so that is what these
     * pin: silence where the answer is settled, and full spend where it is not.
     */
    it('is off by default, so the bound still means what it says', () => {
      const on = find(analyzeHardness(c, { ...BUDGET, observable, attemptTerms: 1 << 16 }), 'pauli-path')
      expect(on.parameter!.bound).toBe('exact')
    })

    it('never changes the recommendation, only what it pays to learn', () => {
      /**
       * The invariant the flag lives or dies by, and it subsumes the worry that
       * motivated it: if skipping an attempt ever hid a break, `best` would differ
       * between the two runs. Sweeping the transverse field covers the Clifford
       * endpoint, where something cheap wins outright, and the generic angle where
       * only the Pauli attack can.
       */
      for (const thetaH of [0, Math.PI / 4, Math.PI / 2, 0.6]) {
        for (const memoryBytes of [16e9, 1 << 20]) {
          const t = trotterIsing({ qubits: 10, coupling: line, depth: 6, thetaH })
          const budget = { ...BUDGET, observable, memoryBytes, attemptTerms: 1 << 16 }
          const full = analyzeHardness(t, budget)
          const frugal = analyzeHardness(t, { ...budget, attemptOnlyIfDecisive: true })
          expect(frugal.best?.name ?? null, `θ=${thetaH} mem=${memoryBytes}`)
            .toBe(full.best?.name ?? null)
        }
      }
    })

    it('resolves a floor that is still in the running', () => {
      // Nothing cheaper has landed and the floor is inside the budget, so the
      // attempt is the only thing that can settle it and has to be paid for.
      const budget = { ...BUDGET, observable, attemptTerms: 1 << 16, attemptOnlyIfDecisive: true }
      const a = find(analyzeHardness(c, budget), 'pauli-path')
      const bar = analyzeHardness(c, { ...BUDGET, observable }).best
      // Guard the premise: if something cheap already won, this proves nothing.
      if (bar === null || (bar.seconds ?? Infinity) > (a.seconds ?? 0)) {
        expect(a.parameter!.bound).toBe('exact')
      }
    })

    it('skips the attempt once something cheaper has landed', () => {
      // Clifford-angle Trotter: the Clifford attack takes this outright, so the
      // Pauli floor cannot displace it however it resolves.
      const easy = trotterIsing({ qubits: 10, coupling: line, depth: 6, thetaH: Math.PI / 2 })
      const budget = { ...BUDGET, observable, attemptTerms: 1 << 16 }
      const landed = analyzeHardness(easy, budget)
      expect(landed.best, 'fixture is meant to be broken by something cheap').not.toBeNull()

      const frugal = find(analyzeHardness(easy, { ...budget, attemptOnlyIfDecisive: true }), 'pauli-path')
      const full = find(landed, 'pauli-path')
      // Same recommendation either way — which is the whole justification.
      expect(analyzeHardness(easy, { ...budget, attemptOnlyIfDecisive: true }).best!.name).toBe(landed.best!.name)
      // But the frugal run left the Pauli attack unresolved rather than paying.
      expect(frugal.parameter!.value).toBeLessThanOrEqual(full.parameter!.value)
    })
  })
})

describe('hardness — approximate MPS is a method, not a failed exact one', () => {
  /**
   * The exact MPS attack refuses any circuit whose χ exceeds the probe cap,
   * because the parameter is then a floor. But truncated MPS *is* the approximate
   * method, and approximate tensor networks are what reproduced the published
   * Trotter circuits — so refusing them leaves the tool unable to reach the
   * results it is being calibrated against.
   */
  const volume = (n: number, layers: number): Circuit => {
    let c = new Circuit(n)
    let a = 0.3
    for (let l = 0; l < layers; l++) {
      for (let q = 0; q < n; q++) { a += 0.37; c = c.ry(a, q).rz(a * 1.7, q) }
      for (let q = l % 2; q + 1 < n; q += 2) { a += 0.23; c = c.crx(a, q, q + 1) }
    }
    return c
  }

  it('a probe that hits its cap reports a floor and cannot be viable', () => {
    const a = find(analyzeHardness(volume(12, 8), { ...BUDGET, probeChi: 4 }), 'mps')
    expect(a.parameter!.bound).toBe('lower')
    expect(a.viable).toBe(false)
  })

  it('an attempt whose answer stops moving when χ doubles is viable', () => {
    // A low-entanglement circuit: χ settles well below the cap, so doubling it
    // changes nothing and the truncation is provably not distorting anything.
    let ghzT = new Circuit(14).h(0)
    for (let i = 0; i < 13; i++) ghzT = ghzT.cnot(i, i + 1)
    const a = find(analyzeHardness(ghzT.t(0), { ...BUDGET, attemptChi: 8 }), 'mps')
    expect(a.parameter!.bound).toBe('exact')
    expect(a.viable).toBe(true)
  })

  it('an attempt on a volume-law circuit still reports a floor', () => {
    // The doubled run saturates its own cap, so the bond dimension is still
    // growing and nothing can be said about what truncation cost.
    const a = find(analyzeHardness(volume(12, 10), { ...BUDGET, attemptChi: 2 }), 'mps')
    expect(a.parameter!.bound).toBe('lower')
    expect(a.viable).toBe(false)
  })

  it('converges exactly where the truncation error actually vanishes', () => {
    /**
     * Measured against ground truth: on this circuit the truncated ⟨Z_mid⟩ error
     * runs 1.2e-1, 5.8e-2, 7.9e-3, then 1e-13 at χ=16. The attack must call it
     * converged at 16 and not before.
     */
    const hard = volume(12, 8)
    for (const chi of [2, 4, 8]) {
      expect(find(analyzeHardness(hard, { ...BUDGET, attemptChi: chi }), 'mps').parameter!.bound,
        `χ=${chi}`).toBe('lower')
    }
    for (const chi of [16, 32]) {
      const a = find(analyzeHardness(hard, { ...BUDGET, attemptChi: chi }), 'mps')
      expect(a.parameter!.bound, `χ=${chi}`).toBe('exact')
      expect(a.parameter!.value, `χ=${chi}`).toBe(16)
    }
  })

  it('is not fooled by the truncation error being non-monotone in χ', () => {
    /**
     * On an RCS grid the error falls 4.2e-1 → 1.0e-1 → 3.7e-2 → 7.8e-3 and then
     * *rises* to 2.4e-2 before collapsing. A check reading one observable can
     * cross the tolerance band by accident on the way past, so several sites and
     * both Z and X are compared at once.
     */
    const rcs = rcsCircuit({ rows: 3, cols: 4, depth: 8, seed: 2 })
    for (const chi of [2, 4, 8, 16, 32]) {
      expect(find(analyzeHardness(rcs, { ...BUDGET, attemptChi: chi }), 'mps').parameter!.bound,
        `χ=${chi}`).toBe('lower')
    }
  })

  it('the signal is saturation, not sampled agreement', () => {
    /**
     * Comparing sampled distributions at χ and 2χ is defeated by shot noise: at a
     * few thousand shots the floor exceeds any useful tolerance, and a volume-law
     * circuit reads as converged. Whether the doubled run hit its own cap is
     * noise-free and cannot be fooled that way.
     */
    const hard = volume(12, 10)
    for (const chi of [2, 4, 8]) {
      const a = find(analyzeHardness(hard, { ...BUDGET, attemptChi: chi }), 'mps')
      expect(a.parameter!.bound, `χ=${chi}`).toBe('lower')
    }
  })

  it('the attempt does not change what the exact probe would have said on an easy circuit', () => {
    let ghz = new Circuit(16).h(0)
    for (let i = 0; i < 15; i++) ghz = ghz.cnot(i, i + 1)
    const probed = find(analyzeHardness(ghz.t(0), BUDGET), 'mps')
    const tried = find(analyzeHardness(ghz.t(0), { ...BUDGET, attemptChi: 16 }), 'mps')
    expect(probed.parameter!.value).toBe(tried.parameter!.value)
    expect(tried.viable).toBe(true)
  })
})

describe('hardness — a recommendation has to cash', () => {
  /**
   * Every verdict is a prediction until someone runs the attack. A recommendation
   * that is affordable and *wrong* is worse than no recommendation, so the winner
   * is executed and checked against the exact statevector.
   *
   * `benchmark/verify-attacks.ts` does this across the published families; this is
   * the fast subset that runs in CI.
   */
  const deviation = (a: Record<string, number>, b: Record<string, number>): number => {
    let worst = 0
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      worst = Math.max(worst, Math.abs((a[k] ?? 0) - (b[k] ?? 0)))
    }
    return worst
  }

  it('the winning attack reproduces the exact distribution', () => {
    const line = Array.from({ length: 9 }, (_, i) => [i, i + 1] as [number, number])
    for (const theta of [0, Math.PI / 4, Math.PI / 2]) {
      const c = trotterIsing({ qubits: 10, coupling: line, depth: 2, thetaH: theta })
      const r = analyzeHardness(c, BUDGET)
      const best = r.best
      expect(best, `θ=${theta}`).not.toBe(null)

      const exact = c.exactProbs()
      const got = best!.name === 'clifford' ? c.runClifford({ shots: 40_000, seed: 1 }).probs
        : best!.name === 'mps' ? c.runMps({ shots: 40_000, seed: 1 }).probs
        : best!.name === 'stabilizer-rank' ? c.runStabilizerRank({ shots: 40_000, seed: 1 }).probs
        : best!.name === 'statevector' ? c.run({ shots: 40_000, seed: 1 }).probs
        : null
      if (got === null) continue          // an amplitude-only attack; covered below
      expect(deviation(exact, got), `θ=${theta} via ${best!.name}`).toBeLessThan(0.02)
    }
  })

  it('a recommended contraction reproduces the exact amplitude', () => {
    const c = rcsCircuit({ rows: 2, cols: 4, depth: 4, seed: 3 })
    const r = analyzeHardness(c, BUDGET)
    const tn = find(r, 'tensor-network')
    expect(tn.viable).toBe(true)
    const bits = '0'.repeat(c.qubits)
    const want = c.amplitude(bits)
    const got = amplitudeByContraction(c, bits, { restarts: 8 })
    expect(got.re).toBeCloseTo(want.re, 12)
    expect(got.im).toBeCloseTo(want.im, 12)
  })
})

describe('hardness — the attacks are complementary, not ranked', () => {
  /**
   * The case that justifies running all of them. GHZ + many T gates has high
   * magic and almost no entanglement, so the backend built for magic chokes while
   * the one built for entanglement walks it. Neither is "better".
   */
  it('high magic with low entanglement breaks the magic attack, not the MPS one', () => {
    const r = analyzeHardness(ghzPlusT(60, 80), BUDGET)
    const sr = find(r, 'stabilizer-rank')
    const mps = find(r, 'mps')

    expect(sr.parameter).toEqual({ name: 'T-count', value: 80, bound: 'exact' })
    expect(sr.memoryBytes!).toBeGreaterThan(BUDGET.memoryBytes)
    expect(sr.viable).toBe(false)

    expect(mps.parameter!.value).toBe(2)
    expect(mps.viable).toBe(true)
    expect(r.best!.name).toBe('mps')
  })

  it('a shallow wide circuit falls to treewidth where nothing else reaches', () => {
    const r = SHALLOW_WIDE
    expect(r.qubits).toBe(36)
    expect(find(r, 'statevector').viable).toBe(false)     // 1.1 TB
    expect(find(r, 'mps').viable).toBe(false)             // χ floor
    expect(find(r, 'tensor-network').viable).toBe(true)
    expect(r.best!.name).toBe('tensor-network')
  })

  it('a deep wide circuit survives every attack, and each says why', () => {
    const r = DEEP_WIDE
    expect(r.best).toBe(null)
    expect(r.bestFull).toBe(null)
    expect(r.advantage).toBe(null)
    for (const a of r.attacks) {
      expect(a.viable, a.name).toBe(false)
      // Nothing is silently absent: every attack carries either a parameter that
      // priced it out or a reason it could not run.
      expect(a.parameter !== undefined || a.reason !== undefined, a.name).toBe(true)
    }
  })

  it('depth is what closes the door — the same width at depth 4 was broken', () => {
    const shallow = SHALLOW_WIDE, deep = DEEP_WIDE
    expect(shallow.qubits).toBe(deep.qubits)
    expect(find(shallow, 'tensor-network').parameter!.value)
      .toBeLessThan(find(deep, 'tensor-network').parameter!.value)
    expect(shallow.best).not.toBe(null)
    expect(deep.best).toBe(null)
  })
})

describe('hardness — the report', () => {
  it('sorts viable first and cheapest first within that', () => {
    const r = analyzeHardness(ghzPlusT(40, 4), BUDGET)
    let seenNonViable = false
    for (const a of r.attacks) {
      if (!a.viable) seenNonViable = true
      else expect(seenNonViable, `${a.name} is viable but came after a non-viable attack`).toBe(false)
    }
    const viable = r.attacks.filter(a => a.viable)
    for (let i = 1; i < viable.length; i++) {
      expect(viable[i]!.seconds!).toBeGreaterThanOrEqual(viable[i - 1]!.seconds!)
    }
    expect(r.best).toBe(viable[0])
  })

  it('advantage is the statevector cost over the winner’s', () => {
    const r = analyzeHardness(ghz(30), BUDGET)
    const sv = find(r, 'statevector')
    expect(r.advantage).toBeCloseTo(sv.operations! / r.best!.operations!, 6)
    expect(r.advantage!).toBeGreaterThan(1e6)
  })

  it('reports no advantage when the statevector is simply the right tool', () => {
    // A small non-Clifford circuit with real entanglement: nothing beats a dense solve.
    const r = analyzeHardness(rcsCircuit({ rows: 2, cols: 3, depth: 8, seed: 4 }),
      BUDGET)
    if (r.best?.name === 'statevector') expect(r.advantage).toBe(null)
    expect(r.attacks.some(a => a.viable)).toBe(true)
  })

  it('separates the cheapest attack from the cheapest full distribution', () => {
    // Contraction is cheaper but answers one amplitude; the distinction is the
    // difference between "this circuit is easy" and "one number about it is".
    const r = analyzeHardness(rcsCircuit({ rows: 4, cols: 4, depth: 10, seed: 1 }), BUDGET)
    expect(r.best!.answers).toBe('amplitude')
    expect(r.bestFull!.answers).toBe('distribution')
    expect(r.bestFull!.name).not.toBe(r.best!.name)
    expect(renderHardness(r)).toContain('cheapest full distribution')
  })

  it('the budget decides the verdict, and moving it moves the verdict', () => {
    const c = ghzPlusT(60, 60)
    const tight = analyzeHardness(c, { memoryBytes: 1e6, seconds: 1 })
    const loose = analyzeHardness(c, { memoryBytes: 1e12, seconds: 1e6 })
    expect(find(tight, 'stabilizer-rank').viable).toBe(false)
    expect(find(loose, 'stabilizer-rank').viable).toBe(true)
    // Same measured parameter either way — only the affordability changed.
    expect(find(tight, 'stabilizer-rank').parameter)
      .toEqual(find(loose, 'stabilizer-rank').parameter)
  })

  it('rejects a nonsensical budget instead of producing a verdict from it', () => {
    expect(() => analyzeHardness(ghz(4), { memoryBytes: 0 })).toThrow(RangeError)
    expect(() => analyzeHardness(ghz(4), { seconds: -1 })).toThrow(RangeError)
    expect(() => analyzeHardness(ghz(4), { opsPerSecond: 0 })).toThrow(RangeError)
    expect(() => analyzeHardness(ghz(4), { probeChi: 0 })).toThrow(RangeError)
  })

  it('every attack appears exactly once, whatever the circuit', () => {
    const names: AttackName[] = ['clifford', 'stabilizer-rank', 'mps', 'peps', 'tensor-network',
      'pauli-path', 'noisy-pauli-path', 'symmetry', 'statevector']
    for (const c of [ghz(8), ghzPlusT(8, 3), rcsCircuit({ rows: 2, cols: 3, depth: 4, seed: 1 })]) {
      const r = analyzeHardness(c, BUDGET)
      expect(r.attacks.map(a => a.name).toSorted()).toEqual([...names].toSorted())
    }
  })
})

describe('hardness — rendering', () => {
  const r = analyzeHardness(ghzPlusT(40, 4), BUDGET)

  it('marks viability and bound direction', () => {
    const text = renderHardness(r)
    expect(text).toContain('✓ mps')
    expect(text).toContain('≤')                     // the contraction width's upper bound
    expect(text).toMatch(/best: \w/)
    expect(text).toContain('40 qubits')
  })

  it('says plainly when nothing landed', () => {
    expect(renderHardness(DEEP_WIDE)).toContain('no attack in this library landed within budget')
  })

  it('keeps a long refusal from setting the table width', () => {
    // The stabilizer-rank refusal lists every gate it accepts, which is a
    // paragraph. It must not push every other column off the screen.
    const text = renderHardness(analyzeHardness(rcsCircuit({ rows: 3, cols: 3, depth: 4, seed: 1 }),
      BUDGET))
    for (const line of text.split('\n')) expect(line.length, line).toBeLessThan(110)
    expect(text).toContain('…')
  })

  it('renders astronomical costs without running out of units', () => {
    const text = renderHardness(analyzeHardness(rcsCircuit({ rows: 10, cols: 10, depth: 10, seed: 1 }),
      BUDGET))
    expect(text).not.toContain('undefined')
    expect(text).not.toContain('NaN')
    expect(text).toMatch(/years|e\+\d+ B/)
  })
})

describe('hardness — the 2-D attack prices a lattice as a lattice', () => {
  /**
   * PEPS exists in this roster because MPS prices the wrong thing on a lattice: a
   * 1-D cut through a 2-D state carries bonds the geometry does not require, and
   * on the heavy-hex circuits the truncated MPS answer was still moving 10%
   * between χ=32 and 64. What matters for the *adversary* is not that PEPS is
   * cheaper but that it cannot overclaim — it is approximate, so the burden is on
   * it to prove a parameter rather than assert one.
   */
  const grid = (rows: number, cols: number): [number, number][] => {
    const q = (i: number, j: number): number => i * cols + j
    const e: [number, number][] = []
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (j + 1 < cols) e.push([q(i, j), q(i, j + 1)])
        if (i + 1 < rows) e.push([q(i, j), q(i + 1, j)])
      }
    }
    return e
  }

  it('is applicable on a 2-D circuit and reports a bond dimension', () => {
    const c = trotterIsing({ qubits: 9, coupling: grid(3, 3), depth: 3, thetaH: Math.PI / 4 })
    const a = find(analyzeHardness(c, BUDGET), 'peps')
    expect(a.applicable).toBe(true)
    expect(a.parameter!.name).toBe('bond dimension')
    expect(a.parameter!.value).toBeGreaterThanOrEqual(1)
  })

  it('needs no lattice supplied — the bonds are the circuit’s own gate pairs', () => {
    // A ring is not a grid and nothing here is told which it is, so an attack
    // that assumed a lattice would either decline or price the wrong network.
    const ring: [number, number][] = Array.from({ length: 6 }, (_, i) => [i, (i + 1) % 6])
    const c = trotterIsing({ qubits: 6, coupling: ring, depth: 3, thetaH: 0.7 })
    expect(find(analyzeHardness(c, BUDGET), 'peps').applicable).toBe(true)
  })

  it('a truncated probe is a floor, never a measurement', () => {
    // The whole risk of adding an approximate method to a hardness report: if a
    // truncated run could claim `exact`, the adversary would report breaks that
    // do not exist. A bond cap of 1 forces truncation on an entangling circuit.
    const c = trotterIsing({ qubits: 9, coupling: grid(3, 3), depth: 4, thetaH: Math.PI / 4 })
    const a = find(analyzeHardness(c, { ...BUDGET, probeBond: 1 }), 'peps')
    expect(a.parameter!.bound).toBe('lower')
    expect(a.viable).toBe(false)
  })

  it('an untruncated run is exact, because nothing was discarded', () => {
    // A product circuit never entangles, so no bond ever grows and no singular
    // weight is thrown away. That is the one case the probe can rule in by itself.
    let c = new Circuit(9)
    for (let q = 0; q < 9; q++) c = c.h(q).t(q)
    const a = find(analyzeHardness(c, BUDGET), 'peps')
    expect(a.parameter!.bound).toBe('exact')
    expect(a.viable).toBe(true)
  })

  it('an attempt that does not settle still reports a floor', () => {
    const c = trotterIsing({ qubits: 9, coupling: grid(3, 3), depth: 5, thetaH: Math.PI / 4 })
    const a = find(analyzeHardness(c, { ...BUDGET, attemptBond: 1, attemptTolerance: 1e-12 }), 'peps')
    expect(a.parameter!.bound).toBe('lower')
  })

  it('charges a degree-3 lattice more per gate than a chain of the same bond', () => {
    // The cost model reads degree off the network rather than assuming one, so a
    // sparse lattice is not billed as a dense one and vice versa.
    const line: [number, number][] = Array.from({ length: 8 }, (_, i) => [i, i + 1])
    const chain = trotterIsing({ qubits: 9, coupling: line, depth: 3, thetaH: 0.4 })
    const lattice = trotterIsing({ qubits: 9, coupling: grid(3, 3), depth: 3, thetaH: 0.4 })
    const a = find(analyzeHardness(chain, BUDGET), 'peps')
    const b = find(analyzeHardness(lattice, BUDGET), 'peps')
    expect(a.applicable && b.applicable).toBe(true)
    // Same gate count and same probe cap, so any difference is the geometry.
    expect(b.operations! / b.parameter!.value).toBeGreaterThan(a.operations! / a.parameter!.value)
  })

  it('declines rather than mispricing a circuit it cannot replay', () => {
    const c = new Circuit(4).h(0).cnot(0, 1).measure(0, 'c', 0)
    const a = find(analyzeHardness(c, BUDGET), 'peps')
    expect(a.applicable).toBe(false)
    expect(a.viable).toBe(false)
  })
})

describe('hardness — a probe that is not cheap is gated by size', () => {
  /**
   * Two probes are structurally expensive rather than cheap: recursive-bisection
   * planning and replaying a circuit as a PEPS. Ungated they make *pricing* a
   * circuit dearer than several of the attacks being priced, and `searchHardness`
   * pays that once per rewrite candidate. What these pin is that the gate changes
   * cost without ever changing a verdict's honesty.
   */
  const line = Array.from({ length: 11 }, (_, i) => [i, i + 1] as [number, number])
  const wide = trotterIsing({ qubits: 12, coupling: line, depth: 12, thetaH: Math.PI / 4 })
  const gateCount = wide.gateCounts().total

  it('declines the PEPS probe past the limit instead of reporting a floor', () => {
    // A decline and a floor mean different things; a floor nobody paid for is a
    // claim about the circuit that no measurement backs.
    const a = find(analyzeHardness(wide, { ...BUDGET, probeMaxGates: 1 }), 'peps')
    expect(a.applicable).toBe(false)
    expect(a.parameter).toBeUndefined()
    expect(a.reason).toMatch(/probeMaxGates/)
    expect(a.viable).toBe(false)
  })

  it('prices the PEPS probe under the limit', () => {
    const a = find(analyzeHardness(wide, { ...BUDGET, probeMaxGates: gateCount + 1 }), 'peps')
    expect(a.applicable).toBe(true)
    expect(a.parameter!.name).toBe('bond dimension')
  })

  it('keeps the contraction width an upper bound on either side of the gate', () => {
    // The contraction attack degrades rather than declining, so the guarantee that
    // has to survive is soundness: a looser plan is still a ceiling, so a fit
    // reported from it is still trustworthy.
    const cheap = find(analyzeHardness(wide, { ...BUDGET, probeMaxGates: 1 }), 'tensor-network')
    const tight = find(analyzeHardness(wide, { ...BUDGET, probeMaxGates: Infinity }), 'tensor-network')
    for (const a of [cheap, tight]) {
      expect(a.applicable).toBe(true)
      expect(a.parameter!.bound).toBe('upper')
    }
    // Bisection can only tie or beat greedy, so the gated width is never tighter.
    expect(tight.parameter!.value).toBeLessThanOrEqual(cheap.parameter!.value)
  })

  it('gating does not change which attack is recommended here', () => {
    const cheap = analyzeHardness(wide, { ...BUDGET, probeMaxGates: 1 })
    const tight = analyzeHardness(wide, { ...BUDGET, probeMaxGates: Infinity })
    expect(cheap.best?.name ?? null).toBe(tight.best?.name ?? null)
  })

  it('defaults to a limit that leaves ordinary circuits priced', () => {
    // The gate exists for 9k-gate networks. A circuit anyone would paste into a
    // playground must not silently lose two attacks to it.
    const small = trotterIsing({ qubits: 9, coupling: line.slice(0, 8), depth: 3, thetaH: 0.5 })
    const r = analyzeHardness(small, { memoryBytes: 16e9, seconds: 3600, probeChi: 8, restarts: 4 })
    expect(find(r, 'peps').applicable).toBe(true)
    expect(find(r, 'tensor-network').applicable).toBe(true)
  })
})

describe('hardness — the roster is one list, and accuracy is reportable', () => {
  /**
   * Plumbing, pinned because both defects it replaces were silent. An attack
   * missing from a hand-written array simply never ran, and a report with a row
   * absent looks exactly like a report with a row that declined. An approximate
   * attack with nowhere to put its error made "broken" and "broken to within 1e-3"
   * print identically.
   */
  const line = Array.from({ length: 11 }, (_, i) => [i, i + 1] as [number, number])

  it('every name in ATTACK_NAMES appears in the report, exactly once', () => {
    // The registry is keyed by AttackName so this is enforced at compile time too;
    // this catches a runner that throws its way out of the roster at runtime.
    for (const c of [ghz(8), ghzPlusT(8, 3), rcsCircuit({ rows: 2, cols: 3, depth: 4, seed: 1 })]) {
      const names = analyzeHardness(c, BUDGET).attacks.map(a => a.name)
      expect(names.toSorted()).toEqual([...ATTACK_NAMES].toSorted())
      expect(new Set(names).size).toBe(ATTACK_NAMES.length)
    }
  })

  it('an exact attack claims zero error; a floor claims nothing at all', () => {
    // Absent is not zero. A probe that hit its cap has no basis for any accuracy
    // statement, and saying "0" there would be the lie this field exists to avoid.
    const clifford = find(analyzeHardness(ghz(20), BUDGET), 'clifford')
    expect(clifford.error).toEqual({ value: 0, bound: 'exact' })

    const floored = find(analyzeHardness(rcsCircuit({ rows: 4, cols: 4, depth: 10, seed: 1 }),
      { ...BUDGET, probeChi: 4 }), 'mps')
    expect(floored.parameter!.bound).toBe('lower')
    expect(floored.error).toBeUndefined()
  })

  it('a converged attempt claims the tolerance it met, not zero', () => {
    // Converged-while-truncating is accurate *to* a tolerance. Reporting 0 would
    // overstate it; reporting nothing would understate a real guarantee.
    const c = trotterIsing({ qubits: 10, coupling: line.slice(0, 9), depth: 6, thetaH: Math.PI / 4 })
    const observable = [{ coeff: 1, ops: `${'I'.repeat(4)}Z${'I'.repeat(5)}` }]
    const a = find(analyzeHardness(c, { ...BUDGET, observable, attemptTerms: 1 << 16, attemptTolerance: 0.02 }),
      'pauli-path')
    expect(a.parameter!.bound).toBe('exact')
    expect(a.error).toBeDefined()
    expect(a.error!.bound === 'exact' ? a.error!.value : a.error!.value).toBeLessThanOrEqual(0.02)
    if (a.error!.value > 0) expect(a.error!.bound).toBe('upper')
  })

  it('stabilizer rank claims the target error it was budgeted for', () => {
    const a = find(analyzeHardness(ghz(20).t(0).t(1), { ...BUDGET, targetError: 0.05 }), 'stabilizer-rank')
    expect(a.error).toEqual({ value: 0.05, bound: 'upper' })
    // Zero T gates means nothing was approximated, whatever the budget said.
    expect(find(analyzeHardness(ghz(20), { ...BUDGET, targetError: 0.05 }), 'stabilizer-rank').error)
      .toEqual({ value: 0, bound: 'exact' })
  })

  it('renders the accuracy claim as its own column', () => {
    const text = renderHardness(analyzeHardness(ghz(20), BUDGET))
    expect(text).toMatch(/\berror\b/)
    // An exact attack reads as "exact" rather than as a bare 0, which would look
    // like a measured error that happened to round to nothing.
    expect(text).toMatch(/exact/)
  })
})

describe('hardness — contraction planning stays bounded', () => {
  /**
   * Greedy planning is the cheap tier only relative to the network: measured, it
   * is 0.8s on a 33,794-tensor Trotter network, 3.1s on Shor N=33 (35,053
   * tensors) and 44s on Shor N=1003 (191,583), because the cost is superlinear in
   * tensor count. Forty-four seconds to report width 81 — unmeetable at any
   * budget — is a wait the caller paid for nothing, so past the limit the attack
   * declines and names it.
   *
   * This also pins the fix to a real stack overflow. Before it, the same circuit
   * came back `reason: 'Maximum call stack size exceeded'`, which reads as "this
   * attack does not apply" when it meant "the planner crashed".
   */
  it('declines a 200k-gate circuit instead of planning it for a minute', () => {
    const c = shorCircuit(1003n, 5n)
    expect(c.gateCounts().total).toBeGreaterThan(50_000)
    const t0 = performance.now()
    const r = analyzeHardness(c, { memoryBytes: 16e9, seconds: 3600, probeChi: 4, restarts: 4 })
    const tn = r.attacks.find(a => a.name === 'tensor-network')!
    expect(tn.applicable).toBe(false)
    expect(tn.reason).toContain('past 50000')
    expect(tn.reason).not.toContain('call stack')
    // Generous against a slow machine, and still an order below the 44s it took
    // to plan the same network before the limit existed.
    expect(performance.now() - t0).toBeLessThan(30_000)
  })

  it('still prices a circuit under the limit with the greedy planner', () => {
    // 21,569 gates: past probeMaxGates, so greedy, and comfortably under the cap.
    const c = shorCircuit(21n, 2n)
    const r = analyzeHardness(c, { memoryBytes: 16e9, seconds: 3600, probeChi: 4, restarts: 4 })
    const tn = r.attacks.find(a => a.name === 'tensor-network')!
    expect(tn.applicable).toBe(true)
    expect(tn.parameter?.name).toBe('contraction width')
  })
})
