import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { lightCone, orderingCost, peephole, qubitsOf, searchQubitOrder, spectralOrder, toZBasis } from './rewrite.js'
import { searchHardness, renderSearch, analyzeHardness } from './hardness.js'
import { rcsCircuit, trotterIsing } from './rcs.js'
import { pauliPathExpectation } from './pauli-path.js'

/**
 * Soundness is the whole contract.
 *
 * A rewrite that quietly changed the answer turns the adversary from an attack
 * into a wrong result delivered confidently. Every rewrite here is checked
 * against `exactProbs()` rather than argued for — the claim on `preserves` is
 * what is under test, not the reasoning behind it.
 */

/** Sum the distribution down onto a subset of qubits. Keys are q0-leftmost. */
const marginal = (probs: Record<string, number>, qubits: readonly number[]): Record<string, number> => {
  const out: Record<string, number> = {}
  for (const [bits, p] of Object.entries(probs)) {
    const key = qubits.map(q => bits[q]!).join('')
    out[key] = (out[key] ?? 0) + p
  }
  return out
}

const agree = (a: Record<string, number>, b: Record<string, number>, label: string, tol = 1e-9): void => {
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    expect(Math.abs((a[k] ?? 0) - (b[k] ?? 0)), `${label} outcome ${k}`).toBeLessThan(tol)
  }
}

const ghz = (n: number): Circuit => {
  let c = new Circuit(n).h(0)
  for (let i = 0; i < n - 1; i++) c = c.cnot(i, i + 1)
  return c
}

/** Two independent chains, so a light cone on one must delete the other. */
const twoChains = (): Circuit => {
  let c = new Circuit(8)
  c = c.h(0).cnot(0, 1).cnot(1, 2).t(2).ry(0.4, 1)
  c = c.h(4).cnot(4, 5).cnot(5, 6).t(6).ry(0.7, 5)
  return c
}

describe('rewrite — qubitsOf covers every op shape', () => {
  it('finds the qubits of each kind', () => {
    const c = new Circuit(5).h(0).cnot(0, 1).swap(1, 2).crx(0.3, 2, 3).ccx(0, 1, 4)
      .cswap(0, 2, 3).unitary([[1, 0], [0, 1]], 4).barrier()
    const ops = c.toJSON().ops as (Record<string, unknown> & { kind: string })[]
    for (const op of ops) {
      const qs = qubitsOf(op)
      expect(qs.every(q => Number.isInteger(q) && q >= 0 && q < 5), op.kind).toBe(true)
      if (op.kind !== 'barrier') expect(qs.length, op.kind).toBeGreaterThan(0)
    }
  })
})

describe('rewrite — light cone preserves the marginal it keeps', () => {
  it('deletes an independent half of the circuit entirely', () => {
    const c = twoChains()
    const r = lightCone(c, [0, 1, 2])
    expect(r.preserves).toBe('observable')
    expect(r.circuit.qubits).toBe(3)
    expect(r.after).toBeLessThan(r.before)

    // The marginal on the kept qubits is untouched.
    const want = marginal(c.exactProbs(), [0, 1, 2])
    agree(want, r.circuit.exactProbs(), 'two chains')
  })

  it('keeps the marginal on a real grid circuit', () => {
    const c = rcsCircuit({ rows: 3, cols: 4, depth: 3, seed: 2 })
    for (const about of [[5], [0, 11], [2, 6, 9]]) {
      const r = lightCone(c, about)
      const want = marginal(c.exactProbs(), about)
      const kept = about.map(q => r.mapping![q]!)
      expect(kept.every(q => q >= 0), `about=${about}`).toBe(true)
      agree(want, marginal(r.circuit.exactProbs(), kept), `about=${about}`)
    }
  })

  it('the cone runs backward in time, so downstream gates are not in it', () => {
    /**
     * Worth being explicit about, because it looks wrong at first. A GHZ chain
     * looks fully connected, but asked about qubit 0 the cone keeps only h(0),
     * cnot(0,1) and t(0) — everything after that acts on qubits 1..5, and a
     * unitary on other qubits cannot change qubit 0's reduced state.
     *
     * Six qubits collapse to two, and the marginal is identical.
     */
    const c = ghz(6).t(0)
    const r = lightCone(c, [0])
    expect(r.circuit.qubits).toBe(2)
    expect(r.after).toBeLessThan(r.before)
    agree(marginal(c.exactProbs(), [0]), marginal(r.circuit.exactProbs(), [r.mapping![0]!]), 'ghz q0')
  })

  it('asking about the last qubit keeps the whole chain', () => {
    // The other direction: qubit 5's marginal needs every gate before it.
    const c = ghz(6)
    const r = lightCone(c, [5])
    expect(r.circuit.qubits).toBe(6)
    expect(r.after).toBe(r.before)
    agree(c.exactProbs(), r.circuit.exactProbs(), 'ghz q5')
  })

  it('drops idle qubits rather than keeping them, since 2^n counts them', () => {
    let c = new Circuit(10).h(0).cnot(0, 1)
    c = c.h(7)                                   // untouched by the cone on {0,1}
    const r = lightCone(c, [0, 1])
    expect(r.circuit.qubits).toBe(2)
    expect(r.mapping![7]).toBe(-1)
  })

  it('refuses a circuit whose causality it cannot analyse', () => {
    const impure = new Circuit(3).h(0).creg('m', 1).measure(0, 'm', 0)
    expect(() => lightCone(impure, [0])).toThrow(/pure circuit/)
    expect(() => lightCone(ghz(4), [])).toThrow(RangeError)
    expect(() => lightCone(ghz(4), [9])).toThrow(RangeError)
  })
})

describe('rewrite — peephole preserves the whole state', () => {
  const cases: [string, Circuit][] = [
    ['self-inverse pairs', new Circuit(4).h(0).h(0).x(1).x(1).cnot(0, 1).cnot(0, 1).h(2)],
    ['named inverses', new Circuit(4).s(0).sdg(0).t(1).tdg(1).h(2).cnot(2, 3)],
    ['additive rotations', new Circuit(3).rz(0.3, 0).rz(0.4, 0).rx(0.1, 1).rx(-0.1, 1).ry(0.2, 2)],
    ['nothing to do', ghz(5).t(0)],
    ['interleaved, so no pairing', new Circuit(3).h(0).cnot(0, 1).h(0)],
    ['cascading after a cancel', new Circuit(2).h(0).x(0).x(0).h(0).t(1)],
  ]

  for (const [name, c] of cases) {
    it(`${name}: the distribution is unchanged`, () => {
      const r = peephole(c)
      expect(r.preserves).toBe('state')
      expect(r.after).toBeLessThanOrEqual(r.before)
      agree(c.exactProbs(), r.circuit.exactProbs(), name)
    })
  }

  it('cancels what it should and leaves what it should not', () => {
    expect(peephole(new Circuit(2).h(0).h(0)).after).toBe(0)
    expect(peephole(new Circuit(2).s(0).sdg(0)).after).toBe(0)
    expect(peephole(new Circuit(2).cnot(0, 1).cnot(0, 1)).after).toBe(0)
    // A gate between them means they are not adjacent on that wire.
    expect(peephole(new Circuit(2).h(0).t(0).h(0)).after).toBe(3)
  })

  it('cascades: removing a pair exposes the pair outside it', () => {
    const r = peephole(new Circuit(1).h(0).x(0).x(0).h(0))
    expect(r.after).toBe(0)
  })

  it('merging rotations removes T gates, which is what the magic attack is priced on', () => {
    // Two T gates in a row are an S gate — a Clifford. A circuit can look
    // expensive purely because nobody simplified it.
    let c = new Circuit(6).h(0)
    for (let i = 0; i < 6; i++) c = c.rz(Math.PI / 4, i).rz(Math.PI / 4, i)
    const before = analyzeHardness(c, { probeChi: 8, restarts: 2 })
    const after = analyzeHardness(peephole(c).circuit, { probeChi: 8, restarts: 2 })
    const t = (r: typeof before): number =>
      r.attacks.find(a => a.name === 'stabilizer-rank')!.parameter!.value
    expect(t(after)).toBeLessThan(t(before))
    agree(c.exactProbs(), peephole(c).circuit.exactProbs(), 'rotation merge')
  })
})

describe('rewrite — the basis change buys eligibility, not size', () => {
  it('Rx and Ry become Z-basis rotations without moving the state', () => {
    for (const c of [
      new Circuit(3).rx(0.7, 0).cnot(0, 1).rx(-0.3, 2),
      new Circuit(3).ry(1.1, 1).h(0).ry(0.2, 2).cz(0, 1),
      new Circuit(4).h(0).rx(Math.PI / 4, 1).ry(0.9, 2).cnot(1, 2).rx(0.5, 3),
    ]) {
      const r = toZBasis(c)
      expect(r.preserves).toBe('state')
      agree(c.exactProbs(), r.circuit.exactProbs(), 'z-basis')
    }
  })

  it('it makes the circuit bigger, which is the cost it is trading', () => {
    const c = new Circuit(2).rx(0.4, 0).rx(0.4, 1)
    const r = toZBasis(c)
    expect(r.after).toBeGreaterThan(r.before)     // H·Rz·H is three gates, not one
  })

  it('leaves a circuit with no transverse rotations exactly as it was', () => {
    const c = ghz(5).t(0).rz(0.4, 1)
    const r = toZBasis(c)
    expect(r.after).toBe(r.before)
    agree(c.exactProbs(), r.circuit.exactProbs(), 'no-op')
  })

  it('exposes T-count the magic attack could not otherwise see', () => {
    /**
     * The case this exists for. `Rzz(−π/2)` is Clifford, so a Trotter circuit's
     * only non-Clifford gate is `Rx(θ)` — and the stabilizer-rank backend takes
     * `rz` and not `rx`, so it declines the whole circuit on an axis convention.
     * Rewritten, the same circuit is Clifford plus Z rotations and can be priced.
     */
    let c = new Circuit(6)
    for (let i = 0; i + 1 < 6; i++) c = c.cnot(i, i + 1).rz(-Math.PI / 2, i + 1).cnot(i, i + 1)
    for (let i = 0; i < 6; i++) c = c.rx(Math.PI / 4, i)

    expect(() => c.toSrOps()).toThrow(/not supported/)
    const z = toZBasis(c)
    expect(() => z.circuit.toSrOps()).not.toThrow()

    const before = analyzeHardness(c, { probeChi: 4, restarts: 2 })
    const after = analyzeHardness(z.circuit, { probeChi: 4, restarts: 2 })
    expect(before.attacks.find(a => a.name === 'stabilizer-rank')!.applicable).toBe(false)
    const sr = after.attacks.find(a => a.name === 'stabilizer-rank')!
    expect(sr.applicable).toBe(true)
    expect(sr.parameter!.value).toBeGreaterThan(0)     // the T-count is finally visible
    agree(c.exactProbs(), z.circuit.exactProbs(), 'trotter z-basis')
  })
})

describe('rewrite — reordering preserves the state up to the relabelling', () => {
  it('the distribution comes back when the permutation is undone', () => {
    const c = rcsCircuit({ rows: 2, cols: 4, depth: 3, seed: 5 })
    const r = searchQubitOrder(c, { restarts: 4 })
    expect(r.preserves).toBe('permuted-state')

    const map = r.mapping!
    const want = c.exactProbs()
    const got = r.circuit.exactProbs()
    for (const [bits, p] of Object.entries(want)) {
      const moved = Array.from({ length: c.qubits }, (_, q) => bits[q]!)
      const permuted = new Array<string>(c.qubits)
      for (let q = 0; q < c.qubits; q++) permuted[map[q]!] = moved[q]!
      expect(got[permuted.join('')] ?? 0, `bits ${bits}`).toBeCloseTo(p, 10)
    }
  })

  it('never lengthens the total two-qubit span it is minimising', () => {
    const c = rcsCircuit({ rows: 3, cols: 4, depth: 4, seed: 7 })
    const identity = Array.from({ length: c.qubits }, (_, i) => i)
    const r = searchQubitOrder(c, { restarts: 6 })
    expect(orderingCost(c, r.mapping!)).toBeLessThanOrEqual(orderingCost(c, identity))
  })

  it('spectral ordering is a seed, not a guarantee', () => {
    /**
     * Measured: it shortens an RCS 4x4 from span 108 to 72, and *lengthens* a 6x6
     * from 210 to 226 and an IBM light cone from 1420 to 2230. So it is seeded
     * into the search alongside the identity rather than trusted, and the search
     * keeps whichever wins. Greedy swaps alone cannot reach a snake layout from a
     * random start, which is why the seed is worth having at all.
     */
    for (const c of [
      rcsCircuit({ rows: 4, cols: 4, depth: 6, seed: 1 }),
      rcsCircuit({ rows: 6, cols: 6, depth: 4, seed: 1 }),
      rcsCircuit({ rows: 2, cols: 5, depth: 5, seed: 3 }),
    ]) {
      const identity = Array.from({ length: c.qubits }, (_, i) => i)
      const searched = searchQubitOrder(c, { restarts: 4 }).mapping!
      // The search may never do worse than doing nothing, whatever the seed did.
      expect(orderingCost(c, searched)).toBeLessThanOrEqual(orderingCost(c, identity))
    }
  })

  it('spectral ordering is a permutation', () => {
    for (const c of [rcsCircuit({ rows: 3, cols: 4, depth: 4, seed: 2 }), ghz(9)]) {
      const m = spectralOrder(c)
      expect(m.length).toBe(c.qubits)
      expect([...m].sort((a, b) => a - b)).toEqual(Array.from({ length: c.qubits }, (_, i) => i))
    }
  })

  it('a circuit already laid out on a line has nothing to find', () => {
    const r = searchQubitOrder(ghz(8), { restarts: 4 })
    const identity = Array.from({ length: 8 }, (_, i) => i)
    expect(orderingCost(ghz(8), r.mapping!)).toBe(orderingCost(ghz(8), identity))
  })
})

describe('rewrite — the search is judged on the report, not a proxy', () => {
  const B = { memoryBytes: 16e9, seconds: 3600, probeChi: 8, restarts: 4 }

  it('never accepts a rewrite that makes the report worse', () => {
    /**
     * The bug this exists to prevent. An earlier version accepted the qubit
     * relabelling whenever it shortened the total two-qubit span — a plausible
     * stand-in for bond dimension — and measured a result 4% *worse*. A search
     * that moves on a proxy can walk downhill.
     */
    for (const c of [
      rcsCircuit({ rows: 4, cols: 6, depth: 6, seed: 3 }),
      rcsCircuit({ rows: 3, cols: 3, depth: 5, seed: 8 }),
      ghz(10).t(0),
    ]) {
      const s = searchHardness(c, B)
      if (s.before.best && s.after.best) {
        expect(s.after.best.operations!).toBeLessThanOrEqual(s.before.best.operations!)
      }
      if (s.before.best === null) expect(true).toBe(true)      // breaking is always an improvement
      else expect(s.after.best).not.toBe(null)                 // never loses a break it already had
      expect(s.speedup === null || s.speedup > 1).toBe(true)
    }
  })

  it('reports the rewrites it kept, and what each one preserves', () => {
    const c = twoChains()
    const s = searchHardness(c, { ...B, about: [0, 1, 2] })
    expect(s.applied.some(r => r.name === 'light-cone')).toBe(true)
    for (const r of s.applied) expect(['state', 'observable', 'permuted-state']).toContain(r.preserves)
    expect(renderSearch(s)).toContain('light-cone')
  })

  it('reports rewrites it tried and discarded', () => {
    /**
     * A discarded rewrite leaves the *pre-rewrite* refusal in the report.
     * `toZBasis` makes the magic attack applicable to a circuit written with
     * `rx`; when the search rejects it because the T-count is hopeless anyway,
     * the surviving row still says "gate 'rx' is not supported" — which reads as
     * a hard incompatibility rather than a decision unless the attempt is shown.
     */
    const c = trotterIsing({ qubits: 10, coupling: Array.from({ length: 9 }, (_, i) => [i, i + 1] as [number, number]), depth: 3, thetaH: Math.PI / 4 })
    const s = searchHardness(c, { ...B, about: [5] })
    expect(s.rejected.some(r => r.name === 'z-basis')).toBe(true)
    const text = renderSearch(s)
    expect(text).toContain('tried, discarded')
    // And the pre-rewrite refusal it explains is still visible.
    expect(s.after.attacks.find(a => a.name === 'stabilizer-rank')!.reason).toMatch(/rx/)
  })

  it('every rewrite is either applied or reported as discarded', () => {
    const c = rcsCircuit({ rows: 3, cols: 3, depth: 4, seed: 2 })
    const s = searchHardness(c, { ...B, about: [4] })
    const seen = [...s.applied.map(r => r.name), ...s.rejected.map(r => r.name)]
    for (const name of ['light-cone', 'peephole', 'z-basis', 'qubit-order']) {
      expect(seen, `${name} vanished from the report`).toContain(name)
    }
  })

  it('says so plainly when nothing helped', () => {
    const s = searchHardness(ghz(6), B)
    expect(s.applied).toEqual([])
    expect(s.speedup).toBe(null)
    expect(renderSearch(s)).toContain('no rewrite helped')
  })

  it('breaks a circuit that nothing could break as written', () => {
    // The headline. 36 qubits at depth 14 defeats every attack; asked about one
    // qubit, the light cone deletes a third of the gates and treewidth lands.
    const c = rcsCircuit({ rows: 6, cols: 6, depth: 14, seed: 1 })
    const s = searchHardness(c, { ...B, about: [14] })
    expect(s.before.best).toBe(null)
    expect(s.after.best).not.toBe(null)
    expect(s.broke).toBe(true)
    expect(renderSearch(s)).toContain('broke a circuit')
  }, 120_000)

  it('carries the observable through a rewrite that renumbers qubits', () => {
    /**
     * The bug this exists to prevent, found by pointing the adversary at a
     * published circuit. A light cone shrank 127 qubits to 88, the observable
     * stayed 127 letters long, and both Pauli attacks refused it as malformed —
     * silently deleting the two attacks the light cone had just made affordable.
     * The report looked like "nothing else applies" rather than "I broke it".
     */
    const c = twoChains()
    const n = c.qubits
    const obs = [{ coeff: 1, ops: Array.from({ length: n }, (_, k) => k === n - 1 - 2 ? 'Z' : 'I').join('') }]
    const s = searchHardness(c, { ...B, about: [0, 1, 2], observable: obs })

    expect(s.applied.some(r => r.name === 'light-cone')).toBe(true)
    for (const name of ['pauli-path', 'noisy-pauli-path'] as const) {
      const a = s.after.attacks.find(x => x.name === name)!
      expect(a.reason ?? '', name).not.toMatch(/one letter per qubit/)
    }
    expect(s.after.attacks.find(x => x.name === 'pauli-path')!.applicable).toBe(true)
  })

  it('the remapped observable computes the same expectation value', () => {
    // Stronger than "it did not throw": the light cone plus the remap has to
    // give the number the full circuit gives.
    const c = twoChains()
    const n = c.qubits
    const obs = { coeff: 1, ops: Array.from({ length: n }, (_, k) => k === n - 1 - 2 ? 'Z' : 'I').join('') }
    const full = pauliPathExpectation(c, [obs], { threshold: 0 }).value

    const r = lightCone(c, [0, 1, 2])
    const m = r.mapping!
    const k = r.circuit.qubits
    const letters = new Array<string>(k).fill('I')
    for (let q = 0; q < n; q++) {
      if (m[q]! >= 0) letters[k - 1 - m[q]!] = obs.ops[n - 1 - q]!
    }
    const reduced = pauliPathExpectation(r.circuit, [{ coeff: 1, ops: letters.join('') }], { threshold: 0 }).value
    expect(reduced).toBeCloseTo(full, 10)
  })

  it('declines a cone that would cost it the observable', () => {
    /**
     * An observable on a qubit the cone discards cannot be carried across, so
     * carrying it would mean dropping it — and dropping it costs both Pauli
     * attacks. Because acceptance is judged on the whole report, the search
     * refuses the cone instead. It gives up the gate reduction to keep an attack,
     * which is the right trade and one a size-based rule would have got backwards.
     */
    const c = twoChains()
    const n = c.qubits
    const outside = [{ coeff: 1, ops: Array.from({ length: n }, (_, k) => k === n - 1 - 5 ? 'Z' : 'I').join('') }]
    const s = searchHardness(c, { ...B, about: [0, 1, 2], observable: outside })

    expect(s.applied.some(r => r.name === 'light-cone')).toBe(false)
    expect(s.after.qubits).toBe(n)
    expect(s.after.attacks.find(x => x.name === 'pauli-path')!.applicable).toBe(true)
  })

  it('no attack is ever declined for a stale observable length', () => {
    // The invariant behind the fix: whatever rewrites are applied, an attack that
    // declines must do so for a real reason, never because the search handed it a
    // string sized for a circuit that no longer exists.
    for (const about of [[0], [0, 1, 2], [4, 5, 6]]) {
      for (const q of [0, 2, 5]) {
        const c = twoChains()
        const n = c.qubits
        const obs = [{ coeff: 1, ops: Array.from({ length: n }, (_, k) => k === n - 1 - q ? 'Z' : 'I').join('') }]
        const s = searchHardness(c, { ...B, about, observable: obs })
        for (const a of s.after.attacks) {
          expect(a.reason ?? '', `about=${about} q=${q} ${a.name}`).not.toMatch(/one letter per qubit/)
        }
      }
    }
  })

  it('the light cone narrows what the answer covers, and the result says so', () => {
    // The reduced circuit answers about the named qubits only. Anyone quoting the
    // verdict has to see that, which is why `preserves` is on every rewrite.
    const c = rcsCircuit({ rows: 4, cols: 4, depth: 4, seed: 1 })
    const s = searchHardness(c, { ...B, about: [5] })
    if (s.applied.some(r => r.name === 'light-cone')) {
      expect(s.applied.find(r => r.name === 'light-cone')!.preserves).toBe('observable')
      expect(s.circuit.qubits).toBeLessThanOrEqual(c.qubits)
    }
  })
})
