import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { CliffordSim } from './clifford.js'
import {
  CHSH_OPTIMAL, chshAnalyze, chshCircuits, chshIdeal, runChsh,
  cliffordGroup, rbFit, rbSequences, rbSurvival, runRb,
} from './protocols.js'
import type { Outcomes, RbPoint } from './protocols.js'

const S_MAX = 2 * Math.SQRT2

/** Replay a Clifford-only circuit's serialized ops into a tableau and return its key. */
function tableauOf(c: Circuit): string {
  const sim = new CliffordSim(c.qubits)
  for (const raw of c.toJSON().ops) {
    const op = raw as { kind: string; q?: number; control?: number; target?: number; meta?: { name: string } }
    if (op.kind === 'cnot') sim.cnot(op.control!, op.target!)
    else if (op.kind === 'single') {
      const q = op.q!
      switch (op.meta!.name) {
        case 'h':   sim.h(q);  break
        case 's':   sim.s(q);  break
        case 'sdg': sim.si(q); break
        default: throw new Error(`unexpected gate ${op.meta!.name}`)
      }
    } else throw new Error(`unexpected op ${op.kind}`)
  }
  return sim.tableau().join('')
}

/**
 * Analytic RB prediction for ket's depolarizing model. A gate that applies a
 * random non-identity Pauli with probability p has process fidelity 1 − p on
 * the whole register, which under the Clifford twirl is a depolarizing
 * parameter λ = 1 − p·d²/(d² − 1). The per-Clifford decay is the group
 * average of the product over its gates, and r = (1 − p_rb)(1 − 1/d).
 */
function expectedError(n: 1 | 2, p1: number, p2: number): number {
  const d2 = 4 ** n
  const l1 = 1 - p1 * d2 / (d2 - 1), l2 = 1 - p2 * d2 / (d2 - 1)
  const group = cliffordGroup(n)
  const pRb = group.reduce((s, c) => {
    const g = c.gateCounts()
    return s + l1 ** g.oneQubit * l2 ** g.twoQubit
  }, 0) / group.length
  return (1 - pRb) * (1 - 2 ** -n)
}

// ── CHSH ─────────────────────────────────────────────────────────────────────

describe('chshCircuits', () => {
  it('returns four 2-qubit circuits in the order (a,b), (a,b′), (a′,b), (a′,b′)', () => {
    const cs = chshCircuits()
    expect(cs).toHaveLength(4)
    for (const c of cs) expect(c.qubits).toBe(2)
    // (a,b) = (0, π/4): P(same) = (1 + cos π/4)/2
    const p = cs[0]!.exactProbs()
    expect(p['00']).toBeCloseTo((1 + Math.SQRT1_2) / 4, 10)
    expect(p['11']).toBeCloseTo((1 + Math.SQRT1_2) / 4, 10)
    expect(p['01']).toBeCloseTo((1 - Math.SQRT1_2) / 4, 10)
  })

  it('reaches the Tsirelson bound 2√2 on exact probabilities', () => {
    const r = chshAnalyze(chshCircuits().map(c => c.exactProbs()))
    expect(r.S).toBeCloseTo(S_MAX, 10)
    const c = Math.SQRT1_2
    expect(r.correlations[0]).toBeCloseTo(c, 10)
    expect(r.correlations[1]).toBeCloseTo(-c, 10)
    expect(r.correlations[2]).toBeCloseTo(c, 10)
    expect(r.correlations[3]).toBeCloseTo(c, 10)
    expect(r.stderr).toBeUndefined()
    expect(r.sigma).toBeUndefined()
  })

  it('matches chshIdeal for arbitrary angles', () => {
    const angles = { a: 0.3, a2: 1.9, b: -0.7, b2: 2.4 }
    const r = chshAnalyze(chshCircuits({ angles }).map(c => c.exactProbs()))
    expect(r.S).toBeCloseTo(chshIdeal(angles), 10)
    expect(chshIdeal()).toBeCloseTo(S_MAX, 12)
    expect(chshIdeal(CHSH_OPTIMAL)).toBeCloseTo(S_MAX, 12)
  })

  it('collapses to the classical value 2 when both parties use one setting', () => {
    const angles = { a: 0, a2: 0, b: 0, b2: 0 }
    const r = chshAnalyze(chshCircuits({ angles }).map(c => c.exactProbs()))
    expect(r.S).toBeCloseTo(2, 10)
  })

  it('places the pair on arbitrary qubits in a wider register', () => {
    const qubits = [1, 3] as const
    const cs = chshCircuits({ qubits, width: 4 })
    for (const c of cs) expect(c.qubits).toBe(4)
    const probs = cs.map(c => c.exactProbs())
    expect(chshAnalyze(probs, { qubits }).S).toBeCloseTo(S_MAX, 10)
    // Analysing the wrong pair sees an unentangled qubit and no violation.
    expect(Math.abs(chshAnalyze(probs, { qubits: [0, 1] }).S)).toBeLessThanOrEqual(2 + 1e-9)
  })

  it('defaults width to one past the highest qubit', () => {
    expect(chshCircuits({ qubits: [0, 2] })[0]!.qubits).toBe(3)
  })

  it('signs an infinite sigma by which side of 2 a noiseless S falls on', () => {
    const same = { '00': 1 }
    expect(chshAnalyze([same, same, same, same], { shots: 100 }).sigma).toBe(0)           // S = 2 exactly
    const diff = { '01': 1 }
    expect(chshAnalyze([same, diff, same, same], { shots: 100 }).sigma).toBe(Infinity)    // S = 4
    expect(chshAnalyze([diff, same, diff, diff], { shots: 100 }).sigma).toBe(-Infinity)   // S = −4
  })

  it('recognises a Distribution structurally, not by class identity', () => {
    const d = chshCircuits()[0]!.run({ shots: 500, seed: 1 })
    const foreign = { probs: { ...d.probs }, shots: d.shots } as unknown as Outcomes
    expect(chshAnalyze([foreign, foreign, foreign, foreign]).stderr).toBeDefined()
  })

  it('rejects unknown options with a suggestion', () => {
    expect(() => chshCircuits({ angle: 1 } as never)).toThrow(/did you mean 'angles'/)
    expect(() => chshAnalyze([], { shot: 1 } as never)).toThrow(/unknown option 'shot'/)
    expect(() => rbSequences({ lengths: [1], qubit: [3] } as never)).toThrow(/did you mean 'qubits'/)
    expect(() => rbFit([], { bootstraps: 0 } as never)).toThrow(/did you mean 'bootstrap'/)
  })

  it('rejects a degenerate pair, an out-of-range qubit, and the wrong number of outcomes', () => {
    expect(() => chshCircuits({ qubits: [1, 1] })).toThrow(TypeError)
    expect(() => chshCircuits({ qubits: [0, 2], width: 2 })).toThrow(RangeError)
    expect(() => chshAnalyze([{ '00': 1 }])).toThrow(/4 settings/)
    expect(() => chshAnalyze([{ '0': 1 }, { '0': 1 }, { '0': 1 }, { '0': 1 }])).toThrow(RangeError)
    expect(() => chshAnalyze([{}, { '00': 1 }, { '00': 1 }, { '00': 1 }])).toThrow(/setting 0: outcomes are empty/)
  })

  it('derives error bars from Distribution shots or an explicit shot count', () => {
    const dists = chshCircuits().map(c => c.run({ shots: 2000, seed: 5 }))
    const r = chshAnalyze(dists)
    expect(r.stderr).toBeGreaterThan(0)
    expect(r.stderr!).toBeLessThan(0.05)
    expect(r.sigma).toBeGreaterThan(10)
    expect(Math.abs(r.S - S_MAX)).toBeLessThan(4 * r.stderr!)

    const probs = dists.map(d => d.probs)
    expect(chshAnalyze(probs).stderr).toBeUndefined()
    expect(chshAnalyze(probs, { shots: 2000 }).stderr).toBeCloseTo(r.stderr!, 12)
  })

  it('accepts un-normalised count maps', () => {
    const counts = chshCircuits().map(c => {
      const d = c.run({ shots: 1000, seed: 9 })
      return Object.fromEntries(Object.entries(d.probs).map(([k, p]) => [k, Math.round(p * 1000)]))
    })
    const fromCounts = chshAnalyze(counts)
    const fromProbs  = chshAnalyze(chshCircuits().map(c => c.run({ shots: 1000, seed: 9 })))
    expect(fromCounts.S).toBeCloseTo(fromProbs.S, 10)
  })

  it('shows a smaller violation under two-qubit depolarizing noise', () => {
    const noisy = chshAnalyze(chshCircuits().map(c => c.run({ shots: 4000, seed: 2, noise: { p2: 0.3 } })))
    expect(noisy.S).toBeLessThan(2.4)
    expect(noisy.S).toBeGreaterThan(0)
  })
})

describe('runChsh', () => {
  it('threads a synchronous executor and attaches the circuits', async () => {
    const r = await runChsh(c => c.run({ shots: 4096, seed: 11 }))
    expect(r.circuits).toHaveLength(4)
    expect(Math.abs(r.S - S_MAX)).toBeLessThan(4 * r.stderr!)
  })

  it('accepts an async executor returning probability maps, with shots for the error bar', async () => {
    const seen: number[] = []
    const r = await runChsh(async (c, i) => {
      seen.push(i)
      await Promise.resolve()
      return c.run({ shots: 1000, seed: 3 }).probs
    }, { shots: 1000, qubits: [2, 0], width: 3 })
    expect(seen).toEqual([0, 1, 2, 3])
    expect(r.circuits[0]!.qubits).toBe(3)
    expect(r.stderr).toBeGreaterThan(0)
    expect(Math.abs(r.S - S_MAX)).toBeLessThan(4 * r.stderr!)
  })
})

// ── Hardware path ────────────────────────────────────────────────────────────

describe('protocol circuits are submittable as generated', () => {
  const circuits = [
    ...chshCircuits(),
    rbSequences({ lengths: [8], samples: 1, seed: 1 })[0]!.circuit,
    rbSequences({ qubits: [0, 1], lengths: [8], samples: 1, seed: 1 })[0]!.circuit,
  ]

  it('serialize to IonQ JSON without basis expansion and pass the device check', () => {
    for (const c of circuits) {
      const { circuit } = c.toIonQ()
      expect(circuit.length).toBe(c.gateCounts().total)
      expect(() => c.checkDevice('forte-1')).not.toThrow()
    }
  })

  it('round-trip through OpenQASM with identical probabilities', () => {
    for (const c of circuits) {
      const back = Circuit.fromQASM(c.toQASM())
      const want = c.exactProbs(), got = back.exactProbs()
      for (const k of new Set([...Object.keys(want), ...Object.keys(got)]))
        expect(got[k] ?? 0).toBeCloseTo(want[k] ?? 0, 10)
    }
  })
})

// ── Clifford group ───────────────────────────────────────────────────────────

describe('cliffordGroup', () => {
  it('has 24 one-qubit and 11,520 two-qubit elements, all distinct', () => {
    const g1 = cliffordGroup(1), g2 = cliffordGroup(2)
    expect(g1).toHaveLength(24)
    expect(g2).toHaveLength(11520)
    expect(new Set(g1.map(tableauOf)).size).toBe(24)
    expect(new Set(g2.map(tableauOf)).size).toBe(11520)
  })

  it('starts with the identity and keeps every word short', () => {
    const g1 = cliffordGroup(1), g2 = cliffordGroup(2)
    expect(g1[0]!.gateCounts().total).toBe(0)
    expect(g2[0]!.gateCounts().total).toBe(0)
    expect(Math.max(...g1.map(c => c.gateCounts().total))).toBeLessThanOrEqual(6)
    expect(Math.max(...g2.map(c => c.gateCounts().total))).toBeLessThanOrEqual(10)
    // Breadth-first: word length never decreases along the list.
    let last = 0
    for (const c of g2) { const t = c.gateCounts().total; expect(t).toBeGreaterThanOrEqual(last); last = t }
  })

  it('uses only gates the Clifford backend accepts', () => {
    for (const c of cliffordGroup(2).filter((_, i) => i % 997 === 0))
      expect(() => c.runClifford({ shots: 1 })).not.toThrow()
  })

  it('is cached, so repeated calls are cheap and identical', () => {
    const a = cliffordGroup(2)
    const t = performance.now()
    const b = cliffordGroup(2)
    expect(performance.now() - t).toBeLessThan(200)
    expect(b.map(c => c.toJSON())).toEqual(a.map(c => c.toJSON()))
  })
})

// ── Randomized benchmarking ──────────────────────────────────────────────────

describe('rbSequences', () => {
  it('produces lengths × samples sequences, labelled and reproducible by seed', () => {
    const opts = { lengths: [1, 4, 16], samples: 3, seed: 42 }
    const a = rbSequences(opts), b = rbSequences(opts), c = rbSequences({ ...opts, seed: 43 })
    expect(a).toHaveLength(9)
    expect(a.map(s => [s.length, s.sample])).toEqual([[1,0],[1,1],[1,2],[4,0],[4,1],[4,2],[16,0],[16,1],[16,2]])
    expect(a.map(s => s.circuit.toJSON())).toEqual(b.map(s => s.circuit.toJSON()))
    expect(a.map(s => s.circuit.toJSON())).not.toEqual(c.map(s => s.circuit.toJSON()))
    for (const s of a) expect(s.circuit.qubits).toBe(1)
  })

  it('returns to |0⟩ exactly when noiseless — one qubit', () => {
    for (const s of rbSequences({ lengths: [1, 2, 7, 50], samples: 6, seed: 1 }))
      expect(s.circuit.exactProbs()['0']).toBeCloseTo(1, 10)
  })

  it('returns to |00⟩ exactly when noiseless — two qubits', () => {
    for (const s of rbSequences({ qubits: [0, 1], lengths: [1, 3, 10, 40], samples: 4, seed: 2 }))
      expect(s.circuit.exactProbs()['00']).toBeCloseTo(1, 10)
  })

  it('closes every sequence with a single bounded-depth inverse', () => {
    for (const s of rbSequences({ lengths: [1, 10, 100], samples: 2, seed: 8 }))
      expect(s.circuit.gateCounts().total).toBeLessThanOrEqual((s.length + 1) * 6)
    for (const s of rbSequences({ qubits: [0, 1], lengths: [1, 10, 50], samples: 2, seed: 8 }))
      expect(s.circuit.gateCounts().total).toBeLessThanOrEqual((s.length + 1) * 10)
  })

  it('benchmarks a chosen qubit inside a wider register and leaves the rest alone', () => {
    const seqs = rbSequences({ qubits: [1], width: 3, lengths: [1, 2, 4], samples: 2, seed: 4 })
    for (const s of seqs) {
      expect(s.circuit.qubits).toBe(3)
      const p = s.circuit.exactProbs()
      expect(p['000']).toBeCloseTo(1, 10)
      expect(rbSurvival(p, [1])).toBeCloseTo(1, 10)
    }
    const pair = rbSequences({ qubits: [2, 0], lengths: [3], samples: 1, seed: 4 })[0]!
    expect(pair.circuit.qubits).toBe(3)
    expect(pair.circuit.exactProbs()['000']).toBeCloseTo(1, 10)
  })

  it('rejects bad options', () => {
    expect(() => rbSequences({ qubits: [0, 1, 2], lengths: [1] })).toThrow(RangeError)
    expect(() => rbSequences({ qubits: [1, 1], lengths: [1] })).toThrow(TypeError)
    expect(() => rbSequences({ qubits: [3], width: 2, lengths: [1] })).toThrow(RangeError)
    expect(() => rbSequences({ lengths: [] })).toThrow(TypeError)
    expect(() => rbSequences({ lengths: [0] })).toThrow(RangeError)
    expect(() => rbSequences({ lengths: [1.5] })).toThrow(RangeError)
    expect(() => rbSequences({ lengths: [1], samples: 0 })).toThrow(RangeError)
  })
})

describe('rbSurvival', () => {
  it('is the probability that every benchmarked qubit read 0, ignoring the others', () => {
    const o: Outcomes = { '00': 0.4, '10': 0.3, '01': 0.2, '11': 0.1 }
    expect(rbSurvival(o, [0])).toBeCloseTo(0.6, 12)
    expect(rbSurvival(o, [1])).toBeCloseTo(0.7, 12)
    expect(rbSurvival(o, [0, 1])).toBeCloseTo(0.4, 12)
    expect(rbSurvival(new Circuit(2).x(1).run({ shots: 10 }), [0])).toBe(1)
    expect(rbSurvival(new Circuit(2).x(1).run({ shots: 10 }), [1])).toBe(0)
  })

  it('normalises raw counts and rejects short bitstrings', () => {
    expect(rbSurvival({ '0': 30, '1': 10 })).toBeCloseTo(0.75, 12)
    expect(() => rbSurvival({ '0': 1 }, [1])).toThrow(RangeError)
  })
})

describe('rbFit', () => {
  it('recovers the parameters of an exact decay', () => {
    const pts: RbPoint[] = []
    for (const m of [1, 2, 4, 8, 16, 32, 64]) pts.push({ length: m, survival: 0.5 * 0.97 ** m + 0.5 })
    const fit = rbFit(pts, { bootstrap: 0 })
    expect(fit.p).toBeCloseTo(0.97, 5)
    expect(fit.A).toBeCloseTo(0.5, 5)
    expect(fit.B).toBeCloseTo(0.5, 5)
    expect(fit.errorPerClifford).toBeCloseTo(0.015, 5)
    expect(fit.fidelity).toBeCloseTo(0.985, 5)
    expect(fit.errorPerCliffordStderr).toBeUndefined()
    expect(fit.qubits).toBe(1)
    expect(fit.points.map(p => p.length)).toEqual([1, 2, 4, 8, 16, 32, 64])
    expect(fit.points[0]!.stderr).toBe(0)
    expect(fit.points[0]!.samples).toBe(1)
  })

  it('scales the error by the dimension for two qubits', () => {
    const pts: RbPoint[] = [1, 2, 4, 8, 16].map(m => ({ length: m, survival: 0.75 * 0.9 ** m + 0.25 }))
    const fit = rbFit(pts, { qubits: 2, bootstrap: 0 })
    expect(fit.p).toBeCloseTo(0.9, 5)
    expect(fit.errorPerClifford).toBeCloseTo(0.1 * 0.75, 5)
  })

  it('summarises repeated points per length and bootstraps an error bar', () => {
    const pts: RbPoint[] = []
    const rng = (() => { let s = 7; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x80000000 } })()
    for (const m of [1, 4, 16, 64]) for (let k = 0; k < 10; k++)
      pts.push({ length: m, survival: 0.5 * 0.98 ** m + 0.5 + (rng() - 0.5) * 0.02 })
    const fit = rbFit(pts, { bootstrap: 50, seed: 1 })
    expect(fit.points).toHaveLength(4)
    for (const p of fit.points) { expect(p.samples).toBe(10); expect(p.stderr).toBeGreaterThan(0) }
    expect(fit.errorPerCliffordStderr).toBeGreaterThan(0)
    expect(fit.errorPerCliffordStderr!).toBeLessThan(0.005)
    expect(Math.abs(fit.errorPerClifford - 0.01)).toBeLessThan(0.003)
  })

  it('keeps A and B physical when the decay is slow over the sampled range', () => {
    // Unconstrained, these points are fit by p → 1 with A ≈ +7000, B ≈ −7000.
    const pts: RbPoint[] = [[1, 0.996], [4, 0.992], [16, 0.982], [64, 0.935]]
      .map(([length, survival]) => ({ length: length!, survival: survival! }))
    const fit = rbFit(pts, { bootstrap: 0 })
    expect(fit.A).toBeGreaterThanOrEqual(0); expect(fit.A).toBeLessThanOrEqual(1)
    expect(fit.B).toBeGreaterThanOrEqual(0); expect(fit.B).toBeLessThanOrEqual(1)
    expect(fit.p).toBeGreaterThan(0.99)
    expect(fit.p).toBeLessThan(0.9995)
    expect(fit.errorPerClifford).toBeGreaterThan(2e-4)
  })

  it('handles a flat curve without blowing up', () => {
    const pts: RbPoint[] = [1, 2, 4].map(m => ({ length: m, survival: 1 }))
    const fit = rbFit(pts, { bootstrap: 0 })
    expect(Number.isFinite(fit.A)).toBe(true)
    expect(Number.isFinite(fit.B)).toBe(true)
    expect(fit.A + fit.B).toBeCloseTo(1, 6)
  })

  it('fits A·pᵐ with B held fixed', () => {
    const pts: RbPoint[] = [1, 2, 4, 8].map(m => ({ length: m, survival: 0.5 * 0.95 ** m + 0.5 }))
    const fit = rbFit(pts, { fixB: 0.5, bootstrap: 0 })
    expect(fit.B).toBe(0.5)
    expect(fit.p).toBeCloseTo(0.95, 5)
    expect(fit.A).toBeCloseTo(0.5, 5)
    // Two lengths suffice once B is fixed.
    expect(() => rbFit(pts.slice(0, 2), { fixB: 0.5, bootstrap: 0 })).not.toThrow()
    expect(() => rbFit(pts, { fixB: 1.5 })).toThrow(RangeError)
  })

  it('requires three distinct lengths and a valid qubit count', () => {
    expect(() => rbFit([{ length: 1, survival: 1 }, { length: 2, survival: 1 }])).toThrow(/3 distinct lengths/)
    expect(() => rbFit([], { qubits: 0 })).toThrow(RangeError)
    expect(() => rbFit([{ length: 1, survival: NaN }, { length: 2, survival: 1 }, { length: 3, survival: 1 }])).toThrow(TypeError)
    expect(() => rbFit([{ length: NaN, survival: 1 }, { length: 2, survival: 1 }, { length: 3, survival: 1 }])).toThrow(/length/)
    expect(() => rbFit([{ length: 1.5, survival: 1 }, { length: 2, survival: 1 }, { length: 3, survival: 1 }])).toThrow(/length/)
  })
})

describe('runRb', () => {
  it('reports no error on a noiseless simulator', async () => {
    const rb = await runRb(c => c.runClifford({ shots: 200 }), { lengths: [1, 4, 16, 64], samples: 3, seed: 1, bootstrap: 0 })
    expect(rb.p).toBeCloseTo(1, 6)
    expect(rb.errorPerClifford).toBeLessThan(1e-6)
    expect(rb.sequences).toHaveLength(12)
    expect(rb.survivals).toHaveLength(12)
    for (const s of rb.survivals) expect(s.survival).toBe(1)
    expect(rb.qubits).toBe(1)
  })

  it('recovers the single-qubit depolarizing error per Clifford', async () => {
    const p1 = 0.002
    const rb = await runRb((c, i) => c.runClifford({ shots: 1000, seed: i + 1, noise: { p1 } }), {
      lengths: [1, 2, 4, 8, 16, 32, 64, 128], samples: 20, seed: 7,
    })
    const want = expectedError(1, p1, 0)
    expect(want).toBeCloseTo(0.004, 3)        // three gates per Clifford on average, 2p/3 error each
    expect(rb.errorPerCliffordStderr).toBeGreaterThan(0)
    expect(Math.abs(rb.errorPerClifford - want)).toBeLessThan(3 * rb.errorPerCliffordStderr! + 5e-4)
    expect(rb.A + rb.B).toBeGreaterThan(0.98)  // no SPAM error in the model
  })

  it('recovers the two-qubit error per Clifford, dominated by CNOTs', async () => {
    const p1 = 0.002, p2 = 0.01
    const rb = await runRb((c, i) => c.runClifford({ shots: 1000, seed: i + 1, noise: { p1, p2 } }), {
      qubits: [0, 1], lengths: [1, 2, 4, 8, 16, 32, 64], samples: 20, seed: 7,
    })
    const want = expectedError(2, p1, p2)
    expect(rb.qubits).toBe(2)
    expect(Math.abs(rb.errorPerClifford - want)).toBeLessThan(3 * rb.errorPerCliffordStderr! + 1e-3)
    expect(rb.B).toBeCloseTo(0.25, 1)
  })

  it('attributes readout error to the SPAM terms, not to p', async () => {
    const p1 = 0.005, pMeas = 0.05
    const rb = await runRb((c, i) => c.runClifford({ shots: 1000, seed: i + 1, noise: { p1, pMeas } }), {
      lengths: [1, 2, 4, 8, 16, 32, 64, 128], samples: 20, seed: 3,
    })
    // Same decay as without readout error…
    const want = expectedError(1, p1, 0)
    expect(Math.abs(rb.errorPerClifford - want)).toBeLessThan(3 * rb.errorPerCliffordStderr! + 5e-4)
    // …while the intercept drops by the readout flip probability: A + B = 1 − 2·pMeas·(1 − ½).
    expect(rb.A + rb.B).toBeCloseTo(1 - pMeas, 1)
    expect(rb.B).toBeCloseTo(0.5, 1)
  })

  it('stays physical on a weak decay over few lengths, and pins it down when B is fixed', async () => {
    const p1 = 0.0005
    const opts = { qubits: [3], width: 5, lengths: [1, 4, 16, 64], samples: 5, seed: 1 }
    const run = (c: Circuit, i: number) => c.runClifford({ shots: 500, seed: i + 1, noise: { p1 } })
    const want = expectedError(1, p1, 0)

    // Free B: the range is too short to find the plateau, so r is only known to a factor of ~2…
    const free = await runRb(run, opts)
    expect(free.A).toBeLessThanOrEqual(1); expect(free.B).toBeGreaterThanOrEqual(0)
    expect(free.errorPerClifford).toBeGreaterThan(want / 2.5)
    expect(free.errorPerClifford).toBeLessThan(want * 2.5)

    // …fixing B = ½ removes the ambiguity.
    const fixed = await runRb(run, { ...opts, fixB: 0.5 })
    expect(fixed.B).toBe(0.5)
    expect(Math.abs(fixed.errorPerClifford - want)).toBeLessThan(3 * fixed.errorPerCliffordStderr! + 3e-4)
  })

  it('runs circuits in order and works with an async executor', async () => {
    const seen: number[] = []
    const rb = await runRb(async (c, i) => { seen.push(i); await Promise.resolve(); return c.run({ shots: 100, seed: i }) },
      { lengths: [1, 2, 3], samples: 2, seed: 5, bootstrap: 0 })
    expect(seen).toEqual([0, 1, 2, 3, 4, 5])
    expect(rb.errorPerClifford).toBeLessThan(1e-6)
  })
})
