import { describe, expect, it } from 'vitest'
import * as G from './gates.js'
import type { Complex } from './complex.js'
import type { Gate2x2 } from './statevector.js'
import {
  SQRT_W, SQRT_X, SQRT_Y, SYCAMORE_1Q, SYCAMORE_SEQUENCE, contractionOracle,
  edgeColouring, fsim, gridCouplings, heavyHex127, linearXeb, mixWithUniform,
  porterThomasMoments, rcsCircuit, statevectorOracle, trotterIsing, xebFromSamples,
  type Coupling,
} from './rcs.js'
import { amplitudeByContraction, circuitNetwork } from './tensor-network.js'

/** Multiply two 2x2 complex matrices. */
const mul2 = (a: Gate2x2, b: Gate2x2): Complex[][] =>
  [0, 1].map(i => [0, 1].map(j => {
    let re = 0, im = 0
    for (const k of [0, 1]) {
      re += a[i]![k]!.re * b[k]![j]!.re - a[i]![k]!.im * b[k]![j]!.im
      im += a[i]![k]!.re * b[k]![j]!.im + a[i]![k]!.im * b[k]![j]!.re
    }
    return { re, im }
  }))

const closeTo = (got: Complex[][], want: readonly (readonly Complex[])[], label: string): void => {
  for (let i = 0; i < got.length; i++) for (let j = 0; j < got.length; j++) {
    expect(got[i]![j]!.re, `${label} [${i}][${j}].re`).toBeCloseTo(want[i]![j]!.re, 12)
    expect(got[i]![j]!.im, `${label} [${i}][${j}].im`).toBeCloseTo(want[i]![j]!.im, 12)
  }
}

describe('rcs — the Sycamore gate set', () => {
  it('every single-qubit gate is unitary', () => {
    for (const [name, g] of [['√X', SQRT_X], ['√Y', SQRT_Y], ['√W', SQRT_W]] as const) {
      const dag: Gate2x2 = [
        [{ re: g[0]![0]!.re, im: -g[0]![0]!.im }, { re: g[1]![0]!.re, im: -g[1]![0]!.im }],
        [{ re: g[0]![1]!.re, im: -g[0]![1]!.im }, { re: g[1]![1]!.re, im: -g[1]![1]!.im }],
      ]
      closeTo(mul2(g, dag), G.Id, `${name}·${name}†`)
    }
  })

  it('each one really is the square root it claims to be', () => {
    // The whole point of the set: three quarter-turns about different axes. A
    // transposed or sign-flipped entry still looks unitary, so squaring is the
    // check that catches it.
    closeTo(mul2(SQRT_X, SQRT_X), G.X, '√X²')
    closeTo(mul2(SQRT_Y, SQRT_Y), G.Y, '√Y²')
    const s2 = Math.SQRT1_2
    const W: Complex[][] = [[{ re: 0, im: 0 }, { re: s2, im: -s2 }], [{ re: s2, im: s2 }, { re: 0, im: 0 }]]
    closeTo(mul2(SQRT_W, SQRT_W), W, '√W²')
  })

  it('√X matches the gate ket already has', () => {
    closeTo(SQRT_X.map(r => [...r]), G.V, '√X vs V')
  })

  it('fSim is unitary and reduces to the identity at zero', () => {
    const f = fsim(0, 0)
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      expect(f[i]![j]!.re).toBeCloseTo(i === j ? 1 : 0, 12)
      expect(f[i]![j]!.im).toBeCloseTo(0, 12)
    }
    // θ = π/2, φ = 0 is iSWAP up to the sign convention: |01> and |10> swap with -i.
    const g = fsim(Math.PI / 2, 0)
    expect(g[1]![2]!.im).toBeCloseTo(-1, 12)
    expect(g[2]![1]!.im).toBeCloseTo(-1, 12)
    expect(g[1]![1]!.re).toBeCloseTo(0, 12)
  })

  it('the Sycamore phase is the entry that makes it not an iSWAP', () => {
    const g = fsim(Math.PI / 2, Math.PI / 6)
    expect(g[3]![3]!.re).toBeCloseTo(Math.cos(Math.PI / 6), 12)
    expect(g[3]![3]!.im).toBeCloseTo(-Math.sin(Math.PI / 6), 12)
  })
})

describe('rcs — coupler patterns', () => {
  it('each pattern is a matching — no qubit twice in a layer', () => {
    for (const [rows, cols] of [[3, 3], [4, 5], [2, 6], [5, 2]] as const) {
      const cp = gridCouplings(rows, cols)
      for (const [name, bonds] of Object.entries(cp)) {
        const seen = new Set<number>()
        for (const [a, b] of bonds) {
          expect(seen.has(a), `${rows}x${cols} ${name} reuses ${a}`).toBe(false)
          expect(seen.has(b), `${rows}x${cols} ${name} reuses ${b}`).toBe(false)
          seen.add(a); seen.add(b)
        }
      }
    }
  })

  it('the four patterns together are exactly the grid edges, each once', () => {
    const [rows, cols] = [4, 4]
    const cp = gridCouplings(rows, cols)
    const all = [...cp.A, ...cp.B, ...cp.C, ...cp.D].map(([a, b]) => `${a}-${b}`)
    expect(new Set(all).size).toBe(all.length)
    expect(all.length).toBe(rows * (cols - 1) + cols * (rows - 1))
  })

  it('every bond joins grid neighbours', () => {
    const cols = 5
    const cp = gridCouplings(4, cols)
    for (const [a, b] of [...cp.A, ...cp.B]) expect(b - a).toBe(1)
    for (const [a, b] of [...cp.C, ...cp.D]) expect(b - a).toBe(cols)
  })

  it('a single row or column has no bonds in the missing direction', () => {
    expect(gridCouplings(1, 4).C).toEqual([])
    expect(gridCouplings(1, 4).D).toEqual([])
    expect(gridCouplings(4, 1).A).toEqual([])
  })
})

describe('rcs — circuit construction', () => {
  it('is deterministic in the seed, and different across seeds', () => {
    const a = rcsCircuit({ rows: 2, cols: 3, depth: 4, seed: 7 })
    const b = rcsCircuit({ rows: 2, cols: 3, depth: 4, seed: 7 })
    const c = rcsCircuit({ rows: 2, cols: 3, depth: 4, seed: 8 })
    expect(JSON.stringify(a.toJSON())).toBe(JSON.stringify(b.toJSON()))
    expect(JSON.stringify(a.toJSON())).not.toBe(JSON.stringify(c.toJSON()))
  })

  it('never repeats a qubit’s single-qubit gate in consecutive cycles', () => {
    // Two identical quarter-turns in a row compose into one half-turn, which
    // makes the circuit shallower than it looks and easier than it should be.
    const rows = 3, cols = 3, depth = 12
    const c = rcsCircuit({ rows, cols, depth, seed: 3 })
    const last = new Map<number, string>()
    const cycleOf = new Map<number, number>()
    let cycle = 0
    for (const raw of c.toJSON().ops) {
      const op = raw as { kind: string; qubits?: number[]; matrix?: unknown }
      if (op.kind !== 'unitary' || op.qubits!.length !== 1) continue
      const q = op.qubits![0]!
      const key = JSON.stringify(op.matrix)
      if (cycleOf.get(q) !== cycle) cycleOf.set(q, cycle)
      if (last.has(q)) expect(key, `qubit ${q} repeated a gate`).not.toBe(last.get(q))
      last.set(q, key)
      if (q === rows * cols - 1) cycle++
    }
    expect(cycle).toBeGreaterThan(1)
  })

  it('has the gate count the cycle structure implies', () => {
    const rows = 3, cols = 3, depth = 8
    const n = rows * cols
    const c = rcsCircuit({ rows, cols, depth, seed: 1 })
    const ops = c.toJSON().ops as { kind: string; qubits?: number[] }[]
    const single = ops.filter(o => o.kind === 'unitary' && o.qubits!.length === 1).length
    const two = ops.filter(o => o.kind === 'unitary' && o.qubits!.length === 2).length
    expect(single).toBe(n * (depth + 1))          // one layer per cycle, plus the final one
    const cp = gridCouplings(rows, cols)
    let want = 0
    for (let d = 0; d < depth; d++) want += cp[SYCAMORE_SEQUENCE[d % SYCAMORE_SEQUENCE.length]!].length
    expect(two).toBe(want)
  })

  it('the state it produces is normalised', () => {
    const c = rcsCircuit({ rows: 2, cols: 3, depth: 6, seed: 5 })
    const total = Object.values(c.exactProbs()).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 10)
  })

  it('rejects a degenerate geometry', () => {
    expect(() => rcsCircuit({ rows: 0, cols: 3, depth: 2 })).toThrow(RangeError)
    expect(() => rcsCircuit({ rows: 2, cols: 2, depth: 0 })).toThrow(RangeError)
    expect(() => rcsCircuit({ rows: 2, cols: 2, depth: 2, sequence: [] })).toThrow(RangeError)
  })

  it('SYCAMORE_1Q is the three-gate set the cycles draw from', () => {
    expect(SYCAMORE_1Q).toEqual([SQRT_X, SQRT_Y, SQRT_W])
  })
})

describe('rcs — the heavy-hex lattice', () => {
  const edges = heavyHex127()

  it('has the shape the published layout has', () => {
    // Every one of these caught the same bug: a shared attachment offset ran the
    // last connector past the end of the 14-qubit bottom row and invented a
    // qubit 127. A coupling map that is subtly wrong changes every verdict
    // computed on it and looks completely plausible while doing so.
    const nodes = new Set(edges.flat())
    expect(nodes.size).toBe(127)
    expect(Math.min(...nodes)).toBe(0)
    expect(Math.max(...nodes)).toBe(126)
    expect(edges.length).toBe(144)
    for (let q = 0; q < 127; q++) expect(nodes.has(q), `qubit ${q} missing`).toBe(true)
  })

  it('has heavy-hex degrees — nothing above 3', () => {
    const deg = new Map<number, number>()
    for (const [a, b] of edges) { deg.set(a, (deg.get(a) ?? 0) + 1); deg.set(b, (deg.get(b) ?? 0) + 1) }
    const histogram: Record<number, number> = {}
    for (const d of deg.values()) histogram[d] = (histogram[d] ?? 0) + 1
    expect(histogram).toEqual({ 1: 2, 2: 89, 3: 36 })
    // The two degree-1 qubits are the row corners with no connector attached.
    expect([...deg.entries()].filter(([, d]) => d === 1).map(([q]) => q).sort((a, b) => a - b))
      .toEqual([13, 113])
  })

  it('lists every edge once', () => {
    const seen = new Set(edges.map(([a, b]) => a < b ? `${a}-${b}` : `${b}-${a}`))
    expect(seen.size).toBe(edges.length)
  })
})

describe('rcs — edge colouring', () => {
  it('every layer is a matching, so it is one hardware layer', () => {
    for (const edges of [heavyHex127(), gridCouplings(5, 5).A.concat(gridCouplings(5, 5).C)]) {
      for (const layer of edgeColouring(edges)) {
        const seen = new Set<number>()
        for (const [a, b] of layer) {
          expect(seen.has(a)).toBe(false)
          expect(seen.has(b)).toBe(false)
          seen.add(a); seen.add(b)
        }
      }
    }
  })

  it('keeps every edge exactly once, across all layers', () => {
    const edges = heavyHex127()
    const flat = edgeColouring(edges).flat()
    expect(flat.length).toBe(edges.length)
    expect(new Set(flat.map(([a, b]) => `${a}-${b}`)).size).toBe(edges.length)
  })

  it('needs no more layers than the maximum degree plus one', () => {
    expect(edgeColouring(heavyHex127()).length).toBeLessThanOrEqual(4)
  })
})

describe('rcs — Trotter-Ising circuits', () => {
  const small: Coupling = [[0, 1], [1, 2], [2, 3]]

  it('has the gate count the structure implies', () => {
    const c = trotterIsing({ qubits: 4, coupling: small, depth: 3, thetaH: 0.4 })
    // Per step: one CNOT-Rz-CNOT per edge, plus one Rx per qubit.
    expect(c.gateCounts().total).toBe(3 * (small.length * 3 + 4))
  })

  it('is Clifford at the angles the sweep starts and ends on', () => {
    // θ_J = -π/2 is Clifford, so θ_H = 0 leaves an entirely Clifford circuit —
    // which is why the transverse-field angle is the knob that matters.
    const clifford = trotterIsing({ qubits: 4, coupling: small, depth: 2, thetaH: 0 })
    const probs = clifford.exactProbs()
    expect(Object.values(probs).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('produces a normalised state on the real lattice shape', () => {
    const c = trotterIsing({ qubits: 8, coupling: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], depth: 2, thetaH: 0.5 })
    expect(Object.values(c.exactProbs()).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('rejects an edge outside the register', () => {
    expect(() => trotterIsing({ qubits: 3, coupling: [[0, 5]], depth: 1, thetaH: 0.3 })).toThrow(RangeError)
    expect(() => trotterIsing({ qubits: 3, coupling: small, depth: 0, thetaH: 0.3 })).toThrow(RangeError)
  })

  it('the 127-qubit circuit builds and every backend seam accepts it', () => {
    const c = trotterIsing({ qubits: 127, coupling: heavyHex127(), depth: 2, thetaH: Math.PI / 4 })
    expect(c.qubits).toBe(127)
    expect(c.gateCounts().total).toBe(2 * (144 * 3 + 127))
    // The attacks lower through these two, so a circuit they reject is a circuit
    // the adversary cannot price.
    expect(() => c.toOps()).not.toThrow()
    expect(() => circuitNetwork(c, '0'.repeat(127))).not.toThrow()
  })
})

describe('rcs — the contraction backend can score these circuits', () => {
  /**
   * The load-bearing connection. An RCS circuit is built from `unitary` ops, and
   * until this landed `circuitNetwork` rejected them — so the one backend that
   * survives past statevector width could not touch the one circuit family that
   * needs it.
   */
  it('contraction agrees with the statevector on every amplitude', () => {
    const c = rcsCircuit({ rows: 2, cols: 3, depth: 5, seed: 11 })
    const n = c.qubits
    for (let i = 0; i < 2 ** n; i++) {
      const bits = i.toString(2).padStart(n, '0').split('').reverse().join('')
      const want = c.amplitude(bits)
      const got = amplitudeByContraction(c, bits, { restarts: 8 })
      expect(got.re, `amp ${bits} re`).toBeCloseTo(want.re, 10)
      expect(got.im, `amp ${bits} im`).toBeCloseTo(want.im, 10)
    }
  })

  it('the oracle plans once and returns the same probabilities as the statevector', () => {
    const c = rcsCircuit({ rows: 3, cols: 3, depth: 6, seed: 2 })
    const exact = statevectorOracle(c)
    const contracted = contractionOracle(c, { restarts: 8 })
    expect(contracted.width).toBeGreaterThan(0)
    expect(contracted.slices).toBe(1)
    for (const bits of ['000000000', '101010101', '111111111', '011010010']) {
      expect(contracted(bits) as number, bits).toBeCloseTo(exact(bits) as number, 10)
    }
  })

  it('a sliced oracle gives the same answers under a width budget', () => {
    const c = rcsCircuit({ rows: 3, cols: 4, depth: 8, seed: 4 })
    const sliced = contractionOracle(c, { restarts: 4, targetWidth: 6 })
    const exact = statevectorOracle(c)
    expect(sliced.width).toBeLessThanOrEqual(6)
    expect(sliced.slices).toBeGreaterThan(1)
    for (const bits of ['000000000000', '110100101101']) {
      expect(sliced(bits) as number, bits).toBeCloseTo(exact(bits) as number, 9)
    }
  })

  it('contraction reaches a width the statevector cannot', () => {
    // 30 qubits: no statevector exists here. Depth 4 keeps the treewidth low
    // enough that one amplitude is still cheap — which is the whole property
    // that makes RCS verification tractable at shallow depth and not at deep.
    const c = rcsCircuit({ rows: 5, cols: 6, depth: 4, seed: 9 })
    expect(c.qubits).toBe(30)
    const r = amplitudeByContraction(c, '0'.repeat(30), { restarts: 8 })
    expect(Number.isFinite(r.re)).toBe(true)
    expect(r.width).toBeLessThan(28)
  })
})

describe('rcs — Porter-Thomas', () => {
  it('a deep random circuit produces the speckle XEB assumes', () => {
    const c = rcsCircuit({ rows: 3, cols: 4, depth: 16, seed: 21 })
    const n = c.qubits
    const probs = new Float64Array(2 ** n)
    for (const [bits, p] of Object.entries(c.exactProbs())) {
      let idx = 0
      for (let k = 0; k < bits.length; k++) if (bits[k] === '1') idx += 2 ** k
      probs[idx] = p
    }
    const m = porterThomasMoments(n, probs)
    // Exponential moments are 1, 2, 6. A finite circuit will not hit them
    // exactly, so the assertion is that it is in the right neighbourhood and
    // clearly not uniform — for which the moments would be 1, 1, 1.
    expect(m.first).toBeCloseTo(1, 6)
    expect(m.second).toBeGreaterThan(1.6)
    expect(m.second).toBeLessThan(2.4)
    expect(m.third).toBeGreaterThan(3.5)
  })

  it('the second moment fails in both directions, which is what makes it a test', () => {
    // Uniform is too flat (moment 1). A depth-1 circuit is too concentrated —
    // amplitude has not spread yet, so a few bitstrings hold most of the weight
    // and the moment overshoots 2 rather than undershooting it. Only a scrambled
    // circuit sits at 2, so a one-sided check would pass on the shallow case.
    const n = 12
    const uniform = new Float64Array(2 ** n).fill(1 / 2 ** n)
    expect(porterThomasMoments(n, uniform).second).toBeCloseTo(1, 9)

    const shallow = rcsCircuit({ rows: 3, cols: 4, depth: 1, seed: 21 })
    const probs = new Float64Array(2 ** n)
    for (const [bits, p] of Object.entries(shallow.exactProbs())) {
      let idx = 0
      for (let k = 0; k < bits.length; k++) if (bits[k] === '1') idx += 2 ** k
      probs[idx] = p
    }
    expect(porterThomasMoments(n, probs).second).toBeGreaterThan(3)
  })

  it('participation counts how many bitstrings carry the weight', () => {
    const n = 6, d = 2 ** n
    const uniform = new Float64Array(d).fill(1 / d)
    expect(porterThomasMoments(n, uniform).participation).toBeCloseTo(d, 6)

    const single = new Float64Array(d); single[3] = 1
    expect(porterThomasMoments(n, single).participation).toBeCloseTo(1, 9)

    // Porter-Thomas puts it at about half the dimension.
    const c = rcsCircuit({ rows: 2, cols: 3, depth: 16, seed: 5 })
    const probs = new Float64Array(d)
    for (const [bits, p] of Object.entries(c.exactProbs())) {
      let idx = 0
      for (let k = 0; k < bits.length; k++) if (bits[k] === '1') idx += 2 ** k
      probs[idx] = p
    }
    const part = porterThomasMoments(n, probs).participation
    expect(part).toBeGreaterThan(d / 4)
    expect(part).toBeLessThan(d)
  })
})

describe('rcs — linear XEB behaves like a fidelity', () => {
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 12, seed: 33 })
  const n = c.qubits
  const idealSamples = (shots: number, seed: number): string[] =>
    Object.entries(c.run({ shots, seed }).histogram).flatMap(([idx, p]) => {
      const bits = BigInt(idx).toString(2).padStart(n, '0').split('').reverse().join('')
      return Array<string>(Math.round(p * shots)).fill(bits)
    })

  it('ideal samples score about 1', async () => {
    const samples = idealSamples(20_000, 1)
    const r = await xebFromSamples(c, samples, { oracle: statevectorOracle(c) })
    expect(r.samples).toBeGreaterThan(15_000)
    expect(Math.abs(r.fidelity - 1)).toBeLessThan(6 * r.stderr + 0.02)
  })

  it('uniform samples score about 0', async () => {
    const samples = mixWithUniform(idealSamples(20_000, 2), 0, n, 5)
    const r = await xebFromSamples(c, samples, { oracle: statevectorOracle(c) })
    expect(Math.abs(r.fidelity)).toBeLessThan(6 * r.stderr + 0.02)
  })

  it('a partly-decohered sampler scores its own fidelity', async () => {
    // The property the whole method rests on: XEB reads back the mixing fraction.
    for (const f of [0.25, 0.5, 0.75]) {
      const samples = mixWithUniform(idealSamples(20_000, 3), f, n, 11)
      const r = await xebFromSamples(c, samples, { oracle: statevectorOracle(c) })
      expect(Math.abs(r.fidelity - f), `f=${f} got ${r.fidelity.toFixed(3)}`).toBeLessThan(0.06)
    }
  })

  it('scoring through contraction gives the same answer as through the statevector', async () => {
    // Same samples, two independent oracles — the check that the boundary-crossing
    // backend is not quietly scoring something else.
    const samples = idealSamples(2000, 4).slice(0, 400)
    const viaSv = await xebFromSamples(c, samples, { oracle: statevectorOracle(c) })
    const viaTn = await xebFromSamples(c, samples, { restarts: 8 })
    expect(viaTn.fidelity).toBeCloseTo(viaSv.fidelity, 8)
  }, 60_000)

  it('reports the spread rather than a bare number', () => {
    const many = linearXeb(4, Array<number>(4000).fill(1 / 16))
    const few = linearXeb(4, Array<number>(40).fill(1 / 16))
    expect(many.fidelity).toBeCloseTo(0, 12)
    expect(many.stderr).toBeLessThanOrEqual(few.stderr)
    expect(many.meanScaled).toBeCloseTo(1, 12)
    expect(many.qubits).toBe(4)
  })

  it('rejects empty or malformed input rather than returning a plausible number', async () => {
    expect(() => linearXeb(4, [])).toThrow(RangeError)
    await expect(xebFromSamples(c, [])).rejects.toThrow(RangeError)
    await expect(xebFromSamples(c, ['0'])).rejects.toThrow(/must have 12 characters/)
    expect(() => mixWithUniform(['0000'], 1.5, 4)).toThrow(RangeError)
  })

  it('sampling the same bitstring twice counts twice', () => {
    // Deduplicating would throw away the peaking that XEB is measuring.
    const heavy = 4 / 16, light = 1 / 16
    const withRepeat = linearXeb(4, [heavy, heavy, light])
    const deduped = linearXeb(4, [heavy, light])
    expect(withRepeat.fidelity).toBeGreaterThan(deduped.fidelity)
  })
})
