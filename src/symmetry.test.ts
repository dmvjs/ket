import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import * as G from './gates.js'
import {
  binomial, particleConservingAnsatz, preservesWeight1, preservesWeight2, runSector,
  weightOf, weightSymmetry,
} from './symmetry.js'
import { analyzeHardness, type Attack, type AttackName } from './hardness.js'
import { fsim, rcsCircuit, trotterIsing } from './rcs.js'
import { trotter } from './algorithms.js'
import type { PauliTerm } from './algorithms.js'

const dev = (a: Record<string, number>, b: Record<string, number>): number => {
  let w = 0
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) w = Math.max(w, Math.abs((a[k] ?? 0) - (b[k] ?? 0)))
  return w
}
const find = (r: { attacks: Attack[] }, name: AttackName): Attack => r.attacks.find(a => a.name === name)!
const line = (n: number): [number, number][] => Array.from({ length: n - 1 }, (_, i) => [i, i + 1])
const filled = (n: number, k: number): string => Array.from({ length: n }, (_, i) => i < k ? '1' : '0').join('')

describe('symmetry — detection is numeric, not by gate name', () => {
  it('recognises the gates that move an excitation without creating one', () => {
    // The point of testing the matrix: nothing here shares a naming convention,
    // and fSim's conservation is not something a name would tell you.
    for (const [name, g] of [
      ['iSWAP', G.ISwap], ['√iSWAP', G.SrSwap], ['XY(0.7)', G.Xy(0.7)],
      ['ZZ(0.4)', G.Zz(0.4)], ['fSim', fsim(Math.PI / 2, Math.PI / 6)],
    ] as const) {
      expect(preservesWeight2(g), name).toBe(true)
    }
  })

  it('rejects the gates that do create or destroy one', () => {
    for (const [name, g] of [['XX(0.3)', G.Xx(0.3)], ['YY(0.3)', G.Yy(0.3)]] as const) {
      expect(preservesWeight2(g), name).toBe(false)
    }
    for (const [name, g] of [['H', G.H], ['X', G.X], ['Rx', G.Rx(0.4)], ['Ry', G.Ry(0.4)], ['V', G.V]] as const) {
      expect(preservesWeight1(g), name).toBe(false)
    }
  })

  it('accepts every diagonal one-qubit gate', () => {
    for (const [name, g] of [['Z', G.Z], ['S', G.S], ['T', G.T], ['Rz', G.Rz(0.9)], ['U1', G.U1(0.3)], ['Id', G.Id]] as const) {
      expect(preservesWeight1(g), name).toBe(true)
    }
  })
})

describe('symmetry — a decomposition must not hide the symmetry', () => {
  /**
   * `Rzz` is diagonal and conserving. Written as `CNOT · Rz · CNOT` it is three
   * gates of which two are not, so a gate-by-gate test reports the CNOT and
   * points at the wrong thing entirely. Runs confined to one pair are fused
   * before testing, which asks about the operator rather than the spelling.
   */
  it('CNOT·Rz·CNOT is seen as the diagonal gate it is', () => {
    const c = new Circuit(4).cnot(0, 1).rz(0.7, 1).cnot(0, 1)
    expect(weightSymmetry(c, '1100').conserved).toBe(true)
  })

  it('a bare CNOT is still correctly rejected', () => {
    expect(weightSymmetry(new Circuit(4).cnot(0, 1), '1100').conserved).toBe(false)
  })

  it('the Trotter family conserves at θ_h = 0 and not beyond it', () => {
    // Rzz is conserving; the transverse field is what breaks it. Before fusing,
    // this reported the CNOT inside Rzz and was misleading at every angle.
    const at = (thetaH: number) =>
      weightSymmetry(trotterIsing({ qubits: 10, coupling: line(10), depth: 3, thetaH }), filled(10, 3))
    expect(at(0).conserved).toBe(true)
    for (const th of [Math.PI / 4, Math.PI / 2, 0.3]) {
      expect(at(th).conserved, `θ=${th}`).toBe(false)
      expect(at(th).brokenBy, `θ=${th}`).toMatch(/gates on q/)
    }
  })

  it('RCS breaks it at the single-qubit layer', () => {
    const s = weightSymmetry(rcsCircuit({ rows: 2, cols: 3, depth: 3, seed: 1 }), '110000')
    expect(s.conserved).toBe(false)
  })
})

describe('symmetry — found in a circuit that was never labelled with it', () => {
  /**
   * Heisenberg dynamics conserves total magnetisation, and `trotter()` decomposes
   * it into gates of which *not one* does: `Rxx` and `Ryy` each move weight by
   * two, and their product does not because `XX + YY` annihilates |00⟩ and |11⟩.
   * Nothing tells the detector this — it comes out of multiplying the run.
   *
   * The bug this guards against is subtler than it looks. Growing a run greedily
   * to the widest support absorbs a trailing single-qubit gate belonging to the
   * *next* bond's decomposition, which does not widen the support past two qubits
   * and so joins the run and breaks a product that conserved one gate earlier.
   * Every bond of a chain became a false negative. The fix is to keep the longest
   * prefix whose running product conserves.
   */
  const heisenberg = (n: number, steps: number): Circuit => {
    const terms: PauliTerm[] = []
    for (let i = 0; i + 1 < n; i++) for (const P of ['XX', 'YY', 'ZZ'] as const) {
      const a = Array<string>(n).fill('I')
      a[n - 1 - i] = P[0]!
      a[n - 1 - (i + 1)] = P[1]!
      terms.push({ coeff: 1, ops: a.join('') })
    }
    return trotter(n, terms, 0.3, steps)
  }

  it('individually the pieces do not conserve', () => {
    expect(preservesWeight2(G.Xx(0.6))).toBe(false)
    expect(preservesWeight2(G.Yy(0.6))).toBe(false)
  })

  it('but a Heisenberg chain does, at every width', () => {
    for (const n of [4, 6, 10, 12]) {
      const s = weightSymmetry(heisenberg(n, 3), filled(n, 2))
      expect(s.conserved, `n=${n}`).toBe(true)
      expect(s.sectorDimension, `n=${n}`).toBe(binomial(n, 2))
    }
  })

  it('and a transverse field breaks it again', () => {
    for (const n of [6, 12]) {
      const terms: PauliTerm[] = []
      for (let i = 0; i + 1 < n; i++) for (const P of ['XX', 'YY', 'ZZ'] as const) {
        const a = Array<string>(n).fill('I')
        a[n - 1 - i] = P[0]!; a[n - 1 - (i + 1)] = P[1]!
        terms.push({ coeff: 1, ops: a.join('') })
      }
      for (let i = 0; i < n; i++) {
        const a = Array<string>(n).fill('I'); a[n - 1 - i] = 'X'
        terms.push({ coeff: 0.7, ops: a.join('') })
      }
      expect(weightSymmetry(trotter(n, terms, 0.3, 3), filled(n, 2)).conserved, `n=${n}`).toBe(false)
    }
  })

  it('the sector simulation still reproduces the statevector on it', () => {
    const n = 10
    const c = heisenberg(n, 2)
    const init = filled(n, 3)
    expect(dev(c.run({ shots: 40_000, seed: 4, initialState: init }).probs,
      runSector(c, { initialState: init, shots: 40_000, seed: 4 }).probs)).toBe(0)
  })
})

describe('symmetry — what the sector buys', () => {
  it('binomial is exact where it matters', () => {
    expect(binomial(14, 2)).toBe(91)
    expect(binomial(50, 10)).toBe(10272278170)
    expect(binomial(24, 3)).toBe(2024)
    expect(binomial(10, 0)).toBe(1)
    expect(binomial(10, 10)).toBe(1)
    expect(binomial(5, 7)).toBe(0)
  })

  it('the reported dimension is C(n, k)', () => {
    const c = particleConservingAnsatz({ qubits: 14, layers: 4, seed: 2 })
    for (const k of [1, 2, 4, 7]) {
      const s = weightSymmetry(c, filled(14, k))
      expect(s.conserved, `k=${k}`).toBe(true)
      expect(s.sectorDimension, `k=${k}`).toBe(binomial(14, k))
      expect(s.reduction, `k=${k}`).toBeCloseTo(2 ** 14 / binomial(14, k), 6)
    }
  })

  it('the reduction is large away from half filling and small at it', () => {
    const c = particleConservingAnsatz({ qubits: 16, layers: 3, seed: 5 })
    expect(weightSymmetry(c, filled(16, 2)).reduction).toBeGreaterThan(500)
    expect(weightSymmetry(c, filled(16, 8)).reduction).toBeLessThan(10)
  })

  it('from the vacuum the sector is one state, which is true and useless', () => {
    const c = particleConservingAnsatz({ qubits: 12, layers: 3, seed: 1 })
    expect(weightSymmetry(c, '0'.repeat(12)).sectorDimension).toBe(1)
  })

  it('weightOf counts what it should', () => {
    expect(weightOf('0000')).toBe(0)
    expect(weightOf('1011')).toBe(3)
  })
})

describe('symmetry — the sector simulation is exact', () => {
  it('matches the full statevector, bit for bit', () => {
    for (const [n, layers, k, seed] of [[10, 4, 2, 1], [12, 5, 3, 2], [8, 6, 4, 3]] as const) {
      const c = particleConservingAnsatz({ qubits: n, layers, seed })
      const init = filled(n, k)
      const full = c.run({ shots: 50_000, seed: 9, initialState: init }).probs
      const sector = runSector(c, { initialState: init, shots: 50_000, seed: 9 })
      // Same PRNG, same ordering — the two are the same computation over a
      // different basis, so this should be exact rather than merely close.
      expect(dev(full, sector.probs), `n=${n} k=${k}`).toBe(0)
      expect(sector.sectorDimension, `n=${n} k=${k}`).toBe(binomial(n, k))
    }
  })

  it('never leaves the sector it started in', () => {
    const n = 12, k = 3
    const c = particleConservingAnsatz({ qubits: n, layers: 6, seed: 4 })
    const r = runSector(c, { initialState: filled(n, k), shots: 20_000, seed: 2 })
    for (const bits of Object.keys(r.probs)) expect(weightOf(bits), bits).toBe(k)
  })

  it('probabilities are a distribution', () => {
    const c = particleConservingAnsatz({ qubits: 10, layers: 4, seed: 6 })
    const r = runSector(c, { initialState: filled(10, 3), shots: 20_000, seed: 3 })
    expect(Object.values(r.probs).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
  })

  it('refuses a circuit that does not conserve, naming the culprit', () => {
    const c = rcsCircuit({ rows: 2, cols: 3, depth: 3, seed: 1 })
    expect(() => runSector(c, { initialState: '110000' })).toThrow(/does not conserve excitation count/)
  })

  it('refuses a width where enumeration would defeat the point', () => {
    const c = particleConservingAnsatz({ qubits: 30, layers: 1, seed: 1 })
    expect(() => runSector(c, { initialState: filled(30, 2) })).toThrow(/defeats the purpose/)
  })

  it('rejects an initial state of the wrong length', () => {
    const c = particleConservingAnsatz({ qubits: 8, layers: 2, seed: 1 })
    expect(() => weightSymmetry(c, '111')).toThrow(RangeError)
  })
})

describe('symmetry — as an attack', () => {
  const B = { memoryBytes: 16e9, seconds: 3600, probeChi: 8, restarts: 4 }
  const c = particleConservingAnsatz({ qubits: 20, layers: 6, seed: 7 })
  const init = filled(20, 3)

  it('declines without an initial state, because a law needs a sector', () => {
    expect(find(analyzeHardness(c, B), 'symmetry').reason).toMatch(/no initial state given/)
  })

  it('prices the sector when given one', () => {
    const a = find(analyzeHardness(c, { ...B, initialState: init }), 'symmetry')
    expect(a.applicable).toBe(true)
    expect(a.parameter!.name).toBe('sector dimension C(n,k)')
    expect(a.parameter!.value).toBe(binomial(20, 3))
    // Sufficient, not necessary: more symmetry would only shrink the space.
    expect(a.parameter!.bound).toBe('upper')
    expect(a.viable).toBe(true)
  })

  it('declines on a circuit that breaks the law, and says where', () => {
    const rcs = rcsCircuit({ rows: 3, cols: 4, depth: 4, seed: 2 })
    const a = find(analyzeHardness(rcs, { ...B, initialState: filled(12, 3) }), 'symmetry')
    expect(a.applicable).toBe(false)
    expect(a.reason).toMatch(/not conserved/)
  })

  it('every attack prices the same problem when an initial state is given', () => {
    /**
     * The inconsistency this prevents. Most backends start from |0…0⟩ and the
     * U(1) attack is priced from the state it was handed, so with a populated
     * start they answered different questions — the MPS probe reported χ=1 for a
     * circuit that is trivial from the vacuum and entangling from anywhere else,
     * and then "won".
     */
    const withState = find(analyzeHardness(c, { ...B, probeChi: 32, initialState: init }), 'mps')
    const fromVacuum = find(analyzeHardness(c, { ...B, probeChi: 32 }), 'mps')
    expect(fromVacuum.parameter!.value).toBe(1)              // nothing happens from |0…0⟩
    expect(withState.parameter!.value).toBeGreaterThan(1)    // and plenty does otherwise
  })

  it('appears in every report so its absence is always explained', () => {
    for (const b of [B, { ...B, initialState: init }]) {
      expect(analyzeHardness(c, b).attacks.filter(a => a.name === 'symmetry')).toHaveLength(1)
    }
  })
})
