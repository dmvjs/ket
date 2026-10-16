import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import * as G from './gates.js'
import { pepsApply1, pepsEnvironments, pepsExpectation, pepsExpectationExact, pepsInit, pepsRun } from './peps.js'
import { rcsCircuit } from './rcs.js'

const zExact = (c: Circuit, q: number): number => {
  let z = 0
  for (const [bits, p] of Object.entries(c.exactProbs())) z += (bits[q] === '1' ? -1 : 1) * p
  return z
}
const grid = (r: number, c: number): [number, number][] => {
  const q = (i: number, j: number): number => i * c + j
  const e: [number, number][] = []
  for (let i = 0; i < r; i++) for (let j = 0; j + 1 < c; j++) e.push([q(i, j), q(i, j + 1)])
  for (let j = 0; j < c; j++) for (let i = 0; i + 1 < r; i++) e.push([q(i, j), q(i + 1, j)])
  return e
}
const worstError = (c: Circuit, st: ReturnType<typeof pepsRun>): number => {
  let w = 0
  for (let q = 0; q < c.qubits; q++) w = Math.max(w, Math.abs(pepsExpectation(st, q, 'Z') - zExact(c, q)))
  return w
}

describe('peps — exact where simple update is exact', () => {
  it('reproduces the statevector on a chain, which is a tree', () => {
    let c = new Circuit(7)
    for (let i = 0; i < 7; i++) c = c.ry(0.3 + 0.2 * i, i)
    for (let i = 0; i < 6; i++) c = c.cnot(i, i + 1).rz(0.4, i + 1)
    expect(worstError(c, pepsRun(c, { maxBond: 32 }))).toBeLessThan(1e-12)
  })

  it('a product state is a product state', () => {
    const st = pepsInit(4, [[0, 1], [1, 2], [2, 3]])
    pepsApply1(st, 1, G.X)
    pepsApply1(st, 2, G.H)
    expect(pepsExpectation(st, 0, 'Z')).toBeCloseTo(1, 12)
    expect(pepsExpectation(st, 1, 'Z')).toBeCloseTo(-1, 12)
    expect(pepsExpectation(st, 2, 'Z')).toBeCloseTo(0, 12)
    expect(pepsExpectation(st, 2, 'X')).toBeCloseTo(1, 12)
  })

  it('bonds grow only as far as the gates require', () => {
    // A shallow grid circuit needs very little bond dimension, and the engine
    // must not manufacture any: nothing truncated means nothing approximated.
    const c = rcsCircuit({ rows: 3, cols: 4, depth: 4, seed: 2 })
    const st = pepsRun(c, { maxBond: 32, coupling: grid(3, 4) })
    expect(st.discarded).toBeLessThan(1e-20)
    expect(worstError(c, st)).toBeLessThan(1e-12)
  })

  it('rejects a gate across a bond the network does not have', () => {
    const c = new Circuit(4).h(0).cnot(0, 2)
    expect(() => pepsRun(c, { coupling: [[0, 1], [1, 2], [2, 3]] })).toThrow(/do not share a bond/)
  })

  it('rejects a malformed coupling graph', () => {
    expect(() => pepsInit(4, [[1, 1]])).toThrow(/self-loop/)
    expect(() => pepsInit(4, [[0, 9]])).toThrow(/outside/)
  })
})

describe('peps — the error floor is the environment, not the bond dimension', () => {
  /**
   * The measurement that decides how this may be used. On a loopy grid at χ=16
   * the discarded weight falls to 1e-30 — *nothing is being truncated* — and the
   * error in ⟨Z⟩ stays at 3.5e-2, unchanged at χ=32. That residual is the
   * simple-update approximation: each site is updated against its own bond
   * weights rather than its true environment, and no bond dimension repairs it.
   */
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 8, seed: 2 })
  const at = (chi: number) => pepsRun(c, { maxBond: chi, coupling: grid(3, 4) })

  it('stops truncating well before it stops being wrong', () => {
    const st = at(16)
    expect(st.discarded).toBeLessThan(1e-20)     // no truncation at all
    expect(worstError(c, st)).toBeGreaterThan(1e-3)  // and still wrong
  })

  it('doubling the bond dimension changes nothing once truncation has stopped', () => {
    /**
     * Why the χ-doubling convergence check used for MPS must **not** be reused
     * here: both runs agree to the last digit and both are wrong by 3%. A check
     * built on their agreement would report convergence on a wrong answer, which
     * is the one failure mode this library is arranged to prevent.
     */
    const lo = at(16), hi = at(32)
    for (let q = 0; q < c.qubits; q++) {
      expect(pepsExpectation(hi, q, 'Z'), `q${q}`).toBeCloseTo(pepsExpectation(lo, q, 'Z'), 12)
    }
    expect(worstError(c, lo)).toBeGreaterThan(1e-3)
    expect(worstError(c, hi)).toBeGreaterThan(1e-3)
  })

  it('an exact environment removes the floor entirely', () => {
    /**
     * The measurement that located the error. On a state with *zero* truncation,
     * the bond-weight estimate is wrong by 7.5e-2 and contracting the norm
     * network exactly is wrong by 5.8e-16. The state was right the whole time;
     * the tree-shaped environment was not.
     */
    const k = rcsCircuit({ rows: 3, cols: 3, depth: 10, seed: 2 })
    const st = pepsRun(k, { maxBond: 8, coupling: grid(3, 3) })
    expect(st.discarded).toBeLessThan(1e-20)          // nothing truncated
    const env = pepsEnvironments(st, { maxIterations: 200 })

    let bondWeight = 0, exact = 0
    for (let q = 0; q < k.qubits; q++) {
      const truth = zExact(k, q)
      bondWeight = Math.max(bondWeight, Math.abs(pepsExpectation(st, q, 'Z') - truth))
      const r = pepsExpectationExact(st, q, 'Z', env, { maxWidth: 26 })!
      exact = Math.max(exact, Math.abs(r.value - truth))
    }
    expect(bondWeight).toBeGreaterThan(1e-3)
    expect(exact).toBeLessThan(1e-10)
  })

  it('but a better environment on a truncated state reads worse', () => {
    /**
     * The trap. Simple update truncates *in the bond-weight gauge*, so its
     * truncation error and its environment error partially cancel and the
     * estimate is self-consistent. Swapping in an exact environment removes one
     * of the two and exposes the full truncation damage — measured at 2.2e-1
     * against the bond-weight estimate's 3.3e-2 on the same state.
     *
     * The two errors cannot be attacked separately: accuracy needs a larger bond
     * dimension *and* a better environment together.
     */
    const k = rcsCircuit({ rows: 3, cols: 4, depth: 8, seed: 2 })
    const st = pepsRun(k, { maxBond: 4, coupling: grid(3, 4) })
    expect(st.discarded).toBeGreaterThan(1e-3)        // genuinely truncated
    const env = pepsEnvironments(st, { maxIterations: 200 })

    let bondWeight = 0, exact = 0
    for (let q = 0; q < k.qubits; q++) {
      const truth = zExact(k, q)
      bondWeight = Math.max(bondWeight, Math.abs(pepsExpectation(st, q, 'Z') - truth))
      const r = pepsExpectationExact(st, q, 'Z', env, { maxWidth: 26 })!
      exact = Math.max(exact, Math.abs(r.value - truth))
    }
    expect(exact).toBeGreaterThan(bondWeight)
  })

  it('reports the width rather than attempting a contraction it cannot finish', () => {
    const k = rcsCircuit({ rows: 4, cols: 4, depth: 6, seed: 1 })
    const st = pepsRun(k, { maxBond: 8, coupling: grid(4, 4) })
    const env = pepsEnvironments(st, { maxIterations: 20 })
    const r = pepsExpectationExact(st, 0, 'Z', env, { maxWidth: 6 })!
    expect(r.width).toBeGreaterThan(6)
    expect(Number.isNaN(r.value)).toBe(true)
  })

  it('more bond dimension does help while truncation is still happening', () => {
    // Below the saturation point the error does fall, so the engine is working;
    // it is the floor beneath it that is the approximation.
    expect(worstError(c, at(4))).toBeLessThan(worstError(c, at(2)))
  })
})
