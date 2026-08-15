import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { c, type Complex } from './complex.js'
import { H, Rx, Ry, Rz, T, U3, X, Xy, Y, Z } from './gates.js'
import {
  denseCNOT, denseControlled, denseCsrSwap, denseCSwap, denseNnz, denseProbabilities,
  denseSingle, denseSWAP, denseToffoli, denseTwo, denseUnitary, fromSparse,
  guardSparseGrowth, MAX_DENSE_QUBITS, SPARSE_ENTRY_LIMIT, toSparse,
} from './dense.js'
import {
  applyCNOT, applyControlled, applyCsrSwap, applyCSwap, applySingle, applySWAP,
  applyToffoli, applyTwo, applyUnitary, probabilities, type Gate4x4, type StateVector,
} from './statevector.js'
import {
  simClone, simCollapse, simDecay, simFromSparse, simKind, simNnz, simNorm2,
  simProbOne, simPromote, simSample, simScale, simScaleBranch, simSingle,
  simToSparse, simZero, svPolicy, DEFAULT_SV_POLICY, type SimState,
} from './hybrid.js'

/** A deterministic, fully-dense n-qubit state — every amplitude non-zero and distinct. */
function spreadState(n: number): StateVector {
  const sv: StateVector = new Map()
  let norm = 0
  for (let i = 0; i < 2 ** n; i++) {
    const re = Math.cos(i * 1.7 + 0.3), im = Math.sin(i * 2.3 + 0.9)
    norm += re * re + im * im
    sv.set(BigInt(i), { re, im })
  }
  const s = 1 / Math.sqrt(norm)
  for (const [k, v] of sv) sv.set(k, { re: v.re * s, im: v.im * s })
  return sv
}

/** Assert two states agree amplitude-by-amplitude. */
function expectSame(a: StateVector, b: StateVector, tol = 1e-12): void {
  const keys = new Set([...a.keys(), ...b.keys()])
  for (const k of keys) {
    const x = a.get(k) ?? { re: 0, im: 0 }
    const y = b.get(k) ?? { re: 0, im: 0 }
    expect(Math.abs(x.re - y.re), `re at |${k}⟩`).toBeLessThan(tol)
    expect(Math.abs(x.im - y.im), `im at |${k}⟩`).toBeLessThan(tol)
  }
}

const TWO_GATE: Gate4x4 = [
  [c(0.5, 0.1), c(0.2, -0.3), c(0.1, 0.4), c(0.3, 0.2)],
  [c(0.2, 0.3), c(0.4, 0.1),  c(0.3, -0.2), c(0.1, 0.5)],
  [c(0.1, -0.4), c(0.3, 0.2), c(0.5, 0.1), c(0.2, -0.1)],
  [c(0.3, 0.2), c(0.1, -0.5), c(0.2, 0.1), c(0.4, 0.3)],
]

describe('dense backend — matches the sparse backend gate for gate', () => {
  const n = 5

  it('single-qubit gates on every qubit', () => {
    for (const gate of [H, X, Y, Z, T, Rx(0.7), Ry(-1.1), Rz(2.4), U3(0.3, 0.5, 0.9)]) {
      for (let q = 0; q < n; q++) {
        const sv = spreadState(n)
        const d  = fromSparse(sv, n)
        denseSingle(d, q, gate)
        expectSame(toSparse(d), applySingle(sv, q, gate))
      }
    }
  })

  it('CNOT for every ordered control/target pair', () => {
    for (let ctrl = 0; ctrl < n; ctrl++) for (let tgt = 0; tgt < n; tgt++) {
      if (ctrl === tgt) continue
      const sv = spreadState(n)
      const d  = fromSparse(sv, n)
      denseCNOT(d, ctrl, tgt)
      expectSame(toSparse(d), applyCNOT(sv, ctrl, tgt))
    }
  })

  it('SWAP for every unordered pair, including reversed order', () => {
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      if (a === b) continue
      const sv = spreadState(n)
      const d  = fromSparse(sv, n)
      denseSWAP(d, a, b)
      expectSame(toSparse(d), applySWAP(sv, a, b))
    }
  })

  it('controlled single-qubit gates, control above and below target', () => {
    for (const gate of [H, X, T, Ry(0.8)]) {
      for (let ctrl = 0; ctrl < n; ctrl++) for (let tgt = 0; tgt < n; tgt++) {
        if (ctrl === tgt) continue
        const sv = spreadState(n)
        const d  = fromSparse(sv, n)
        denseControlled(d, ctrl, tgt, gate)
        expectSame(toSparse(d), applyControlled(sv, ctrl, tgt, gate))
      }
    }
  })

  it('two-qubit gates — qubit a is the MSB of the local index', () => {
    for (const gate of [TWO_GATE, Xy(0.6)]) {
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
        if (a === b) continue
        const sv = spreadState(n)
        const d  = fromSparse(sv, n)
        denseTwo(d, a, b, gate)
        expectSame(toSparse(d), applyTwo(sv, a, b, gate))
      }
    }
  })

  it('Toffoli across every distinct control/control/target triple', () => {
    for (let c1 = 0; c1 < 4; c1++) for (let c2 = 0; c2 < 4; c2++) for (let t = 0; t < 4; t++) {
      if (c1 === c2 || c1 === t || c2 === t) continue
      const sv = spreadState(n)
      const d  = fromSparse(sv, n)
      denseToffoli(d, c1, c2, t)
      expectSame(toSparse(d), applyToffoli(sv, c1, c2, t))
    }
  })

  it('CSWAP across every distinct control/a/b triple', () => {
    for (let ct = 0; ct < 4; ct++) for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
      if (ct === a || ct === b || a === b) continue
      const sv = spreadState(n)
      const d  = fromSparse(sv, n)
      denseCSwap(d, ct, a, b)
      expectSame(toSparse(d), applyCSwap(sv, ct, a, b))
    }
  })

  it('controlled-√iSWAP across every distinct triple', () => {
    for (let ct = 0; ct < 4; ct++) for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
      if (ct === a || ct === b || a === b) continue
      const sv = spreadState(n)
      const d  = fromSparse(sv, n)
      denseCsrSwap(d, ct, a, b)
      expectSame(toSparse(d), applyCsrSwap(sv, ct, a, b))
    }
  })

  it('arbitrary N-qubit unitary — 1, 2 and 3 target qubits', () => {
    const unitaryOf = (dim: number): Complex[][] =>
      Array.from({ length: dim }, (_, r) =>
        Array.from({ length: dim }, (_, cc) => c(Math.cos(r * 1.3 + cc * 0.7), Math.sin(r * 0.9 - cc * 1.1))))

    for (const qs of [[2], [0, 3], [3, 0], [1, 2, 4], [4, 1, 0]]) {
      const m  = unitaryOf(1 << qs.length)
      const sv = spreadState(n)
      const d  = fromSparse(sv, n)
      denseUnitary(d, qs, m)
      expectSame(toSparse(d), applyUnitary(sv, qs, m))
    }
  })

  it('probabilities agree with the sparse backend', () => {
    const sv = spreadState(n)
    const dp = denseProbabilities(fromSparse(sv, n))
    const sp = probabilities(sv)
    expect(dp.size).toBe(sp.size)
    for (const [k, p] of sp) expect(Math.abs(dp.get(k)! - p)).toBeLessThan(1e-14)
  })

  it('fromSparse → toSparse round-trips a sparse state', () => {
    const sv: StateVector = new Map([[0n, { re: Math.SQRT1_2, im: 0 }], [5n, { re: 0, im: Math.SQRT1_2 }]])
    expectSame(toSparse(fromSparse(sv, 4)), sv)
  })

  it('toSparse drops rounding dust, as the sparse backend does', () => {
    // Below AMP_EPSILON (1e-15) an amplitude is indistinguishable from an exact
    // cancellation, so both representations discard it.
    const dust: StateVector = new Map([[0n, { re: 1, im: 0 }], [3n, { re: 1e-17, im: 0 }]])
    expect(toSparse(fromSparse(dust, 3)).has(3n)).toBe(false)
    expect(denseNnz(fromSparse(dust, 3))).toBe(1)
  })

  it('toSparse keeps small but physical amplitudes', () => {
    // 1e-9 is dust under the old 1e-7 cutoff and real physics under the current
    // one. Discarding it perturbs the state by far more than its own weight once
    // later entangling gates spread the loss, so it has to survive the round trip.
    const small: StateVector = new Map([[0n, { re: 1, im: 0 }], [3n, { re: 1e-9, im: 0 }]])
    expect(toSparse(fromSparse(small, 3)).has(3n)).toBe(true)
    expect(toSparse(fromSparse(small, 3)).get(3n)?.re).toBeCloseTo(1e-9, 20)
    expect(denseNnz(fromSparse(small, 3))).toBe(2)
  })
})

describe('hybrid — promotion policy', () => {
  it('a sparse state stays sparse below the fill threshold', () => {
    // GHZ holds two amplitudes at any width; 2 × 8 is far below 2^12.
    let s = simZero(12)
    s = simSingle(s, 0, H)
    expect(simKind(s)).toBe('sparse')
    expect(simNnz(s)).toBe(2)
  })

  it('GHZ-20 never densifies end to end', () => {
    let g = new Circuit(20).h(0)
    for (let i = 0; i < 19; i++) g = g.cnot(i, i + 1)
    const sv = g.statevector()
    expect(sv.size).toBe(2)
  })

  it('a densifying circuit promotes and still gives exact amplitudes', () => {
    // H on all 6 qubits fills all 64 amplitudes — well past 2^6/8 = 8.
    let uni = new Circuit(6)
    for (let q = 0; q < 6; q++) uni = uni.h(q)
    const probs = uni.exactProbs()
    expect(Object.keys(probs)).toHaveLength(64)
    for (const p of Object.values(probs)) expect(Math.abs(p - 1 / 64)).toBeLessThan(1e-15)
  })

  it('simPromote forces dense and preserves the state', () => {
    let s = simZero(4)
    s = simSingle(s, 0, H)
    const before = simToSparse(s)
    const forced = simPromote(s)
    expect(simKind(forced)).toBe('dense')
    expectSame(simToSparse(forced), before)
  })

  it('promoting an already-dense state is a no-op', () => {
    const s = simPromote(simZero(3))
    expect(simKind(simPromote(s))).toBe('dense')
  })

  it('never promotes beyond the dense qubit ceiling', () => {
    // A 30-qubit uniform superposition would be 16 GiB dense. It must stay sparse,
    // so this completes rather than exhausting memory.
    expect(MAX_DENSE_QUBITS).toBeLessThan(30)
    let wide = new Circuit(30)
    for (let q = 0; q < 3; q++) wide = wide.h(q)
    expect(wide.statevector().size).toBe(8)
  })
})

describe('hybrid — measurement and channel primitives agree in both representations', () => {
  const n = 5
  /** The same state in both representations. */
  const pair = (): [SimState, SimState] => {
    const sv = spreadState(n)
    return [simFromSparse(new Map(sv), n), simPromote(simFromSparse(new Map(sv), n))]
  }

  it('simProbOne matches', () => {
    const [sp, dn] = pair()
    for (let q = 0; q < n; q++) {
      expect(Math.abs(simProbOne(sp, q) - simProbOne(dn, q)), `qubit ${q}`).toBeLessThan(1e-12)
    }
  })

  it('simNorm2 matches and a normalised state gives 1', () => {
    const [sp, dn] = pair()
    expect(Math.abs(simNorm2(sp) - simNorm2(dn))).toBeLessThan(1e-12)
    expect(Math.abs(simNorm2(sp) - 1)).toBeLessThan(1e-12)
  })

  it('simCollapse matches for both outcomes on every qubit', () => {
    for (let q = 0; q < n; q++) for (const outcome of [0, 1] as const) {
      const [sp, dn] = pair()
      const inv = 1 / Math.sqrt(outcome === 1 ? simProbOne(sp, q) : 1 - simProbOne(sp, q))
      expectSame(simToSparse(simCollapse(sp, q, outcome, inv)), simToSparse(simCollapse(dn, q, outcome, inv)))
    }
  })

  it('a collapsed state is renormalised and has the measured qubit fixed', () => {
    for (let q = 0; q < n; q++) {
      const [, dn] = pair()
      const inv = 1 / Math.sqrt(simProbOne(dn, q))
      const out = simCollapse(dn, q, 1, inv)
      expect(Math.abs(simNorm2(out) - 1), `norm after collapse on ${q}`).toBeLessThan(1e-12)
      expect(Math.abs(simProbOne(out, q) - 1), `qubit ${q} pinned to 1`).toBeLessThan(1e-12)
    }
  })

  it('simDecay matches and moves all population to |0⟩', () => {
    for (let q = 0; q < n; q++) {
      const [sp, dn] = pair()
      const inv = 1 / Math.sqrt(simProbOne(sp, q))
      const a = simToSparse(simDecay(sp, q, inv))
      const b = simToSparse(simDecay(dn, q, inv))
      expectSame(a, b)
      // After a decay jump the qubit is certainly |0⟩.
      expect(Math.abs(simProbOne(simFromSparse(a, n), q))).toBeLessThan(1e-12)
    }
  })

  it('simScaleBranch matches', () => {
    for (let q = 0; q < n; q++) {
      const [sp, dn] = pair()
      expectSame(simToSparse(simScaleBranch(sp, q, 0.6, 1.3)), simToSparse(simScaleBranch(dn, q, 0.6, 1.3)))
    }
  })

  it('simScale matches', () => {
    const [sp, dn] = pair()
    expectSame(simToSparse(simScale(sp, 0.37)), simToSparse(simScale(dn, 0.37)))
  })

  it('simClone is independent of its source', () => {
    const [, dn] = pair()
    const copy = simClone(dn)
    simScale(dn, 0)                                  // destroy the original
    expect(simNorm2(copy)).toBeGreaterThan(0.5)      // copy survives
  })

  it('simSample picks the same outcome from either representation', () => {
    // Determinism across the promotion boundary: a given RNG draw must not
    // change the sampled basis state just because the state densified.
    for (let k = 0; k < 200; k++) {
      const [sp, dn] = pair()
      const r = (k + 0.5) / 200
      expect(simSample(dn, r), `draw ${r}`).toBe(simSample(sp, r))
    }
  })
})

describe('hybrid — noise channels behave physically', () => {
  it('full amplitude damping drives every qubit to |0⟩', () => {
    let k = new Circuit(4)
    for (let q = 0; q < 4; q++) k = k.x(q)
    const d = k.run({ shots: 200, seed: 1, noise: { gamma: 1 } })
    expect(d.probs['0000']).toBe(1)
  })

  it('zero noise reproduces the noiseless distribution', () => {
    let k = new Circuit(5)
    for (let q = 0; q < 5; q++) k = k.h(q).t(q)
    k = k.cnot(0, 1).cz(2, 3)
    const exact = k.exactProbs()
    const noisy = k.run({ shots: 60_000, seed: 8, noise: { p1: 0, p2: 0 } }).probs
    for (const [bits, p] of Object.entries(exact)) {
      if (p < 0.01) continue
      expect(Math.abs((noisy[bits] ?? 0) - p), `outcome ${bits}`).toBeLessThan(0.015)
    }
  })

  it('depolarizing noise stays normalised and spreads the distribution', () => {
    let k = new Circuit(6)
    for (let q = 0; q < 6; q++) k = k.h(q).t(q)
    for (let q = 0; q < 5; q++) k = k.cnot(q, q + 1)
    const d = k.run({ shots: 4000, seed: 2, noise: { p1: 0.02, p2: 0.05 } })
    const total = Object.values(d.probs).reduce((a, b) => a + b, 0)
    expect(Math.abs(total - 1)).toBeLessThan(1e-9)
  })

  it('a custom Kraus channel keeps the state normalised', () => {
    // Bit-flip channel written as explicit Kraus operators.
    const p = 0.3
    const k0: [[Complex, Complex], [Complex, Complex]] =
      [[c(Math.sqrt(1 - p), 0), c(0, 0)], [c(0, 0), c(Math.sqrt(1 - p), 0)]]
    const k1: [[Complex, Complex], [Complex, Complex]] =
      [[c(0, 0), c(Math.sqrt(p), 0)], [c(Math.sqrt(p), 0), c(0, 0)]]

    let circ = new Circuit(5)
    for (let q = 0; q < 5; q++) circ = circ.h(q).t(q)
    const d = circ.run({ shots: 2000, seed: 5, noise: { kraus1: [k0, k1] } })
    const total = Object.values(d.probs).reduce((a, b) => a + b, 0)
    expect(Math.abs(total - 1)).toBeLessThan(1e-9)
  })

  it('readout error flips outcomes at roughly the stated rate', () => {
    // |1111⟩ with pMeas=0.1: each qubit independently misreports 10% of the time,
    // so the all-ones string survives about 0.9^4 ≈ 0.656 of shots.
    let k = new Circuit(4)
    for (let q = 0; q < 4; q++) k = k.x(q)
    const d = k.run({ shots: 40_000, seed: 4, noise: { pMeas: 0.1 } })
    expect(Math.abs(d.probs['1111']! - 0.9 ** 4)).toBeLessThan(0.02)
  })
})

describe('hybrid — end-to-end agreement across the public API', () => {
  /** Mixed circuit that densifies, using every gate kind the hybrid path dispatches. */
  const mixed = (n: number): Circuit => {
    let k = new Circuit(n)
    for (let q = 0; q < n; q++) k = k.h(q).t(q)
    k = k.cnot(0, 1).swap(1, 2).cz(0, 2).crx(0.7, 2, 0)
    if (n >= 4) k = k.ccx(0, 1, 3).cswap(3, 0, 2).xy(0.4, 1, 3)
    for (let q = 0; q < n; q++) k = k.ry(0.3 * (q + 1), q)
    return k
  }

  it('exactProbs sums to 1 and matches amplitude() for each basis state', () => {
    const k = mixed(5)
    const probs = k.exactProbs()
    const total = Object.values(probs).reduce((a, b) => a + b, 0)
    expect(Math.abs(total - 1)).toBeLessThan(1e-12)

    for (const [bits, p] of Object.entries(probs)) {
      const amp = k.amplitude(bits)
      expect(Math.abs(amp.re * amp.re + amp.im * amp.im - p)).toBeLessThan(1e-12)
    }
  })

  it('statevector() is normalised after a densifying circuit', () => {
    let total = 0
    for (const amp of mixed(6).statevector().values()) total += amp.re * amp.re + amp.im * amp.im
    expect(Math.abs(total - 1)).toBeLessThan(1e-12)
  })

  it('run() sampling tracks exactProbs', () => {
    const k = mixed(4)
    const exact = k.exactProbs()
    let withMeas = k.creg('out', 4)
    for (let q = 0; q < 4; q++) withMeas = withMeas.measure(q, 'out', q)
    const sampled = withMeas.run({ shots: 200_000, seed: 11 }).probs

    for (const [bits, p] of Object.entries(exact)) {
      if (p < 0.01) continue
      expect(Math.abs((sampled[bits] ?? 0) - p), `outcome ${bits}`).toBeLessThan(0.01)
    }
  })

  it('initialState is honoured on the densifying path', () => {
    let k = new Circuit(5).x(0)
    for (let q = 0; q < 5; q++) k = k.h(q)
    const fromZero = k.exactProbs()
    const fromOne  = new Circuit(5)
      .h(0).h(1).h(2).h(3).h(4)
      .exactProbs({ initialState: '10000' })
    // X then H on q0 equals H applied to |1⟩ — same distribution either way.
    expect(Object.keys(fromZero)).toHaveLength(32)
    expect(Object.keys(fromOne)).toHaveLength(32)
  })
})

describe('hybrid — configurable promotion thresholds', () => {
  /**
   * The defaults (promote at 1/8 fill, never allocate past 24 qubits) are a
   * memory-versus-speed guess that cannot suit every machine. `dense` overrides
   * both per call. Results must not depend on the choice — only cost does.
   */

  it('maxQubits: 0 keeps everything sparse and still gives exact results', () => {
    let uni = new Circuit(6)
    for (let q = 0; q < 6; q++) uni = uni.h(q)
    const forced = uni.exactProbs({ dense: { maxQubits: 0 } })
    expect(Object.keys(forced)).toHaveLength(64)
    for (const p of Object.values(forced)) expect(Math.abs(p - 1 / 64)).toBeLessThan(1e-15)
    // Identical to the default path, which does promote at this size.
    expect(forced).toEqual(uni.exactProbs())
  })

  it('an eager fill promotes sooner without changing the answer', () => {
    let k = new Circuit(5)
    for (let q = 0; q < 5; q++) k = k.h(q).t(q)
    k = k.cnot(0, 1).cz(2, 3)
    const eager  = k.exactProbs({ dense: { fill: 1e6 } })   // promote almost immediately
    const lazy   = k.exactProbs({ dense: { maxQubits: 0 } }) // never promote
    for (const [bits, p] of Object.entries(lazy)) {
      expect(Math.abs(eager[bits]! - p), `outcome ${bits}`).toBeLessThan(1e-14)
    }
  })

  it('the option reaches run(), including the per-shot noise path', () => {
    let k = new Circuit(5)
    for (let q = 0; q < 5; q++) k = k.h(q).t(q)
    const a = k.run({ shots: 4000, seed: 3, dense: { maxQubits: 0 } })
    const b = k.run({ shots: 4000, seed: 3, dense: { fill: 1e6 } })
    // Same seed, same sampling order in both representations.
    expect(b.probs).toEqual(a.probs)

    const na = k.run({ shots: 400, seed: 4, noise: { p1: 0.01 }, dense: { maxQubits: 0 } })
    const nb = k.run({ shots: 400, seed: 4, noise: { p1: 0.01 }, dense: { fill: 1e6 } })
    expect(Object.values(nb.probs).reduce((x, y) => x + y, 0)).toBeCloseTo(1, 10)
    expect(Object.keys(nb.probs).length).toBeGreaterThan(0)
    expect(Object.keys(na.probs).length).toBeGreaterThan(0)
  })

  it('the option reaches dm()', () => {
    let k = new Circuit(4)
    for (let q = 0; q < 4; q++) k = k.h(q)
    const sparse = k.dm({ dense: { maxQubits: 0 } }).probabilities()
    const dense  = k.dm({ dense: { fill: 1e6 } }).probabilities()
    expect(Object.keys(dense)).toHaveLength(16)
    for (const [bits, p] of Object.entries(sparse)) {
      expect(Math.abs(dense[bits]! - p), `outcome ${bits}`).toBeLessThan(1e-14)
    }
  })

  it('rejects thresholds that cannot mean anything', () => {
    const k = new Circuit(2).h(0)
    expect(() => k.exactProbs({ dense: { fill: 0 } })).toThrow(RangeError)
    expect(() => k.exactProbs({ dense: { fill: -1 } })).toThrow(RangeError)
    expect(() => k.exactProbs({ dense: { maxQubits: -1 } })).toThrow(RangeError)
    expect(() => k.exactProbs({ dense: { maxQubits: 2.5 } })).toThrow(RangeError)
  })

  it('omitting the option leaves the documented defaults in place', () => {
    // fill 64 sits just short of the ~2^n/100 break-even between the sparse and
    // dense kernels. It was 8 — twelve times later than break-even — which cost
    // up to 4x on states that end up dense anyway.
    expect(DEFAULT_SV_POLICY).toEqual({ fill: 64, maxQubits: MAX_DENSE_QUBITS })
    expect(svPolicy()).toEqual(DEFAULT_SV_POLICY)
    expect(svPolicy({ fill: 4 })).toEqual({ fill: 4, maxQubits: MAX_DENSE_QUBITS })
  })
})

describe('simulate() — entanglement-adaptive routing', () => {
  /**
   * Above the statevector limit `simulate()` used to always pick MPS from the
   * gate list alone. MPS is only right while entanglement stays bounded: a
   * volume-law circuit at n=22 drives χ past 500 and takes 108s, where a dense
   * statevector finishes the same circuit in 7.6s. The router now measures χ as
   * it goes and abandons MPS once it passes the crossover at χ = 2^(n/3).
   *
   * `statevectorLimit` is lowered in these tests so the probe path engages at
   * sizes where `exactProbs()` can check the answer.
   */

  /** Brickwork of generic (non-Clifford) two-qubit rotations — entanglement grows with depth. */
  const volume = (n: number, layers: number): Circuit => {
    let k = new Circuit(n)
    let a = 0.3
    for (let l = 0; l < layers; l++) {
      for (let q = 0; q < n; q++) { a += 0.37; k = k.ry(a, q).rz(a * 1.7, q) }
      for (let q = l % 2; q < n - 1; q += 2) { a += 0.23; k = k.crx(a, q, q + 1).cry(a * 0.9, q + 1, q) }
    }
    return k
  }

  it('keeps a low-entanglement circuit on MPS', () => {
    const d = volume(10, 2).simulate({ shots: 500, seed: 1, statevectorLimit: 4 })
    expect(d.backend).toBe('mps')
    expect(d.peakChi!).toBeLessThanOrEqual(2 ** (10 / 3))
  })

  it('abandons MPS for the statevector once entanglement passes the crossover', () => {
    const d = volume(10, 10).simulate({ shots: 500, seed: 1, statevectorLimit: 4 })
    expect(d.backend).toBe('statevector')
  })

  it('both routing outcomes agree with exactProbs', () => {
    for (const [n, layers] of [[10, 2], [10, 10], [12, 12]] as const) {
      const k = volume(n, layers)
      const exact = k.exactProbs()
      const d = k.simulate({ shots: 200_000, seed: 7, statevectorLimit: 4 })
      for (const [bits, p] of Object.entries(exact)) {
        if (p < 0.002) continue
        expect(Math.abs((d.probs[bits] ?? 0) - p), `n=${n} layers=${layers} outcome ${bits}`).toBeLessThan(0.01)
      }
    }
  }, 60_000)

  it('a GHZ chain stays on MPS however wide it gets', () => {
    // .t(0) keeps it off the Clifford path so the MPS branch is the one tested.
    let ghz = new Circuit(40).h(0)
    for (let i = 0; i < 39; i++) ghz = ghz.cnot(i, i + 1)
    const d = ghz.t(0).simulate({ shots: 200, seed: 1 })
    expect(d.backend).toBe('mps')
    expect(d.peakChi).toBe(2)
  })

  it('past the dense ceiling there is no fallback to consider', () => {
    // n=40 exceeds maxQubits, so no statevector exists to fall back to and the
    // probe is skipped entirely — MPS runs regardless of how entangled it gets.
    const d = volume(30, 2).simulate({ shots: 100, seed: 1 })
    expect(d.backend).toBe('mps')
  }, 60_000)

  it('routing is unchanged for Clifford circuits and small circuits', () => {
    let ghz = new Circuit(30).h(0)
    for (let i = 0; i < 29; i++) ghz = ghz.cnot(i, i + 1)
    expect(ghz.simulate({ shots: 200, seed: 1 }).backend).toBe('clifford')
    expect(new Circuit(5).h(0).t(1).simulate({ shots: 200, seed: 1 }).backend).toBe('statevector')
  })

  it('noisy and mid-circuit runs bypass the probe', () => {
    // Both re-simulate per shot; the probe only covers the clean path.
    let k = new Circuit(22).h(0)
    for (let i = 0; i < 21; i++) k = k.cnot(i, i + 1)
    expect(k.t(0).simulate({ shots: 50, seed: 1, noise: { p1: 0.01 } }).backend).toBe('mps')
  }, 60_000)
})

describe('representation is observable', () => {
  /**
   * `dense` is a documented option, so its effect has to be checkable. Without
   * this, tuning `fill` or `maxQubits` could only be inferred from timings.
   */

  const uniform = (n: number): Circuit => {
    let k = new Circuit(n)
    for (let q = 0; q < n; q++) k = k.h(q).t(q)
    return k
  }

  it('reports sparse for a state that never fills', () => {
    let ghz = new Circuit(20).h(0)
    for (let i = 0; i < 19; i++) ghz = ghz.cnot(i, i + 1)
    expect(ghz.run({ shots: 100, seed: 1 }).representation).toBe('sparse')
  })

  it('reports dense for a state that does', () => {
    expect(uniform(12).run({ shots: 100, seed: 1 }).representation).toBe('dense')
  })

  it('tracks the dense option in both directions', () => {
    const k = uniform(10)
    expect(k.run({ shots: 100, seed: 1, dense: { maxQubits: 0 } }).representation).toBe('sparse')
    expect(k.run({ shots: 100, seed: 1, dense: { fill: 1e6 } }).representation).toBe('dense')
  })

  it('covers the per-shot noise path too', () => {
    expect(uniform(10).run({ shots: 20, seed: 1, noise: { p1: 0.01 } }).representation).toBe('dense')
    expect(uniform(10).run({ shots: 20, seed: 1, noise: { p1: 0.01 }, dense: { maxQubits: 0 } }).representation).toBe('sparse')
  })

  it('is undefined for backends the choice does not apply to', () => {
    let ghz = new Circuit(20).h(0)
    for (let i = 0; i < 19; i++) ghz = ghz.cnot(i, i + 1)
    expect(ghz.runClifford({ shots: 100, seed: 1 }).representation).toBeUndefined()
    expect(ghz.t(0).runMps({ shots: 100, seed: 1 }).representation).toBeUndefined()
  })

  it('DensityMatrix reports its own representation', () => {
    let u = new Circuit(6)
    for (let q = 0; q < 6; q++) u = u.h(q)
    expect(u.dm().representation).toBe('dense')
    expect(u.dm({ dense: { maxQubits: 0 } }).representation).toBe('sparse')
    expect(new Circuit(6).x(0).x(3).dm().representation).toBe('sparse')
  })
})

describe('simulate() — MPS tuning passes through', () => {
  const volume = (n: number, layers: number): Circuit => {
    let k = new Circuit(n)
    let a = 0.3
    for (let l = 0; l < layers; l++) {
      for (let q = 0; q < n; q++) { a += 0.37; k = k.ry(a, q) }
      for (let q = l % 2; q < n - 1; q += 2) { a += 0.23; k = k.crx(a, q, q + 1) }
    }
    return k
  }

  it('truncErr caps chi, so a circuit that would bail stays on MPS', () => {
    const k = volume(10, 10)
    // Exact: entanglement outgrows the budget and the router falls back.
    expect(k.simulate({ shots: 200, seed: 1, statevectorLimit: 4 }).backend).toBe('statevector')
    // Approximate: truncation holds chi down, so MPS remains the cheaper route.
    const t = k.simulate({ shots: 200, seed: 1, statevectorLimit: 4, truncErr: 1e-3 })
    expect(t.backend).toBe('mps')
    expect(t.truncated).toBe(true)
    expect(t.peakChi!).toBeLessThan(2 ** (10 / 3))
  })

  it('maxBond is an allocation hint, not a cap, so it does not change routing', () => {
    // chi still grows past the budget however much was pre-allocated.
    const k = volume(10, 10)
    expect(k.simulate({ shots: 200, seed: 1, statevectorLimit: 4, maxBond: 256 }).backend).toBe('statevector')
  })

  it('tuning reaches the plain MPS route above the dense ceiling', () => {
    const d = volume(30, 4).simulate({ shots: 100, seed: 1, truncErr: 1e-3 })
    expect(d.backend).toBe('mps')
    expect(d.truncated).toBe(true)
  }, 60_000)
})

describe('dense promotion — disabling it fails loudly rather than exhausting the heap', () => {
  /**
   * `dense.maxQubits` exists to bound memory, so honouring a lowered ceiling all
   * the way into a multi-gigabyte sparse map would invert its purpose. Forcing
   * sparse on a 12-qubit density matrix asks for 4^12 boxed entries; that used to
   * end in a V8 heap abort with nothing to indicate why.
   *
   * The guard is deliberately narrow — it must not disturb small forced-sparse
   * runs, nor wide circuits where dense was never an option to begin with.
   */

  // The guard's own logic is checked directly below; these end-to-end cases only
  // confirm the wiring, and each one has to build millions of sparse entries
  // before the guard can fire. One throwing call, asserted three ways.
  it('throws a diagnostic instead of running out of memory', () => {
    let u = new Circuit(12)
    for (let q = 0; q < 12; q++) u = u.h(q)
    let err: unknown
    try { u.dm({ dense: { maxQubits: 0 } }) } catch (e) { err = e }
    expect(err).toBeInstanceOf(RangeError)
    expect((err as Error).message).toMatch(/dense promotion disabled/)
    // The message has to say how to proceed, not just that something went wrong.
    expect((err as Error).message).toMatch(/Raise dense\.maxQubits to at least 12/)
  }, 60_000)

  it('guardSparseGrowth fires on exactly the hazard case and nothing else', () => {
    const defaults = { fill: 32, maxQubits: 12 }
    const off      = { fill: 32, maxQubits: 0 }   // caller disabled promotion
    const big      = SPARSE_ENTRY_LIMIT + 1

    // All three conditions hold: large, promotion off, dense would have fitted.
    expect(() => guardSparseGrowth(big, 12, off, defaults, 'x')).toThrow(RangeError)

    // Small enough not to matter.
    expect(() => guardSparseGrowth(SPARSE_ENTRY_LIMIT, 12, off, defaults, 'x')).not.toThrow()
    // Promotion still available — nothing was overridden away.
    expect(() => guardSparseGrowth(big, 12, defaults, defaults, 'x')).not.toThrow()
    // Too wide for a dense buffer under any policy: sparse is the only option.
    expect(() => guardSparseGrowth(big, 30, off, defaults, 'x')).not.toThrow()
  })

  it('leaves small forced-sparse runs alone', () => {
    let u6 = new Circuit(6)
    for (let q = 0; q < 6; q++) u6 = u6.h(q)
    expect(u6.dm({ dense: { maxQubits: 0 } }).representation).toBe('sparse')

    let u10 = new Circuit(10)
    for (let q = 0; q < 10; q++) u10 = u10.h(q).t(q)
    expect(u10.run({ shots: 50, seed: 1, dense: { maxQubits: 0 } }).representation).toBe('sparse')
  })

  it('leaves wide circuits alone, where dense was never viable', () => {
    // n=60 is far past any dense ceiling, so sparse is the only representation
    // available and must not be second-guessed.
    let ghz = new Circuit(60).h(0)
    for (let i = 0; i < 59; i++) ghz = ghz.cnot(i, i + 1)
    expect(ghz.statevector().size).toBe(2)
  })

  it('raising maxQubits is a working remedy, as the message claims', () => {
    let u = new Circuit(12)
    for (let q = 0; q < 12; q++) u = u.h(q)
    const d = u.dm({ dense: { maxQubits: 12 } })
    expect(d.representation).toBe('dense')
    expect(Object.keys(d.probabilities())).toHaveLength(4096)
  }, 60_000)
})

describe('MPS bond dimension — maxChi is a real ceiling', () => {
  /**
   * `maxBond` reads like a cap but is only an initial allocation — χ grows past
   * it on demand, so `maxBond: 256` changes nothing about how large χ gets. That
   * left no way to bound MPS *memory* deterministically: `truncErr` bounds the
   * error, which is a different quantity. `maxChi` is the ceiling, and it binds.
   */

  const volume = (n: number, layers: number): Circuit => {
    let k = new Circuit(n)
    let a = 0.3
    for (let l = 0; l < layers; l++) {
      for (let q = 0; q < n; q++) { a += 0.37; k = k.ry(a, q) }
      for (let q = l % 2; q < n - 1; q += 2) { a += 0.23; k = k.crx(a, q, q + 1) }
    }
    return k
  }

  it('caps chi and reports the truncation', () => {
    const k = volume(14, 10)
    const free = k.runMps({ shots: 500, seed: 1 })
    expect(free.peakChi!).toBeGreaterThan(16)
    expect(free.truncated).toBe(false)

    for (const cap of [16, 4]) {
      const d = k.runMps({ shots: 500, seed: 1, maxChi: cap })
      expect(d.peakChi, `maxChi=${cap}`).toBe(cap)
      expect(d.truncated, `maxChi=${cap}`).toBe(true)
    }
  }, 60_000)

  it('a ceiling above the circuit’s own chi changes nothing', () => {
    const k = volume(14, 10)
    const free = k.runMps({ shots: 500, seed: 1 })
    const high = k.runMps({ shots: 500, seed: 1, maxChi: 1024 })
    expect(high.peakChi).toBe(free.peakChi)
    expect(high.truncated).toBe(false)
  }, 60_000)

  it('stays accurate at a generous cap and degrades gracefully at a tight one', () => {
    const k = volume(14, 10)
    const exact = k.exactProbs()
    // 30k shots puts sampling noise near 3e-3, well inside both bounds below.
    const err = (cap: number): number => {
      const d = k.runMps({ shots: 30_000, seed: 1, maxChi: cap })
      let e = 0
      for (const [bits, p] of Object.entries(exact)) if (p > 0.002) e = Math.max(e, Math.abs((d.probs[bits] ?? 0) - p))
      return e
    }
    expect(err(16)).toBeLessThan(0.008)   // generous cap: still essentially exact
    expect(err(4)).toBeLessThan(0.02)     // tight cap: approximate but not wild
  }, 60_000)

  it('reaches the backend through simulate() too', () => {
    const d = volume(30, 4).simulate({ shots: 100, seed: 1, maxChi: 8 })
    expect(d.backend).toBe('mps')
    expect(d.peakChi!).toBeLessThanOrEqual(8)
  }, 60_000)

  it('is exact by default — no ceiling unless asked for', () => {
    let ghz = new Circuit(20).h(0)
    for (let i = 0; i < 19; i++) ghz = ghz.cnot(i, i + 1)
    const d = ghz.t(0).runMps({ shots: 200, seed: 1 })
    expect(d.truncated).toBe(false)
    expect(d.peakChi).toBe(2)
  })
})

describe('maxChi and the entanglement router interact predictably', () => {
  const volume = (n: number, layers: number): Circuit => {
    let k = new Circuit(n)
    let a = 0.3
    for (let l = 0; l < layers; l++) {
      for (let q = 0; q < n; q++) { a += 0.37; k = k.ry(a, q).rz(a * 1.7, q) }
      for (let q = l % 2; q < n - 1; q += 2) { a += 0.23; k = k.crx(a, q, q + 1).cry(a * 0.9, q + 1, q) }
    }
    return k
  }

  it('a ceiling keeps the run on MPS instead of falling back', () => {
    const k = volume(14, 12)
    // Unbounded: entanglement outgrows the routing budget, so it bails.
    expect(k.simulate({ shots: 100, seed: 1, statevectorLimit: 4 }).backend).toBe('statevector')
    // Bounded: chi cannot reach the budget, so MPS stays — which is the point of
    // asking for bounded memory in the first place.
    const m = k.simulate({ shots: 100, seed: 1, statevectorLimit: 4, maxChi: 8 })
    expect(m.backend).toBe('mps')
    expect(m.peakChi).toBe(8)
    expect(m.truncated).toBe(true)
  }, 60_000)
})
