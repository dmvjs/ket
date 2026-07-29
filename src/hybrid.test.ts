import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { c, type Complex } from './complex.js'
import { H, Rx, Ry, Rz, T, U3, X, Xy, Y, Z } from './gates.js'
import {
  denseCNOT, denseControlled, denseCsrSwap, denseCSwap, denseNnz, denseProbabilities,
  denseSingle, denseSWAP, denseToffoli, denseTwo, denseUnitary, fromSparse,
  MAX_DENSE_QUBITS, toSparse,
} from './dense.js'
import {
  applyCNOT, applyControlled, applyCsrSwap, applyCSwap, applySingle, applySWAP,
  applyToffoli, applyTwo, applyUnitary, probabilities, type Gate4x4, type StateVector,
} from './statevector.js'
import {
  simClone, simCollapse, simDecay, simFromSparse, simKind, simNnz, simNorm2,
  simProbOne, simPromote, simSample, simScale, simScaleBranch, simSingle,
  simToSparse, simZero, type SimState,
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

  it('toSparse drops negligible amplitudes, as the sparse backend does', () => {
    const sv: StateVector = new Map([[0n, { re: 1, im: 0 }], [3n, { re: 1e-9, im: 0 }]])
    // 1e-9 squared is 1e-18, below the 1e-14 support threshold.
    expect(toSparse(fromSparse(sv, 3)).has(3n)).toBe(false)
    expect(denseNnz(fromSparse(sv, 3))).toBe(1)
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
