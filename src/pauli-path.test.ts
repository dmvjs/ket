import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { vqe } from './algorithms.js'
import type { PauliTerm } from './algorithms.js'
import { pauliPathExpectation } from './pauli-path.js'
import { makePrng } from './prng.js'

/**
 * Pauli-path propagation against the statevector oracle.
 *
 * `vqe()` computes the same quantity by building the state and contracting, so
 * the two share nothing but the answer. Every sign convention in the conjugation
 * tables is exercised by construction: a wrong sign anywhere shows up as a
 * mismatch on some random circuit.
 */

const LETTERS = ['I', 'X', 'Y', 'Z'] as const

function randomObservable(n: number, rand: () => number, terms = 3): PauliTerm[] {
  return Array.from({ length: terms }, () => ({
    coeff: Math.round((rand() * 4 - 2) * 100) / 100,
    ops: Array.from({ length: n }, () => LETTERS[Math.floor(rand() * 4)]!).join(''),
  }))
}

/** Clifford-only circuit: exercises every table entry without branching. */
function randomClifford(n: number, depth: number, rand: () => number): Circuit {
  let c = new Circuit(n)
  const one = ['h', 's', 'sdg', 'x', 'y', 'z'] as const
  for (let d = 0; d < depth; d++) {
    const r = rand()
    if (n > 1 && r < 0.35) {
      const a = Math.floor(rand() * n)
      let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
      c = rand() < 0.5 ? c.cnot(a, b) : c.cz(a, b)
    } else if (n > 1 && r < 0.45) {
      const a = Math.floor(rand() * n)
      let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
      c = c.swap(a, b)
    } else {
      c = c[one[Math.floor(rand() * one.length)]!](Math.floor(rand() * n))
    }
  }
  return c
}

/** Clifford plus rotations: exercises the branching path. */
function randomRotated(n: number, depth: number, rand: () => number): Circuit {
  let c = randomClifford(n, depth, rand)
  for (let i = 0; i < depth; i++) {
    const q = Math.floor(rand() * n)
    const theta = (rand() * 2 - 1) * Math.PI
    const pick = rand()
    c = pick < 0.34 ? c.rz(theta, q) : pick < 0.67 ? c.rx(theta, q) : c.ry(theta, q)
    if (rand() < 0.3) c = c.t(Math.floor(rand() * n))
  }
  return c
}

describe('pauli-path — differential vs the statevector oracle', () => {
  for (const n of [1, 2, 3, 4]) {
    it(`Clifford circuits agree exactly, n=${n}`, () => {
      for (let seed = 0; seed < 30; seed++) {
        const rand = makePrng(seed * 7919 + n * 131 + 3)
        const circuit = randomClifford(n, 4 * n + 6, rand)
        const obs = randomObservable(n, rand)
        const got = pauliPathExpectation(circuit, obs).value
        expect(got, `n=${n} seed=${seed}`).toBeCloseTo(vqe(circuit, obs), 10)
      }
    })
  }

  for (const n of [1, 2, 3, 4]) {
    it(`Clifford + rotations agree exactly, n=${n}`, () => {
      for (let seed = 0; seed < 30; seed++) {
        const rand = makePrng(seed * 104729 + n * 17 + 11)
        const circuit = randomRotated(n, 3 * n + 4, rand)
        const obs = randomObservable(n, rand)
        const got = pauliPathExpectation(circuit, obs).value
        expect(got, `n=${n} seed=${seed}`).toBeCloseTo(vqe(circuit, obs), 9)
      }
    })
  }

  it('matches on a Heisenberg chain, the VQE case it exists for', () => {
    const n = 6
    const rand = makePrng(4242)
    let ansatz = new Circuit(n)
    for (let i = 0; i < n; i++) ansatz = ansatz.ry(0.3 + 0.1 * i, i)
    for (let i = 0; i + 1 < n; i++) ansatz = ansatz.cnot(i, i + 1).rz(0.2 * (i + 1), i + 1)
    void rand

    const h: PauliTerm[] = []
    for (let i = 0; i + 1 < n; i++) {
      const at = (a: string, j: number): string =>
        Array.from({ length: n }, (_, k) => (k === n - 1 - j || k === n - 2 - j ? a : 'I')).join('')
      h.push({ coeff: 1, ops: at('Z', i) }, { coeff: 1, ops: at('X', i) }, { coeff: 1, ops: at('Y', i) })
    }
    expect(pauliPathExpectation(ansatz, h).value).toBeCloseTo(vqe(ansatz, h), 9)
  })
})

describe('pauli-path — cost model and truncation', () => {
  it('a Clifford circuit never branches, at any width', () => {
    // One Pauli in, one Pauli out, per gate — width is close to free.
    const n = 400
    let c = new Circuit(n).h(0)
    for (let i = 0; i + 1 < n; i++) c = c.cnot(i, i + 1)
    const obs: PauliTerm[] = [{ coeff: 1, ops: 'Z'.repeat(n) }]
    const r = pauliPathExpectation(c, obs)
    expect(r.peakTerms).toBe(1)
    expect(r.truncated).toBe(false)
    expect(Number.isFinite(r.value)).toBe(true)
  })

  it('reports the weight it discarded, and the error respects that bound', () => {
    const rand = makePrng(77)
    const n = 4
    const circuit = randomRotated(n, 10, rand)
    const obs = randomObservable(n, rand)

    const exact = pauliPathExpectation(circuit, obs).value
    const cut = pauliPathExpectation(circuit, obs, { maxTerms: 4 })

    expect(cut.truncated).toBe(true)
    expect(cut.droppedWeight).toBeGreaterThan(0)
    // droppedWeight is an upper bound on how far the answer can have moved.
    expect(Math.abs(cut.value - exact)).toBeLessThanOrEqual(cut.droppedWeight + 1e-9)
  })

  it('a tighter budget never reports less dropped weight', () => {
    const rand = makePrng(915)
    const circuit = randomRotated(4, 12, rand)
    const obs = randomObservable(4, rand)
    const loose = pauliPathExpectation(circuit, obs, { maxTerms: 32 })
    const tight = pauliPathExpectation(circuit, obs, { maxTerms: 4 })
    expect(tight.droppedWeight).toBeGreaterThanOrEqual(loose.droppedWeight)
  })

  it('declines gates it cannot propagate, naming them', () => {
    const c = new Circuit(2).h(0).u3(0.1, 0.2, 0.3, 1)
    expect(() => pauliPathExpectation(c, [{ coeff: 1, ops: 'ZZ' }]))
      .toThrow(/does not support gate 'u3'/)
  })

  it('declines an impure circuit', () => {
    const c = new Circuit(2).h(0).creg('m', 1).measure(0, 'm', 0)
    expect(() => pauliPathExpectation(c, [{ coeff: 1, ops: 'ZZ' }]))
      .toThrow(/requires a pure circuit/)
  })

  it('rejects an observable of the wrong width', () => {
    expect(() => pauliPathExpectation(new Circuit(3).h(0), [{ coeff: 1, ops: 'ZZ' }]))
      .toThrow(/one letter per qubit/)
  })
})

/** Tr(ρP) from an exact density matrix — ground truth for the noisy case. */
function dmExpectation(circuit: Circuit, term: PauliTerm, noise: { p1?: number; p2?: number }): number {
  const n = circuit.qubits
  const rho = circuit.dm({ noise })
  // P|j> = phase * |i>, so Tr(ρP) = sum_i rho[i][j(i)] * P[j(i)][i].
  let re = 0
  for (let i = 0; i < 2 ** n; i++) {
    let j = i, pre = 1, pim = 0
    for (let q = 0; q < n; q++) {
      const letter = term.ops[n - 1 - q]!
      const bit = (i >> q) & 1
      if (letter === 'X') { j ^= 1 << q }
      else if (letter === 'Y') { j ^= 1 << q; const s = bit === 0 ? 1 : -1; const nr = -pim * s, ni = pre * s; pre = nr; pim = ni }
      else if (letter === 'Z') { if (bit === 1) { pre = -pre; pim = -pim } }
    }
    // Tr(ρP) = Σ_ij ρ[i,j]·P[j,i]; ρ is Hermitian, so transposing these silently
    // conjugates the result — invisible for real entries, wrong wherever a Y
    // meets a complex amplitude.
    const z = rho.get(BigInt(i), BigInt(j))
    re += z.re * pre - z.im * pim
  }
  return re * term.coeff
}

describe('pauli-path — noise', () => {
  const noise = { p1: 0.01, p2: 0.02 }

  for (const n of [2, 3, 4]) {
    it(`depolarizing damping matches the density matrix, n=${n}`, () => {
      for (let seed = 0; seed < 12; seed++) {
        const rand = makePrng(seed * 5591 + n * 37 + 2)
        const circuit = randomRotated(n, 2 * n + 3, rand)
        const [term] = randomObservable(n, rand, 1)
        const got = pauliPathExpectation(circuit, [term!], { noise }).value
        expect(got, `n=${n} seed=${seed}`).toBeCloseTo(dmExpectation(circuit, term!, noise), 8)
      }
    })
  }

  it('noise shrinks coefficients, so truncation costs less', () => {
    const rand = makePrng(31)
    const circuit = randomRotated(6, 30, rand)
    const obs = randomObservable(6, rand, 2)
    const clean = pauliPathExpectation(circuit, obs, { threshold: 1e-6 })
    const noisy = pauliPathExpectation(circuit, obs, { threshold: 1e-6, noise: { p1: 0.02, p2: 0.04 } })
    expect(noisy.peakTerms).toBeLessThanOrEqual(clean.peakTerms)
  })

  it('weight truncation drops high-weight terms and reports the cost', () => {
    const rand = makePrng(808)
    const circuit = randomRotated(6, 24, rand)
    const obs = randomObservable(6, rand, 2)
    const full = pauliPathExpectation(circuit, obs)
    const capped = pauliPathExpectation(circuit, obs, { maxWeight: 2 })
    expect(capped.peakTerms).toBeLessThan(full.peakTerms)
    expect(capped.truncated).toBe(true)
    expect(Math.abs(capped.value - full.value)).toBeLessThanOrEqual(capped.droppedWeight + 1e-9)
  })
})

describe('pauli-path — gates that are rotations under another name', () => {
  /**
   * `rotationOf` matched on gate names, and eight gates that *are* Pauli-axis
   * rotations were spelled differently: `si` (this library's `sdg`), `v`/`srn`/`sx`
   * (Rx(π/2)), their inverses, and `r2`/`r4`/`r8` (defined in gates.ts as
   * Rz(π/2), Rz(π/4), Rz(π/8)). Each one disqualified the whole attack from any
   * circuit containing it — and this is the attack that reproduced the published
   * 127-qubit results, so the hole mattered.
   *
   * The same name-matching failure once hid `Rz(−π/2)` being Clifford. Accepting a
   * gate is not the point; conjugating it correctly is, so this checks values.
   */
  const exactZ = (c: Circuit, q: number): number => {
    let z = 0
    for (const [bits, p] of Object.entries(c.exactProbs())) z += (bits[q] === '1' ? -1 : 1) * p
    return z
  }

  const CASES: [string, (c: Circuit) => Circuit][] = [
    ['si', c => c.si(1)], ['v', c => c.v(1)], ['vi', c => c.vi(1)],
    ['srn', c => c.srn(1)], ['srndg', c => c.srndg(1)],
    ['r2', c => c.r2(1)], ['r4', c => c.r4(1)], ['r8', c => c.r8(1)],
    // Already worked; kept so a regression in the shared path is visible here too.
    ['t', c => c.t(1)], ['s', c => c.s(1)], ['h', c => c.h(1)],
    ['rx', c => c.rx(0.7, 1)], ['ry', c => c.ry(0.7, 1)], ['rz', c => c.rz(0.7, 1)],
  ]

  /**
   * Qubit 1 has to start in a *tilted* state, or the test proves nothing.
   *
   * The obvious context — `h(0).cnot(0,1)` — leaves qubit 1 maximally mixed, and
   * then ⟨Z₁⟩ is 0 for every unitary on it. That version of this test passed with
   * all fourteen values at exactly zero, which would have accepted any conjugation
   * rule at all. The trailing CNOT commutes with Z₁, so entanglement is exercised
   * without flattening the observable.
   */
  const context = (apply: (c: Circuit) => Circuit): Circuit =>
    apply(new Circuit(3).h(0).ry(0.9, 1).rx(0.35, 1).t(2)).cnot(1, 2).rz(0.4, 0)

  it.each(CASES)('%s conjugates to the exact expectation value', (_name, apply) => {
    const c = context(apply)
    const got = pauliPathExpectation(c, [{ coeff: 1, ops: 'IZI' }], { maxTerms: 8192, threshold: 1e-15 })
    expect(got.value).toBeCloseTo(exactZ(c, 1), 9)
  })

  it('is not measuring a constant', () => {
    // Guards the trap above: if the context ever flattens ⟨Z₁⟩ again, this fails
    // rather than the suite going quietly green on a vacuous check.
    const values = new Set(CASES.map(([, apply]) => exactZ(context(apply), 1).toFixed(6)))
    expect(values.size).toBeGreaterThan(3)
  })
})

describe('pauli-path — two-qubit Pauli rotations', () => {
  /**
   * The native case for Pauli propagation, and it was missing entirely: every
   * `two` op was refused, so the attack declined any circuit built from Ising
   * interactions. That is what Trotterised Hamiltonian simulation *is*, and what
   * the published 127-qubit circuits are made of — so the one method that actually
   * reproduced those results could not be pointed at them.
   *
   * `Xx/Yy/Zz` are exp(-iθ/2 · P⊗P) directly. `Xy` is a rotation about (XX+YY),
   * whose halves commute, so it splits — checked against the gate matrix rather
   * than argued: `Xy(θ) = Xx(-θ/2)·Yy(-θ/2)` agrees to 1e-16, `iSWAP = Xy(π)`.
   */
  const N = 3
  /** `ops` runs one letter per qubit, reversed: `ops[n-1-q]` is qubit q. */
  const zAt = (q: number): PauliTerm[] =>
    [{ coeff: 1, ops: Array.from({ length: N }, (_, k) => (k === N - 1 - q ? 'Z' : 'I')).join('') }]
  const exactZ = (c: Circuit, q: number): number => {
    let z = 0
    for (const [bits, p] of Object.entries(c.exactProbs())) z += (bits[q] === '1' ? -1 : 1) * p
    return z
  }

  const CASES: [string, (c: Circuit) => Circuit][] = [
    ['xx', c => c.xx(0.6, 0, 1)], ['yy', c => c.yy(0.6, 0, 1)], ['zz', c => c.zz(0.6, 0, 1)],
    ['xy', c => c.xy(0.6, 0, 1)], ['iswap', c => c.iswap(0, 1)],
    ['xx at π/2', c => c.xx(Math.PI / 2, 0, 1)], ['zz at π', c => c.zz(Math.PI, 0, 1)],
    // Reversed and non-adjacent operands: the axis is per-operand, not positional.
    ['xx reversed', c => c.xx(0.6, 1, 0)], ['zz non-adjacent', c => c.zz(0.6, 0, 2)],
    ['yy on 1,2', c => c.yy(0.4, 1, 2)],
  ]

  it.each(CASES)('%s conjugates to the exact expectation on every qubit', (_name, apply) => {
    const c = apply(new Circuit(N).h(0).ry(0.9, 1).rx(0.35, 1).t(2))
      .cnot(1, 2).rz(0.4, 0).ry(0.2, 2)
    for (let q = 0; q < N; q++) {
      const got = pauliPathExpectation(c, zAt(q), { maxTerms: 16384, threshold: 1e-15 })
      expect(got.value, `⟨Z${q}⟩`).toBeCloseTo(exactZ(c, q), 9)
    }
  })

  it('refuses Ms rather than approximating it', () => {
    // Ms mixes XX, YY, XY and YX, which do not all commute, so it is not a product
    // of Pauli rotations. Only Ms(0,0) and Ms(π/2,π/2) are — and special-casing a
    // gate by its parameters is how the name-matching bugs here began.
    expect(() => pauliPathExpectation(new Circuit(N).ms(0.3, 0.4, 0, 1), zAt(0), { maxTerms: 64 }))
      .toThrow(/does not support/)
  })
})
