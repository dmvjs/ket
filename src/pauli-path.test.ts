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
