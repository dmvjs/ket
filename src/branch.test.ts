import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { shorCircuit, shorBeauregard, shorCircuitIterative, modPow } from './beauregard.js'

/** Max |Δ amplitude| between the branch expansion and the sparse statevector. */
function maxDiff(c: Circuit): { diff: number; peak: number } {
  const s = c.branchState()
  const sv = c.statevector()
  const amps = s.amplitudes()
  let diff = 0
  for (const k of new Set([...sv.keys(), ...amps.keys()])) {
    const a = sv.get(k) ?? { re: 0, im: 0 }, b = amps.get(k) ?? { re: 0, im: 0 }
    diff = Math.max(diff, Math.hypot(a.re - b.re, a.im - b.im))
  }
  return { diff, peak: s.peakBranches }
}

/** Seeded random circuit over the full op menu the backend dispatches. */
function randomCircuit(n: number, gates: number, seed: number): Circuit {
  let s = seed >>> 0 || 1
  const r = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 0x100000000 }
  const q = () => Math.floor(r() * n)
  const distinct = (k: number) => { const out = new Set<number>(); while (out.size < k) out.add(q()); return [...out] }
  let c = new Circuit(n)
  for (let i = 0; i < gates; i++) {
    const [a, b, d] = distinct(3) as [number, number, number]
    switch (Math.floor(r() * 12)) {
      case 0: c = c.h(a); break
      case 1: c = c.x(a); break
      case 2: c = c.ry(r() * 6, a); break
      case 3: c = c.p(r() * 6, a); break
      case 4: c = c.cnot(a, b); break
      case 5: c = c.cu1(r() * 6, a, b); break
      case 6: c = c.swap(a, b); break
      case 7: c = c.cswap(a, b, d); break
      case 8: c = c.ccx(a, b, d); break
      case 9: c = c.xx(r() * 6, a, b); break
      case 10: c = c.cz(a, b); break
      default: c = c.t(a); break
    }
  }
  return c
}

describe('branch backend — exactness', () => {
  it('matches the statevector on random circuits over the whole gate menu', () => {
    for (let seed = 1; seed <= 12; seed++) {
      const c = randomCircuit(5, 40, seed)
      expect(maxDiff(c).diff).toBeLessThan(1e-10)
      expect(c.runBranch({ shots: 1, seed }).backend).toBe('branch')
    }
  })

  it('reproduces exact amplitudes of the Beauregard QPE circuit to 1e-12', () => {
    for (const [N, a] of [[15n, 7n], [21n, 2n], [33n, 5n]] as const) {
      const c = shorCircuit(N, a, 2)
      const { diff, peak } = maxDiff(c)
      expect(diff).toBeLessThan(1e-12)
      expect(peak).toBe(4)                                   // 2 counting qubits → 4 branches, never more
    }
  })

  it('a Bell pair is two branches with the right probabilities', () => {
    const d = new Circuit(2).h(0).cnot(0, 1).runBranch({ shots: 2000, seed: 1 })
    expect(d.peakBranches).toBe(2)
    expect(Math.abs(d.probs['00']! - 0.5)).toBeLessThan(0.05)
    expect(Math.abs(d.probs['11']! - 0.5)).toBeLessThan(0.05)
    expect(d.probs['01']).toBeUndefined()
  })

  it('a product state never splits, whatever the local bases', () => {
    const d = new Circuit(6).h(0).h(1).ry(0.3, 2).p(1.1, 0).cu1(0.7, 0, 3).cnot(3, 4).runBranch({ shots: 10, seed: 1 })
    expect(d.peakBranches).toBe(1)                           // cu1 with basis-state q3 is a phase; cnot with basis control is an X
  })
})

describe('branch backend — measurement, reset, classical control', () => {
  it('teleports a state exactly, through non-orthogonal branches', () => {
    // Prepare |ψ⟩ = Ry(1.1)|0⟩ on q0, teleport to q2, then undo Ry — q2 must read 0.
    let c = new Circuit(3).ry(1.1, 0).h(1).cnot(1, 2).cnot(0, 1).h(0).measure(0, 'a', 0).measure(1, 'b', 0)
    c = c.if('b', 1, cc => cc.x(2)).if('a', 1, cc => cc.z(2)).ry(-1.1, 2)
    const d = c.runBranch({ shots: 200, seed: 7 })
    for (const [bits, p] of Object.entries(d.probs)) if (bits[2] === '1') expect(p).toBe(0)
    expect(Math.abs(d.cregs['a']![0]! - 0.5)).toBeLessThan(0.15)
  })

  it('agrees with run() on a measured-and-reset circuit', () => {
    const c = new Circuit(2).h(0).cnot(0, 1).measure(0, 'm', 0).reset(0).h(0)
    const a = c.run({ shots: 4000, seed: 3 }), b = c.runBranch({ shots: 4000, seed: 3 })
    for (const k of ['00', '01', '10', '11']) expect(Math.abs((a.probs[k] ?? 0) - (b.probs[k] ?? 0))).toBeLessThan(0.05)
    expect(Math.abs(a.cregs['m']![0]! - b.cregs['m']![0]!)).toBeLessThan(0.05)
  })

  it('reports peakBranches across shots and honours maxBranches', () => {
    const c = new Circuit(3).h(0).h(1).h(2).cnot(0, 1).cnot(1, 2).measure(2, 'c', 0)
    expect(c.runBranch({ shots: 5, seed: 1 }).peakBranches).toBeGreaterThan(1)
    expect(() => c.runBranch({ shots: 1, maxBranches: 1 })).toThrow(/maxBranches/)
  })

  it('rejects unknown options and a wrong-length initialState', () => {
    expect(() => new Circuit(1).runBranch({ bogus: 1 } as never)).toThrow(/unknown option/i)
    expect(() => new Circuit(2).runBranch({ initialState: '1' })).toThrow(RangeError)
    expect(new Circuit(2).runBranch({ initialState: '10', shots: 3 }).most).toBe('10')
  })
})

describe('iterative Shor on the branch backend', () => {
  it('shorCircuitIterative measures the phase bits least-significant first and recovers the period', () => {
    const c = shorCircuitIterative(15n, 7n)
    expect(c.qubits).toBe(2 * 4 + 3)
    const seen = new Set<number>()
    for (let seed = 1; seed <= 8; seed++) {
      const d = c.runBranch({ shots: 1, seed })
      let m = 0
      for (let k = 0; k < 9; k++) if (d.cregs[`m${k}`]![0]! > 0.5) m += 2 ** (8 - k)
      seen.add(m)
      expect(m % 128).toBe(0)                               // phase is exactly s/4 for period 4
      expect(d.peakBranches).toBeLessThanOrEqual(4)
    }
    expect(seen.size).toBeGreaterThan(1)
  })

  it('factors an 8-bit and a 12-bit modulus honestly, cost equal to the period', () => {
    const r8 = shorBeauregard(143n, { a: 73n, iterative: true, seed: 1 })
    expect(r8.method).toBe('quantum')
    expect(r8.factors?.map(Number).sort((a, b) => a - b)).toEqual([11, 13])
    expect(r8.period).toBe(20n)
    expect(r8.qubits).toBe(19)

    const t = performance.now()
    const r12 = shorBeauregard(3233n, { a: 1661n, iterative: true, seed: 1 })
    expect(performance.now() - t).toBeLessThan(20_000)
    expect(r12.method).toBe('quantum')
    expect(r12.factors?.map(Number).sort((a, b) => a - b)).toEqual([53, 61])
    expect(modPow(1661n, r12.period!, 3233n)).toBe(1n)
  })

  it('still takes the classical exits and reports them', () => {
    expect(shorBeauregard(143n, { a: 11n, iterative: true }).method).toBe('classical-gcd')
    expect(shorBeauregard(144n, { iterative: true }).method).toBe('classical-even')
  })
})
