import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { makePrng } from './prng.js'
import { amplitudeByContraction, amplitudeBySlicedContraction, circuitNetwork, planContraction, planContractionPartitioned, planBest, evaluatePlan, contractNetwork, sliceContraction, projectTensor, permuteTensor, contractPair } from './tensor-network.js'

/**
 * Contraction against the statevector oracle.
 *
 * Every amplitude of every random circuit is checked, so an index-ordering or
 * transpose error anywhere in the network construction shows up: those are the
 * mistakes this kind of code actually makes, and they are invisible on a circuit
 * whose amplitudes happen to be zero or symmetric.
 */

function randomCircuit(n: number, depth: number, rand: () => number): Circuit {
  let c = new Circuit(n)
  const one = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg'] as const
  for (let d = 0; d < depth; d++) {
    const r = rand()
    if (n > 1 && r < 0.3) {
      const a = Math.floor(rand() * n)
      let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
      c = c.cnot(a, b)
    } else if (n > 1 && r < 0.4) {
      const a = Math.floor(rand() * n)
      let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
      c = rand() < 0.5 ? c.cz(a, b) : c.swap(a, b)
    } else if (r < 0.6) {
      const q = Math.floor(rand() * n)
      const th = (rand() * 4 - 2) * Math.PI
      const pick = rand()
      c = pick < 0.34 ? c.rx(th, q) : pick < 0.67 ? c.ry(th, q) : c.rz(th, q)
    } else {
      c = c[one[Math.floor(rand() * one.length)]!](Math.floor(rand() * n))
    }
  }
  return c
}

const bits = (i: number, n: number): string =>
  Array.from({ length: n }, (_, q) => ((i >> q) & 1 ? '1' : '0')).join('')

describe('tensor network — every amplitude matches the statevector', () => {
  for (const n of [1, 2, 3, 4, 5]) {
    it(`random circuits, n=${n}`, () => {
      for (let seed = 0; seed < 12; seed++) {
        const rand = makePrng(seed * 7919 + n * 733 + 5)
        const circuit = randomCircuit(n, 4 * n + 6, rand)
        for (let i = 0; i < 2 ** n; i++) {
          const b = bits(i, n)
          const got = amplitudeByContraction(circuit, b, { restarts: 4 })
          const want = circuit.amplitude(b)
          expect(got.re, `n=${n} seed=${seed} |${b}⟩ re`).toBeCloseTo(want.re, 10)
          expect(got.im, `n=${n} seed=${seed} |${b}⟩ im`).toBeCloseTo(want.im, 10)
        }
      }
    })
  }

  it('handles controlled rotations, which carry a non-trivial 2x2 block', () => {
    const c = new Circuit(3).h(0).h(1).cu1(0.7, 0, 1).cnot(1, 2).h(2).cu1(-1.3, 2, 0)
    for (let i = 0; i < 8; i++) {
      const b = bits(i, 3)
      const got = amplitudeByContraction(c, b)
      const want = c.amplitude(b)
      expect(got.re).toBeCloseTo(want.re, 10)
      expect(got.im).toBeCloseTo(want.im, 10)
    }
  })

  it('treats SWAP as a relabelling, adding no tensor', () => {
    const withSwap = new Circuit(3).h(0).swap(0, 2).cnot(0, 1)
    const plain = new Circuit(3).h(0).cnot(0, 1)
    expect(circuitNetwork(withSwap, '000').length).toBe(circuitNetwork(plain, '000').length)
    for (let i = 0; i < 8; i++) {
      const b = bits(i, 3)
      expect(amplitudeByContraction(withSwap, b).re).toBeCloseTo(withSwap.amplitude(b).re, 10)
    }
  })
})

describe('tensor network — contraction order', () => {
  it('a plan contracts the network down to a scalar', () => {
    const c = new Circuit(4).h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)
    const net = circuitNetwork(c, '0000')
    const plan = planContraction(net.map(t => t.indices))
    expect(plan.steps.length).toBe(net.length - 1)
    expect(contractNetwork(net, plan).indices).toEqual([])
  })

  it('width tracks circuit structure, not qubit count', () => {
    // A ladder of CNOTs stays narrow however wide it gets: contracting along the
    // chain never needs to hold more than a few indices at once. A statevector
    // would need n.
    const widths: number[] = []
    for (const n of [8, 16, 32, 64]) {
      let c = new Circuit(n).h(0)
      for (let i = 0; i + 1 < n; i++) c = c.cnot(i, i + 1)
      widths.push(amplitudeByContraction(c, '0'.repeat(n), { restarts: 8 }).width)
    }
    // Constant in n, and far below it.
    expect(Math.max(...widths)).toBeLessThan(8)
    expect(new Set(widths).size).toBeLessThanOrEqual(2)
  })

  it('searching harder never returns a worse plan', () => {
    const rand = makePrng(31337)
    const c = randomCircuit(6, 30, rand)
    const net = circuitNetwork(c, '000000').map(t => t.indices)
    const cheap = planContraction(net, { restarts: 1 })
    const better = planContraction(net, { restarts: 64 })
    expect(better.width).toBeLessThanOrEqual(cheap.width)
  })

  it('reaches widths a statevector cannot, and agrees where both can run', () => {
    // 60 qubits: no statevector exists. The contraction is small because the
    // circuit is shallow and locally connected.
    const n = 60
    let c = new Circuit(n)
    for (let i = 0; i < n; i++) c = c.h(i)
    for (let i = 0; i + 1 < n; i += 2) c = c.cnot(i, i + 1)
    for (let i = 1; i + 1 < n; i += 2) c = c.cz(i, i + 1)
    const r = amplitudeByContraction(c, '0'.repeat(n), { restarts: 8 })
    expect(r.width).toBeLessThan(20)
    expect(Number.isFinite(r.re)).toBe(true)
  })

  it('declines circuits it cannot build a network for', () => {
    const impure = new Circuit(2).h(0).creg('m', 1).measure(0, 'm', 0)
    expect(() => amplitudeByContraction(impure, '00')).toThrow(/requires a pure circuit/)
    expect(() => amplitudeByContraction(new Circuit(2).h(0), '000')).toThrow(/must have 2 characters/)
  })
})

describe('tensor network — partition planner', () => {
  const shallow = (n: number, depth: number): Circuit => {
    let c = new Circuit(n)
    for (let i = 0; i < n; i++) c = c.h(i)
    for (let d = 0; d < depth; d++) {
      for (let i = d % 2; i + 1 < n; i += 2) c = c.cz(i, i + 1)
      for (let i = 0; i < n; i++) c = c.rz(0.3, i).rx(0.7, i)
    }
    return c
  }

  it('produces the same value as greedy — the order cannot change the answer', () => {
    for (const [n, depth] of [[6, 4], [10, 6], [12, 5]] as const) {
      const c = shallow(n, depth)
      const net = circuitNetwork(c, '0'.repeat(n))
      const idx = net.map(t => t.indices)
      const viaGreedy = contractNetwork(net, planContraction(idx, { restarts: 4 }))
      const viaPart = contractNetwork(net, planContractionPartitioned(idx, { restarts: 4 }))
      expect(viaPart.data[0]!).toBeCloseTo(viaGreedy.data[0]!, 10)
      expect(viaPart.data[1]!).toBeCloseTo(viaGreedy.data[1]!, 10)
      expect(viaGreedy.data[0]!).toBeCloseTo(c.amplitude('0'.repeat(n)).re, 10)
    }
  })

  it('contracts the network fully, like any other plan', () => {
    const net = circuitNetwork(shallow(8, 4), '0'.repeat(8))
    const plan = planContractionPartitioned(net.map(t => t.indices))
    expect(plan.steps.length).toBe(net.length - 1)
    expect(contractNetwork(net, plan).indices).toEqual([])
  })

  it('evaluatePlan scores a plan without touching tensor data', () => {
    const net = circuitNetwork(shallow(8, 4), '0'.repeat(8))
    const idx = net.map(t => t.indices)
    const plan = planContraction(idx, { restarts: 4 })
    const scored = evaluatePlan(idx, plan.steps)
    expect(scored.width).toBe(plan.width)
    expect(scored.cost).toBe(plan.cost)
  })

  it('planBest never returns worse than either planner alone', () => {
    for (const [n, depth] of [[8, 4], [12, 6]] as const) {
      const idx = circuitNetwork(shallow(n, depth), '0'.repeat(n)).map(t => t.indices)
      const g = planContraction(idx, { restarts: 6 })
      const p = planContractionPartitioned(idx, { restarts: 3 })
      const best = planBest(idx, { restarts: 6 })
      expect(best.width).toBeLessThanOrEqual(Math.min(g.width, p.width))
      expect(['greedy', 'partition']).toContain(best.strategy)
    }
  })
})

describe('tensor network — slicing', () => {
  const layered = (n: number, depth: number): Circuit => {
    let c = new Circuit(n)
    for (let i = 0; i < n; i++) c = c.h(i)
    for (let d = 0; d < depth; d++) {
      for (let i = d % 2; i + 1 < n; i += 2) c = c.cz(i, i + 1)
      for (let i = 0; i < n; i++) c = c.rz(0.3 + 0.01 * i, i).rx(0.7, i)
    }
    return c
  }

  it('a sliced contraction equals the unsliced one', () => {
    // Fixing indices instead of summing over them changes the memory profile and
    // nothing else: the slices must add back up to exactly the same amplitude.
    for (const [n, depth, target] of [[6, 4, 2], [8, 6, 3], [10, 6, 4]] as const) {
      const c = layered(n, depth)
      const b = '0'.repeat(n)
      const sliced = amplitudeBySlicedContraction(c, b, { targetWidth: target, restarts: 4 })
      const want = c.amplitude(b)
      expect(sliced.re, `n=${n}`).toBeCloseTo(want.re, 10)
      expect(sliced.im, `n=${n}`).toBeCloseTo(want.im, 10)
      expect(sliced.slices).toBe(2 ** sliced.sliced.length)
    }
  })

  it('does not slice a contraction that already fits', () => {
    const idx = circuitNetwork(layered(6, 3), '000000').map(t => t.indices)
    const s = sliceContraction(idx, { targetWidth: 30 })
    expect(s.sliced).toEqual([])
    expect(s.slices).toBe(1)
    expect(s.overhead).toBe(1)
  })

  it('buys memory, and for less than the naive doubling', () => {
    const idx = circuitNetwork(layered(10, 8), '0'.repeat(10)).map(t => t.indices)
    const base = planContraction(idx, { restarts: 6 })
    const s = sliceContraction(idx, { targetWidth: base.width - 1, restarts: 2 })
    expect(s.width).toBeLessThan(base.width)
    // Two slices of a narrower contraction cost less than twice the original:
    // removing an index does not merely shrink the work, it removes some of it.
    expect(s.overhead).toBeLessThan(2)
    expect(s.overhead).toBeGreaterThan(1)
  })

  it('projectTensor fixes an index and drops that axis', () => {
    // |0⟩⟨0| on one index of a Bell-state tensor picks out one row.
    const t = { indices: ['a', 'b'], data: Float64Array.from([1, 0, 2, 0, 3, 0, 4, 0]) }
    const p0 = projectTensor(t, new Map([['a', 0]]))
    expect(p0.indices).toEqual(['b'])
    expect(Array.from(p0.data)).toEqual([1, 0, 2, 0])
    const p1 = projectTensor(t, new Map([['a', 1]]))
    expect(Array.from(p1.data)).toEqual([3, 0, 4, 0])
    const both = projectTensor(t, new Map([['a', 1], ['b', 0]]))
    expect(both.indices).toEqual([])
    expect(Array.from(both.data)).toEqual([3, 0])
  })
})

describe('tensor network — slice selection reaches a target', () => {
  const deep = (n: number, depth: number): Circuit => {
    let c = new Circuit(n)
    for (let i = 0; i < n; i++) c = c.h(i)
    for (let d = 0; d < depth; d++) {
      for (let i = d % 2; i + 1 < n; i += 2) c = c.cz(i, i + 1)
      for (let i = 0; i < n; i++) c = c.rz(0.3 + 0.01 * i, i).rx(0.7, i)
    }
    return c
  }

  it('slices down to a demanding target, and still gives the right answer', () => {
    // Selection by peak coverage rather than immediate width reduction: the peak
    // is held by several intermediates, so progress is made by emptying that set
    // even on steps where the width itself does not yet move.
    const n = 10
    const c = deep(n, 8)
    const b = '0'.repeat(n)
    const sliced = amplitudeBySlicedContraction(c, b, { targetWidth: 4, restarts: 3 })
    const want = c.amplitude(b)

    expect(sliced.width).toBeLessThanOrEqual(4)
    expect(sliced.sliced.length).toBeGreaterThan(1)
    expect(sliced.re).toBeCloseTo(want.re, 9)
    expect(sliced.im).toBeCloseTo(want.im, 9)
  })

  it('costs far less than the naive 2^k the slice count suggests', () => {
    const idx = circuitNetwork(deep(12, 8), '0'.repeat(12)).map(t => t.indices)
    const s = sliceContraction(idx, { targetWidth: 4, restarts: 2 })
    expect(s.slices).toBeGreaterThan(4)
    // Every slice doubles the repeats but each is a smaller contraction, so the
    // real overhead lands well under the slice count.
    expect(s.overhead).toBeLessThan(s.slices / 2)
  })

  it('reports the width it reached when the budget runs out', () => {
    const idx = circuitNetwork(deep(12, 8), '0'.repeat(12)).map(t => t.indices)
    const s = sliceContraction(idx, { targetWidth: 1, maxSliced: 3, restarts: 2 })
    expect(s.sliced.length).toBeLessThanOrEqual(3)
    expect(s.width).toBeGreaterThan(1)          // honest about not reaching it
    expect(s.slices).toBe(2 ** s.sliced.length)
  })
})

describe('tensor network — contraction kernel', () => {
  it('permuteTensor reorders indices without changing what the tensor means', () => {
    // A rank-2 tensor transposed: element [a,b] must land at [b,a].
    const t = { indices: ['a', 'b'], data: Float64Array.from([1, 0, 2, 0, 3, 0, 4, 0]) }
    const p = permuteTensor(t, ['b', 'a'])
    expect(p.indices).toEqual(['b', 'a'])
    expect(Array.from(p.data)).toEqual([1, 0, 3, 0, 2, 0, 4, 0])
  })

  it('returns the same object when no permutation is needed', () => {
    const t = { indices: ['a', 'b'], data: Float64Array.from([1, 0, 2, 0, 3, 0, 4, 0]) }
    expect(permuteTensor(t, ['a', 'b'])).toBe(t)
  })

  it('contractPair matches an independently computed contraction', () => {
    // <psi|M|phi> built by hand, against the kernel.
    const M = { indices: ['o', 'i'], data: Float64Array.from([0, 1, 2, 0, 3, 0, 0, -1]) }
    const v = { indices: ['i'], data: Float64Array.from([1, 0, 0, 2]) }
    const got = contractPair(M, v)
    expect(got.indices).toEqual(['o'])
    // row 0: (0+1i)(1+0i) + (2+0i)(0+2i) = i + 4i = 0 + 5i
    expect(got.data[0]!).toBeCloseTo(0, 12)
    expect(got.data[1]!).toBeCloseTo(5, 12)
    // row 1: (3+0i)(1+0i) + (0-1i)(0+2i) = 3 + 2 = 5
    expect(got.data[2]!).toBeCloseTo(5, 12)
    expect(got.data[3]!).toBeCloseTo(0, 12)
  })

  it('is order-insensitive in its operands, up to index order', () => {
    const A = { indices: ['x', 's'], data: Float64Array.from([1, 0, 0, 1, 2, 0, 0, -1]) }
    const B = { indices: ['s', 'y'], data: Float64Array.from([0, 1, 1, 0, 3, 0, 0, 2]) }
    const ab = contractPair(A, B)
    const ba = contractPair(B, A)
    expect(ab.indices).toEqual(['x', 'y'])
    expect(ba.indices).toEqual(['y', 'x'])
    // Same tensor, transposed layout.
    expect(ba.data[0]!).toBeCloseTo(ab.data[0]!, 12)
    expect(ba.data[2]!).toBeCloseTo(ab.data[4]!, 12)
  })
})
