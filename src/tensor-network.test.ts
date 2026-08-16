import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { makePrng } from './prng.js'
import { amplitudeByContraction, amplitudeBatchByContraction, amplitudeBySlicedContraction, circuitNetwork, planContraction, planContractionPartitioned, planBest, evaluatePlan, contractNetwork, sliceContraction, projectTensor, permuteTensor, contractPair, sampleByContraction } from './tensor-network.js'

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
    // The baseline must be planned with the effort `sliceContraction` spends on
    // its own base plan, or the width it "gains" is really just the better plan
    // more restarts found, with nothing sliced at all.
    const idx = circuitNetwork(layered(14, 16), '0'.repeat(14)).map(t => t.indices)
    const base = planContraction(idx, { restarts: 24, seed: 1 })
    const s = sliceContraction(idx, { targetWidth: base.width - 1, restarts: 2 })
    expect(s.sliced.length).toBeGreaterThan(0)
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
    const n = 14
    const c = deep(n, 16)
    const b = '0'.repeat(n)
    const sliced = amplitudeBySlicedContraction(c, b, { targetWidth: 8, restarts: 3 })
    const want = c.amplitude(b)

    expect(sliced.width).toBeLessThanOrEqual(8)
    expect(sliced.sliced.length).toBeGreaterThan(1)
    expect(sliced.re).toBeCloseTo(want.re, 9)
    expect(sliced.im).toBeCloseTo(want.im, 9)
  })

  it('costs far less than the naive 2^k the slice count suggests', () => {
    const idx = circuitNetwork(deep(14, 16), '0'.repeat(14)).map(t => t.indices)
    const s = sliceContraction(idx, { targetWidth: 6, restarts: 2 })
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

describe('tensor network — batched amplitudes', () => {
  it('every amplitude in a batch matches the statevector', () => {
    // A wrong index-to-qubit mapping is the failure mode here, and it hides on
    // circuits whose amplitudes are zero or symmetric — so this sweeps random
    // circuits, several open-qubit subsets, and every entry of each batch.
    let worst = 0
    let checks = 0
    for (const n of [3, 4, 5]) {
      for (let seed = 0; seed < 4; seed++) {
        const rand = makePrng(seed * 991 + n * 7)
        let c = new Circuit(n)
        for (let i = 0; i < n; i++) c = c.h(i).rz(rand() * 2, i)
        for (let d = 0; d < 3; d++) {
          for (let i = 0; i + 1 < n; i++) if (rand() < 0.6) c = c.cnot(i, i + 1)
          for (let i = 0; i < n; i++) c = c.ry(rand() * 2, i).t(i)
        }
        for (const mask of [1, 3, 5, 6]) {
          if (mask >= 1 << n) continue
          const pattern = Array.from({ length: n }, (_, q) =>
            ((mask >> q) & 1) ? '?' : (seed % 2 ? '1' : '0')).join('')
          const batch = amplitudeBatchByContraction(c, pattern, { restarts: 4 })
          const k = batch.open.length
          for (let v = 0; v < 1 << k; v++) {
            const bitVals = Array.from({ length: k }, (_, i) => (v >> (k - 1 - i)) & 1)
            const full = pattern.split('')
            batch.open.forEach((q, i) => { full[q] = String(bitVals[i]) })
            const want = c.amplitude(full.join(''))
            const got = batch.at(bitVals)
            worst = Math.max(worst, Math.abs(got.re - want.re), Math.abs(got.im - want.im))
            checks++
          }
        }
      }
    }
    expect(checks).toBeGreaterThan(150)
    expect(worst).toBeLessThan(1e-9)
  })

  it('returns 2^k amplitudes from one contraction', () => {
    const c = new Circuit(6).h(0).cnot(0, 1).ry(0.5, 2).cz(1, 2).t(3).cnot(3, 4).rx(0.3, 5)
    const batch = amplitudeBatchByContraction(c, '??0?00')
    expect(batch.open).toEqual([0, 1, 3])
    expect(batch.data.length / 2).toBe(8)
  })

  it('rejects a pattern of the wrong length or alphabet', () => {
    const c = new Circuit(3).h(0)
    expect(() => amplitudeBatchByContraction(c, '??')).toThrow(/must have 3 characters/)
    expect(() => amplitudeBatchByContraction(c, '?X0')).toThrow(/only 0, 1 and \?/)
  })
})

describe('tensor network — diagonal gates as hyper-indices', () => {
  const deepCZ = (n: number, depth: number): Circuit => {
    let c = new Circuit(n)
    for (let i = 0; i < n; i++) c = c.h(i)
    for (let d = 0; d < depth; d++) {
      for (let i = d % 2; i + 1 < n; i += 2) c = c.cz(i, i + 1)
      for (let i = 0; i < n; i++) c = c.rz(0.3 + 0.01 * i, i).rx(0.7, i)
    }
    return c
  }

  it('a diagonal gate sits on its wire instead of renaming it', () => {
    // Two CZs and an rz add tensors but no new indices: a diagonal gate does not
    // cut a wire, so the only fresh index in this circuit comes from the h.
    const plain = circuitNetwork(new Circuit(2).h(0), '00')
    const withDiagonals = circuitNetwork(new Circuit(2).h(0).cz(0, 1).rz(0.4, 0).cz(0, 1), '00')
    const indicesOf = (net: { indices: string[] }[]): Set<string> =>
      new Set(net.flatMap(t => t.indices))

    expect(withDiagonals.length).toBeGreaterThan(plain.length)   // more tensors
    expect(indicesOf(withDiagonals).size).toBe(indicesOf(plain).size)   // same indices
  })

  it('keeps a CZ layer from doubling the width', () => {
    // Pins the mechanism by its consequence. Without hyper-indices these widths
    // were 13, 17 and 23 — each unit of width is a doubling of memory.
    for (const [depth, bound] of [[12, 9], [16, 12], [20, 15]] as const) {
      const idx = circuitNetwork(deepCZ(40, depth), '0'.repeat(40)).map(t => t.indices)
      expect(planContraction(idx, { restarts: 16 }).width).toBeLessThanOrEqual(bound)
    }
  })

  it('an open wire ending in a diagonal gate still comes back', () => {
    // The regression the identity cap exists for. A trailing rz/z/t shares the
    // wire's index rather than renaming it, so without the cap the open wire is
    // held by two tensors, looks contractible, and is summed away — a silently
    // wrong amplitude rather than an error.
    for (const trailing of ['rz', 'z', 't', 's'] as const) {
      let c = new Circuit(3).h(0).cnot(0, 1).ry(0.4, 2).cz(0, 2)
      c = trailing === 'rz' ? c.rz(0.9, 0) : c[trailing](0)
      c = trailing === 'rz' ? c.rz(0.5, 1) : c[trailing](1)

      const batch = amplitudeBatchByContraction(c, '??0')
      for (let k = 0; k < 4; k++) {
        const bits = [(k >> 1) & 1, k & 1]
        const want = c.amplitude(`${bits[0]}${bits[1]}0`)
        const got = batch.at(bits)
        expect(got.re, `${trailing} k=${k}`).toBeCloseTo(want.re, 12)
        expect(got.im, `${trailing} k=${k}`).toBeCloseTo(want.im, 12)
      }
    }
  })

  it('contractPair keeps a batch index instead of summing it', () => {
    // Shared index 'b' is held elsewhere, so it survives as a batch index: the
    // result is elementwise along b, not summed over it.
    const a = { indices: ['b'], data: Float64Array.from([2, 0, 3, 0]) }
    const c = { indices: ['b'], data: Float64Array.from([5, 0, 7, 0]) }
    const batched = contractPair(a, c, new Set(['b']))
    expect(batched.indices).toEqual(['b'])
    expect(Array.from(batched.data)).toEqual([10, 0, 21, 0])

    // Without the keep set the same pair is a full contraction: 2*5 + 3*7 = 31.
    const summed = contractPair(a, c)
    expect(summed.indices).toEqual([])
    expect(Array.from(summed.data)).toEqual([31, 0])
  })
})

describe('tensor network — sampling', () => {
  const messy = (n: number, rand: () => number): Circuit => {
    let c = new Circuit(n)
    const one = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg'] as const
    for (let d = 0; d < 3 * n; d++) {
      const r = rand()
      if (n > 1 && r < 0.35) {
        const a = Math.floor(rand() * n); let b = Math.floor(rand() * (n - 1)); if (b >= a) b++
        c = rand() < 0.5 ? c.cz(a, b) : c.cnot(a, b)
      } else if (r < 0.6) c = c.rx((rand() * 6 - 3), Math.floor(rand() * n))
      else if (r < 0.75) c = c.rz((rand() * 6 - 3), Math.floor(rand() * n))
      else c = c[one[Math.floor(rand() * one.length)]!](Math.floor(rand() * n))
    }
    return c
  }

  it('reproduces the exact distribution, not just its support', () => {
    const shots = 40000
    for (let seed = 1; seed <= 6; seed++) {
      const rand = makePrng(seed * 6151 + 7)
      const n = 2 + Math.floor(rand() * 3)
      const c = messy(n, rand)
      const exact = c.exactProbs()
      const got = sampleByContraction(c, { shots, blockSize: 2, seed: 4242 })

      for (const [outcome, p] of Object.entries(exact)) {
        const empirical = (got.counts.get(outcome) ?? 0) / shots
        // 4 sigma of a binomial at this shot count, floored for tiny p.
        const sigma = Math.sqrt(Math.max(p * (1 - p), 1e-6) / shots)
        expect(Math.abs(empirical - p), `seed=${seed} |${outcome}⟩`).toBeLessThan(4 * sigma + 1e-3)
      }
      // Nothing outside the support may ever be emitted.
      for (const outcome of got.counts.keys()) expect(exact[outcome] ?? 0).toBeGreaterThan(0)
    }
  })

  it('gives the same distribution however the qubits are blocked', () => {
    // Chaining is where a conditional sampler goes wrong: block boundaries are
    // exactly where a mis-normalised marginal would show up.
    const c = new Circuit(8).h(0).cnot(0, 1).cz(1, 2).ry(0.7, 3)
      .cnot(3, 4).rx(0.4, 5).cz(5, 6).t(7).cnot(6, 7).h(2)
    const exact = c.exactProbs()
    const shots = 40000
    for (const blockSize of [1, 2, 3, 8]) {
      const got = sampleByContraction(c, { shots, blockSize, seed: 99 })
      for (const [outcome, p] of Object.entries(exact)) {
        const empirical = (got.counts.get(outcome) ?? 0) / shots
        expect(Math.abs(empirical - p), `blockSize=${blockSize} |${outcome}⟩`)
          .toBeLessThan(4 * Math.sqrt(Math.max(p * (1 - p), 1e-6) / shots) + 1e-3)
      }
    }
  })

  it('a deterministic circuit yields exactly one outcome', () => {
    const got = sampleByContraction(new Circuit(5).x(0).x(3), { shots: 500, blockSize: 2 })
    expect([...got.counts.entries()]).toEqual([['10010', 500]])
  })

  it('reaches widths no statevector could, on a circuit no MPS is shaped for', () => {
    // 60 qubits: 2^60 amplitudes do not exist. Long-range CZs, so this is not a
    // chain an MPS would keep narrow either.
    let c = new Circuit(60)
    for (let i = 0; i < 60; i++) c = c.h(i)
    for (let i = 0; i + 30 < 60; i++) c = c.cz(i, i + 30)
    for (let i = 0; i < 60; i++) c = c.rz(0.2 + 0.01 * i, i)
    const got = sampleByContraction(c, { shots: 200, blockSize: 4, restarts: 4 })
    let total = 0
    for (const [outcome, k] of got.counts) { expect(outcome).toMatch(/^[01]{60}$/); total += k }
    expect(total).toBe(200)
  })

  it('the light cone keeps the conditional cache small without changing it', () => {
    // Same circuit, same seed: whatever the cache does, the samples must be the
    // ones the chain would have produced anyway. The cone only decides how often
    // a conditional is recomputed, never what it is.
    let c = new Circuit(24)
    for (let i = 0; i < 24; i++) c = c.h(i)
    for (let d = 0; d < 3; d++) {
      for (let i = d % 2; i + 1 < 24; i += 2) c = c.cz(i, i + 1)
      for (let i = 0; i < 24; i++) c = c.rx(0.6, i)
    }
    const got = sampleByContraction(c, { shots: 2000, blockSize: 4, seed: 7, restarts: 4 })
    // 6 blocks x 2000 shots is 12,000 conditionals if nothing is shared.
    expect(got.contractions).toBeLessThan(400)
    expect([...got.counts.values()].reduce((a, b) => a + b, 0)).toBe(2000)
  })

  it('rejects nonsense parameters', () => {
    const c = new Circuit(3).h(0)
    expect(() => sampleByContraction(c, { shots: 0 })).toThrow(/positive integer/)
    expect(() => sampleByContraction(c, { blockSize: 0 })).toThrow(/1\.\.16/)
    expect(() => sampleByContraction(c, { blockSize: 17 })).toThrow(/1\.\.16/)
  })
})
