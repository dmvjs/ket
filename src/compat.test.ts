import { describe, it, expect } from 'vitest'
import { QuantumCircuit } from './compat.js'
import { Circuit } from './circuit.js'

// Expected values in this file were captured from the `quantum-circuit` npm
// package (v0.9.248) running the same script, so they pin compatibility rather
// than merely describing what this implementation happens to do.

describe('compat — shape and placement', () => {
  it('defaults to one qubit and grows to fit any wire named', () => {
    const c = new QuantumCircuit()
    expect(c.numQubits).toBe(1)
    c.addGate('h', -1, 3)
    expect(c.numQubits).toBe(4)
  })

  it('counts columns up to the highest occupied one, including gaps', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 5, 0)
    expect(c.numCols()).toBe(6)
    expect(c.getDepth()).toBe(1)      // depth ignores the gaps
  })

  it('appends at the end when the column is negative', () => {
    const c = new QuantumCircuit(1)
    c.addGate('x', -1, 0)
    c.addGate('x', -1, 0)
    expect(c.numCols()).toBe(2)
  })

  it('shares a column between gates on different wires', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('x', 0, 1)
    c.addGate('cx', 1, [0, 1])
    expect(c.numCols()).toBe(2)
    expect(c.getDepth()).toBe(2)
  })

  it('reports used gates in first-use order', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('cx', 1, [0, 1])
    expect(c.usedGates()).toEqual(['h', 'cx'])
  })

  it('reports 2^n amplitudes', () => {
    expect(new QuantumCircuit(3).numAmplitudes()).toBe(8)
  })

  it('removes a gate by id', () => {
    const c = new QuantumCircuit(1)
    const id = c.addGate('x', 0, 0)
    c.addGate('x', 1, 0)
    c.removeGate(id)
    c.run()
    expect(c.probability(0)).toBeCloseTo(1, 12)   // one X left, so |1⟩
  })

  it('rejects a gate given the wrong number of wires', () => {
    expect(() => new QuantumCircuit(2).addGate('cx', 0, 0)).toThrow(/spans 2 wire\(s\)/)
  })

  it('rejects an unknown gate', () => {
    expect(() => new QuantumCircuit(1).addGate('bogus', 0, 0)).toThrow(/unknown gate/)
  })
})

describe('compat — bit order matches quantum-circuit, not ket', () => {
  it('prints wire 0 as the rightmost character', () => {
    const c = new QuantumCircuit(2)
    c.addGate('x', 0, 0)
    c.run()
    // quantum-circuit prints |01>; ket natively prints '10'
    expect(c.stateAsString(true)).toBe(' 1.00000000+0.00000000i|01>\t100.00000%\n')
  })

  it('puts the highest wire leftmost for three qubits', () => {
    const c = new QuantumCircuit(3)
    c.addGate('x', 0, 2)
    c.run()
    expect(c.stateAsString(true)).toBe(' 1.00000000+0.00000000i|100>\t100.00000%\n')
  })

  it('indexes probabilities by wire', () => {
    const c = new QuantumCircuit(2)
    c.addGate('x', 0, 0)
    c.run()
    expect(c.probabilities()).toEqual([1, 0])
  })

  it('indexes measureAll by wire', () => {
    const c = new QuantumCircuit(2)
    c.addGate('x', 0, 0)
    c.run()
    expect(c.measureAll()).toEqual([1, 0])
  })

  it('keys multishot counts in quantum-circuit order', () => {
    const c = new QuantumCircuit(3)
    c.addGate('x', 0, 2)
    c.run()
    expect(c.measureAllMultishot(4)).toEqual({ '100': 4 })
  })

  it('emits the full state in index order when onlyPossible is false', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('cx', 1, [0, 1])
    c.run()
    expect(c.stateAsString(false).split('\n').filter(Boolean)).toHaveLength(4)
    expect(c.stateAsString(true).split('\n').filter(Boolean)).toHaveLength(2)
  })
})

describe('compat — running', () => {
  it('reproduces the Bell state', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('cx', 1, [0, 1])
    c.run()
    expect(c.stateAsString(true)).toBe(
      ' 0.70710678+0.00000000i|00>\t 50.00000%\n 0.70710678+0.00000000i|11>\t 50.00000%\n',
    )
    expect(c.probabilities()).toEqual([0.5, 0.5])
  })

  it('sets initial wire values from run()', () => {
    const c = new QuantumCircuit(2)
    c.run([1, 0])
    expect(c.stateAsString(true)).toBe(' 1.00000000+0.00000000i|01>\t100.00000%\n')
  })

  it('accepts numeric and string gate parameters', () => {
    const s = new QuantumCircuit(1); s.addGate('rx', 0, 0, { params: { theta: 'pi/2' } }); s.run()
    const n = new QuantumCircuit(1); n.addGate('rx', 0, 0, { params: { theta: Math.PI / 2 } }); n.run()
    expect(s.probability(0)).toBeCloseTo(0.5, 12)
    expect(n.probability(0)).toBeCloseTo(0.5, 12)
  })

  it('rejects a parameterised gate with a missing parameter', () => {
    const c = new QuantumCircuit(1)
    c.addGate('rx', 0, 0)
    expect(() => c.run()).toThrow(/missing parameter 'theta'/)
  })

  it('refuses to read state before run()', () => {
    expect(() => new QuantumCircuit(1).probabilities()).toThrow(/not initialized/)
    expect(new QuantumCircuit(1).stateAsString()).toMatch(/^Error: circuit is not initialized/)
  })

  it('draws an independent result on each unseeded run', () => {
    // The README's 8-bit QRNG: a seeded-from-the-clock PRNG would repeat here
    const draw = (): number => {
      const c = new QuantumCircuit()
      for (let i = 0; i < 8; i++) { c.addGate('h', -1, i); c.addMeasure(i, 'c', i) }
      c.run()
      return c.getCregValue('c')
    }
    const draws = new Set(Array.from({ length: 20 }, draw))
    expect(draws.size).toBeGreaterThan(1)
    for (const v of draws) expect(v).toBeGreaterThanOrEqual(0)
    for (const v of draws) expect(v).toBeLessThanOrEqual(255)
  })

  it('repeats exactly when given a seed', () => {
    const draw = (): string => {
      const c = new QuantumCircuit(2)
      c.addGate('h', 0, 0); c.addGate('cx', 1, [0, 1])
      c.addMeasure(0, 'c', 0); c.addMeasure(1, 'c', 1)
      c.run(undefined, { seed: 12345 })
      return c.cregsAsString()
    }
    expect(draw()).toBe(draw())
  })
})

describe('compat — classical registers', () => {
  it('creates a register from addMeasure and reads it back', () => {
    const c = new QuantumCircuit(2)
    c.addGate('x', -1, 0)
    c.addMeasure(0, 'c', 0)
    c.addMeasure(1, 'c', 1)
    c.run()
    expect(c.getCregs()).toEqual({ c: 1 })
    expect(c.getCregValue('c')).toBe(1)
    expect(c.getCregBit('c', 0)).toBe(1)
    expect(c.getCregBit('c', 1)).toBe(0)
    expect(c.cregsAsString()).toBe('reg\tbin\tdec\nc\t01\t1\n')
  })

  it('leaves the state alone when measuring without classical control', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addMeasure(0, 'c', 0)
    c.run()
    // non-destructive: the superposition is still reported
    expect(c.probability(0)).toBeCloseTo(0.5, 12)
  })

  it('collapses the state when a classically-controlled gate is present', () => {
    const c = new QuantumCircuit(2)
    c.addGate('x', -1, 0)
    c.addMeasure(0, 'c', 0)
    c.addGate('x', -1, 1, { condition: { creg: 'c', value: 1 } })
    c.run()
    expect(c.measureAllMultishot(8)).toEqual({ '11': 8 })
  })

  it('does not fire a conditioned gate when the register does not match', () => {
    const c = new QuantumCircuit(2)
    c.addMeasure(0, 'c', 0)                                    // measures |0⟩
    c.addGate('x', -1, 1, { condition: { creg: 'c', value: 1 } })
    c.run()
    expect(c.getCregValue('c')).toBe(0)
    expect(c.probability(1)).toBeCloseTo(0, 12)
  })

  it('correlates the bits of an entangled measurement', () => {
    const seen = new Set<string>()
    for (let i = 0; i < 40; i++) {
      const c = new QuantumCircuit(2)
      c.addGate('h', 0, 0); c.addGate('cx', 1, [0, 1])
      c.addMeasure(0, 'c', 0); c.addMeasure(1, 'c', 1)
      c.run()
      seen.add(`${c.getCregBit('c', 0)}${c.getCregBit('c', 1)}`)
    }
    expect([...seen].sort()).toEqual(['00', '11'])
  })

  it('sets and reads a bit directly', () => {
    const c = new QuantumCircuit(1)
    c.createCreg('ans', 5)
    c.setCregBit('ans', 3, 1)
    expect(c.getCregValue('ans')).toBe(8)
    expect(c.getCregBit('ans', 3)).toBe(1)
  })

  it('rejects a condition on an unknown register', () => {
    const c = new QuantumCircuit(1)
    c.addGate('x', 0, 0, { condition: { creg: 'nope', value: 1 } })
    expect(() => c.run()).toThrow(/unknown register 'nope'/)
  })
})

describe('compat — custom gates, save and load', () => {
  const bell = (): QuantumCircuit => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('cx', 1, [0, 1])
    return c
  }

  it('saves the quantum-circuit object shape', () => {
    const saved = bell().save()
    expect(Object.keys(saved).sort()).toEqual(
      ['cregs', 'customGates', 'gates', 'numQubits', 'options', 'params'],
    )
    expect(saved.numQubits).toBe(2)
    expect(saved.gates).toHaveLength(2)                       // one row per wire
    expect(saved.gates[0]?.[0]?.name).toBe('h')
    expect(saved.gates[0]?.[1]?.connector).toBe(0)            // control end of cx
    expect(saved.gates[1]?.[1]?.connector).toBe(1)            // target end of cx
  })

  it('round-trips through save and load', () => {
    const c = new QuantumCircuit(1)
    c.load(bell().save())
    c.run()
    expect(c.numQubits).toBe(2)
    expect(c.probabilities()).toEqual([0.5, 0.5])
  })

  it('uses a saved circuit as a gate in another circuit', () => {
    const outer = new QuantumCircuit(3)
    outer.registerGate('bell', bell().save())
    outer.addGate('bell', 0, [0, 1])
    outer.run()
    expect(outer.stateAsString(true)).toBe(
      ' 0.70710678+0.00000000i|000>\t 50.00000%\n 0.70710678+0.00000000i|011>\t 50.00000%\n',
    )
  })

  it('accepts a QuantumCircuit directly as a custom gate', () => {
    const outer = new QuantumCircuit(2)
    outer.registerGate('bell', bell())
    outer.addGate('bell', 0, [0, 1])
    outer.run()
    expect(outer.probabilities()).toEqual([0.5, 0.5])
  })

  it('maps a custom gate onto the wires it is given', () => {
    const outer = new QuantumCircuit(4)
    outer.registerGate('bell', bell().save())
    outer.addGate('bell', 0, [2, 3])
    outer.run()
    expect(outer.probabilities()).toEqual([0, 0, 0.5, 0.5])
  })

  it('rejects a custom gate placed on too few wires', () => {
    const outer = new QuantumCircuit(3)
    outer.registerGate('bell', bell().save())
    expect(() => outer.addGate('bell', 0, [0])).toThrow(/spans 2 wire\(s\)/)
  })

  it('appends one circuit after another', () => {
    const a = new QuantumCircuit(2); a.addGate('h', 0, 0)
    const b = new QuantumCircuit(2); b.addGate('cx', 0, [0, 1])
    a.appendCircuit(b)
    a.run()
    expect(a.numCols()).toBe(2)
    expect(a.probabilities()).toEqual([0.5, 0.5])
  })

  it('clears gates and state', () => {
    const c = bell()
    c.run()
    c.clearGates()
    expect(c.numCols()).toBe(0)
    expect(() => c.probabilities()).toThrow(/not initialized/)
  })
})

describe('compat — import and export', () => {
  it('exports OpenQASM', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('cx', 1, [0, 1])
    expect(c.exportQASM()).toContain('h q[0];')
    expect(c.exportQASM()).toContain('cx q[0],q[1];')
  })

  it('imports OpenQASM', () => {
    const c = new QuantumCircuit()
    c.importQASM('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];')
    c.run()
    expect(c.numQubits).toBe(2)
    expect(c.probabilities()).toEqual([0.5, 0.5])
  })

  it('round-trips a parameterised circuit through QASM', () => {
    const c = new QuantumCircuit(1)
    c.addGate('rx', 0, 0, { params: { theta: 'pi/2' } })
    const back = new QuantumCircuit()
    back.importQASM(c.exportQASM())
    back.run()
    expect(back.probability(0)).toBeCloseTo(0.5, 12)
  })

  it('delegates the other exporters to ket', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('cx', 1, [0, 1])
    expect(c.exportToQiskit()).toContain('QuantumCircuit')
    expect(c.exportToCirq()).toContain('cirq')
    expect(c.exportSVG()).toContain('<svg')
  })
})

describe('compat — every quantum-circuit gate is accepted', () => {
  const oneWire = 'id x y z h srn srndg r2 r4 r8 s t sdg tdg'.split(' ')
  const twoWire = 'cx cy cz ch csrn swap srswap iswap cr2 cr4 cr8 cs ct csdg ctdg'.split(' ')
  const threeWire = 'ccx cswap csrswap'.split(' ')
  const parameterised: [string, Record<string, number>][] = [
    ['rx', { theta: Math.PI }], ['ry', { theta: Math.PI }], ['rz', { phi: Math.PI }],
    ['u1', { lambda: Math.PI }], ['u2', { phi: 0, lambda: Math.PI }],
    ['u3', { theta: Math.PI, phi: 0, lambda: Math.PI }],
    ['gpi', { phi: 0 }], ['gpi2', { phi: 0 }], ['vz', { theta: Math.PI }],
  ]

  for (const g of oneWire) {
    it(`runs ${g}`, () => {
      const c = new QuantumCircuit(1); c.addGate(g, 0, 0); c.run()
      expect(Number.isFinite(c.probability(0))).toBe(true)
    })
  }
  for (const [g, params] of parameterised) {
    it(`runs ${g}`, () => {
      const c = new QuantumCircuit(1); c.addGate(g, 0, 0, { params }); c.run()
      expect(Number.isFinite(c.probability(0))).toBe(true)
    })
  }
  for (const g of twoWire) {
    it(`runs ${g}`, () => {
      const c = new QuantumCircuit(2); c.addGate('x', 0, 0); c.addGate(g, 1, [0, 1]); c.run()
      expect(Number.isFinite(c.probability(1))).toBe(true)
    })
  }
  for (const g of threeWire) {
    it(`runs ${g}`, () => {
      const c = new QuantumCircuit(3)
      c.addGate('x', 0, 0); c.addGate('x', 0, 1); c.addGate(g, 1, [0, 1, 2]); c.run()
      expect(Number.isFinite(c.probability(2))).toBe(true)
    })
  }

  it('applies ccx as a Toffoli', () => {
    const c = new QuantumCircuit(3)
    c.addGate('x', 0, 0); c.addGate('x', 0, 1); c.addGate('ccx', 1, [0, 1, 2])
    c.run()
    expect(c.probabilities()).toEqual([1, 1, 1])
  })

  it('applies barrier without changing the state', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0); c.addGate('barrier', 1, [0, 1]); c.addGate('cx', 2, [0, 1])
    c.run()
    expect(c.probabilities()).toEqual([0.5, 0.5])
  })

  it('resets a qubit', () => {
    const c = new QuantumCircuit(1)
    c.addGate('x', 0, 0); c.addGate('reset', 1, 0)
    c.run()
    expect(c.probability(0)).toBeCloseTo(0, 12)
  })
})

describe('compat — escape hatch into ket', () => {
  it('returns an equivalent immutable Circuit', () => {
    const c = new QuantumCircuit(2)
    c.addGate('h', 0, 0)
    c.addGate('cx', 1, [0, 1])
    const k = c.toKet()
    expect(k).toBeInstanceOf(Circuit)
    expect(k.exactProbs()['00']).toBeCloseTo(0.5, 12)
    expect(k.exactProbs()['11']).toBeCloseTo(0.5, 12)
  })

  it('reaches a backend the original package cannot', () => {
    // 200-qubit GHZ: 2^200 amplitudes, but Clifford-simulable in milliseconds
    const c = new QuantumCircuit(200)
    c.addGate('h', -1, 0)
    for (let i = 0; i < 199; i++) c.addGate('cx', -1, [i, i + 1])
    const d = c.toKet().simulate({ shots: 100 })
    expect(d.backend).toBe('clifford')
    expect(Object.keys(d.probs).sort()).toEqual(['0'.repeat(200), '1'.repeat(200)])
  })

  it('carries classical registers through to ket', () => {
    const c = new QuantumCircuit(1)
    c.addMeasure(0, 'c', 0)
    expect(c.toKet().toJSON().cregs).toEqual({ c: 1 })
  })
})
