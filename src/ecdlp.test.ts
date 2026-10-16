import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import {
  rippleAdd, rippleSub, addConst, subConst, ctrlAddConst,
  modAdd, modDouble, ctrlModAdd, modMul,
  costAdd, costModAdd, costModDouble, costCtrlModAdd, costModMul,
  ecdlpResources, compareEcdlp, renderEcdlp, ECDLP_REFERENCES,
  type ModWork,
} from './ecdlp.js'

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Hands out consecutive qubit indices, so a test reads as a register layout. */
function layout() {
  let next = 0
  return {
    reg(w: number): number[] { return Array.from({ length: w }, () => next++) },
    bit(): number { return next++ },
    get qubits(): number { return next },
  }
}

/** |value⟩ on the given register, qubit 0 = LSB. */
function load(c: Circuit, reg: readonly number[], value: bigint): Circuit {
  for (let i = 0; i < reg.length; i++) if ((value >> BigInt(i)) & 1n) c = c.x(reg[i]!)
  return c
}

/** The integer held by `reg` in a measured bitstring. */
function read(bs: string, reg: readonly number[]): bigint {
  let v = 0n
  for (let i = 0; i < reg.length; i++) if (bs[reg[i]!] === '1') v |= 1n << BigInt(i)
  return v
}

/** Run a permutation circuit once — the outcome is deterministic. */
function once(c: Circuit): string {
  return c.run({ shots: 1, seed: 1 }).most
}

/** Toffoli gates in a circuit. */
function toffolis(c: Circuit): number {
  return c.gateCounts().byName['ccx'] ?? 0
}

/** Every prime under 32, for modulus sweeps. */
const PRIMES = [5n, 7n, 11n, 13n]

// ── The adder ─────────────────────────────────────────────────────────────────

describe('ecdlp — ripple-carry adder', () => {
  it('adds every pair of 4-bit operands mod 2^4, restoring a and the carry ancilla', () => {
    const w = 4
    for (let av = 0; av < 16; av++) {
      for (let bv = 0; bv < 16; bv++) {
        const L = layout()
        const a = L.reg(w), b = L.reg(w), cin = L.bit()
        let c = new Circuit(L.qubits)
        c = load(c, a, BigInt(av))
        c = load(c, b, BigInt(bv))
        c = rippleAdd(c, a, b, cin)
        const bs = once(c)
        expect(read(bs, b)).toBe(BigInt((av + bv) % 16))
        expect(read(bs, a)).toBe(BigInt(av))
        expect(bs[cin]).toBe('0')
      }
    }
  })

  it('subtracts, wrapping into two-s complement when the result is negative', () => {
    const w = 4
    for (let av = 0; av < 16; av++) {
      for (let bv = 0; bv < 16; bv++) {
        const L = layout()
        const a = L.reg(w), b = L.reg(w), cin = L.bit()
        let c = new Circuit(L.qubits)
        c = load(c, a, BigInt(av))
        c = load(c, b, BigInt(bv))
        c = rippleSub(c, a, b, cin)
        const bs = once(c)
        expect(read(bs, b)).toBe(BigInt(((bv - av) % 16 + 16) % 16))
        expect(read(bs, a)).toBe(BigInt(av))
        expect(bs[cin]).toBe('0')
      }
    }
  })

  it('reads the sign of a subtraction off the top bit', () => {
    // The whole modular-reduction scheme rests on this: with both operands
    // below 2^(w-1), b - a is negative exactly when the top bit is set.
    const w = 5
    for (const [av, bv] of [[3, 9], [9, 3], [15, 15], [0, 1], [1, 0]] as const) {
      const L = layout()
      const a = L.reg(w), b = L.reg(w), cin = L.bit()
      let c = new Circuit(L.qubits)
      c = load(c, a, BigInt(av))
      c = load(c, b, BigInt(bv))
      c = rippleSub(c, a, b, cin)
      const bs = once(c)
      expect(bs[b[w - 1]!]).toBe(bv - av < 0 ? '1' : '0')
    }
  })
})

// ── Constant addition ─────────────────────────────────────────────────────────

describe('ecdlp — constant addition', () => {
  it('adds and subtracts a classical constant, returning the scratch register clean', () => {
    const w = 5
    for (const k of [0n, 1n, 13n, 31n]) {
      for (let bv = 0; bv < 32; bv += 3) {
        const L = layout()
        const b = L.reg(w), scratch = L.reg(w), cin = L.bit()
        let c = new Circuit(L.qubits)
        c = load(c, b, BigInt(bv))
        c = addConst(c, k, b, scratch, cin)
        let bs = once(c)
        expect(read(bs, b)).toBe((BigInt(bv) + k) % 32n)
        expect(read(bs, scratch)).toBe(0n)

        let d = new Circuit(L.qubits)
        d = load(d, b, BigInt(bv))
        d = subConst(d, k, b, scratch, cin)
        bs = once(d)
        expect(read(bs, b)).toBe(((BigInt(bv) - k) % 32n + 32n) % 32n)
        expect(read(bs, scratch)).toBe(0n)
      }
    }
  })

  it('adds the constant only when the control is set, at no extra Toffoli cost', () => {
    const w = 5
    for (const ctrlOn of [false, true]) {
      const L = layout()
      const b = L.reg(w), scratch = L.reg(w), cin = L.bit(), ctrl = L.bit()
      let c = new Circuit(L.qubits)
      c = load(c, b, 7n)
      if (ctrlOn) c = c.x(ctrl)
      c = ctrlAddConst(c, ctrl, 11n, b, scratch, cin)
      const bs = once(c)
      expect(read(bs, b)).toBe(ctrlOn ? 18n : 7n)
      expect(read(bs, scratch)).toBe(0n)
      expect(bs[ctrl]).toBe(ctrlOn ? '1' : '0')
    }
    // Controlling a classical constant is free: the scratch load becomes CNOTs.
    const L = layout()
    const b = L.reg(w), scratch = L.reg(w), cin = L.bit(), ctrl = L.bit()
    expect(toffolis(ctrlAddConst(new Circuit(L.qubits), ctrl, 11n, b, scratch, cin)))
      .toBe(toffolis(addConst(new Circuit(L.qubits), 11n, b, scratch, cin)))
  })
})

// ── Modular arithmetic ────────────────────────────────────────────────────────

/** Allocate the ancilla a modular routine needs, in one place. */
function work(L: ReturnType<typeof layout>, w: number): ModWork {
  return { scratch: L.reg(w), cin: L.bit(), flag: L.bit() }
}

describe('ecdlp — modular addition', () => {
  it.each(PRIMES)('computes (a + b) mod %s for every residue pair', p => {
    const n = p.toString(2).length
    const w = n + 1
    for (let av = 0; av < Number(p); av++) {
      for (let bv = 0; bv < Number(p); bv++) {
        const L = layout()
        const a = L.reg(w), b = L.reg(w), wk = work(L, w)
        let c = new Circuit(L.qubits)
        c = load(c, a, BigInt(av))
        c = load(c, b, BigInt(bv))
        c = modAdd(c, a, b, p, wk)
        const bs = once(c)
        expect(read(bs, b)).toBe(BigInt(av + bv) % p)
        expect(read(bs, a)).toBe(BigInt(av))
        // The flag is the part that is easy to get wrong: leave it set and it
        // stays entangled with the result, and the multiplier above breaks.
        expect(read(bs, wk.scratch)).toBe(0n)
        expect(bs[wk.cin]).toBe('0')
        expect(bs[wk.flag]).toBe('0')
      }
    }
  })
})

describe('ecdlp — modular doubling', () => {
  it.each(PRIMES)('computes 2a mod %s for every residue', p => {
    const w = p.toString(2).length + 1
    for (let av = 0; av < Number(p); av++) {
      const L = layout()
      const a = L.reg(w), wk = work(L, w)
      let c = new Circuit(L.qubits)
      c = load(c, a, BigInt(av))
      c = modDouble(c, a, p, wk)
      const bs = once(c)
      expect(read(bs, a)).toBe(BigInt(2 * av) % p)
      expect(read(bs, wk.scratch)).toBe(0n)
      expect(bs[wk.cin]).toBe('0')
      expect(bs[wk.flag]).toBe('0')
    }
  })

  it('refuses an even modulus, whose reduction flag it cannot uncompute', () => {
    const L = layout()
    const a = L.reg(4), wk = work(L, 4)
    expect(() => modDouble(new Circuit(L.qubits), a, 12n, wk)).toThrow(/odd/)
  })
})

describe('ecdlp — controlled modular addition', () => {
  it('adds only when the control is set', () => {
    const p = 11n, w = 5
    for (const ctrlOn of [false, true]) {
      const L = layout()
      const a = L.reg(w), copy = L.reg(w), b = L.reg(w), ctrl = L.bit(), wk = work(L, w)
      let c = new Circuit(L.qubits)
      c = load(c, a, 7n)
      c = load(c, b, 9n)
      if (ctrlOn) c = c.x(ctrl)
      c = ctrlModAdd(c, ctrl, a, copy, b, p, wk)
      const bs = once(c)
      expect(read(bs, b)).toBe(ctrlOn ? (7n + 9n) % p : 9n)
      expect(read(bs, a)).toBe(7n)
      expect(read(bs, copy)).toBe(0n)
      expect(bs[wk.flag]).toBe('0')
    }
  })
})

describe('ecdlp — modular multiplication', () => {
  it('computes x * y mod 7 for every pair, restoring both operands', () => {
    const p = 7n, n = 3, w = 4
    for (let xv = 0; xv < Number(p); xv++) {
      for (let yv = 0; yv < Number(p); yv++) {
        const L = layout()
        const x = L.reg(n), y = L.reg(w), copy = L.reg(w), res = L.reg(w), wk = work(L, w)
        let c = new Circuit(L.qubits)
        c = load(c, x, BigInt(xv))
        c = load(c, y, BigInt(yv))
        c = modMul(c, x, y, copy, res, p, wk)
        const bs = once(c)
        expect(read(bs, res)).toBe(BigInt(xv * yv) % p)
        expect(read(bs, x)).toBe(BigInt(xv))
        expect(read(bs, y)).toBe(BigInt(yv))
        expect(read(bs, copy)).toBe(0n)
        expect(read(bs, wk.scratch)).toBe(0n)
        expect(bs[wk.cin]).toBe('0')
        expect(bs[wk.flag]).toBe('0')
      }
    }
  })

  it('holds at 4 bits, on a 26-qubit register file', () => {
    // Wider than a dense statevector reaches. Every routine here is a
    // permutation, so the state stays a single amplitude and the sparse
    // backend runs it in constant memory — the property that lets the
    // validated layer sit as high as it does.
    const p = 13n, n = 4, w = 5
    for (const [xv, yv] of [[5, 7], [12, 12], [1, 11], [0, 9], [11, 6]] as const) {
      const L = layout()
      const x = L.reg(n), y = L.reg(w), copy = L.reg(w), res = L.reg(w), wk = work(L, w)
      expect(L.qubits).toBe(26)
      let c = new Circuit(L.qubits)
      c = load(c, x, BigInt(xv))
      c = load(c, y, BigInt(yv))
      c = modMul(c, x, y, copy, res, p, wk)
      const bs = once(c)
      expect(read(bs, res)).toBe(BigInt(xv * yv) % p)
      expect(read(bs, x)).toBe(BigInt(xv))
      expect(read(bs, y)).toBe(BigInt(yv))
      expect(read(bs, copy)).toBe(0n)
    }
  })
})

// ── The closed forms the model extrapolates ───────────────────────────────────

describe('ecdlp — closed forms match the circuits they stand for', () => {
  const widths = [3, 4, 5, 6, 7, 8, 9, 10]

  it.each(widths)('adder at width %i', w => {
    const L = layout()
    const a = L.reg(w), b = L.reg(w), cin = L.bit()
    expect(toffolis(rippleAdd(new Circuit(L.qubits), a, b, cin))).toBe(costAdd(w))
    expect(toffolis(rippleSub(new Circuit(L.qubits), a, b, cin))).toBe(costAdd(w))
  })

  it.each(widths)('modular addition at width %i', w => {
    const L = layout()
    const a = L.reg(w), b = L.reg(w), wk = work(L, w)
    expect(toffolis(modAdd(new Circuit(L.qubits), a, b, 5n, wk))).toBe(costModAdd(w))
  })

  it.each(widths)('modular doubling at width %i', w => {
    const L = layout()
    const a = L.reg(w), wk = work(L, w)
    expect(toffolis(modDouble(new Circuit(L.qubits), a, 5n, wk))).toBe(costModDouble(w))
  })

  it.each(widths)('controlled modular addition at width %i', w => {
    const L = layout()
    const a = L.reg(w), copy = L.reg(w), b = L.reg(w), ctrl = L.bit(), wk = work(L, w)
    expect(toffolis(ctrlModAdd(new Circuit(L.qubits), ctrl, a, copy, b, 5n, wk))).toBe(costCtrlModAdd(w))
  })

  it.each([3, 4, 5, 6, 7, 8])('modular multiplication at %i bits', n => {
    const w = n + 1
    const L = layout()
    const x = L.reg(n), y = L.reg(w), copy = L.reg(w), res = L.reg(w), wk = work(L, w)
    expect(toffolis(modMul(new Circuit(L.qubits), x, y, copy, res, 5n, wk))).toBe(costModMul(n, w))
  })
})

// ── The estimate ──────────────────────────────────────────────────────────────

describe('ecdlp — resource estimate', () => {
  it('reproduces the published logical qubit count to within a percent', () => {
    // Roetteler et al. give 9n + 2⌈log₂n⌉ + 10. This model counts its own
    // registers and arrives independently at 9w + 12. Agreement is a real
    // cross-check on the register accounting, which nothing else here tests.
    const rnsl = ECDLP_REFERENCES.find(r => r.label === 'Roetteler et al.')!
    const { logicalQubits } = ecdlpResources(256)
    expect(Math.abs(logicalQubits / rnsl.logicalQubits! - 1)).toBeLessThan(0.01)
  })

  it('scales as n³ — 2n point additions of n² arithmetic each', () => {
    const a = ecdlpResources(128).toffoli
    const b = ecdlpResources(256).toffoli
    expect(b / a).toBeGreaterThan(7.5)
    expect(b / a).toBeLessThan(8.5)
  })

  it('is dominated by inversion and multiplication, not by addition', () => {
    const { breakdown, perPointAddition } = ecdlpResources(256)
    const byName = Object.fromEntries(breakdown.map(t => [t.name, t.total]))
    expect(byName['modular addition']! / perPointAddition).toBeLessThan(0.005)
    expect((byName['modular inversion']! + byName['modular multiplication']!) / perPointAddition)
      .toBeGreaterThan(0.99)
  })

  it('brackets the answer even if the modelled inversion term is badly wrong', () => {
    const { toffoli, sensitivity } = ecdlpResources(256)
    expect(sensitivity.inversionFree).toBeLessThan(toffoli)
    expect(sensitivity.inversionTenfold).toBeGreaterThan(toffoli)
    // The published claims sit far below the model at both ends of the range,
    // so no reading of the soft term turns the gap into agreement.
    const ionq = ECDLP_REFERENCES.find(r => r.label === 'IonQ')!.toffoli!
    expect(sensitivity.inversionFree / ionq).toBeGreaterThan(10)
  })

  it('lands below the published upper bound, as a bound requires', () => {
    const rnsl = ECDLP_REFERENCES.find(r => r.label === 'Roetteler et al.')!
    expect(ecdlpResources(256).toffoli).toBeLessThan(rnsl.toffoli!)
  })

  it('rejects widths it cannot represent', () => {
    expect(() => ecdlpResources(1)).toThrow(/integer/)
    expect(() => ecdlpResources(2.5)).toThrow(/integer/)
  })

  it('reports a ratio against every reference that quotes a comparable number', () => {
    const { gaps } = compareEcdlp(256)
    expect(gaps).toHaveLength(ECDLP_REFERENCES.length)
    for (const g of gaps) {
      expect(g.toffoliRatio).not.toBeNull()
      expect(g.toffoliRatio!).toBeGreaterThan(0)
    }
    expect(gaps.find(g => g.reference.label === 'Roetteler et al.')!.toffoliRatio!).toBeLessThan(1)
    expect(gaps.find(g => g.reference.label === 'IonQ')!.toffoliRatio!).toBeGreaterThan(1)
  })

  it('renders a report that says what it does not cover', () => {
    const out = renderEcdlp(256)
    expect(out).toContain('256-bit prime field')
    expect(out).toContain('modular inversion')
    expect(out).toContain('IonQ')
    expect(out).toMatch(/no physical qubit count and no runtime/)
  })
})
