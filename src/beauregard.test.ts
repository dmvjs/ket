import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { phiAdd, cPhiAdd, ccPhiAdd, phiAddMod, ccPhiAddMod, cMultModAdd, beauregardU,
         modPow, modInverse, gcd, continuedFractions,
         applyQft, applyIqft } from './beauregard.js'

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Run a circuit once, return the most probable bitstring. */
function mostLikely(c: Circuit, seed = 1): string {
  return c.run({ shots: 4096, seed }).most
}

/**
 * Build state |x⟩ on n qubits (qubit 0 = LSB).
 * Sets qubit j if bit j of x is 1.
 */
function setState(c: Circuit, x: bigint, offset: number, n: number): Circuit {
  for (let j = 0; j < n; j++) {
    if ((x >> BigInt(j)) & 1n) c = c.x(offset + j)
  }
  return c
}

/** Read the integer value from the most-likely measurement of qubits [offset..offset+n-1]. */
function readInt(bs: string, offset: number, n: number): bigint {
  let val = 0n
  for (let j = 0; j < n; j++) {
    if (bs[offset + j] === '1') val |= 1n << BigInt(j)
  }
  return val
}

// ── Classical arithmetic ──────────────────────────────────────────────────────

describe('beauregard — classical arithmetic', () => {
  it('modPow: 2^10 mod 15 = 4', () => {
    expect(modPow(2n, 10n, 15n)).toBe(4n)
  })

  it('modPow: 7^4 mod 15 = 1 (period 4)', () => {
    expect(modPow(7n, 4n, 15n)).toBe(1n)
  })

  it('modPow: a^0 = 1 for any a', () => {
    expect(modPow(13n, 0n, 21n)).toBe(1n)
  })

  it('modInverse: 7⁻¹ mod 15 = 13 (7·13=91=6·15+1)', () => {
    expect(modInverse(7n, 15n)).toBe(13n)
    expect(7n * 13n % 15n).toBe(1n)
  })

  it('modInverse: 2⁻¹ mod 15 = 8', () => {
    expect(modInverse(2n, 15n)).toBe(8n)
    expect(2n * 8n % 15n).toBe(1n)
  })

  it('gcd: gcd(12, 8) = 4', () => {
    expect(gcd(12n, 8n)).toBe(4n)
  })

  it('gcd: gcd(7, 15) = 1 (coprime)', () => {
    expect(gcd(7n, 15n)).toBe(1n)
  })

  it('continuedFractions: finds period 4 for 15 from measured phase s/r ≈ 1/4', () => {
    // s=1, r=4: phase = 1/4 = 0.25; with 8 precision bits, measured = round(0.25 × 256) = 64
    const r = continuedFractions(64, 8, 15n)
    expect([2n, 4n]).toContain(r)  // 4 or a divisor of 4
  })

  it('continuedFractions: handles measured=0 gracefully', () => {
    expect(continuedFractions(0, 8, 15n)).toBe(1n)
  })
})

// ── QFT helpers ───────────────────────────────────────────────────────────────

describe('beauregard — applyQft / applyIqft at offset', () => {
  it('QFT then IQFT is identity on |x=5⟩, 4 qubits, offset=0', () => {
    let c = new Circuit(4)
    c = setState(c, 5n, 0, 4)  // |0101⟩ = 5
    c = applyQft(c, 4, 0)
    c = applyIqft(c, 4, 0)
    expect(readInt(mostLikely(c), 0, 4)).toBe(5n)
  })

  it('QFT then IQFT at offset=2 leaves other qubits untouched', () => {
    let c = new Circuit(6)
    c = setState(c, 3n, 0, 2)  // qubits 0-1 = |11⟩
    c = setState(c, 5n, 2, 4)  // qubits 2-5 = 5
    c = applyQft(c, 4, 2)
    c = applyIqft(c, 4, 2)
    const bs = mostLikely(c)
    expect(readInt(bs, 0, 2)).toBe(3n)  // untouched
    expect(readInt(bs, 2, 4)).toBe(5n)  // restored
  })
})

// ── Draper phase adder ────────────────────────────────────────────────────────

describe('phiAdd — Draper phase adder', () => {
  /**
   * Test: |x⟩ -QFT-> phiAdd(a) -IQFT-> |x+a mod 2^n⟩
   */
  function testAdd(n: number, x: bigint, a: bigint): void {
    let c = new Circuit(n)
    c = setState(c, x, 0, n)
    c = applyQft(c, n, 0)
    c = phiAdd(c, n, a)
    c = applyIqft(c, n, 0)
    const mod = 1n << BigInt(n)
    const expected = ((x + a) % mod + mod) % mod
    expect(readInt(mostLikely(c), 0, n)).toBe(expected)
  }

  it('3+4=7 on 4 qubits', () => testAdd(4, 3n, 4n))
  it('5+3=0 mod 8 on 3 qubits (overflow wraps)', () => testAdd(3, 5n, 3n))
  it('0+7=7 on 4 qubits', () => testAdd(4, 0n, 7n))
  it('15+1=0 mod 16 on 4 qubits', () => testAdd(4, 15n, 1n))
  it('subtract: 7-3=4 via negative a', () => testAdd(4, 7n, -3n))
  it('subtract with wrap: 2-5 = 13 mod 16 on 4 qubits', () => testAdd(4, 2n, -5n))
  it('large n=8: 200+55=255', () => testAdd(8, 200n, 55n))
  it('large n=8: 200+56=0 mod 256 (overflow)', () => testAdd(8, 200n, 56n))
})

describe('cPhiAdd — controlled Draper adder', () => {
  /**
   * Circuit: ctrl | n-qubit register
   * When ctrl=1: register → x+a mod 2^n
   * When ctrl=0: register unchanged
   */
  function testCAdd(n: number, x: bigint, a: bigint, ctrlVal: 0 | 1): bigint {
    const total = 1 + n
    let c = new Circuit(total)
    if (ctrlVal === 1) c = c.x(0)   // ctrl qubit = qubit 0
    c = setState(c, x, 1, n)         // register = qubits 1..n
    c = applyQft(c, n, 1)
    c = cPhiAdd(c, n, a, 0, 1)      // ctrl=0, offset=1
    c = applyIqft(c, n, 1)
    return readInt(mostLikely(c), 1, n)
  }

  it('ctrl=1: 3+4=7', () => {
    expect(testCAdd(4, 3n, 4n, 1)).toBe(7n)
  })

  it('ctrl=0: register unchanged (3 stays 3)', () => {
    expect(testCAdd(4, 3n, 4n, 0)).toBe(3n)
  })

  it('ctrl=1 with wrap: 13+5=2 mod 16', () => {
    expect(testCAdd(4, 13n, 5n, 1)).toBe(2n)
  })
})

describe('ccPhiAdd — doubly-controlled Draper adder', () => {
  /**
   * Circuit: ctrl1 | ctrl2 | n-qubit register
   * Adds only when both controls are |1⟩.
   */
  function testCCAdd(n: number, x: bigint, a: bigint, c1: 0|1, c2: 0|1): bigint {
    const total = 2 + n
    let c = new Circuit(total)
    if (c1) c = c.x(0)
    if (c2) c = c.x(1)
    c = setState(c, x, 2, n)
    c = applyQft(c, n, 2)
    c = ccPhiAdd(c, n, a, 0, 1, 2)
    c = applyIqft(c, n, 2)
    return readInt(mostLikely(c), 2, n)
  }

  it('c1=1,c2=1: 3+4=7', () => expect(testCCAdd(4, 3n, 4n, 1, 1)).toBe(7n))
  it('c1=1,c2=0: unchanged',  () => expect(testCCAdd(4, 3n, 4n, 1, 0)).toBe(3n))
  it('c1=0,c2=1: unchanged',  () => expect(testCCAdd(4, 3n, 4n, 0, 1)).toBe(3n))
  it('c1=0,c2=0: unchanged',  () => expect(testCCAdd(4, 3n, 4n, 0, 0)).toBe(3n))
  it('c1=1,c2=1 with wrap: 15+1=0 mod 16', () => expect(testCCAdd(4, 15n, 1n, 1, 1)).toBe(0n))
})

// ── Modular adder ─────────────────────────────────────────────────────────────

describe('phiAddMod — modular adder', () => {
  /**
   * Circuit: (n+1)-qubit b register | 1 ancilla
   * b (n+1 qubits) starts at x, ancilla starts at |0⟩.
   * After phiAddMod: b = x+a mod N.
   * Ancilla must return to |0⟩.
   */
  function testAddMod(n: number, x: bigint, a: bigint, N: bigint): { result: bigint; ancilla: bigint } {
    const nb = n + 1
    const total = nb + 1  // b register + ancilla
    let c = new Circuit(total)
    c = setState(c, x, 0, nb)      // b = x (using n+1 qubits)
    c = applyQft(c, nb, 0)
    c = phiAddMod(c, n, a, N, 0, nb)  // ancilla at qubit nb
    c = applyIqft(c, nb, 0)
    const bs = mostLikely(c)
    return {
      result:   readInt(bs, 0, nb),
      ancilla:  readInt(bs, nb, 1),
    }
  }

  it('3+4 mod 7 = 0', () => {
    const r = testAddMod(3, 3n, 4n, 7n)
    expect(r.result).toBe(0n)
    expect(r.ancilla).toBe(0n)  // ancilla restored
  })

  it('2+3 mod 7 = 5', () => {
    const r = testAddMod(3, 2n, 3n, 7n)
    expect(r.result).toBe(5n)
    expect(r.ancilla).toBe(0n)
  })

  it('6+6 mod 7 = 5', () => {
    const r = testAddMod(3, 6n, 6n, 7n)
    expect(r.result).toBe(5n)
    expect(r.ancilla).toBe(0n)
  })

  it('0+0 mod 15 = 0', () => {
    const r = testAddMod(4, 0n, 0n, 15n)
    expect(r.result).toBe(0n)
    expect(r.ancilla).toBe(0n)
  })

  it('7+8 mod 15 = 0', () => {
    const r = testAddMod(4, 7n, 8n, 15n)
    expect(r.result).toBe(0n)
    expect(r.ancilla).toBe(0n)
  })

  it('14+14 mod 15 = 13', () => {
    const r = testAddMod(4, 14n, 14n, 15n)
    expect(r.result).toBe(13n)
    expect(r.ancilla).toBe(0n)
  })
})

describe('ccPhiAddMod — doubly-controlled modular adder', () => {
  function testCCAddMod(n: number, x: bigint, a: bigint, N: bigint, c1: 0|1, c2: 0|1): { result: bigint; ancilla: bigint } {
    const nb = n + 1
    const total = 2 + nb + 1  // ctrl1 | ctrl2 | b(n+1) | ancilla
    let c = new Circuit(total)
    if (c1) c = c.x(0)
    if (c2) c = c.x(1)
    c = setState(c, x, 2, nb)
    c = applyQft(c, nb, 2)
    c = ccPhiAddMod(c, n, a, N, 0, 1, 2, 2 + nb)
    c = applyIqft(c, nb, 2)
    const bs = mostLikely(c)
    return { result: readInt(bs, 2, nb), ancilla: readInt(bs, 2 + nb, 1) }
  }

  it('c1=1,c2=1: 7+8 mod 15 = 0', () => {
    const r = testCCAddMod(4, 7n, 8n, 15n, 1, 1)
    expect(r.result).toBe(0n)
    expect(r.ancilla).toBe(0n)
  })

  it('c1=1,c2=0: b unchanged (7 stays 7)', () => {
    const r = testCCAddMod(4, 7n, 8n, 15n, 1, 0)
    expect(r.result).toBe(7n)
    expect(r.ancilla).toBe(0n)
  })

  it('c1=0,c2=1: b unchanged', () => {
    const r = testCCAddMod(4, 7n, 8n, 15n, 0, 1)
    expect(r.result).toBe(7n)
    expect(r.ancilla).toBe(0n)
  })

  it('c1=0,c2=0: b unchanged', () => {
    const r = testCCAddMod(4, 7n, 8n, 15n, 0, 0)
    expect(r.result).toBe(7n)
    expect(r.ancilla).toBe(0n)
  })
})

// ── Modular multiplier ────────────────────────────────────────────────────────

describe('cMultModAdd — controlled modular multiply-add', () => {
  /**
   * Layout: ctrl | x[0..n-1] | b[0..n] | ancilla
   * When ctrl=1: b → b + a·x mod N
   * When ctrl=0: b unchanged
   */
  function testMultAdd(n: number, x: bigint, b0: bigint, a: bigint, N: bigint, ctrlVal: 0|1): bigint {
    const nb = n + 1
    const total = 1 + n + nb + 1
    let c = new Circuit(total)
    if (ctrlVal) c = c.x(0)           // ctrl
    c = setState(c, x,  1,     n)     // x register
    c = setState(c, b0, 1 + n, nb)    // b register
    c = cMultModAdd(c, n, a, N, 0, 1, 1 + n, 1 + n + nb)
    const bs = mostLikely(c)
    return readInt(bs, 1 + n, nb)
  }

  it('ctrl=1: b=0, x=1, a=7, N=15 → b=7', () => {
    expect(testMultAdd(4, 1n, 0n, 7n, 15n, 1)).toBe(7n)
  })

  it('ctrl=1: b=0, x=2, a=7, N=15 → b=14', () => {
    expect(testMultAdd(4, 2n, 0n, 7n, 15n, 1)).toBe(14n)
  })

  it('ctrl=1: b=0, x=3, a=7, N=15 → b=6 (21 mod 15)', () => {
    expect(testMultAdd(4, 3n, 0n, 7n, 15n, 1)).toBe(6n)
  })

  it('ctrl=0: b unchanged when ctrl=0', () => {
    expect(testMultAdd(4, 3n, 0n, 7n, 15n, 0)).toBe(0n)
  })

  it('ctrl=1: b=0, x=4, a=2, N=15 → b=8', () => {
    expect(testMultAdd(4, 4n, 0n, 2n, 15n, 1)).toBe(8n)
  })
})

describe('beauregardU — controlled modular multiplier U_a', () => {
  /**
   * Layout: ctrl | x[0..n-1] | b[0..n] | ancilla
   * When ctrl=1: x → ax mod N (b must start at |0⟩, returns to |0⟩)
   * When ctrl=0: x unchanged
   */
  function testU(n: number, x: bigint, a: bigint, N: bigint, ctrlVal: 0|1): bigint {
    const nb = n + 1
    const total = 1 + n + nb + 1
    let c = new Circuit(total)
    if (ctrlVal) c = c.x(0)
    c = setState(c, x, 1, n)
    const aInv = modInverse(a, N)
    c = beauregardU(c, n, a, aInv, N, 0, 1, 1 + n, 1 + n + nb)
    const bs = mostLikely(c)
    return readInt(bs, 1, n)
  }

  it('ctrl=1: 1×7 mod 15 = 7', () => {
    expect(testU(4, 1n, 7n, 15n, 1)).toBe(7n)
  })

  it('ctrl=1: 7×2 mod 15 = 14', () => {
    expect(testU(4, 7n, 2n, 15n, 1)).toBe(14n)
  })

  it('ctrl=1: 7×7 mod 15 = 4 (49 mod 15)', () => {
    expect(testU(4, 7n, 7n, 15n, 1)).toBe(4n)
  })

  it('ctrl=1: 1×2 mod 21 = 2', () => {
    expect(testU(5, 1n, 2n, 21n, 1)).toBe(2n)
  })

  it('ctrl=1: applying U_2 four times gives 16 mod 15 = 1 (period=4)', () => {
    // x: 1 → 2 → 4 → 8 → 16≡1 mod 15
    // Instead of chaining, verify each step:
    expect(testU(4, 1n,  2n, 15n, 1)).toBe(2n)
    expect(testU(4, 2n,  2n, 15n, 1)).toBe(4n)
    expect(testU(4, 4n,  2n, 15n, 1)).toBe(8n)
    expect(testU(4, 8n,  2n, 15n, 1)).toBe(1n)
  })

  it('ctrl=0: x unchanged', () => {
    expect(testU(4, 7n, 7n, 15n, 0)).toBe(7n)
  })

  it('b register returns to |0⟩ after ctrl=1', () => {
    const n = 4, nb = n + 1, a = 7n, N = 15n
    const total = 1 + n + nb + 1
    let c = new Circuit(total)
    c = c.x(0)  // ctrl=1
    c = setState(c, 1n, 1, n)
    const aInv = modInverse(a, N)
    c = beauregardU(c, n, a, aInv, N, 0, 1, 1 + n, 1 + n + nb)
    const bs = mostLikely(c)
    // b register qubits 1+n..1+2n should all be 0
    expect(readInt(bs, 1 + n, nb)).toBe(0n)
  })
})

// ── Entanglement profiling ────────────────────────────────────────────────────
//
// Empirical MPS bond dimension scaling for the full Beauregard QPE circuit.
//
// Empirical results (measured, not assumed):
//   n=4 (N=15,  a=7):  peakChi =   4  (~0.4s)
//   n=5 (N=21,  a=2):  peakChi =  27  (~2.7s)
//   n=6 (N=35,  a=3):  peakChi =  44  (~29s)
//   n=7 (N=77,  a=2):  peakChi = 143  (~907s)
//
// Conclusion: chi grows super-linearly with n (not bounded by a small polynomial).
// The full QPE circuit generates significant intermediate entanglement from the
// controlled-U_a structure; the workspace (b-register) entangles with the counting
// register during each U_a even after the multiplier disentangles it at the end.
// Exact MPS simulation is therefore not efficient for large N.  For approximate
// simulation with a χ cap, accuracy degrades as χ_actual > χ_cap.
//
// The three small-N tests below serve as regression guards for the gate-level
// correctness of the circuit rather than MPS efficiency claims.

describe('beauregard — MPS bond dimension scaling', () => {
  /** Run QPE for factoring N with base a, return peakChi. */
  function measurePeakChi(N: bigint, a: bigint, maxBond = 512): number {
    const n         = Math.ceil(Math.log2(Number(N)))
    const precision = 2 * n + 1
    const xOff      = precision
    const bOff      = precision + n
    const anc       = precision + 2 * n + 1
    const totalQ    = anc + 1

    let c = new Circuit(totalQ)
    for (let k = 0; k < precision; k++) c = c.h(k)
    c = c.x(xOff)

    let ak = a % N
    for (let k = 0; k < precision; k++) {
      const akInv = modInverse(ak, N)
      c = beauregardU(c, n, ak, akInv, N, k, xOff, bOff, anc)
      ak = ak * ak % N
    }
    c = applyIqft(c, precision, 0)

    const dist = c.runMps({ shots: 1, seed: 1, maxBond })
    return dist.peakChi!
  }

  it('N=15 (4-bit): peakChi=4', () => {
    const chi = measurePeakChi(15n, 7n)
    console.log(`N=15, a=7, peakChi=${chi}`)
    expect(chi).toBe(4)
  })

  it('N=21 (5-bit): peakChi=27', () => {
    const chi = measurePeakChi(21n, 2n)
    console.log(`N=21, a=2, peakChi=${chi}`)
    expect(chi).toBe(27)
  })

  it('N=35 (6-bit): peakChi=44', () => {
    const chi = measurePeakChi(35n, 3n)
    console.log(`N=35, a=3, peakChi=${chi}`)
    expect(chi).toBe(44)
  })

  // N=77 peakChi=143 measured (907s); excluded from CI — run scripts/measure-chi.ts manually.
})
