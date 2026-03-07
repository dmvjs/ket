/**
 * Immutable circuit builder with IonQ QIS gate names.
 *
 * Each gate method returns a new Circuit — circuits are values, not mutated objects.
 * Simulation is lazy: nothing runs until `.run()` is called.
 */

import * as G from './gates.js'
import { applyCNOT, applyControlled, applyCSwap, applySingle, applySWAP, applyToffoli, applyTwo, Gate2x2, Gate4x4, probabilities, StateVector, zero } from './statevector.js'
import { Complex, ZERO } from './complex.js'

// ─── Operation types ─────────────────────────────────────────────────────────

type SingleOp     = { kind: 'single';     q: number;                      gate: Gate2x2 }
type CNOTOp       = { kind: 'cnot';       control: number; target: number               }
type SWAPOp       = { kind: 'swap';       a: number;       b: number                    }
type TwoOp        = { kind: 'two';        a: number;       b: number;    gate: Gate4x4  }
type ControlledOp = { kind: 'controlled'; control: number; target: number; gate: Gate2x2 }
type ToffoliOp    = { kind: 'toffoli';    c1: number; c2: number; target: number }
type CSwapOp      = { kind: 'cswap';      control: number; a: number; b: number }
type MeasureOp = { kind: 'measure'; q: number; creg: string; bit: number }
type ResetOp   = { kind: 'reset';   q: number }
type IfOp      = { kind: 'if';      creg: string; value: number; ops: readonly Op[] }
type Op = SingleOp | CNOTOp | SWAPOp | TwoOp | ControlledOp | ToffoliOp | CSwapOp | MeasureOp | ResetOp | IfOp

// ─── Mid-circuit measurement helpers ─────────────────────────────────────────

/**
 * Project the statevector onto the given qubit outcome and renormalize.
 * rand: a uniform random number in [0, 1) used to sample the outcome.
 */
function collapseQubit(sv: StateVector, q: number, rand: number): { outcome: 0 | 1; sv: StateVector } {
  const mask = 1n << BigInt(q)
  let p1 = 0
  for (const [idx, amp] of sv) {
    if ((idx & mask) !== 0n) p1 += amp.re * amp.re + amp.im * amp.im
  }
  const outcome: 0 | 1 = rand < p1 ? 1 : 0
  const invNorm = 1 / Math.sqrt(outcome === 1 ? p1 : 1 - p1)
  const next: StateVector = new Map()
  for (const [idx, amp] of sv) {
    if (outcome === 1 ? (idx & mask) !== 0n : (idx & mask) === 0n) {
      next.set(idx, { re: amp.re * invNorm, im: amp.im * invNorm })
    }
  }
  return { outcome, sv: next }
}

/** Sample one basis-state index from a statevector using a uniform random number. */
function sampleSV(sv: StateVector, rand: number): bigint {
  const sorted = sv.entries().toArray().toSorted(([a], [b]) => (a < b ? -1 : 1))
  let cum = 0
  for (const [idx, amp] of sorted) {
    cum += amp.re * amp.re + amp.im * amp.im
    if (rand <= cum) return idx
  }
  return sorted.at(-1)?.[0] ?? 0n
}

/** Simulate a pure (no measure/reset/if) circuit and return the statevector. */
function simulatePure(ops: readonly Op[], qubits: number): StateVector {
  let sv: StateVector = zero(qubits)
  for (const op of ops) {
    if      (op.kind === 'single')     sv = applySingle(sv, op.q, op.gate)
    else if (op.kind === 'cnot')       sv = applyCNOT(sv, op.control, op.target)
    else if (op.kind === 'controlled') sv = applyControlled(sv, op.control, op.target, op.gate)
    else if (op.kind === 'swap')       sv = applySWAP(sv, op.a, op.b)
    else if (op.kind === 'toffoli')    sv = applyToffoli(sv, op.c1, op.c2, op.target)
    else if (op.kind === 'cswap')      sv = applyCSwap(sv, op.control, op.a, op.b)
    else if (op.kind === 'two')        sv = applyTwo(sv, op.a, op.b, op.gate)
  }
  return sv
}

/** Read a classical register as a little-endian integer (bit 0 = LSB). */
function cregValue(shotCregs: Map<string, boolean[]>, name: string): number {
  return (shotCregs.get(name) ?? []).reduce((acc, b, i) => acc | (b ? 1 << i : 0), 0)
}

/** Apply ops to `sv`, handling mid-circuit measurement with `rng`. Recursive for IfOp. */
function applyOps(ops: readonly Op[], svIn: StateVector, shotCregs: Map<string, boolean[]>, rng: () => number): StateVector {
  let sv = svIn
  for (const op of ops) {
    if      (op.kind === 'single')     sv = applySingle(sv, op.q, op.gate)
    else if (op.kind === 'cnot')       sv = applyCNOT(sv, op.control, op.target)
    else if (op.kind === 'controlled') sv = applyControlled(sv, op.control, op.target, op.gate)
    else if (op.kind === 'swap')       sv = applySWAP(sv, op.a, op.b)
    else if (op.kind === 'toffoli')    sv = applyToffoli(sv, op.c1, op.c2, op.target)
    else if (op.kind === 'cswap')      sv = applyCSwap(sv, op.control, op.a, op.b)
    else if (op.kind === 'two')        sv = applyTwo(sv, op.a, op.b, op.gate)
    else if (op.kind === 'measure') {
      const { outcome, sv: next } = collapseQubit(sv, op.q, rng())
      sv = next
      const reg = shotCregs.get(op.creg)
      if (reg) reg[op.bit] = outcome === 1
    } else if (op.kind === 'reset') {
      const { outcome, sv: next } = collapseQubit(sv, op.q, rng())
      sv = next
      if (outcome === 1) sv = applySingle(sv, op.q, G.X)
    } else {  // if
      if (cregValue(shotCregs, op.creg) === op.value) sv = applyOps(op.ops, sv, shotCregs, rng)
    }
  }
  return sv
}

// ─── Distribution ─────────────────────────────────────────────────────────────

/** Seeded xorshift32 PRNG — same algorithm used by qsim for reproducibility. */
function makePrng(seed?: number): () => number {
  let s = seed !== undefined ? ((seed >>> 0) || 1) : ((Date.now() & 0xffffffff) >>> 0) || 1
  return () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5
    return (s >>> 0) / 0x100000000
  }
}

export interface RunOptions {
  shots?: number
  seed?: number
}

/**
 * Measurement result — the output of running a circuit.
 *
 * `probs` keys are IonQ bitstrings: q0 is the rightmost character.
 * `histogram` keys are decimal integers (IonQ API convention).
 */
export class Distribution {
  readonly qubits: number
  readonly shots:  number
  readonly probs:  Readonly<Record<string, number>>
  readonly histogram: Readonly<Record<string, number>>
  /** Classical register results: `cregs[name][bit]` = fraction of shots where that bit was 1. */
  readonly cregs: Readonly<Record<string, readonly number[]>>

  constructor(
    qubits: number,
    shots: number,
    counts: Map<bigint, number>,
    cregCounts: Map<string, number[]> = new Map(),
  ) {
    this.qubits = qubits
    this.shots  = shots

    const probs: Record<string, number>     = {}
    const histogram: Record<string, number> = {}

    for (const [idx, count] of counts) {
      const prob      = count / shots
      const bitstring = idx.toString(2).padStart(qubits, '0')
      probs[bitstring]         = prob
      histogram[String(idx)]   = prob
    }

    this.probs     = Object.freeze(probs)
    this.histogram = Object.freeze(histogram)

    const cregs: Record<string, readonly number[]> = {}
    for (const [name, bitCounts] of cregCounts) {
      cregs[name] = Object.freeze(bitCounts.map(c => c / shots))
    }
    this.cregs = Object.freeze(cregs)
  }

  /** Most probable bitstring. */
  get most(): string {
    let best = '', bestP = -1
    for (const [bs, p] of Object.entries(this.probs)) {
      if (p > bestP) { best = bs; bestP = p }
    }
    return best
  }

  /** Shannon entropy of the distribution (in bits). */
  get entropy(): number {
    let h = 0
    for (const p of Object.values(this.probs)) {
      if (p > 0) h -= p * Math.log2(p)
    }
    return h
  }

  /** ASCII bar chart of measurement outcomes. */
  render(): string {
    const entries = Object.entries(this.probs).toSorted(([a], [b]) => a.localeCompare(b))
    const maxP    = Math.max(...entries.map(([, p]) => p))
    const width   = 40
    const lines   = entries.map(([bs, p]) => {
      const bar   = '█'.repeat(Math.round((p / maxP) * width))
      const pct   = (p * 100).toFixed(1).padStart(5)
      return `|${bs}⟩ ${pct}%  ${bar}`
    })
    return lines.join('\n')
  }
}

// ─── Circuit ──────────────────────────────────────────────────────────────────

export class Circuit {
  readonly qubits: number
  readonly #ops:   readonly Op[]
  readonly #cregs: ReadonlyMap<string, number>  // name → declared size

  constructor(qubits: number, ops: readonly Op[] = [], cregs: ReadonlyMap<string, number> = new Map()) {
    this.qubits  = qubits
    this.#ops    = ops
    this.#cregs  = cregs
  }

  #add(op: Op): Circuit {
    return new Circuit(this.qubits, [...this.#ops, op], this.#cregs)
  }

  #ctrl(control: number, target: number, gate: Gate2x2): Circuit {
    return this.#add({ kind: 'controlled', control, target, gate })
  }

  // ── IonQ single-qubit gates ──────────────────────────────────────────────

  h(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.H  }) }
  x(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.X  }) }
  y(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Y  }) }
  z(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Z  }) }
  s(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.S  }) }
  si(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Si }) }
  t(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.T  }) }
  ti(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Ti }) }
  v(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.V  }) }
  vi(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Vi }) }

  // ── Rotation gates ───────────────────────────────────────────────────────

  rx(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Rx(theta) }) }
  ry(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Ry(theta) }) }
  rz(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Rz(theta) }) }

  // ── Named phase rotation gates ───────────────────────────────────────────

  /** Rz(π/2) — phase rotation by a half-turn; S up to global phase. */
  r2(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.R2 }) }

  /** Rz(π/4) — phase rotation by a quarter-turn; T up to global phase. */
  r4(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.R4 }) }

  /** Rz(π/8) — phase rotation by an eighth-turn. */
  r8(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.R8 }) }

  // ── OpenQASM basis gates ─────────────────────────────────────────────────

  /** U1(λ) — phase gate; equal to Rz(λ) up to global phase. */
  u1(lambda: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.U1(lambda) }) }

  /** U2(φ, λ) = U3(π/2, φ, λ) — equatorial gate. U2(0, π) = H. */
  u2(phi: number, lambda: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.U2(phi, lambda) }) }

  /** U3(θ, φ, λ) — general single-qubit unitary; OpenQASM 2.0 basis gate. */
  u3(theta: number, phi: number, lambda: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.U3(theta, phi, lambda) }) }

  // ── Two-qubit gates ──────────────────────────────────────────────────────

  /** Controlled-NOT. IonQ name: cnot. */
  cnot(control: number, target: number): Circuit {
    return this.#add({ kind: 'cnot', control, target })
  }

  swap(a: number, b: number): Circuit {
    return this.#add({ kind: 'swap', a, b })
  }

  // ── Two-qubit interaction gates ─────────────────────────────────────────

  /** XX(θ) = exp(−iθ/2 · X⊗X) — Ising-XX interaction; IonQ native. */
  xx(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Xx(theta) }) }

  /** YY(θ) = exp(−iθ/2 · Y⊗Y) — Ising-YY interaction; IonQ native. */
  yy(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Yy(theta) }) }

  /** ZZ(θ) = exp(−iθ/2 · Z⊗Z) — Ising-ZZ interaction; IonQ native. */
  zz(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Zz(theta) }) }

  /** XY(θ) interaction gate. XY(π) = iSWAP, XY(π/2) = √iSWAP. */
  xy(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Xy(theta) }) }

  /** iSWAP = XY(π): swaps qubits and multiplies each by i. */
  iswap(a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.ISwap }) }

  /** √iSWAP = XY(π/2): square root of iSWAP. */
  srswap(a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.SrSwap }) }

  // ── Controlled single-qubit gates ────────────────────────────────────────

  /** Controlled-NOT; alias for cnot. IBM/OpenQASM name. */
  cx(control: number, target: number): Circuit { return this.cnot(control, target) }

  cy(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Y) }
  cz(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Z) }
  ch(control: number, target: number): Circuit { return this.#ctrl(control, target, G.H) }

  // ── Controlled rotation gates ────────────────────────────────────────────

  crx(theta: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.Rx(theta)) }
  cry(theta: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.Ry(theta)) }
  crz(theta: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.Rz(theta)) }

  /** Controlled-Rz(π/2) — controlled phase half-turn. */
  cr2(control: number, target: number): Circuit { return this.#ctrl(control, target, G.R2) }

  /** Controlled-Rz(π/4) — controlled phase quarter-turn. */
  cr4(control: number, target: number): Circuit { return this.#ctrl(control, target, G.R4) }

  /** Controlled-Rz(π/8) — controlled phase eighth-turn. */
  cr8(control: number, target: number): Circuit { return this.#ctrl(control, target, G.R8) }

  // ── Controlled parameterized unitaries ───────────────────────────────────

  /** CU1(λ) — controlled phase gate; CU1(π) = CZ. */
  cu1(lambda: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.U1(lambda)) }

  /** CU3(θ,φ,λ) — controlled general unitary; CU3(π,0,π) = CX. */
  cu3(theta: number, phi: number, lambda: number, control: number, target: number): Circuit {
    return this.#ctrl(control, target, G.U3(theta, phi, lambda))
  }

  // ── Controlled phase gates ────────────────────────────────────────────────

  cs(control: number, target: number):   Circuit { return this.#ctrl(control, target, G.S)  }
  ct(control: number, target: number):   Circuit { return this.#ctrl(control, target, G.T)  }
  csdg(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Si) }
  ctdg(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Ti) }

  // ── Native IonQ gates ────────────────────────────────────────────────────

  /** GPI(φ) — IonQ hardware-native single-qubit gate. GPI(0) = X, GPI(π/2) = Y. */
  gpi(phi: number, q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Gpi(phi)  }) }

  /** GPI2(φ) — IonQ hardware-native half-rotation. GPI2(0) = Rx(π/2), GPI2(π/2) = Ry(π/2). */
  gpi2(phi: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Gpi2(phi) }) }

  /** MS(φ₀, φ₁) — Mølmer-Sørensen entangling gate; IonQ's native two-qubit operation. MS(0,0) = XX(π/2). */
  ms(phi0: number, phi1: number, a: number, b: number): Circuit {
    return this.#add({ kind: 'two', a, b, gate: G.Ms(phi0, phi1) })
  }

  // ── Three-qubit gates ────────────────────────────────────────────────────

  /** Toffoli (CCX): flip target if both c1 and c2 are |1⟩. Universal for reversible computation. */
  ccx(c1: number, c2: number, target: number): Circuit {
    return this.#add({ kind: 'toffoli', c1, c2, target })
  }

  /** Fredkin (CSWAP): swap qubits a and b if control is |1⟩. */
  cswap(control: number, a: number, b: number): Circuit {
    return this.#add({ kind: 'cswap', control, a, b })
  }

  // ── Statevector inspection ────────────────────────────────────────────────

  /**
   * Simulate the circuit and return the full sparse amplitude map.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   */
  statevector(): Map<bigint, Complex> {
    if (this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      throw new TypeError('statevector() requires a pure circuit — remove measure/reset/if ops')
    }
    return simulatePure(this.#ops, this.qubits)
  }

  /**
   * Return the complex amplitude for the basis state identified by `bitstring`.
   * Bitstring format: q0 is the rightmost character (IonQ convention), e.g. `'01'` = q0=1, q1=0.
   */
  amplitude(bitstring: string): Complex {
    return this.statevector().get(BigInt('0b' + bitstring)) ?? ZERO
  }

  /** Return the measurement probability (|amplitude|²) for the given basis state bitstring. */
  probability(bitstring: string): number {
    const { re, im } = this.amplitude(bitstring)
    return re * re + im * im
  }

  // ── Classical registers and mid-circuit measurement ──────────────────────

  /** Declare a classical register of `size` bits. */
  creg(name: string, size: number): Circuit {
    return new Circuit(this.qubits, this.#ops, new Map(this.#cregs).set(name, size))
  }

  /**
   * Measure qubit `q` in the computational basis, storing the outcome in
   * `creg[bit]`. Collapses the statevector for that shot.
   * Auto-registers the creg if not yet declared.
   */
  measure(q: number, creg: string, bit: number): Circuit {
    const size = Math.max(this.#cregs.get(creg) ?? 0, bit + 1)
    return new Circuit(
      this.qubits,
      [...this.#ops, { kind: 'measure', q, creg, bit }],
      new Map(this.#cregs).set(creg, size),
    )
  }

  /** Reset qubit `q` to |0⟩ by measuring and conditionally flipping. */
  reset(q: number): Circuit { return this.#add({ kind: 'reset', q }) }

  /**
   * Conditionally apply a gate (or sequence of gates) only when the classical
   * register `creg` equals `value`.
   *
   * `value` is compared against the register as a little-endian integer:
   * bit 0 is the LSB.  For a 1-bit register, `value` is simply 0 or 1.
   *
   * @example
   * // Apply X to q2 only if register 'c' == 1
   * circuit.if('c', 1, c => c.x(2))
   */
  if(creg: string, value: number, build: (c: Circuit) => Circuit): Circuit {
    const inner = build(new Circuit(this.qubits))
    return this.#add({ kind: 'if', creg, value, ops: inner.#ops })
  }

  // ── Execution ────────────────────────────────────────────────────────────

  /** Run the circuit and return a probability distribution. */
  run({ shots = 1024, seed }: RunOptions = {}): Distribution {
    const rng        = makePrng(seed)
    const cregCounts = new Map<string, number[]>(
      this.#cregs.entries().map(([name, size]) => [name, new Array<number>(size).fill(0)])
    )

    // ── Fast path: pure circuit — simulate once, sample N times ──────────────
    if (!this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      const sv     = simulatePure(this.#ops, this.qubits)
      const probs  = probabilities(sv)
      const sorted = probs.entries().toArray().toSorted(([a], [b]) => (a < b ? -1 : 1))

      const cdf: { idx: bigint; cumP: number }[] = []
      let cum = 0
      for (const [idx, p] of sorted) {
        cum += p
        cdf.push({ idx, cumP: cum })
      }
      const last = cdf.at(-1)
      if (last) last.cumP = 1.0

      const counts = new Map<bigint, number>()
      for (let i = 0; i < shots; i++) {
        const r  = rng()
        let lo   = 0
        let hi   = cdf.length - 1
        while (lo < hi) {
          const mid = (lo + hi) >> 1
          if (cdf[mid]!.cumP < r) lo = mid + 1
          else hi = mid
        }
        const idx = cdf[lo]?.idx ?? 0n
        counts.set(idx, (counts.get(idx) ?? 0) + 1)
      }

      return new Distribution(this.qubits, shots, counts, cregCounts)
    }

    // ── Per-shot path: mid-circuit measurement — one full simulation per shot ──
    const counts = new Map<bigint, number>()

    for (let i = 0; i < shots; i++) {
      const shotCregs = new Map<string, boolean[]>(
        this.#cregs.entries().map(([name, size]) => [name, new Array<boolean>(size).fill(false)])
      )

      const sv       = applyOps(this.#ops, zero(this.qubits), shotCregs, rng)
      const finalIdx = sampleSV(sv, rng())
      counts.set(finalIdx, (counts.get(finalIdx) ?? 0) + 1)

      for (const [name, bits] of shotCregs) {
        const acc = cregCounts.get(name)!
        for (const [j, b] of bits.entries()) if (b) acc[j]! += 1
      }
    }

    return new Distribution(this.qubits, shots, counts, cregCounts)
  }
}
