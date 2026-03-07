/**
 * Immutable circuit builder with IonQ QIS gate names.
 *
 * Each gate method returns a new Circuit — circuits are values, not mutated objects.
 * Simulation is lazy: nothing runs until `.run()` is called.
 */

import * as G from './gates.js'
import { applyCNOT, applyControlled, applyCSwap, applySingle, applySWAP, applyToffoli, applyTwo, Gate2x2, Gate4x4, probabilities, StateVector, zero } from './statevector.js'
import { Complex, ZERO } from './complex.js'
import { CNOT4, controlledGate, mpsApply1, mpsApply2, mpsInit, mpsSample, SWAP4 } from './mps.js'

// ─── Operation types ─────────────────────────────────────────────────────────

/** Serialization tag carried on ops that have an IonQ JSON representation. */
type GateMeta = { name: string; params?: readonly number[] }

type SingleOp     = { kind: 'single';     q: number;                      gate: Gate2x2; meta?: GateMeta }
type CNOTOp       = { kind: 'cnot';       control: number; target: number                                }
type SWAPOp       = { kind: 'swap';       a: number;       b: number                                    }
type TwoOp        = { kind: 'two';        a: number;       b: number;    gate: Gate4x4;  meta?: GateMeta }
type ControlledOp = { kind: 'controlled'; control: number; target: number; gate: Gate2x2; meta?: GateMeta }
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
function applyOps(ops: readonly Op[], svIn: StateVector, shotCregs: Map<string, boolean[]>, rng: () => number, noise?: NoiseParams): StateVector {
  let sv = svIn
  const p1 = noise?.p1 ?? 0
  const p2 = noise?.p2 ?? 0
  const pM = noise?.pMeas ?? 0
  for (const op of ops) {
    if      (op.kind === 'single')     { sv = applySingle(sv, op.q, op.gate);                         if (p1) sv = dep1(sv, op.q, p1, rng()) }
    else if (op.kind === 'cnot')       { sv = applyCNOT(sv, op.control, op.target);                   if (p2) sv = dep2(sv, op.control, op.target, p2, rng()) }
    else if (op.kind === 'controlled') { sv = applyControlled(sv, op.control, op.target, op.gate);    if (p2) sv = dep2(sv, op.control, op.target, p2, rng()) }
    else if (op.kind === 'swap')       { sv = applySWAP(sv, op.a, op.b);                              if (p2) sv = dep2(sv, op.a, op.b, p2, rng()) }
    else if (op.kind === 'toffoli')      sv = applyToffoli(sv, op.c1, op.c2, op.target)
    else if (op.kind === 'cswap')        sv = applyCSwap(sv, op.control, op.a, op.b)
    else if (op.kind === 'two')        { sv = applyTwo(sv, op.a, op.b, op.gate);                      if (p2) sv = dep2(sv, op.a, op.b, p2, rng()) }
    else if (op.kind === 'measure') {
      const { outcome, sv: next } = collapseQubit(sv, op.q, rng())
      const reported: 0 | 1 = pM && rng() < pM ? (outcome === 1 ? 0 : 1) : outcome
      sv = next
      const reg = shotCregs.get(op.creg)
      if (reg) reg[op.bit] = reported === 1
    } else if (op.kind === 'reset') {
      const { outcome, sv: next } = collapseQubit(sv, op.q, rng())
      sv = next
      if (outcome === 1) sv = applySingle(sv, op.q, G.X)
    } else {  // if
      if (cregValue(shotCregs, op.creg) === op.value) sv = applyOps(op.ops, sv, shotCregs, rng, noise)
    }
  }
  return sv
}

// ─── Noise simulation ─────────────────────────────────────────────────────────

/** Per-gate error parameters for stochastic noise simulation. */
export interface NoiseParams {
  /** Single-qubit depolarizing error probability per gate (0–1). */
  p1?: number
  /** Two-qubit depolarizing error probability per gate (0–1). */
  p2?: number
  /** SPAM: probability of flipping each measured bit (0–1). */
  pMeas?: number
}

/** Published IonQ device noise characteristics (conservative, from public benchmarks). */
const DEVICE_NOISE: Readonly<Record<string, NoiseParams>> = {
  'aria-1':  { p1: 0.0003,  p2: 0.005,  pMeas: 0.004  },
  'forte-1': { p1: 0.0001,  p2: 0.002,  pMeas: 0.002  },
  'harmony': { p1: 0.001,   p2: 0.015,  pMeas: 0.01   },
}

// 15 non-identity two-qubit Paulis for depolarizing channel: {I,X,Y,Z}⊗{I,X,Y,Z} \ {II}
const TWO_PAULI: readonly (readonly [Gate2x2 | null, Gate2x2 | null])[] = [
  [null, G.X], [null, G.Y], [null, G.Z],
  [G.X, null], [G.X, G.X], [G.X, G.Y], [G.X, G.Z],
  [G.Y, null], [G.Y, G.X], [G.Y, G.Y], [G.Y, G.Z],
  [G.Z, null], [G.Z, G.X], [G.Z, G.Y], [G.Z, G.Z],
]

/** Apply single-qubit depolarizing channel: random Pauli X/Y/Z with total probability p. */
function dep1(sv: StateVector, q: number, p: number, rand: number): StateVector {
  if (rand >= p) return sv
  const r = rand / p
  if (r < 1/3) return applySingle(sv, q, G.X)
  if (r < 2/3) return applySingle(sv, q, G.Y)
  return applySingle(sv, q, G.Z)
}

/** Apply two-qubit depolarizing channel: random non-identity 2-qubit Pauli with total probability p. */
function dep2(sv: StateVector, a: number, b: number, p: number, rand: number): StateVector {
  if (rand >= p) return sv
  const [pa, pb] = TWO_PAULI[Math.min(Math.floor(rand / p * 15), 14)]!
  if (pa) sv = applySingle(sv, a, pa)
  if (pb) sv = applySingle(sv, b, pb)
  return sv
}

// ─── IonQ JSON types ──────────────────────────────────────────────────────────

/** A single gate entry in the IonQ `ionq.circuit.v0` JSON format. */
export interface IonQGate {
  gate:     string
  target?:  number
  targets?: [number, number]
  control?: number
  rotation?: number
  phase?:   number
  phases?:  [number, number]
}

/** The `ionq.circuit.v0` circuit object accepted by IonQ Cloud and qsim. */
export interface IonQCircuit {
  format:  'ionq.circuit.v0'
  qubits:  number
  circuit: readonly IonQGate[]
}

// ─── OpenQASM 2.0 helpers ─────────────────────────────────────────────────────

/**
 * Format a radian value as an angle expression using `piToken` for π.
 * Recognises rational multiples of π (up to denominator 16) for clean output.
 */
function fmtAngle(r: number, piToken: string): string {
  if (Math.abs(r) < 1e-14) return '0'
  const f = r / Math.PI
  for (const d of [1, 2, 3, 4, 6, 8, 12, 16]) {
    for (let n = -16; n <= 16; n++) {
      if (n === 0) continue
      if (Math.abs(f - n / d) < 1e-12) {
        const sign = n < 0 ? '-' : ''
        const a = Math.abs(n)
        if (d === 1) return a === 1 ? `${sign}${piToken}` : `${sign}${a}*${piToken}`
        return a === 1 ? `${sign}${piToken}/${d}` : `${sign}${a}*${piToken}/${d}`
      }
    }
  }
  return String(r)
}

function qasmAngle(r: number): string { return fmtAngle(r, 'pi') }
function pyAngle(r: number):   string { return fmtAngle(r, 'math.pi') }

/** Parse a QASM angle expression (supports `pi`, `*`, `/`, `+`, `-`, parentheses). */
function parseAngle(expr: string): number {
  const s = expr.replace(/\s/g, '')
  let i = 0
  function parseFactor(): number {
    if (s[i] === '-') { i++; return -parseFactor() }
    if (s[i] === '+') { i++; return parseFactor() }
    if (s[i] === '(') { i++; const v = parseExpr(); if (s[i] === ')') i++; return v }
    if (s.startsWith('pi', i)) { i += 2; return Math.PI }
    const j = i
    while (i < s.length && /[0-9.]/.test(s[i]!)) i++
    return parseFloat(s.slice(j, i))
  }
  function parseTerm(): number {
    let v = parseFactor()
    while (i < s.length && (s[i] === '*' || s[i] === '/')) {
      const op = s[i++]!; v = op === '*' ? v * parseFactor() : v / parseFactor()
    }
    return v
  }
  function parseExpr(): number {
    let v = parseTerm()
    while (i < s.length && (s[i] === '+' || s[i] === '-')) {
      const op = s[i++]!; v = op === '+' ? v + parseTerm() : v - parseTerm()
    }
    return v
  }
  return parseExpr()
}

/**
 * Map a GateMeta to a QASM gate name and parameters.
 * Some gates have different names in QASM (si→sdg) or fixed params (r2→rz(π/2)).
 */
function qasmGateName(meta: GateMeta): { qname: string; qparams: number[] } {
  switch (meta.name) {
    case 'si':   return { qname: 'sdg',  qparams: [] }
    case 'ti':   return { qname: 'tdg',  qparams: [] }
    case 'v':    return { qname: 'sx',   qparams: [] }
    case 'vi':   return { qname: 'sxdg', qparams: [] }
    case 'r2':   return { qname: 'rz',   qparams: [Math.PI / 2] }
    case 'r4':   return { qname: 'rz',   qparams: [Math.PI / 4] }
    case 'r8':   return { qname: 'rz',   qparams: [Math.PI / 8] }
    case 'cr2':  return { qname: 'crz',  qparams: [Math.PI / 2] }
    case 'cr4':  return { qname: 'crz',  qparams: [Math.PI / 4] }
    case 'cr8':  return { qname: 'crz',  qparams: [Math.PI / 8] }
    case 'cs':   return { qname: 'cu1',  qparams: [Math.PI / 2] }
    case 'ct':   return { qname: 'cu1',  qparams: [Math.PI / 4] }
    case 'csdg': return { qname: 'cu1',  qparams: [-Math.PI / 2] }
    case 'ctdg': return { qname: 'cu1',  qparams: [-Math.PI / 4] }
    default:     return { qname: meta.name, qparams: [...(meta.params ?? [])] }
  }
}

/** Dispatch a parsed QASM gate onto a Circuit. */
function applyQASMGate(c: Circuit, name: string, params: number[], qs: number[]): Circuit {
  const [a, b, d] = qs
  const [p0, p1, p2] = params
  switch (name) {
    case 'h':     return c.h(a!)
    case 'x':     return c.x(a!)
    case 'y':     return c.y(a!)
    case 'z':     return c.z(a!)
    case 's':     return c.s(a!)
    case 'sdg':   return c.si(a!)
    case 't':     return c.t(a!)
    case 'tdg':   return c.ti(a!)
    case 'sx':    return c.v(a!)
    case 'sxdg':  return c.vi(a!)
    case 'rx':    return c.rx(p0!, a!)
    case 'ry':    return c.ry(p0!, a!)
    case 'rz':    return c.rz(p0!, a!)
    case 'u1':    return c.u1(p0!, a!)
    case 'u2':    return c.u2(p0!, p1!, a!)
    case 'u3':    return c.u3(p0!, p1!, p2!, a!)
    case 'cx':    return c.cnot(a!, b!)
    case 'cy':    return c.cy(a!, b!)
    case 'cz':    return c.cz(a!, b!)
    case 'ch':    return c.ch(a!, b!)
    case 'crx':   return c.crx(p0!, a!, b!)
    case 'cry':   return c.cry(p0!, a!, b!)
    case 'crz':   return c.crz(p0!, a!, b!)
    case 'cu1':   return c.cu1(p0!, a!, b!)
    case 'cu3':   return c.cu3(p0!, p1!, p2!, a!, b!)
    case 'swap':  return c.swap(a!, b!)
    case 'ccx':   return c.ccx(a!, b!, d!)
    case 'cswap': return c.cswap(a!, b!, d!)
    default: throw new TypeError(`Unknown QASM gate: '${name}'`)
  }
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
  /** Named device profile ('aria-1' | 'forte-1' | 'harmony') or custom NoiseParams. */
  noise?: string | NoiseParams
}

export interface MpsRunOptions {
  shots?: number
  seed?: number
  /** Maximum bond dimension χ (default 64). Larger = more accurate for high-entanglement circuits. */
  maxBond?: number
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

  #ctrl(control: number, target: number, gate: Gate2x2, meta?: GateMeta): Circuit {
    return this.#add({ kind: 'controlled', control, target, gate, meta })
  }

  // ── IonQ single-qubit gates ──────────────────────────────────────────────

  h(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.H,  meta: { name: 'h'  } }) }
  x(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.X,  meta: { name: 'x'  } }) }
  y(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Y,  meta: { name: 'y'  } }) }
  z(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Z,  meta: { name: 'z'  } }) }
  s(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.S,  meta: { name: 's'  } }) }
  si(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Si, meta: { name: 'si' } }) }
  t(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.T,  meta: { name: 't'  } }) }
  ti(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Ti, meta: { name: 'ti' } }) }
  v(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.V,  meta: { name: 'v'  } }) }
  vi(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Vi, meta: { name: 'vi' } }) }

  // ── Rotation gates ───────────────────────────────────────────────────────

  rx(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Rx(theta), meta: { name: 'rx', params: [theta] } }) }
  ry(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Ry(theta), meta: { name: 'ry', params: [theta] } }) }
  rz(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Rz(theta), meta: { name: 'rz', params: [theta] } }) }

  // ── Named phase rotation gates ───────────────────────────────────────────

  /** Rz(π/2) — phase rotation by a half-turn; S up to global phase. */
  r2(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.R2, meta: { name: 'r2' } }) }

  /** Rz(π/4) — phase rotation by a quarter-turn; T up to global phase. */
  r4(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.R4, meta: { name: 'r4' } }) }

  /** Rz(π/8) — phase rotation by an eighth-turn. */
  r8(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.R8, meta: { name: 'r8' } }) }

  // ── OpenQASM basis gates ─────────────────────────────────────────────────

  /** U1(λ) — phase gate; equal to Rz(λ) up to global phase. */
  u1(lambda: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.U1(lambda), meta: { name: 'u1', params: [lambda] } }) }

  /** U2(φ, λ) = U3(π/2, φ, λ) — equatorial gate. U2(0, π) = H. */
  u2(phi: number, lambda: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.U2(phi, lambda), meta: { name: 'u2', params: [phi, lambda] } }) }

  /** U3(θ, φ, λ) — general single-qubit unitary; OpenQASM 2.0 basis gate. */
  u3(theta: number, phi: number, lambda: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.U3(theta, phi, lambda), meta: { name: 'u3', params: [theta, phi, lambda] } }) }

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
  xx(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Xx(theta), meta: { name: 'xx', params: [theta] } }) }

  /** YY(θ) = exp(−iθ/2 · Y⊗Y) — Ising-YY interaction; IonQ native. */
  yy(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Yy(theta), meta: { name: 'yy', params: [theta] } }) }

  /** ZZ(θ) = exp(−iθ/2 · Z⊗Z) — Ising-ZZ interaction; IonQ native. */
  zz(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Zz(theta), meta: { name: 'zz', params: [theta] } }) }

  /** XY(θ) interaction gate. XY(π) = iSWAP, XY(π/2) = √iSWAP. */
  xy(theta: number, a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.Xy(theta), meta: { name: 'xy', params: [theta] } }) }

  /** iSWAP = XY(π): swaps qubits and multiplies each by i. */
  iswap(a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.ISwap, meta: { name: 'iswap' } }) }

  /** √iSWAP = XY(π/2): square root of iSWAP. */
  srswap(a: number, b: number): Circuit { return this.#add({ kind: 'two', a, b, gate: G.SrSwap, meta: { name: 'srswap' } }) }

  // ── Controlled single-qubit gates ────────────────────────────────────────

  /** Controlled-NOT; alias for cnot. IBM/OpenQASM name. */
  cx(control: number, target: number): Circuit { return this.cnot(control, target) }

  cy(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Y, { name: 'cy' }) }
  cz(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Z, { name: 'cz' }) }
  ch(control: number, target: number): Circuit { return this.#ctrl(control, target, G.H, { name: 'ch' }) }

  // ── Controlled rotation gates ────────────────────────────────────────────

  crx(theta: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.Rx(theta), { name: 'crx', params: [theta] }) }
  cry(theta: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.Ry(theta), { name: 'cry', params: [theta] }) }
  crz(theta: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.Rz(theta), { name: 'crz', params: [theta] }) }

  /** Controlled-Rz(π/2) — controlled phase half-turn. */
  cr2(control: number, target: number): Circuit { return this.#ctrl(control, target, G.R2, { name: 'cr2' }) }

  /** Controlled-Rz(π/4) — controlled phase quarter-turn. */
  cr4(control: number, target: number): Circuit { return this.#ctrl(control, target, G.R4, { name: 'cr4' }) }

  /** Controlled-Rz(π/8) — controlled phase eighth-turn. */
  cr8(control: number, target: number): Circuit { return this.#ctrl(control, target, G.R8, { name: 'cr8' }) }

  // ── Controlled parameterized unitaries ───────────────────────────────────

  /** CU1(λ) — controlled phase gate; CU1(π) = CZ. */
  cu1(lambda: number, control: number, target: number): Circuit { return this.#ctrl(control, target, G.U1(lambda), { name: 'cu1', params: [lambda] }) }

  /** CU3(θ,φ,λ) — controlled general unitary; CU3(π,0,π) = CX. */
  cu3(theta: number, phi: number, lambda: number, control: number, target: number): Circuit {
    return this.#ctrl(control, target, G.U3(theta, phi, lambda), { name: 'cu3', params: [theta, phi, lambda] })
  }

  // ── Controlled phase gates ────────────────────────────────────────────────

  cs(control: number, target: number):   Circuit { return this.#ctrl(control, target, G.S,  { name: 'cs'   }) }
  ct(control: number, target: number):   Circuit { return this.#ctrl(control, target, G.T,  { name: 'ct'   }) }
  csdg(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Si, { name: 'csdg' }) }
  ctdg(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Ti, { name: 'ctdg' }) }

  // ── Native IonQ gates ────────────────────────────────────────────────────

  /** GPI(φ) — IonQ hardware-native single-qubit gate. GPI(0) = X, GPI(π/2) = Y. */
  gpi(phi: number, q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Gpi(phi),  meta: { name: 'gpi',  params: [phi] } }) }

  /** GPI2(φ) — IonQ hardware-native half-rotation. GPI2(0) = Rx(π/2), GPI2(π/2) = Ry(π/2). */
  gpi2(phi: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Gpi2(phi), meta: { name: 'gpi2', params: [phi] } }) }

  /** MS(φ₀, φ₁) — Mølmer-Sørensen entangling gate; IonQ's native two-qubit operation. MS(0,0) = XX(π/2). */
  ms(phi0: number, phi1: number, a: number, b: number): Circuit {
    return this.#add({ kind: 'two', a, b, gate: G.Ms(phi0, phi1), meta: { name: 'ms', params: [phi0, phi1] } })
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

  // ── IonQ JSON import / export ────────────────────────────────────────────

  /**
   * Parse an `ionq.circuit.v0` JSON object into a `Circuit`.
   *
   * Angle convention: `rotation` fields are in π-radians (1.0 = π rad);
   * `phase` / `phases` fields are in turns (1.0 = 2π rad).
   */
  static fromIonQ({ qubits, circuit }: IonQCircuit): Circuit {
    let c = new Circuit(qubits)
    for (const g of circuit) {
      const t  = g.target  ?? 0
      const [a, b] = g.targets ?? [0, 1]
      const rot = (g.rotation ?? 0) * Math.PI
      const ph  = (g.phase   ?? 0) * 2 * Math.PI
      switch (g.gate) {
        case 'h':    c = c.h(t);  break
        case 'x':    c = c.x(t);  break
        case 'y':    c = c.y(t);  break
        case 'z':    c = c.z(t);  break
        case 's':    c = c.s(t);  break
        case 'si':   c = c.si(t); break
        case 't':    c = c.t(t);  break
        case 'ti':   c = c.ti(t); break
        case 'v':    c = c.v(t);  break
        case 'vi':   c = c.vi(t); break
        case 'rx':   c = c.rx(rot, t); break
        case 'ry':   c = c.ry(rot, t); break
        case 'rz':   c = c.rz(rot, t); break
        case 'r2':   c = c.r2(t); break
        case 'r4':   c = c.r4(t); break
        case 'r8':   c = c.r8(t); break
        case 'gpi':  c = c.gpi(ph, t);  break
        case 'gpi2': c = c.gpi2(ph, t); break
        case 'cnot': c = c.cnot(g.control ?? 0, t); break
        case 'swap': c = c.swap(a, b); break
        case 'xx':   c = c.xx(rot, a, b); break
        case 'yy':   c = c.yy(rot, a, b); break
        case 'zz':   c = c.zz(rot, a, b); break
        case 'ms': {
          const [p0, p1] = (g.phases ?? [0, 0]).map(p => p * 2 * Math.PI)
          c = c.ms(p0!, p1!, a, b)
          break
        }
        default: throw new TypeError(`Unknown IonQ gate: '${g.gate}'`)
      }
    }
    return c
  }

  /**
   * Serialize to an `ionq.circuit.v0` JSON object ready for IonQ Cloud or qsim.
   *
   * Throws `TypeError` for any gate that has no IonQ JSON representation
   * (controlled variants, U-gates, XY/iSWAP, mid-circuit measurement, etc.).
   */
  toIonQ(): IonQCircuit {
    const IONQ_SINGLE = new Set(['h','x','y','z','s','si','t','ti','v','vi','rx','ry','rz','r2','r4','r8','gpi','gpi2'])
    const IONQ_TWO    = new Set(['xx','yy','zz','ms'])
    const circuit: IonQGate[] = []
    for (const op of this.#ops) {
      if (op.kind === 'cnot') {
        circuit.push({ gate: 'cnot', control: op.control, target: op.target })
      } else if (op.kind === 'swap') {
        circuit.push({ gate: 'swap', targets: [op.a, op.b] })
      } else if (op.kind === 'single' && op.meta && IONQ_SINGLE.has(op.meta.name)) {
        const { name, params } = op.meta
        const g: IonQGate = { gate: name, target: op.q }
        if (params) {
          if (name === 'gpi' || name === 'gpi2') g.phase    = params[0]! / (2 * Math.PI)
          else                                   g.rotation = params[0]! / Math.PI
        }
        circuit.push(g)
      } else if (op.kind === 'two' && op.meta && IONQ_TWO.has(op.meta.name)) {
        const { name, params } = op.meta
        const g: IonQGate = { gate: name, targets: [op.a, op.b] }
        if (params) {
          if (name === 'ms') g.phases   = [params[0]! / (2 * Math.PI), params[1]! / (2 * Math.PI)]
          else               g.rotation = params[0]! / Math.PI
        }
        circuit.push(g)
      } else {
        const n = (op as { meta?: GateMeta }).meta?.name ?? op.kind
        throw new TypeError(`Gate '${n}' is not serializable to IonQ JSON`)
      }
    }
    return { format: 'ionq.circuit.v0', qubits: this.qubits, circuit }
  }

  // ── OpenQASM 2.0 import / export ─────────────────────────────────────────

  /**
   * Emit a valid OpenQASM 2.0 string for this circuit.
   *
   * Gate name mapping: si→sdg, ti→tdg, v→sx, vi→sxdg; r2/r4/r8→rz(π/n);
   * cs/ct/csdg/ctdg→cu1(±π/n); cr2/cr4/cr8→crz(π/n).
   * Throws `TypeError` for gates with no QASM 2.0 representation (gpi, gpi2, xx, yy, zz, ms, xy, iswap, srswap, if).
   */
  toQASM(): string {
    const lines: string[] = [
      'OPENQASM 2.0;',
      'include "qelib1.inc";',
      '',
      `qreg q[${this.qubits}];`,
    ]
    for (const [name, size] of this.#cregs) lines.push(`creg ${name}[${size}];`)
    if (this.#cregs.size) lines.push('')

    for (const op of this.#ops) {
      switch (op.kind) {
        case 'cnot':    lines.push(`cx q[${op.control}],q[${op.target}];`); break
        case 'swap':    lines.push(`swap q[${op.a}],q[${op.b}];`);         break
        case 'toffoli': lines.push(`ccx q[${op.c1}],q[${op.c2}],q[${op.target}];`); break
        case 'cswap':   lines.push(`cswap q[${op.control}],q[${op.a}],q[${op.b}];`); break
        case 'measure': lines.push(`measure q[${op.q}] -> ${op.creg}[${op.bit}];`); break
        case 'reset':   lines.push(`reset q[${op.q}];`); break
        case 'if':      throw new TypeError('if ops cannot be serialized to OpenQASM 2.0')
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          if (op.meta.name === 'gpi' || op.meta.name === 'gpi2')
            throw new TypeError(`Gate '${op.meta.name}' has no OpenQASM 2.0 representation`)
          const { qname, qparams } = qasmGateName(op.meta)
          const ps = qparams.length ? `(${qparams.map(qasmAngle).join(',')})` : ''
          lines.push(`${qname}${ps} q[${op.q}];`)
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { qname, qparams } = qasmGateName(op.meta)
          const ps = qparams.length ? `(${qparams.map(qasmAngle).join(',')})` : ''
          lines.push(`${qname}${ps} q[${op.control}],q[${op.target}];`)
          break
        }
        case 'two': {
          const n = op.meta?.name ?? 'two'
          throw new TypeError(`Gate '${n}' has no OpenQASM 2.0 representation`)
        }
      }
    }
    return lines.join('\n')
  }

  /**
   * Parse an OpenQASM 2.0 string into a `Circuit`.
   *
   * Supports: qreg/creg, all qelib1.inc gates, measure, reset, single-line comments.
   * Does not support: gate definitions, barrier, opaque, if statements, multi-qubit qregs.
   */
  static fromQASM(source: string): Circuit {
    // Strip single-line comments, split into statements
    const stmts = source
      .replace(/\/\/[^\n]*/g, '')
      .split(';')
      .map(s => s.trim())
      .filter(Boolean)

    let qubits = 0
    const cregSizes = new Map<string, number>()

    // First pass: collect qreg / creg declarations
    for (const stmt of stmts) {
      const qr = stmt.match(/^qreg\s+\w+\[(\d+)\]$/)
      if (qr) { qubits = parseInt(qr[1]!); continue }
      const cr = stmt.match(/^creg\s+(\w+)\[(\d+)\]$/)
      if (cr) cregSizes.set(cr[1]!, parseInt(cr[2]!))
    }

    let c = new Circuit(qubits)
    for (const [name, size] of cregSizes) c = c.creg(name, size)

    // Second pass: apply gates
    for (const stmt of stmts) {
      if (/^(OPENQASM|include|qreg|creg)\b/.test(stmt)) continue

      // measure q[i] -> creg[j]
      const meas = stmt.match(/^measure\s+\w+\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]$/)
      if (meas) { c = c.measure(parseInt(meas[1]!), meas[2]!, parseInt(meas[3]!)); continue }

      // reset q[i]
      const rst = stmt.match(/^reset\s+\w+\[(\d+)\]$/)
      if (rst) { c = c.reset(parseInt(rst[1]!)); continue }

      // gatename[(params)] q[i](,q[j])*
      const gate = stmt.match(/^(\w+)(?:\(([^)]*)\))?\s+([\w\[\],\s]+)$/)
      if (gate) {
        const name   = gate[1]!
        const params = gate[2] ? gate[2].split(',').map(p => parseAngle(p)) : []
        const qs     = [...gate[3]!.matchAll(/\[(\d+)\]/g)].map(m => parseInt(m[1]!))
        c = applyQASMGate(c, name, params, qs)
      }
    }
    return c
  }

  // ── Export targets ───────────────────────────────────────────────────────

  /**
   * Emit Python code for Qiskit's `QuantumCircuit` API.
   * Gate coverage: full standard gate set, rx/ry/rz, u1/u2/u3, controlled family,
   * rxx/ryy/rzz/iswap. Throws for gpi/gpi2/ms/xy/srswap/if.
   */
  toQiskit(): string {
    const lines: string[] = []
    const imports = ['from qiskit import QuantumCircuit']
    const cregLines: string[] = []

    if (this.#cregs.size) imports.push('from qiskit.circuit import ClassicalRegister')
    imports.push('import math', '')

    lines.push(`qc = QuantumCircuit(${this.qubits})`)
    for (const [name, size] of this.#cregs) {
      cregLines.push(`${name} = ClassicalRegister(${size}, '${name}')`)
      cregLines.push(`qc.add_register(${name})`)
    }
    if (cregLines.length) lines.push(...cregLines, '')

    for (const op of this.#ops) {
      switch (op.kind) {
        case 'cnot':    lines.push(`qc.cx(${op.control}, ${op.target})`);             break
        case 'swap':    lines.push(`qc.swap(${op.a}, ${op.b})`);                      break
        case 'toffoli': lines.push(`qc.ccx(${op.c1}, ${op.c2}, ${op.target})`);      break
        case 'cswap':   lines.push(`qc.cswap(${op.control}, ${op.a}, ${op.b})`);     break
        case 'measure': lines.push(`qc.measure(${op.q}, ${op.creg}[${op.bit}])`);    break
        case 'reset':   lines.push(`qc.reset(${op.q})`);                              break
        case 'if':      throw new TypeError('if ops cannot be serialized to Qiskit')
        case 'two': {
          const n = op.meta?.name
          if (n === 'xx') { lines.push(`qc.rxx(${pyAngle(op.meta!.params![0]!)}, ${op.a}, ${op.b})`); break }
          if (n === 'yy') { lines.push(`qc.ryy(${pyAngle(op.meta!.params![0]!)}, ${op.a}, ${op.b})`); break }
          if (n === 'zz') { lines.push(`qc.rzz(${pyAngle(op.meta!.params![0]!)}, ${op.a}, ${op.b})`); break }
          if (n === 'iswap') { lines.push(`qc.iswap(${op.a}, ${op.b})`); break }
          throw new TypeError(`Gate '${n ?? 'two'}' has no Qiskit representation`)
        }
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          if (n === 'gpi' || n === 'gpi2') throw new TypeError(`Gate '${n}' has no Qiskit representation`)
          const angle = () => pyAngle(p![0]!)
          switch (n) {
            case 'si': lines.push(`qc.sdg(${q})`);                   break
            case 'ti': lines.push(`qc.tdg(${q})`);                   break
            case 'v':  lines.push(`qc.sx(${q})`);                    break
            case 'vi': lines.push(`qc.sxdg(${q})`);                  break
            case 'r2': lines.push(`qc.rz(math.pi/2, ${q})`);        break
            case 'r4': lines.push(`qc.rz(math.pi/4, ${q})`);        break
            case 'r8': lines.push(`qc.rz(math.pi/8, ${q})`);        break
            case 'rx': lines.push(`qc.rx(${angle()}, ${q})`);        break
            case 'ry': lines.push(`qc.ry(${angle()}, ${q})`);        break
            case 'rz': lines.push(`qc.rz(${angle()}, ${q})`);        break
            case 'u1': lines.push(`qc.u1(${angle()}, ${q})`);        break
            case 'u2': lines.push(`qc.u2(${pyAngle(p![0]!)}, ${pyAngle(p![1]!)}, ${q})`); break
            case 'u3': lines.push(`qc.u3(${pyAngle(p![0]!)}, ${pyAngle(p![1]!)}, ${pyAngle(p![2]!)}, ${q})`); break
            default:   lines.push(`qc.${n}(${q})`)
          }
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { name: n, params: p } = op.meta
          const [c, t] = [op.control, op.target]
          const angle = () => pyAngle(p![0]!)
          switch (n) {
            case 'cr2':  lines.push(`qc.crz(math.pi/2, ${c}, ${t})`);                          break
            case 'cr4':  lines.push(`qc.crz(math.pi/4, ${c}, ${t})`);                          break
            case 'cr8':  lines.push(`qc.crz(math.pi/8, ${c}, ${t})`);                          break
            case 'cs':   lines.push(`qc.cu1(math.pi/2, ${c}, ${t})`);                          break
            case 'ct':   lines.push(`qc.cu1(math.pi/4, ${c}, ${t})`);                          break
            case 'csdg': lines.push(`qc.cu1(-math.pi/2, ${c}, ${t})`);                         break
            case 'ctdg': lines.push(`qc.cu1(-math.pi/4, ${c}, ${t})`);                         break
            case 'crx':  lines.push(`qc.crx(${angle()}, ${c}, ${t})`);                         break
            case 'cry':  lines.push(`qc.cry(${angle()}, ${c}, ${t})`);                         break
            case 'crz':  lines.push(`qc.crz(${angle()}, ${c}, ${t})`);                         break
            case 'cu1':  lines.push(`qc.cu1(${angle()}, ${c}, ${t})`);                         break
            case 'cu3':  lines.push(`qc.cu3(${pyAngle(p![0]!)}, ${pyAngle(p![1]!)}, ${pyAngle(p![2]!)}, ${c}, ${t})`); break
            default:     lines.push(`qc.${n}(${c}, ${t})`)
          }
          break
        }
      }
    }
    return [...imports, ...lines].join('\n')
  }

  /**
   * Emit Python code for Google Cirq.
   * Gate coverage: H/X/Y/Z/S/T, rx/ry/rz, r2/r4/r8, u1/u3,
   * CNOT/CZ/CY/CH/swap/CCNOT/CSWAP, crx/cry/crz/cu1/cu3.
   * Throws for gpi/gpi2/ms/xx/yy/zz/xy/iswap/srswap/if.
   */
  toCirq(): string {
    const ops: string[] = []

    const gateOp = (gate: string, qs: number[]) =>
      `    ${gate}(${qs.map(q => `q[${q}]`).join(', ')}),`
    const rads = (r: number) => `rads=${pyAngle(r)}`

    for (const op of this.#ops) {
      switch (op.kind) {
        case 'cnot':    ops.push(gateOp('cirq.CNOT', [op.control, op.target]));  break
        case 'swap':    ops.push(gateOp('cirq.SWAP', [op.a, op.b]));             break
        case 'toffoli': ops.push(gateOp('cirq.CCNOT', [op.c1, op.c2, op.target])); break
        case 'cswap':   ops.push(gateOp('cirq.CSWAP', [op.control, op.a, op.b])); break
        case 'measure': throw new TypeError('measure ops cannot be serialized to Cirq via toCirc(); use cirq.measure() manually')
        case 'reset':   throw new TypeError('reset ops cannot be serialized to Cirq via toCirq()')
        case 'if':      throw new TypeError('if ops cannot be serialized to Cirq')
        case 'two': {
          const n = op.meta?.name
          throw new TypeError(`Gate '${n ?? 'two'}' has no Cirq representation`)
        }
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          switch (n) {
            case 'h':   ops.push(gateOp('cirq.H', [q]));                         break
            case 'x':   ops.push(gateOp('cirq.X', [q]));                         break
            case 'y':   ops.push(gateOp('cirq.Y', [q]));                         break
            case 'z':   ops.push(gateOp('cirq.Z', [q]));                         break
            case 's':   ops.push(gateOp('cirq.S', [q]));                         break
            case 'si':  ops.push(`    cirq.ZPowGate(exponent=-0.5)(q[${q}]),`);  break
            case 't':   ops.push(gateOp('cirq.T', [q]));                         break
            case 'ti':  ops.push(`    cirq.ZPowGate(exponent=-0.25)(q[${q}]),`); break
            case 'v':   ops.push(`    cirq.X**0.5(q[${q}]),`);                   break
            case 'vi':  ops.push(`    (cirq.X**-0.5)(q[${q}]),`);                break
            case 'r2':  ops.push(`    cirq.rz(${rads(Math.PI / 2)})(q[${q}]),`); break
            case 'r4':  ops.push(`    cirq.rz(${rads(Math.PI / 4)})(q[${q}]),`); break
            case 'r8':  ops.push(`    cirq.rz(${rads(Math.PI / 8)})(q[${q}]),`); break
            case 'rx':  ops.push(`    cirq.rx(${rads(p![0]!)})(q[${q}]),`);      break
            case 'ry':  ops.push(`    cirq.ry(${rads(p![0]!)})(q[${q}]),`);      break
            case 'rz':  ops.push(`    cirq.rz(${rads(p![0]!)})(q[${q}]),`);      break
            case 'u1':  ops.push(`    cirq.ZPowGate(exponent=${pyAngle(p![0]! / Math.PI)})(q[${q}]),`); break
            case 'u3':  throw new TypeError('U3 has no direct Cirq equivalent; use cirq.MatrixGate(np.array([...])) with the explicit unitary')
            case 'gpi': case 'gpi2': throw new TypeError(`Gate '${n}' has no Cirq representation`)
            default:    throw new TypeError(`Gate '${n}' has no Cirq representation`)
          }
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { name: n, params: p } = op.meta
          const [c, t] = [op.control, op.target]
          switch (n) {
            case 'cy':   ops.push(`    cirq.Y.controlled()(q[${c}], q[${t}]),`);                       break
            case 'cz':   ops.push(gateOp('cirq.CZ', [c, t]));                                          break
            case 'ch':   ops.push(`    cirq.H.controlled()(q[${c}], q[${t}]),`);                       break
            case 'cr2':  ops.push(`    cirq.rz(${rads(Math.PI/2)}).controlled()(q[${c}], q[${t}]),`);  break
            case 'cr4':  ops.push(`    cirq.rz(${rads(Math.PI/4)}).controlled()(q[${c}], q[${t}]),`);  break
            case 'cr8':  ops.push(`    cirq.rz(${rads(Math.PI/8)}).controlled()(q[${c}], q[${t}]),`);  break
            case 'crx':  ops.push(`    cirq.rx(${rads(p![0]!)}).controlled()(q[${c}], q[${t}]),`);     break
            case 'cry':  ops.push(`    cirq.ry(${rads(p![0]!)}).controlled()(q[${c}], q[${t}]),`);     break
            case 'crz':  ops.push(`    cirq.rz(${rads(p![0]!)}).controlled()(q[${c}], q[${t}]),`);     break
            case 'cu1':  ops.push(`    cirq.ZPowGate(exponent=${pyAngle(p![0]! / Math.PI)}).controlled()(q[${c}], q[${t}]),`); break
            case 'cs':   ops.push(`    cirq.ZPowGate(exponent=0.5).controlled()(q[${c}], q[${t}]),`);  break
            case 'ct':   ops.push(`    cirq.ZPowGate(exponent=0.25).controlled()(q[${c}], q[${t}]),`); break
            case 'csdg': ops.push(`    cirq.ZPowGate(exponent=-0.5).controlled()(q[${c}], q[${t}]),`); break
            case 'ctdg': ops.push(`    cirq.ZPowGate(exponent=-0.25).controlled()(q[${c}], q[${t}]),`);break
            default:     throw new TypeError(`Gate '${n}' has no Cirq representation`)
          }
          break
        }
      }
    }

    const body = ops.length ? ops.join('\n') : '    # empty circuit'
    return [
      'import cirq',
      'import math',
      '',
      `q = cirq.LineQubit.range(${this.qubits})`,
      'circuit = cirq.Circuit([',
      body,
      '])',
    ].join('\n')
  }

  /**
   * Emit a Q# operation for Microsoft Azure Quantum.
   * Gate coverage: H/X/Y/Z/S/T and adjoints, Rx/Ry/Rz, CNOT/CZ/SWAP/CCNOT,
   * controlled rotations via Controlled Rx/Ry/Rz. Throws for gpi/gpi2/ms/two-qubit interaction gates/if.
   */
  toQSharp(): string {
    const pi = 'PI()'
    const a = (r: number) => fmtAngle(r, pi).replace(/(\d+)\*PI/, '$1.0*PI').replace(/PI\(?\)?\/(\d+)/, `PI()/${`$1`.padStart(1)}`)
    // Q# needs float literals for division: PI()/2.0 not PI()/2
    const qsharpAngle = (r: number): string => {
      if (Math.abs(r) < 1e-14) return '0.0'
      const f = r / Math.PI
      for (const d of [1, 2, 3, 4, 6, 8, 12, 16]) {
        for (let n = -16; n <= 16; n++) {
          if (n === 0) continue
          if (Math.abs(f - n / d) < 1e-12) {
            const sign = n < 0 ? '-' : ''
            const abs = Math.abs(n)
            if (d === 1) return abs === 1 ? `${sign}PI()` : `${sign}${abs}.0*PI()`
            return abs === 1 ? `${sign}PI()/${d}.0` : `${sign}${abs}.0*PI()/${d}.0`
          }
        }
      }
      return r.toFixed(15)
    }

    const body: string[] = []
    for (const op of this.#ops) {
      switch (op.kind) {
        case 'cnot':    body.push(`        CNOT(q[${op.control}], q[${op.target}]);`);                                         break
        case 'swap':    body.push(`        SWAP(q[${op.a}], q[${op.b}]);`);                                                    break
        case 'toffoli': body.push(`        CCNOT(q[${op.c1}], q[${op.c2}], q[${op.target}]);`);                               break
        case 'cswap':   body.push(`        Controlled SWAP([q[${op.control}]], (q[${op.a}], q[${op.b}]));`);                   break
        case 'measure': body.push(`        set ${op.creg}w${op.bit} = M(q[${op.q}]) == One;`);                                 break
        case 'reset':   body.push(`        Reset(q[${op.q}]);`);                                                               break
        case 'if':      throw new TypeError('if ops cannot be serialized to Q#')
        case 'two':     throw new TypeError(`Gate '${op.meta?.name ?? 'two'}' has no Q# representation`)
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          const ang = () => qsharpAngle(p![0]!)
          switch (n) {
            case 'h':   body.push(`        H(q[${q}]);`);                                   break
            case 'x':   body.push(`        X(q[${q}]);`);                                   break
            case 'y':   body.push(`        Y(q[${q}]);`);                                   break
            case 'z':   body.push(`        Z(q[${q}]);`);                                   break
            case 's':   body.push(`        S(q[${q}]);`);                                   break
            case 'si':  body.push(`        Adjoint S(q[${q}]);`);                           break
            case 't':   body.push(`        T(q[${q}]);`);                                   break
            case 'ti':  body.push(`        Adjoint T(q[${q}]);`);                           break
            case 'v':   body.push(`        Rx(PI()/2.0, q[${q}]);`);                        break  // √X ≡ Rx(π/2) up to global phase
            case 'vi':  body.push(`        Rx(-PI()/2.0, q[${q}]);`);                       break
            case 'r2':  body.push(`        Rz(PI()/2.0, q[${q}]);`);                        break
            case 'r4':  body.push(`        Rz(PI()/4.0, q[${q}]);`);                        break
            case 'r8':  body.push(`        Rz(PI()/8.0, q[${q}]);`);                        break
            case 'rx':  body.push(`        Rx(${ang()}, q[${q}]);`);                        break
            case 'ry':  body.push(`        Ry(${ang()}, q[${q}]);`);                        break
            case 'rz':  body.push(`        Rz(${ang()}, q[${q}]);`);                        break
            case 'u1':  body.push(`        Rz(${ang()}, q[${q}]);`);                        break  // U1 = Rz up to global phase
            case 'gpi': case 'gpi2': throw new TypeError(`Gate '${n}' has no Q# representation`)
            default:    throw new TypeError(`Gate '${n}' has no Q# representation`)
          }
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { name: n, params: p } = op.meta
          const [c, t] = [op.control, op.target]
          const ang = () => qsharpAngle(p![0]!)
          switch (n) {
            case 'cy':   body.push(`        Controlled Y([q[${c}]], q[${t}]);`);                    break
            case 'cz':   body.push(`        CZ(q[${c}], q[${t}]);`);                               break
            case 'ch':   body.push(`        Controlled H([q[${c}]], q[${t}]);`);                    break
            case 'crx':  body.push(`        Controlled Rx([q[${c}]], (${ang()}, q[${t}]));`);       break
            case 'cry':  body.push(`        Controlled Ry([q[${c}]], (${ang()}, q[${t}]));`);       break
            case 'crz':  body.push(`        Controlled Rz([q[${c}]], (${ang()}, q[${t}]));`);       break
            case 'cr2':  body.push(`        Controlled Rz([q[${c}]], (PI()/2.0, q[${t}]));`);      break
            case 'cr4':  body.push(`        Controlled Rz([q[${c}]], (PI()/4.0, q[${t}]));`);      break
            case 'cr8':  body.push(`        Controlled Rz([q[${c}]], (PI()/8.0, q[${t}]));`);      break
            case 'cs':   body.push(`        Controlled S([q[${c}]], q[${t}]);`);                    break
            case 'csdg': body.push(`        Controlled Adjoint S([q[${c}]], q[${t}]);`);            break
            case 'ct':   body.push(`        Controlled T([q[${c}]], q[${t}]);`);                    break
            case 'ctdg': body.push(`        Controlled Adjoint T([q[${c}]], q[${t}]);`);            break
            case 'cu1':  body.push(`        Controlled Rz([q[${c}]], (${ang()}, q[${t}]));`);       break
            default:     throw new TypeError(`Gate '${n}' has no Q# representation`)
          }
          break
        }
      }
    }

    const qregs = `        use q = Qubit[${this.qubits}];`
    const reset = `        ResetAll(q);`
    return [
      'namespace KetCircuit {',
      '    open Microsoft.Quantum.Intrinsic;',
      '    open Microsoft.Quantum.Canon;',
      '    open Microsoft.Quantum.Math;',
      '',
      '    operation Run() : Unit {',
      qregs,
      ...(body.length ? ['', ...body, ''] : []),
      reset,
      '    }',
      '}',
    ].join('\n')
  }

  /**
   * Emit Python code for Rigetti's pyQuil.
   * Gate coverage: H/X/Y/Z/S/T (Sdg/Tdg via DAGGER), RX/RY/RZ,
   * CNOT/CZ/SWAP/CCNOT/CSWAP/ISWAP. Throws for controlled-rotation gates, U-gates, gpi/gpi2/ms/if.
   */
  toPyQuil(): string {
    const used = new Set<string>()
    const body: string[] = []

    for (const op of this.#ops) {
      switch (op.kind) {
        case 'cnot':    used.add('CNOT');  body.push(`p += CNOT(${op.control}, ${op.target})`);          break
        case 'swap':    used.add('SWAP');  body.push(`p += SWAP(${op.a}, ${op.b})`);                     break
        case 'toffoli': used.add('CCNOT'); body.push(`p += CCNOT(${op.c1}, ${op.c2}, ${op.target})`);   break
        case 'cswap':   used.add('CSWAP'); body.push(`p += CSWAP(${op.control}, ${op.a}, ${op.b})`);    break
        case 'measure': used.add('MEASURE'); body.push(`p += MEASURE(${op.q}, ro[${op.bit}])`);          break
        case 'reset':   body.push(`p += RESET(${op.q})`);                                                break
        case 'if':      throw new TypeError('if ops cannot be serialized to pyQuil')
        case 'two': {
          const n = op.meta?.name
          if (n === 'iswap') { used.add('ISWAP'); body.push(`p += ISWAP(${op.a}, ${op.b})`); break }
          throw new TypeError(`Gate '${n ?? 'two'}' has no pyQuil representation`)
        }
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          const ang = (i = 0) => pyAngle(p![i]!)
          switch (n) {
            case 'h':   used.add('H');   body.push(`p += H(${q})`);              break
            case 'x':   used.add('X');   body.push(`p += X(${q})`);              break
            case 'y':   used.add('Y');   body.push(`p += Y(${q})`);              break
            case 'z':   used.add('Z');   body.push(`p += Z(${q})`);              break
            case 's':   used.add('S');   body.push(`p += S(${q})`);              break
            case 'si':  used.add('S'); used.add('DAGGER');  body.push(`p += DAGGER(S)(${q})`); break
            case 't':   used.add('T');   body.push(`p += T(${q})`);              break
            case 'ti':  used.add('T'); used.add('DAGGER');  body.push(`p += DAGGER(T)(${q})`); break
            case 'v':   used.add('RX');  body.push(`p += RX(math.pi/2, ${q})`); break  // √X up to global phase
            case 'vi':  used.add('RX');  body.push(`p += RX(-math.pi/2, ${q})`);break
            case 'r2':  used.add('RZ');  body.push(`p += RZ(math.pi/2, ${q})`); break
            case 'r4':  used.add('RZ');  body.push(`p += RZ(math.pi/4, ${q})`); break
            case 'r8':  used.add('RZ');  body.push(`p += RZ(math.pi/8, ${q})`); break
            case 'rx':  used.add('RX');  body.push(`p += RX(${ang()}, ${q})`);   break
            case 'ry':  used.add('RY');  body.push(`p += RY(${ang()}, ${q})`);   break
            case 'rz':  used.add('RZ');  body.push(`p += RZ(${ang()}, ${q})`);   break
            case 'gpi': case 'gpi2': throw new TypeError(`Gate '${n}' has no pyQuil representation`)
            default: throw new TypeError(`Gate '${n}' has no pyQuil representation`)
          }
          break
        }
        case 'controlled':
          throw new TypeError(`Gate '${op.meta?.name ?? 'controlled'}' has no standard pyQuil representation`)
      }
    }

    const gateImports = [...used].filter(g => g !== 'RESET' && g !== 'MEASURE').sort()
    const extras: string[] = []
    if (used.has('MEASURE')) extras.push('from pyquil.quilbase import MemoryReference')
    if (used.has('RESET'))   extras.push('# Reset: use p += RESET() or p.reset() as appropriate')

    return [
      'from pyquil import Program',
      gateImports.length ? `from pyquil.gates import ${gateImports.join(', ')}` : '',
      ...extras,
      'import math',
      '',
      'p = Program()',
      ...body,
    ].filter((l, i) => l !== '' || i > 3).join('\n')  // collapse leading blank lines
  }

  // ── Execution ────────────────────────────────────────────────────────────

  /** Run the circuit and return a probability distribution. */
  run({ shots = 1024, seed, noise }: RunOptions = {}): Distribution {
    const rng = makePrng(seed)

    // Resolve noise: named device profile → NoiseParams, or use as-is
    const noiseParams: NoiseParams | undefined =
      noise == null        ? undefined :
      typeof noise === 'string' ? (() => {
        const p = DEVICE_NOISE[noise]
        if (!p) throw new TypeError(`Unknown device profile '${noise}'. Known: ${Object.keys(DEVICE_NOISE).join(', ')}`)
        return p
      })() : noise

    const cregCounts = new Map<string, number[]>(
      this.#cregs.entries().map(([name, size]) => [name, new Array<number>(size).fill(0)])
    )

    // ── Fast path: pure circuit without noise — simulate once, sample N times ──
    if (!noiseParams && !this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
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

    // ── Per-shot path: noise or mid-circuit ops — one full simulation per shot ──
    const counts = new Map<bigint, number>()
    const pMeas  = noiseParams?.pMeas ?? 0

    for (let i = 0; i < shots; i++) {
      const shotCregs = new Map<string, boolean[]>(
        this.#cregs.entries().map(([name, size]) => [name, new Array<boolean>(size).fill(false)])
      )

      const sv = applyOps(this.#ops, zero(this.qubits), shotCregs, rng, noiseParams)

      // Final readout: sample then apply SPAM noise per qubit
      let finalIdx = sampleSV(sv, rng())
      if (pMeas) {
        for (let q = 0; q < this.qubits; q++) {
          if (rng() < pMeas) finalIdx ^= (1n << BigInt(q))
        }
      }

      counts.set(finalIdx, (counts.get(finalIdx) ?? 0) + 1)
      for (const [name, bits] of shotCregs) {
        const acc = cregCounts.get(name)!
        for (const [j, b] of bits.entries()) if (b) acc[j]! += 1
      }
    }

    return new Distribution(this.qubits, shots, counts, cregCounts)
  }

  /**
   * Run the circuit using MPS (tensor-network) simulation.
   *
   * Efficient for circuits with bounded entanglement (GHZ, BV, shallow QFT, QAOA low-depth).
   * Memory: O(n · χ² · 2) vs O(2ⁿ) for full statevector.
   * Handles 50+ qubit circuits that would be intractable with the statevector backend.
   *
   * Limitations: no noise, no mid-circuit measure/reset/if; Toffoli and CSWAP
   * must be decomposed into single- and two-qubit gates first.
   *
   * @param maxBond Maximum bond dimension χ. Default 64 — exact for GHZ/BV, approximate for deep random circuits.
   */
  runMps({ shots = 1024, seed, maxBond = 64 }: MpsRunOptions = {}): Distribution {
    const rng = makePrng(seed)
    let state = mpsInit(this.qubits)

    for (const op of this.#ops) {
      switch (op.kind) {
        case 'single':     state = mpsApply1(state, op.q, op.gate);                                        break
        case 'cnot':       state = mpsApply2(state, op.control, op.target, CNOT4, maxBond);                break
        case 'swap':       state = mpsApply2(state, op.a, op.b, SWAP4, maxBond);                           break
        case 'two':        state = mpsApply2(state, op.a, op.b, op.gate, maxBond);                         break
        case 'controlled': state = mpsApply2(state, op.control, op.target, controlledGate(op.gate), maxBond); break
        case 'toffoli': throw new TypeError('CCX (Toffoli) not supported in MPS mode; decompose into CX gates')
        case 'cswap':   throw new TypeError('CSWAP (Fredkin) not supported in MPS mode; decompose into CX gates')
        case 'measure': case 'reset': case 'if':
          throw new TypeError(`'${op.kind}' not supported in MPS mode`)
      }
    }

    const counts = new Map<bigint, number>()
    for (let i = 0; i < shots; i++) {
      const idx = mpsSample(state, rng)
      counts.set(idx, (counts.get(idx) ?? 0) + 1)
    }

    const cregCounts = new Map<string, number[]>(
      this.#cregs.entries().map(([name, size]) => [name, new Array<number>(size).fill(0)])
    )
    return new Distribution(this.qubits, shots, counts, cregCounts)
  }

  /**
   * Return exact floating-point probabilities from the statevector — no sampling variance.
   *
   * Keys are IonQ bitstrings (q0 rightmost). Only non-negligible amplitudes are included.
   * Throws for circuits containing mid-circuit measure, reset, or conditional ops.
   */
  exactProbs(): Readonly<Record<string, number>> {
    if (this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      throw new TypeError('exactProbs() requires a pure circuit — no measure, reset, or if ops')
    }
    const sv = simulatePure(this.#ops, this.qubits)
    const out: Record<string, number> = {}
    for (const [idx, p] of probabilities(sv)) {
      out[idx.toString(2).padStart(this.qubits, '0')] = p
    }
    return Object.freeze(out)
  }
}
