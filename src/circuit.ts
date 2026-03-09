/**
 * Immutable circuit builder with IonQ QIS gate names.
 *
 * Each gate method returns a new Circuit — circuits are values, not mutated objects.
 * Simulation is lazy: nothing runs until `.run()` is called.
 */

import * as G from './gates.js'
import { applyCNOT, applyControlled, applyCsrSwap, applyCSwap, applySingle, applySWAP, applyToffoli, applyTwo, applyUnitary, Gate2x2, Gate4x4, probabilities, StateVector, zero } from './statevector.js'
import { Complex, ZERO } from './complex.js'
import { CNOT4, controlledGate, mpsApply1, mpsApply2, mpsInit, mpsSample, SWAP4 } from './mps.js'
import { DensityMatrix, DM_DEVICE_NOISE, DmNoiseParams, runDM } from './density.js'
import { CliffordSim } from './clifford.js'

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
type CsrSwapOp    = { kind: 'csrswap';    control: number; a: number; b: number }
type MeasureOp    = { kind: 'measure';    q: number; creg: string; bit: number }
type ResetOp      = { kind: 'reset';      q: number }
type IfOp         = { kind: 'if';         creg: string; value: number; ops: readonly Op[] }
/** A user-defined named gate applied to a specific set of parent-circuit qubits. */
type SubcircuitOp = { kind: 'subcircuit'; name: string; qubits: readonly number[]; def: readonly Op[] }
/** Scheduling/grouping hint — no effect on the statevector; emitted as `barrier` in QASM. */
type BarrierOp    = { kind: 'barrier';    qubits: readonly number[] }
/** Arbitrary N-qubit unitary gate defined by its 2^N × 2^N matrix. */
type UnitaryOp    = { kind: 'unitary';    qubits: readonly number[]; matrix: readonly (readonly Complex[])[] }
type Op = SingleOp | CNOTOp | SWAPOp | TwoOp | ControlledOp | ToffoliOp | CSwapOp | CsrSwapOp | MeasureOp | ResetOp | IfOp | SubcircuitOp | BarrierOp | UnitaryOp

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
  const sorted = Array.from(sv.entries()).toSorted(([a], [b]) => (a < b ? -1 : 1))
  let cum = 0
  for (const [idx, amp] of sorted) {
    cum += amp.re * amp.re + amp.im * amp.im
    if (rand <= cum) return idx
  }
  return sorted.at(-1)?.[0] ?? 0n
}

/** Remap every qubit index in `op` through `qmap` (qmap[subcircuit-qubit] = parent-qubit). */
function remapOp(op: Op, qmap: readonly number[]): Op {
  const q = (i: number) => qmap[i]!
  switch (op.kind) {
    case 'single':     return { ...op, q: q(op.q) }
    case 'cnot':       return { ...op, control: q(op.control), target: q(op.target) }
    case 'swap':       return { ...op, a: q(op.a), b: q(op.b) }
    case 'two':        return { ...op, a: q(op.a), b: q(op.b) }
    case 'controlled': return { ...op, control: q(op.control), target: q(op.target) }
    case 'toffoli':    return { ...op, c1: q(op.c1), c2: q(op.c2), target: q(op.target) }
    case 'cswap':      return { ...op, control: q(op.control), a: q(op.a), b: q(op.b) }
    case 'csrswap':    return { ...op, control: q(op.control), a: q(op.a), b: q(op.b) }
    case 'subcircuit': return { ...op, qubits: op.qubits.map(i => q(i)) }
    case 'unitary':    return { ...op, qubits: op.qubits.map(i => q(i)) }
    case 'barrier':    return { ...op, qubits: op.qubits.map(i => q(i)) }
    case 'measure':    return op  // measure/reset/if are forbidden inside defineGate bodies
    case 'reset':      return op
    case 'if':         return op
    default: {
      const _exhaustive: never = op
      return _exhaustive
    }
  }
}

/**
 * Recursively expand all SubcircuitOps into their constituent primitive ops,
 * remapping qubit indices at each level. Returns a flat array with no subcircuit ops.
 */
function flattenOps(ops: readonly Op[]): readonly Op[] {
  let hasSubcircuit = false
  for (const op of ops) if (op.kind === 'subcircuit') { hasSubcircuit = true; break }
  if (!hasSubcircuit) return ops  // fast path — no allocation for pure-primitive circuits

  const result: Op[] = []
  for (const op of ops) {
    if (op.kind === 'subcircuit') {
      result.push(...flattenOps(op.def.map(inner => remapOp(inner, op.qubits))))
    } else {
      result.push(op)
    }
  }
  return result
}

/** Build a computational-basis statevector from a bitstring (q0 rightmost, IonQ convention). */
function svFromBitstring(s: string, qubits: number): StateVector {
  if (s.length !== qubits || !/^[01]+$/.test(s))
    throw new TypeError(`initialState '${s}' must be a ${qubits}-character binary string`)
  return new Map([[BigInt('0b' + s), { re: 1, im: 0 }]])
}

/** Simulate a pure (no measure/reset/if) circuit and return the statevector. */
function simulatePure(ops: readonly Op[], qubits: number, init?: StateVector): StateVector {
  let sv: StateVector = init ?? zero(qubits)
  for (const op of flattenOps(ops)) {
    switch (op.kind) {
      case 'single':     sv = applySingle(sv, op.q, op.gate); break
      case 'cnot':       sv = applyCNOT(sv, op.control, op.target); break
      case 'controlled': sv = applyControlled(sv, op.control, op.target, op.gate); break
      case 'swap':       sv = applySWAP(sv, op.a, op.b); break
      case 'toffoli':    sv = applyToffoli(sv, op.c1, op.c2, op.target); break
      case 'cswap':      sv = applyCSwap(sv, op.control, op.a, op.b); break
      case 'csrswap':    sv = applyCsrSwap(sv, op.control, op.a, op.b); break
      case 'two':        sv = applyTwo(sv, op.a, op.b, op.gate); break
      case 'unitary':    sv = applyUnitary(sv, op.qubits, op.matrix); break
      case 'barrier': case 'measure': case 'reset': case 'if': case 'subcircuit': break
      default: { const _exhaustive: never = op; void _exhaustive }
    }
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
  for (const op of flattenOps(ops)) {
    switch (op.kind) {
      case 'single':     sv = applySingle(sv, op.q, op.gate);                         if (p1) sv = dep1(sv, op.q, p1, rng()); break
      case 'cnot':       sv = applyCNOT(sv, op.control, op.target);                   if (p2) sv = dep2(sv, op.control, op.target, p2, rng()); break
      case 'controlled': sv = applyControlled(sv, op.control, op.target, op.gate);    if (p2) sv = dep2(sv, op.control, op.target, p2, rng()); break
      case 'swap':       sv = applySWAP(sv, op.a, op.b);                              if (p2) sv = dep2(sv, op.a, op.b, p2, rng()); break
      case 'toffoli':    sv = applyToffoli(sv, op.c1, op.c2, op.target); break
      case 'cswap':      sv = applyCSwap(sv, op.control, op.a, op.b); break
      case 'csrswap':    sv = applyCsrSwap(sv, op.control, op.a, op.b); break
      case 'two':        sv = applyTwo(sv, op.a, op.b, op.gate);                      if (p2) sv = dep2(sv, op.a, op.b, p2, rng()); break
      case 'unitary':    sv = applyUnitary(sv, op.qubits, op.matrix); break
      case 'measure': {
        const { outcome, sv: next } = collapseQubit(sv, op.q, rng())
        const reported: 0 | 1 = pM && rng() < pM ? (outcome === 1 ? 0 : 1) : outcome
        sv = next
        const reg = shotCregs.get(op.creg)
        if (reg) reg[op.bit] = reported === 1
        break
      }
      case 'reset': {
        const { outcome, sv: next } = collapseQubit(sv, op.q, rng())
        sv = next
        if (outcome === 1) sv = applySingle(sv, op.q, G.X)
        break
      }
      case 'if':
        if (cregValue(shotCregs, op.creg) === op.value) sv = applyOps(op.ops, sv, shotCregs, rng, noise)
        break
      case 'barrier': case 'subcircuit': break
      default: { const _exhaustive: never = op; void _exhaustive }
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

/** Per-device specs: qubit capacity, hardware-native gate set, and published noise figures. */
export interface IonQDeviceInfo {
  /** Maximum number of qubits the device supports. */
  readonly qubits:      number
  /** Gates that run directly on hardware without compiler translation. */
  readonly nativeGates: readonly string[]
  /** Published depolarizing + readout noise parameters (conservative estimates). */
  readonly noise:       Readonly<NoiseParams>
}

/**
 * Published IonQ device specifications.
 * Keys match the strings accepted by `run({ noise })`, `dm({ noise })`, and `checkDevice()`.
 */
export const IONQ_DEVICES: Readonly<Record<string, IonQDeviceInfo>> = {
  'aria-1':  { qubits: 25, nativeGates: ['gpi', 'gpi2', 'ms', 'vz'],        noise: { p1: 0.0003, p2: 0.005,  pMeas: 0.004 } },
  'forte-1': { qubits: 36, nativeGates: ['gpi', 'gpi2', 'ms', 'vz', 'zz'],  noise: { p1: 0.0001, p2: 0.002,  pMeas: 0.002 } },
  'harmony': { qubits: 11, nativeGates: ['gpi', 'gpi2', 'ms', 'vz'],         noise: { p1: 0.001,  p2: 0.015,  pMeas: 0.01  } },
}

const DEVICE_NOISE: Readonly<Record<string, NoiseParams>> =
  Object.fromEntries(Object.entries(IONQ_DEVICES).map(([k, v]) => [k, v.noise]))

// 15 non-identity two-qubit Paulis for depolarizing channel: {I,X,Y,Z}⊗{I,X,Y,Z} \ {II}
const TWO_PAULI: readonly (readonly [Gate2x2 | null, Gate2x2 | null])[] = [
  [null, G.X], [null, G.Y], [null, G.Z],
  [G.X, null], [G.X, G.X], [G.X, G.Y], [G.X, G.Z],
  [G.Y, null], [G.Y, G.X], [G.Y, G.Y], [G.Y, G.Z],
  [G.Z, null], [G.Z, G.X], [G.Z, G.Y], [G.Z, G.Z],
]

// Same 15 Paulis encoded as (0=I,1=X,2=Y,3=Z) pairs — used by Clifford depolarizing channel.
const TWO_PAULI_IDX: readonly [number, number][] = [
  [0,1],[0,2],[0,3],
  [1,0],[1,1],[1,2],[1,3],
  [2,0],[2,1],[2,2],[2,3],
  [3,0],[3,1],[3,2],[3,3],
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
    case 'vz':   return { qname: 'rz',   qparams: [...(meta.params ?? [])] }  // VirtualZ ≡ Rz
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
    case 'id':      return c.id(a!)
    case 'barrier': return qs.length ? c.barrier(...qs) : c.barrier()
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
    case 'srn':   return c.srn(a!)
    case 'srndg': return c.srndg(a!)
    case 'p':     return c.p(p0!, a!)
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
    case 'cu2':   return c.cu2(p0!, p1!, a!, b!)
    case 'cu3':   return c.cu3(p0!, p1!, p2!, a!, b!)
    case 'swap':  return c.swap(a!, b!)
    case 'ccx':   return c.ccx(a!, b!, d!)
    case 'cswap': return c.cswap(a!, b!, d!)
    case 'csrn':  return c.csrn(a!, b!)
    default: throw new TypeError(`Unknown QASM gate: '${name}'`)
  }
}

// ─── JSON serialization helpers ───────────────────────────────────────────────

/** The versioned JSON schema produced by `circuit.toJSON()` and consumed by `Circuit.fromJSON()`. */
export interface CircuitJSON {
  /** Format version — always 1 in this release. */
  readonly ket: 1
  readonly qubits: number
  /** Classical register sizes: `{ c: 2 }` means register "c" has 2 bits. */
  readonly cregs: Readonly<Record<string, number>>
  /** Named gate definitions registered via `defineGate()`. */
  readonly gates: Readonly<Record<string, { readonly qubits: number; readonly ops: readonly unknown[] }>>
  readonly ops: readonly unknown[]
}

/** Reconstruct the Gate2x2 matrix for a single-qubit op from its serialized meta. */
function gate2x2FromMeta(name: string, params: readonly number[]): Gate2x2 {
  const p = params
  switch (name) {
    case 'h':    return G.H;  case 'x':   return G.X;  case 'y': return G.Y;  case 'z': return G.Z
    case 's':    return G.S;  case 'si':  return G.Si; case 'sdg': return G.Si
    case 't':    return G.T;  case 'ti':  return G.Ti; case 'tdg': return G.Ti
    case 'v':    return G.V;  case 'vi':  return G.Vi; case 'srn': return G.V; case 'srndg': return G.Vi
    case 'sx':   return G.V;  case 'sxdg': return G.Vi
    case 'id':   return G.Id
    case 'rx':   return G.Rx(p[0]!);   case 'ry': return G.Ry(p[0]!); case 'rz': return G.Rz(p[0]!)
    case 'vz':   return G.Rz(p[0]!)   // VirtualZ ≡ Rz
    case 'r2':   return G.R2;          case 'r4': return G.R4;         case 'r8': return G.R8
    case 'u1':   return G.U1(p[0]!);  case 'p': return G.U1(p[0]!)   // P gate = U1 (Qiskit 1.0+)
    case 'u2':   return G.U2(p[0]!, p[1]!)
    case 'u3':   return G.U3(p[0]!, p[1]!, p[2]!)
    case 'gpi':  return G.Gpi(p[0]!);  case 'gpi2': return G.Gpi2(p[0]!)
    default: throw new TypeError(`fromJSON: unknown single-qubit gate '${name}'`)
  }
}

/** Reconstruct the Gate4x4 matrix for a two-qubit op from its serialized meta. */
function gate4x4FromMeta(name: string, params: readonly number[]): Gate4x4 {
  const p = params
  switch (name) {
    case 'xx':     return G.Xx(p[0]!)
    case 'yy':     return G.Yy(p[0]!)
    case 'zz':     return G.Zz(p[0]!)
    case 'xy':     return G.Xy(p[0]!)
    case 'iswap':  return G.ISwap
    case 'srswap': return G.SrSwap
    case 'ms':     return G.Ms(p[0]!, p[1]!)
    default: throw new TypeError(`fromJSON: unknown two-qubit gate '${name}'`)
  }
}

/** Reconstruct the Gate2x2 for the target qubit of a controlled op. */
function ctrlGate2x2FromMeta(name: string, params: readonly number[]): Gate2x2 {
  const p = params
  switch (name) {
    case 'cx':   return G.X;  case 'cy':   return G.Y;  case 'cz':   return G.Z;  case 'ch':   return G.H
    case 'crx':  return G.Rx(p[0]!);  case 'cry': return G.Ry(p[0]!);  case 'crz': return G.Rz(p[0]!)
    case 'cu1':  return G.U1(p[0]!)
    case 'cu2':  return G.U2(p[0]!, p[1]!)
    case 'cu3':  return G.U3(p[0]!, p[1]!, p[2]!)
    case 'cs':   return G.S;  case 'ct':   return G.T;  case 'csdg': return G.Si; case 'ctdg': return G.Ti
    case 'cr2':  return G.R2; case 'cr4':  return G.R4; case 'cr8':  return G.R8
    case 'csrn': return G.V   // Controlled-√NOT
    default: throw new TypeError(`fromJSON: unknown controlled gate '${name}'`)
  }
}

/** Deserialize a raw JSON op array into typed Op objects. */
function opsFromJSON(raw: readonly unknown[]): Op[] {
  return (raw as Record<string, unknown>[]).map(o => {
    const kind = o['kind'] as string
    switch (kind) {
      case 'single': {
        const meta = o['meta'] as GateMeta
        return { kind: 'single', q: o['q'] as number, gate: gate2x2FromMeta(meta.name, meta.params ?? []), meta } satisfies SingleOp
      }
      case 'cnot':
        return { kind: 'cnot', control: o['control'] as number, target: o['target'] as number } satisfies CNOTOp
      case 'swap':
        return { kind: 'swap', a: o['a'] as number, b: o['b'] as number } satisfies SWAPOp
      case 'two': {
        const meta = o['meta'] as GateMeta
        return { kind: 'two', a: o['a'] as number, b: o['b'] as number, gate: gate4x4FromMeta(meta.name, meta.params ?? []), meta } satisfies TwoOp
      }
      case 'controlled': {
        const meta = o['meta'] as GateMeta
        return { kind: 'controlled', control: o['control'] as number, target: o['target'] as number, gate: ctrlGate2x2FromMeta(meta.name, meta.params ?? []), meta } satisfies ControlledOp
      }
      case 'toffoli':
        return { kind: 'toffoli', c1: o['c1'] as number, c2: o['c2'] as number, target: o['target'] as number } satisfies ToffoliOp
      case 'cswap':
        return { kind: 'cswap', control: o['control'] as number, a: o['a'] as number, b: o['b'] as number } satisfies CSwapOp
      case 'csrswap':
        return { kind: 'csrswap', control: o['control'] as number, a: o['a'] as number, b: o['b'] as number } satisfies CsrSwapOp
      case 'measure':
        return { kind: 'measure', q: o['q'] as number, creg: o['creg'] as string, bit: o['bit'] as number } satisfies MeasureOp
      case 'reset':
        return { kind: 'reset', q: o['q'] as number } satisfies ResetOp
      case 'if':
        return { kind: 'if', creg: o['creg'] as string, value: o['value'] as number, ops: opsFromJSON(o['ops'] as unknown[]) } satisfies IfOp
      case 'subcircuit':
        return { kind: 'subcircuit', name: o['name'] as string, qubits: o['qubits'] as number[], def: opsFromJSON(o['def'] as unknown[]) } satisfies SubcircuitOp
      case 'barrier':
        return { kind: 'barrier', qubits: o['qubits'] as number[] } satisfies BarrierOp
      case 'unitary': {
        const rawMatrix = o['matrix'] as [number, number][][]
        const matrix: Complex[][] = rawMatrix.map(row => row.map(([re, im]) => ({ re, im })))
        return { kind: 'unitary', qubits: o['qubits'] as number[], matrix } satisfies UnitaryOp
      }
      default:
        throw new TypeError(`fromJSON: unknown op kind '${kind}'`)
    }
  })
}

/** Serialize ops to a plain JSON-safe array (no gate matrices — reconstructed on load). */
function opsToJSON(ops: readonly Op[]): unknown[] {
  return ops.map(op => {
    switch (op.kind) {
      case 'single':     return { kind: 'single', q: op.q, meta: op.meta }
      case 'cnot':       return { kind: 'cnot', control: op.control, target: op.target }
      case 'swap':       return { kind: 'swap', a: op.a, b: op.b }
      case 'two':        return { kind: 'two', a: op.a, b: op.b, meta: op.meta }
      case 'controlled': return { kind: 'controlled', control: op.control, target: op.target, meta: op.meta }
      case 'toffoli':    return { kind: 'toffoli', c1: op.c1, c2: op.c2, target: op.target }
      case 'cswap':      return { kind: 'cswap', control: op.control, a: op.a, b: op.b }
      case 'csrswap':    return { kind: 'csrswap', control: op.control, a: op.a, b: op.b }
      case 'measure':    return { kind: 'measure', q: op.q, creg: op.creg, bit: op.bit }
      case 'reset':      return { kind: 'reset', q: op.q }
      case 'if':         return { kind: 'if', creg: op.creg, value: op.value, ops: opsToJSON(op.ops) }
      case 'subcircuit': return { kind: 'subcircuit', name: op.name, qubits: [...op.qubits], def: opsToJSON(op.def) }
      case 'barrier':    return { kind: 'barrier', qubits: [...op.qubits] }
      case 'unitary':    return { kind: 'unitary', qubits: [...op.qubits], matrix: op.matrix.map(row => row.map(({ re, im }) => [re, im])) }
      default: {
        const _exhaustive: never = op
        return _exhaustive
      }
    }
  })
}

// ─── LaTeX helpers ────────────────────────────────────────────────────────────

/** Format a radian value as a LaTeX math expression using \frac{}{} for clean fractions. */
function latexAngle(r: number): string {
  if (Math.abs(r) < 1e-14) return '0'
  const f = r / Math.PI
  for (const d of [1, 2, 3, 4, 6, 8, 12, 16]) {
    for (let n = -16; n <= 16; n++) {
      if (n === 0) continue
      if (Math.abs(f - n / d) < 1e-12) {
        const sign = n < 0 ? '-' : ''
        const a = Math.abs(n)
        if (d === 1) return a === 1 ? `${sign}\\pi` : `${sign}${a}\\pi`
        return a === 1 ? `${sign}\\frac{\\pi}{${d}}` : `${sign}\\frac{${a}\\pi}{${d}}`
      }
    }
  }
  return String(r)
}

function latexSingleLabel(op: SingleOp): string {
  const nm = op.meta?.name, p = op.meta?.params ?? []
  const a  = (i: number) => latexAngle(p[i] ?? 0)
  switch (nm) {
    case 'h': return 'H';  case 'x': return 'X';  case 'y': return 'Y';  case 'z': return 'Z'
    case 's': return 'S';  case 'si': return 'S^\\dagger'; case 't': return 'T'; case 'ti': return 'T^\\dagger'
    case 'v': return '\\sqrt{X}'; case 'vi': return '\\sqrt{X}^\\dagger'; case 'id': return 'I'
    case 'rx': return `R_x(${a(0)})`; case 'ry': return `R_y(${a(0)})`
    case 'rz': case 'vz': return `R_z(${a(0)})`
    case 'r2': return 'R_2'; case 'r4': return 'R_4'; case 'r8': return 'R_8'
    case 'u1': return `U_1(${a(0)})`; case 'u2': return `U_2(${a(0)},${a(1)})`
    case 'u3': return `U_3(${a(0)},${a(1)},${a(2)})`
    case 'gpi': return `\\text{GPI}(${a(0)})`; case 'gpi2': return `\\text{GPI2}(${a(0)})`
    default: return nm ? nm.toUpperCase() : 'U'
  }
}

function latexTwoLabel(op: TwoOp): string {
  const nm = op.meta?.name, p = op.meta?.params ?? []
  const a  = (i: number) => latexAngle(p[i] ?? 0)
  switch (nm) {
    case 'xx': return `XX(${a(0)})`; case 'yy': return `YY(${a(0)})`
    case 'zz': return `ZZ(${a(0)})`; case 'xy': return `XY(${a(0)})`
    case 'iswap': return '\\text{iSWAP}'; case 'srswap': return '\\sqrt{\\text{iSWAP}}'
    case 'ms': return `\\text{MS}(${a(0)},${a(1)})`
    default: return nm ? nm.toUpperCase() : 'U'
  }
}

function latexCtrlTargetLabel(op: ControlledOp): string {
  const nm = op.meta?.name, p = op.meta?.params ?? []
  const a  = (i: number) => latexAngle(p[i] ?? 0)
  switch (nm) {
    case 'cx': return 'X';  case 'cy': return 'Y';  case 'cz': return 'Z';  case 'ch': return 'H'
    case 'crx': return `R_x(${a(0)})`; case 'cry': return `R_y(${a(0)})`; case 'crz': return `R_z(${a(0)})`
    case 'cu1': return `U_1(${a(0)})`; case 'cu2': return `U_2(${a(0)},${a(1)})`
    case 'cu3': return `U_3(${a(0)},${a(1)},${a(2)})`
    case 'cs': return 'S';  case 'ct': return 'T';  case 'csdg': return 'S^\\dagger'; case 'ctdg': return 'T^\\dagger'
    case 'cr2': return 'R_2'; case 'cr4': return 'R_4'; case 'cr8': return 'R_8'
    default: return nm ? nm.slice(1).toUpperCase() : 'U'
  }
}

// ─── Visualization helpers ────────────────────────────────────────────────────

/** Format a radian angle using Unicode π for draw() / toSVG() labels. */
function drawAngle(r: number): string { return fmtAngle(r, 'π') }

/** All qubit indices touched by an op (the full span is handled by the caller). */
function opQubits(op: Op): number[] {
  switch (op.kind) {
    case 'single':     return [op.q]
    case 'cnot':       return [op.control, op.target]
    case 'swap':       return [op.a, op.b]
    case 'two':        return [op.a, op.b]
    case 'controlled': return [op.control, op.target]
    case 'toffoli':    return [op.c1, op.c2, op.target]
    case 'cswap':      return [op.control, op.a, op.b]
    case 'csrswap':    return [op.control, op.a, op.b]
    case 'measure':    return [op.q]
    case 'reset':      return [op.q]
    case 'barrier':    return [...op.qubits]
    case 'subcircuit': return [...op.qubits]
    case 'unitary':    return [...op.qubits]
    default:           return []
  }
}

/** Label displayed for qubit `q`'s role in `op`. */
function opLabel(op: Op, q: number): string {
  const a = (p: readonly number[], i: number) => drawAngle(p[i] ?? 0)
  switch (op.kind) {
    case 'single': {
      const name = op.meta?.name; const p = op.meta?.params ?? []
      switch (name) {
        case 'h': return 'H'; case 'x': return 'X'; case 'y': return 'Y'; case 'z': return 'Z'
        case 's': return 'S'; case 'si': return 'S†'; case 't': return 'T'; case 'ti': return 'T†'
        case 'v': return 'V'; case 'vi': return 'V†'; case 'id': return 'I'
        case 'rx': return `Rx(${a(p,0)})`; case 'ry': return `Ry(${a(p,0)})`
        case 'rz': case 'vz': return `Rz(${a(p,0)})`
        case 'r2': return 'R₂'; case 'r4': return 'R₄'; case 'r8': return 'R₈'
        case 'u1': return `U1(${a(p,0)})`
        case 'u2': return `U2(${a(p,0)},${a(p,1)})`
        case 'u3': return `U3(${a(p,0)},${a(p,1)},${a(p,2)})`
        case 'gpi': return `GPI(${a(p,0)})`; case 'gpi2': return `GPI2(${a(p,0)})`
        default: return name ? name.toUpperCase() : 'U'
      }
    }
    case 'cnot':       return op.control === q ? '●' : '⊕'
    case 'swap':       return '╳'
    case 'two': {
      const name = op.meta?.name; const p = op.meta?.params ?? []
      switch (name) {
        case 'xx': return `XX(${a(p,0)})`; case 'yy': return `YY(${a(p,0)})`
        case 'zz': return `ZZ(${a(p,0)})`; case 'xy': return `XY(${a(p,0)})`
        case 'iswap': return 'iSWAP'; case 'srswap': return '√iSWAP'
        case 'ms': return `MS(${a(p,0)},${a(p,1)})`
        default: return name ? name.toUpperCase() : 'U'
      }
    }
    case 'controlled': {
      if (op.control === q) return '●'
      const name = op.meta?.name; const p = op.meta?.params ?? []
      switch (name) {
        case 'cx': return 'X'; case 'cy': return 'Y'; case 'cz': return 'Z'; case 'ch': return 'H'
        case 'crx': return `Rx(${a(p,0)})`; case 'cry': return `Ry(${a(p,0)})`; case 'crz': return `Rz(${a(p,0)})`
        case 'cu1': return `U1(${a(p,0)})`; case 'cu2': return `U2(${a(p,0)},${a(p,1)})`
        case 'cu3': return `U3(${a(p,0)},${a(p,1)},${a(p,2)})`
        case 'cs': return 'S'; case 'ct': return 'T'; case 'csdg': return 'S†'; case 'ctdg': return 'T†'
        case 'cr2': return 'R₂'; case 'cr4': return 'R₄'; case 'cr8': return 'R₈'
        default: return name ? name.slice(1).toUpperCase() : 'U'
      }
    }
    case 'toffoli':    return op.target === q ? '⊕' : '●'
    case 'cswap':      return op.control === q ? '●' : '╳'
    case 'csrswap':    return op.control === q ? '●' : '√SW'
    case 'measure':    return 'M'
    case 'reset':      return '|0⟩'
    case 'barrier':    return '░'
    case 'subcircuit': return op.name
    case 'unitary':    return 'U'
    default:           return '?'
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
  /** Starting computational basis state as a bitstring (q0 rightmost). E.g. `'110'` = q0=0, q1=1, q2=1. */
  initialState?: string
}

export interface MpsRunOptions {
  shots?: number
  seed?: number
  /** Maximum bond dimension χ (default 64). Larger = more accurate for high-entanglement circuits. */
  maxBond?: number
  /** Starting computational basis state as a bitstring (q0 rightmost). E.g. `'110'` = q0=0, q1=1, q2=1. */
  initialState?: string
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
  readonly #gates: ReadonlyMap<string, Circuit>  // name → registered sub-circuit

  constructor(
    qubits: number,
    ops:   readonly Op[]                 = [],
    cregs: ReadonlyMap<string, number>   = new Map(),
    gates: ReadonlyMap<string, Circuit>  = new Map(),
  ) {
    this.qubits  = qubits
    this.#ops    = ops
    this.#cregs  = cregs
    this.#gates  = gates
  }

  #add(op: Op): Circuit {
    this.#checkOp(op)
    return new Circuit(this.qubits, [...this.#ops, op], this.#cregs, this.#gates)
  }

  #checkOp(op: Op): void {
    const q = (i: number) => {
      if (i < 0 || i >= this.qubits)
        throw new RangeError(`qubit index ${i} is out of range for a ${this.qubits}-qubit circuit`)
    }
    switch (op.kind) {
      case 'single':     q(op.q); break
      case 'cnot':       q(op.control); q(op.target); break
      case 'swap':       q(op.a); q(op.b); break
      case 'two':        q(op.a); q(op.b); break
      case 'controlled': q(op.control); q(op.target); break
      case 'toffoli':    q(op.c1); q(op.c2); q(op.target); break
      case 'cswap':      q(op.control); q(op.a); q(op.b); break
      case 'csrswap':    q(op.control); q(op.a); q(op.b); break
      case 'measure':    q(op.q); break
      case 'reset':      q(op.q); break
      case 'barrier':    op.qubits.forEach(q); break
      case 'unitary': {
        if (op.qubits.length === 0) throw new TypeError('unitary: qubits must be non-empty')
        const expected = 1 << op.qubits.length
        if (op.matrix.length !== expected || op.matrix.some(row => row.length !== expected))
          throw new TypeError(`unitary: matrix must be ${expected}×${expected} for ${op.qubits.length} qubit(s), got ${op.matrix.length}×${op.matrix[0]?.length ?? 0}`)
        op.qubits.forEach(q)
        break
      }
    }
  }

  #ctrl(control: number, target: number, gate: Gate2x2, meta?: GateMeta): Circuit {
    return this.#add({ kind: 'controlled', control, target, gate, ...(meta !== undefined && { meta }) })
  }

  // ── IonQ single-qubit gates ──────────────────────────────────────────────

  /** Identity gate — no-op on the statevector; preserved by name through import/export. */
  id(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Id, meta: { name: 'id' } }) }

  h(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.H,  meta: { name: 'h'  } }) }
  x(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.X,  meta: { name: 'x'  } }) }
  y(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Y,  meta: { name: 'y'  } }) }
  z(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Z,  meta: { name: 'z'  } }) }
  s(q: number):    Circuit { return this.#add({ kind: 'single', q, gate: G.S,  meta: { name: 's'     } }) }
  si(q: number):   Circuit { return this.#add({ kind: 'single', q, gate: G.Si, meta: { name: 'si'    } }) }
  /** S† — alias `sdg` (Qiskit / OpenQASM convention). */
  sdg(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Si, meta: { name: 'sdg'   } }) }
  t(q: number):    Circuit { return this.#add({ kind: 'single', q, gate: G.T,  meta: { name: 't'     } }) }
  ti(q: number):   Circuit { return this.#add({ kind: 'single', q, gate: G.Ti, meta: { name: 'ti'    } }) }
  /** T† — alias `tdg` (Qiskit / OpenQASM convention). */
  tdg(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.Ti, meta: { name: 'tdg'   } }) }
  v(q: number):    Circuit { return this.#add({ kind: 'single', q, gate: G.V,  meta: { name: 'v'     } }) }
  vi(q: number):   Circuit { return this.#add({ kind: 'single', q, gate: G.Vi, meta: { name: 'vi'    } }) }
  /** √NOT — alias `srn`; same as `v`. */
  srn(q: number):  Circuit { return this.#add({ kind: 'single', q, gate: G.V,  meta: { name: 'srn'   } }) }
  /** (√NOT)† — alias `srndg`; same as `vi`. */
  srndg(q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Vi, meta: { name: 'srndg' } }) }

  // ── Rotation gates ───────────────────────────────────────────────────────

  rx(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Rx(theta), meta: { name: 'rx', params: [theta] } }) }
  ry(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Ry(theta), meta: { name: 'ry', params: [theta] } }) }
  rz(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Rz(theta), meta: { name: 'rz', params: [theta] } }) }

  /**
   * VirtualZ(θ) — named Rz alias common in superconducting hardware native gate sets
   * (IBM, Rigetti). Functionally identical to `rz(θ)` but carries the `vz` name through
   * import/export for hardware compilation pass awareness.
   */
  vz(theta: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.Rz(theta), meta: { name: 'vz', params: [theta] } }) }

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

  /** P(λ) — phase gate alias for U1(λ); Qiskit 1.0+ name. P(π) = Z, P(π/2) = S, P(π/4) = T. */
  p(lambda: number, q: number): Circuit { return this.#add({ kind: 'single', q, gate: G.U1(lambda), meta: { name: 'p', params: [lambda] } }) }

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

  /** CU2(φ,λ) = CU3(π/2,φ,λ) — controlled equatorial gate. */
  cu2(phi: number, lambda: number, control: number, target: number): Circuit {
    return this.#ctrl(control, target, G.U2(phi, lambda), { name: 'cu2', params: [phi, lambda] })
  }

  /** CU3(θ,φ,λ) — controlled general unitary; CU3(π,0,π) = CX. */
  cu3(theta: number, phi: number, lambda: number, control: number, target: number): Circuit {
    return this.#ctrl(control, target, G.U3(theta, phi, lambda), { name: 'cu3', params: [theta, phi, lambda] })
  }

  // ── Controlled phase gates ────────────────────────────────────────────────

  cs(control: number, target: number):   Circuit { return this.#ctrl(control, target, G.S,  { name: 'cs'   }) }
  ct(control: number, target: number):   Circuit { return this.#ctrl(control, target, G.T,  { name: 'ct'   }) }
  csdg(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Si, { name: 'csdg' }) }
  ctdg(control: number, target: number): Circuit { return this.#ctrl(control, target, G.Ti, { name: 'ctdg' }) }

  /** Controlled-√NOT (C-V); applies V = √X to target when control is |1⟩. */
  csrn(control: number, target: number): Circuit { return this.#ctrl(control, target, G.V,  { name: 'csrn' }) }

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

  /** C-√iSWAP: apply √iSWAP to qubits a and b if control is |1⟩. Completes the three-qubit gate set. */
  csrswap(control: number, a: number, b: number): Circuit {
    return this.#add({ kind: 'csrswap', control, a, b })
  }

  // ── Scheduling hints ─────────────────────────────────────────────────────

  /**
   * Barrier — scheduling/grouping hint with no effect on the statevector.
   * In QASM export it emits `barrier q[a],q[b],...;`. Pass the qubit indices to barrier.
   * Calling with no arguments barriers all qubits.
   */
  barrier(...qubits: number[]): Circuit {
    const qs = qubits.length ? qubits : Array.from({ length: this.qubits }, (_, i) => i)
    return this.#add({ kind: 'barrier', qubits: qs })
  }

  // ── Custom unitary gate ──────────────────────────────────────────────────

  /**
   * Apply a custom N-qubit unitary gate defined by its 2^N × 2^N matrix.
   *
   * `matrix` must be 2^N × 2^N where N = qubits.length. Entries may be
   * `Complex` objects `{ re, im }` or plain `number` (treated as real).
   * The qubit ordering matches all other multi-qubit gates: `qubits[0]` is the
   * MSB of the local state index.
   *
   * @example
   * // Real matrix
   * circuit.unitary([[1,0],[0,1]], 0)
   *
   * // Complex matrix
   * const S = [[{re:1,im:0},{re:0,im:0}],[{re:0,im:0},{re:0,im:1}]]
   * circuit.unitary(S, 0)
   */
  unitary(matrix: readonly (readonly (number | Complex)[])[], ...qubits: number[]): Circuit {
    const normalized: Complex[][] = matrix.map(row =>
      row.map(v => typeof v === 'number' ? { re: v, im: 0 } : v)
    )
    return this.#add({ kind: 'unitary', qubits, matrix: normalized })
  }

  // ── Statevector inspection ────────────────────────────────────────────────

  /**
   * Simulate the circuit and return the full sparse amplitude map.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   *
   * @param initialState Optional starting computational basis state as a bitstring (q0 rightmost).
   */
  statevector({ initialState }: { initialState?: string } = {}): Map<bigint, Complex> {
    if (this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      throw new TypeError('statevector() requires a pure circuit — remove measure/reset/if ops')
    }
    const init = initialState !== undefined ? svFromBitstring(initialState, this.qubits) : undefined
    return simulatePure(this.#ops, this.qubits, init)
  }

  /**
   * Return the 2^n × 2^n unitary matrix of the circuit.
   *
   * `matrix[row][col]` is the amplitude of basis state `|row⟩` after starting
   * from `|col⟩`. Row and column indices use **standard convention**: q0 is the
   * MSB, so the ordering is |00…0⟩, |00…1⟩, …, |11…1⟩ with the first qubit
   * varying slowest. This matches the convention of `unitary()`, textbooks, and
   * most quantum computing libraries.
   *
   * Note: this is different from the IonQ bitstring convention (q0 rightmost)
   * used by `amplitude()`, `exactProbs()`, and `statevector()`.
   *
   * Throws `TypeError` for circuits with mid-circuit measurement, reset, or
   * conditional ops. Throws `RangeError` for circuits wider than 12 qubits
   * (matrix would be 4096×4096 = 16M entries).
   *
   * @example
   * new Circuit(2).cnot(0, 1).circuitMatrix()
   * // [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]  — standard CNOT matrix
   */
  circuitMatrix(): Complex[][] {
    if (this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      throw new TypeError('circuitMatrix() requires a pure circuit — remove measure/reset/if ops')
    }
    const n = this.qubits
    const dim = 1 << n
    if (n > 12) {
      throw new RangeError(`circuitMatrix(): circuit too large (${n} qubits = ${dim}×${dim} matrix)`)
    }
    // Convert between standard index (q0=MSB) and global IonQ index (q0=LSB)
    // by reversing the n-bit representation. Bit reversal is its own inverse.
    const flip = (i: number): number => {
      let r = 0
      for (let b = 0; b < n; b++) if (i & (1 << b)) r |= 1 << (n - 1 - b)
      return r
    }
    const matrix: Complex[][] = Array.from({ length: dim }, () => new Array<Complex>(dim).fill(ZERO))
    for (let col = 0; col < dim; col++) {
      const sv = simulatePure(this.#ops, this.qubits, new Map([[BigInt(flip(col)), { re: 1, im: 0 }]]))
      for (const [idx, amp] of sv) matrix[flip(Number(idx))]![col] = amp
    }
    return matrix
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

  /**
   * Return the marginal probability P(qubit q = |1⟩) for each qubit.
   * Result[q] is the probability of measuring qubit q as 1, summed over all other qubits.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   */
  marginals({ initialState }: { initialState?: string } = {}): number[] {
    const sv  = this.statevector(initialState !== undefined ? { initialState } : {})
    const out = new Array<number>(this.qubits).fill(0)
    for (const [idx, amp] of sv) {
      const p = amp.re * amp.re + amp.im * amp.im
      for (let q = 0; q < this.qubits; q++) {
        if ((idx >> BigInt(q)) & 1n) out[q]! += p
      }
    }
    return out
  }

  /**
   * Return a human-readable representation of the statevector, e.g.:
   *   `0.7071|00⟩ + 0.7071|11⟩`
   *   `0.5|00⟩ + (0.5+0.5i)|01⟩ - 0.5i|10⟩`
   *
   * Amplitudes with magnitude² < 1e-10 are omitted.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   */
  stateAsString({ initialState }: { initialState?: string } = {}): string {
    const sv  = this.statevector(initialState !== undefined ? { initialState } : {})
    const eps = 1e-10
    const n   = (x: number) => parseFloat(x.toPrecision(4)).toString()

    const fmtAmp = ({ re, im }: Complex): string => {
      const rz = Math.abs(re) < eps, iz = Math.abs(im) < eps
      if (rz && iz) return '0'
      if (iz) return n(re)
      if (rz) return `${n(im)}i`
      const sign = im < 0 ? '' : '+'
      return `(${n(re)}${sign}${n(im)}i)`
    }

    const entries = [...sv.entries()]
      .filter(([, { re, im }]) => re * re + im * im > eps * eps)
      .sort(([a], [b]) => (a < b ? -1 : 1))

    if (entries.length === 0) return '0'

    const terms = entries.map(([idx, amp]) =>
      `${fmtAmp(amp)}|${idx.toString(2).padStart(this.qubits, '0')}⟩`
    )

    let result = terms[0]!
    for (let i = 1; i < terms.length; i++) {
      const t = terms[i]!
      result += t.startsWith('-') ? ` - ${t.slice(1)}` : ` + ${t}`
    }
    return result
  }

  // ── Classical registers and mid-circuit measurement ──────────────────────

  /** Declare a classical register of `size` bits. */
  creg(name: string, size: number): Circuit {
    return new Circuit(this.qubits, this.#ops, new Map(this.#cregs).set(name, size), this.#gates)
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
      this.#gates,
    )
  }

  /**
   * Reset qubit `q` to the given computational basis state (default |0⟩).
   *
   * - `reset(q)` / `reset(q, 0)` — unconditionally collapses `q` to |0⟩.
   * - `reset(q, 1)` — collapses to |0⟩ then flips to |1⟩, equivalent to `reset(q).x(q)`.
   */
  reset(q: number, value: 0 | 1 = 0): Circuit {
    const r = this.#add({ kind: 'reset', q })
    return value === 1 ? r.x(q) : r
  }

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
    const inner = build(new Circuit(this.qubits, [], new Map(), this.#gates))
    return this.#add({ kind: 'if', creg, value, ops: inner.#ops })
  }

  // ── Named sub-circuit gates ──────────────────────────────────────────────

  /**
   * Register a named reusable gate defined by `sub`.
   *
   * The registered gate can be used on this circuit (and any circuit derived
   * from it) via `.gate(name, ...qubits)`.
   *
   * @example
   * const bell = new Circuit(2).h(0).cnot(0, 1)
   * const c = new Circuit(4)
   *   .defineGate('bell', bell)
   *   .gate('bell', 0, 1)   // Bell pair on qubits 0,1
   *   .gate('bell', 2, 3)   // Bell pair on qubits 2,3
   */
  defineGate(name: string, sub: Circuit): Circuit {
    if (sub.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      throw new TypeError(`Gate '${name}' contains classical ops (measure/reset/if), which are not supported inside named gates`)
    }
    return new Circuit(this.qubits, this.#ops, this.#cregs, new Map(this.#gates).set(name, sub))
  }

  /**
   * Apply a previously registered named gate to the given parent-circuit qubits.
   *
   * The number of `qubits` must match the qubit count of the registered gate.
   * Qubit 0 of the gate maps to `qubits[0]`, qubit 1 to `qubits[1]`, etc.
   *
   * @example
   * circuit.gate('bell', 2, 3)  // apply 'bell' gate to parent qubits 2 and 3
   */
  gate(name: string, ...qubits: number[]): Circuit {
    const sub = this.#gates.get(name)
    if (!sub) throw new TypeError(`Unknown gate '${name}'. Register it first with .defineGate(name, subcircuit).`)
    if (qubits.length !== sub.qubits) {
      throw new TypeError(`Gate '${name}' expects ${sub.qubits} qubit(s), got ${qubits.length}`)
    }
    return this.#add({ kind: 'subcircuit', name, qubits, def: sub.#ops })
  }

  /**
   * Inline all named gates, returning a new `Circuit` containing only primitive ops.
   *
   * Required before serialization (toQASM, toQiskit, etc.) when the circuit
   * contains named gates applied via `.gate()`.
   *
   * @example
   * circuit.defineGate('bell', bell).gate('bell', 0, 1).decompose()
   */
  decompose(): Circuit {
    return new Circuit(this.qubits, flattenOps(this.#ops), this.#cregs, this.#gates)
  }

  // ── IonQ device targeting ────────────────────────────────────────────────

  /**
   * Return the published specs for a named IonQ device.
   * Throws if the device name is not recognised.
   */
  static ionqDevice(name: string): IonQDeviceInfo {
    const info = IONQ_DEVICES[name]
    if (!info) throw new TypeError(`Unknown IonQ device '${name}'. Known: ${Object.keys(IONQ_DEVICES).join(', ')}`)
    return info
  }

  /**
   * Validate that this circuit can be submitted to the named IonQ device.
   * Throws a `TypeError` listing every issue found:
   *   - qubit count exceeds device capacity
   *   - gates that have no IonQ JSON representation (use `decompose()` or replace them)
   *
   * Call this before `toIonQ()` to get a complete error report rather than a
   * first-failure throw.
   */
  checkDevice(name: string): void {
    const info = Circuit.ionqDevice(name)
    const issues: string[] = []

    if (this.qubits > info.qubits)
      issues.push(`circuit uses ${this.qubits} qubits; ${name} supports at most ${info.qubits}`)

    const IONQ_SINGLE = new Set(['h','x','y','z','s','si','t','ti','v','vi','rx','ry','rz','r2','r4','r8','gpi','gpi2','vz','id'])
    const IONQ_TWO    = new Set(['xx','yy','zz','ms'])
    const seen = new Set<string>()

    for (const op of flattenOps(this.#ops)) {
      if (op.kind === 'cnot' || op.kind === 'swap' || op.kind === 'barrier') continue
      if (op.kind === 'single' && op.meta && IONQ_SINGLE.has(op.meta.name)) continue
      if (op.kind === 'two'    && op.meta && IONQ_TWO.has(op.meta.name))    continue
      const label = (op as { meta?: GateMeta }).meta?.name ?? op.kind
      if (!seen.has(label)) { seen.add(label); issues.push(`gate '${label}' is not supported on ${name}`) }
    }

    if (issues.length) throw new TypeError(`Circuit is not compatible with ${name}:\n  - ${issues.join('\n  - ')}`)
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
    for (const op of flattenOps(this.#ops)) {
      if (op.kind === 'cnot') {
        circuit.push({ gate: 'cnot', control: op.control, target: op.target })
      } else if (op.kind === 'swap') {
        circuit.push({ gate: 'swap', targets: [op.a, op.b] })
      } else if (op.kind === 'csrswap') {
        throw new TypeError(`Gate 'csrswap' is not serializable to IonQ JSON`)
      } else if (op.kind === 'single' && op.meta?.name === 'id') {
        // Identity gate has no IonQ JSON representation; omit (no effect on state)
      } else if (op.kind === 'single' && op.meta?.name === 'vz') {
        // VirtualZ = Rz; IonQ has no vz gate, emit as rz
        circuit.push({ gate: 'rz', target: op.q, rotation: op.meta.params![0]! / Math.PI })
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

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    lines.push(`cx q[${op.control}],q[${op.target}];`); break
        case 'swap':    lines.push(`swap q[${op.a}],q[${op.b}];`);         break
        case 'toffoli': lines.push(`ccx q[${op.c1}],q[${op.c2}],q[${op.target}];`); break
        case 'cswap':   lines.push(`cswap q[${op.control}],q[${op.a}],q[${op.b}];`); break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no OpenQASM 2.0 representation")
        case 'measure': lines.push(`measure q[${op.q}] -> ${op.creg}[${op.bit}];`); break
        case 'reset':   lines.push(`reset q[${op.q}];`); break
        case 'barrier': lines.push(`barrier ${op.qubits.map(q => `q[${q}]`).join(',')};`); break
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
        case 'two':     throw new TypeError(`Gate '${op.meta?.name ?? 'two'}' has no OpenQASM 2.0 representation`)
        case 'unitary': throw new TypeError("Gate 'unitary' has no OpenQASM 2.0 representation")
      }
    }
    return lines.join('\n')
  }

  /**
   * Parse an OpenQASM 2.0 or 3.0 string into a `Circuit`. Auto-detects the version.
   *
   * **2.0 syntax supported:** `qreg`/`creg`, `measure q[i] -> c[j]`, `//` comments,
   * all qelib1.inc gates (h, x, cx, rz, u1, u2, u3, ccx, cswap, …).
   *
   * **3.0 syntax supported:** qubit[N]/bit[N] declarations, "c[j] = measure q[i]"
   * assignment form, block comments, stdgates.inc, p/sx/sdg/tdg gate names.
   *
   * Not supported: gate definitions (`gate foo …`), gate modifiers (`ctrl @`, `inv @`),
   * `if`/`else` blocks, `gphase`, multi-register qubit indexing.
   */
  static fromQASM(source: string): Circuit {
    // Strip block comments then single-line comments; split into statements
    const stmts = source
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/[^\n]*/g, '')
      .split(';')
      .map(s => s.trim())
      .filter(Boolean)

    let qubits = 0
    const cregSizes = new Map<string, number>()

    // First pass: collect register declarations (both 2.0 and 3.0 syntax)
    for (const stmt of stmts) {
      // QASM 2.0: qreg q[N]
      const qr2 = stmt.match(/^qreg\s+\w+\[(\d+)\]$/)
      if (qr2) { qubits += parseInt(qr2[1]!); continue }

      // QASM 3.0: qubit[N] q  |  qubit q  (single qubit)
      const qr3 = stmt.match(/^qubit(?:\[(\d+)\])?\s+\w+$/)
      if (qr3) { qubits += parseInt(qr3[1] ?? '1'); continue }

      // QASM 2.0: creg name[N]
      const cr2 = stmt.match(/^creg\s+(\w+)\[(\d+)\]$/)
      if (cr2) { cregSizes.set(cr2[1]!, parseInt(cr2[2]!)); continue }

      // QASM 3.0: bit[N] name  |  bit name
      const cr3 = stmt.match(/^bit(?:\[(\d+)\])?\s+(\w+)$/)
      if (cr3) { cregSizes.set(cr3[2]!, parseInt(cr3[1] ?? '1')); continue }
    }

    let c = new Circuit(qubits)
    for (const [name, size] of cregSizes) c = c.creg(name, size)

    // Second pass: apply gates and operations
    for (const stmt of stmts) {
      // Skip header / declaration lines
      if (/^(OPENQASM|include|qreg|creg|qubit|bit)\b/.test(stmt)) continue

      // QASM 2.0: measure q[i] -> c[j]
      const meas2 = stmt.match(/^measure\s+\w+\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]$/)
      if (meas2) { c = c.measure(parseInt(meas2[1]!), meas2[2]!, parseInt(meas2[3]!)); continue }

      // QASM 3.0: c[j] = measure q[i]
      const meas3 = stmt.match(/^(\w+)\[(\d+)\]\s*=\s*measure\s+\w+\[(\d+)\]$/)
      if (meas3) { c = c.measure(parseInt(meas3[3]!), meas3[1]!, parseInt(meas3[2]!)); continue }

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

  /**
   * Parse a Quil 2.0 program and return an equivalent Circuit.
   *
   * Supported: all standard single/two/three-qubit gates, `CONTROLLED` prefix,
   * `DAGGER` prefix, `MEASURE`, `RESET`, `DECLARE BIT[]`.
   * Qubit count is inferred from the highest qubit index referenced.
   * `DEFGATE` and `PRAGMA` are silently skipped.
   *
   * @example
   * const c = Circuit.fromQuil('H 0\nCNOT 0 1\nMEASURE 0 ro[0]')
   */
  static fromQuil(source: string): Circuit {
    // Strip comments (# to end of line), then split into non-empty lines
    const lines = source
      .split('\n')
      .map(l => l.replace(/#.*/g, '').trim())
      .filter(Boolean)

    const cregs = new Map<string, number>()
    let maxQubit = 0

    // ── Pass 1: DECLARE registers + infer qubit count ──────────────────────
    for (const line of lines) {
      const decl = line.match(/^DECLARE\s+(\w+)\s+BIT\[(\d+)\]/)
      if (decl) { cregs.set(decl[1]!, parseInt(decl[2]!)); continue }
      // Strip parenthesised params before scanning qubit indices
      const withoutParams = line.replace(/\([^)]*\)/g, '')
      for (const m of withoutParams.matchAll(/\b(\d+)\b/g)) {
        maxQubit = Math.max(maxQubit, parseInt(m[1]!))
      }
    }

    let c = new Circuit(maxQubit + 1)
    for (const [name, size] of cregs) c = c.creg(name, size)

    // ── Pass 2: apply gates ─────────────────────────────────────────────────
    for (const line of lines) {
      if (/^(DECLARE|DEFGATE|DEFCIRCUIT|PRAGMA|HALT|WAIT|NOP)\b/.test(line)) continue

      // MEASURE qubit reg[bit]
      const meas = line.match(/^MEASURE\s+(\d+)\s+(\w+)\[(\d+)\]$/)
      if (meas) { c = c.measure(parseInt(meas[1]!), meas[2]!, parseInt(meas[3]!)); continue }

      // RESET [qubit?]
      const rst = line.match(/^RESET(?:\s+(\d+))?$/)
      if (rst) {
        if (rst[1] !== undefined) {
          c = c.reset(parseInt(rst[1]!))
        } else {
          for (let q = 0; q < c.qubits; q++) c = c.reset(q)
        }
        continue
      }

      // CONTROLLED gateName[(params)] ctrl tgt
      const ctrlLine = line.match(/^CONTROLLED\s+(\w+)(?:\(([^)]*)\))?\s+(\d+)\s+(\d+)$/)
      if (ctrlLine) {
        const [, gName, paramStr, ctrlStr, tgtStr] = ctrlLine
        const p = paramStr ? paramStr.split(',').map(parseAngle) : []
        const [con, tgt] = [parseInt(ctrlStr!), parseInt(tgtStr!)]
        switch (gName!.toUpperCase()) {
          case 'H':     c = c.ch(con, tgt);             break
          case 'Y':     c = c.cy(con, tgt);             break
          case 'Z':     c = c.cz(con, tgt);             break
          case 'RX':    c = c.crx(p[0]!, con, tgt);    break
          case 'RY':    c = c.cry(p[0]!, con, tgt);    break
          case 'RZ':    c = c.crz(p[0]!, con, tgt);    break
          case 'PHASE': c = c.cu1(p[0]!, con, tgt);    break
          default: throw new TypeError(`fromQuil: unknown CONTROLLED gate '${gName}'`)
        }
        continue
      }

      // DAGGER gateName qubit  (S†, T†)
      const daggerLine = line.match(/^DAGGER\s+(\w+)\s+(\d+)$/)
      if (daggerLine) {
        const [, gName, qStr] = daggerLine
        const q = parseInt(qStr!)
        switch (gName!.toUpperCase()) {
          case 'S': c = c.si(q); break
          case 'T': c = c.ti(q); break
          default: throw new TypeError(`fromQuil: unknown DAGGER gate '${gName}'`)
        }
        continue
      }

      // Generic: GATENAME[(params)] qubit [qubit ...]
      const gate = line.match(/^(\w+)(?:\(([^)]*)\))?\s+([\d\s]+)$/)
      if (gate) {
        const [, gName, paramStr, qStr] = gate
        const p  = paramStr ? paramStr.split(',').map(parseAngle) : []
        const qs = qStr!.trim().split(/\s+/).map(Number)
        const [q0, q1, q2] = qs
        switch (gName!.toUpperCase()) {
          case 'I':      c = c.id(q0!);                        break
          case 'H':      c = c.h(q0!);                         break
          case 'X':      c = c.x(q0!);                         break
          case 'Y':      c = c.y(q0!);                         break
          case 'Z':      c = c.z(q0!);                         break
          case 'S':      c = c.s(q0!);                         break
          case 'T':      c = c.t(q0!);                         break
          case 'RX':     c = c.rx(p[0]!, q0!);                 break
          case 'RY':     c = c.ry(p[0]!, q0!);                 break
          case 'RZ':     c = c.rz(p[0]!, q0!);                 break
          case 'PHASE':  c = c.u1(p[0]!, q0!);                 break
          case 'CNOT':   c = c.cnot(q0!, q1!);                 break
          case 'CZ':     c = c.cz(q0!, q1!);                   break
          case 'SWAP':   c = c.swap(q0!, q1!);                  break
          case 'ISWAP':  c = c.iswap(q0!, q1!);                break
          case 'CPHASE': c = c.cu1(p[0]!, q0!, q1!);           break
          case 'CCNOT':  c = c.ccx(q0!, q1!, q2!);             break
          case 'CSWAP':  c = c.cswap(q0!, q1!, q2!);           break
          default: throw new TypeError(`fromQuil: unknown gate '${gName}'`)
        }
        continue
      }
    }
    return c
  }

  /**
   * Parse Qiskit Python QuantumCircuit code back into a Circuit.
   * Accepts the output of toQiskit() — round-trips all gates supported by that method.
   */
  static fromQiskit(source: string): Circuit {
    const pa = (e: string) => parseAngle(e.replace(/math\.pi/g, 'pi').trim())

    const nm = source.match(/QuantumCircuit\((\d+)\)/)
    if (!nm) throw new TypeError('fromQiskit: cannot find QuantumCircuit(N)')
    let c = new Circuit(parseInt(nm[1]!))

    const cregRe = /ClassicalRegister\((\d+),\s*['"](\w+)['"]\)/g
    let cm: RegExpExecArray | null
    while ((cm = cregRe.exec(source)) !== null) c = c.creg(cm[2]!, parseInt(cm[1]!))

    for (const rawLine of source.split('\n')) {
      const m = rawLine.trim().match(/^qc\.(\w+)\(([^)]*(?:\[[^\]]*\][^)]*)*)\)$/)
      if (!m) continue
      const [, method, argStr] = m
      const args = argStr!.split(',').map(s => s.trim()).filter(Boolean)
      const qi = (s: string) => parseInt(s)
      switch (method!) {
        case 'h':      c = c.h(qi(args[0]!));                                                             break
        case 'x':      c = c.x(qi(args[0]!));                                                             break
        case 'y':      c = c.y(qi(args[0]!));                                                             break
        case 'z':      c = c.z(qi(args[0]!));                                                             break
        case 's':      c = c.s(qi(args[0]!));                                                             break
        case 'sdg':    c = c.si(qi(args[0]!));                                                            break
        case 't':      c = c.t(qi(args[0]!));                                                             break
        case 'tdg':    c = c.ti(qi(args[0]!));                                                            break
        case 'sx':     c = c.v(qi(args[0]!));                                                             break
        case 'sxdg':   c = c.vi(qi(args[0]!));                                                            break
        case 'id':     c = c.id(qi(args[0]!));                                                            break
        case 'rx':     c = c.rx(pa(args[0]!), qi(args[1]!));                                              break
        case 'ry':     c = c.ry(pa(args[0]!), qi(args[1]!));                                              break
        case 'rz':     c = c.rz(pa(args[0]!), qi(args[1]!));                                              break
        case 'u1':     c = c.u1(pa(args[0]!), qi(args[1]!));                                              break
        case 'u2':     c = c.u2(pa(args[0]!), pa(args[1]!), qi(args[2]!));                                break
        case 'u3':     c = c.u3(pa(args[0]!), pa(args[1]!), pa(args[2]!), qi(args[3]!));                  break
        case 'p':      c = c.p(pa(args[0]!), qi(args[1]!));                                               break
        case 'cx':     c = c.cnot(qi(args[0]!), qi(args[1]!));                                            break
        case 'cy':     c = c.cy(qi(args[0]!), qi(args[1]!));                                              break
        case 'cz':     c = c.cz(qi(args[0]!), qi(args[1]!));                                              break
        case 'ch':     c = c.ch(qi(args[0]!), qi(args[1]!));                                              break
        case 'swap':   c = c.swap(qi(args[0]!), qi(args[1]!));                                            break
        case 'crx':    c = c.crx(pa(args[0]!), qi(args[1]!), qi(args[2]!));                               break
        case 'cry':    c = c.cry(pa(args[0]!), qi(args[1]!), qi(args[2]!));                               break
        case 'crz':    c = c.crz(pa(args[0]!), qi(args[1]!), qi(args[2]!));                               break
        case 'cu1':    c = c.cu1(pa(args[0]!), qi(args[1]!), qi(args[2]!));                               break
        case 'cu2':    c = c.cu2(pa(args[0]!), pa(args[1]!), qi(args[2]!), qi(args[3]!));                 break
        case 'cu3':    c = c.cu3(pa(args[0]!), pa(args[1]!), pa(args[2]!), qi(args[3]!), qi(args[4]!));   break
        case 'rxx':    c = c.xx(pa(args[0]!), qi(args[1]!), qi(args[2]!));                                break
        case 'ryy':    c = c.yy(pa(args[0]!), qi(args[1]!), qi(args[2]!));                                break
        case 'rzz':    c = c.zz(pa(args[0]!), qi(args[1]!), qi(args[2]!));                                break
        case 'iswap':  c = c.iswap(qi(args[0]!), qi(args[1]!));                                           break
        case 'ccx':    c = c.ccx(qi(args[0]!), qi(args[1]!), qi(args[2]!));                               break
        case 'cswap':  c = c.cswap(qi(args[0]!), qi(args[1]!), qi(args[2]!));                             break
        case 'reset':  c = c.reset(qi(args[0]!));                                                         break
        case 'barrier': {
          const qs = args.filter(a => /^\d+$/.test(a)).map(Number)
          c = qs.length ? c.barrier(...qs) : c.barrier()
          break
        }
        case 'measure': {
          const mt = args[1]!.match(/(\w+)\[(\d+)\]/)
          if (!mt) throw new TypeError(`fromQiskit: invalid measure target '${args[1]}'`)
          c = c.measure(qi(args[0]!), mt[1]!, parseInt(mt[2]!))
          break
        }
        case 'add_register': break
        default: throw new TypeError(`fromQiskit: unknown method 'qc.${method}'`)
      }
    }
    return c
  }

  /**
   * Parse Cirq Python circuit code back into a Circuit.
   * Accepts the output of toCirq() — round-trips all gates supported by that method.
   */
  static fromCirq(source: string): Circuit {
    const pa = (e: string) => parseAngle(e.replace(/math\.pi/g, 'pi').trim())

    const nm = source.match(/LineQubit\.range\((\d+)\)/)
    if (!nm) throw new TypeError('fromCirq: cannot find LineQubit.range(N)')
    let c = new Circuit(parseInt(nm[1]!))

    const qis = (s: string): number[] =>
      Array.from(s.matchAll(/q\[(\d+)\]/g), m => parseInt(m[1]!))

    for (const rawLine of source.split('\n')) {
      const line = rawLine.trim().replace(/,$/, '')
      if (!line.startsWith('cirq.') && !line.startsWith('(cirq.')) continue
      if (line.startsWith('cirq.Circuit') || line.startsWith('cirq.LineQubit')) continue

      // cirq.ZPowGate(exponent=E).controlled()(q[c], q[t])
      const zpowCtrl = line.match(/^cirq\.ZPowGate\(exponent=([^)]+)\)\.controlled\(\)\(q\[(\d+)\],\s*q\[(\d+)\]\)$/)
      if (zpowCtrl) {
        const exp = parseFloat(zpowCtrl[1]!);  const ci = parseInt(zpowCtrl[2]!);  const ti = parseInt(zpowCtrl[3]!)
        if (Math.abs(exp - 0.5)  < 1e-9) c = c.cs(ci, ti)
        else if (Math.abs(exp - 0.25) < 1e-9) c = c.ct(ci, ti)
        else if (Math.abs(exp + 0.5)  < 1e-9) c = c.csdg(ci, ti)
        else if (Math.abs(exp + 0.25) < 1e-9) c = c.ctdg(ci, ti)
        else c = c.cu1(exp * Math.PI, ci, ti)
        continue
      }

      // cirq.ZPowGate(exponent=E)(q[i])
      const zpow = line.match(/^cirq\.ZPowGate\(exponent=([^)]+)\)\(q\[(\d+)\]\)$/)
      if (zpow) {
        const exp = parseFloat(zpow[1]!);  const q = parseInt(zpow[2]!)
        if (Math.abs(exp - 0.5)  < 1e-9) c = c.s(q)
        else if (Math.abs(exp - 0.25) < 1e-9) c = c.t(q)
        else if (Math.abs(exp + 0.5)  < 1e-9) c = c.si(q)
        else if (Math.abs(exp + 0.25) < 1e-9) c = c.ti(q)
        else c = c.u1(exp * Math.PI, q)
        continue
      }

      // cirq.rx(rads=E).controlled()(q[c], q[t])
      const radsCtrl = line.match(/^cirq\.(rx|ry|rz)\(rads=([^)]+)\)\.controlled\(\)\(q\[(\d+)\],\s*q\[(\d+)\]\)$/)
      if (radsCtrl) {
        const r = pa(radsCtrl[2]!);  const ci = parseInt(radsCtrl[3]!);  const ti = parseInt(radsCtrl[4]!)
        if (radsCtrl[1] === 'rx') c = c.crx(r, ci, ti)
        else if (radsCtrl[1] === 'ry') c = c.cry(r, ci, ti)
        else c = c.crz(r, ci, ti)
        continue
      }

      // cirq.rx(rads=E)(q[i])
      const radsSingle = line.match(/^cirq\.(rx|ry|rz)\(rads=([^)]+)\)\(q\[(\d+)\]\)$/)
      if (radsSingle) {
        const r = pa(radsSingle[2]!);  const q = parseInt(radsSingle[3]!)
        if (radsSingle[1] === 'rx') c = c.rx(r, q)
        else if (radsSingle[1] === 'ry') c = c.ry(r, q)
        else c = c.rz(r, q)
        continue
      }

      // cirq.X**0.5(q[i]) or (cirq.X**-0.5)(q[i])
      const xpow = line.match(/^\(?cirq\.X\*\*([^)(]+)\)?\(q\[(\d+)\]\)$/)
      if (xpow) {
        const exp = parseFloat(xpow[1]!.trim());  const q = parseInt(xpow[2]!)
        if (Math.abs(exp - 0.5)  < 1e-9) c = c.v(q)
        else if (Math.abs(exp + 0.5) < 1e-9) c = c.vi(q)
        else throw new TypeError(`fromCirq: unsupported X**${exp}`)
        continue
      }

      // cirq.GATE.controlled()(q[c], q[t])
      const namedCtrl = line.match(/^cirq\.(\w+)\.controlled\(\)\(q\[(\d+)\],\s*q\[(\d+)\]\)$/)
      if (namedCtrl) {
        const ci = parseInt(namedCtrl[2]!);  const ti = parseInt(namedCtrl[3]!)
        switch (namedCtrl[1]!) {
          case 'Y': c = c.cy(ci, ti); break
          case 'H': c = c.ch(ci, ti); break
          default:  throw new TypeError(`fromCirq: unknown controlled gate '${namedCtrl[1]}'`)
        }
        continue
      }

      // cirq.GATE(q[i], ...)
      const simple = line.match(/^cirq\.(\w+)\((.+)\)$/)
      if (simple) {
        const qs = qis(simple[2]!)
        switch (simple[1]!) {
          case 'I':     c = c.id(qs[0]!);                       break
          case 'H':     c = c.h(qs[0]!);                        break
          case 'X':     c = c.x(qs[0]!);                        break
          case 'Y':     c = c.y(qs[0]!);                        break
          case 'Z':     c = c.z(qs[0]!);                        break
          case 'S':     c = c.s(qs[0]!);                        break
          case 'T':     c = c.t(qs[0]!);                        break
          case 'CNOT':  c = c.cnot(qs[0]!, qs[1]!);            break
          case 'CX':    c = c.cnot(qs[0]!, qs[1]!);            break
          case 'CZ':    c = c.cz(qs[0]!, qs[1]!);              break
          case 'SWAP':  c = c.swap(qs[0]!, qs[1]!);            break
          case 'CCNOT': c = c.ccx(qs[0]!, qs[1]!, qs[2]!);    break
          case 'CSWAP': c = c.cswap(qs[0]!, qs[1]!, qs[2]!);  break
          default:      throw new TypeError(`fromCirq: unknown gate '${simple[1]}'`)
        }
        continue
      }
    }
    return c
  }

  /**
   * Parse a Qiskit Qobj JSON object into a Circuit.
   * Accepts the object returned by `qc.qobj()` or IBM Quantum job results.
   * Only the first experiment is used.
   */
  static fromQobj(qobj: {
    experiments: ReadonlyArray<{
      header:       { n_qubits: number; creg_sizes?: ReadonlyArray<readonly [string, number]> }
      instructions: ReadonlyArray<{ name: string; qubits: number[]; params?: number[]; memory?: number[] }>
    }>
  }): Circuit {
    const exp = qobj.experiments[0]
    if (!exp) throw new TypeError('fromQobj: no experiments found')

    let c = new Circuit(exp.header.n_qubits)

    // Build memory-slot → (cregName, bitOffset) map
    const slotMap: Array<{ creg: string; bit: number }> = []
    for (const [name, size] of exp.header.creg_sizes ?? []) {
      c = c.creg(name, size)
      for (let b = 0; b < size; b++) slotMap.push({ creg: name, bit: b })
    }

    for (const { name, qubits: qs, params: p = [], memory: mem = [] } of exp.instructions) {
      switch (name) {
        case 'id':    c = c.id(qs[0]!);                                        break
        case 'h':     c = c.h(qs[0]!);                                         break
        case 'x':     c = c.x(qs[0]!);                                         break
        case 'y':     c = c.y(qs[0]!);                                         break
        case 'z':     c = c.z(qs[0]!);                                         break
        case 's':     c = c.s(qs[0]!);                                         break
        case 'sdg':   c = c.si(qs[0]!);                                        break
        case 't':     c = c.t(qs[0]!);                                         break
        case 'tdg':   c = c.ti(qs[0]!);                                        break
        case 'sx':    c = c.v(qs[0]!);                                         break
        case 'sxdg':  c = c.vi(qs[0]!);                                        break
        case 'rx':    c = c.rx(p[0]!, qs[0]!);                                 break
        case 'ry':    c = c.ry(p[0]!, qs[0]!);                                 break
        case 'rz':    c = c.rz(p[0]!, qs[0]!);                                 break
        case 'u1': case 'p':
                      c = c.u1(p[0]!, qs[0]!);                                 break
        case 'u2':    c = c.u2(p[0]!, p[1]!, qs[0]!);                          break
        case 'u3':    c = c.u3(p[0]!, p[1]!, p[2]!, qs[0]!);                   break
        case 'cx': case 'cnot':
                      c = c.cnot(qs[0]!, qs[1]!);                              break
        case 'cy':    c = c.cy(qs[0]!, qs[1]!);                                break
        case 'cz':    c = c.cz(qs[0]!, qs[1]!);                                break
        case 'ch':    c = c.ch(qs[0]!, qs[1]!);                                break
        case 'swap':  c = c.swap(qs[0]!, qs[1]!);                              break
        case 'iswap': c = c.iswap(qs[0]!, qs[1]!);                             break
        case 'crx':   c = c.crx(p[0]!, qs[0]!, qs[1]!);                        break
        case 'cry':   c = c.cry(p[0]!, qs[0]!, qs[1]!);                        break
        case 'crz':   c = c.crz(p[0]!, qs[0]!, qs[1]!);                        break
        case 'cu1':   c = c.cu1(p[0]!, qs[0]!, qs[1]!);                        break
        case 'cu2':   c = c.cu2(p[0]!, p[1]!, qs[0]!, qs[1]!);                 break
        case 'cu3':   c = c.cu3(p[0]!, p[1]!, p[2]!, qs[0]!, qs[1]!);          break
        case 'rxx':   c = c.xx(p[0]!, qs[0]!, qs[1]!);                         break
        case 'ryy':   c = c.yy(p[0]!, qs[0]!, qs[1]!);                         break
        case 'rzz':   c = c.zz(p[0]!, qs[0]!, qs[1]!);                         break
        case 'ccx':   c = c.ccx(qs[0]!, qs[1]!, qs[2]!);                       break
        case 'cswap': c = c.cswap(qs[0]!, qs[1]!, qs[2]!);                     break
        case 'reset': c = c.reset(qs[0]!);                                      break
        case 'barrier': c = c.barrier(...qs);                                   break
        case 'measure': {
          const slot = slotMap[mem[0]!]
          if (slot) c = c.measure(qs[0]!, slot.creg, slot.bit)
          break
        }
        case 'snapshot': case 'save_statevector': break  // Qiskit metadata — skip
        default: throw new TypeError(`fromQobj: unknown instruction '${name}'`)
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

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    lines.push(`qc.cx(${op.control}, ${op.target})`);             break
        case 'swap':    lines.push(`qc.swap(${op.a}, ${op.b})`);                      break
        case 'toffoli': lines.push(`qc.ccx(${op.c1}, ${op.c2}, ${op.target})`);      break
        case 'cswap':   lines.push(`qc.cswap(${op.control}, ${op.a}, ${op.b})`);     break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no Qiskit representation")
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
        case 'unitary': throw new TypeError("Gate 'unitary' has no Qiskit representation")
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
            case 'vz': lines.push(`qc.rz(${angle()}, ${q})`);        break
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
            case 'cu2':  lines.push(`qc.cu2(${pyAngle(p![0]!)}, ${pyAngle(p![1]!)}, ${c}, ${t})`);                    break
            case 'cu3':  lines.push(`qc.cu3(${pyAngle(p![0]!)}, ${pyAngle(p![1]!)}, ${pyAngle(p![2]!)}, ${c}, ${t})`); break
            default:     lines.push(`qc.${n}(${c}, ${t})`)
          }
          break
        }
      }
    }
    return [...imports, ...lines].join('\n')
  }

  /** Build the list of `    cirq.*` op strings shared by toCirq() and toTFQ(). */
  #cirqOps(): string[] {
    const ops: string[] = []
    const go   = (gate: string, qs: number[]) =>
      `    ${gate}(${qs.map(q => `q[${q}]`).join(', ')}),`
    const rads = (r: number) => `rads=${pyAngle(r)}`

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    ops.push(go('cirq.CNOT', [op.control, op.target]));    break
        case 'swap':    ops.push(go('cirq.SWAP', [op.a, op.b]));               break
        case 'toffoli': ops.push(go('cirq.CCNOT', [op.c1, op.c2, op.target])); break
        case 'cswap':   ops.push(go('cirq.CSWAP', [op.control, op.a, op.b])); break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no Cirq representation")
        case 'measure': throw new TypeError('measure ops cannot be serialized to Cirq via toCirq(); use cirq.measure() manually')
        case 'reset':   throw new TypeError('reset ops cannot be serialized to Cirq via toCirq()')
        case 'if':      throw new TypeError('if ops cannot be serialized to Cirq')
        case 'two':     throw new TypeError(`Gate '${op.meta?.name ?? 'two'}' has no Cirq representation`)
        case 'unitary': throw new TypeError("Gate 'unitary' has no Cirq representation")
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          switch (n) {
            case 'id':  ops.push(go('cirq.I', [q]));                                                  break
            case 'h':   ops.push(go('cirq.H', [q]));                                                  break
            case 'x':   ops.push(go('cirq.X', [q]));                                                  break
            case 'y':   ops.push(go('cirq.Y', [q]));                                                  break
            case 'z':   ops.push(go('cirq.Z', [q]));                                                  break
            case 's':   ops.push(go('cirq.S', [q]));                                                  break
            case 'si':  ops.push(`    cirq.ZPowGate(exponent=-0.5)(q[${q}]),`);                       break
            case 't':   ops.push(go('cirq.T', [q]));                                                  break
            case 'ti':  ops.push(`    cirq.ZPowGate(exponent=-0.25)(q[${q}]),`);                      break
            case 'v':   ops.push(`    cirq.X**0.5(q[${q}]),`);                                        break
            case 'vi':  ops.push(`    (cirq.X**-0.5)(q[${q}]),`);                                     break
            case 'r2':  ops.push(`    cirq.rz(${rads(Math.PI / 2)})(q[${q}]),`);                      break
            case 'r4':  ops.push(`    cirq.rz(${rads(Math.PI / 4)})(q[${q}]),`);                      break
            case 'r8':  ops.push(`    cirq.rz(${rads(Math.PI / 8)})(q[${q}]),`);                      break
            case 'rx':  ops.push(`    cirq.rx(${rads(p![0]!)})(q[${q}]),`);                           break
            case 'ry':  ops.push(`    cirq.ry(${rads(p![0]!)})(q[${q}]),`);                           break
            case 'rz':  ops.push(`    cirq.rz(${rads(p![0]!)})(q[${q}]),`);                           break
            case 'vz':  ops.push(`    cirq.rz(${rads(p![0]!)})(q[${q}]),`);                           break
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
            case 'cy':   ops.push(`    cirq.Y.controlled()(q[${c}], q[${t}]),`);                                        break
            case 'cz':   ops.push(go('cirq.CZ', [c, t]));                                                               break
            case 'ch':   ops.push(`    cirq.H.controlled()(q[${c}], q[${t}]),`);                                        break
            case 'cr2':  ops.push(`    cirq.rz(${rads(Math.PI/2)}).controlled()(q[${c}], q[${t}]),`);                   break
            case 'cr4':  ops.push(`    cirq.rz(${rads(Math.PI/4)}).controlled()(q[${c}], q[${t}]),`);                   break
            case 'cr8':  ops.push(`    cirq.rz(${rads(Math.PI/8)}).controlled()(q[${c}], q[${t}]),`);                   break
            case 'crx':  ops.push(`    cirq.rx(${rads(p![0]!)}).controlled()(q[${c}], q[${t}]),`);                      break
            case 'cry':  ops.push(`    cirq.ry(${rads(p![0]!)}).controlled()(q[${c}], q[${t}]),`);                      break
            case 'crz':  ops.push(`    cirq.rz(${rads(p![0]!)}).controlled()(q[${c}], q[${t}]),`);                      break
            case 'cu1':  ops.push(`    cirq.ZPowGate(exponent=${pyAngle(p![0]! / Math.PI)}).controlled()(q[${c}], q[${t}]),`); break
            case 'cs':   ops.push(`    cirq.ZPowGate(exponent=0.5).controlled()(q[${c}], q[${t}]),`);                   break
            case 'ct':   ops.push(`    cirq.ZPowGate(exponent=0.25).controlled()(q[${c}], q[${t}]),`);                  break
            case 'csdg': ops.push(`    cirq.ZPowGate(exponent=-0.5).controlled()(q[${c}], q[${t}]),`);                  break
            case 'ctdg': ops.push(`    cirq.ZPowGate(exponent=-0.25).controlled()(q[${c}], q[${t}]),`);                 break
            default:     throw new TypeError(`Gate '${n}' has no Cirq representation`)
          }
          break
        }
      }
    }
    return ops
  }

  /**
   * Emit Python code for Google Cirq.
   * Gate coverage: H/X/Y/Z/S/T, rx/ry/rz, r2/r4/r8, u1/u3,
   * CNOT/CZ/CY/CH/swap/CCNOT/CSWAP, crx/cry/crz/cu1/cu3.
   * Throws for gpi/gpi2/ms/xx/yy/zz/xy/iswap/srswap/if.
   */
  toCirq(): string {
    const ops  = this.#cirqOps()
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
   * Emit Python code for TensorFlow Quantum (TFQ).
   *
   * TFQ wraps Cirq circuits; qubits must be `cirq.GridQubit` instances.
   * The output includes the `tfq.convert_to_tensor` call needed to feed
   * the circuit into a TFQ layer.
   *
   * ```python
   * tensor = tfq.convert_to_tensor([circuit])
   * ```
   *
   * Same gate coverage and restrictions as `toCirq()`.
   */
  toTFQ(): string {
    const ops  = this.#cirqOps()
    const body = ops.length ? ops.join('\n') : '    # empty circuit'
    return [
      'import cirq',
      'import math',
      'import tensorflow_quantum as tfq',
      '',
      `q = [cirq.GridQubit(0, i) for i in range(${this.qubits})]`,
      'circuit = cirq.Circuit([',
      body,
      '])',
      'tensor = tfq.convert_to_tensor([circuit])',
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
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    body.push(`        CNOT(q[${op.control}], q[${op.target}]);`);                                         break
        case 'swap':    body.push(`        SWAP(q[${op.a}], q[${op.b}]);`);                                                    break
        case 'toffoli': body.push(`        CCNOT(q[${op.c1}], q[${op.c2}], q[${op.target}]);`);                               break
        case 'cswap':   body.push(`        Controlled SWAP([q[${op.control}]], (q[${op.a}], q[${op.b}]));`);                   break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no Q# representation")
        case 'measure': body.push(`        set ${op.creg}w${op.bit} = M(q[${op.q}]) == One;`);                                 break
        case 'reset':   body.push(`        Reset(q[${op.q}]);`);                                                               break
        case 'if':      throw new TypeError('if ops cannot be serialized to Q#')
        case 'two':     throw new TypeError(`Gate '${op.meta?.name ?? 'two'}' has no Q# representation`)
        case 'unitary': throw new TypeError("Gate 'unitary' has no Q# representation")
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          const ang = () => qsharpAngle(p![0]!)
          switch (n) {
            case 'id':  body.push(`        I(q[${q}]);`);                                   break
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
            case 'vz':  body.push(`        Rz(${ang()}, q[${q}]);`);                        break
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

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    used.add('CNOT');  body.push(`p += CNOT(${op.control}, ${op.target})`);          break
        case 'swap':    used.add('SWAP');  body.push(`p += SWAP(${op.a}, ${op.b})`);                     break
        case 'toffoli': used.add('CCNOT'); body.push(`p += CCNOT(${op.c1}, ${op.c2}, ${op.target})`);   break
        case 'cswap':   used.add('CSWAP'); body.push(`p += CSWAP(${op.control}, ${op.a}, ${op.b})`);    break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no pyQuil representation")
        case 'measure': used.add('MEASURE'); body.push(`p += MEASURE(${op.q}, ro[${op.bit}])`);          break
        case 'reset':   body.push(`p += RESET(${op.q})`);                                                break
        case 'if':      throw new TypeError('if ops cannot be serialized to pyQuil')
        case 'two': {
          const n = op.meta?.name
          if (n === 'iswap') { used.add('ISWAP'); body.push(`p += ISWAP(${op.a}, ${op.b})`); break }
          throw new TypeError(`Gate '${n ?? 'two'}' has no pyQuil representation`)
        }
        case 'unitary': throw new TypeError("Gate 'unitary' has no pyQuil representation")
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          const ang = (i = 0) => pyAngle(p![i]!)
          switch (n) {
            case 'id':  used.add('I');   body.push(`p += I(${q})`);              break
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
            case 'vz':  used.add('RZ');  body.push(`p += RZ(${ang()}, ${q})`);   break
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

  /**
   * Emit a Quil (Quantum Instruction Language) program for Rigetti hardware.
   *
   * Gate coverage: I/H/X/Y/Z/S/T and daggers, RX/RY/RZ, PHASE,
   * CNOT/CZ/SWAP/ISWAP/CCNOT/CSWAP, controlled family via CONTROLLED/CPHASE.
   * Throws for gates with no Quil representation: gpi/gpi2/ms/xx/yy/zz/xy/srswap/u2/u3/cu2/cu3/if.
   */
  toQuil(): string {
    const lines: string[] = []
    const ang  = (r: number) => fmtAngle(r, 'pi')

    for (const [name, size] of this.#cregs) lines.push(`DECLARE ${name} BIT[${size}]`)
    if (this.#cregs.size) lines.push('')

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    lines.push(`CNOT ${op.control} ${op.target}`);              break
        case 'swap':    lines.push(`SWAP ${op.a} ${op.b}`);                         break
        case 'toffoli': lines.push(`CCNOT ${op.c1} ${op.c2} ${op.target}`);        break
        case 'cswap':   lines.push(`CSWAP ${op.control} ${op.a} ${op.b}`);         break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no Quil representation")
        case 'measure': lines.push(`MEASURE ${op.q} ${op.creg}[${op.bit}]`);       break
        case 'reset':   lines.push(`RESET ${op.q}`);                                break
        case 'if':      throw new TypeError('if ops cannot be serialized to Quil')
        case 'two': {
          const n = op.meta?.name
          if (n === 'iswap') { lines.push(`ISWAP ${op.a} ${op.b}`); break }
          throw new TypeError(`Gate '${n ?? 'two'}' has no Quil representation`)
        }
        case 'unitary': throw new TypeError("Gate 'unitary' has no Quil representation")
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          switch (n) {
            case 'id':  lines.push(`I ${q}`);                     break
            case 'h':   lines.push(`H ${q}`);                     break
            case 'x':   lines.push(`X ${q}`);                     break
            case 'y':   lines.push(`Y ${q}`);                     break
            case 'z':   lines.push(`Z ${q}`);                     break
            case 's':   lines.push(`S ${q}`);                     break
            case 'si':  lines.push(`DAGGER S ${q}`);              break
            case 't':   lines.push(`T ${q}`);                     break
            case 'ti':  lines.push(`DAGGER T ${q}`);              break
            case 'v':   lines.push(`RX(pi/2) ${q}`);              break  // √X up to global phase
            case 'vi':  lines.push(`RX(-pi/2) ${q}`);             break
            case 'r2':  lines.push(`RZ(pi/2) ${q}`);              break
            case 'r4':  lines.push(`RZ(pi/4) ${q}`);              break
            case 'r8':  lines.push(`RZ(pi/8) ${q}`);              break
            case 'rx':  lines.push(`RX(${ang(p![0]!)}) ${q}`);    break
            case 'ry':  lines.push(`RY(${ang(p![0]!)}) ${q}`);    break
            case 'rz':  lines.push(`RZ(${ang(p![0]!)}) ${q}`);    break
            case 'vz':  lines.push(`RZ(${ang(p![0]!)}) ${q}`);    break
            case 'u1':  lines.push(`PHASE(${ang(p![0]!)}) ${q}`); break
            case 'gpi': case 'gpi2':
              throw new TypeError(`Gate '${n}' has no Quil representation`)
            default:
              throw new TypeError(`Gate '${n}' has no Quil representation`)
          }
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { name: n, params: p } = op.meta
          const [c, t] = [op.control, op.target]
          switch (n) {
            case 'cy':   lines.push(`CONTROLLED Y ${c} ${t}`);                     break
            case 'cz':   lines.push(`CZ ${c} ${t}`);                               break
            case 'ch':   lines.push(`CONTROLLED H ${c} ${t}`);                     break
            case 'crx':  lines.push(`CONTROLLED RX(${ang(p![0]!)}) ${c} ${t}`);   break
            case 'cry':  lines.push(`CONTROLLED RY(${ang(p![0]!)}) ${c} ${t}`);   break
            case 'crz':  lines.push(`CONTROLLED RZ(${ang(p![0]!)}) ${c} ${t}`);   break
            case 'cu1':  lines.push(`CPHASE(${ang(p![0]!)}) ${c} ${t}`);          break
            case 'cs':   lines.push(`CPHASE(pi/2) ${c} ${t}`);                    break
            case 'ct':   lines.push(`CPHASE(pi/4) ${c} ${t}`);                    break
            case 'csdg': lines.push(`CPHASE(-pi/2) ${c} ${t}`);                   break
            case 'ctdg': lines.push(`CPHASE(-pi/4) ${c} ${t}`);                   break
            case 'cr2':  lines.push(`CPHASE(pi/2) ${c} ${t}`);                    break
            case 'cr4':  lines.push(`CPHASE(pi/4) ${c} ${t}`);                    break
            case 'cr8':  lines.push(`CPHASE(pi/8) ${c} ${t}`);                    break
            default:
              throw new TypeError(`Gate '${n}' has no Quil representation`)
          }
          break
        }
      }
    }
    return lines.join('\n')
  }

  /**
   * Emit a `quantikz` LaTeX environment for the circuit.
   *
   * The output is a self-contained `\begin{quantikz}...\end{quantikz}` block
   * using the `tikz-quantikz` package.  Paste it into any LaTeX document that
   * loads `\usepackage{quantikz}` (and `\usepackage{amsmath}` for `\ket{}`).
   *
   * Gate coverage: all single-qubit gates, CNOT, SWAP, controlled family,
   * Toffoli, Fredkin, two-qubit interaction gates, measure, reset.
   * `if` ops are silently skipped (no standard quantikz representation).
   *
   * @example
   * console.log(new Circuit(2).h(0).cnot(0, 1).toLatex())
   * // \begin{quantikz}
   * // \lstick{$q_{0}$} & \gate{H} & \ctrl{1} & \qw \\
   * // \lstick{$q_{1}$} & \qw & \targ{} & \qw
   * // \end{quantikz}
   */
  toLatex(): string {
    const n = this.qubits
    if (n === 0) return '\\begin{quantikz}\n\\end{quantikz}'

    const ops = flattenOps(this.#ops).filter(op => op.kind !== 'if')

    // Column assignment (same greedy algorithm as draw() / toSVG())
    const colOf = new Array<number>(n).fill(0)
    type PlacedOp = { op: Op; col: number }
    const placed: PlacedOp[] = []
    for (const op of ops) {
      const qs = opQubits(op)
      if (qs.length === 0) continue
      const minQ = Math.min(...qs), maxQ = Math.max(...qs)
      let col = 0
      for (let q = minQ; q <= maxQ; q++) col = Math.max(col, colOf[q]!)
      for (let q = minQ; q <= maxQ; q++) colOf[q] = col + 1
      placed.push({ op, col })
    }
    const numCols = Math.max(0, ...colOf)

    // cell[q][c] — quantikz command for each (qubit, column) slot; '' = absorbed by a spanning gate
    const cell: string[][] = Array.from({ length: n }, () =>
      new Array<string>(numCols).fill('\\qw')
    )

    for (const { op, col } of placed) {
      const qs = opQubits(op)
      const minQ = Math.min(...qs), maxQ = Math.max(...qs)

      switch (op.kind) {
        case 'single':
          cell[op.q]![col] = `\\gate{${latexSingleLabel(op)}}`
          break
        case 'cnot':
          cell[op.control]![col] = `\\ctrl{${op.target - op.control}}`
          cell[op.target]![col]  = '\\targ{}'
          break
        case 'swap':
          cell[op.a]![col] = `\\swap{${op.b - op.a}}`
          cell[op.b]![col] = `\\swap{${op.a - op.b}}`
          break
        case 'controlled':
          cell[op.control]![col] = `\\ctrl{${op.target - op.control}}`
          cell[op.target]![col]  = `\\gate{${latexCtrlTargetLabel(op)}}`
          break
        case 'toffoli':
          cell[op.c1]![col]     = `\\ctrl{${op.target - op.c1}}`
          cell[op.c2]![col]     = `\\ctrl{${op.target - op.c2}}`
          cell[op.target]![col] = '\\targ{}'
          break
        case 'cswap':
          cell[op.control]![col] = `\\ctrl{${Math.min(op.a, op.b) - op.control}}`
          cell[Math.min(op.a, op.b)]![col] = `\\swap{${Math.max(op.a, op.b) - Math.min(op.a, op.b)}}`
          cell[Math.max(op.a, op.b)]![col] = `\\swap{${Math.min(op.a, op.b) - Math.max(op.a, op.b)}}`
          break
        case 'csrswap': {
          const [tA, tB] = [Math.min(op.a, op.b), Math.max(op.a, op.b)]
          const span = tB - tA + 1
          cell[op.control]![col] = `\\ctrl{${tA - op.control}}`
          cell[tA]![col]         = `\\gate[${span}]{\\sqrt{\\text{iSWAP}}}`
          for (let q = tA + 1; q <= tB; q++) cell[q]![col] = ''
          break
        }
        case 'two': {
          const span = maxQ - minQ + 1
          cell[minQ]![col] = `\\gate[${span}]{${latexTwoLabel(op)}}`
          for (let q = minQ + 1; q <= maxQ; q++) cell[q]![col] = ''
          break
        }
        case 'unitary': {
          const span = maxQ - minQ + 1
          cell[minQ]![col] = span === 1 ? `\\gate{U}` : `\\gate[${span}]{U}`
          for (let q = minQ + 1; q <= maxQ; q++) cell[q]![col] = ''
          break
        }
        case 'measure':
          cell[op.q]![col] = '\\meter{}'
          break
        case 'reset':
          cell[op.q]![col] = '\\gate{\\ket{0}}'
          break
      }
    }

    const rows = Array.from({ length: n }, (_, q) => {
      const cells = cell[q]!.filter(c => c !== '').join(' & ')
      return `\\lstick{$q_{${q}}$} & ${cells} & \\qw`
    })

    return `\\begin{quantikz}\n${rows.join(' \\\\\n')}\n\\end{quantikz}`
  }

  /**
   * Emit Python code for Amazon Braket's `Circuit` API.
   *
   * Gate coverage: full single-qubit set, rx/ry/rz, phaseshift (u1), xx/yy/zz/xy,
   * cnot/cy/cz/swap/iswap, ccnot/cswap, controlled family via control= kwarg.
   * Throws for gates with no Braket representation: gpi/gpi2/ms/srswap/u2/u3/cu2/cu3/measure/reset/if.
   */
  toBraket(): string {
    const lines: string[] = []
    const a = (r: number) => pyAngle(r)

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    lines.push(`circ.cnot(${op.control}, ${op.target})`);           break
        case 'swap':    lines.push(`circ.swap(${op.a}, ${op.b})`);                      break
        case 'toffoli': lines.push(`circ.ccnot(${op.c1}, ${op.c2}, ${op.target})`);    break
        case 'cswap':   lines.push(`circ.cswap(${op.control}, ${op.a}, ${op.b})`);     break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no Braket representation")
        case 'measure': throw new TypeError('measure ops cannot be serialized to Braket via toBraket()')
        case 'reset':   throw new TypeError('reset ops cannot be serialized to Braket via toBraket()')
        case 'if':      throw new TypeError('if ops cannot be serialized to Braket')
        case 'two': {
          const n = op.meta?.name
          const [qa, qb] = [op.a, op.b]
          if (n === 'xx')    { lines.push(`circ.xx(${qa}, ${qb}, ${a(op.meta!.params![0]!)})`);    break }
          if (n === 'yy')    { lines.push(`circ.yy(${qa}, ${qb}, ${a(op.meta!.params![0]!)})`);    break }
          if (n === 'zz')    { lines.push(`circ.zz(${qa}, ${qb}, ${a(op.meta!.params![0]!)})`);    break }
          if (n === 'xy')    { lines.push(`circ.xy(${qa}, ${qb}, ${a(op.meta!.params![0]!)})`);    break }
          if (n === 'iswap') { lines.push(`circ.iswap(${qa}, ${qb})`);                             break }
          throw new TypeError(`Gate '${n ?? 'two'}' has no Braket representation`)
        }
        case 'unitary': throw new TypeError("Gate 'unitary' has no Braket representation")
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = op.q
          switch (n) {
            case 'id':  lines.push(`circ.i(${q})`);                             break
            case 'h':   lines.push(`circ.h(${q})`);                             break
            case 'x':   lines.push(`circ.x(${q})`);                             break
            case 'y':   lines.push(`circ.y(${q})`);                             break
            case 'z':   lines.push(`circ.z(${q})`);                             break
            case 's':   lines.push(`circ.s(${q})`);                             break
            case 'si':  lines.push(`circ.si(${q})`);                            break
            case 't':   lines.push(`circ.t(${q})`);                             break
            case 'ti':  lines.push(`circ.ti(${q})`);                            break
            case 'v':   lines.push(`circ.v(${q})`);                             break
            case 'vi':  lines.push(`circ.vi(${q})`);                            break
            case 'r2':  lines.push(`circ.rz(${q}, math.pi/2)`);                 break
            case 'r4':  lines.push(`circ.rz(${q}, math.pi/4)`);                 break
            case 'r8':  lines.push(`circ.rz(${q}, math.pi/8)`);                 break
            case 'rx':  lines.push(`circ.rx(${q}, ${a(p![0]!)})`);              break
            case 'ry':  lines.push(`circ.ry(${q}, ${a(p![0]!)})`);              break
            case 'rz':  lines.push(`circ.rz(${q}, ${a(p![0]!)})`);              break
            case 'vz':  lines.push(`circ.rz(${q}, ${a(p![0]!)})`);              break
            case 'u1':  lines.push(`circ.phaseshift(${q}, ${a(p![0]!)})`);      break
            case 'gpi': case 'gpi2':
              throw new TypeError(`Gate '${n}' has no Braket representation`)
            default:
              throw new TypeError(`Gate '${n}' has no Braket representation`)
          }
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { name: n, params: p } = op.meta
          const [c, t] = [op.control, op.target]
          switch (n) {
            case 'cy':   lines.push(`circ.cy(${c}, ${t})`);                                          break
            case 'cz':   lines.push(`circ.cz(${c}, ${t})`);                                          break
            case 'ch':   lines.push(`circ.h(${t}, control=${c})`);                                   break
            case 'crx':  lines.push(`circ.rx(${t}, ${a(p![0]!)}, control=${c})`);                    break
            case 'cry':  lines.push(`circ.ry(${t}, ${a(p![0]!)}, control=${c})`);                    break
            case 'crz':  lines.push(`circ.rz(${t}, ${a(p![0]!)}, control=${c})`);                    break
            case 'cr2':  lines.push(`circ.rz(${t}, math.pi/2, control=${c})`);                       break
            case 'cr4':  lines.push(`circ.rz(${t}, math.pi/4, control=${c})`);                       break
            case 'cr8':  lines.push(`circ.rz(${t}, math.pi/8, control=${c})`);                       break
            case 'cu1':  lines.push(`circ.phaseshift(${t}, ${a(p![0]!)}, control=${c})`);            break
            case 'cs':   lines.push(`circ.phaseshift(${t}, math.pi/2, control=${c})`);               break
            case 'ct':   lines.push(`circ.phaseshift(${t}, math.pi/4, control=${c})`);               break
            case 'csdg': lines.push(`circ.phaseshift(${t}, -math.pi/2, control=${c})`);              break
            case 'ctdg': lines.push(`circ.phaseshift(${t}, -math.pi/4, control=${c})`);              break
            default:
              throw new TypeError(`Gate '${n}' has no Braket representation`)
          }
          break
        }
      }
    }

    return [
      'import math',
      'from braket.circuits import Circuit',
      '',
      'circ = Circuit()',
      ...lines,
    ].join('\n')
  }

  /**
   * Emit a CUDA Quantum (cudaq) Python kernel.
   *
   * ```python
   * import math
   * import cudaq
   *
   * kernel = cudaq.make_kernel()
   * q = kernel.qalloc(2)
   * kernel.h(q[0])
   * kernel.cx(q[0], q[1])
   * ```
   *
   * Limitations: classical ops (measure/reset/if), IonQ-native gates (gpi/gpi2/ms),
   * √X variants (v/vi), and interaction gates (xx/yy/zz/xy/iswap/srswap) have no
   * direct CudaQ equivalent and will throw.
   */
  toCudaQ(): string {
    const lines: string[] = []
    const a  = (r: number) => pyAngle(r)
    const qi = (i: number) => `q[${i}]`

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    lines.push(`kernel.cx(${qi(op.control)}, ${qi(op.target)})`);              break
        case 'swap':    lines.push(`kernel.swap(${qi(op.a)}, ${qi(op.b)})`);                       break
        case 'toffoli': lines.push(`kernel.ccx(${qi(op.c1)}, ${qi(op.c2)}, ${qi(op.target)})`);   break
        case 'cswap':   lines.push(`kernel.cswap(${qi(op.control)}, ${qi(op.a)}, ${qi(op.b)})`);  break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no CudaQ representation")
        case 'measure': throw new TypeError('measure ops cannot be serialized to CudaQ via toCudaQ()')
        case 'reset':   throw new TypeError('reset ops cannot be serialized to CudaQ via toCudaQ()')
        case 'if':      throw new TypeError('if ops cannot be serialized to CudaQ via toCudaQ()')
        case 'two':     throw new TypeError(`Gate '${op.meta?.name ?? 'two'}' has no CudaQ representation`)
        case 'unitary': throw new TypeError("Gate 'unitary' has no CudaQ representation")
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: n, params: p } = op.meta
          const q = qi(op.q)
          switch (n) {
            case 'id':  /* no-op — CudaQ has no identity gate */             break
            case 'h':   lines.push(`kernel.h(${q})`);                        break
            case 'x':   lines.push(`kernel.x(${q})`);                        break
            case 'y':   lines.push(`kernel.y(${q})`);                        break
            case 'z':   lines.push(`kernel.z(${q})`);                        break
            case 's':   lines.push(`kernel.s(${q})`);                        break
            case 'si':  lines.push(`kernel.sdg(${q})`);                      break
            case 't':   lines.push(`kernel.t(${q})`);                        break
            case 'ti':  lines.push(`kernel.tdg(${q})`);                      break
            case 'r2':  lines.push(`kernel.rz(math.pi/2, ${q})`);            break
            case 'r4':  lines.push(`kernel.rz(math.pi/4, ${q})`);            break
            case 'r8':  lines.push(`kernel.rz(math.pi/8, ${q})`);            break
            case 'rx':  lines.push(`kernel.rx(${a(p![0]!)}, ${q})`);         break
            case 'ry':  lines.push(`kernel.ry(${a(p![0]!)}, ${q})`);         break
            case 'rz':  lines.push(`kernel.rz(${a(p![0]!)}, ${q})`);         break
            case 'vz':  lines.push(`kernel.rz(${a(p![0]!)}, ${q})`);         break
            case 'u1':  lines.push(`kernel.r1(${a(p![0]!)}, ${q})`);         break
            case 'u2':  lines.push(`kernel.u3(math.pi/2, ${a(p![0]!)}, ${a(p![1]!)}, ${q})`); break
            case 'u3':  lines.push(`kernel.u3(${a(p![0]!)}, ${a(p![1]!)}, ${a(p![2]!)}, ${q})`); break
            case 'v': case 'vi':
              throw new TypeError(`Gate '${n}' has no CudaQ representation`)
            case 'gpi': case 'gpi2':
              throw new TypeError(`Gate '${n}' has no CudaQ representation`)
            default:
              throw new TypeError(`Gate '${n}' has no CudaQ representation`)
          }
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { name: n, params: p } = op.meta
          const [c, t] = [qi(op.control), qi(op.target)]
          switch (n) {
            case 'cy':   lines.push(`kernel.cy(${c}, ${t})`);                         break
            case 'cz':   lines.push(`kernel.cz(${c}, ${t})`);                         break
            case 'ch':   lines.push(`kernel.ch(${c}, ${t})`);                         break
            case 'crx':  lines.push(`kernel.crx(${a(p![0]!)}, ${c}, ${t})`);          break
            case 'cry':  lines.push(`kernel.cry(${a(p![0]!)}, ${c}, ${t})`);          break
            case 'crz':  lines.push(`kernel.crz(${a(p![0]!)}, ${c}, ${t})`);          break
            case 'cr2':  lines.push(`kernel.crz(math.pi/2, ${c}, ${t})`);             break
            case 'cr4':  lines.push(`kernel.crz(math.pi/4, ${c}, ${t})`);             break
            case 'cr8':  lines.push(`kernel.crz(math.pi/8, ${c}, ${t})`);             break
            case 'cu1':  lines.push(`kernel.cr1(${a(p![0]!)}, ${c}, ${t})`);          break
            case 'cs':   lines.push(`kernel.cr1(math.pi/2, ${c}, ${t})`);             break
            case 'ct':   lines.push(`kernel.cr1(math.pi/4, ${c}, ${t})`);             break
            case 'csdg': lines.push(`kernel.cr1(-math.pi/2, ${c}, ${t})`);            break
            case 'ctdg': lines.push(`kernel.cr1(-math.pi/4, ${c}, ${t})`);            break
            default:
              throw new TypeError(`Gate '${n}' has no CudaQ representation`)
          }
          break
        }
      }
    }

    return [
      'import math',
      'import cudaq',
      '',
      'kernel = cudaq.make_kernel()',
      `q = kernel.qalloc(${this.qubits})`,
      ...lines,
    ].join('\n')
  }

  /**
   * Emit a Quirk (algassert.com/quirk) JSON circuit descriptor.
   *
   * Returns a JSON string `{"cols":[...]}` that can be pasted into Quirk's
   * "Load" dialog or appended to the URL as `#circuit=<encoded>`.
   *
   * Column structure: each column is an array indexed by qubit.
   * `1` = idle wire, `"•"` = control, named strings or `{id, arg}` = gates.
   * Angles are in half-turns: arg = θ/π (Rx(π/2) → arg=0.5).
   *
   * Limitations: U2/U3/gpi/gpi2/ms, interaction gates (xx/yy/zz/xy/iswap/srswap),
   * and if/reset ops have no Quirk equivalent and will throw.
   * Measure ops emit `"Measure"`. U1 is approximated as Rz (same unitary up to global phase).
   */
  toQuirk(): string {
    type QuirkGate = string | number | { id: string; arg: number }
    const cols: QuirkGate[][] = []
    const n   = this.qubits
    const rot = (id: string, theta: number): { id: string; arg: number } =>
      ({ id, arg: theta / Math.PI })

    const col = (entries: Partial<Record<number, QuirkGate>>) => {
      const c: QuirkGate[] = new Array(n).fill(1)
      for (const [q, g] of Object.entries(entries)) c[+q] = g!
      while (c.length > 1 && c[c.length - 1] === 1) c.pop()
      cols.push(c)
    }

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'cnot':    col({ [op.control]: '•',    [op.target]: 'X' });                           break
        case 'swap':    col({ [op.a]: 'Swap',        [op.b]: 'Swap' });                            break
        case 'toffoli': col({ [op.c1]: '•', [op.c2]: '•', [op.target]: 'X' });                   break
        case 'cswap':   col({ [op.control]: '•', [op.a]: 'Swap', [op.b]: 'Swap' });               break
        case 'csrswap': throw new TypeError("Gate 'csrswap' has no Quirk representation")
        case 'measure': col({ [op.q]: 'Measure' });                                                break
        case 'reset':   throw new TypeError('reset ops cannot be serialized to Quirk via toQuirk()')
        case 'if':      throw new TypeError('if ops cannot be serialized to Quirk via toQuirk()')
        case 'two': {
          const n = op.meta?.name
          throw new TypeError(`Gate '${n ?? 'two'}' has no Quirk representation`)
        }
        case 'single': {
          if (!op.meta) throw new TypeError('Single-qubit op missing serialization meta')
          const { name: nm, params: p } = op.meta
          const q = op.q
          switch (nm) {
            case 'id':  /* skip — no-op */                                  break
            case 'h':   col({ [q]: 'H' });                                  break
            case 'x':   col({ [q]: 'X' });                                  break
            case 'y':   col({ [q]: 'Y' });                                  break
            case 'z':   col({ [q]: 'Z' });                                  break
            case 's':   col({ [q]: 'S' });                                  break
            case 'si':  col({ [q]: 'S†' });                                 break
            case 't':   col({ [q]: 'T' });                                  break
            case 'ti':  col({ [q]: 'T†' });                                 break
            case 'v':   col({ [q]: 'X^½' });                                break
            case 'vi':  col({ [q]: 'X^-½' });                               break
            case 'r2':  col({ [q]: rot('Rz', Math.PI / 2) });               break
            case 'r4':  col({ [q]: rot('Rz', Math.PI / 4) });               break
            case 'r8':  col({ [q]: rot('Rz', Math.PI / 8) });               break
            case 'rx':  col({ [q]: rot('Rx', p![0]!) });                    break
            case 'ry':  col({ [q]: rot('Ry', p![0]!) });                    break
            case 'rz':  col({ [q]: rot('Rz', p![0]!) });                    break
            case 'vz':  col({ [q]: rot('Rz', p![0]!) });                    break
            case 'u1':  col({ [q]: rot('Rz', p![0]!) });                    break  // U1 ≡ Rz (global phase)
            case 'gpi': case 'gpi2':
              throw new TypeError(`Gate '${nm}' has no Quirk representation`)
            default:
              throw new TypeError(`Gate '${nm}' has no Quirk representation`)
          }
          break
        }
        case 'controlled': {
          if (!op.meta) throw new TypeError('Controlled op missing serialization meta')
          const { name: nm, params: p } = op.meta
          const [c, t] = [op.control, op.target]
          let g: QuirkGate
          switch (nm) {
            case 'cy':   g = 'Y';                              break
            case 'cz':   g = 'Z';                              break
            case 'ch':   g = 'H';                              break
            case 'cr2':  g = rot('Rz', Math.PI / 2);           break
            case 'cr4':  g = rot('Rz', Math.PI / 4);           break
            case 'cr8':  g = rot('Rz', Math.PI / 8);           break
            case 'crx':  g = rot('Rx', p![0]!);                break
            case 'cry':  g = rot('Ry', p![0]!);                break
            case 'crz':  g = rot('Rz', p![0]!);                break
            case 'cu1':  g = rot('Rz', p![0]!);                break
            case 'cs':   g = 'S';                              break
            case 'ct':   g = 'T';                              break
            case 'csdg': g = 'S†';                             break
            case 'ctdg': g = 'T†';                             break
            default:
              throw new TypeError(`Gate '${nm}' has no Quirk representation`)
          }
          col({ [c]: '•', [t]: g })
          break
        }
        case 'unitary': throw new TypeError("Gate 'unitary' has no Quirk representation")
      }
    }

    return JSON.stringify({ cols })
  }

  // ── Execution ────────────────────────────────────────────────────────────

  /** Run the circuit and return a probability distribution. */
  run({ shots = 1024, seed, noise, initialState }: RunOptions = {}): Distribution {
    const rng  = makePrng(seed)
    const init = initialState !== undefined ? svFromBitstring(initialState, this.qubits) : undefined

    // Resolve noise: named device profile → NoiseParams, or use as-is
    const noiseParams: NoiseParams | undefined =
      noise == null        ? undefined :
      typeof noise === 'string' ? (() => {
        const p = DEVICE_NOISE[noise]
        if (!p) throw new TypeError(`Unknown device profile '${noise}'. Known: ${Object.keys(DEVICE_NOISE).join(', ')}`)
        return p
      })() : noise

    const cregCounts = new Map<string, number[]>(
      Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array<number>(size).fill(0)])
    )

    // ── Fast path: pure circuit without noise — simulate once, sample N times ──
    if (!noiseParams && !this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      const sv     = simulatePure(this.#ops, this.qubits, init)
      const probs  = probabilities(sv)
      const sorted = Array.from(probs.entries()).toSorted(([a], [b]) => (a < b ? -1 : 1))

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
        Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array<boolean>(size).fill(false)])
      )

      const sv = applyOps(this.#ops, init ?? zero(this.qubits), shotCregs, rng, noiseParams)

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
  runMps({ shots = 1024, seed, maxBond = 64, initialState }: MpsRunOptions = {}): Distribution {
    const rng = makePrng(seed)
    let state = mpsInit(this.qubits)
    if (initialState !== undefined) {
      svFromBitstring(initialState, this.qubits) // validate
      const rev = initialState.split('').reverse() // rev[q] = bit for qubit q
      for (let q = 0; q < this.qubits; q++) {
        if (rev[q] === '1') state = mpsApply1(state, q, G.X)
      }
    }

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'single':     state = mpsApply1(state, op.q, op.gate);                                        break
        case 'cnot':       state = mpsApply2(state, op.control, op.target, CNOT4, maxBond);                break
        case 'swap':       state = mpsApply2(state, op.a, op.b, SWAP4, maxBond);                           break
        case 'two':        state = mpsApply2(state, op.a, op.b, op.gate, maxBond);                         break
        case 'controlled': state = mpsApply2(state, op.control, op.target, controlledGate(op.gate), maxBond); break
        case 'toffoli': throw new TypeError('CCX (Toffoli) not supported in MPS mode; decompose into CX gates')
        case 'cswap':   throw new TypeError('CSWAP (Fredkin) not supported in MPS mode; decompose into CX gates')
        case 'csrswap': throw new TypeError('csrswap not supported in MPS mode; decompose into CX gates')
        case 'unitary': {
          const n = op.qubits.length
          if (n === 1)      state = mpsApply1(state, op.qubits[0]!, op.matrix as Gate2x2)
          else if (n === 2) state = mpsApply2(state, op.qubits[0]!, op.qubits[1]!, op.matrix as Gate4x4, maxBond)
          else throw new TypeError(`unitary gate with ${n} qubits is not supported in MPS mode; use run() instead`)
          break
        }
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
      Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array<number>(size).fill(0)])
    )
    return new Distribution(this.qubits, shots, counts, cregCounts)
  }

  /**
   * Return exact floating-point probabilities from the statevector — no sampling variance.
   *
   * Keys are IonQ bitstrings (q0 rightmost). Only non-negligible amplitudes are included.
   * Throws for circuits containing mid-circuit measure, reset, or conditional ops.
   */
  exactProbs({ initialState }: { initialState?: string } = {}): Readonly<Record<string, number>> {
    if (this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      throw new TypeError('exactProbs() requires a pure circuit — no measure, reset, or if ops')
    }
    const init = initialState !== undefined ? svFromBitstring(initialState, this.qubits) : undefined
    const sv = simulatePure(this.#ops, this.qubits, init)
    const out: Record<string, number> = {}
    for (const [idx, p] of probabilities(sv)) {
      out[idx.toString(2).padStart(this.qubits, '0')] = p
    }
    return Object.freeze(out)
  }

  // ── JSON save / load ─────────────────────────────────────────────────────

  /**
   * Serialize the circuit to a lossless JSON object.
   *
   * Gate matrices are **not** stored — they are reconstructed from their
   * names and parameters on load, so the output is compact and stable
   * across library versions.
   *
   * @example
   * const json = circuit.toJSON()
   * fs.writeFileSync('circuit.json', JSON.stringify(json, null, 2))
   */
  toJSON(): CircuitJSON {
    const cregs: Record<string, number> = {}
    for (const [name, size] of this.#cregs) cregs[name] = size

    const gates: Record<string, { qubits: number; ops: unknown[] }> = {}
    for (const [name, sub] of this.#gates) {
      gates[name] = { qubits: sub.qubits, ops: opsToJSON(sub.#ops) }
    }

    return { ket: 1, qubits: this.qubits, cregs, gates, ops: opsToJSON(this.#ops) }
  }

  /**
   * Deserialize a circuit from a `CircuitJSON` object or a JSON string.
   *
   * The loaded circuit is fully functional — all gate methods, simulation,
   * serialization, and visualization APIs work identically to a hand-built circuit.
   *
   * @throws TypeError for unrecognised op kinds, unknown gate names, or wrong schema version.
   *
   * @example
   * const circuit = Circuit.fromJSON(fs.readFileSync('circuit.json', 'utf8'))
   */
  static fromJSON(json: CircuitJSON | string): Circuit {
    const j: CircuitJSON = typeof json === 'string' ? JSON.parse(json) : json
    if (j.ket !== 1) throw new TypeError(`fromJSON: unsupported schema version ${j.ket}`)

    const cregs = new Map<string, number>(Object.entries(j.cregs))

    const gates = new Map<string, Circuit>()
    for (const [name, def] of Object.entries(j.gates)) {
      gates.set(name, new Circuit(def.qubits, opsFromJSON(def.ops), new Map(), new Map()))
    }

    return new Circuit(j.qubits, opsFromJSON(j.ops), cregs, gates)
  }

  // ── Visualization ────────────────────────────────────────────────────────

  /**
   * Render a minimal ASCII circuit diagram.
   *
   * @example
   * new Circuit(2).h(0).cnot(0, 1).draw()
   * // q0: ─H──●─
   * //          │
   * // q1: ─────⊕─
   */
  draw(): string {
    const n = this.qubits
    if (n === 0) return ''

    const ops = flattenOps(this.#ops).filter(op => op.kind !== 'if')

    // ── Assign ops to columns (greedy, preserve order) ─────────────────────
    // colOf[q] = next free column for qubit q
    const colOf = new Array<number>(n).fill(0)
    type PlacedOp = { op: Op; col: number }
    const placed: PlacedOp[] = []

    for (const op of ops) {
      const qs = opQubits(op)
      if (qs.length === 0) continue
      const minQ = Math.min(...qs), maxQ = Math.max(...qs)
      // Block the full qubit span so vertical wires don't overlap other gates
      let col = 0
      for (let q = minQ; q <= maxQ; q++) col = Math.max(col, colOf[q]!)
      for (let q = minQ; q <= maxQ; q++) colOf[q] = col + 1
      placed.push({ op, col })
    }

    const numCols = Math.max(0, ...colOf)
    if (numCols === 0) {
      // Empty circuit: just wires
      const pw = `q${n - 1}: `.length
      return Array.from({ length: n }, (_, q) => `q${q}: `.padStart(pw) + '─').join('\n')
    }

    // ── Build label grid: label[q][c] ──────────────────────────────────────
    // '' means wire (─), null means "pass-through" (qubit in span but not active)
    const label: (string | null)[][] = Array.from({ length: n }, () =>
      new Array<string | null>(numCols).fill('')
    )
    // hasVert[c][gap] = true if a vertical wire crosses the gap between q=gap and q=gap+1
    const hasVert: boolean[][] = Array.from({ length: numCols }, () =>
      new Array<boolean>(n - 1).fill(false)
    )

    for (const { op, col } of placed) {
      const qs = opQubits(op)
      const minQ = Math.min(...qs), maxQ = Math.max(...qs)
      for (let gap = minQ; gap < maxQ; gap++) hasVert[col]![gap] = true
      for (const q of qs) label[q]![col] = opLabel(op, q)
    }

    // ── Compute per-column body width ──────────────────────────────────────
    const colW = Array.from({ length: numCols }, (_, c) => {
      let w = 1
      for (let q = 0; q < n; q++) {
        const lbl = label[q]![c]!
        if (lbl !== null && lbl !== '') w = Math.max(w, lbl.length)
      }
      return w
    })

    // ── Render ─────────────────────────────────────────────────────────────
    const prefixW = `q${n - 1}: `.length
    const lines: string[] = []

    for (let q = 0; q < n; q++) {
      let line = `q${q}: `.padStart(prefixW)
      for (let c = 0; c < numCols; c++) {
        const lbl = label[q]![c]!
        const w   = colW[c]!
        if (lbl === '' || lbl === null) {
          // wire
          line += '─'.repeat(w + 2)
        } else {
          // gate box: ─[label centered in w with ─ padding]─
          const pad  = w - lbl.length
          const padL = Math.floor(pad / 2)
          const padR = pad - padL
          line += '─' + '─'.repeat(padL) + lbl + '─'.repeat(padR) + '─'
        }
      }
      line += '─'  // trailing wire
      lines.push(line)

      // Spacer row between q and q+1
      if (q < n - 1) {
        let spacer = ' '.repeat(prefixW)
        for (let c = 0; c < numCols; c++) {
          const w = colW[c]!
          if (hasVert[c]![q]) {
            const center = Math.floor((w + 2) / 2)
            spacer += ' '.repeat(center) + '│' + ' '.repeat(w + 2 - center - 1)
          } else {
            spacer += ' '.repeat(w + 2)
          }
        }
        lines.push(spacer)
      }
    }

    return lines.join('\n')
  }

  /**
   * Export the circuit as a self-contained SVG string.
   *
   * The diagram uses the same column layout as `draw()`.  No external fonts or
   * stylesheets are required — the SVG embeds a monospace `font-family` stack.
   *
   * @example
   * fs.writeFileSync('bell.svg', new Circuit(2).h(0).cnot(0, 1).toSVG())
   */
  toSVG(): string {
    const n = this.qubits
    const ROW_H  = 40   // px between qubit rows
    const COL_W  = 52   // base px per column (may be wider for long labels)
    const CHAR_W = 7.5  // rough px per monospace character at font-size 13
    const BOX_H  = 22   // gate box height
    const R      = 5    // box corner radius
    const ML     = 52   // left margin (qubit labels)
    const MR     = 24   // right margin
    const MT     = 24   // top margin
    const MB     = 16   // bottom margin

    const ops = flattenOps(this.#ops).filter(op => op.kind !== 'if')

    // Column assignment (same algorithm as draw())
    const colOf = new Array<number>(n).fill(0)
    type PlacedOp = { op: Op; col: number }
    const placed: PlacedOp[] = []
    for (const op of ops) {
      const qs = opQubits(op)
      if (qs.length === 0) continue
      const minQ = Math.min(...qs), maxQ = Math.max(...qs)
      let col = 0
      for (let q = minQ; q <= maxQ; q++) col = Math.max(col, colOf[q]!)
      for (let q = minQ; q <= maxQ; q++) colOf[q] = col + 1
      placed.push({ op, col })
    }
    const numCols = Math.max(0, ...colOf)

    // Per-column body widths (in px), based on label length
    const colPx = Array.from({ length: numCols }, (_, c) => {
      let maxLabel = 1
      for (const { op, col } of placed) {
        if (col !== c) continue
        for (const q of opQubits(op)) {
          const lbl = opLabel(op, q)
          if (lbl !== '●' && lbl !== '⊕' && lbl !== '╳') maxLabel = Math.max(maxLabel, lbl.length)
        }
      }
      return Math.max(COL_W, Math.ceil(maxLabel * CHAR_W) + 20)
    })

    // Column center x positions
    const colX: number[] = []
    let cx = ML
    for (let c = 0; c < numCols; c++) {
      colX.push(cx + colPx[c]! / 2)
      cx += colPx[c]!
    }
    const totalW = ML + (numCols > 0 ? cx - ML : 0) + MR
    const totalH = MT + (n - 1) * ROW_H + MB + ROW_H

    const qy = (q: number) => MT + q * ROW_H + ROW_H / 2

    const svgParts: string[] = []

    // Background
    svgParts.push(`<rect width="${totalW}" height="${totalH}" fill="#ffffff"/>`)

    // Horizontal wires
    for (let q = 0; q < n; q++) {
      const y = qy(q)
      svgParts.push(`<line x1="${ML - 8}" y1="${y}" x2="${totalW - MR}" y2="${y}" stroke="#334155" stroke-width="1.5"/>`)
    }

    // Qubit labels
    for (let q = 0; q < n; q++) {
      svgParts.push(`<text x="${ML - 12}" y="${qy(q) + 4}" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" fill="#334155">q${q}:</text>`)
    }

    // Gates
    for (const { op, col } of placed) {
      const x = colX[col]!
      const qs = opQubits(op)
      const minQ = Math.min(...qs), maxQ = Math.max(...qs)

      // Vertical connector line between min and max qubit
      if (minQ !== maxQ) {
        svgParts.push(`<line x1="${x}" y1="${qy(minQ)}" x2="${x}" y2="${qy(maxQ)}" stroke="#334155" stroke-width="1.5"/>`)
      }

      for (const q of qs) {
        const lbl = opLabel(op, q)
        const y   = qy(q)

        if (lbl === '●') {
          // Control dot
          svgParts.push(`<circle cx="${x}" cy="${y}" r="5" fill="#334155"/>`)
        } else if (lbl === '⊕') {
          // CNOT target: circle with cross
          svgParts.push(`<circle cx="${x}" cy="${y}" r="10" fill="none" stroke="#334155" stroke-width="1.5"/>`)
          svgParts.push(`<line x1="${x}" y1="${y - 10}" x2="${x}" y2="${y + 10}" stroke="#334155" stroke-width="1.5"/>`)
          svgParts.push(`<line x1="${x - 10}" y1="${y}" x2="${x + 10}" y2="${y}" stroke="#334155" stroke-width="1.5"/>`)
        } else if (lbl === '╳') {
          // SWAP target: X mark
          const d = 7
          svgParts.push(`<line x1="${x - d}" y1="${y - d}" x2="${x + d}" y2="${y + d}" stroke="#334155" stroke-width="2"/>`)
          svgParts.push(`<line x1="${x + d}" y1="${y - d}" x2="${x - d}" y2="${y + d}" stroke="#334155" stroke-width="2"/>`)
        } else {
          // Gate box
          const labelPx = Math.max(20, Math.ceil(lbl.length * CHAR_W) + 14)
          const bx = x - labelPx / 2, by = y - BOX_H / 2
          svgParts.push(`<rect x="${bx}" y="${by}" width="${labelPx}" height="${BOX_H}" rx="${R}" fill="#f8fafc" stroke="#334155" stroke-width="1.5"/>`)
          svgParts.push(`<text x="${x}" y="${y + 4}" text-anchor="middle" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" fill="#1e293b">${lbl}</text>`)
        }
      }
    }

    const w = totalW, h = totalH
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">\n${svgParts.join('\n')}\n</svg>`
  }

  /**
   * Return the Bloch sphere angles (θ, φ) for qubit `q`.
   *
   * Computes the reduced single-qubit density matrix by tracing out all other
   * qubits, then extracts the Bloch vector (rx, ry, rz) and converts to
   * standard spherical coordinates:
   *   - θ ∈ [0, π]  — polar angle from |0⟩ (north pole)
   *   - φ ∈ (-π, π] — azimuthal angle in the equatorial plane
   *
   * For pure product states the result is exact.  For entangled qubits the
   * Bloch vector has |r| < 1 (mixed state) and the angles are still well
   * defined as long as the qubit is not maximally mixed (|r| > 0).
   *
   * @throws TypeError when called on circuits with measure/reset/if ops.
   */
  blochAngles(q: number, { initialState }: { initialState?: string } = {}): { theta: number; phi: number } {
    const sv  = this.statevector(initialState !== undefined ? { initialState } : {})
    const mask = 1n << BigInt(q)

    // Reduced density matrix elements — single pass
    let rho00 = 0, rho11 = 0, rho01re = 0, rho01im = 0

    for (const [idx, amp] of sv) {
      const p = amp.re * amp.re + amp.im * amp.im
      if ((idx & mask) !== 0n) { rho11 += p; continue }
      rho00 += p
      const amp1 = sv.get(idx | mask)
      if (amp1) {
        // ρ₀₁ = Σ ψ(idx) · ψ(idx|mask)*
        rho01re += amp.re * amp1.re + amp.im * amp1.im
        rho01im += amp.im * amp1.re - amp.re * amp1.im
      }
    }

    // Bloch vector: rz = ρ₀₀ - ρ₁₁, rx = 2·Re(ρ₀₁), ry = -2·Im(ρ₀₁)
    const rz = rho00 - rho11
    const rx = 2 * rho01re
    const ry = -2 * rho01im

    const theta = Math.acos(Math.max(-1, Math.min(1, rz)))
    const phi   = Math.atan2(ry, rx)
    return { theta, phi }
  }

  // ── Circuit depth ─────────────────────────────────────────────────────────

  /**
   * Return the critical path length — the minimum number of time steps needed
   * to execute this circuit when gates on independent qubits run in parallel.
   *
   * Barriers are scheduling hints only and do not increment depth.
   * IfOps recurse into their inner ops.
   */
  depth(): number {
    const stepOf = new Array<number>(this.qubits).fill(0)

    function processOps(ops: readonly Op[]): void {
      for (const op of flattenOps(ops)) {
        if (op.kind === 'barrier') continue  // no depth increment for barriers

        if (op.kind === 'if') {
          processOps(op.ops)
          continue
        }

        let qubits: number[]
        switch (op.kind) {
          case 'single':  qubits = [op.q];                              break
          case 'cnot':    qubits = [op.control, op.target];             break
          case 'swap':    qubits = [op.a, op.b];                        break
          case 'two':     qubits = [op.a, op.b];                        break
          case 'controlled': qubits = [op.control, op.target];          break
          case 'toffoli': qubits = [op.c1, op.c2, op.target];           break
          case 'cswap':   qubits = [op.control, op.a, op.b];            break
          case 'csrswap': qubits = [op.control, op.a, op.b];            break
          case 'measure': qubits = [op.q];                               break
          case 'reset':   qubits = [op.q];                               break
          case 'subcircuit': qubits = [...op.qubits];                    break
          case 'unitary':    qubits = [...op.qubits];                    break
          default:           qubits = [];                                break
        }

        if (qubits.length === 0) continue

        // Find the earliest step this op can start (= max step already used among its qubits)
        let maxStep = 0
        for (const q of qubits) maxStep = Math.max(maxStep, stepOf[q] ?? 0)

        // Place the op at maxStep (occupies one step)
        const nextStep = maxStep + 1
        for (const q of qubits) stepOf[q] = nextStep
      }
    }

    processOps(this.#ops)
    return Math.max(0, ...stepOf)
  }

  // ── Bloch sphere visualization ────────────────────────────────────────────

  /**
   * Return a self-contained SVG string showing the Bloch sphere for qubit `q`.
   *
   * Uses `blochAngles(q)` to get the state angles, then renders a 300×300 SVG
   * with the sphere outline, equatorial ellipse, axes, and a blue arrow for the
   * state vector using cavalier projection.
   *
   * @throws TypeError if the circuit contains measure/reset/if ops (see blochAngles).
   */
  blochSphere(q: number): string {
    const { theta, phi } = this.blochAngles(q)

    const bx = Math.sin(theta) * Math.cos(phi)
    const by = Math.sin(theta) * Math.sin(phi)
    const bz = Math.cos(theta)

    const cx = 150, cy = 150, R = 110

    // Cavalier projection: x-axis right, y-axis lower-right, z-axis up
    const px = cx + R * (bx - by * 0.4)
    const py = cy - R * (bz + by * 0.1)

    const parts: string[] = []

    // Background
    parts.push(`<rect width="300" height="300" fill="white"/>`)

    // Sphere outline
    parts.push(`<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="#94a3b8" stroke-width="1.5"/>`)

    // Equatorial ellipse (dashed)
    parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${R}" ry="${Math.round(R * 0.35)}" fill="none" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,3"/>`)

    // Z-axis: dashed above equator, solid below
    // Above equator: from cy to cy-R
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy - R}" stroke="#475569" stroke-width="1.2" stroke-dasharray="5,3"/>`)
    // Below equator: solid
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy + R}" stroke="#475569" stroke-width="1.2"/>`)

    // Z-axis labels
    parts.push(`<text x="${cx}" y="${cy - R - 8}" text-anchor="middle" font-family="serif" font-size="14" fill="#1e293b">|0⟩</text>`)
    parts.push(`<text x="${cx}" y="${cy + R + 18}" text-anchor="middle" font-family="serif" font-size="14" fill="#1e293b">|1⟩</text>`)

    // X-axis (right-horizontal)
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx + R}" y2="${cy}" stroke="#475569" stroke-width="1.2"/>`)
    parts.push(`<text x="${cx + R + 8}" y="${cy + 4}" text-anchor="start" font-family="serif" font-size="13" fill="#1e293b">|+⟩</text>`)

    // Y-axis (diagonal lower-right — cavalier projection of +Y)
    const yTipX = cx + Math.round(R * 0.4)
    const yTipY = cy + Math.round(R * 0.1) + Math.round(R * 0.35)
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${yTipX}" y2="${yTipY}" stroke="#475569" stroke-width="1.2"/>`)
    parts.push(`<text x="${yTipX + 6}" y="${yTipY + 4}" text-anchor="start" font-family="serif" font-size="13" fill="#1e293b">|i⟩</text>`)

    // State vector arrow (blue line)
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${px.toFixed(1)}" y2="${py.toFixed(1)}" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round"/>`)

    // State vector dot at tip
    parts.push(`<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5" fill="#3b82f6"/>`)

    return `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">\n${parts.join('\n')}\n</svg>`
  }

  // ── Clifford stabilizer simulation ───────────────────────────────────────

  /**
   * Simulate this circuit using the CHP (Aaronson-Gottesman 2004) Clifford
   * stabilizer algorithm — exponentially faster than the statevector for
   * Clifford circuits, with exact probabilities.
   *
   * Only Clifford gates are supported:
   *   h, x, y, z, s, si/sdg, measure, reset, barrier, cnot, swap,
   *   and controlled gates cx/cy/cz.
   *
   * Non-Clifford gates (T, Rx(θ≠kπ/2), etc.) cause a TypeError.
   *
   * @param opts.shots  Number of measurement shots (default 1024).
   * @param opts.seed   Optional PRNG seed for reproducibility.
   * @param opts.noise  Device name (`'aria-1'` / `'forte-1'` / `'harmony'`) or
   *                    `{ p1?, p2?, pMeas? }` depolarizing + readout error rates.
   */
  runClifford({ shots = 1024, seed, noise }: { shots?: number; seed?: number; noise?: string | NoiseParams } = {}): Distribution {
    // ── Validate: check all ops are Clifford ─────────────────────────────
    const CLIFFORD_SINGLE = new Set(['h', 'x', 'y', 'z', 's', 'si', 'sdg'])
    const CLIFFORD_CTRL   = new Set(['cx', 'cy', 'cz'])

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'barrier': case 'measure': case 'reset': break
        case 'cnot': case 'swap': break
        case 'single': {
          const name = op.meta?.name ?? '?'
          if (!CLIFFORD_SINGLE.has(name))
            throw new TypeError(`runClifford: gate '${name}' is not a Clifford gate`)
          break
        }
        case 'controlled': {
          const name = op.meta?.name ?? '?'
          if (!CLIFFORD_CTRL.has(name))
            throw new TypeError(`runClifford: gate '${name}' is not a Clifford gate`)
          break
        }
        default: {
          const name = (op as { meta?: GateMeta }).meta?.name ?? op.kind
          throw new TypeError(`runClifford: gate '${name}' is not a Clifford gate`)
        }
      }
    }

    // Resolve noise profile
    const noiseParams: NoiseParams | undefined =
      noise == null         ? undefined :
      typeof noise === 'string' ? (() => {
        const p = DEVICE_NOISE[noise]
        if (!p) throw new TypeError(`Unknown device profile '${noise}'. Known: ${Object.keys(DEVICE_NOISE).join(', ')}`)
        return p
      })() : noise
    const p1    = noiseParams?.p1    ?? 0
    const p2    = noiseParams?.p2    ?? 0
    const pMeas = noiseParams?.pMeas ?? 0

    const rng = makePrng(seed)
    const counts = new Map<bigint, number>()

    // Precompute which qubits are explicitly measured in this circuit
    const flatOps = flattenOps(this.#ops)
    const measuredQubits = new Set<number>()
    for (const op of flatOps) {
      if (op.kind === 'measure') measuredQubits.add(op.q)
    }

    // Clifford depolarizing: random Pauli X/Y/Z with total prob p
    const cDep1 = (q: number, p: number): void => {
      const r = rng()
      if (r >= p) return
      const s = r / p
      if (s < 1/3) sim.x(q)
      else if (s < 2/3) sim.y(q)
      else sim.z(q)
    }

    // Clifford two-qubit depolarizing: random non-identity {I,X,Y,Z}⊗{I,X,Y,Z} with total prob p
    const applyPauli = (q: number, p: number): void => {
      if (p === 1) sim.x(q); else if (p === 2) sim.y(q); else if (p === 3) sim.z(q)
    }
    const cDep2 = (a: number, b: number, p: number): void => {
      const r = rng()
      if (r >= p) return
      const [ea, eb] = TWO_PAULI_IDX[Math.min(Math.floor(r / p * 15), 14)]!
      applyPauli(a, ea); applyPauli(b, eb)
    }

    // ── Per-shot simulation ───────────────────────────────────────────────
    let sim!: CliffordSim
    for (let shot = 0; shot < shots; shot++) {
      sim = new CliffordSim(this.qubits)

      // Track measurement outcomes for this shot (to form the output bitstring)
      const measured = new Array<number>(this.qubits).fill(0)

      const applyCliffords = (ops: readonly Op[]): void => {
        for (const op of ops) {
          switch (op.kind) {
            case 'barrier': break
            case 'single': {
              const name = op.meta?.name ?? ''
              switch (name) {
                case 'h':  sim.h(op.q);  break
                case 'x':  sim.x(op.q);  break
                case 'y':  sim.y(op.q);  break
                case 'z':  sim.z(op.q);  break
                case 's':  sim.s(op.q);  break
                case 'si': case 'sdg': sim.si(op.q); break
              }
              if (p1) cDep1(op.q, p1)
              break
            }
            case 'cnot':
              sim.cnot(op.control, op.target)
              if (p2) cDep2(op.control, op.target, p2)
              break
            case 'swap':
              sim.swap(op.a, op.b)
              if (p2) cDep2(op.a, op.b, p2)
              break
            case 'controlled': {
              const name = op.meta?.name ?? ''
              switch (name) {
                case 'cx': sim.cnot(op.control, op.target); break
                case 'cy': sim.cy(op.control, op.target);   break
                case 'cz': sim.cz(op.control, op.target);   break
              }
              if (p2) cDep2(op.control, op.target, p2)
              break
            }
            case 'measure': {
              const raw = sim.measure(op.q, rng())
              const outcome = pMeas && rng() < pMeas ? (raw ^ 1) : raw
              measured[op.q] = outcome
              break
            }
            case 'reset': {
              // Reset: measure then conditionally flip back to |0⟩
              const outcome = sim.measure(op.q, rng())
              if (outcome === 1) sim.x(op.q)
              measured[op.q] = 0
              break
            }
            case 'if': {
              // Recurse into inner ops (creg evaluation not tracked in Clifford mode)
              applyCliffords(op.ops)
              break
            }
          }
        }
      }

      applyCliffords(flatOps)

      // Final readout: for qubits not explicitly measured, measure now
      let idx = 0n
      for (let q = 0; q < this.qubits; q++) {
        const bit = measuredQubits.has(q) ? measured[q]! : sim.measure(q, rng())
        if (bit) idx |= (1n << BigInt(q))
      }

      counts.set(idx, (counts.get(idx) ?? 0) + 1)
    }

    const cregCounts = new Map<string, number[]>(
      Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array<number>(size).fill(0)])
    )
    return new Distribution(this.qubits, shots, counts, cregCounts)
  }

  // ── Hardware compilation ──────────────────────────────────────────────────

  /**
   * Transpile this circuit to the native gate set of the specified IonQ device.
   *
   * Supported devices: 'aria-1', 'forte-1', 'harmony' — all use {GPI, GPI2, MS, VZ}.
   *
   * Single-qubit decompositions (all exact up to global phase):
   *   h  → vz(π/2) · gpi2(0) · vz(π/2)
   *   x  → gpi(0)
   *   y  → gpi(π/2)
   *   z  → vz(π)
   *   s  → vz(π/2)
   *   si/sdg → vz(-π/2)
   *   t  → vz(π/4)
   *   ti/tdg → vz(-π/4)
   *   rz/vz → vz (pass through)
   *   gpi/gpi2 → pass through
   *
   * Two-qubit:
   *   cnot(a,b) → gpi2(π/2,a) · ms(0,0,a,b) · gpi2(3π/2,a) · vz(-π/2,a) · vz(-π/2,b)
   *   swap(a,b) → three CNOT decompositions (via above)
   *   ms → pass through
   *
   * Barriers pass through. All other gates throw TypeError.
   *
   * @param device  Target device name ('aria-1', 'forte-1', 'harmony').
   * @returns A new Circuit containing only native-gate ops.
   * @throws TypeError for unsupported gates or unknown device names.
   */
  compile(device: string): Circuit {
    const KNOWN_DEVICES = new Set(Object.keys(IONQ_DEVICES))
    if (!KNOWN_DEVICES.has(device)) {
      throw new TypeError(`compile: unknown device '${device}'. Known IonQ devices: ${[...KNOWN_DEVICES].join(', ')}`)
    }

    const PI  = Math.PI
    let result = new Circuit(this.qubits, [], this.#cregs, this.#gates)

    // Helper: compile a CNOT(a→b) → native GPI2/MS/VZ sequence.
    // Derived from MS(0,0) = XX(π/4) and the standard CX decomposition via Mølmer–Sørensen:
    // CNOT(ctrl=a, tgt=b) = GPI2(π/2, a) · GPI2(π, b) · MS(0,0, a,b) · GPI2(-π/2, a) · VZ(-π/2, a)
    const compileCnot = (c: Circuit, a: number, b: number): Circuit => {
      c = c.gpi2(PI / 2,  a)   // GPI2(π/2) on control
      c = c.gpi2(PI,      b)   // GPI2(π)   on target
      c = c.ms(0, 0,      a, b) // MS entangling gate
      c = c.gpi2(-PI / 2, a)   // GPI2(-π/2) on control
      c = c.vz(-PI / 2,   a)   // VZ(-π/2) on control
      return c
    }

    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case 'barrier':
          result = result.barrier(...op.qubits)
          break

        case 'single': {
          const name = op.meta?.name ?? '?'
          const q    = op.q
          switch (name) {
            case 'gpi':  result = result.gpi(op.meta!.params![0]!, q);  break
            case 'gpi2': result = result.gpi2(op.meta!.params![0]!, q); break
            case 'vz':   result = result.vz(op.meta!.params![0]!, q);   break
            case 'rz':   result = result.vz(op.meta!.params![0]!, q);   break
            case 'h':
              result = result.vz(PI / 2, q).gpi2(0, q).vz(PI / 2, q)
              break
            case 'x':  result = result.gpi(0, q);        break
            case 'y':  result = result.gpi(PI / 2, q);   break
            case 'z':  result = result.vz(PI, q);         break
            case 's':  result = result.vz(PI / 2, q);    break
            case 'si': case 'sdg':
              result = result.vz(-PI / 2, q); break
            case 't':  result = result.vz(PI / 4, q);    break
            case 'ti': case 'tdg':
              result = result.vz(-PI / 4, q); break
            case 'id': break  // identity — no-op
            default:
              throw new TypeError(`compile: gate '${name}' cannot be compiled to ${device} native gates`)
          }
          break
        }

        case 'cnot':
          result = compileCnot(result, op.control, op.target)
          break

        case 'swap': {
          // SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
          result = compileCnot(result, op.a, op.b)
          result = compileCnot(result, op.b, op.a)
          result = compileCnot(result, op.a, op.b)
          break
        }

        case 'two': {
          const name = op.meta?.name ?? '?'
          if (name === 'ms') {
            // Pass through MS gate natively
            result = result.ms(op.meta!.params![0]!, op.meta!.params![1]!, op.a, op.b)
          } else {
            throw new TypeError(`compile: gate '${name}' cannot be compiled to ${device} native gates`)
          }
          break
        }

        case 'measure':
          result = result.measure(op.q, op.creg, op.bit)
          break

        case 'reset':
          // Reset is not a gate — pass through as-is (preserves collapse semantics)
          result = result.reset(op.q)
          break

        default: {
          const name = (op as { meta?: GateMeta }).meta?.name ?? op.kind
          throw new TypeError(`compile: gate '${name}' cannot be compiled to ${device} native gates`)
        }
      }
    }

    return result
  }

  /**
   * Simulate the circuit as an exact density matrix and return it.
   *
   * Unlike `run()` (which samples) and `statevector()` (which is pure-state only),
   * `dm()` computes the full ρ = |ψ⟩⟨ψ| evolution and applies optional per-gate
   * depolarizing noise channels exactly — no sampling, no variance.
   *
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   * Complexity: O(4ⁿ) — practical up to ~12 qubits.
   *
   * @param options.noise  Device name (`'aria-1'` / `'forte-1'` / `'harmony'`) or
   *                       `{ p1?, p2? }` noise parameters.
   */
  dm(options?: { noise?: DmNoiseParams | string }): DensityMatrix {
    if (this.#ops.some(op => op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if')) {
      throw new TypeError('dm() requires a pure circuit — remove measure/reset/if ops')
    }

    const { noise } = options ?? {}
    const noiseParams: DmNoiseParams | undefined =
      noise == null           ? undefined :
      typeof noise === 'string' ? (() => {
        const p = DM_DEVICE_NOISE[noise]
        if (!p) throw new TypeError(`Unknown device profile '${noise}'. Known: ${Object.keys(DM_DEVICE_NOISE).join(', ')}`)
        return p
      })() : noise

    // DmOp is a structural subset of Op; safety guaranteed by the classical-op guard above.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return runDM(flattenOps(this.#ops) as any, this.qubits, noiseParams)
  }
}
