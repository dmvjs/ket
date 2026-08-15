/**
 * Compatibility layer for the `quantum-circuit` npm package.
 *
 * Implements that package's mutable, column-indexed `QuantumCircuit` API on top
 * of ket's engine, so existing code can be pointed at ket by changing one
 * import:
 *
 * ```javascript
 * // const QuantumCircuit = require('quantum-circuit')
 * import { QuantumCircuit } from '@kirkelliott/ket/compat'
 *
 * const circuit = new QuantumCircuit(2)
 * circuit.addGate('h', 0, 0)
 * circuit.addGate('cx', 1, [0, 1])
 * circuit.run()
 * circuit.print()
 * ```
 *
 * All 55 of that package's gates are supported — ket implements every one under
 * the same name. `toKet()` returns the equivalent immutable `Circuit`, which is
 * the way out of this API and into the rest of the library (five backends, noise,
 * hardware submission).
 *
 * Bit ordering is converted at the boundary, so bitstrings printed or returned
 * here read in the original package's order (highest wire leftmost), not ket's
 * native order (wire 0 leftmost). The one known cosmetic difference is
 * `exportQASM()`, which uses ket's whitespace.
 */

import { Circuit, parseAngle } from './circuit.js'
import type { Complex } from './complex.js'

// ─── Gate tables ──────────────────────────────────────────────────────────────

/** Wires each gate spans. `barrier` spans however many it is given. */
const GATE_WIRES: Readonly<Record<string, number>> = {
  id: 1, x: 1, y: 1, z: 1, h: 1, srn: 1, srndg: 1, r2: 1, r4: 1, r8: 1,
  rx: 1, ry: 1, rz: 1, u1: 1, u2: 1, u3: 1, s: 1, t: 1, sdg: 1, tdg: 1,
  gpi: 1, gpi2: 1, vz: 1, reset: 1, measure: 1,
  cx: 2, cy: 2, cz: 2, ch: 2, csrn: 2, swap: 2, srswap: 2, iswap: 2,
  xx: 2, yy: 2, zz: 2, xy: 2, ms: 2, cr2: 2, cr4: 2, cr8: 2,
  crx: 2, cry: 2, crz: 2, cu1: 2, cu2: 2, cu3: 2, cs: 2, ct: 2, csdg: 2, ctdg: 2,
  ccx: 3, cswap: 3, csrswap: 3,
}

/** Parameter names per gate, in the order ket's method takes them. */
const GATE_PARAMS: Readonly<Record<string, readonly string[]>> = {
  rx: ['theta'], ry: ['theta'], rz: ['phi'], u1: ['lambda'], u2: ['phi', 'lambda'],
  u3: ['theta', 'phi', 'lambda'], gpi: ['phi'], gpi2: ['phi'], vz: ['theta'],
  xx: ['theta'], yy: ['theta'], zz: ['theta'], xy: ['phi'], ms: ['phi0', 'phi1'],
  crx: ['theta'], cry: ['theta'], crz: ['phi'], cu1: ['lambda'],
  cu2: ['phi', 'lambda'], cu3: ['theta', 'phi', 'lambda'],
}

// ─── Public shapes ────────────────────────────────────────────────────────────

/** Run a gate only when a classical register holds `value`. */
export interface CompatCondition { creg: string; value: number }

/** The `options` object accepted as `addGate`'s fourth argument. */
export interface CompatGateOptions {
  params?: Record<string, number | string>
  condition?: CompatCondition
  creg?: { name: string; bit: number }
}

/** One gate as stored in a saved circuit's `gates[wire][column]` grid. */
export interface CompatSavedGate {
  id: string
  name: string
  connector: number
  options: CompatGateOptions
}

/** The object produced by `save()` and consumed by `load()` / `registerGate()`. */
export interface CompatSavedCircuit {
  numQubits: number
  params: unknown[]
  options: Record<string, unknown>
  gates: (CompatSavedGate | null)[][]
  customGates: Record<string, CompatSavedCircuit>
  cregs: Record<string, number>
}

/** Options accepted by `run`. `seed` is a ket extension for reproducible shots. */
export interface CompatRunOptions { seed?: number }

/** A gate placed on the grid. */
interface PlacedGate {
  id: string
  name: string
  column: number
  wires: number[]
  options: CompatGateOptions
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Convert between ket's bitstring order (character *i* is qubit *i*) and
 * quantum-circuit's (character *i* is qubit *n−1−i*). The conversion is its own
 * inverse, and is applied at every boundary where a bitstring is read or written.
 */
function flipBitOrder(bits: string): string {
  return bits.split('').reverse().join('')
}

/** Round to 14 decimal places, matching the original package's display rounding. */
function round14(x: number): number {
  const r = Math.round(x * 1e14) / 1e14
  return Object.is(r, -0) ? 0 : r
}

/** Format one amplitude as ` 0.70710678+0.00000000i`. */
function formatComplex(z: Complex): string {
  const re = round14(z.re), im = round14(z.im)
  return `${re < 0 ? '-' : ' '}${Math.abs(re).toFixed(8)}${im < 0 ? '-' : '+'}${Math.abs(im).toFixed(8)}i`
}

/** Call a ket gate method by name: parameters first, then wires. */
function applyNamed(c: Circuit, name: string, args: number[]): Circuit {
  const fn = (c as unknown as Record<string, unknown>)[name]
  if (typeof fn !== 'function') throw new TypeError(`compat: unknown gate '${name}'`)
  return (fn as (...a: number[]) => Circuit).apply(c, args)
}

let gateSeq = 0
/** Stable, collision-free gate id — the original package uses a random string. */
function nextGateId(): string { return `g${(++gateSeq).toString(36)}` }

/**
 * A fresh seed for an unseeded run, drawn from the platform CSPRNG.
 *
 * ket seeds from `Date.now()` when no seed is given, which repeats inside a tight
 * loop — the `quantum-random-number-generator` pattern of running an unseeded
 * circuit in a `for` loop would return the same value every iteration. Seeding
 * each run separately fixes that.
 *
 * Note this seeds a deterministic sampler: it makes unseeded runs unpredictable
 * and non-repeating, but the resulting bits are no stronger than the PRNG behind
 * them. For cryptographic randomness use `crypto.getRandomValues` directly.
 */
function randomSeed(): number {
  const c = globalThis.crypto
  if (c?.getRandomValues) return c.getRandomValues(new Uint32Array(1))[0]!
  return (Math.random() * 0x1_0000_0000) >>> 0     // last resort: no WebCrypto in this host
}

// ─── QuantumCircuit ───────────────────────────────────────────────────────────

/**
 * Drop-in replacement for the `quantum-circuit` package's default export.
 *
 * Mutable and column-indexed, like the original: gates are placed at an explicit
 * column (or appended with a negative one), the circuit grows to fit any wire you
 * name, and `run()` stores state on the instance for later inspection.
 */
export class QuantumCircuit {
  numQubits: number

  #gates: PlacedGate[] = []
  #cregSizes  = new Map<string, number>()
  #cregBits   = new Map<string, number[]>()
  #customGates = new Map<string, CompatSavedCircuit>()
  #state: Map<bigint, Complex> | null = null
  #collapsed: number[] | null = null

  constructor(numQubits = 1) { this.numQubits = numQubits }

  // ── Construction ───────────────────────────────────────────────────────────

  /** Reset the circuit to `numQubits` wires and no gates. */
  init(numQubits = 1): void {
    this.numQubits = numQubits
    this.clearGates()
    this.#cregSizes.clear()
    this.#cregBits.clear()
    this.#customGates.clear()
  }

  /** Remove every gate, keeping registers and custom gate definitions. */
  clearGates(): void {
    this.#gates = []
    this.resetState()
  }

  /** Discard the state produced by the last `run()`. */
  resetState(): void {
    this.#state = null
    this.#collapsed = null
    this.#cregBits.clear()
  }

  /**
   * Place a gate at `column`. A negative column appends to the end. `wires` is a
   * single wire index or an array of them; the circuit grows to fit.
   */
  addGate(gateName: string, column: number, wires: number | number[], options: CompatGateOptions = {}): string {
    const ws = Array.isArray(wires) ? [...wires] : [wires]
    if (!ws.length) throw new TypeError(`compat: gate '${gateName}' needs at least one wire`)
    for (const w of ws) {
      if (!Number.isInteger(w) || w < 0) throw new TypeError(`compat: invalid wire ${w} for gate '${gateName}'`)
      this.numQubits = Math.max(this.numQubits, w + 1)
    }

    const known = this.#customGates.get(gateName)
    const arity = known ? known.numQubits : GATE_WIRES[gateName]
    if (arity === undefined && gateName !== 'barrier')
      throw new TypeError(`compat: unknown gate '${gateName}'`)
    if (arity !== undefined && gateName !== 'barrier' && ws.length !== arity)
      throw new TypeError(`compat: gate '${gateName}' spans ${arity} wire(s), got ${ws.length}`)

    const col = column < 0 ? this.numCols() : column
    const gate: PlacedGate = { id: nextGateId(), name: gateName, column: col, wires: ws, options }
    this.#gates.push(gate)
    this.resetState()
    return gate.id
  }

  /** Add a gate at the end of the circuit. */
  appendGate(gateName: string, wires: number | number[], options: CompatGateOptions = {}): string {
    return this.addGate(gateName, -1, wires, options)
  }

  /** Add a measurement gate storing `wire` into `creg[cbit]`, creating the register if needed. */
  addMeasure(wire: number, creg: string, cbit: number): string {
    this.createCreg(creg, Math.max(this.#cregSizes.get(creg) ?? 0, cbit + 1))
    return this.addGate('measure', -1, wire, { creg: { name: creg, bit: cbit } })
  }

  /** Remove the gate with this id. Unknown ids are ignored, as in the original. */
  removeGate(gateId: string): void {
    const before = this.#gates.length
    this.#gates = this.#gates.filter(g => g.id !== gateId)
    if (this.#gates.length !== before) this.resetState()
  }

  /** Append every gate of another circuit after this one's last column. */
  appendCircuit(circuit: QuantumCircuit): void {
    const offset = this.numCols()
    this.numQubits = Math.max(this.numQubits, circuit.numQubits)
    for (const [name, size] of circuit.#cregSizes) this.createCreg(name, size)
    for (const [name, def] of circuit.#customGates) this.#customGates.set(name, def)
    for (const g of circuit.#ordered())
      this.#gates.push({ ...g, id: nextGateId(), column: g.column + offset, wires: [...g.wires] })
    this.resetState()
  }

  // ── Shape ──────────────────────────────────────────────────────────────────

  /** One past the highest occupied column, counting gaps. */
  numCols(): number {
    return this.#gates.reduce((m, g) => Math.max(m, g.column + 1), 0)
  }

  /** Number of amplitudes in the state: 2^numQubits. */
  numAmplitudes(): number { return 2 ** this.numQubits }

  /** Number of *occupied* columns, ignoring gaps. */
  getDepth(): number { return new Set(this.#gates.map(g => g.column)).size }

  /** Names of the gates used, in first-use order. */
  usedGates(): string[] { return [...new Set(this.#ordered().map(g => g.name))] }

  /** Gates sorted by column, ties broken by insertion order. */
  #ordered(): PlacedGate[] {
    return this.#gates.map((g, i) => ({ g, i }))
      .sort((a, b) => a.g.column - b.g.column || a.i - b.i)
      .map(({ g }) => g)
  }

  // ── Classical registers ────────────────────────────────────────────────────

  /** Create a classical register, or grow an existing one to `len` bits. */
  createCreg(creg: string, len: number): void {
    this.#cregSizes.set(creg, Math.max(len, this.#cregSizes.get(creg) ?? 0))
  }

  /** All classical registers as `{ name: integerValue }`. */
  getCregs(): Record<string, number> {
    const out: Record<string, number> = {}
    for (const name of this.#cregSizes.keys()) out[name] = this.getCregValue(name)
    return out
  }

  /** A classical register's value as an integer, bit 0 least significant. */
  getCregValue(creg: string): number {
    const bits = this.#bitsOf(creg)
    return bits.reduce((acc, bit, i) => acc + (bit ? 2 ** i : 0), 0)
  }

  /** One bit of a classical register, as 0 or 1. */
  getCregBit(creg: string, cbit: number): number {
    const bits = this.#bitsOf(creg)
    if (cbit >= bits.length) throw new TypeError(`compat: bit ${cbit} out of range for register '${creg}'`)
    return bits[cbit] ?? 0
  }

  /** Set one bit of a classical register. */
  setCregBit(creg: string, cbit: number, value: number): void {
    this.createCreg(creg, cbit + 1)
    const bits = [...this.#bitsOf(creg)]
    while (bits.length <= cbit) bits.push(0)
    bits[cbit] = value ? 1 : 0
    this.#cregBits.set(creg, bits)
  }

  /** Every register as a tab-delimited table: `reg`, `bin`, `dec`. */
  cregsAsString(): string {
    let s = 'reg\tbin\tdec\n'
    for (const [name, size] of this.#cregSizes) {
      const value = this.getCregValue(name)
      s += `${name}\t${value.toString(2).padStart(size, '0')}\t${value}\n`
    }
    return s
  }

  #bitsOf(creg: string): number[] {
    const size = this.#cregSizes.get(creg)
    if (size === undefined) throw new TypeError(`compat: unknown classical register '${creg}'`)
    const bits = this.#cregBits.get(creg) ?? []
    return Array.from({ length: size }, (_, i) => bits[i] ?? 0)
  }

  // ── Custom gates ───────────────────────────────────────────────────────────

  /** Register a saved circuit as a named gate usable in `addGate`. */
  registerGate(name: string, obj: CompatSavedCircuit | QuantumCircuit): void {
    this.#customGates.set(name, obj instanceof QuantumCircuit ? obj.save() : obj)
  }

  // ── Translation to ket ─────────────────────────────────────────────────────

  /**
   * Build the equivalent immutable ket `Circuit` — the escape hatch out of this
   * API. Everything ket can do (MPS, stabilizer, density-matrix and stabilizer-rank
   * backends, noise, device checks, hardware submission) is reached through here.
   */
  toKet(initialValues?: readonly (number | boolean)[]): Circuit {
    let c = new Circuit(this.numQubits)
    for (const [name, size] of this.#cregSizes) c = c.creg(name, size)
    if (initialValues) initialValues.forEach((v, w) => { if (v) c = c.x(w) })
    for (const g of this.#ordered()) c = this.#apply(c, g, 0)
    return c
  }

  #apply(c: Circuit, g: PlacedGate, depth: number): Circuit {
    const cond = g.options.condition
    if (cond) {
      if (!this.#cregSizes.has(cond.creg))
        throw new TypeError(`compat: gate '${g.name}' is conditioned on unknown register '${cond.creg}'`)
      return c.if(cond.creg, cond.value, inner => this.#applyBare(inner, g, depth))
    }
    return this.#applyBare(c, g, depth)
  }

  #applyBare(c: Circuit, g: PlacedGate, depth: number): Circuit {
    const custom = this.#customGates.get(g.name)
    if (custom) {
      if (depth > 32) throw new TypeError(`compat: custom gates nest too deeply at '${g.name}' (recursive?)`)
      for (const inner of savedGates(custom)) {
        const mapped = inner.wires.map(w => {
          const outer = g.wires[w]
          if (outer === undefined)
            throw new TypeError(`compat: custom gate '${g.name}' uses wire ${w}, but only ${g.wires.length} were given`)
          return outer
        })
        c = this.#apply(c, { ...inner, wires: mapped }, depth + 1)
      }
      return c
    }

    const [a] = g.wires
    switch (g.name) {
      case 'measure': {
        const target = g.options.creg
        if (!target) throw new TypeError('compat: measure gate is missing its creg option')
        return c.measure(a!, target.name, target.bit)
      }
      case 'reset':   return c.reset(a!)
      case 'barrier': return g.wires.length ? c.barrier(...g.wires) : c.barrier()
      default:        return applyNamed(c, g.name, [...this.#params(g), ...g.wires])
    }
  }

  /** Resolve a gate's parameters, in ket's argument order. Strings are angle expressions. */
  #params(g: PlacedGate): number[] {
    const names = GATE_PARAMS[g.name] ?? []
    return names.map(n => {
      const raw = g.options.params?.[n]
      if (raw === undefined) throw new TypeError(`compat: gate '${g.name}' is missing parameter '${n}'`)
      return typeof raw === 'number' ? raw : parseAngle(raw)
    })
  }

  // ── Execution ──────────────────────────────────────────────────────────────

  /**
   * Evaluate the circuit. `initialValues` sets wires to |1⟩ before the first gate.
   *
   * As in the original, a measurement gate is *non-destructive* — it writes to the
   * classical register but leaves the state alone — unless the circuit contains a
   * classically-controlled gate or a reset, in which case it collapses.
   *
   * Each unseeded run draws a fresh seed, so running the same circuit in a loop
   * gives independent results. Pass `options.seed` for a reproducible run; this is
   * a ket extension the original does not have.
   */
  run(initialValues?: readonly (number | boolean)[], options: CompatRunOptions = {}): this {
    const ordered = this.#ordered()
    const destructive = ordered.some(g => g.options.condition || g.name === 'reset')
    const measures    = ordered.filter(g => g.name === 'measure')

    this.#cregBits.clear()
    this.#collapsed = null

    if (measures.length || destructive) {
      const sampled = this.toKet(initialValues).run({ shots: 1, seed: options.seed ?? randomSeed() })
      for (const [name, bits] of Object.entries(sampled.cregs))
        this.#cregBits.set(name, bits.map(p => (p > 0.5 ? 1 : 0)))

      const outcome = Object.keys(sampled.probs)[0]
      if (destructive && outcome !== undefined) {
        this.#collapsed = outcome.split('').map(ch => (ch === '1' ? 1 : 0))
        this.#state = basisState(outcome)
        return this
      }
    }

    // Non-destructive: report the state as though the measurements had not happened.
    const pure = this.#pureCircuit(initialValues)
    this.#state = pure.statevector()
    return this
  }

  /** The circuit with measure/reset/condition stripped, for the non-destructive state. */
  #pureCircuit(initialValues?: readonly (number | boolean)[]): Circuit {
    let c = new Circuit(this.numQubits)
    if (initialValues) initialValues.forEach((v, w) => { if (v) c = c.x(w) })
    for (const g of this.#ordered()) {
      if (g.name === 'measure' || g.name === 'reset' || g.options.condition) continue
      c = this.#applyBare(c, g, 0)
    }
    return c
  }

  #requireState(): Map<bigint, Complex> {
    if (!this.#state) throw new TypeError('compat: circuit is not initialized — call run() first')
    return this.#state
  }

  // ── Reading the state ──────────────────────────────────────────────────────

  /** Probability of |1⟩ on each wire, indexed by wire. */
  probabilities(): number[] {
    const state = this.#requireState()
    const out = new Array<number>(this.numQubits).fill(0)
    for (const [idx, amp] of state) {
      const p = amp.re * amp.re + amp.im * amp.im
      for (let w = 0; w < this.numQubits; w++) if ((idx >> BigInt(w)) & 1n) out[w]! += p
    }
    return out.map(round14)
  }

  /** Probability of |1⟩ on one wire. */
  probability(wire: number): number {
    const p = this.probabilities()[wire]
    if (p === undefined) throw new TypeError(`compat: wire ${wire} out of range`)
    return p
  }

  /** Sample every wire once, returning an array of 0/1 indexed by wire. */
  measureAll(forceSampling = false): number[] {
    if (this.#collapsed && !forceSampling) return [...this.#collapsed]
    const state = this.#requireState()
    let weight = Math.random()
    let last = 0n
    for (const [idx, amp] of state) {
      last = idx
      weight -= amp.re * amp.re + amp.im * amp.im
      if (weight <= 0) return bitsOfIndex(idx, this.numQubits)
    }
    return bitsOfIndex(last, this.numQubits)
  }

  /** Sample one wire, optionally storing the result into a classical register. */
  measure(wire: number, creg?: string, cbit?: number, forceSampling = false): number {
    const bit = this.measureAll(forceSampling)[wire]
    if (bit === undefined) throw new TypeError(`compat: wire ${wire} out of range`)
    if (creg !== undefined && cbit !== undefined) this.setCregBit(creg, cbit, bit)
    return bit
  }

  /**
   * Run `shots` times and return `{ bitstring: count }`, with the highest wire
   * leftmost — the original package's ordering, not ket's.
   */
  measureAllMultishot(shots = 1, _forceRerun?: boolean): Record<string, number> {
    const d = this.toKet().run({ shots, seed: randomSeed() })
    const counts: Record<string, number> = {}
    for (const [bits, p] of Object.entries(d.probs)) {
      const n = Math.round(p * shots)
      if (n > 0) counts[flipBitOrder(bits)] = n
    }
    return counts
  }

  /** The state as text, one amplitude per line. Pass `true` for non-zero terms only. */
  stateAsString(onlyPossible = false): string {
    if (!this.#state) return 'Error: circuit is not initialized. Please call initState() or run() method.'
    let s = ''
    for (let i = 0; i < this.numAmplitudes(); i++) {
      const amp = this.#state.get(BigInt(i)) ?? { re: 0, im: 0 }
      const re = round14(amp.re), im = round14(amp.im)
      if (onlyPossible && !re && !im) continue
      const percent = ((re * re + im * im) * 100).toFixed(5).padStart(9, ' ')
      s += `${formatComplex(amp)}|${i.toString(2).padStart(this.numQubits, '0')}>\t${percent}%\n`
    }
    return s
  }

  /** Print the state to the console. */
  print(onlyPossible = false): void {
    // eslint-disable-next-line no-console
    console.log(this.stateAsString(onlyPossible))
  }

  // ── Save / load ────────────────────────────────────────────────────────────

  /** Export to the original package's plain-object format. */
  save(): CompatSavedCircuit {
    const cols = this.numCols()
    const grid: (CompatSavedGate | null)[][] = Array.from(
      { length: this.numQubits }, () => new Array<CompatSavedGate | null>(cols).fill(null),
    )
    for (const g of this.#ordered()) {
      g.wires.forEach((w, connector) => {
        grid[w]![g.column] = { id: g.id, name: g.name, connector, options: g.options }
      })
    }
    return {
      numQubits: this.numQubits,
      params: [],
      options: {},
      gates: grid,
      customGates: Object.fromEntries(this.#customGates),
      cregs: Object.fromEntries(this.#cregSizes),
    }
  }

  /** Replace this circuit's contents with a previously saved object. */
  load(obj: CompatSavedCircuit): void {
    this.init(obj.numQubits)
    for (const [name, size] of Object.entries(obj.cregs ?? {})) this.createCreg(name, size)
    for (const [name, def] of Object.entries(obj.customGates ?? {})) this.#customGates.set(name, def)
    this.#gates = savedGates(obj)
  }

  // ── Import / export ────────────────────────────────────────────────────────

  /** Replace this circuit's contents with a parsed OpenQASM program. */
  importQASM(input: string): void {
    this.#loadKet(Circuit.fromQASM(input))
  }

  /** Serialize to OpenQASM 2.0. */
  exportQASM(): string { return `${this.toKet().toQASM()}\n` }

  /** Serialize to Qiskit (Python). */
  exportToQiskit(): string { return this.toKet().toQiskit() }
  /** Serialize to Cirq (Python). */
  exportToCirq(): string { return this.toKet().toCirq() }
  /** Serialize to Quil. */
  exportToQuil(): string { return this.toKet().toQuil() }
  /** Serialize to pyQuil (Python). */
  exportToPyquil(): string { return this.toKet().toPyQuil() }
  /** Serialize to Q#. */
  exportToQSharp(): string { return this.toKet().toQSharp() }
  /** Serialize to TensorFlow Quantum (Python). */
  exportToTFQ(): string { return this.toKet().toTFQ() }
  /** Serialize to Amazon Braket (Python). */
  exportToBraket(): string { return this.toKet().toBraket() }
  /** Serialize to IonQ's JSON job format. */
  exportToIonq(): unknown { return this.toKet().toIonQ() }
  /** Render to SVG. */
  exportSVG(): string { return this.toKet().toSVG() }

  /** Rebuild this circuit's gate list from a ket `Circuit`. */
  #loadKet(c: Circuit): void {
    const json = c.toJSON()
    this.init(json.qubits)
    for (const [name, size] of Object.entries(json.cregs ?? {})) this.createCreg(name, size)
    for (const op of json.ops) {
      const placed = compatGateFromOp(op as Record<string, unknown>)
      if (placed) this.addGate(placed.name, -1, placed.wires, placed.options)
    }
  }
}

// ─── Module-level helpers ─────────────────────────────────────────────────────

/** Rebuild the flat gate list from a saved `gates[wire][column]` grid. */
function savedGates(obj: CompatSavedCircuit): PlacedGate[] {
  const byId = new Map<string, PlacedGate>()
  obj.gates.forEach((row, wire) => {
    row?.forEach((cell, column) => {
      if (!cell) return
      const existing = byId.get(cell.id)
      if (existing) existing.wires[cell.connector] = wire
      else {
        const wires: number[] = []
        wires[cell.connector] = wire
        byId.set(cell.id, { id: cell.id, name: cell.name, column, wires, options: cell.options ?? {} })
      }
    })
  })
  return [...byId.values()].sort((a, b) => a.column - b.column)
}

/** Split a state index into per-wire bits, wire 0 first. */
function bitsOfIndex(idx: bigint, numQubits: number): number[] {
  return Array.from({ length: numQubits }, (_, w) => Number((idx >> BigInt(w)) & 1n))
}

/** Build a single-basis-state statevector from a ket bitstring. */
function basisState(bits: string): Map<bigint, Complex> {
  let idx = 0n
  bits.split('').forEach((ch, w) => { if (ch === '1') idx |= 1n << BigInt(w) })
  return new Map([[idx, { re: 1, im: 0 }]])
}

/** Translate one op from `Circuit.toJSON()` into a compat gate placement. */
function compatGateFromOp(op: Record<string, unknown>): { name: string; wires: number[]; options: CompatGateOptions } | null {
  const meta = op['meta'] as { name?: string; params?: number[] } | undefined
  const named = (name: string, wires: number[]): { name: string; wires: number[]; options: CompatGateOptions } => {
    const names = GATE_PARAMS[name] ?? []
    const params: Record<string, number> = {}
    names.forEach((p, i) => { params[p] = meta?.params?.[i] ?? 0 })
    return { name, wires, options: names.length ? { params } : {} }
  }
  switch (op['kind']) {
    case 'single':     return named(meta?.name ?? '', [op['q'] as number])
    case 'controlled': return named(meta?.name ?? '', [op['control'] as number, op['target'] as number])
    case 'cnot':       return named('cx',    [op['control'] as number, op['target'] as number])
    case 'swap':       return named('swap',  [op['a'] as number, op['b'] as number])
    case 'two':        return named(meta?.name ?? '', [op['a'] as number, op['b'] as number])
    case 'toffoli':    return named('ccx',   [op['c1'] as number, op['c2'] as number, op['target'] as number])
    case 'cswap':      return named('cswap', [op['control'] as number, op['a'] as number, op['b'] as number])
    case 'reset':      return { name: 'reset',   wires: [op['q'] as number], options: {} }
    case 'barrier':    return { name: 'barrier', wires: op['qubits'] as number[], options: {} }
    case 'measure':    return {
      name: 'measure',
      wires: [op['q'] as number],
      options: { creg: { name: op['creg'] as string, bit: op['bit'] as number } },
    }
    default: return null
  }
}

export default QuantumCircuit
