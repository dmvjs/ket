/**
 * Making the adversary adversarial.
 *
 * `analyzeHardness` prices a circuit as written. But a circuit as written is only
 * one representative of everything that computes the same answer, and an attacker
 * is not obliged to take the presentation they were handed. The hardness
 * parameters move under rewriting: deleting gates that cannot affect the answer
 * shrinks every one of them, merging rotations removes T gates outright, and
 * relabelling qubits changes the bond dimension without changing the state.
 *
 * So this searches. Each rewrite here is answer-preserving — and what exactly it
 * preserves is recorded on the result, because the three kinds are not
 * interchangeable:
 *
 * - **`state`** — the full statevector is unchanged. Safe for anything.
 * - **`observable`** — only the marginal on the named qubits survives. A light
 *   cone throws away everything outside it, so a question about other qubits has
 *   no answer any more.
 * - **`permuted-state`** — the state is unchanged up to a relabelling of the
 *   qubits, which is given so the caller can undo it.
 *
 * A rewrite that quietly changed the answer would turn this from an attack into a
 * wrong result delivered confidently, so the suite checks every one of them
 * against `exactProbs()` rather than reasoning about why they must be correct.
 */
import { Circuit } from './circuit.js'
import type { CircuitJSON } from './circuit.js'

/** What a rewrite leaves intact. */
export type Preserves = 'state' | 'observable' | 'permuted-state'

export interface Rewrite {
  name: string
  preserves: Preserves
  circuit: Circuit
  /** Gates before the rewrite. */
  before: number
  /** Gates after. */
  after: number
  /**
   * `newIndex[oldQubit]`, when the rewrite moved qubits. Absent when it did not.
   * A dropped qubit maps to −1.
   */
  mapping?: number[]
}

type JsonOp = Record<string, unknown> & { kind: string }

/** Every qubit an op touches. */
export function qubitsOf(op: JsonOp): number[] {
  const n = (k: string): number | undefined => op[k] as number | undefined
  const out: number[] = []
  for (const k of ['q', 'control', 'target', 'a', 'b', 'c1', 'c2']) {
    const v = n(k)
    if (typeof v === 'number') out.push(v)
  }
  for (const k of ['qubits']) {
    const v = op[k] as number[] | undefined
    if (Array.isArray(v)) out.push(...v)
  }
  return out
}

/** Rewrite every qubit index in `op` through `map`. */
function remapOp(op: JsonOp, map: readonly number[]): JsonOp {
  const out: JsonOp = { ...op }
  for (const k of ['q', 'control', 'target', 'a', 'b', 'c1', 'c2']) {
    if (typeof op[k] === 'number') out[k] = map[op[k] as number]!
  }
  if (Array.isArray(op['qubits'])) out['qubits'] = (op['qubits'] as number[]).map(q => map[q]!)
  return out
}

const rebuild = (json: CircuitJSON, qubits: number, ops: JsonOp[]): Circuit =>
  Circuit.fromJSON({ ...json, qubits, ops })

/** Gates, excluding the bookkeeping ops that carry no cost. */
const countGates = (ops: readonly JsonOp[]): number =>
  ops.filter(o => o.kind !== 'barrier').length

// ── Light cone ────────────────────────────────────────────────────────────────

/**
 * Delete everything that cannot reach the qubits you are asking about.
 *
 * Walking backward from the end, a gate matters only if it touches a qubit
 * already known to matter; if it does, everything it touches starts mattering
 * too. Anything else acts on a part of the state that is traced away, and can go.
 *
 * This is the first thing a real attack does and it is often the whole attack.
 * For a single-site observable on a depth-d circuit the cone spans about 2d+1
 * qubits however wide the circuit is, so a 100-qubit depth-10 circuit collapses to
 * around 21 qubits — and 2^100 becomes 2^21.
 *
 * Qubits left untouched are dropped rather than kept idle, because a statevector's
 * cost is 2^n and an idle qubit still costs its factor of two. They were in |0⟩
 * and stayed there, so nothing is lost but the width.
 */
export function lightCone(circuit: Circuit, qubits: readonly number[]): Rewrite {
  const n = circuit.qubits
  for (const q of qubits) {
    if (!Number.isInteger(q) || q < 0 || q >= n) throw new RangeError(`lightCone: qubit ${q} out of range for ${n} qubits`)
  }
  if (qubits.length === 0) throw new RangeError('lightCone: name at least one qubit to keep')

  const json = circuit.toJSON()
  const ops = json.ops as JsonOp[]
  const live = new Set<number>(qubits)
  const keep = new Array<boolean>(ops.length).fill(false)

  for (let i = ops.length - 1; i >= 0; i--) {
    const op = ops[i]!
    if (op.kind === 'barrier') continue
    // Mid-circuit ops write classical state that later ops may read, so the
    // causal analysis above does not cover them. Keeping everything is the only
    // safe answer, and saying so beats silently dropping a measurement.
    if (op.kind === 'measure' || op.kind === 'reset' || op.kind === 'if') {
      throw new TypeError('lightCone requires a pure circuit — remove measure/reset/if ops')
    }
    const touched = qubitsOf(op)
    if (touched.some(q => live.has(q))) {
      keep[i] = true
      for (const q of touched) live.add(q)
    }
  }

  // Compact to the qubits that survived, in ascending order.
  const survivors = [...live].sort((a, b) => a - b)
  const mapping = new Array<number>(n).fill(-1)
  survivors.forEach((q, i) => { mapping[q] = i })

  const kept = ops.filter((_, i) => keep[i]).map(op => remapOp(op, mapping))
  return {
    name: 'light-cone',
    preserves: 'observable',
    circuit: rebuild(json, survivors.length, kept),
    before: countGates(ops),
    after: countGates(kept),
    mapping,
  }
}

// ── Peephole ──────────────────────────────────────────────────────────────────

/** Gates that are their own inverse, so two in a row vanish. */
const SELF_INVERSE = new Set(['h', 'x', 'y', 'z', 'id'])
/** Rotations that merge by adding angles. */
const ADDITIVE = new Set(['rx', 'ry', 'rz', 'u1', 'p'])
/** Named inverse pairs. */
const INVERSE_OF: Record<string, string> = {
  s: 'sdg', sdg: 's', si: 's', t: 'tdg', tdg: 't', ti: 't', v: 'vi', vi: 'v',
}

const nameOf = (op: JsonOp): string => ((op['meta'] as { name?: string } | undefined)?.name) ?? ''
const paramsOf = (op: JsonOp): number[] => ((op['meta'] as { params?: number[] } | undefined)?.params) ?? []

const sameTwoQubit = (a: JsonOp, b: JsonOp): boolean => {
  if (a.kind !== b.kind) return false
  if (a.kind === 'cnot') return a['control'] === b['control'] && a['target'] === b['target']
  if (a.kind === 'swap') return (a['a'] === b['a'] && a['b'] === b['b']) || (a['a'] === b['b'] && a['b'] === b['a'])
  return false
}

/**
 * Cancel and merge neighbouring gates, to a fixed point.
 *
 * Three rules, all exact circuit identities: a self-inverse gate twice in a row
 * vanishes, a named inverse pair vanishes, and two rotations about the same axis
 * add. "In a row" means nothing else touches those qubits in between, which is
 * what the per-qubit last-op index tracks.
 *
 * Worth having beyond the gate count, because a merged pair of T gates is a T
 * gate *removed* — and T-count is what the stabilizer-rank attack is priced on.
 * A circuit can look expensive purely because nobody simplified it.
 */
export function peephole(circuit: Circuit): Rewrite {
  const json = circuit.toJSON()
  let ops = json.ops as JsonOp[]
  const originally = countGates(ops)

  for (let pass = 0; pass < 64; pass++) {
    const alive = ops.map(() => true)
    /** Index of the most recent surviving op on each qubit, or -1. */
    const last = new Map<number, number>()
    let changed = false

    for (let i = 0; i < ops.length; i++) {
      const op = ops[i]!
      if (op.kind === 'barrier') { for (const q of qubitsOf(op)) last.set(q, -1); continue }
      const qs = qubitsOf(op)
      // Only a gate whose every qubit's previous op is the *same* op can pair
      // with it; otherwise something intervened on at least one wire.
      const prevs = new Set(qs.map(q => last.get(q) ?? -1))
      const prev = prevs.size === 1 ? [...prevs][0]! : -1
      const p = prev >= 0 ? ops[prev]! : null

      if (p && qubitsOf(p).length === qs.length) {
        const an = nameOf(op), pn = nameOf(p)
        const oneQubit = qs.length === 1 && op.kind === 'single' && p.kind === 'single'
        if (oneQubit && an === pn && SELF_INVERSE.has(an)) {
          alive[i] = false; alive[prev] = false; changed = true
          for (const q of qs) last.set(q, -1)
          continue
        }
        if (oneQubit && INVERSE_OF[pn] === an) {
          alive[i] = false; alive[prev] = false; changed = true
          for (const q of qs) last.set(q, -1)
          continue
        }
        if (oneQubit && an === pn && ADDITIVE.has(an)) {
          const sum = (paramsOf(p)[0] ?? 0) + (paramsOf(op)[0] ?? 0)
          alive[prev] = false
          ops[i] = { ...op, meta: { name: an, params: [sum] } }
          changed = true
          for (const q of qs) last.set(q, i)
          continue
        }
        if (sameTwoQubit(op, p)) {
          alive[i] = false; alive[prev] = false; changed = true
          for (const q of qs) last.set(q, -1)
          continue
        }
      }
      for (const q of qs) last.set(q, i)
    }

    if (!changed) break
    ops = ops.filter((_, i) => alive[i]!)
  }

  return {
    name: 'peephole',
    preserves: 'state',
    circuit: rebuild(json, circuit.qubits, ops),
    before: originally,
    after: countGates(ops),
  }
}

// ── Basis change ──────────────────────────────────────────────────────────────

/**
 * Rewrite transverse rotations into the Z basis: `Rx(θ) = H·Rz(θ)·H`.
 *
 * Gate count goes *up*, which is why this is not an optimisation, and why the
 * search has to judge it on the report rather than on size. What it buys is
 * eligibility: several backends accept `rz` and not `rx`, so a circuit whose only
 * non-Clifford gates are transverse rotations looks unattackable to them purely
 * because of the axis it is written in.
 *
 * The case this exists for is the 127-qubit Trotter family, where `Rzz(−π/2)` is
 * Clifford and `Rx(θ)` is the only thing that is not. Written as-is, the magic
 * attack declines the circuit outright. Rewritten, the whole thing is Clifford
 * plus Z rotations, and at θ = π/4 those are literally T gates — so the T-count
 * becomes visible and the circuit can be priced instead of refused.
 *
 * `Ry(θ) = S·H·Rz(θ)·H·S†` is the same trick one conjugation deeper.
 */
export function toZBasis(circuit: Circuit): Rewrite {
  const json = circuit.toJSON()
  const ops = json.ops as JsonOp[]
  const out: JsonOp[] = []

  for (const op of ops) {
    const name = nameOf(op)
    if (op.kind !== 'single' || (name !== 'rx' && name !== 'ry')) { out.push(op); continue }
    const q = op['q'] as number
    const theta = paramsOf(op)[0] ?? 0
    const h = { kind: 'single', q, meta: { name: 'h' } } as JsonOp
    const rz = { kind: 'single', q, meta: { name: 'rz', params: [theta] } } as JsonOp
    if (name === 'rx') { out.push(h, rz, h); continue }
    // Ry(θ) = S H Rz(θ) H S†  —  a further quarter turn about z on each side.
    out.push(
      { kind: 'single', q, meta: { name: 's' } } as JsonOp, h, rz, h,
      { kind: 'single', q, meta: { name: 'sdg' } } as JsonOp,
    )
  }

  return {
    name: 'z-basis',
    preserves: 'state',
    circuit: rebuild(json, circuit.qubits, out),
    before: countGates(ops),
    after: countGates(out),
  }
}

// ── Qubit ordering ────────────────────────────────────────────────────────────

/**
 * Total distance spanned by two-qubit gates under an ordering.
 *
 * The MPS backend treats the qubit index as position on a chain, so a gate
 * between distant indices costs a SWAP network and drives χ up. Total span is the
 * standard cheap proxy for that, and minimising it is the linear-arrangement
 * problem — NP-hard, hence a search rather than a solve.
 */
export function orderingCost(circuit: Circuit, order: readonly number[]): number {
  let cost = 0
  for (const raw of circuit.toJSON().ops) {
    const op = raw as JsonOp
    if (op.kind === 'barrier') continue
    const qs = qubitsOf(op)
    if (qs.length < 2) continue
    const at = qs.map(q => order[q]!)
    cost += Math.max(...at) - Math.min(...at)
  }
  return cost
}

/**
 * Order qubits by the Fiedler vector of the connectivity Laplacian.
 *
 * Greedy pairwise swaps are a local search and get stuck: on a grid they cannot
 * find the snake ordering from a random start, because every single swap out of a
 * bad layout makes the span worse before it makes it better. Spectral ordering is
 * the standard principled answer to minimum linear arrangement — embed the
 * connectivity graph on a line by the second-smallest Laplacian eigenvector and
 * read off the order — and it lands close to a snake on lattice-like graphs
 * without needing to know it is a lattice.
 *
 * The eigenvector comes from power iteration on `μI − L` with the all-ones vector
 * projected out at every step, since all-ones is the zero eigenvector and would
 * otherwise dominate.
 */
export function spectralOrder(circuit: Circuit): number[] {
  const n = circuit.qubits
  if (n < 3) return Array.from({ length: n }, (_, i) => i)

  // Weighted adjacency: how often each pair is acted on together.
  const adj = new Float64Array(n * n)
  const deg = new Float64Array(n)
  for (const raw of circuit.toJSON().ops) {
    const op = raw as JsonOp
    if (op.kind === 'barrier') continue
    const qs = [...new Set(qubitsOf(op))]
    for (let i = 0; i < qs.length; i++) for (let j = i + 1; j < qs.length; j++) {
      adj[qs[i]! * n + qs[j]!] = adj[qs[i]! * n + qs[j]!]! + 1
      adj[qs[j]! * n + qs[i]!] = adj[qs[j]! * n + qs[i]!]! + 1
      deg[qs[i]!] = deg[qs[i]!]! + 1
      deg[qs[j]!] = deg[qs[j]!]! + 1
    }
  }

  // Shift so the iteration converges to the *smallest* nonzero eigenvector of L.
  let shift = 0
  for (let i = 0; i < n; i++) shift = Math.max(shift, 2 * deg[i]!)
  if (shift === 0) return Array.from({ length: n }, (_, i) => i)

  let v = Array.from({ length: n }, (_, i) => Math.sin(i * 1.7) + Math.cos(i * 0.9))
  const centre = (x: number[]): void => {
    let m = 0
    for (const y of x) m += y
    m /= n
    for (let i = 0; i < n; i++) x[i] = x[i]! - m
    let norm = 0
    for (const y of x) norm += y * y
    norm = Math.sqrt(norm)
    if (norm > 1e-300) for (let i = 0; i < n; i++) x[i] = x[i]! / norm
  }
  centre(v)

  for (let iter = 0; iter < 300; iter++) {
    // w = (shift·I − L)·v, with L = D − A.
    const w = new Array<number>(n).fill(0)
    for (let i = 0; i < n; i++) {
      let s = (shift - deg[i]!) * v[i]!
      for (let j = 0; j < n; j++) if (adj[i * n + j]! !== 0) s += adj[i * n + j]! * v[j]!
      w[i] = s
    }
    centre(w)                       // projects out the all-ones eigenvector
    let delta = 0
    for (let i = 0; i < n; i++) delta = Math.max(delta, Math.abs(w[i]! - v[i]!))
    v = w
    if (delta < 1e-12) break
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => v[a]! - v[b]!)
  const mapping = new Array<number>(n)
  order.forEach((q, pos) => { mapping[q] = pos })
  return mapping
}

/**
 * Search for a qubit relabelling that shortens the two-qubit gates.
 *
 * Greedy pairwise swaps from several random starts, keeping the best. The state
 * is unchanged up to the relabelling, which is returned so a caller can map
 * results back — this is why `preserves` is `permuted-state` and not `state`.
 *
 * A circuit already laid out on a line has nothing to find here and the search
 * says so by returning the identity. Where it pays is a circuit whose natural
 * indices do not match its connectivity — a 2-D grid flattened row-major, for
 * instance, where vertical bonds span a whole row.
 */
export function searchQubitOrder(
  circuit: Circuit,
  { restarts = 8, sweeps = 20, seed = 1 }: { restarts?: number; sweeps?: number; seed?: number } = {},
): Rewrite {
  const n = circuit.qubits
  const json = circuit.toJSON()
  let rng = seed >>> 0 || 1
  const rand = (): number => { rng ^= rng << 13; rng ^= rng >>> 17; rng ^= rng << 5; return (rng >>> 0) / 0x100000000 }

  let best = Array.from({ length: n }, (_, i) => i)
  let bestCost = orderingCost(circuit, best)

  // Seed with the spectral ordering, then let local search polish it. Greedy
  // swaps from a random start cannot reach a snake layout on a lattice — every
  // single swap out of a bad arrangement makes the span worse first.
  const spectral = spectralOrder(circuit)
  const spectralCost = orderingCost(circuit, spectral)
  if (spectralCost < bestCost) { best = spectral; bestCost = spectralCost }

  for (let r = 0; r < restarts + 1; r++) {
    const order = r === 0
      ? spectral.slice()
      : r === 1
      ? Array.from({ length: n }, (_, i) => i)
      : (() => {
          const o = Array.from({ length: n }, (_, i) => i)
          for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [o[i], o[j]] = [o[j]!, o[i]!] }
          return o
        })()
    let cost = orderingCost(circuit, order)
    for (let s = 0; s < sweeps; s++) {
      let improved = false
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        ;[order[i], order[j]] = [order[j]!, order[i]!]
        const trial = orderingCost(circuit, order)
        if (trial < cost) { cost = trial; improved = true }
        else [order[i], order[j]] = [order[j]!, order[i]!]
      }
      if (!improved) break
    }
    if (cost < bestCost) { bestCost = cost; best = order.slice() }
  }

  const ops = (json.ops as JsonOp[]).map(op => remapOp(op, best))
  return {
    name: 'qubit-order',
    preserves: 'permuted-state',
    circuit: rebuild(json, n, ops),
    before: countGates(json.ops as JsonOp[]),
    after: countGates(ops),
    mapping: best,
  }
}
