/**
 * Circuit amplitudes by tensor-network contraction, with contraction-order search.
 *
 * A circuit is a tensor network: every gate is a tensor, every wire segment an
 * index, and ⟨x|U|0⟩ is the network contracted to a scalar. The cost of that has
 * nothing to do with 2ⁿ — it is governed by how wide the largest intermediate
 * tensor gets, which is a property of the circuit's connectivity (its treewidth)
 * rather than its qubit count. A shallow circuit on 60 qubits can be contracted
 * in milliseconds; a statevector for it does not fit in memory.
 *
 * Which means the whole problem is *the order you contract in*. The same network
 * contracted well or badly differs by many orders of magnitude, so the search for
 * an order is the algorithm, and the contraction itself is bookkeeping. This is
 * the technique behind the classical simulations of random-circuit sampling
 * experiments.
 *
 * The search here is randomized greedy with restarts: repeatedly contract the
 * pair that looks cheapest, with the tie-break jittered, and keep the best plan
 * found. Restarts are the whole search, and they are cheap — a plan for a
 * 100-qubit network takes about 4ms — so the default runs many of them. That is the standard baseline. It is not hypergraph partitioning
 * (KaHyPar-class, as `cotengra` uses), which does better on large hard networks —
 * so `planContraction` is deliberately separate from the contraction itself, and
 * a better planner can be dropped in without touching anything else.
 */

import type { Circuit } from './circuit.js'
import * as G from './gates.js'
import type { Gate4x4 } from './statevector.js'

/** A tensor over `indices`, complex, stored re/im interleaved in row-major order. */
export interface Tensor {
  /** Index labels; every index has dimension 2 for a qubit circuit. */
  indices: string[]
  /** 2 × 2^indices.length numbers: `data[2k]` real, `data[2k+1]` imaginary. */
  data: Float64Array
}

/** One step of a contraction plan: contract these two tensor slots. */
export type ContractionStep = readonly [number, number]

export interface ContractionPlan {
  steps: ContractionStep[]
  /** Widest intermediate, as a count of indices — memory goes as 2^width. */
  width: number
  /** Total multiply-accumulate operations the plan will perform. */
  cost: number
}

const size = (n: number): number => 2 ** n

/**
 * Largest of a collection, by iteration.
 *
 * `Math.max(0, ...xs)` passes one argument per element, and V8 overflows the
 * call stack past about 124,500 of them. A 43-qubit Shor network is 191,583
 * tensors, so every planner here reported `Maximum call stack size exceeded` on
 * circuits of that size — an error where a width belongs, and one that reads as
 * "not applicable" rather than "not measured". Nothing on this path is hot; the
 * spread was only shorter to write.
 */
const maxOf = (xs: Iterable<number>): number => {
  let m = 0
  for (const x of xs) if (x > m) m = x
  return m
}

/**
 * Search for a contraction order.
 *
 * Greedy on the pair whose contraction produces the smallest result, restarted
 * with jittered tie-breaking. Width is minimised first because it sets memory,
 * which is what actually makes a contraction impossible; cost breaks ties.
 */
export function planContraction(
  network: readonly (readonly string[])[],
  { restarts = 64, seed = 1 }: { restarts?: number; seed?: number } = {},
): ContractionPlan {
  let rng = seed >>> 0 || 1
  const rand = (): number => {
    rng ^= rng << 13; rng ^= rng >>> 17; rng ^= rng << 5
    return (rng >>> 0) / 0x100000000
  }

  let best: ContractionPlan | null = null

  for (let attempt = 0; attempt < restarts; attempt++) {
    const jitter = attempt === 0 ? 0 : 0.35
    const live: (Set<string> | undefined)[] = network.map(idx => new Set(idx))

    // Candidates come from the network's own connectivity, and are kept in a heap
    // so a step costs the merged tensor's degree rather than a scan of every
    // pair. Rescanning was O(tensors x candidates) per step: at n=28 that made
    // planning 957ms against 17ms of actual contraction.
    const held = holdCounts(network)
    const holders = new Map<string, Set<number>>()
    live.forEach((idx, i) => {
      for (const x of idx!) {
        let set = holders.get(x)
        if (!set) { set = new Set(); holders.set(x, set) }
        set.add(i)
      }
    })

    const neighbours: Set<number>[] = live.map(() => new Set<number>())
    for (const set of holders.values()) {
      const members = [...set]
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < members.length; j++) {
          neighbours[members[i]!]!.add(members[j]!)
          neighbours[members[j]!]!.add(members[i]!)
        }
      }
    }

    // Version counters make invalidation lazy: an entry is stale if either of its
    // tensors has changed since the entry was pushed, so nothing has to be found
    // and removed from the heap when a merge happens.
    const version = new Int32Array(live.length)
    const heap = new MinHeap()

    // A shared index only leaves the network when this pair holds the last two
    // copies of it; while a third tensor still carries it the index survives the
    // step as a batch index. Work is 2^|union| either way — it is the *rank* of
    // the result that a hyper-index changes, and rank is what sets memory.
    const scoreOf = (a: number, b: number): { score: number; outRank: number; cost: number } => {
      const A = live[a]!, B = live[b]!
      let shared = 0, summed = 0
      for (const x of A) if (B.has(x)) { shared++; if (held.get(x) === 2) summed++ }
      const union = A.size + B.size - shared
      const outRank = union - summed
      const cost = size(union)
      return { score: outRank + Math.log2(cost) * 0.001 + rand() * jitter, outRank, cost }
    }

    const push = (a: number, b: number): void => {
      const { score } = scoreOf(a, b)
      heap.push(score, a, b, version[a]!, version[b]!)
    }

    for (let i = 0; i < live.length; i++) {
      for (const j of neighbours[i]!) if (j > i) push(i, j)
    }

    const steps: ContractionStep[] = []
    let width = maxOf(live.map(t => t!.size))
    let cost = 0
    let remaining = live.length

    while (remaining > 1) {
      let a = -1, b = -1
      while (!heap.empty()) {
        const top = heap.pop()!
        if (!live[top.a] || !live[top.b]) continue
        if (top.va !== version[top.a] || top.vb !== version[top.b]) continue   // stale
        a = top.a; b = top.b
        break
      }

      // Nothing connected left: separate components, which is legal — an idle
      // wire is |0> meeting <0| and touches nothing else. Join two and continue.
      if (a < 0) {
        const alive: number[] = []
        for (let i = 0; i < live.length && alive.length < 2; i++) if (live[i]) alive.push(i)
        if (alive.length < 2) break
        a = alive[0]!; b = alive[1]!
      }

      const { outRank, cost: stepCost } = scoreOf(a, b)
      const A = live[a]!, B = live[b]!
      const merged = new Set<string>()
      for (const x of A) if (!B.has(x) || held.get(x)! > 2) merged.add(x)
      for (const x of B) if (!A.has(x) || held.get(x)! > 2) merged.add(x)

      for (const x of A) held.set(x, held.get(x)! - 1)
      for (const x of B) held.set(x, held.get(x)! - 1)
      for (const x of merged) held.set(x, held.get(x)! + 1)

      live[a] = merged
      live[b] = undefined
      remaining--
      steps.push([a, b])
      width = Math.max(width, outRank)
      cost += stepCost

      // b's neighbours become a's. Only these pairs need rescoring.
      version[a]!++
      for (const x of neighbours[b]!) {
        if (x === a || !live[x]) continue
        neighbours[a]!.add(x)
        neighbours[x]!.delete(b)
        neighbours[x]!.add(a)
      }
      neighbours[a]!.delete(b)
      neighbours[b]!.clear()
      for (const x of neighbours[a]!) if (live[x]) push(a, x)
    }

    if (!best || width < best.width || (width === best.width && cost < best.cost)) {
      best = { steps, width, cost }
    }
  }

  if (!best) throw new Error('planContraction: no plan found')
  return best
}

/** Binary min-heap over candidate contractions, with the versions that validate them. */
class MinHeap {
  #score: number[] = []
  #a: number[] = []
  #b: number[] = []
  #va: number[] = []
  #vb: number[] = []

  empty(): boolean { return this.#score.length === 0 }

  push(score: number, a: number, b: number, va: number, vb: number): void {
    this.#score.push(score); this.#a.push(a); this.#b.push(b); this.#va.push(va); this.#vb.push(vb)
    let i = this.#score.length - 1
    while (i > 0) {
      const parent = (i - 1) >> 1
      if (this.#score[parent]! <= this.#score[i]!) break
      this.#swap(i, parent)
      i = parent
    }
  }

  pop(): { a: number; b: number; va: number; vb: number } | null {
    if (this.empty()) return null
    const out = { a: this.#a[0]!, b: this.#b[0]!, va: this.#va[0]!, vb: this.#vb[0]! }
    const last = this.#score.length - 1
    this.#swap(0, last)
    this.#score.pop(); this.#a.pop(); this.#b.pop(); this.#va.pop(); this.#vb.pop()
    let i = 0
    for (;;) {
      const l = 2 * i + 1, r = l + 1
      let small = i
      if (l < this.#score.length && this.#score[l]! < this.#score[small]!) small = l
      if (r < this.#score.length && this.#score[r]! < this.#score[small]!) small = r
      if (small === i) break
      this.#swap(i, small)
      i = small
    }
    return out
  }

  #swap(i: number, j: number): void {
    ;[this.#score[i], this.#score[j]] = [this.#score[j]!, this.#score[i]!]
    ;[this.#a[i], this.#a[j]] = [this.#a[j]!, this.#a[i]!]
    ;[this.#b[i], this.#b[j]] = [this.#b[j]!, this.#b[i]!]
    ;[this.#va[i], this.#va[j]] = [this.#va[j]!, this.#va[i]!]
    ;[this.#vb[i], this.#vb[j]] = [this.#vb[j]!, this.#vb[i]!]
  }
}

/**
 * Reorder a tensor's indices, producing data in the new layout.
 *
 * Returns the tensor untouched when it is already in the requested order, which
 * is common: half the operands in a circuit network need no permutation at all.
 */
export function permuteTensor(t: Tensor, order: readonly string[]): Tensor {
  if (order.length === t.indices.length && order.every((x, i) => x === t.indices[i])) return t

  const rank = t.indices.length
  const srcStride = order.map(name => 1 << (rank - 1 - t.indices.indexOf(name)))
  const total = 1 << rank
  const out = new Float64Array(total * 2)

  for (let o = 0; o < total; o++) {
    let src = 0
    for (let k = 0; k < rank; k++) if ((o >> (rank - 1 - k)) & 1) src += srcStride[k]!
    out[2 * o] = t.data[2 * src]!
    out[2 * o + 1] = t.data[2 * src + 1]!
  }
  return { indices: [...order], data: out }
}

/**
 * Contract two tensors over their shared indices.
 *
 * Permute both operands so the shared indices are contiguous and trailing on the
 * left, leading on the right, and the contraction becomes an ordinary matrix
 * product: (free_a × shared) · (shared × free_b). The naive alternative — walking
 * output positions and re-deriving each operand's offset a bit at a time — pays
 * that decomposition on every element, and reads memory in a stride pattern the
 * hardware cannot prefetch.
 *
 * The zero test in the middle loop is not a micro-optimisation here. Gate tensors
 * are mostly zeros — a CNOT is 4 non-zero entries out of 16 — so skipping a zero
 * row of the left operand skips an entire pass over the right one.
 *
 * `keep` names shared indices that must survive rather than be summed, because
 * some third tensor still holds them — a hyper-index. Those become batch indices:
 * both operands are addressed at the same value and the output carries it, so the
 * matrix product runs once per assignment over a contiguous block of each operand.
 */
export function contractPair(a: Tensor, b: Tensor, keep?: ReadonlySet<string>): Tensor {
  const shared = a.indices.filter(i => b.indices.includes(i))
  const batch = keep && keep.size ? shared.filter(i => keep.has(i)) : []
  const summed = batch.length ? shared.filter(i => !keep!.has(i)) : shared
  const aFree = a.indices.filter(i => !shared.includes(i))
  const bFree = b.indices.filter(i => !shared.includes(i))

  const A = permuteTensor(a, [...batch, ...aFree, ...summed])
  const B = permuteTensor(b, [...batch, ...summed, ...bFree])

  const G = 1 << batch.length
  const M = 1 << aFree.length
  const K = 1 << summed.length
  const N = 1 << bFree.length
  const data = new Float64Array(G * M * N * 2)

  for (let g = 0; g < G; g++) {
    const aBase = g * M * K
    const bBase = g * K * N
    const cBase = g * M * N
    for (let i = 0; i < M; i++) {
      const aRow = aBase + i * K
      const cRow = cBase + i * N
      for (let k = 0; k < K; k++) {
        const ar = A.data[2 * (aRow + k)]!
        const ai = A.data[2 * (aRow + k) + 1]!
        if (ar === 0 && ai === 0) continue
        const bRow = bBase + k * N
        for (let j = 0; j < N; j++) {
          const br = B.data[2 * (bRow + j)]!
          const bi = B.data[2 * (bRow + j) + 1]!
          data[2 * (cRow + j)] = (data[2 * (cRow + j)] ?? 0) + ar * br - ai * bi
          data[2 * (cRow + j) + 1] = (data[2 * (cRow + j) + 1] ?? 0) + ar * bi + ai * br
        }
      }
    }
  }

  return { indices: [...batch, ...aFree, ...bFree], data }
}

/**
 * How many live tensors hold each index.
 *
 * A shared index may only be summed when the pair contracting it holds the last
 * two copies; while a third tensor still carries it, it is a hyper-index and has
 * to survive the step.
 */
function holdCounts(tensors: readonly (readonly string[])[]): Map<string, number> {
  const held = new Map<string, number>()
  for (const idx of tensors) for (const x of idx) held.set(x, (held.get(x) ?? 0) + 1)
  return held
}

/** Contract a whole network following a plan. */
export function contractNetwork(tensors: readonly Tensor[], plan: ContractionPlan): Tensor {
  const live: (Tensor | undefined)[] = tensors.slice()
  const held = holdCounts(tensors.map(t => t.indices))

  for (const [a, b] of plan.steps) {
    const A = live[a]!, B = live[b]!
    const keep = new Set<string>()
    for (const x of A.indices) if (B.indices.includes(x) && (held.get(x) ?? 0) > 2) keep.add(x)

    const merged = contractPair(A, B, keep)
    for (const x of A.indices) held.set(x, held.get(x)! - 1)
    for (const x of B.indices) held.set(x, held.get(x)! - 1)
    for (const x of merged.indices) held.set(x, (held.get(x) ?? 0) + 1)

    live[a] = merged
    live[b] = undefined
  }
  const remaining = live.filter((t): t is Tensor => t !== undefined)
  if (remaining.length !== 1) throw new Error(`contractNetwork: ${remaining.length} tensors left, expected 1`)
  return remaining[0]!
}

// ── Building a network from a circuit ─────────────────────────────────────────

const S2 = Math.SQRT1_2

/** Gate matrices as flat complex arrays, row-major. */
function matrixOf(name: string, params: readonly number[] | undefined): Float64Array | null {
  const p = params?.[0] ?? 0
  const c = Math.cos(p / 2), s = Math.sin(p / 2)
  const m = (...v: number[]): Float64Array => Float64Array.from(v)
  switch (name) {
    case 'id':  return m(1,0, 0,0, 0,0, 1,0)
    case 'h':   return m(S2,0, S2,0, S2,0, -S2,0)
    case 'x':   return m(0,0, 1,0, 1,0, 0,0)
    case 'y':   return m(0,0, 0,-1, 0,1, 0,0)
    case 'z':   return m(1,0, 0,0, 0,0, -1,0)
    case 's':   return m(1,0, 0,0, 0,0, 0,1)
    case 'sdg': case 'si': return m(1,0, 0,0, 0,0, 0,-1)
    case 't':   return m(1,0, 0,0, 0,0, S2,S2)
    case 'tdg': case 'ti': return m(1,0, 0,0, 0,0, S2,-S2)
    case 'rz':  return m(Math.cos(p / 2), -Math.sin(p / 2), 0,0, 0,0, Math.cos(p / 2), Math.sin(p / 2))
    case 'rx':  return m(c,0, 0,-s, 0,-s, c,0)
    case 'ry':  return m(c,0, -s,0, s,0, c,0)
    case 'u1': case 'p': return m(1,0, 0,0, 0,0, Math.cos(p), Math.sin(p))
    default: return fromGateTable(name, params)
  }
}

/**
 * Last resort: build the matrix out of `gates.ts` instead of hand-copying it.
 *
 * The switch above duplicates entries that already exist in the gate table, and
 * duplication is exactly how this drifted — `v`, `vi`, `u2`, `u3`, `r2`, `r4`,
 * `r8` and `srn` were all missing, which silently disqualified the contraction
 * attack from any circuit containing them. `u3` is the *universal* single-qubit
 * gate, so that was most circuits.
 *
 * Falling through to the table means a gate added to ket is contractible the day
 * it lands, with nothing here to remember to update.
 */
function fromGateTable(name: string, params: readonly number[] | undefined): Float64Array | null {
  const table = G as unknown as Record<string, unknown>
  // Aliases the serialiser emits that are not the table's own spelling.
  const alias: Record<string, string> = {
    srn: 'V', sx: 'V', srndg: 'Vi', sxdg: 'Vi', sdg: 'Si', tdg: 'Ti', p: 'U1',
  }
  const key = alias[name] ?? Object.keys(table).find(k => k.toLowerCase() === name.toLowerCase())
  if (key === undefined) return null
  const entry = table[key]
  const gate = typeof entry === 'function'
    ? (entry as (...a: number[]) => unknown)(...(params ?? []))
    : entry
  const rows = gate as readonly (readonly { re: number; im: number }[])[] | undefined
  if (!Array.isArray(rows) || rows.length !== 2) return null
  const out = new Float64Array(8)
  for (let r = 0; r < 2; r++) {
    for (let c2 = 0; c2 < 2; c2++) {
      const v = rows[r]![c2]
      if (v === undefined || typeof v.re !== 'number') return null
      out[(r * 2 + c2) * 2] = v.re
      out[(r * 2 + c2) * 2 + 1] = v.im
    }
  }
  return out
}

/**
 * Named two-qubit gates as a flat 4×4, row-major, |ab⟩ with `a` the high bit.
 *
 * Reconstructed from the serialized name and parameters rather than carried
 * through, because that is all `toJSON` records for a named gate. A gate whose
 * matrix cannot be rebuilt this way arrives as a `unitary` op instead, which
 * carries its entries.
 */
function twoMatrixOf(name: string, params: readonly number[] | undefined): Float64Array | null {
  const build = (g: Gate4x4): Float64Array => {
    const d = new Float64Array(32)
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      d[8 * r + 2 * c] = g[r]![c]!.re
      d[8 * r + 2 * c + 1] = g[r]![c]!.im
    }
    return d
  }
  const p0 = params?.[0] ?? 0
  switch (name) {
    case 'xx':     return build(G.Xx(p0))
    case 'yy':     return build(G.Yy(p0))
    case 'zz':     return build(G.Zz(p0))
    case 'xy':     return build(G.Xy(p0))
    case 'iswap':  return build(G.ISwap)
    case 'srswap': return build(G.SrSwap)
    case 'ms':     return build(G.Ms(p0, params?.[1] ?? 0))
    default:       return null
  }
}

/**
 * A general two-qubit gate as a rank-4 tensor over [outA, outB, inA, inB].
 *
 * The 4×4's row index is the pair of output bits and its column index the pair of
 * input bits, so this is a reshape and nothing more — but the bit order is the
 * part that is easy to get backwards, and getting it backwards still produces a
 * normalised state, just the wrong one.
 */
function twoQubitTensor(m: Float64Array): Float64Array {
  const d = new Float64Array(16 * 2)
  for (let oa = 0; oa < 2; oa++) for (let ob = 0; ob < 2; ob++)
    for (let ia = 0; ia < 2; ia++) for (let ib = 0; ib < 2; ib++) {
      const dst = ((oa * 2 + ob) * 2 + ia) * 2 + ib
      const src = (oa * 2 + ob) * 4 + (ia * 2 + ib)
      d[2 * dst] = m[2 * src]!
      d[2 * dst + 1] = m[2 * src + 1]!
    }
  return d
}

/** An explicit `unitary` op's matrix, flattened row-major. */
function flattenMatrix(rows: readonly (readonly [number, number])[][]): Float64Array {
  const n = rows.length
  const d = new Float64Array(n * n * 2)
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    d[2 * (r * n + c)] = rows[r]![c]![0]
    d[2 * (r * n + c) + 1] = rows[r]![c]![1]
  }
  return d
}

/** Gates that are diagonal in the computational basis. */
const DIAGONAL_1Q = new Set(['id', 'z', 's', 'sdg', 'si', 't', 'tdg', 'ti', 'rz', 'u1', 'p'])

/** diag(d00, d11) as a rank-1 tensor's data, or null when the gate is not diagonal. */
function diagonalOf(name: string, params: readonly number[] | undefined): Float64Array | null {
  if (!DIAGONAL_1Q.has(name)) return null
  const m = matrixOf(name, params)
  return m ? Float64Array.from([m[0]!, m[1]!, m[6]!, m[7]!]) : null
}

/**
 * A controlled-diagonal gate as a rank-2 tensor over [control, target].
 *
 * Controlling a diagonal gate leaves it diagonal, so a CZ or a CPHASE never
 * cuts either wire — the whole gate is 4 numbers on the two existing indices.
 */
function controlledDiagonalOf(name: string, params: readonly number[] | undefined): Float64Array | null {
  if (!name.startsWith('c')) return null
  const base = diagonalOf(name.slice(1), params)
  if (!base) return null
  return Float64Array.from([1, 0, 1, 0, base[0]!, base[1]!, base[2]!, base[3]!])
}

/**
 * Build the tensor network for ⟨bitstring|circuit|0…0⟩.
 *
 * Each wire carries an index that is renamed every time a gate touches it, so
 * index identity encodes the circuit's connectivity and nothing else has to.
 *
 * Diagonal gates are the exception, and they are why deep circuits are reachable
 * at all. A diagonal gate does not mix basis states, so it does not cut its
 * wires: it sits on the indices already there, shared with whatever produced and
 * consumes them. That makes those indices *hyper-indices*, held by three tensors
 * or more, and it stops a CZ layer from doubling the index count.
 */
export function circuitNetwork(circuit: Circuit, bitstring: string): Tensor[] {
  const n = circuit.qubits
  if (bitstring.length !== n) throw new TypeError(`bitstring '${bitstring}' must have ${n} characters`)
  if (!/^[01?]+$/.test(bitstring))
    throw new TypeError(`bitstring '${bitstring}' may contain only 0, 1 and ? (open)`)

  const tensors: Tensor[] = []
  const wire: string[] = []
  let fresh = 0
  const next = (): string => `i${fresh++}`

  // |0⟩ on every wire.
  for (let q = 0; q < n; q++) {
    wire[q] = next()
    tensors.push({ indices: [wire[q]!], data: Float64Array.from([1, 0, 0, 0]) })
  }

  const json = circuit.toJSON()
  for (const raw of json.ops) {
    const op = raw as Record<string, unknown>
    const kind = op['kind'] as string
    if (kind === 'barrier') continue
    if (kind === 'measure' || kind === 'reset' || kind === 'if')
      throw new TypeError('circuitNetwork requires a pure circuit — remove measure/reset/if ops')

    if (kind === 'single') {
      const q = op['q'] as number
      const meta = op['meta'] as { name: string; params?: number[] }
      const diag = diagonalOf(meta.name, meta.params)
      if (diag) { tensors.push({ indices: [wire[q]!], data: diag }); continue }
      const mat = matrixOf(meta.name, meta.params)
      if (!mat) throw new TypeError(`circuitNetwork does not support gate '${meta.name}'`)
      const out = next()
      tensors.push({ indices: [out, wire[q]!], data: mat })
      wire[q] = out
    } else if (kind === 'cnot' || (kind === 'controlled' && (op['meta'] as { name?: string })?.name === 'cx')) {
      const c = op['control'] as number, t = op['target'] as number
      const oc = next(), ot = next()
      tensors.push({ indices: [oc, ot, wire[c]!, wire[t]!], data: cnotTensor() })
      wire[c] = oc; wire[t] = ot
    } else if (kind === 'controlled') {
      const meta = op['meta'] as { name: string; params?: number[] }
      const c = op['control'] as number, t = op['target'] as number
      const cdiag = controlledDiagonalOf(meta.name, meta.params)
      if (cdiag) { tensors.push({ indices: [wire[c]!, wire[t]!], data: cdiag }); continue }
      const base = matrixOf(meta.name.replace(/^c/, ''), meta.params)
      if (!base) throw new TypeError(`circuitNetwork does not support gate '${meta.name}'`)
      const oc = next(), ot = next()
      tensors.push({ indices: [oc, ot, wire[c]!, wire[t]!], data: controlledTensor(base) })
      wire[c] = oc; wire[t] = ot
    } else if (kind === 'swap') {
      const a = op['a'] as number, b = op['b'] as number
      // A SWAP is just a relabelling — no tensor needed at all.
      const tmp = wire[a]!; wire[a] = wire[b]!; wire[b] = tmp
    } else if (kind === 'two') {
      const meta = op['meta'] as { name: string; params?: number[] }
      const mat = twoMatrixOf(meta.name, meta.params)
      if (!mat) throw new TypeError(`circuitNetwork does not support gate '${meta.name}'`)
      const a = op['a'] as number, b = op['b'] as number
      const oa = next(), ob = next()
      tensors.push({ indices: [oa, ob, wire[a]!, wire[b]!], data: twoQubitTensor(mat) })
      wire[a] = oa; wire[b] = ob
    } else if (kind === 'toffoli' || kind === 'cswap' || kind === 'csrswap') {
      // Shor's modular arithmetic is built from these, so without them the
      // contraction attack silently sat out the one family it matters most on.
      const qs = kind === 'toffoli'
        ? [op['c1'] as number, op['c2'] as number, op['target'] as number]
        : [op['control'] as number, op['a'] as number, op['b'] as number]
      const data3 = kind === 'toffoli' ? toffoliTensor()
                  : kind === 'cswap' ? cswapTensor() : csrSwapTensor()
      const outs = [next(), next(), next()]
      tensors.push({
        indices: [outs[0]!, outs[1]!, outs[2]!, wire[qs[0]!]!, wire[qs[1]!]!, wire[qs[2]!]!],
        data: data3,
      })
      for (let i = 0; i < 3; i++) wire[qs[i]!] = outs[i]!
    } else if (kind === 'unitary') {
      // The general escape hatch, and the one an RCS circuit's fSim arrives as.
      const qs = op['qubits'] as number[]
      const rows = op['matrix'] as (readonly [number, number])[][]
      if (qs.length === 1) {
        const out = next()
        tensors.push({ indices: [out, wire[qs[0]!]!], data: flattenMatrix(rows) })
        wire[qs[0]!] = out
      } else if (qs.length === 2) {
        const oa = next(), ob = next()
        tensors.push({
          indices: [oa, ob, wire[qs[0]!]!, wire[qs[1]!]!],
          data: twoQubitTensor(flattenMatrix(rows)),
        })
        wire[qs[0]!] = oa; wire[qs[1]!] = ob
      } else {
        throw new TypeError(`circuitNetwork supports 1- and 2-qubit unitary ops, got ${qs.length} qubits`)
      }
    } else {
      throw new TypeError(`circuitNetwork does not support op '${kind}'`)
    }
  }

  // Close each wire with ⟨0| or ⟨1|. A '?' leaves the wire open, so the
  // contraction returns a tensor over those qubits instead of a scalar — 2^k
  // amplitudes from one contraction rather than 2^k contractions.
  for (let q = 0; q < n; q++) {
    const ch = bitstring[q]
    if (ch === '?') continue
    const bit = ch === '1' ? 1 : 0
    tensors.push({ indices: [wire[q]!], data: Float64Array.from(bit ? [0, 0, 1, 0] : [1, 0, 0, 0]) })
  }

  // An open wire gets an identity cap, in ascending qubit order. A trailing
  // diagonal gate shares the wire's index rather than renaming it, so without
  // the cap an open wire can end on an index two tensors hold — which the
  // contraction would sum away instead of returning. The cap restores the
  // invariant the rest of the module relies on: an index held once is an output.
  for (let q = 0; q < n; q++) {
    if (bitstring[q] !== '?') continue
    const out = next()
    tensors.push({ indices: [out, wire[q]!], data: matrixOf('id', undefined)! })
    wire[q] = out
  }
  return tensors
}

/** CNOT as a rank-4 tensor with index order [outC, outT, inC, inT]. */
function cnotTensor(): Float64Array {
  const d = new Float64Array(16 * 2)
  for (let ic = 0; ic < 2; ic++) for (let it = 0; it < 2; it++) {
    const oc = ic, ot = ic === 1 ? it ^ 1 : it
    d[2 * (((oc * 2 + ot) * 2 + ic) * 2 + it)] = 1
  }
  return d
}

/**
 * A three-qubit permutation as a rank-6 tensor over [out…, in…].
 *
 * Toffoli and Fredkin are both permutations of the computational basis, so each
 * is a 0/1 tensor with exactly one entry per input — `move` says where that input
 * goes. Building them directly rather than decomposing into CNOTs keeps the
 * network one tensor per gate: a Shor circuit is tens of thousands of these, and
 * tripling the tensor count would cost the contraction planner far more than the
 * rank-6 tensor costs to hold.
 */
function permTensor3(move: (a: number, b: number, c: number) => [number, number, number]): Float64Array {
  const d = new Float64Array(64 * 2)
  for (let i0 = 0; i0 < 2; i0++) for (let i1 = 0; i1 < 2; i1++) for (let i2 = 0; i2 < 2; i2++) {
    const [o0, o1, o2] = move(i0, i1, i2)
    d[2 * (((((o0 * 2 + o1) * 2 + o2) * 2 + i0) * 2 + i1) * 2 + i2)] = 1
  }
  return d
}

/**
 * Fredkin's square root: a controlled-√SWAP, as a rank-6 tensor.
 *
 * Not a permutation, so it cannot go through {@link permTensor3}. With the
 * control set, |01⟩ and |10⟩ mix through [[s, is], [is, s]] with s = 1/√2, while
 * |00⟩ and |11⟩ — and everything with the control clear — pass through untouched.
 * Transcribed from the dense kernel rather than the prose, so the two cannot
 * disagree: out0 = s·in0 + i·s·in1, out1 = i·s·in0 + s·in1.
 */
function csrSwapTensor(): Float64Array {
  const d = new Float64Array(64 * 2)
  const at = (o0: number, o1: number, o2: number, i0: number, i1: number, i2: number): number =>
    2 * (((((o0 * 2 + o1) * 2 + o2) * 2 + i0) * 2 + i1) * 2 + i2)
  const s = Math.SQRT1_2
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
    d[at(0, a, b, 0, a, b)] = 1                       // control clear: identity
  }
  d[at(1, 0, 0, 1, 0, 0)] = 1                         // |c00> untouched
  d[at(1, 1, 1, 1, 1, 1)] = 1                         // |c11> untouched
  d[at(1, 0, 1, 1, 0, 1)] = s                         // <01|M|01> = s
  d[at(1, 1, 0, 1, 0, 1) + 1] = s                     // <10|M|01> = i s
  d[at(1, 0, 1, 1, 1, 0) + 1] = s                     // <01|M|10> = i s
  d[at(1, 1, 0, 1, 1, 0)] = s                         // <10|M|10> = s
  return d
}

/** Toffoli: flip the target when both controls are set. */
const toffoliTensor = (): Float64Array =>
  permTensor3((c1, c2, t) => [c1, c2, c1 === 1 && c2 === 1 ? t ^ 1 : t])

/** Fredkin: swap the two targets when the control is set. */
const cswapTensor = (): Float64Array =>
  permTensor3((c, a, b) => (c === 1 ? [c, b, a] : [c, a, b]))

/** A controlled-U as a rank-4 tensor, from U's 2×2 matrix. */
function controlledTensor(u: Float64Array): Float64Array {
  const d = new Float64Array(16 * 2)
  /** Flat offset for [outC, outT, inC, inT] in row-major order. */
  const at = (oc: number, ot: number, ic: number, itIn: number): number =>
    ((oc * 2 + ot) * 2 + ic) * 2 + itIn

  for (let it = 0; it < 2; it++) {
    d[2 * at(0, it, 0, it)] = 1                    // control 0: target untouched
    for (let ot = 0; ot < 2; ot++) {               // control 1: apply U
      const k = at(1, ot, 1, it)
      d[2 * k]     = u[2 * (ot * 2 + it)]!
      d[2 * k + 1] = u[2 * (ot * 2 + it) + 1]!
    }
  }
  return d
}

export interface AmplitudeResult {
  re: number
  im: number
  /** Widest intermediate tensor, in indices — memory goes as 2^width. */
  width: number
  /** Multiply-accumulate operations performed. */
  cost: number
  /** Tensors in the network before contraction. */
  tensors: number
}

/**
 * Compute the amplitude ⟨bitstring|circuit|0…0⟩ by tensor-network contraction.
 *
 * Reports the contraction width alongside the value, because the width is what
 * decides whether a circuit is reachable at all — a statevector is the special
 * case of a contraction whose width is the qubit count.
 */
export function amplitudeByContraction(
  circuit: Circuit,
  bitstring: string,
  { restarts = 24, seed = 1 }: { restarts?: number; seed?: number } = {},
): AmplitudeResult {
  const tensors = circuitNetwork(circuit, bitstring)
  // planBest, not planContraction: the hardness attack prices this network with
  // planBest, and on a 54-qubit RCS network that is width 12 against greedy's 14.
  // Executing the worse plan would mean the reported cost of an attack was one a
  // caller could not actually get by running it.
  const plan = planBest(tensors.map(t => t.indices), { restarts, seed })
  const out = contractNetwork(tensors, plan)
  if (out.indices.length !== 0) throw new Error(`contraction left ${out.indices.length} open indices`)
  return { re: out.data[0]!, im: out.data[1]!, width: plan.width, cost: plan.cost, tensors: tensors.length }
}

// ── Partition-based planning ──────────────────────────────────────────────────

/**
 * Plan a contraction by recursive balanced bisection.
 *
 * Greedy planning is local: it takes the cheapest step available now, which on a
 * structured network walks into a corner it cannot see coming. The partitioning
 * approach is global — split the network into two halves joined by as few indices
 * as possible, plan each half the same way, and contract the halves last.
 *
 * The cut *is* the width. Contract one half to a single tensor and its remaining
 * open indices are exactly the edges crossing to the other half, so minimising
 * the cut minimises the widest intermediate directly, rather than hoping a
 * sequence of locally-cheap steps adds up to a globally cheap one.
 *
 * In a circuit amplitude network every index joins exactly two tensors — a wire
 * segment between consecutive gates — so the hypergraph is a graph and bisection
 * is ordinary min-cut. Refinement is Fiduccia–Mattheyses: repeatedly move the
 * vertex with the best gain, keep the best prefix of moves, discard the rest.
 *
 * This is the shape of what `cotengra` does with KaHyPar; the multilevel
 * coarsening that makes KaHyPar strong on very large hypergraphs is not here.
 */
export function planContractionPartitioned(
  network: readonly (readonly string[])[],
  { restarts = 6, seed = 1, imbalance }: { restarts?: number; seed?: number; imbalance?: number } = {},
): ContractionPlan {
  const T = network.length
  if (T === 0) throw new Error('planContractionPartitioned: empty network')

  let rng = seed >>> 0 || 1
  const rand = (): number => {
    rng ^= rng << 13; rng ^= rng >>> 17; rng ^= rng << 5
    return (rng >>> 0) / 0x100000000
  }

  // Adjacency: how many indices each pair of tensors shares.
  const holders = new Map<string, number[]>()
  network.forEach((idx, i) => {
    for (const x of idx) {
      const list = holders.get(x)
      if (list) list.push(i)
      else holders.set(x, [i])
    }
  })
  const adj: Map<number, number>[] = Array.from({ length: T }, () => new Map())
  for (const list of holders.values()) {
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = list[i]!, b = list[j]!
        adj[a]!.set(b, (adj[a]!.get(b) ?? 0) + 1)
        adj[b]!.set(a, (adj[b]!.get(a) ?? 0) + 1)
      }
    }
  }

  /**
   * Split `verts` in two, minimising the widest resulting block boundary.
   *
   * Not the cut: contracting block A yields a tensor whose indices are the edges
   * from A to B *plus* the edges from A to everything outside this subproblem.
   * Minimising the cut alone ignores that inherited boundary.
   *
   * Multilevel, because flat refinement does not work here. Moving one vertex at
   * a time can only escape a local minimum if some single move improves things,
   * and on these graphs none does — the earlier flat version lost to greedy
   * everywhere. Coarsening collapses clusters into single vertices, so one move
   * at a coarse level relocates a whole region at once; the partition is then
   * projected back down and polished at every level on the way.
   */
  const bisect = (verts: number[], ext: Map<number, number>, tolerance: number): [number[], number[]] => {
    const m = verts.length
    if (m <= 1) return [verts, []]
    if (m === 2) return [[verts[0]!], [verts[1]!]]

    // Level 0: the subproblem itself, indexed locally.
    const pos = new Map<number, number>()
    verts.forEach((v, i) => pos.set(v, i))
    type Level = {
      count: number
      weight: number[]
      ext: number[]
      adj: Map<number, number>[]
      /** Where each vertex of the *finer* level went. Empty on level 0. */
      from: number[]
    }
    const base: Level = {
      count: m,
      weight: new Array(m).fill(1),
      ext: verts.map(v => ext.get(v) ?? 0),
      adj: verts.map(v => {
        const local = new Map<number, number>()
        for (const [nb, w] of adj[v]!) {
          const p = pos.get(nb)
          if (p !== undefined) local.set(p, w)
        }
        return local
      }),
      from: [],
    }

    // ── Coarsen: heavy-edge matching, until small or no longer shrinking ──────
    const levels: Level[] = [base]
    while (true) {
      const cur = levels[levels.length - 1]!
      if (cur.count <= COARSEST_SIZE) break

      const match = new Int32Array(cur.count).fill(-1)
      const order = Array.from({ length: cur.count }, (_, i) => i)
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1))
        const t = order[i]!; order[i] = order[j]!; order[j] = t
      }
      let groups = 0
      for (const v of order) {
        if (match[v] !== -1) continue
        let bestNb = -1, bestW = -1
        for (const [nb, w] of cur.adj[v]!) {
          if (match[nb] !== -1) continue
          if (w > bestW) { bestW = w; bestNb = nb }
        }
        match[v] = groups
        if (bestNb >= 0) match[bestNb] = groups
        groups++
      }
      if (groups >= cur.count) break                    // nothing merged; stop

      const next: Level = {
        count: groups,
        weight: new Array(groups).fill(0),
        ext: new Array(groups).fill(0),
        adj: Array.from({ length: groups }, () => new Map<number, number>()),
        from: Array.from(match),
      }
      for (let v = 0; v < cur.count; v++) {
        const g = match[v]!
        next.weight[g] = (next.weight[g] ?? 0) + cur.weight[v]!
        next.ext[g] = (next.ext[g] ?? 0) + cur.ext[v]!
      }
      for (let v = 0; v < cur.count; v++) {
        const g = match[v]!
        for (const [nb, w] of cur.adj[v]!) {
          const h = match[nb]!
          if (h === g) continue                          // internal to the group now
          next.adj[g]!.set(h, (next.adj[g]!.get(h) ?? 0) + w)
        }
      }
      levels.push(next)
    }

    // ── Score and refine, shared by every level ──────────────────────────────
    const totalWeight = base.weight.reduce((a, b) => a + b, 0)
    const low = totalWeight * (0.5 - tolerance / 2)
    const high = totalWeight * (0.5 + tolerance / 2)

    const evaluate = (lv: Level, side: Uint8Array): { cut: number; extA: number; extB: number } => {
      let cut = 0, extA = 0, extB = 0
      for (let i = 0; i < lv.count; i++) {
        if (side[i] === 0) extA += lv.ext[i]!; else extB += lv.ext[i]!
        for (const [nb, w] of lv.adj[i]!) if (nb > i && side[nb] !== side[i]) cut += w
      }
      return { cut, extA, extB }
    }
    const widest = (cut: number, extA: number, extB: number): number => Math.max(cut + extA, cut + extB)

    const refine = (lv: Level, side: Uint8Array): void => {
      for (let pass = 0; pass < FM_PASSES; pass++) {
        const locked = new Uint8Array(lv.count)
        let { cut, extA, extB } = evaluate(lv, side)
        let wA = 0
        for (let i = 0; i < lv.count; i++) if (side[i] === 0) wA += lv.weight[i]!
        const moves: number[] = []
        let running = 0, bestRunning = 0, bestStep = -1

        for (let step = 0; step < lv.count; step++) {
          let pickV = -1, pickGain = -Infinity, pickCut = 0, pickExtA = 0, pickExtB = 0
          for (let i = 0; i < lv.count; i++) {
            if (locked[i]) continue
            const nextWA = side[i] === 0 ? wA - lv.weight[i]! : wA + lv.weight[i]!
            if (nextWA < low || nextWA > high) continue
            let delta = 0
            for (const [nb, w] of lv.adj[i]!) delta += side[nb] === side[i] ? -w : w
            const e = lv.ext[i]!
            const nExtA = side[i] === 0 ? extA - e : extA + e
            const nExtB = side[i] === 0 ? extB + e : extB - e
            const gain = widest(cut, extA, extB) - widest(cut - delta, nExtA, nExtB)
            if (gain > pickGain) {
              pickGain = gain; pickV = i
              pickCut = cut - delta; pickExtA = nExtA; pickExtB = nExtB
            }
          }
          if (pickV < 0) break
          cut = pickCut; extA = pickExtA; extB = pickExtB
          wA += side[pickV] === 0 ? -lv.weight[pickV]! : lv.weight[pickV]!
          side[pickV] = side[pickV] === 0 ? 1 : 0
          locked[pickV] = 1
          moves.push(pickV)
          running += pickGain
          if (running > bestRunning) { bestRunning = running; bestStep = moves.length - 1 }
        }
        for (let k = moves.length - 1; k > bestStep; k--) {
          const v = moves[k]!
          side[v] = side[v] === 0 ? 1 : 0
        }
        if (bestRunning <= 0) break
      }
    }

    // ── Initial partition on the coarsest level, then uncoarsen ──────────────
    const coarsest = levels[levels.length - 1]!
    let bestSide: Uint8Array | null = null
    let bestScore = Infinity

    for (let attempt = 0; attempt < restarts; attempt++) {
      const side = new Uint8Array(coarsest.count).fill(1)
      // Grow one side from a random seed; locality is real structure here.
      const queue = [Math.floor(rand() * coarsest.count)]
      const seen = new Uint8Array(coarsest.count)
      seen[queue[0]!] = 1
      let wA = 0
      const half = totalWeight / 2
      while (queue.length > 0 && wA < half) {
        const v = queue.shift()!
        side[v] = 0
        wA += coarsest.weight[v]!
        for (const nb of coarsest.adj[v]!.keys()) if (!seen[nb]) { seen[nb] = 1; queue.push(nb) }
      }
      for (let i = 0; i < coarsest.count && wA < half; i++)
        if (side[i] === 1) { side[i] = 0; wA += coarsest.weight[i]! }

      refine(coarsest, side)

      // Project down, refining at each level.
      let cur = side
      for (let l = levels.length - 2; l >= 0; l--) {
        const finer = levels[l]!
        const mapping = levels[l + 1]!.from
        const projected = new Uint8Array(finer.count)
        for (let i = 0; i < finer.count; i++) projected[i] = cur[mapping[i]!]!
        refine(finer, projected)
        cur = projected
      }

      const { cut, extA, extB } = evaluate(base, cur)
      const sc = widest(cut, extA, extB)
      if (sc < bestScore) { bestScore = sc; bestSide = cur.slice() }
    }

    const A: number[] = [], B: number[] = []
    verts.forEach((v, i) => (bestSide![i] === 0 ? A : B).push(v))
    if (A.length === 0 || B.length === 0) return [verts.slice(0, m >> 1), verts.slice(m >> 1)]
    return [A, B]
  }

  const externalOf = (verts: number[]): Map<number, number> => {
    const inside = new Set(verts)
    const ext = new Map<number, number>()
    for (const v of verts) {
      let e = 0
      for (const [nb, w] of adj[v]!) if (!inside.has(nb)) e += w
      ext.set(v, e)
    }
    return ext
  }

  const buildPlan = (tolerance: number): ContractionStep[] => {
    const steps: ContractionStep[] = []
    const build = (verts: number[]): number => {
      if (verts.length === 1) return verts[0]!
      const [A, B] = bisect(verts, externalOf(verts), tolerance)
      const slotA = build(A)
      const slotB = build(B)
      steps.push([slotA, slotB])
      return slotA
    }
    build(Array.from({ length: T }, (_, i) => i))
    return steps
  }

  // Select across whole plans, the way the greedy planner does. Scoring only the
  // local cut inside `bisect` meant more search could return a worse plan.
  //
  // Imbalance is searched rather than assumed. A balanced split is the wrong
  // shape for these networks: the best order for a shallow circuit is a sweep,
  // which is maximally lopsided, and forcing halves actively fights it. Measured
  // across shallow layers, width falls from 10 to 8 to 6 as the tolerance opens
  // up, and the best value differs per circuit — so it is another axis of the
  // search, not a constant.
  const tolerances = imbalance !== undefined ? [imbalance] : IMBALANCE_LADDER
  let best: ContractionPlan | null = null
  for (const tol of tolerances) {
    for (let attempt = 0; attempt < Math.max(1, Math.ceil(restarts / 2)); attempt++) {
      const steps = buildPlan(tol)
      const { width, cost } = evaluatePlan(network, steps)
      if (!best || width < best.width || (width === best.width && cost < best.cost)) {
        best = { steps, width, cost }
      }
    }
  }
  return best!
}

/**
 * Replay a plan over index sets alone to score it.
 *
 * Costing a plan without touching tensor data is what lets planners be compared
 * and searched cheaply — the arithmetic is the part you are trying not to do yet.
 */
export function evaluatePlan(
  network: readonly (readonly string[])[],
  steps: readonly ContractionStep[],
): { width: number; cost: number } {
  const live: (Set<string> | undefined)[] = network.map(idx => new Set(idx))
  const held = holdCounts(network)
  let width = maxOf(live.map(s => s!.size))
  let cost = 0
  for (const [a, b] of steps) {
    const A = live[a]!, B = live[b]!
    let shared = 0
    for (const x of A) if (B.has(x)) shared++
    cost += 2 ** (A.size + B.size - shared)
    const merged = new Set<string>()
    for (const x of A) if (!B.has(x) || held.get(x)! > 2) merged.add(x)
    for (const x of B) if (!A.has(x) || held.get(x)! > 2) merged.add(x)
    for (const x of A) held.set(x, held.get(x)! - 1)
    for (const x of B) held.set(x, held.get(x)! - 1)
    for (const x of merged) held.set(x, held.get(x)! + 1)
    width = Math.max(width, merged.size)
    live[a] = merged
    live[b] = undefined
  }
  return { width, cost }
}

/**
 * Plan with every available strategy and keep the best.
 *
 * Which planner wins is a property of the network, not of the planner, so the
 * practical answer is to run both and score them — the same thing `cotengra`
 * does with its own set. Scoring is cheap because `evaluatePlan` replays a plan
 * over index sets without touching tensor data.
 *
 * Which one wins is decided by the network, and on the published families it is
 * usually bisection — by margins that dwarf the cost of trying both. Measured on
 * the light cone of a single-site observable, greedy against partitioned:
 *
 *   IBM heavy-hex,  5 steps   w 20 → 19     cost 5.9e8  → 2.6e8
 *   IBM heavy-hex, 10 steps   w 78 → 55     cost 2.9e33 → 1.4e24
 *   IBM heavy-hex, 20 steps   w 183 → 127   cost 2.3e74 → 4.1e57
 *   Google RCS,    depth 12   w 15 → 15     cost 3.6e6  → 5.0e6   (greedy wins)
 *   Google RCS,    depth 20   w 39 → 30     cost 1.8e16 → 1.9e13
 *
 * Greedy holds its own only on the shallowest RCS instance. Sparse degree-3
 * lattices are exactly the regime where multilevel coarsening earns its keep, and
 * greedy's local choice walks into a corner it cannot see coming.
 *
 * The cost is planning time, and it is not small: on the 20-step heavy-hex network
 * greedy plans in 349ms and bisection in 349s, a thousandfold. `planBest` pays
 * both. On a deep circuit that makes *planning* the dominant cost of pricing a
 * contraction, which is worth knowing before calling this in a loop.
 */
export function planBest(
  network: readonly (readonly string[])[],
  { restarts = 12, seed = 1 }: { restarts?: number; seed?: number } = {},
): ContractionPlan & { strategy: 'greedy' | 'partition' } {
  const greedy = planContraction(network, { restarts, seed })
  let best: ContractionPlan & { strategy: 'greedy' | 'partition' } = { ...greedy, strategy: 'greedy' }
  try {
    const part = planContractionPartitioned(network, { restarts: Math.max(2, restarts >> 1), seed })
    if (part.width < best.width || (part.width === best.width && part.cost < best.cost)) {
      best = { ...part, strategy: 'partition' }
    }
  } catch {
    // A planner that cannot handle a network simply does not compete.
  }
  return best
}

// ── Slicing ───────────────────────────────────────────────────────────────────

/**
 * A contraction split into independent slices.
 *
 * When a network's width is too large, no contraction order saves you: memory
 * goes as 2^width and that is that. Slicing sidesteps it by *fixing* a set of
 * indices rather than summing over them, contracting once per assignment and
 * adding the results. Each slice is a smaller contraction — the sliced indices
 * are gone from every intermediate — so k sliced indices trade 2^k repeats for
 * roughly 2^k less memory.
 *
 * That trade is what made petabyte-scale contractions run on ordinary hardware:
 * the repeats are completely independent, so they parallelise perfectly and can
 * be spread over as many machines as you have. The overhead is usually far below
 * the naive 2^k, because a slice is not merely a smaller version of the same
 * contraction — removing an index often collapses work that was being duplicated.
 */
export interface SlicedPlan {
  /** Contraction order for a single slice. */
  plan: ContractionPlan
  /** Indices held fixed rather than summed over. */
  sliced: string[]
  /** Independent contractions to perform: 2^sliced.length. */
  slices: number
  /** Width of one slice — this is what has to fit in memory. */
  width: number
  /** Total operations across all slices. */
  totalCost: number
  /** Total cost divided by the unsliced cost. 1 means slicing was free. */
  overhead: number
}

/** Remove a set of indices from every tensor's index list. */
const withoutIndices = (
  network: readonly (readonly string[])[], drop: ReadonlySet<string>,
): string[][] => network.map(idx => idx.filter(i => !drop.has(i)))


/** Coarsening stops once a level is this small; the initial partition is solved there. */
const COARSEST_SIZE = 24
/** Fiduccia–Mattheyses passes per level. */
const FM_PASSES = 3
/**
 * Balance tolerances to try when none is given.
 *
 * Circuit networks want lopsided splits — a shallow circuit's best contraction
 * is a sweep — but not degenerate ones, which stop reducing anything at all.
 */
const IMBALANCE_LADDER = [0.4, 0.7, 0.85, 0.95]

/** How many slice candidates to re-plan for at each step. */
const SLICE_CANDIDATES = 12
/**
 * Restarts used when scoring a candidate slice.
 *
 * Progress is judged by comparing two plans, so it is only as trustworthy as the
 * planner is repeatable. Too few restarts and the comparison measures planner
 * variance instead of the slice: candidates that genuinely help look like noise
 * and the search stops early. Planning is cheap enough now to buy that certainty.
 */
const SLICE_TRIAL_RESTARTS = 24


/**
 * Score a plan by width, how many intermediates reach that width, and cost.
 *
 * The count matters for slicing. A peak held by six intermediates does not fall
 * when one index leaves five of them, so width alone reports no progress and a
 * width-only rule stops before it has started. The count does move, and it is
 * what says whether slicing is converging.
 */
function planProfile(
  network: readonly (readonly string[])[],
  steps: readonly ContractionStep[],
): { width: number; peakCount: number; cost: number; peakIndices: Map<string, number> } {
  const live: (Set<string> | undefined)[] = network.map(idx => new Set(idx))
  const held = holdCounts(network)
  const sizes: number[] = []
  const holders: Set<string>[] = []
  let cost = 0

  for (const [a, b] of steps) {
    const A = live[a]!, B = live[b]!
    let shared = 0
    for (const x of A) if (B.has(x)) shared++
    cost += 2 ** (A.size + B.size - shared)
    const merged = new Set<string>()
    for (const x of A) if (!B.has(x) || held.get(x)! > 2) merged.add(x)
    for (const x of B) if (!A.has(x) || held.get(x)! > 2) merged.add(x)
    for (const x of A) held.set(x, held.get(x)! - 1)
    for (const x of B) held.set(x, held.get(x)! - 1)
    for (const x of merged) held.set(x, held.get(x)! + 1)
    live[a] = merged
    live[b] = undefined
    sizes.push(merged.size)
    holders.push(merged)
  }

  const width = Math.max(maxOf(sizes), maxOf(network.map(idx => idx.length)))
  let peakCount = 0
  const peakIndices = new Map<string, number>()
  sizes.forEach((sz, i) => {
    if (sz < width) return
    peakCount++
    for (const x of holders[i]!) peakIndices.set(x, (peakIndices.get(x) ?? 0) + 1)
  })
  return { width, peakCount, cost, peakIndices }
}

/**
 * Choose indices to slice until the contraction fits within `targetWidth`.
 *
 * Greedy on what actually matters: at each step, try every remaining index and
 * keep the one whose removal leaves the narrowest plan, breaking ties on cost.
 * Each candidate is scored by re-planning, which is affordable because planning
 * touches only index sets.
 */
export function sliceContraction(
  network: readonly (readonly string[])[],
  { targetWidth = 24, maxSliced = 12, restarts = 6, seed = 1 }: {
    targetWidth?: number; maxSliced?: number; restarts?: number; seed?: number
  } = {},
): SlicedPlan {
  const base = planContraction(network, { restarts: Math.max(restarts, SLICE_TRIAL_RESTARTS), seed })
  const sliced: string[] = []
  let current = base

  let profile = planProfile(network, current.steps)

  while (profile.width > targetWidth && sliced.length < maxSliced) {
    const drop = new Set(sliced)

    // Only indices carried by the widest intermediates can lower the peak, and
    // among those, the ones appearing in the *most* of them are the ones that
    // make progress. Ranking by that coverage is what turns a stalled search into
    // a converging one; scoring every index instead costs more than slicing saves.
    const ranked = [...profile.peakIndices.entries()]
      .filter(([name]) => !drop.has(name))
      .sort((a, b) => b[1] - a[1])
      .slice(0, SLICE_CANDIDATES)
      .map(([name]) => name)

    let bestIndex: string | null = null
    let bestPlan: ContractionPlan | null = null
    let bestProfile: ReturnType<typeof planProfile> | null = null

    for (const candidate of ranked) {
      drop.add(candidate)
      const reduced = withoutIndices(network, drop)
      const trial = planContraction(reduced, { restarts: SLICE_TRIAL_RESTARTS, seed })
      const prof = planProfile(reduced, trial.steps)
      drop.delete(candidate)
      // Lexicographic: narrower first, then fewer intermediates at that width,
      // then cheaper.
      const better = !bestProfile
        || prof.width < bestProfile.width
        || (prof.width === bestProfile.width && prof.peakCount < bestProfile.peakCount)
        || (prof.width === bestProfile.width && prof.peakCount === bestProfile.peakCount
            && trial.cost < bestPlan!.cost)
      if (better) { bestPlan = trial; bestIndex = candidate; bestProfile = prof }
    }

    // Accept real progress: a narrower contraction, or the same width held by
    // fewer intermediates, which is the step before it narrows. Anything else
    // would double the work for nothing, and a run of those would quietly turn
    // one contraction into millions.
    const progress = bestProfile && (
      bestProfile.width < profile.width ||
      (bestProfile.width === profile.width && bestProfile.peakCount < profile.peakCount))
    if (!bestIndex || !bestPlan || !bestProfile || !progress) break

    sliced.push(bestIndex)
    current = bestPlan
    profile = bestProfile
  }

  const slices = 2 ** sliced.length
  const totalCost = current.cost * slices
  return { plan: current, sliced, slices, width: current.width, totalCost, overhead: totalCost / base.cost }
}

/** Fix `index` to `value` in a tensor, dropping that axis. */
export function projectTensor(t: Tensor, fixed: ReadonlyMap<string, number>): Tensor {
  const keep = t.indices.filter(i => !fixed.has(i))
  if (keep.length === t.indices.length) return t

  const strideOf = (name: string): number => 1 << (t.indices.length - 1 - t.indices.indexOf(name))
  let base = 0
  for (const [name, value] of fixed) if (t.indices.includes(name) && value === 1) base += strideOf(name)
  const keepStrides = keep.map(strideOf)

  const out = new Float64Array((1 << keep.length) * 2)
  for (let o = 0; o < 1 << keep.length; o++) {
    let src = base
    for (let k = 0; k < keep.length; k++) if ((o >> (keep.length - 1 - k)) & 1) src += keepStrides[k]!
    out[2 * o] = t.data[2 * src]!
    out[2 * o + 1] = t.data[2 * src + 1]!
  }
  return { indices: keep, data: out }
}

/**
 * Contract slices `[lo, hi)` and sum them.
 *
 * This is the unit of distributable work. A slice reads only the network and the
 * plan and writes only its own partial sum, so a range of them is a self-contained
 * job: ship the network and two integers, get back one tensor. That is the whole
 * reason slicing scales — the jobs never talk to each other, and the reduction at
 * the end is addition.
 *
 * Splitting a contraction across ranges changes the *order* the partial sums are
 * added in, which moves the result in the last bits or two. Any given partition is
 * deterministic; two different partitions are not bitwise equal. Compare sliced
 * results with a tolerance, never with `toBe`.
 */
export function contractSliceRange(
  tensors: readonly Tensor[], sliced: SlicedPlan, lo: number, hi: number,
): Tensor {
  if (!Number.isInteger(lo) || !Number.isInteger(hi)) throw new RangeError('contractSliceRange: lo and hi must be integers')
  if (lo < 0 || hi > sliced.slices) throw new RangeError(`contractSliceRange: [${lo}, ${hi}) is outside [0, ${sliced.slices})`)

  let acc: Tensor | null = null

  for (let s = lo; s < hi; s++) {
    const fixed = new Map<string, number>()
    sliced.sliced.forEach((name, k) => fixed.set(name, (s >> k) & 1))
    const projected = tensors.map(t => projectTensor(t, fixed))
    const part = contractNetwork(projected, sliced.plan)
    if (!acc) acc = { indices: part.indices, data: new Float64Array(part.data.length) }
    for (let i = 0; i < part.data.length; i++) acc.data[i]! += part.data[i]!
  }

  if (!acc) throw new RangeError(`contractSliceRange: empty range [${lo}, ${hi})`)
  return acc
}

/**
 * Contract a network slice by slice and sum the results.
 *
 * Each slice is independent, so this loop is the sequential form of what would
 * normally be distributed: nothing is shared between iterations but the running
 */
export function contractSliced(tensors: readonly Tensor[], sliced: SlicedPlan): Tensor {
  return contractSliceRange(tensors, sliced, 0, sliced.slices)
}

/** Amplitude via a sliced contraction, so width can be held under a memory budget. */
export function amplitudeBySlicedContraction(
  circuit: Circuit,
  bitstring: string,
  { targetWidth = 24, maxSliced = 24, restarts = 6, seed = 1 }: {
    targetWidth?: number; maxSliced?: number; restarts?: number; seed?: number
  } = {},
): AmplitudeResult & { slices: number; sliced: string[]; overhead: number } {
  const tensors = circuitNetwork(circuit, bitstring)
  const spec = sliceContraction(tensors.map(t => t.indices), { targetWidth, maxSliced, restarts, seed })
  const out = contractSliced(tensors, spec)
  if (out.indices.length !== 0) throw new Error(`sliced contraction left ${out.indices.length} open indices`)
  return {
    re: out.data[0]!, im: out.data[1]!,
    width: spec.width, cost: spec.totalCost, tensors: tensors.length,
    slices: spec.slices, sliced: spec.sliced, overhead: spec.overhead,
  }
}

/** A batch of amplitudes: every assignment of the open qubits, in one contraction. */
export interface AmplitudeBatch {
  /** Qubits left open, ascending. */
  open: number[]
  /** 2^open.length complex amplitudes, re/im interleaved, open[0] most significant. */
  data: Float64Array
  width: number
  cost: number
  /** Look one amplitude up by the values of the open qubits, in `open` order. */
  at(bits: readonly number[]): { re: number; im: number }
}

/**
 * Contract once, returning every amplitude consistent with a partial bitstring.
 *
 * Write `?` for a qubit to leave open. Its wire is not closed off, so the
 * contraction ends on a tensor over those qubits rather than a scalar: 2^k
 * amplitudes for roughly the price of one, instead of 2^k separate contractions.
 *
 * This is what makes contraction usable for sampling rather than for spot checks.
 * The open indices widen every intermediate that carries them, so the batch size
 * trades directly against contraction width — which is the same currency slicing
 * spends, and the reason the two are usually used together.
 */
export function amplitudeBatchByContraction(
  circuit: Circuit,
  pattern: string,
  { restarts = 32, seed = 1 }: { restarts?: number; seed?: number } = {},
): AmplitudeBatch {
  const tensors = circuitNetwork(circuit, pattern)
  const plan = planContraction(tensors.map(t => t.indices), { restarts, seed })
  const out = contractNetwork(tensors, plan)

  const open: number[] = []
  for (let q = 0; q < circuit.qubits; q++) if (pattern[q] === '?') open.push(q)
  if (out.indices.length !== open.length)
    throw new Error(`contraction left ${out.indices.length} open indices, expected ${open.length}`)

  // The contracted tensor's index order follows the network, not the qubit
  // order, so permute it into ascending-qubit order before handing it back.
  const wanted = open.map(q => openIndexFor(tensors, circuit, pattern, q))
  const ordered = permuteTensor(out, wanted.filter(x => out.indices.includes(x)))

  return {
    open,
    data: ordered.data,
    width: plan.width,
    cost: plan.cost,
    at(bits: readonly number[]) {
      if (bits.length !== open.length) throw new TypeError(`expected ${open.length} bit values`)
      let k = 0
      for (const b of bits) k = (k << 1) | (b ? 1 : 0)
      return { re: ordered.data[2 * k]!, im: ordered.data[2 * k + 1]! }
    },
  }
}

/** The dangling index belonging to an open qubit, found from the built network. */
function openIndexFor(tensors: readonly Tensor[], circuit: Circuit, pattern: string, qubit: number): string {
  // An open wire's final index appears exactly once across the whole network.
  const seen = new Map<string, number>()
  for (const t of tensors) for (const i of t.indices) seen.set(i, (seen.get(i) ?? 0) + 1)
  const dangling = [...seen.entries()].filter(([, c]) => c === 1).map(([i]) => i)

  // Rebuild the wire assignment to know which dangling index is which qubit.
  const open: number[] = []
  for (let q = 0; q < circuit.qubits; q++) if (pattern[q] === '?') open.push(q)
  const rank = open.indexOf(qubit)
  // Wires are created in qubit order and renamed in gate order, so the dangling
  // indices sort by their numeric suffix in the same order the wires were last
  // touched — not by qubit. Match by position in the network's own tensor order.
  const byLastUse = dangling.sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)))
  return byLastUse[rank]!
}
