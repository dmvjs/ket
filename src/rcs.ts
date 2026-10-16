/**
 * Random circuit sampling, and cross-entropy benchmarking of the result.
 *
 * RCS is the task quantum-advantage experiments are built on, and XEB is how they
 * are scored. Both are short to state. A random circuit on a 2-D grid produces an
 * output distribution that is speckled rather than flat — heavy on a few
 * bitstrings, near-zero on most — and a machine that really ran the circuit will
 * sample the heavy ones more often than chance. Linear XEB measures exactly that:
 *
 *     F = 2^n · mean over samples of p_ideal(x) − 1
 *
 * which is 1 for perfect sampling, 0 for uniform noise, and — this is the part
 * that makes it useful — approximately the fidelity for anything in between.
 *
 * The catch is `p_ideal`. Scoring a sample requires the ideal probability of that
 * specific bitstring, from a classical simulation of the very circuit that is
 * supposed to be hard to simulate. That is the whole game, and it is why this
 * module sits on top of the contraction backend rather than the statevector: a
 * contraction computes one amplitude at a time at a cost set by treewidth, so it
 * keeps working at widths where no statevector exists.
 *
 * The plan is built once and reused across samples. Every bitstring gives the same
 * network apart from the rank-1 tensors that close the wires, so the contraction
 * order — which is the expensive part to find — is computed once and replayed.
 */
import { Circuit } from './circuit.js'
import type { Complex } from './complex.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'
import { makePrng } from './prng.js'
import {
  circuitNetwork, contractNetwork, contractSliced, planContraction, sliceContraction,
  type ContractionPlan, type SlicedPlan,
} from './tensor-network.js'

const S2 = Math.SQRT1_2

// ── The Sycamore single-qubit set ─────────────────────────────────────────────
//
// √X, √Y and √W, where W = (X+Y)/√2. Written as explicit matrices and applied as
// `unitary` ops rather than named gates, because √Y and √W have no name in ket
// and a named gate every backend does not know would restrict which backends can
// score the circuit — which is the one thing this module cannot afford.

/** √X — a quarter turn about x. */
export const SQRT_X: Gate2x2 = [
  [{ re: 0.5, im: 0.5 }, { re: 0.5, im: -0.5 }],
  [{ re: 0.5, im: -0.5 }, { re: 0.5, im: 0.5 }],
]

/** √Y — a quarter turn about y. */
export const SQRT_Y: Gate2x2 = [
  [{ re: 0.5, im: 0.5 }, { re: -0.5, im: -0.5 }],
  [{ re: 0.5, im: 0.5 }, { re: 0.5, im: 0.5 }],
]

/**
 * √W, with W = (X+Y)/√2.
 *
 * W squares to the identity, so √W is `e^{iπ/4}(I − iW)/√2`. The global phase is
 * unobservable but is kept so that the matrix is literally the square root, which
 * is what the unitarity tests check.
 */
export const SQRT_W: Gate2x2 = [
  [{ re: 0.5, im: 0.5 }, { re: 0, im: -S2 }],
  [{ re: S2, im: 0 }, { re: 0.5, im: 0.5 }],
]

/** The three-gate set a cycle draws from, in the order the tests expect. */
export const SYCAMORE_1Q: readonly Gate2x2[] = [SQRT_X, SQRT_Y, SQRT_W]

/**
 * The fSim entangler.
 *
 * ```
 *   | 1      0        0     0     |
 *   | 0    cos θ   −i sin θ 0     |
 *   | 0  −i sin θ   cos θ   0     |
 *   | 0      0        0   e^{−iφ} |
 * ```
 *
 * Sycamore's calibrated values were θ ≈ π/2 (a full iSWAP-like swap angle) and
 * φ ≈ π/6 (a conditional phase), which are the defaults here.
 */
export function fsim(theta: number, phi: number): Gate4x4 {
  const c = Math.cos(theta), s = Math.sin(theta)
  const z = { re: 0, im: 0 }
  return [
    [{ re: 1, im: 0 }, z, z, z],
    [z, { re: c, im: 0 }, { re: 0, im: -s }, z],
    [z, { re: 0, im: -s }, { re: c, im: 0 }, z],
    [z, z, z, { re: Math.cos(phi), im: -Math.sin(phi) }],
  ]
}

// ── Coupler patterns ──────────────────────────────────────────────────────────

/** One layer's worth of qubit pairs. */
export type Coupling = readonly (readonly [number, number])[]

/**
 * The four disjoint bond sets of a rectangular grid.
 *
 * A two-qubit layer must be a matching — no qubit in two gates at once — so the
 * bonds are split by orientation and parity. `A`/`B` are horizontal, `C`/`D`
 * vertical.
 */
export function gridCouplings(rows: number, cols: number): Record<'A' | 'B' | 'C' | 'D', Coupling> {
  const q = (r: number, c: number): number => r * cols + c
  const out = { A: [] as [number, number][], B: [] as [number, number][], C: [] as [number, number][], D: [] as [number, number][] }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c + 1 < cols; c++) (c % 2 === 0 ? out.A : out.B).push([q(r, c), q(r, c + 1)])
  }
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r + 1 < rows; r++) (r % 2 === 0 ? out.C : out.D).push([q(r, c), q(r + 1, c)])
  }
  return out
}

/**
 * Sycamore's cycle order.
 *
 * Not `ABCD` repeated: the published sequence interleaves so that consecutive
 * cycles act on different bonds, which is what makes entanglement spread as fast
 * as it does and is therefore load-bearing for the circuit being hard.
 */
export const SYCAMORE_SEQUENCE = ['A', 'B', 'C', 'D', 'C', 'D', 'A', 'B'] as const

export interface RcsOptions {
  rows: number
  cols: number
  /** Cycles. Each is one random single-qubit layer followed by one coupler layer. */
  depth: number
  seed?: number
  /** fSim swap angle. Default π/2. */
  theta?: number
  /** fSim conditional phase. Default π/6. */
  phi?: number
  /** Coupler pattern order. Default {@link SYCAMORE_SEQUENCE}. */
  sequence?: readonly ('A' | 'B' | 'C' | 'D')[]
  /** Append a final single-qubit layer, as the reference experiments do. Default true. */
  finalSingleLayer?: boolean
}

/**
 * A Sycamore-shaped random circuit.
 *
 * Each cycle gives every qubit a single-qubit gate drawn at random from
 * {√X, √Y, √W} — never the same one that qubit had last cycle, which is what
 * stops a qubit's rotations collapsing into a single smaller one — and then
 * applies fSim across one coupler pattern.
 */
export function rcsCircuit(options: RcsOptions): Circuit {
  const { rows, cols, depth, seed = 1, theta = Math.PI / 2, phi = Math.PI / 6,
    sequence = SYCAMORE_SEQUENCE, finalSingleLayer = true } = options
  for (const [name, v] of [['rows', rows], ['cols', cols], ['depth', depth]] as const) {
    if (!Number.isInteger(v) || v < 1) throw new RangeError(`rcsCircuit: ${name} must be a positive integer, got ${v}`)
  }
  if (sequence.length === 0) throw new RangeError('rcsCircuit: sequence must not be empty')

  const n = rows * cols
  const rand = makePrng(seed)
  const couplings = gridCouplings(rows, cols)
  const entangler = fsim(theta, phi)
  const asRows = (g: Gate4x4 | Gate2x2): Complex[][] => (g as Complex[][]).map(r => [...r])

  /** Last gate index applied to each qubit, so the next one can differ. */
  const previous = new Int8Array(n).fill(-1)
  const singleLayer = (c: Circuit): Circuit => {
    let out = c
    for (let q = 0; q < n; q++) {
      // Choose among the two that are not the previous gate. With none applied
      // yet, all three are available.
      const choices = [0, 1, 2].filter(i => i !== previous[q])
      const pick = choices[Math.min(choices.length - 1, Math.floor(rand() * choices.length))]!
      previous[q] = pick
      out = out.unitary(asRows(SYCAMORE_1Q[pick]!), q)
    }
    return out
  }

  let c = new Circuit(n)
  for (let d = 0; d < depth; d++) {
    c = singleLayer(c)
    for (const [a, b] of couplings[sequence[d % sequence.length]!]) {
      c = c.unitary(asRows(entangler), a, b)
    }
  }
  if (finalSingleLayer) c = singleLayer(c)
  return c
}

// ── Heavy-hex, and the Trotter circuits run on it ─────────────────────────────

/**
 * The 127-qubit heavy-hex coupling map — IBM's Eagle topology.
 *
 * Seven rows joined by four connector qubits apiece: 14, 15, 15, 15, 15, 15, 14
 * along the rows, with the connector groups sitting between them and attaching
 * every fourth qubit, alternating offset 0 and 2 row to row. That gives 127
 * qubits, 144 edges and a maximum degree of 3 — the numbers the layout is checked
 * against, since a coupling map that is subtly wrong would silently change every
 * hardness verdict computed on it.
 *
 * The labelling follows the standard published layout. It does not have to match
 * any particular calibration snapshot: hardness is a property of the graph, and
 * relabelling qubits cannot change treewidth, T-count or light-cone size.
 */
export function heavyHex127(): Coupling {
  const rows = [
    [0, 13], [18, 32], [37, 51], [56, 70], [75, 89], [94, 108], [113, 126],
  ] as const
  /**
   * Connector groups, with the attachment offset into each row they join.
   *
   * The offsets are per side, not per group. They agree for five of the six
   * groups, which makes a single shared offset look right — but the bottom row
   * holds 14 qubits rather than 15 and sits shifted, so its attachments land at
   * offset 1. Sharing one offset there runs the last connector off the end of the
   * row and invents a qubit 127.
   */
  const links = [
    { base: 14, above: 0, below: 1, up: 0, down: 0 },
    { base: 33, above: 1, below: 2, up: 2, down: 2 },
    { base: 52, above: 2, below: 3, up: 0, down: 0 },
    { base: 71, above: 3, below: 4, up: 2, down: 2 },
    { base: 90, above: 4, below: 5, up: 0, down: 0 },
    { base: 109, above: 5, below: 6, up: 2, down: 1 },
  ] as const

  const edges: [number, number][] = []
  for (const [lo, hi] of rows) for (let q = lo; q < hi; q++) edges.push([q, q + 1])
  for (const { base, above, below, up, down } of links) {
    for (let k = 0; k < 4; k++) {
      const c = base + k
      const a = rows[above]![0] + up + 4 * k
      const b = rows[below]![0] + down + 4 * k
      if (a > rows[above]![1] || b > rows[below]![1]) {
        throw new Error(`heavyHex127: connector ${c} attaches outside its row (${a}, ${b})`)
      }
      edges.push([a, c], [c, b])
    }
  }
  return edges
}

/** Partition edges into matchings, so each becomes one hardware layer. */
export function edgeColouring(edges: Coupling): Coupling[] {
  const layers: [number, number][][] = []
  const used: Set<number>[] = []
  for (const [a, b] of edges) {
    let placed = false
    for (let i = 0; i < layers.length; i++) {
      if (!used[i]!.has(a) && !used[i]!.has(b)) {
        layers[i]!.push([a, b]); used[i]!.add(a); used[i]!.add(b); placed = true; break
      }
    }
    if (!placed) { layers.push([[a, b]]); used.push(new Set([a, b])) }
  }
  return layers
}

export interface TrotterIsingOptions {
  qubits: number
  coupling: Coupling
  /** Trotter steps. */
  depth: number
  /** ZZ angle. The utility experiment fixed this at −π/2, which is Clifford. */
  thetaJ?: number
  /** Transverse-field angle. The knob that was swept; 0 and π/2 are Clifford. */
  thetaH: number
}

/**
 * Trotterized transverse-field Ising dynamics on a given coupling map.
 *
 * One step is `Rzz(θ_J)` across every edge — in matching layers, as hardware must
 * run it — followed by `Rx(θ_H)` on every qubit. This is the circuit family of
 * the 127-qubit "utility" experiments.
 *
 * `Rzz` is emitted as CNOT · Rz · CNOT rather than as a named two-qubit gate,
 * because that decomposition is what every backend here accepts — including the
 * Pauli-path propagation, which has no `zz`.
 */
export function trotterIsing(
  { qubits, coupling, depth, thetaJ = -Math.PI / 2, thetaH }: TrotterIsingOptions,
): Circuit {
  if (!Number.isInteger(qubits) || qubits < 1) throw new RangeError(`trotterIsing: qubits must be a positive integer, got ${qubits}`)
  if (!Number.isInteger(depth) || depth < 1) throw new RangeError(`trotterIsing: depth must be a positive integer, got ${depth}`)
  for (const [a, b] of coupling) {
    if (a < 0 || b < 0 || a >= qubits || b >= qubits) throw new RangeError(`trotterIsing: edge ${a}-${b} outside ${qubits} qubits`)
  }
  const layers = edgeColouring(coupling)
  let c = new Circuit(qubits)
  for (let d = 0; d < depth; d++) {
    for (const layer of layers) {
      for (const [a, b] of layer) c = c.cnot(a, b).rz(thetaJ, b).cnot(a, b)
    }
    for (let q = 0; q < qubits; q++) c = c.rx(thetaH, q)
  }
  return c
}

// ── Scoring ───────────────────────────────────────────────────────────────────

export interface XebResult {
  /** Linear XEB: 2^n·mean(p) − 1. 1 is ideal sampling, 0 is uniform noise. */
  fidelity: number
  /** Standard error of `fidelity`, from the spread of the sampled probabilities. */
  stderr: number
  /** Samples scored. */
  samples: number
  qubits: number
  /** Mean of 2^n·p over the samples — `fidelity` plus one. */
  meanScaled: number
}

/**
 * Linear XEB from the ideal probabilities of a set of samples.
 *
 * `probs[i]` is the ideal probability of the i-th sampled bitstring, so a
 * bitstring sampled twice appears twice. Repeats carry real information here —
 * they are how a peaked distribution shows itself — so deduplicating the input
 * would bias the estimate.
 */
export function linearXeb(qubits: number, probs: readonly number[]): XebResult {
  if (probs.length === 0) throw new RangeError('linearXeb: no samples to score')
  const d = 2 ** qubits
  let sum = 0, sumSq = 0
  for (const p of probs) { const x = d * p; sum += x; sumSq += x * x }
  const m = probs.length
  const meanScaled = sum / m
  // Sample variance of d·p, then the standard error of its mean.
  const variance = Math.max(0, sumSq / m - meanScaled * meanScaled)
  return {
    fidelity: meanScaled - 1,
    stderr: Math.sqrt(variance / m),
    samples: m,
    qubits,
    meanScaled,
  }
}

/**
 * How speckled a distribution is, against the Porter-Thomas prediction.
 *
 * A random circuit deep enough to scramble produces probabilities distributed as
 * `P(d·p = x) = e^{-x}`, whose moments are 1, 2 and 6. Checking them is the
 * cheapest way to tell whether a circuit is actually in the hard regime or is
 * still shallow enough to be structured — a claim about XEB on a circuit that
 * fails this is a claim about the wrong thing.
 *
 * The second moment misses in both directions, which is what makes it worth
 * checking rather than assuming: a uniform distribution sits at 1, and a circuit
 * too shallow to have spread its amplitude overshoots 2 rather than falling short
 * of it, because the weight is still concentrated on a few bitstrings.
 *
 * Pass the whole distribution. `participation` is `1/Σp²` and only means "how
 * many bitstrings carry the weight" when nothing has been left out.
 */
export function porterThomasMoments(qubits: number, probs: ArrayLike<number>): {
  first: number; second: number; third: number; participation: number
} {
  const d = 2 ** qubits
  let m1 = 0, m2 = 0, m3 = 0
  for (let i = 0; i < probs.length; i++) {
    const x = d * probs[i]!
    m1 += x; m2 += x * x; m3 += x * x * x
  }
  const k = probs.length
  return {
    first: m1 / k,
    second: m2 / k,
    third: m3 / k,
    // Inverse participation ratio: how many bitstrings carry the weight.
    participation: 1 / (m2 / k / d),
  }
}

/** Probability of one bitstring, however the caller wants it computed. */
export type ProbabilityOracle = (bitstring: string) => number | Promise<number>

export interface OracleOptions {
  /** Restarts for the contraction-order search. Default 24. */
  restarts?: number
  seed?: number
  /**
   * Slice until the contraction fits this width, trading repeats for memory.
   * Left undefined, the plan is used as found — set it when the circuit is wide
   * enough that the unsliced contraction will not fit.
   */
  targetWidth?: number
  maxSliced?: number
}

/**
 * A reusable probability oracle backed by tensor-network contraction.
 *
 * Planning is the expensive half and it does not depend on the bitstring, so it
 * happens once here and every later call replays it. That is what makes scoring
 * thousands of samples affordable, and it is why this returns a function rather
 * than taking the samples up front.
 *
 * `plan.width` is exposed so a caller can see what the contraction will cost
 * before running it — memory goes as 2^width.
 */
export function contractionOracle(circuit: Circuit, options: OracleOptions = {}):
  ProbabilityOracle & { width: number; slices: number } {
  const { restarts = 24, seed = 1, targetWidth, maxSliced = 24 } = options
  const probe = circuitNetwork(circuit, '0'.repeat(circuit.qubits))
  const indices = probe.map(t => t.indices)

  let plan: ContractionPlan | null = null
  let sliced: SlicedPlan | null = null
  if (targetWidth === undefined) plan = planContraction(indices, { restarts, seed })
  else sliced = sliceContraction(indices, { targetWidth, maxSliced, restarts, seed })

  const oracle = (bitstring: string): number => {
    const tensors = circuitNetwork(circuit, bitstring)
    const out = sliced ? contractSliced(tensors, sliced) : contractNetwork(tensors, plan!)
    if (out.indices.length !== 0) throw new Error(`contractionOracle: ${out.indices.length} open indices left`)
    const re = out.data[0]!, im = out.data[1]!
    return re * re + im * im
  }
  return Object.assign(oracle, {
    width: sliced ? sliced.width : plan!.width,
    slices: sliced ? sliced.slices : 1,
  })
}

/**
 * An oracle from the exact statevector — one dense solve, then every lookup is free.
 *
 * The right choice whenever the circuit fits, and much faster than contraction
 * there. It stops sooner than the statevector backend itself does, though:
 * `exactProbs()` returns a record keyed by bitstring, so this materialises 2^n
 * string keys and runs out of room around 20-24 qubits — well before the
 * statevector runs out of memory. Past that, use {@link contractionOracle}.
 */
export function statevectorOracle(circuit: Circuit): ProbabilityOracle {
  const probs = circuit.exactProbs()
  return (bitstring: string) => probs[bitstring] ?? 0
}

/**
 * Score samples against a circuit.
 *
 * Defaults to contraction, which is the only oracle that keeps working once the
 * circuit is too wide to hold — pass `statevectorOracle(circuit)` when it is not,
 * because for a small circuit one dense solve beats thousands of contractions.
 *
 * Distinct bitstrings are scored once each; repeats share an ideal probability.
 */
export async function xebFromSamples(
  circuit: Circuit,
  samples: readonly string[],
  options: { oracle?: ProbabilityOracle } & OracleOptions = {},
): Promise<XebResult> {
  if (samples.length === 0) throw new RangeError('xebFromSamples: no samples to score')
  for (const s of samples) {
    if (s.length !== circuit.qubits) {
      throw new RangeError(`xebFromSamples: sample '${s}' must have ${circuit.qubits} characters`)
    }
  }
  const { oracle, ...oracleOptions } = options

  // Distinct bitstrings only: repeats have the same ideal probability, and at a
  // contraction apiece that is the difference between scoring 1000 samples and
  // scoring however many of them happened to be distinct.
  const distinct = [...new Set(samples)]
  const cache = new Map<string, number>()
  const score = oracle ?? contractionOracle(circuit, oracleOptions)
  for (const s of distinct) cache.set(s, await score(s))

  return linearXeb(circuit.qubits, samples.map(s => cache.get(s)!))
}

/**
 * Replace a fraction of samples with uniform-random ones.
 *
 * The standard depolarizing picture of a noisy sampler: with probability `1 − f`
 * the run decoheres completely and returns noise, otherwise it returns an ideal
 * sample. XEB should then read back `f`, and that is the property that lets a
 * fidelity be quoted from samples at all — which is worth being able to
 * demonstrate rather than assert.
 */
export function mixWithUniform(
  samples: readonly string[], fidelity: number, qubits: number, seed = 1,
): string[] {
  if (fidelity < 0 || fidelity > 1) throw new RangeError(`mixWithUniform: fidelity must be in [0, 1], got ${fidelity}`)
  const rand = makePrng(seed)
  return samples.map(s => {
    if (rand() < fidelity) return s
    let out = ''
    for (let q = 0; q < qubits; q++) out += rand() < 0.5 ? '0' : '1'
    return out
  })
}
