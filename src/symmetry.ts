/**
 * U(1) symmetry: when a circuit cannot change how many excitations it has.
 *
 * Every other attack in this library exploits something about the circuit's
 * *dynamics* — how entangled it gets, how much magic it uses, how its gates
 * connect. This one exploits a conservation law. If no gate can move an
 * excitation into or out of existence, the state never leaves the Hamming-weight
 * sector it started in, and the reachable space is C(n, k) rather than 2^n.
 *
 * At half filling that buys a factor of about √n and is barely worth having. Away
 * from half filling it is enormous: 50 spin-orbitals holding 10 electrons is
 * C(50,10) ≈ 1.0e10 against 2^50 ≈ 1.1e15, five orders of magnitude, and that is
 * the regime quantum chemistry actually runs in. Particle-conserving ansätze are
 * standard there, and general-purpose simulators mostly do not exploit them —
 * the reduction is left to domain-specific chemistry codes.
 *
 * **Detected numerically, never from gate names.** A gate preserves weight when
 * its matrix has no entry connecting basis states of different weight, which is a
 * property of the numbers. Name-based detection is how this library previously
 * managed to miss that `Rz(−π/2)` is Clifford, and the same mistake here would
 * miss that `fSim` conserves particle number while `iSWAP` and `√iSWAP` do too.
 * `toOps()` already lowers every circuit to explicit one- and two-qubit
 * matrices, so the test is applied to what the gate *is*.
 */
import { Circuit } from './circuit.js'
import type { Complex } from './complex.js'
import type { Gate2x2, Gate4x4 } from './statevector.js'
import { makePrng } from './prng.js'

/** Below this an amplitude counts as structurally zero rather than small. */
const ZERO_TOL = 1e-12

const isZero = (z: Complex): boolean => Math.abs(z.re) < ZERO_TOL && Math.abs(z.im) < ZERO_TOL

/** Excitations in a two-qubit basis index, where index = 2·bit(a) + bit(b). */
const WEIGHT2 = [0, 1, 1, 2] as const

/**
 * Does this one-qubit gate preserve excitation count?
 *
 * Only if it is diagonal: an off-diagonal entry is exactly the amplitude for
 * creating or destroying one.
 */
export function preservesWeight1(g: Gate2x2): boolean {
  return isZero(g[0][1]) && isZero(g[1][0])
}

/**
 * Does this two-qubit gate preserve excitation count?
 *
 * The matrix must be block diagonal over the weight classes {00}, {01,10}, {11}.
 * The middle block is where the interesting gates live — an fSim or an iSWAP
 * moves an excitation between the two qubits without changing how many there are.
 */
export function preservesWeight2(g: Gate4x4): boolean {
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    if (WEIGHT2[i] !== WEIGHT2[j] && !isZero(g[i]![j]!)) return false
  }
  return true
}

export interface WeightSymmetry {
  /**
   * True when the circuit provably cannot change the excitation count.
   *
   * A **sufficient** condition, not a necessary one. It is established gate by
   * gate, so `true` is trustworthy and `false` only means no proof was found —
   * a circuit can conserve weight globally while individual gates do not. The
   * clearest case is `Rzz`, which is diagonal and conserving, but is written here
   * as `CNOT · Rz · CNOT`, and a CNOT on its own moves an excitation. Runs of
   * gates confined to one pair of qubits are therefore fused before testing,
   * which recovers exactly that case; a symmetry spread more widely than two
   * qubits will still be missed.
   */
  conserved: boolean
  /** The first gate that broke it, described. Absent when the symmetry holds. */
  brokenBy?: string
  /** Excitations in the initial state. */
  weight: number
  qubits: number
  /** C(n, k) — states the circuit can ever reach. */
  sectorDimension: number
  /** 2^n over the sector dimension. How much the conservation law removes. */
  reduction: number
}

/** C(n, k), exactly for the sizes that matter and without overflowing early. */
export function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0
  const m = Math.min(k, n - k)
  let out = 1
  for (let i = 0; i < m; i++) out = out * (n - i) / (i + 1)
  return Math.round(out)
}

/** Excitations in a bitstring, in ket's q0-leftmost convention. */
export const weightOf = (bitstring: string): number => {
  let w = 0
  for (const c of bitstring) if (c === '1') w++
  return w
}

/**
 * Does this circuit conserve excitation count, and what does that buy?
 *
 * `initialState` defaults to |0…0⟩, whose sector is a single state — so a
 * conserving circuit started from the vacuum is trivial, and the interesting
 * question is always asked with a populated initial state.
 */
export function weightSymmetry(circuit: Circuit, initialState?: string): WeightSymmetry {
  const n = circuit.qubits
  if (initialState !== undefined && initialState.length !== n) {
    throw new RangeError(`weightSymmetry: initialState '${initialState}' must have ${n} characters`)
  }
  const weight = initialState === undefined ? 0 : weightOf(initialState)

  const brokenBy = firstBreak(circuit)

  const sectorDimension = brokenBy === undefined ? binomial(n, weight) : 2 ** n
  return {
    conserved: brokenBy === undefined,
    ...(brokenBy !== undefined && { brokenBy }),
    weight, qubits: n, sectorDimension,
    reduction: 2 ** n / sectorDimension,
  }
}

/** 4x4 complex identity, as a mutable array of rows. */
const eye4 = (): Complex[][] =>
  [0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => ({ re: i === j ? 1 : 0, im: 0 })))

/** Embed a 1-qubit gate acting on `slot` (0 = high bit) into a 4x4. */
function embed(g: Gate2x2, slot: 0 | 1): Complex[][] {
  const out = eye4().map(r => r.map(() => ({ re: 0, im: 0 })))
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    const [ia, ib] = [(i >> 1) & 1, i & 1]
    const [ja, jb] = [(j >> 1) & 1, j & 1]
    const same = slot === 0 ? ib === jb : ia === ja
    if (!same) continue
    const gi = slot === 0 ? ia : ib
    const gj = slot === 0 ? ja : jb
    out[i]![j] = g[gi]![gj]!
  }
  return out
}

const mul4 = (a: Complex[][], b: Complex[][]): Complex[][] =>
  [0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => {
    let re = 0, im = 0
    for (let k = 0; k < 4; k++) {
      re += a[i]![k]!.re * b[k]![j]!.re - a[i]![k]!.im * b[k]![j]!.im
      im += a[i]![k]!.re * b[k]![j]!.im + a[i]![k]!.im * b[k]![j]!.re
    }
    return { re, im }
  }))

/** A run of gates fused into one operator that provably conserves weight. */
export type FusedOp =
  | { kind: 'one'; q: number; gate: Gate2x2 }
  | { kind: 'two'; a: number; b: number; gate: Gate4x4 }

/**
 * Split a circuit into fused, individually-conserving operators.
 *
 * Both the detector and the simulator need this, and they must agree: a circuit
 * can conserve only as a *product* — Heisenberg's `Rxx · Ryy` does, while each
 * factor moves weight by two — and applying such a circuit gate by gate inside a
 * sector is wrong, because the intermediate amplitude leaves the sector and the
 * basis has nowhere to put it. Detecting on runs and then simulating on gates
 * gave a viable verdict and a wrong answer, which is the one failure this whole
 * library is arranged to prevent. So the runs are the unit of both.
 *
 * Consecutive gates confined to the same pair of qubits are multiplied together,
 * because a decomposition can break a symmetry the gate it decomposes actually
 * has: `Rzz` is diagonal and conserving, and `CNOT · Rz · CNOT` is the same
 * operator written in three steps of which two are not. Fusing asks about the
 * operator rather than the spelling.
 */
function fusedRuns(circuit: Circuit): { runs: FusedOp[] } | { brokenBy: string } {
  const ops = circuit.toOps()
  const runs: FusedOp[] = []
  let i = 0
  while (i < ops.length) {
    // Grow a run while everything in it fits inside one pair of qubits.
    const support = new Set<number>()
    let j = i
    for (; j < ops.length; j++) {
      const op = ops[j]!
      const next = new Set(support)
      if (op.kind === 'one') next.add(op.q)
      else { next.add(op.a); next.add(op.b) }
      if (next.size > 2) break
      support.clear()
      for (const q of next) support.add(q)
    }
    if (j === i) j = i + 1                     // a single op always forms a run

    const qs = [...support].sort((a, b) => a - b)
    const qa = qs[0] ?? 0
    const qb = qs.length > 1 ? qs[1]! : -1

    /**
     * Take the longest *prefix* of the run whose product conserves, not the
     * whole run.
     *
     * Growing greedily to the widest support absorbs gates belonging to the next
     * bond's decomposition — a trailing single-qubit gate on the shared qubit
     * does not widen the support past two, so it joins the run and breaks a
     * product that was conserving one gate earlier. On a Heisenberg chain that
     * turned every bond into a false negative. Tracking the running product and
     * remembering the last index at which it was still conserving costs one
     * matrix multiply per gate and gets the boundary right.
     */
    let m: Complex[][] = eye4()
    let lastGood = -1
    for (let k = i; k < j; k++) {
      const op = ops[k]!
      m = op.kind === 'one'
        ? mul4(embed(op.gate, qb < 0 || op.q === qa ? 0 : 1), m)
        : mul4(op.a === qa ? (op.gate as unknown as Complex[][]) : swapArgs(op.gate), m)
      if (preservesWeight2(m as Gate4x4)) lastGood = k + 1
    }
    if (lastGood <= i) {
      return { brokenBy: qb < 0 ? `a 1-qubit gate on q${qa}` : `a run of gates on q${qa},q${qb}` }
    }
    // Recompute the product for exactly the accepted prefix.
    let acc: Complex[][] = eye4()
    for (let k = i; k < lastGood; k++) {
      const op = ops[k]!
      acc = op.kind === 'one'
        ? mul4(embed(op.gate, qb < 0 || op.q === qa ? 0 : 1), acc)
        : mul4(op.a === qa ? (op.gate as unknown as Complex[][]) : swapArgs(op.gate), acc)
    }
    if (qb < 0) {
      // The run touched one wire, so the 4x4 is `g ⊗ I` and g sits at these slots.
      runs.push({ kind: 'one', q: qa, gate: [[acc[0]![0]!, acc[0]![2]!], [acc[2]![0]!, acc[2]![2]!]] })
    } else {
      runs.push({ kind: 'two', a: qa, b: qb, gate: acc as Gate4x4 })
    }
    i = lastGood
  }
  return { runs }
}

/** The first place the excitation count can change, or undefined if nowhere. */
function firstBreak(circuit: Circuit): string | undefined {
  const r = fusedRuns(circuit)
  return 'brokenBy' in r ? r.brokenBy : undefined
}

/** The same 4x4 with its two qubits exchanged. */
function swapArgs(g: Gate4x4): Complex[][] {
  const swap = (i: number): number => ((i & 1) << 1) | ((i >> 1) & 1)
  return [0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => g[swap(i)]![swap(j)]!))
}

// ── Simulating inside the sector ──────────────────────────────────────────────

/** Basis of a fixed-weight sector: the masks, and where each one lives. */
interface Sector {
  masks: number[]
  index: Map<number, number>
}

/** Every n-bit mask of the given popcount, ascending. */
function sectorBasis(n: number, k: number): Sector {
  const masks: number[] = []
  for (let m = 0; m < 1 << n; m++) {
    let w = 0, x = m
    while (x) { w += x & 1; x >>>= 1 }
    if (w === k) masks.push(m)
  }
  const index = new Map<number, number>()
  masks.forEach((m, i) => index.set(m, i))
  return { masks, index }
}

/**
 * Enumerating the sector by scanning 2^n defeats the point past ~28 qubits.
 *
 * Combinatorial enumeration would lift that, and is the obvious next step if this
 * attack is ever pointed somewhere the scan cannot reach. Refusing loudly beats
 * quietly taking exponential time inside a routine whose whole claim is that it
 * does not.
 */
const MAX_ENUMERATED_QUBITS = 26

export interface SectorRunOptions {
  /** Initial computational basis state. Its weight fixes the sector. */
  initialState: string
  shots?: number
  seed?: number
}

export interface SectorResult {
  /** Probabilities keyed by bitstring, q0 leftmost, as elsewhere in ket. */
  probs: Record<string, number>
  /** States the simulation actually held. */
  sectorDimension: number
  /** 2^n over that. */
  reduction: number
  shots: number
}

/**
 * Simulate inside the conserved sector.
 *
 * An ordinary statevector over C(n,k) states rather than 2^n. Diagonal gates
 * scale amplitudes in place; a weight-preserving two-qubit gate mixes only the
 * `|01⟩ ↔ |10⟩` pair, so it is a 2×2 rotation applied to pairs of sector indices
 * and the `|00⟩` and `|11⟩` components are phases.
 *
 * @throws TypeError when the circuit does not conserve weight, naming the gate.
 */
export function runSector(circuit: Circuit, options: SectorRunOptions): SectorResult {
  const { initialState, shots = 1024, seed } = options
  const n = circuit.qubits
  const sym = weightSymmetry(circuit, initialState)
  if (!sym.conserved) {
    throw new TypeError(`runSector: this circuit does not conserve excitation count — broken by ${sym.brokenBy}`)
  }
  if (n > MAX_ENUMERATED_QUBITS) {
    throw new RangeError(
      `runSector: enumerating a sector of ${n} qubits scans 2^${n} masks, which defeats the purpose. ` +
      `Combinatorial enumeration would lift this ceiling.`)
  }

  const k = sym.weight
  const { masks, index } = sectorBasis(n, k)
  const dim = masks.length
  const re = new Float64Array(dim)
  const im = new Float64Array(dim)

  // ket's bitstrings are q0-leftmost; the masks are q0 = bit 0.
  let start = 0
  for (let q = 0; q < n; q++) if (initialState[q] === '1') start |= 1 << q
  re[index.get(start)!] = 1

  const bit = (m: number, q: number): number => (m >> q) & 1

  const fused = fusedRuns(circuit)
  if ('brokenBy' in fused) {
    throw new TypeError(`runSector: this circuit does not conserve excitation count — broken by ${fused.brokenBy}`)
  }
  for (const op of fused.runs) {
    if (op.kind === 'one') {
      const [[a], [, d]] = op.gate
      for (let i = 0; i < dim; i++) {
        const z = bit(masks[i]!, op.q) === 0 ? a : d
        const r = re[i]!, m = im[i]!
        re[i] = r * z.re - m * z.im
        im[i] = r * z.im + m * z.re
      }
      continue
    }
    const g = op.gate
    const { a: qa, b: qb } = op
    const done = new Uint8Array(dim)
    for (let i = 0; i < dim; i++) {
      if (done[i]) continue
      const m = masks[i]!
      const ba = bit(m, qa), bb = bit(m, qb)
      if (ba === bb) {
        // |00> and |11> are one-dimensional blocks: a phase.
        const z = ba === 0 ? g[0]![0]! : g[3]![3]!
        const r = re[i]!, s = im[i]!
        re[i] = r * z.re - s * z.im
        im[i] = r * z.im + s * z.re
        done[i] = 1
        continue
      }
      // The |01> / |10> block. Matrix index 1 is (a=0,b=1), index 2 is (a=1,b=0).
      const partner = index.get(m ^ (1 << qa) ^ (1 << qb))!
      const [i01, i10] = ba === 0 ? [i, partner] : [partner, i]
      const r01 = re[i01]!, s01 = im[i01]!, r10 = re[i10]!, s10 = im[i10]!
      const mul = (z: Complex, r: number, s: number): [number, number] =>
        [z.re * r - z.im * s, z.re * s + z.im * r]
      const [ar, ai] = mul(g[1]![1]!, r01, s01)
      const [br, bi] = mul(g[1]![2]!, r10, s10)
      const [cr, ci] = mul(g[2]![1]!, r01, s01)
      const [dr, di] = mul(g[2]![2]!, r10, s10)
      re[i01] = ar + br; im[i01] = ai + bi
      re[i10] = cr + dr; im[i10] = ci + di
      done[i01] = 1; done[i10] = 1
    }
  }

  const probs: Record<string, number> = {}
  const weights = new Float64Array(dim)
  let total = 0
  for (let i = 0; i < dim; i++) { weights[i] = re[i]! * re[i]! + im[i]! * im[i]!; total += weights[i]! }

  const rand = makePrng(seed)
  const counts = new Int32Array(dim)
  for (let s = 0; s < shots; s++) {
    let r = rand() * total
    let pick = dim - 1
    for (let i = 0; i < dim; i++) { r -= weights[i]!; if (r <= 0) { pick = i; break } }
    counts[pick]!++
  }
  for (let i = 0; i < dim; i++) {
    if (counts[i] === 0) continue
    let bits = ''
    for (let q = 0; q < n; q++) bits += (masks[i]! >> q) & 1 ? '1' : '0'
    probs[bits] = counts[i]! / shots
  }

  return { probs, sectorDimension: dim, reduction: 2 ** n / dim, shots }
}

// ── A circuit family that actually has the symmetry ───────────────────────────

/**
 * A particle-conserving ansatz, of the kind quantum chemistry runs.
 *
 * Layers of Givens-like rotations — `XY(θ)` mixes `|01⟩` and `|10⟩` and nothing
 * else — interleaved with `Rz` phases, which are diagonal and so also conserving.
 * Started from a state with `k` excitations it explores exactly the C(n,k)
 * sector, which is what makes it a sensible ansatz for a molecule with a fixed
 * electron count.
 *
 * Neither published family in this library has this symmetry: IBM's Trotter
 * circuits break it with the transverse field and Google's RCS breaks it with
 * √X/√Y/√W, in both cases at the single-qubit layer. It is worth knowing where
 * the axis does *not* fire.
 */
export function particleConservingAnsatz(
  { qubits, layers, seed = 1 }: { qubits: number; layers: number; seed?: number },
): Circuit {
  if (!Number.isInteger(qubits) || qubits < 2) throw new RangeError(`particleConservingAnsatz: qubits must be at least 2, got ${qubits}`)
  if (!Number.isInteger(layers) || layers < 1) throw new RangeError(`particleConservingAnsatz: layers must be a positive integer, got ${layers}`)
  const rand = makePrng(seed)
  let c = new Circuit(qubits)
  for (let l = 0; l < layers; l++) {
    // Brickwork of excitation-preserving mixers.
    for (let q = l % 2; q + 1 < qubits; q += 2) c = c.xy(0.3 + rand() * 2, q, q + 1)
    // Diagonal phases: conserving, and what carries the variational freedom that
    // the mixers alone cannot express.
    for (let q = 0; q < qubits; q++) c = c.rz(0.2 + rand() * 2, q)
  }
  return c
}
