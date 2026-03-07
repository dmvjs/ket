/**
 * Immutable circuit builder with IonQ QIS gate names.
 *
 * Each gate method returns a new Circuit — circuits are values, not mutated objects.
 * Simulation is lazy: nothing runs until `.run()` is called.
 */

import { c } from './complex.js'
import * as G from './gates.js'
import { applyCNOT, applySingle, applySWAP, probabilities, StateVector, zero } from './statevector.js'

// ─── Operation types ─────────────────────────────────────────────────────────

type SingleOp = { kind: 'single'; q: number;  gate: G.Gate2x2 extends infer T ? T : never }
type CNOTOp   = { kind: 'cnot';   control: number; target: number }
type SWAPOp   = { kind: 'swap';   a: number; b: number }
type Op = SingleOp | CNOTOp | SWAPOp

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

  constructor(qubits: number, shots: number, counts: Map<bigint, number>) {
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
    const entries = Object.entries(this.probs).sort(([a], [b]) => a.localeCompare(b))
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
  private readonly ops: readonly Op[]

  constructor(qubits: number, ops: readonly Op[] = []) {
    this.qubits = qubits
    this.ops    = ops
  }

  private add(op: Op): Circuit {
    return new Circuit(this.qubits, [...this.ops, op])
  }

  // ── IonQ single-qubit gates ──────────────────────────────────────────────

  h(q: number):  Circuit { return this.add({ kind: 'single', q, gate: G.H  }) }
  x(q: number):  Circuit { return this.add({ kind: 'single', q, gate: G.X  }) }
  y(q: number):  Circuit { return this.add({ kind: 'single', q, gate: G.Y  }) }
  z(q: number):  Circuit { return this.add({ kind: 'single', q, gate: G.Z  }) }
  s(q: number):  Circuit { return this.add({ kind: 'single', q, gate: G.S  }) }
  si(q: number): Circuit { return this.add({ kind: 'single', q, gate: G.Si }) }
  t(q: number):  Circuit { return this.add({ kind: 'single', q, gate: G.T  }) }
  ti(q: number): Circuit { return this.add({ kind: 'single', q, gate: G.Ti }) }
  v(q: number):  Circuit { return this.add({ kind: 'single', q, gate: G.V  }) }
  vi(q: number): Circuit { return this.add({ kind: 'single', q, gate: G.Vi }) }

  // ── Rotation gates ───────────────────────────────────────────────────────

  rx(theta: number, q: number): Circuit { return this.add({ kind: 'single', q, gate: G.Rx(theta) }) }
  ry(theta: number, q: number): Circuit { return this.add({ kind: 'single', q, gate: G.Ry(theta) }) }
  rz(theta: number, q: number): Circuit { return this.add({ kind: 'single', q, gate: G.Rz(theta) }) }

  // ── Named phase rotation gates ───────────────────────────────────────────

  /** Rz(π/2) — phase rotation by a half-turn; S up to global phase. */
  r2(q: number): Circuit { return this.add({ kind: 'single', q, gate: G.R2 }) }

  /** Rz(π/4) — phase rotation by a quarter-turn; T up to global phase. */
  r4(q: number): Circuit { return this.add({ kind: 'single', q, gate: G.R4 }) }

  /** Rz(π/8) — phase rotation by an eighth-turn. */
  r8(q: number): Circuit { return this.add({ kind: 'single', q, gate: G.R8 }) }

  // ── Two-qubit gates ──────────────────────────────────────────────────────

  /** Controlled-NOT. IonQ name: cnot. */
  cnot(control: number, target: number): Circuit {
    return this.add({ kind: 'cnot', control, target })
  }

  swap(a: number, b: number): Circuit {
    return this.add({ kind: 'swap', a, b })
  }

  // ── Execution ────────────────────────────────────────────────────────────

  /** Run the circuit and return a probability distribution. */
  run({ shots = 1024, seed }: RunOptions = {}): Distribution {
    let sv: StateVector = zero(this.qubits)

    for (const op of this.ops) {
      if (op.kind === 'single') {
        sv = applySingle(sv, op.q, op.gate)
      } else if (op.kind === 'cnot') {
        sv = applyCNOT(sv, op.control, op.target)
      } else {
        sv = applySWAP(sv, op.a, op.b)
      }
    }

    // Sample shots from the probability distribution
    const probs  = probabilities(sv)
    const sorted = [...probs.entries()].sort(([a], [b]) => (a < b ? -1 : 1))

    const cdf: { idx: bigint; cumP: number }[] = []
    let cum = 0
    for (const [idx, p] of sorted) {
      cum += p
      cdf.push({ idx, cumP: cum })
    }
    if (cdf.length > 0) cdf[cdf.length - 1]!.cumP = 1.0 // clamp

    const rng    = makePrng(seed)
    const counts = new Map<bigint, number>()

    for (let i = 0; i < shots; i++) {
      const r  = rng()
      let lo   = 0
      let hi   = cdf.length - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if ((cdf[mid]!.cumP) < r) lo = mid + 1
        else hi = mid
      }
      const idx = cdf[lo]?.idx ?? 0n
      counts.set(idx, (counts.get(idx) ?? 0) + 1)
    }

    return new Distribution(this.qubits, shots, counts)
  }
}
