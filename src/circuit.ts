/**
 * Immutable circuit builder with IonQ QIS gate names.
 *
 * Each gate method returns a new Circuit — circuits are values, not mutated objects.
 * Simulation is lazy: nothing runs until `.run()` is called.
 */

import * as G from './gates.js'
import { applyCNOT, applyControlled, applyCSwap, applySingle, applySWAP, applyToffoli, applyTwo, Gate2x2, Gate4x4, probabilities, StateVector, zero } from './statevector.js'

// ─── Operation types ─────────────────────────────────────────────────────────

type SingleOp     = { kind: 'single';     q: number;                      gate: Gate2x2 }
type CNOTOp       = { kind: 'cnot';       control: number; target: number               }
type SWAPOp       = { kind: 'swap';       a: number;       b: number                    }
type TwoOp        = { kind: 'two';        a: number;       b: number;    gate: Gate4x4  }
type ControlledOp = { kind: 'controlled'; control: number; target: number; gate: Gate2x2 }
type ToffoliOp    = { kind: 'toffoli';    c1: number; c2: number; target: number }
type CSwapOp      = { kind: 'cswap';      control: number; a: number; b: number }
type Op = SingleOp | CNOTOp | SWAPOp | TwoOp | ControlledOp | ToffoliOp | CSwapOp

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
  readonly #ops: readonly Op[]

  constructor(qubits: number, ops: readonly Op[] = []) {
    this.qubits = qubits
    this.#ops   = ops
  }

  #add(op: Op): Circuit {
    return new Circuit(this.qubits, [...this.#ops, op])
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

  // ── Three-qubit gates ────────────────────────────────────────────────────

  /** Toffoli (CCX): flip target if both c1 and c2 are |1⟩. Universal for reversible computation. */
  ccx(c1: number, c2: number, target: number): Circuit {
    return this.#add({ kind: 'toffoli', c1, c2, target })
  }

  /** Fredkin (CSWAP): swap qubits a and b if control is |1⟩. */
  cswap(control: number, a: number, b: number): Circuit {
    return this.#add({ kind: 'cswap', control, a, b })
  }

  // ── Execution ────────────────────────────────────────────────────────────

  /** Run the circuit and return a probability distribution. */
  run({ shots = 1024, seed }: RunOptions = {}): Distribution {
    let sv: StateVector = zero(this.qubits)

    for (const op of this.#ops) {
      if (op.kind === 'single') {
        sv = applySingle(sv, op.q, op.gate)
      } else if (op.kind === 'cnot') {
        sv = applyCNOT(sv, op.control, op.target)
      } else if (op.kind === 'controlled') {
        sv = applyControlled(sv, op.control, op.target, op.gate)
      } else if (op.kind === 'swap') {
        sv = applySWAP(sv, op.a, op.b)
      } else if (op.kind === 'toffoli') {
        sv = applyToffoli(sv, op.c1, op.c2, op.target)
      } else if (op.kind === 'cswap') {
        sv = applyCSwap(sv, op.control, op.a, op.b)
      } else {
        sv = applyTwo(sv, op.a, op.b, op.gate)
      }
    }

    // Sample shots from the probability distribution
    const probs  = probabilities(sv)
    const sorted = probs.entries().toArray().toSorted(([a], [b]) => (a < b ? -1 : 1))

    const cdf: { idx: bigint; cumP: number }[] = []
    let cum = 0
    for (const [idx, p] of sorted) {
      cum += p
      cdf.push({ idx, cumP: cum })
    }
    const last = cdf.at(-1)
    if (last) last.cumP = 1.0 // clamp floating-point drift

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
