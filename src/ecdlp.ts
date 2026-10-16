/**
 * What a 256-bit ECDLP attack actually costs, counted rather than asserted.
 *
 * Resource estimates for breaking elliptic-curve signatures are published as
 * single numbers — "39 million Toffoli gates", "50 million Toffoli gates",
 * "2330 logical qubits" — and a reader has no way to take one apart. The
 * circuits are far beyond any simulator, so the usual check, run it and see,
 * is unavailable at 256 bits and will stay unavailable.
 *
 * What is available is the *composition*. Shor's algorithm for the elliptic
 * curve discrete logarithm is a double-and-add ladder over 2n controlled point
 * additions; a point addition is a fixed number of modular multiplications,
 * inversions and additions; a modular multiplication is n rounds of modular
 * doubling and controlled modular addition; and a modular addition is five
 * ripple-carry adders. Every layer but one is small enough to build as an
 * actual gate network, simulate, and count — which is what this module does.
 * The Toffoli counts in the report below are not formulas someone typed in;
 * they are measured off circuits in `ecdlp.test.ts` that are separately checked
 * to compute the right arithmetic, and the closed forms are asserted against
 * those measurements at every width from 3 to 10 bits.
 *
 * **The one exception is modular inversion.** Kaliski's almost-inverse over 2n
 * rounds is the only subroutine here that is priced from a round-count model
 * rather than a built circuit — its per-round cost is composed from primitives
 * that *are* measured, but the round schedule and its control logic are taken
 * from Roetteler et al. and are not simulated. Every term in the output carries
 * its `source` for exactly this reason, and {@link EcdlpEstimate.sensitivity}
 * reports the total with that term set to zero and to ten times the model, so a
 * reader can see how much of the answer rests on it. (It is not much: the
 * conclusion below survives both.)
 *
 * **What this cannot do.** The counts stop at the logical layer. Nothing here
 * models an error-correcting code, a decoder, magic-state distillation or
 * cultivation, routing, or a code cycle, so nothing here speaks to a physical
 * qubit count or a wall-clock runtime. A published estimate that compiles to
 * qLDPC primitives and quotes a number of days is making a claim this module
 * cannot reach, and a reader should not treat agreement on Toffolis as
 * agreement on anything downstream of them.
 *
 * **What it is for.** Given a published logical-layer number, this says where
 * it sits relative to a construction whose arithmetic has been verified gate by
 * gate, and how much algorithmic improvement over that construction the number
 * requires. That turns "is 39 million plausible?" into "is a 180x improvement
 * over textbook shift-and-add plausible?", which is a question with a
 * literature attached to it.
 *
 * References, all quoted in {@link ECDLP_REFERENCES}:
 *   - Roetteler, Naehrig, Svore, Lauter (2017), arXiv:1706.06752
 *   - Häner, Roetteler, Svore (2016), arXiv:1611.07995 — constant adders
 *   - Litinski (2023), arXiv:2306.08585
 *   - Cuccaro, Draper, Kutin, Moulton (2004), quant-ph/0410184 — the adder
 */
import { Circuit } from './circuit.js'

// ── Gate-level reversible arithmetic ─────────────────────────────────────────
//
// Registers are passed as explicit arrays of qubit indices, least-significant
// first, matching ket's qubit-0-is-LSB convention. Every routine below is a
// permutation of basis states, so a circuit built from them stays a single
// amplitude and the sparse statevector simulates it in constant memory however
// wide the registers are — which is what makes exhaustive validation cheap.
//
// Arithmetic is two's complement over the full register width w. Callers work
// with w = n + 1 for an n-bit modulus, so that a sum of two residues and a
// subtraction that goes negative both stay representable and the top bit reads
// as a sign.

/** CDKM majority block: writes the carry into `z`. One Toffoli. */
function maj(c: Circuit, x: number, y: number, z: number): Circuit {
  return c.cx(z, y).cx(z, x).ccx(x, y, z)
}

/** CDKM un-majority-and-add: undoes {@link maj} and leaves the sum in `y`. One Toffoli. */
function uma(c: Circuit, x: number, y: number, z: number): Circuit {
  return c.ccx(x, y, z).cx(z, x).cx(x, y)
}

/**
 * Ripple-carry addition in place: `b += a` mod 2^w, with `a` restored.
 *
 * Cuccaro et al.'s adder. `cin` is one clean ancilla, returned to |0⟩. The
 * carry out of the top bit is discarded, which is what makes this addition
 * modulo 2^w — the wrap is deliberate and the two's-complement subtraction
 * below depends on it.
 *
 * Cost: 2w Toffoli gates, w = a.length.
 */
export function rippleAdd(c: Circuit, a: readonly number[], b: readonly number[], cin: number): Circuit {
  const w = a.length
  if (b.length !== w) throw new RangeError(`rippleAdd: registers differ in width (${w} vs ${b.length})`)
  c = maj(c, cin, b[0]!, a[0]!)
  for (let i = 1; i < w; i++) c = maj(c, a[i - 1]!, b[i]!, a[i]!)
  for (let i = w - 1; i >= 1; i--) c = uma(c, a[i - 1]!, b[i]!, a[i]!)
  return uma(c, cin, b[0]!, a[0]!)
}

/**
 * Ripple-carry subtraction in place: `b -= a` mod 2^w, with `a` restored.
 *
 * `b - a = ~(~b + a)`, so this is {@link rippleAdd} conjugated by bit flips on
 * `b`. Same cost, 2w Toffoli gates: the flips are single-qubit.
 */
export function rippleSub(c: Circuit, a: readonly number[], b: readonly number[], cin: number): Circuit {
  for (const q of b) c = c.x(q)
  c = rippleAdd(c, a, b, cin)
  for (const q of b) c = c.x(q)
  return c
}

/** Bits of a non-negative integer, least-significant first, padded to `w`. */
function bits(k: bigint, w: number): boolean[] {
  const out: boolean[] = []
  for (let i = 0; i < w; i++) out.push(((k >> BigInt(i)) & 1n) === 1n)
  return out
}

/**
 * Add the classical constant `k`: `b += k` mod 2^w.
 *
 * The constant is loaded into a clean scratch register with X gates, added, and
 * unloaded. That costs one full quantum-quantum addition where a dedicated
 * constant adder (Häner, Roetteler, Svore, arXiv:1611.07995) needs roughly
 * half as many Toffoli gates and no scratch register — a known factor this
 * construction leaves on the table, and one of the reasons the baseline below
 * is a baseline rather than a best effort.
 *
 * Cost: 2w Toffoli gates.
 */
export function addConst(
  c: Circuit, k: bigint, b: readonly number[], scratch: readonly number[], cin: number,
): Circuit {
  const w = b.length
  const kb = bits(k, w)
  for (let i = 0; i < w; i++) if (kb[i]) c = c.x(scratch[i]!)
  c = rippleAdd(c, scratch, b, cin)
  for (let i = 0; i < w; i++) if (kb[i]) c = c.x(scratch[i]!)
  return c
}

/** Subtract the classical constant `k`: `b -= k` mod 2^w. Cost: 2w Toffoli gates. */
export function subConst(
  c: Circuit, k: bigint, b: readonly number[], scratch: readonly number[], cin: number,
): Circuit {
  for (const q of b) c = c.x(q)
  c = addConst(c, k, b, scratch, cin)
  for (const q of b) c = c.x(q)
  return c
}

/**
 * Add the classical constant `k` if `ctrl` is set.
 *
 * Controlling a *classical* constant costs nothing extra: the scratch register
 * is loaded with CNOTs from the control instead of X gates, so an unset control
 * loads zero and the adder adds nothing. Cost is the same 2w Toffoli gates as
 * the unconditional version — the reason modular reduction is cheap.
 */
export function ctrlAddConst(
  c: Circuit, ctrl: number, k: bigint, b: readonly number[], scratch: readonly number[], cin: number,
): Circuit {
  const w = b.length
  const kb = bits(k, w)
  for (let i = 0; i < w; i++) if (kb[i]) c = c.cx(ctrl, scratch[i]!)
  c = rippleAdd(c, scratch, b, cin)
  for (let i = 0; i < w; i++) if (kb[i]) c = c.cx(ctrl, scratch[i]!)
  return c
}

/** The ancilla a modular routine needs beyond its operands. */
export interface ModWork {
  /** Constant-loading register, w qubits, clean in and out. */
  scratch: readonly number[]
  /** Adder carry ancilla, one qubit, clean in and out. */
  cin: number
  /** Reduction flag, one qubit, clean in and out. */
  flag: number
}

/**
 * Modular addition: `b = (a + b) mod p`, with `a` restored.
 *
 * Requires `a, b < p < 2^(w-1)` so that the sum fits in w bits and the top bit
 * of the register reads as a two's-complement sign.
 *
 * Five additions: add, subtract p, conditionally add p back, then two more to
 * clear the reduction flag. The flag is the subtle part — it must be returned
 * to |0⟩ or it stays entangled with the result and the multiplier above breaks.
 * The identity used is that `(a+b) mod p < a` exactly when the reduction fired,
 * so subtracting `a` from the reduced result exposes the flag as a sign bit and
 * lets it be uncomputed. `p` odd is not required here.
 *
 * Cost: 10w Toffoli gates.
 */
export function modAdd(
  c: Circuit, a: readonly number[], b: readonly number[], p: bigint, work: ModWork,
): Circuit {
  const { scratch, cin, flag } = work
  const top = b[b.length - 1]!
  c = rippleAdd(c, a, b, cin)             // b = a + b, < 2p, fits in w bits
  c = subConst(c, p, b, scratch, cin)     // b = a + b - p, negative iff a + b < p
  c = c.cx(top, flag)                     // flag = 1 iff no reduction was needed
  c = ctrlAddConst(c, flag, p, b, scratch, cin)  // b = (a + b) mod p
  c = rippleSub(c, a, b, cin)             // b = result - a, negative iff flag = 0
  c = c.x(top).cx(top, flag).x(top)       // flag ^= NOT sign = flag, so flag = 0
  c = rippleAdd(c, a, b, cin)             // b = result
  return c
}

/**
 * Modular doubling in place: `a = 2a mod p`, for odd `p`.
 *
 * The shift is a chain of SWAPs, which is Clifford and costs no Toffoli gates;
 * only the conditional subtraction of `p` does. Clearing the reduction flag is
 * free here and needs no extra arithmetic: `2a` is even, so if `p` is odd the
 * result is odd exactly when the reduction fired, and the low bit *is* the
 * flag. This is why the two reductions differ in cost.
 *
 * Requires `a < p < 2^(w-1)` and `p` odd — every prime field of cryptographic
 * interest qualifies.
 *
 * Cost: 4w Toffoli gates.
 */
export function modDouble(c: Circuit, a: readonly number[], p: bigint, work: ModWork): Circuit {
  if ((p & 1n) === 0n) throw new RangeError('modDouble: p must be odd')
  const { scratch, cin, flag } = work
  const w = a.length
  const top = a[w - 1]!
  for (let i = w - 1; i >= 1; i--) c = c.swap(a[i]!, a[i - 1]!)  // a = 2a, exact in w bits
  c = subConst(c, p, a, scratch, cin)             // a = 2a - p, negative iff 2a < p
  c = c.cx(top, flag)
  c = ctrlAddConst(c, flag, p, a, scratch, cin)   // a = 2a mod p
  c = c.x(a[0]!).cx(a[0]!, flag).x(a[0]!)         // result even iff reduction did not fire
  return c
}

/**
 * Controlled modular addition: `b = (a + b) mod p` if `ctrl` is set.
 *
 * The control is ANDed into a copy of `a` and the unconditional adder runs on
 * the copy, which is the simple construction rather than the tight one — a
 * controlled adder that gates only the writes into `b` avoids the copy and its
 * 2w Toffoli gates. Counted honestly rather than optimised.
 *
 * Cost: 12w Toffoli gates.
 */
export function ctrlModAdd(
  c: Circuit, ctrl: number, a: readonly number[], copy: readonly number[],
  b: readonly number[], p: bigint, work: ModWork,
): Circuit {
  const w = a.length
  for (let i = 0; i < w; i++) c = c.ccx(ctrl, a[i]!, copy[i]!)
  c = modAdd(c, copy, b, p, work)
  for (let i = 0; i < w; i++) c = c.ccx(ctrl, a[i]!, copy[i]!)
  return c
}

/**
 * Modular multiplication: `res = (x * y) mod p`, with `x` and `y` restored.
 *
 * Horner's method over the bits of `x`, most significant first, doubling the
 * accumulator rather than the multiplicand so that `y` is never touched and
 * needs no unwinding at the end. `res` must arrive clean.
 *
 * Cost: 16nw Toffoli gates, for an n-bit multiplier and width-w registers.
 */
export function modMul(
  c: Circuit, x: readonly number[], y: readonly number[], copy: readonly number[],
  res: readonly number[], p: bigint, work: ModWork,
): Circuit {
  for (let i = x.length - 1; i >= 0; i--) {
    c = modDouble(c, res, p, work)
    c = ctrlModAdd(c, x[i]!, y, copy, res, p, work)
  }
  return c
}

// ── Closed forms, asserted against the circuits above ────────────────────────
//
// Every one of these is checked in ecdlp.test.ts against a real gate count from
// the corresponding builder, at each width from 3 to 10. They are the bridge
// from circuits small enough to simulate to a width nobody will ever simulate.

/** Toffoli gates in {@link rippleAdd} and {@link rippleSub} on width-w registers. */
export const costAdd = (w: number): number => 2 * w
/** Toffoli gates in {@link modAdd} on width-w registers. */
export const costModAdd = (w: number): number => 10 * w
/** Toffoli gates in {@link modDouble} on width-w registers. */
export const costModDouble = (w: number): number => 4 * w
/** Toffoli gates in {@link ctrlModAdd} on width-w registers. */
export const costCtrlModAdd = (w: number): number => 12 * w
/** Toffoli gates in {@link modMul} for an n-bit multiplier on width-w registers. */
export const costModMul = (n: number, w: number): number => n * (costModDouble(w) + costCtrlModAdd(w))

// ── The resource model ───────────────────────────────────────────────────────

/** Where a Toffoli count came from, and therefore how far to trust it. */
export type EcdlpSource =
  /** Counted off a gate network in this module that is also checked to compute the right arithmetic. */
  | 'measured'
  /** Arithmetic on measured terms — a composition whose factors are each measured. */
  | 'composed'
  /** A round count or a structure taken from the literature, not built here. Treat as the soft term. */
  | 'modelled'

/** One line of the cost breakdown. */
export interface EcdlpTerm {
  name: string
  /** Toffoli gates in one invocation. */
  each: number
  /** Invocations by the level above. */
  calls: number
  /** Contribution to the level above, `each * calls`. */
  total: number
  source: EcdlpSource
  note?: string
}

/** Knobs on the one part of the model that is not measured. */
export interface EcdlpOptions {
  /**
   * Toffoli gates per Kaliski round, in units of the register width w.
   *
   * Roetteler et al. §3.4 run 2n rounds, each testing the parities of u and v,
   * comparing them, and conditionally subtracting and shifting. Priced from the
   * measured primitives here: one comparison is two additions (4w), one
   * controlled subtraction is an addition plus the control copy (4w), and the
   * two controlled register swaps are w Toffoli-equivalents apiece (2w), for
   * 10w — rounded to 12w for the round's control logic and counter.
   */
  inversionRoundWidths?: number
  /** Kaliski rounds, as a multiple of n. Roetteler et al. use 2n. */
  inversionRounds?: number
}

/** A full estimate for one field width, with everything it was built from. */
export interface EcdlpEstimate {
  /** Bit width of the prime field. 256 for secp256k1 and P-256. */
  n: number
  /** Register width, `n + 1` — one bit of headroom for sums and sign. */
  width: number
  /** Total Toffoli gates for the discrete logarithm. */
  toffoli: number
  /** Logical qubits held for the duration. */
  logicalQubits: number
  /** Controlled point additions in the double-and-add ladder: 2n. */
  pointAdditions: number
  /** Toffoli gates in one controlled point addition. */
  perPointAddition: number
  /** What a point addition is made of. */
  breakdown: EcdlpTerm[]
  /**
   * The total if the one modelled term were wrong.
   *
   * `inversionFree` sets modular inversion to zero, `inversionTenfold` to ten
   * times the model. If a conclusion holds across that range it does not rest
   * on the term that was not built.
   */
  sensitivity: { inversionFree: number; inversionTenfold: number }
}

/**
 * Resource estimate for Shor's algorithm on an n-bit prime-field ECDLP.
 *
 * The ladder is 2n controlled point additions in affine coordinates. Each one
 * computes the slope `λ = (y₂-y₁)/(x₂-x₁)`, which needs a modular inversion,
 * uses it in four multiplications and two squarings, and then runs the
 * inversion backwards to uncompute it — hence two inversions per point
 * addition, the dominant term. The eight modular additions are the coordinate
 * differences and the final `x₃ = λ² - x₁ - x₂`, `y₃ = λ(x₁ - x₃) - y₁`, and
 * the two controlled swaps carry the ladder's control bit.
 */
export function ecdlpResources(n: number, options: EcdlpOptions = {}): EcdlpEstimate {
  if (!Number.isInteger(n) || n < 2) throw new RangeError(`ecdlpResources: n must be an integer >= 2, got ${n}`)
  const { inversionRoundWidths = 12, inversionRounds = 2 } = options
  const w = n + 1

  // Kaliski almost-inverse, plus the two multiplications that undo the
  // Montgomery factor it leaves behind. The only 'modelled' line in the table.
  const inversion = inversionRounds * n * inversionRoundWidths * w + 2 * costModMul(n, w)

  const breakdown: EcdlpTerm[] = [
    {
      name: 'modular inversion', each: inversion, calls: 2, total: 2 * inversion, source: 'modelled',
      note: `${inversionRounds}n Kaliski rounds at ${inversionRoundWidths}w, + Montgomery correction`,
    },
    {
      name: 'modular multiplication', each: costModMul(n, w), calls: 6, total: 6 * costModMul(n, w),
      source: 'composed', note: 'n rounds of doubling and controlled addition; 2 of the 6 are squarings',
    },
    {
      name: 'modular addition', each: costModAdd(w), calls: 8, total: 8 * costModAdd(w),
      source: 'measured', note: 'coordinate differences and the two output coordinates',
    },
    {
      name: 'controlled swap', each: w, calls: 2, total: 2 * w,
      source: 'composed', note: 'the ladder control bit, one Toffoli-equivalent per qubit',
    },
  ]

  const perPointAddition = breakdown.reduce((s, t) => s + t.total, 0)
  const pointAdditions = 2 * n
  const toffoli = pointAdditions * perPointAddition

  // Registers held for the duration: the accumulator point (2w), the
  // multiplier's accumulator, multiplicand copy and constant scratch (3w),
  // Kaliski's u, v, r, s (4w), and a dozen carries and flags. The exponents are
  // not stored — a semi-classical QFT feeds the ladder one bit at a time.
  const logicalQubits = 9 * w + 12

  const withInversion = (each: number): number =>
    pointAdditions * (perPointAddition - 2 * inversion + 2 * each)

  return {
    n, width: w, toffoli, logicalQubits, pointAdditions, perPointAddition, breakdown,
    sensitivity: { inversionFree: withInversion(0), inversionTenfold: withInversion(10 * inversion) },
  }
}

// ── Published estimates, for comparison ──────────────────────────────────────

/** A published resource estimate, quoted for comparison against the model. */
export interface EcdlpReference {
  label: string
  year: number
  cite: string
  /** Toffoli gates at n = 256, or null if the source does not give one. */
  toffoli: number | null
  /** Logical qubits at n = 256, or null. */
  logicalQubits: number | null
  /** What the number does and does not include. */
  note: string
}

/**
 * Published logical-layer estimates for a 256-bit prime-field ECDLP.
 *
 * Quoted, not recomputed. Where a source gives a formula the value here is that
 * formula evaluated at n = 256, and the formula is in the note so it can be
 * checked. These are the yardsticks; nothing in this list is verified by ket.
 */
export const ECDLP_REFERENCES: readonly EcdlpReference[] = Object.freeze([
  {
    label: 'Roetteler et al.', year: 2017, cite: 'arXiv:1706.06752',
    // 448 n^3 log2(n) + 4090 n^3 at n = 256, log2(256) = 8.
    toffoli: (448 * 8 + 4090) * 256 ** 3,
    logicalQubits: 9 * 256 + 2 * 8 + 10,
    note: 'upper bound 448n³log₂n + 4090n³; qubits 9n + 2⌈log₂n⌉ + 10',
  },
  {
    label: 'Litinski', year: 2023, cite: 'arXiv:2306.08585',
    toffoli: 44e6, logicalQubits: null,
    note: 'asymptotic cost per key; title quotes 50M. Windowed arithmetic, shared division, state reuse',
  },
  {
    label: 'IonQ', year: 2026, cite: 'ionq.com press release',
    toffoli: 39e6, logicalQubits: 1457,
    note: 'also claims 19,397 physical qubits and 25.7 days — neither is a logical-layer number',
  },
])

/** How a published estimate sits against the model. */
export interface EcdlpGap {
  reference: EcdlpReference
  /** Model Toffolis divided by the reference's, or null if it quotes none. */
  toffoliRatio: number | null
  /** Model logical qubits divided by the reference's, or null. */
  qubitRatio: number | null
}

/** Model versus every published estimate at the given width. */
export function compareEcdlp(n = 256, options: EcdlpOptions = {}): { estimate: EcdlpEstimate; gaps: EcdlpGap[] } {
  const estimate = ecdlpResources(n, options)
  const gaps = ECDLP_REFERENCES.map(reference => ({
    reference,
    toffoliRatio: reference.toffoli === null ? null : estimate.toffoli / reference.toffoli,
    qubitRatio: reference.logicalQubits === null ? null : estimate.logicalQubits / reference.logicalQubits,
  }))
  return { estimate, gaps }
}

// ── Rendering ────────────────────────────────────────────────────────────────

const sci = (x: number): string => x.toExponential(2)

/**
 * A ratio, at a precision that survives the interesting cases.
 *
 * Fixed decimals would round the model-versus-upper-bound ratio to "0.1x" and
 * lose the fact that it is a factor of eighteen.
 */
const ratio = (x: number): string =>
  x >= 10 ? `${x.toFixed(0)}x` : x >= 1 ? `${x.toFixed(2)}x` : `${x.toPrecision(3)}x`

function table(head: string[], rows: string[][], rightFrom: number): string[] {
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map(r => r[i]!.length)))
  const line = (cells: string[]): string =>
    cells.map((c, i) => i >= rightFrom ? c.padStart(widths[i]!) : c.padEnd(widths[i]!)).join('  ').trimEnd()
  return [line(head), widths.map(w => '─'.repeat(w)).join('  '), ...rows.map(line)]
}

/** The estimate, its breakdown, and the published numbers beside it. */
export function renderEcdlp(n = 256, options: EcdlpOptions = {}): string {
  const { estimate: e, gaps } = compareEcdlp(n, options)

  const breakdown = table(
    ['subroutine', 'source', 'each', 'calls', 'toffoli', 'share'],
    e.breakdown.map(t => [
      t.name, t.source, sci(t.each), String(t.calls), sci(t.total),
      `${(100 * t.total / e.perPointAddition).toFixed(1)}%`,
    ]),
    2,
  )

  const comparison = table(
    ['estimate', 'cite', 'toffoli', 'ket/theirs', 'qubits', 'ket/theirs'],
    [
      ['ket (this model)', 'measured + modelled', sci(e.toffoli), '1.00x', String(e.logicalQubits), '1.00x'],
      ...gaps.map(g => [
        `${g.reference.label} ${g.reference.year}`, g.reference.cite,
        g.reference.toffoli === null ? '' : sci(g.reference.toffoli),
        g.toffoliRatio === null ? '' : ratio(g.toffoliRatio),
        g.reference.logicalQubits === null ? '' : String(g.reference.logicalQubits),
        g.qubitRatio === null ? '' : ratio(g.qubitRatio),
      ]),
    ],
    2,
  )

  return [
    `ECDLP over a ${e.n}-bit prime field — ${e.pointAdditions} controlled point additions,`,
    `${sci(e.perPointAddition)} Toffoli gates each.`,
    '',
    'One controlled point addition:',
    ...breakdown,
    '',
    'Against the literature:',
    ...comparison,
    '',
    `Sensitivity to the one modelled term: inversion at 0x gives ${sci(e.sensitivity.inversionFree)},`,
    `at 10x gives ${sci(e.sensitivity.inversionTenfold)}.`,
    '',
    'Logical Toffoli gates and logical qubits only. No error-correcting code, no',
    'magic-state cost, no code cycle, so no physical qubit count and no runtime.',
  ].join('\n')
}
