/**
 * How hard is this circuit, actually — measured by trying to break it.
 *
 * Every backend in ket exists because some circuits have structure that collapses
 * their cost: Clifford closure, bounded magic, bounded entanglement, low
 * treewidth, Heisenberg sparsity. Each one turns an exponential in the qubit
 * count into an exponential in something else, and for a given circuit that
 * something else may be small. A circuit is hard only if it is hard for *all* of
 * them, so the way to find out is to run every attack against it and see which
 * one lands.
 *
 * Running them would defeat the point, so each attack is *probed* instead —
 * measured cheaply for the one structural quantity that decides its cost, then
 * priced. Probing is not free, but it is bounded: counting T gates is a scan,
 * planning a contraction is milliseconds to seconds, and the MPS probe runs under
 * a bond-dimension cap so it stops as soon as the answer is "more than this".
 *
 * **What a negative result means.** `best === null` says that these attacks, at
 * these budgets, in this library, did not break the circuit. It does not say the
 * circuit is hard. A better contraction planner, a smarter decomposition, or a
 * technique not implemented here would each move the line, and two of the
 * parameters reported below are bounds rather than measurements for exactly that
 * reason. The honest reading of this module is an upper bound on difficulty that
 * gets tighter as the attacks get better.
 */
import { Circuit } from './circuit.js'
import type { PauliTerm } from './algorithms.js'
import { bytesPerTerm } from './stabilizer-capacity.js'
import { countSplits, termBudget } from './stabilizer-rank.js'
import { circuitNetwork, planBest, planContraction } from './tensor-network.js'
import { pauliPathExpectation } from './pauli-path.js'
import { lightCone, peephole, searchQubitOrder, toZBasis, type Rewrite } from './rewrite.js'
import { weightSymmetry } from './symmetry.js'
import { mpsExpectation, mpsFromOps } from './mps.js'
import { pepsExpectation, pepsRun, type PepsState } from './peps.js'

/** Depolarising rates, as the noisy attacks price them. */
export interface NoiseRates {
  /** Per single-qubit gate. */
  p1: number
  /** Per two-qubit gate. */
  p2: number
}

/**
 * Every attack, in the order a report lists them before sorting.
 *
 * The single source of truth for the roster. `AttackName` is derived from it and
 * the runner table below is keyed by it, so adding an attack is this list plus
 * one entry — and forgetting the entry is a compile error rather than an attack
 * that silently never runs.
 */
export const ATTACK_NAMES = [
  'clifford', 'stabilizer-rank', 'mps', 'peps', 'tensor-network',
  'pauli-path', 'noisy-pauli-path', 'symmetry', 'statevector',
] as const

export type AttackName = typeof ATTACK_NAMES[number]

/**
 * What an attack actually produces.
 *
 * They are not interchangeable, and comparing their costs without this is how a
 * report ends up claiming a circuit is easy because one amplitude of it is.
 */
export type Answers = 'distribution' | 'amplitude' | 'expectation'

/** How much to trust the reported parameter. */
export type Bound =
  /** Measured. The cost follows. */
  | 'exact'
  /** The true value is at least this — the probe stopped early, so the cost is a floor. */
  | 'lower'
  /** The true value is at most this — a better search might do better, so the cost is a ceiling. */
  | 'upper'

export interface Attack {
  name: AttackName
  answers: Answers
  applicable: boolean
  /** Why this attack cannot be used at all. Present only when `applicable` is false. */
  reason?: string
  /** The structural quantity that decides this attack's cost. */
  parameter?: { name: string; value: number; bound: Bound }
  /**
   * The accuracy claim this attack is making, when it makes one.
   *
   * `parameter` says what the attack costs; this says how wrong its answer may be,
   * and the two are genuinely independent. An approximate attack can be cheap and
   * inaccurate at once, and until this existed the report could only express the
   * cheap half — every approximate attack computed an error figure and dropped it,
   * which left "broken" and "broken to within 1e-3" looking identical.
   *
   * **Present only when there is a real claim.** Absent means the attack asserts
   * nothing about its accuracy, which is the honest reading of a bare probe. So
   * `error === undefined` is not "no error", it is "no claim" — and for an attack
   * whose `parameter` is a floor, that is the only truthful thing to say.
   *
   * `bound` qualifies the claim the same way it does for `parameter`: `upper` is a
   * guarantee the error is no worse than this, which is what a converged attempt
   * and a sparsification budget each provide.
   */
  error?: { value: number; bound: Bound }
  memoryBytes?: number
  /** Elementary operations, before any throughput assumption. */
  operations?: number
  /** `operations` divided by the budget's `opsPerSecond`. An estimate, not a measurement. */
  seconds?: number
  /** Fits both the memory and the time budget, with a parameter solid enough to believe. */
  viable: boolean
}

export interface Budget {
  /** Host memory. Default 8 GB. */
  memoryBytes?: number
  /** Wall-clock the caller will accept. Default one hour. */
  seconds?: number
  /**
   * Elementary operations per second, used only to turn `operations` into
   * `seconds`. Default 2e8, measured for tensor contraction in Node on Apple
   * silicon. Memory is arithmetic; time is this constant, so treat the two
   * differently — memory decides what is impossible, this decides what is slow.
   */
  opsPerSecond?: number
  /** Approximation the stabilizer-rank attack is allowed. Default 0.1. */
  targetError?: number
  /** Observable for the Pauli-path attacks. Without one, they report why they were skipped. */
  observable?: readonly PauliTerm[]
  /**
   * The device's error rates.
   *
   * Given both this and an `observable`, the adversary also attacks the *noisy*
   * circuit — which is a different and easier problem than the ideal one, and is
   * the one a real device actually solves.
   */
  noise?: NoiseRates
  /** Bond dimension the MPS probe stops at. Default 64. Raising it costs χ³ per gate. */
  probeChi?: number
  /**
   * Terms to *attempt* the Pauli-path attacks with, instead of probing them.
   *
   * The distinction this module otherwise lacks. A probe is cheap and can only
   * rule an attack out: it caps the propagation, so the term count comes back as
   * a floor, and a floor never establishes viability. That systematically
   * under-reports the Pauli attacks, because on most circuits the propagation
   * does not converge inside a cheap cap — including the published Trotter
   * circuits that sparse Pauli dynamics actually reproduced, at millions of terms.
   *
   * Set this and the attack is attempted rather than probed: run at the budget,
   * run again at a quarter of it, and if the answer barely moved the propagation
   * has converged and the parameter is a measurement. Costs terms x gates, so a
   * million terms on a 5000-gate circuit is a minute, not a probe.
   */
  attemptTerms?: number
  /** Drift allowed between the two budgets before an attempt counts as converged. Default 0.01. */
  attemptTolerance?: number
  /**
   * Spend an attempt only where it could change the recommendation. Default false.
   *
   * Cost is monotone in the attempt budget — a larger cap retains more terms or a
   * wider bond, never fewer — so a probe's floor is a true lower bound, and two
   * cases are decided before paying anything: a floor already over budget stays
   * over budget, and a floor already dearer than an attack that landed cannot
   * displace it.
   *
   * Off by default, because it costs the meaning of `bound`. Normally `'lower'`
   * says "measured, and still moving"; under this flag it may also mean "not
   * attempted, because nothing it could have said would have mattered", and the
   * report cannot tell you which. Worth it when the output is a recommendation —
   * a 127-qubit target already broken by a 16 MB contraction should not spend ten
   * minutes resolving a Pauli attack that cannot win — and wrong when the output
   * is a cost table you intend to read row by row.
   */
  attemptOnlyIfDecisive?: boolean
  /**
   * Bond dimension to probe the 2-D (PEPS) attack with. Default 4.
   *
   * Deliberately not shared with {@link probeChi}. Both are called a bond
   * dimension and they are not the same quantity: an MPS bond cuts a chain, a
   * PEPS bond cuts one lattice edge, and a site of degree d holds χ^d rather than
   * χ². χ=64 is unremarkable for an MPS and would be enormous here, so one knob
   * serving both would mean any value that suited one was wrong for the other.
   */
  probeBond?: number
  /**
   * Bond dimension to *attempt* the 2-D attack with. Default 0, meaning probe only.
   *
   * Same contract as {@link attemptChi}: run at this and at twice it, and accept
   * the parameter as a measurement only if the answer barely moved. PEPS cannot
   * certify itself — its `discarded` figure bounds each local truncation, not the
   * final answer — so a truncated run stays a floor until two bond dimensions
   * agree.
   */
  attemptBond?: number
  /**
   * Gate count above which the *structurally expensive* probes are skipped.
   * Default 2000.
   *
   * A probe is supposed to be cheap — that is the whole basis for pricing every
   * attack before running any of them. Two of them are not cheap on a large
   * circuit: recursive-bisection contraction planning costs 349s on a 9k-gate
   * heavy-hex network where greedy costs 349ms, and replaying a circuit as a PEPS
   * costs 87s at bond 4. Left ungated they make *pricing* a circuit more expensive
   * than several of the attacks being priced, and `searchHardness` pays it once
   * per rewrite candidate.
   *
   * Past the limit the two degrade differently, and both stay honest:
   *
   * - The contraction attack falls back to greedy planning. Its width is an upper
   *   bound either way, so the verdict remains sound — just looser. Measured, that
   *   is width 78 instead of 55 at ten Trotter steps, and 183 instead of 127 at
   *   twenty. Neither is viable at any budget, which is why the cheap plan is the
   *   right default and the tight one is opt-in.
   * - The PEPS attack declines outright rather than reporting a floor it did not
   *   pay for, because a floor and a decline mean different things and conflating
   *   them is how a report starts lying.
   *
   * Raise it when the width matters more than the wait — a calibration run, say.
   * `Infinity` disables the gate.
   */
  probeMaxGates?: number
  /**
   * Bond dimension to *attempt* MPS with, accepting truncation.
   *
   * The exact MPS attack refuses any circuit whose χ exceeds the probe cap,
   * because the parameter is then a floor. But a truncated MPS is not a failed
   * exact one — it is the approximate method, and approximate tensor networks are
   * what actually reproduced the published Trotter circuits. Set this and the
   * attack runs at this χ and again at twice it: if the distribution stops moving,
   * the truncation is not costing anything measurable and the attack is real.
   *
   * Approximate, so the answer carries error the convergence check bounds
   * empirically and nothing certifies.
   */
  attemptChi?: number
  /**
   * Term cap for the Pauli-path probes. Default 4096.
   *
   * Same role as `probeChi`: it bounds the probe, not the attack. Cost is terms ×
   * gates, so on a 127-qubit depth-20 circuit a generous cap turns a probe into a
   * full run and the analysis takes minutes per candidate rewrite. A probe that
   * hits the cap reports a floor, which cannot establish viability anyway.
   */
  probeTerms?: number
  /** Restarts for the contraction planner. Default 16. */
  restarts?: number
  /**
   * The basis state the circuit starts from.
   *
   * Only the U(1) attack uses it, and it needs it: a conservation law is only
   * worth anything once you know which sector you are in. Started from |0…0⟩ a
   * conserving circuit never leaves a one-dimensional space, which is true and
   * useless, so this defaults to absent and that attack declines without it.
   */
  initialState?: string
}

type FilledBudget = Required<Omit<Budget, 'observable' | 'noise' | 'initialState'>>
  & { observable?: readonly PauliTerm[]; noise?: NoiseRates; initialState?: string }

export interface HardnessReport {
  qubits: number
  /** Gates, excluding barriers, measurements and resets. */
  gates: number
  /** Every attack, viable ones first and cheapest first within that. */
  attacks: Attack[]
  /** The cheapest viable attack, or null when none of them landed. */
  best: Attack | null
  /** The cheapest viable attack that produces a full distribution, which is the strongest product. */
  bestFull: Attack | null
  /**
   * How many times cheaper `best` is than the dense statevector.
   *
   * The number that answers "how much complexity did the structure remove".
   * Null when nothing landed, or when the statevector itself is the best attack.
   */
  advantage: number | null
  budget: FilledBudget
}

const GB = 1e9

function fill(b: Budget): FilledBudget {
  const out: FilledBudget = {
    memoryBytes: b.memoryBytes ?? 8 * GB,
    seconds: b.seconds ?? 3600,
    opsPerSecond: b.opsPerSecond ?? 2e8,
    targetError: b.targetError ?? 0.1,
    probeChi: b.probeChi ?? 64,
    probeTerms: b.probeTerms ?? 4096,
    attemptTerms: b.attemptTerms ?? 0,
    attemptChi: b.attemptChi ?? 0,
    attemptTolerance: b.attemptTolerance ?? 0.01,
    attemptOnlyIfDecisive: b.attemptOnlyIfDecisive ?? false,
    probeBond: b.probeBond ?? 4,
    attemptBond: b.attemptBond ?? 0,
    probeMaxGates: b.probeMaxGates ?? 2000,
    restarts: b.restarts ?? 16,
  }
  if (b.observable !== undefined) out.observable = b.observable
  if (b.noise !== undefined) out.noise = b.noise
  if (b.initialState !== undefined) out.initialState = b.initialState
  for (const [k, v] of [['memoryBytes', out.memoryBytes], ['seconds', out.seconds],
    ['opsPerSecond', out.opsPerSecond], ['probeChi', out.probeChi],
    ['probeTerms', out.probeTerms]] as const) {
    if (!(v > 0)) throw new RangeError(`analyzeHardness: ${k} must be positive, got ${v}`)
  }
  return out
}

const message = (e: unknown): string => e instanceof Error ? e.message : String(e)

/** Assemble an attack, deciding viability from the budget and the parameter's bound. */
function priced(
  name: AttackName, answers: Answers, budget: FilledBudget,
  parameter: { name: string; value: number; bound: Bound },
  memoryBytes: number, operations: number,
  error?: { value: number; bound: Bound },
): Attack {
  const seconds = operations / budget.opsPerSecond
  // A lower bound cannot establish viability: the true cost is at least this and
  // may be anything above it. Claiming a fit from a floor is the one way this
  // module could report a break that does not exist.
  const viable = parameter.bound !== 'lower'
    && memoryBytes <= budget.memoryBytes
    && seconds <= budget.seconds
  const a: Attack = { name, answers, applicable: true, parameter, memoryBytes, operations, seconds, viable }
  if (error !== undefined) a.error = error
  return a
}

const declined = (name: AttackName, answers: Answers, reason: string): Attack =>
  ({ name, answers, applicable: false, reason, viable: false })

// ── The attacks ───────────────────────────────────────────────────────────────

/** Dense statevector: the baseline every other attack is measured against. */
function statevectorAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  const n = circuit.qubits
  return priced('statevector', 'distribution', budget,
    { name: 'qubits', value: n, bound: 'exact' },
    // Two f64 per amplitude.
    2 ** n * 16,
    gates * 2 ** n,
    // Exact method: this break is the real answer, not an approximation of it.
    { value: 0, bound: 'exact' })
}

const CLIFFORD_1Q = new Set(['h', 'x', 'y', 'z', 's', 'si', 'sdg', 'id', 'v', 'vi'])
const CLIFFORD_2Q = new Set(['cx', 'cy', 'cz'])
/** Rotations that land in the Clifford group at the right angle. */
const ROTATION_1Q = new Set(['rx', 'ry', 'rz', 'u1', 'p'])

/**
 * Is this rotation angle a Clifford one?
 *
 * `Rz(θ)` is `S` at π/2, `Z` at π and `S†` at 3π/2 — Clifford at every multiple of
 * π/2 and nowhere else, and the same holds for `Rx`, `Ry` and `u1`. Deciding
 * Clifford-ness from the gate *name* alone therefore gets it wrong on exactly the
 * circuits where it matters most: the published Trotter family fixes its coupling
 * angle at −π/2, so `Rzz` is Clifford, and at the ends of the transverse-field
 * sweep the whole circuit is. Name-based detection reports those as unbreakable
 * when Gottesman–Knill solves them in polynomial time.
 */
const cliffordAngle = (theta: number): boolean => {
  const r = Math.abs(theta) % (Math.PI / 2)
  return Math.min(r, Math.PI / 2 - r) < 1e-9
}

/**
 * Clifford closure — the attack that costs nothing when it applies.
 *
 * Detected from the serialized gate names rather than by attempting a run,
 * because a circuit one T gate away from Clifford should be rejected instantly
 * rather than after a simulation.
 */
function cliffordAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  const n = circuit.qubits
  for (const raw of circuit.toJSON().ops) {
    const op = raw as { kind: string; meta?: { name?: string; params?: number[] } }
    switch (op.kind) {
      case 'barrier': case 'measure': case 'reset': case 'cnot': case 'swap': break
      case 'single': {
        const name = op.meta?.name ?? '?'
        if (CLIFFORD_1Q.has(name)) break
        if (ROTATION_1Q.has(name)) {
          const theta = (op.meta?.params?.[0]) ?? 0
          if (cliffordAngle(theta)) break
          return declined('clifford', 'distribution', `${name}(${theta.toFixed(4)}) is not a Clifford angle`)
        }
        return declined('clifford', 'distribution', `gate '${name}' is not Clifford`)
      }
      case 'controlled':
        if (!CLIFFORD_2Q.has(op.meta?.name ?? '')) {
          return declined('clifford', 'distribution', `gate '${op.meta?.name ?? '?'}' is not Clifford`)
        }
        break
      default:
        return declined('clifford', 'distribution', `op '${op.kind}' is not Clifford`)
    }
  }
  // Two n x n bit tableaus plus sign vectors, packed 32 bits to a word.
  return priced('clifford', 'distribution', budget,
    { name: 'T-count', value: 0, bound: 'exact' },
    2 * n * Math.ceil(n / 32) * 4 + n,
    gates * n * Math.ceil(n / 32),
    // Exact method: this break is the real answer, not an approximation of it.
    { value: 0, bound: 'exact' })
}

/**
 * Low-rank stabilizer decomposition — the attack that prices magic.
 *
 * `countSplits` is the T-count and `termBudget` turns it into a term count at the
 * allowed approximation, both of which are arithmetic on the gate list. Nothing
 * is simulated to get here.
 */
function stabilizerRankAttack(circuit: Circuit, budget: FilledBudget): Attack {
  let ops
  try { ops = circuit.toSrOps() } catch (e) { return declined('stabilizer-rank', 'distribution', message(e)) }
  const t = countSplits(ops)
  const terms = termBudget(ops, budget.targetError)
  // PEAK_FACTOR 4 — a split allocates the doubled arrays while the originals are
  // still live, and GC lags. Mirrors stabilizer-capacity.ts.
  return priced('stabilizer-rank', 'distribution', budget,
    { name: 'T-count', value: t, bound: 'exact' },
    terms * bytesPerTerm(circuit.qubits) * 4,
    terms * ops.length * circuit.qubits,
    // The term budget is derived *from* the target error, so this attack is the one
    // place accuracy is an input rather than an outcome — it buys exactly the δ it
    // was asked for. A ceiling, and only for a single application: repeated
    // sparsification in a streaming run compounds beyond what the lemma bounds.
    t > 0 ? { value: budget.targetError, bound: 'upper' } : { value: 0, bound: 'exact' })
}

/**
 * Matrix product state — the attack that prices entanglement.
 *
 * χ cannot be known without running, so the probe runs with a cap and reads back
 * whether it bound. Not truncated means χ is exact and the cost follows; truncated
 * means χ is only known to exceed the cap, which is a floor and cannot establish
 * that anything fits.
 */
function mpsAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  let peakChi: number, truncated: boolean
  try {
    if (budget.attemptChi > 0) {
      /**
       * Raise the ceiling the probe can confirm, without claiming more.
       *
       * The obvious design — run at χ and 2χ and compare the sampled
       * distributions — is defeated by shot noise: at a few thousand shots the
       * noise floor is larger than any tolerance worth setting, so a volume-law
       * circuit whose truncation is doing real damage reads as converged. That
       * is precisely the failure mode this module exists to avoid.
       *
       * The noise-free signal is whether the doubled run *saturated its own cap*.
       * If it did not, 2χ exceeded the circuit's true bond dimension, the run was
       * exact, and χ is a measurement. If it did, the bond dimension is still
       * growing and nothing here can say what the truncation cost.
       *
       * This therefore does **not** deliver the literature's approximate method —
       * a genuinely truncated run stays a floor. It only lets the attack confirm
       * a larger χ than a cheap probe could afford.
       */
      /**
       * Run truncated at χ and at 2χ and compare *expectation values*, which
       * carry no shot noise, rather than sampled distributions, whose noise floor
       * at any affordable shot count exceeds a useful tolerance.
       *
       * Compared across a spread of sites rather than one, because the truncation
       * error is **not monotone in χ**: measured on an RCS grid it fell
       * 4.2e-1 → 1.0e-1 → 3.7e-2 → 7.8e-3 and then rose to 2.4e-2 before
       * collapsing. A single observable can cross the tolerance band by accident
       * on the way past; several agreeing at once is far harder to fake.
       */
      const n = circuit.qubits
      const ops = circuit.toOps()
      const lo = mpsFromOps(n, ops, budget.attemptChi)
      const hi = mpsFromOps(n, ops, budget.attemptChi * 2)
      if (!hi.truncated) {
        // 2χ never bound, so that run was exact and χ is a measurement.
        const chiExact = Math.max(1, hi.peakChi)
        return priced('mps', 'distribution', budget,
          { name: 'bond dimension χ', value: chiExact, bound: 'exact' },
          n * 2 * chiExact * chiExact * 16, gates * chiExact ** 3,
          // 2χ never bound, so nothing was truncated and the state is the exact one.
          { value: 0, bound: 'exact' })
      }
      const sites = Array.from({ length: Math.min(n, 8) }, (_, i) => Math.floor(i * n / Math.min(n, 8)))
      let drift = 0
      for (const q of sites) {
        for (const p of ['Z', 'X'] as const) {
          drift = Math.max(drift, Math.abs(mpsExpectation(lo.mps, q, p) - mpsExpectation(hi.mps, q, p)))
        }
      }
      const settled = drift <= budget.attemptTolerance
      const chiUsed = Math.max(1, lo.peakChi)
      return priced('mps', 'distribution', budget,
        { name: 'bond dimension χ', value: settled ? chiUsed : budget.attemptChi * 2, bound: settled ? 'exact' : 'lower' },
        n * 2 * chiUsed * chiUsed * 16,
        gates * chiUsed ** 3,
        // Truncated but settled: accurate *to* the tolerance it met, not exactly.
        // Unsettled makes no claim at all.
        settled ? { value: budget.attemptTolerance, bound: 'upper' } : undefined)
    }
    const d = circuit.runMps({ shots: 1, seed: 1, maxChi: budget.probeChi })
    peakChi = d.peakChi ?? 1
    truncated = d.truncated
  } catch (e) {
    return declined('mps', 'distribution', message(e))
  }
  const chi = Math.max(1, peakChi)
  // n sites of chi x 2 x chi complex doubles.
  return priced('mps', 'distribution', budget,
    { name: 'bond dimension χ', value: chi, bound: truncated ? 'lower' : 'exact' },
    circuit.qubits * 2 * chi * chi * 16,
    gates * chi ** 3,
    truncated ? undefined : { value: 0, bound: 'exact' })
}

/**
 * Projected entangled-pair state — the attack that prices *two-dimensional*
 * entanglement.
 *
 * The MPS attack prices entanglement across a chain, and a lattice has no good
 * linear embedding: on the heavy-hex circuits the truncated MPS answer was still
 * moving 10% between χ=32 and 64, because a 1-D cut through a 2-D state carries
 * bonds it should not have to. This keeps one tensor per qubit and one bond per
 * coupling edge, so the network has the circuit's own shape.
 *
 * No lattice is assumed or required: the bonds are the pairs the circuit's own
 * two-qubit gates connect.
 *
 * Approximate, and it cannot certify itself — `discarded` bounds each local
 * truncation rather than the final answer — so anything thrown away leaves the
 * parameter a floor, and only an attempt at two bond dimensions can turn it into
 * a measurement. Same contract as {@link mpsAttack}.
 */
function pepsAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  const n = circuit.qubits

  /** Price a state from the tensors it actually holds, rather than from a model. */
  const price = (state: PepsState, bond: number, bound: Bound,
                 error?: { value: number; bound: Bound }): Attack => {
    let bytes = 0
    let degree = 1
    for (const s of state.sites) {
      bytes += s.data.length * 8
      degree = Math.max(degree, s.bonds.length)
    }
    // A two-site update decomposes a block whose side is 2·χ^(d−1), and that
    // decomposition is cubic in the side. MPS is the d=2 case, where this reduces
    // to the χ³ the MPS attack charges. Degree is read off the network, so a
    // sparse lattice is not charged for a dense one.
    const perGate = 8 * bond ** (3 * Math.max(1, degree - 1))
    return priced('peps', 'expectation', budget,
      { name: 'bond dimension', value: bond, bound }, bytes, gates * perGate, error)
  }

  // Replaying a circuit as a PEPS is not a cheap probe — 87s at bond 4 on a
  // 9k-gate heavy-hex network — and a probe that costs more than the attacks it
  // is pricing defeats the point. Declining is deliberate: a floor and a decline
  // say different things, and reporting a floor nobody paid for would be a lie.
  if (gates > budget.probeMaxGates) {
    return declined('peps', 'expectation',
      `circuit is ${gates} gates, past probeMaxGates ${budget.probeMaxGates} — raise it to price this attack`)
  }

  try {
    if (budget.attemptBond > 0) {
      const lo = pepsRun(circuit, { maxBond: budget.attemptBond })
      const hi = pepsRun(circuit, { maxBond: budget.attemptBond * 2 })
      // Nothing discarded at 2χ means that run was exact, so its bond is a
      // measurement and no comparison is needed.
      if (hi.discarded === 0) return price(hi, Math.max(1, hi.maxBondUsed), 'exact', { value: 0, bound: 'exact' })
      // Compared across a spread of sites, and on both Z and X, for the reason
      // given on the MPS attack: truncation error is not monotone in the bond
      // dimension, so a single observable can cross a tolerance band on the way
      // past. Expectation values carry no shot noise.
      const probes = Math.min(n, 8)
      let drift = 0
      for (let i = 0; i < probes; i++) {
        const q = Math.floor(i * n / probes)
        for (const p of ['Z', 'X'] as const) {
          drift = Math.max(drift, Math.abs(pepsExpectation(lo, q, p) - pepsExpectation(hi, q, p)))
        }
      }
      const settled = drift <= budget.attemptTolerance
      return price(lo, settled ? Math.max(1, lo.maxBondUsed) : budget.attemptBond * 2,
        settled ? 'exact' : 'lower',
        settled ? { value: budget.attemptTolerance, bound: 'upper' } : undefined)
    }
    const state = pepsRun(circuit, { maxBond: budget.probeBond })
    return price(state, Math.max(1, state.maxBondUsed), state.discarded > 0 ? 'lower' : 'exact',
      state.discarded > 0 ? undefined : { value: 0, bound: 'exact' })
  } catch (e) {
    return declined('peps', 'expectation', message(e))
  }
}

/** Gates past which even greedy contraction planning is refused. See below. */
const GREEDY_MAX_GATES = 50_000

/**
 * Tensor-network contraction — the attack that prices treewidth.
 *
 * The width is whatever the planner found, so it is an upper bound on the best
 * order: a stronger search would only lower it, never raise it. That makes the
 * cost a ceiling, and a viable verdict here is trustworthy in the direction that
 * matters — if this width fits, the true optimum fits too.
 */
function tensorNetworkAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  // The cheap tier is only cheap relative to the network. Greedy planning is
  // 0.8s on a 33,794-tensor Trotter network and 3.1s on Shor N=33 (35,053), but
  // 44s on Shor N=1003 (191,583) — the cost is superlinear in tensor count. That
  // last one spends 44 seconds to report width 81, which no budget could ever
  // meet, so the caller paid a wait for nothing. Tensors track gates within 10%
  // on every circuit measured, so this is stated in gates as probeMaxGates is.
  // Only reachable on the greedy tier: `probeMaxGates: Infinity` opts into the
  // thorough planner and out of this, which is what a calibration run wants.
  if (gates > budget.probeMaxGates && gates > GREEDY_MAX_GATES) {
    return declined('tensor-network', 'amplitude',
      `circuit is ${gates} gates, past ${GREEDY_MAX_GATES} — planning a contraction ` +
      `order costs longer than the verdict is worth`)
  }
  let plan
  let netBytes = 0
  try {
    const net = circuitNetwork(circuit, '0'.repeat(circuit.qubits))
    for (const t of net) netBytes += t.data.length * 8
    const indices = net.map(t => t.indices)
    // `planBest` over `planContraction` because which planner wins is a property
    // of the network, and pricing an attack against the weaker of two available
    // plans understates it — bisection reaches width 55 where greedy reaches 78 on
    // a ten-step heavy-hex network. But it costs a thousandfold in planning time
    // on a deep circuit, so past `probeMaxGates` the cheap plan is used instead.
    // The width is an upper bound either way, so a looser plan loses tightness,
    // never soundness.
    plan = gates > budget.probeMaxGates
      ? planContraction(indices, { restarts: budget.restarts })
      : planBest(indices, { restarts: budget.restarts })
  } catch (e) {
    return declined('tensor-network', 'amplitude', message(e))
  }
  // The largest intermediate is 2^width amplitudes, but the input network has to
  // be resident too — 624 small tensors are 61 KB on a 54-qubit RCS circuit, which
  // is the same order as a width-12 intermediate. Reporting only the intermediate
  // understated peak memory by about half on exactly the circuits that land.
  return priced('tensor-network', 'amplitude', budget,
    { name: 'contraction width', value: plan.width, bound: 'upper' },
    2 ** plan.width * 16 + netBytes,
    plan.cost,
    // Exact method: this break is the real answer, not an approximation of it.
    { value: 0, bound: 'exact' })
}

/**
 * Pauli-path propagation — the attack that prices Heisenberg-picture sparsity.
 *
 * Costed against a specific observable, because the term growth is a property of
 * the observable as much as of the circuit. Without one there is nothing to
 * price, and saying so beats inventing a default and reporting its cost as the
 * circuit's.
 */
/**
 * Run a Pauli propagation as a probe, or as an attempt with a convergence check.
 *
 * The attempt is what turns a floor into a measurement: converged means the
 * answer stopped moving when the budget quadrupled, so the term count is real and
 * the attack can be ruled in. Not converged means the cap still binds, which is
 * exactly what the cheap probe would have said.
 */
function propagate(
  circuit: Circuit, budget: FilledBudget, noise: NoiseRates | undefined,
): { terms: number; bound: Bound; error?: { value: number; bound: Bound } } {
  const observable = budget.observable!
  const run = (maxTerms: number): { value: number; peakTerms: number; truncated: boolean } =>
    pauliPathExpectation(circuit, observable, { maxTerms, ...(noise !== undefined && { noise }) })

  // Nothing discarded means nothing approximated: the answer is the exact one, and
  // that is a stronger claim than any tolerance. Kept distinct from a converged
  // truncation, which is only accurate *to* a tolerance.
  const exact = { value: 0, bound: 'exact' as const }

  if (budget.attemptTerms <= 0) {
    const r = run(budget.probeTerms)
    return r.truncated
      ? { terms: r.peakTerms, bound: 'lower' }
      : { terms: r.peakTerms, bound: 'exact', error: exact }
  }
  const coarse = run(Math.max(1, Math.floor(budget.attemptTerms / 4)))
  const fine = run(budget.attemptTerms)
  if (!fine.truncated) return { terms: fine.peakTerms, bound: 'exact', error: exact }
  const drift = Math.abs(fine.value - coarse.value) / Math.max(Math.abs(fine.value), 1e-12)
  // Converged under a 4x budget increase: the truncation is no longer moving the
  // answer, so the cost stands even though terms were formally discarded. The
  // accuracy that buys is the tolerance it met, not zero.
  return drift <= budget.attemptTolerance
    ? { terms: fine.peakTerms, bound: 'exact', error: { value: budget.attemptTolerance, bound: 'upper' } }
    : { terms: fine.peakTerms, bound: 'lower' }
}

function pauliPathAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  if (budget.observable === undefined) {
    return declined('pauli-path', 'expectation', 'no observable given — set budget.observable to price this attack')
  }
  let got: { terms: number; bound: Bound; error?: { value: number; bound: Bound } }
  try {
    got = propagate(circuit, budget, undefined)
  } catch (e) {
    return declined('pauli-path', 'expectation', message(e))
  }
  const peakTerms = got.terms
  return priced('pauli-path', 'expectation', budget,
    { name: 'Pauli terms', value: peakTerms, bound: got.bound },
    // One letter per qubit plus a coefficient and object overhead.
    peakTerms * (circuit.qubits * 2 + 64),
    gates * peakTerms, got.error)
}

/**
 * The same propagation, against the noisy circuit — the problem hardware solves.
 *
 * Depolarizing damps a weight-w term by λ^w per layer, so the heavy terms that
 * make the ideal propagation expensive are precisely the ones noise removes. This
 * is the only attack here that gets *cheaper* as the device gets worse, and its
 * verdict is about a different quantity: not ⟨O⟩ of the circuit, but ⟨O⟩ of the
 * machine.
 */
function noisyPauliPathAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  if (budget.observable === undefined) {
    return declined('noisy-pauli-path', 'expectation', 'no observable given — set budget.observable to price this attack')
  }
  if (budget.noise === undefined) {
    return declined('noisy-pauli-path', 'expectation', 'no device noise given — set budget.noise to attack the noisy circuit')
  }
  let got: { terms: number; bound: Bound; error?: { value: number; bound: Bound } }
  try {
    got = propagate(circuit, budget, budget.noise)
  } catch (e) {
    return declined('noisy-pauli-path', 'expectation', message(e))
  }
  const peakTerms = got.terms
  return priced('noisy-pauli-path', 'expectation', budget,
    { name: 'Pauli terms (noisy)', value: peakTerms, bound: got.bound },
    peakTerms * (circuit.qubits * 2 + 64),
    gates * peakTerms, got.error)
}

/**
 * U(1) symmetry — the attack that prices a conservation law.
 *
 * The only attack here that exploits something the circuit *cannot do* rather
 * than something it does cheaply. If no gate can change the excitation count, the
 * reachable space is C(n,k) instead of 2^n, and away from half filling that is
 * orders of magnitude: 50 qubits holding 10 excitations is 1.0e10 against 1.1e15.
 *
 * The parameter is an upper bound because the detection is sufficient rather than
 * necessary — a circuit may carry more symmetry than a gate-by-gate test finds,
 * which would only make the reachable space smaller.
 */
function symmetryAttack(circuit: Circuit, gates: number, budget: FilledBudget): Attack {
  if (budget.initialState === undefined) {
    return declined('symmetry', 'distribution',
      'no initial state given — set budget.initialState, since a conservation law needs a sector')
  }
  let sym
  try { sym = weightSymmetry(circuit, budget.initialState) }
  catch (e) { return declined('symmetry', 'distribution', message(e)) }
  if (!sym.conserved) {
    return declined('symmetry', 'distribution', `excitation count is not conserved — broken by ${sym.brokenBy}`)
  }
  return priced('symmetry', 'distribution', budget,
    { name: 'sector dimension C(n,k)', value: sym.sectorDimension, bound: 'upper' },
    sym.sectorDimension * 16,
    gates * sym.sectorDimension,
    // Exact method: this break is the real answer, not an approximation of it.
    { value: 0, bound: 'exact' })
}

// ── The report ────────────────────────────────────────────────────────────────

/**
 * Run every classical attack's probe against a circuit and price the results.
 *
 * @example
 * const r = analyzeHardness(circuit, { memoryBytes: 16e9 })
 * if (r.best) console.log(`broken by ${r.best.name}, ${r.advantage?.toFixed(0)}x cheaper than a statevector`)
 * else        console.log('no attack in this library landed at this budget')
 */
export function analyzeHardness(circuit: Circuit, budget: Budget = {}): HardnessReport {
  const b = fill(budget)

  /**
   * Every attack has to be pricing the same problem.
   *
   * Most backends here start from |0…0⟩, and the U(1) attack is priced from the
   * initial state it was given — so with a populated start they were answering
   * different questions, and the MPS probe would report χ=1 for a circuit that is
   * trivial from the vacuum and entangling from anywhere else. Prepending the
   * state-preparation layer puts them all on the same footing.
   *
   * The symmetry attack keeps the original circuit, because those X gates are
   * exactly what a conservation law forbids and prepending them would destroy the
   * symmetry being measured.
   */
  const prepared = b.initialState === undefined || !b.initialState.includes('1')
    ? circuit
    : (() => {
        let out = new Circuit(circuit.qubits)
        for (let q = 0; q < circuit.qubits; q++) if (b.initialState![q] === '1') out = out.x(q)
        return out.compose(circuit)
      })()
  const gates = Math.max(1, prepared.gateCounts().total)

  /**
   * With {@link Budget.attemptOnlyIfDecisive}, price from the cheap probes first
   * and pay for resolution only where it could still change the recommendation.
   *
   * An attempt runs the whole propagation twice — at a quarter budget and at full
   * — so it costs orders of magnitude more than a probe, and most of the time it
   * cannot alter a single verdict. The flag's doc comment has the argument for
   * why a floor is enough to decide that in advance, and what the flag costs.
   */
  /**
   * How to run each attack, and which budget knob can resolve its floor.
   *
   * Keyed by {@link AttackName}, so the table is exhaustive by construction —
   * an attack added to `ATTACK_NAMES` without a runner here does not compile.
   * That matters because the failure it replaces was silent: an attack missing
   * from a hand-written array simply never ran, and a report with a row absent
   * looks exactly like a report with a row that declined.
   *
   * `attempt` names the field whose value turns this attack's probe into an
   * attempt. Absent means the attack has no approximation to resolve, so a bigger
   * budget buys it nothing.
   */
  const runners: Record<AttackName, {
    run: (budget: FilledBudget) => Attack
    attempt?: 'attemptTerms' | 'attemptChi' | 'attemptBond'
  }> = {
    'clifford':         { run: bud => cliffordAttack(prepared, gates, bud) },
    'stabilizer-rank':  { run: bud => stabilizerRankAttack(prepared, bud) },
    'mps':              { run: bud => mpsAttack(prepared, gates, bud), attempt: 'attemptChi' },
    'peps':             { run: bud => pepsAttack(prepared, gates, bud), attempt: 'attemptBond' },
    'tensor-network':   { run: bud => tensorNetworkAttack(prepared, gates, bud) },
    'pauli-path':       { run: bud => pauliPathAttack(prepared, gates, bud), attempt: 'attemptTerms' },
    'noisy-pauli-path': { run: bud => noisyPauliPathAttack(prepared, gates, bud), attempt: 'attemptTerms' },
    'symmetry':         { run: bud => symmetryAttack(circuit, gates, bud) },
    'statevector':      { run: bud => statevectorAttack(prepared, gates, bud) },
  }

  /**
   * With {@link Budget.attemptOnlyIfDecisive}, price from the cheap probes first
   * and pay for resolution only where it could still change the recommendation.
   *
   * An attempt runs the whole propagation twice — at a quarter budget and at full
   * — so it costs orders of magnitude more than a probe, and most of the time it
   * cannot alter a single verdict. The flag's doc comment has the argument for
   * why a floor is enough to decide that in advance, and what the flag costs.
   *
   * The set of knobs to suppress is folded out of the runner table rather than
   * listed, so a new attempt budget cannot be forgotten here.
   */
  const knobs = [...new Set(ATTACK_NAMES.map(n => runners[n].attempt).filter(k => k !== undefined))]
  const staged = b.attemptOnlyIfDecisive && knobs.some(k => b[k] > 0)
  const first: FilledBudget = staged
    ? { ...b, ...Object.fromEntries(knobs.map(k => [k, 0])) }
    : b

  const attacks: Attack[] = ATTACK_NAMES.map(n => runners[n].run(first))

  if (staged) {
    // The bar a floor has to beat to be worth resolving: the cheapest attack that
    // already landed. Infinity when nothing did, which is exactly the case where
    // every attempt is worth paying for.
    let bar = Infinity
    for (const a of attacks) {
      if (a.viable && a.seconds !== undefined) bar = Math.min(bar, a.seconds)
    }
    for (let i = 0; i < attacks.length; i++) {
      const a = attacks[i]!
      const { run, attempt } = runners[ATTACK_NAMES[i]!]
      // Only a floor can move, and only if a budget was actually offered. 'exact'
      // is settled and 'upper' is already a ceiling, so neither gains anything.
      if (attempt === undefined || b[attempt] <= 0 || a.parameter?.bound !== 'lower') continue
      const seconds = a.seconds ?? 0
      // Monotone cost: over budget at the floor is over budget for real.
      if ((a.memoryBytes ?? 0) > b.memoryBytes || seconds > b.seconds) continue
      if (seconds >= bar) continue
      attacks[i] = run(b)
    }
  }

  // Viable first, then cheapest. An inapplicable attack has no cost to sort on
  // and goes last regardless.
  attacks.sort((x, y) => {
    if (x.viable !== y.viable) return x.viable ? -1 : 1
    if (x.applicable !== y.applicable) return x.applicable ? -1 : 1
    return (x.seconds ?? Infinity) - (y.seconds ?? Infinity)
  })

  const best = attacks.find(a => a.viable) ?? null
  const bestFull = attacks.find(a => a.viable && a.answers === 'distribution') ?? null
  const sv = attacks.find(a => a.name === 'statevector')!
  const advantage = best && best.name !== 'statevector' && best.operations !== undefined && best.operations > 0
    ? sv.operations! / best.operations
    : null

  return { qubits: circuit.qubits, gates, attacks, best, bestFull, advantage, budget: b }
}

// ── Searching for a cheaper presentation ──────────────────────────────────────

export interface SearchOptions extends Budget {
  /**
   * Qubits the answer is about. Given these, the light cone can delete everything
   * that cannot reach them — usually the largest single reduction available, and
   * the reason to supply them.
   */
  about?: readonly number[]
  /** Try cancelling and merging neighbouring gates. Default true. */
  peephole?: boolean
  /** Search for a qubit relabelling that shortens two-qubit gates. Default true. */
  reorder?: boolean
  /**
   * Try rewriting transverse rotations as `H·Rz·H`. Default true.
   *
   * Raises the gate count, so it only survives when the eligibility it buys is
   * worth more than the size it costs — which the search decides by re-pricing,
   * not by guessing.
   */
  zBasis?: boolean
  /** Restarts for the ordering search. Default 8. */
  orderRestarts?: number
}

export interface HardnessSearch {
  /** The circuit as handed over. */
  before: HardnessReport
  /** The cheapest presentation found. */
  after: HardnessReport
  /** Rewrites applied, in order. */
  applied: Rewrite[]
  /**
   * Rewrites that were tried and did not help.
   *
   * Worth reporting because a discarded rewrite leaves the *pre-rewrite* refusal
   * in the report. `toZBasis` can make the magic attack applicable to a circuit
   * written with `rx`, and when the search rejects it — because the T-count is
   * hopeless either way — the surviving row still reads "gate 'rx' is not
   * supported", which looks like a hard incompatibility rather than a decision.
   */
  rejected: { name: string; before: number; after: number }[]
  /** The rewritten circuit, so a caller can run the attack the report describes. */
  circuit: Circuit
  /**
   * Whether searching turned a circuit nothing could break into one something can.
   *
   * The outcome worth naming: not "it got cheaper" but "it stopped being hard".
   */
  broke: boolean
  /** Cheapest viable attack after over cheapest before, when both exist. */
  speedup: number | null
}

/**
 * Carry a Pauli observable through a rewrite that moved the qubits.
 *
 * A Pauli string is written qubit n−1 first, so a rewrite that renumbers or drops
 * qubits invalidates it — and an observable of the wrong length is refused by the
 * Pauli backends, which silently removes exactly the attacks a light cone was
 * supposed to enable. Returns null when a non-identity letter sits on a qubit the
 * rewrite discarded, since there is then no honest way to carry it across.
 */
function remapObservable(
  observable: readonly PauliTerm[], mapping: readonly number[], oldQubits: number, newQubits: number,
): PauliTerm[] | null {
  const out: PauliTerm[] = []
  for (const { coeff, ops } of observable) {
    const letters = new Array<string>(newQubits).fill('I')
    for (let q = 0; q < oldQubits; q++) {
      const letter = ops[oldQubits - 1 - q]!
      const to = mapping[q]
      if (to === undefined || to < 0) {
        if (letter !== 'I') return null           // the observable reached outside the cone
        continue
      }
      letters[newQubits - 1 - to] = letter
    }
    out.push({ coeff, ops: letters.join('') })
  }
  return out
}

/**
 * Attack the circuit's *presentation*, not just the circuit.
 *
 * A circuit as written is one representative of everything computing the same
 * answer, and the hardness parameters are properties of the representative. An
 * attacker rewrites first. This applies the sound rewrites in `rewrite.ts`, keeps
 * whichever actually help, and re-prices.
 *
 * The rewrites narrow what the result answers, and that narrowing is reported
 * rather than hidden: with `about` supplied, everything outside those qubits'
 * light cone is gone and the report describes a circuit that answers only about
 * them. Check `applied` for what was done before quoting the verdict.
 */
export function searchHardness(circuit: Circuit, options: SearchOptions = {}): HardnessSearch {
  const { about, peephole: doPeephole = true, reorder = true, zBasis = true,
    orderRestarts = 8, ...budget } = options

  // Attempting is for the winner, not for every candidate. A search tries several
  // rewrites and prices each, so leaving `attemptTerms` set here would run a
  // million-term propagation per candidate per attack — minutes of work to decide
  // whether a rewrite was worth keeping. Search on probes, then attempt once.
  const { attemptTerms, ...searchBudget } = budget
  const attempting = (attemptTerms ?? 0) > 0
  const finalBudget: Budget = attempting ? budget : searchBudget

  const before = analyzeHardness(circuit, attempting ? finalBudget : searchBudget)

  const applied: Rewrite[] = []
  const rejected: { name: string; before: number; after: number }[] = []
  let current = circuit
  let currentBudget: Budget = searchBudget
  let currentReport = attempting ? analyzeHardness(circuit, searchBudget) : before

  /**
   * Is `a` a better presentation than `b`?
   *
   * Judged on the report, never on a proxy. An earlier version accepted the
   * qubit relabelling whenever it shortened the total two-qubit span — a
   * reasonable-sounding stand-in for bond dimension — and it made the measured
   * result *worse*, because a shorter span is not the same as a cheaper
   * contraction. A search that accepts moves on a proxy can walk downhill.
   */
  const better = (a: HardnessReport, b: HardnessReport): boolean => {
    if ((a.best === null) !== (b.best === null)) return a.best !== null   // breaking beats everything
    if (a.best === null || b.best === null) return a.gates < b.gates      // neither breaks: prefer smaller
    return (a.best.operations ?? Infinity) < (b.best.operations ?? Infinity)
  }

  // Light cone first: it deletes qubits, which makes every later search smaller.
  const candidates: (() => Rewrite)[] = []
  if (about !== undefined && about.length > 0) candidates.push(() => lightCone(current, about))
  if (doPeephole) candidates.push(() => peephole(current))
  if (zBasis) candidates.push(() => toZBasis(current))
  if (reorder) candidates.push(() => searchQubitOrder(current, { restarts: orderRestarts }))

  for (const make of candidates) {
    if (reorder && current.qubits <= 2) continue
    const r = make()
    // The observable is indexed by qubit, so a rewrite that renumbers or drops
    // qubits has to carry it across. Skipping this leaves a stale-length string
    // that both Pauli attacks refuse — which quietly deletes them from the report
    // precisely when the light cone has just made them affordable.
    let nextBudget: Budget = currentBudget
    if (r.mapping !== undefined && currentBudget.observable !== undefined) {
      const moved = remapObservable(currentBudget.observable, r.mapping, current.qubits, r.circuit.qubits)
      if (moved === null) {
        // The observable reached outside the cone, so it cannot be carried. Drop
        // it rather than pass a stale one: the Pauli attacks then decline with
        // "no observable given", which is true, instead of a length mismatch.
        const { observable: _dropped, ...rest } = currentBudget
        nextBudget = rest
      } else {
        nextBudget = { ...currentBudget, observable: moved }
      }
    }
    const report = analyzeHardness(r.circuit, nextBudget)
    if (!better(report, currentReport)) { rejected.push({ name: r.name, before: r.before, after: r.after }); continue }
    applied.push(r)
    current = r.circuit
    currentBudget = nextBudget
    currentReport = report
  }

  // One attempt, on the presentation the search settled on.
  const after = attempting
    ? analyzeHardness(current, { ...currentBudget, attemptTerms: attemptTerms! })
    : currentReport
  return {
    before, after, applied, rejected, circuit: current,
    broke: before.best === null && after.best !== null,
    // Only a real improvement is a speedup; the search never accepts a move that
    // makes things worse, so anything at or below 1 is reported as no change.
    speedup: before.best?.operations !== undefined && after.best?.operations
      && after.best.operations < before.best.operations
      ? before.best.operations / after.best.operations
      : null,
  }
}

/** A search result as an ASCII summary, with the rewrites that got there. */
export function renderSearch(s: HardnessSearch): string {
  const out = [
    `search: ${s.before.qubits} qubits / ${s.before.gates} gates` +
    ` → ${s.after.qubits} qubits / ${s.after.gates} gates`,
  ]
  for (const r of s.applied) {
    out.push(`  ${r.name.padEnd(12)} ${r.before} → ${r.after} gates` +
      `${r.circuit.qubits !== s.before.qubits ? `, ${r.circuit.qubits} qubits` : ''}` +
      `  (preserves ${r.preserves})`)
  }
  if (s.applied.length === 0) out.push('  no rewrite helped')
  for (const r of s.rejected) {
    out.push(`  ${r.name.padEnd(12)} tried, discarded (${r.before} → ${r.after} gates, no improvement)`)
  }
  out.push('')
  out.push(`before: ${s.before.best ? s.before.best.name : 'nothing landed'}`)
  out.push(`after:  ${s.after.best ? s.after.best.name : 'nothing landed'}` +
    (s.speedup ? ` — ${s.speedup.toExponential(1)}x cheaper` : ''))
  if (s.broke) out.push('the search broke a circuit that nothing could break as written')
  return out.join('\n')
}

/** Bytes as something a person reads at a glance. */
function humanBytes(n: number): string {
  if (!Number.isFinite(n)) return '∞'
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  let i = 0
  while (n >= 1000 && i < units.length - 1) { n /= 1000; i++ }
  // Past yottabytes the unit ladder stops meaning anything; the exponent is the
  // only honest rendering, and these numbers do show up at 100 qubits.
  if (n >= 1000) return `${(n * 1e24).toExponential(1)} B`
  return `${n < 10 ? n.toFixed(1) : Math.round(n)} ${units[i]}`
}

/** Seconds as something a person reads at a glance. */
function humanTime(s: number): string {
  if (!Number.isFinite(s)) return '∞'
  if (s < 1e-3) return `${(s * 1e6).toFixed(0)} µs`
  if (s < 1) return `${(s * 1e3).toFixed(0)} ms`
  if (s < 90) return `${s.toFixed(1)} s`
  if (s < 5400) return `${(s / 60).toFixed(1)} min`
  if (s < 1.3e5) return `${(s / 3600).toFixed(1)} h`
  if (s < 3e9) return `${(s / 86400).toFixed(0)} days`
  return `${(s / 3.15e7).toExponential(1)} years`
}

/**
 * The report as an ASCII table.
 *
 * The bound marker on each parameter is the column to read second: `≥` means the
 * probe stopped early and the cost shown is a floor, `≤` means a better search
 * could lower it.
 */
export function renderHardness(report: HardnessReport): string {
  const mark = (b: Bound): string => b === 'exact' ? ' ' : b === 'lower' ? '≥' : '≤'
  // A declined attack's reason can be a whole paragraph of gate names, and one of
  // those would set the column width for the entire table.
  const clip = (s: string, max = 46): string => s.length <= max ? s : `${s.slice(0, max - 1)}…`
  const rows = report.attacks.map(a => {
    const p = a.parameter
    return [
      `${a.viable ? '✓' : a.applicable ? '·' : ' '} ${a.name}`,
      a.answers,
      p ? `${p.name} ${mark(p.bound)}${p.value}` : clip(a.reason ?? ''),
      // An approximate break and an exact one are different results, so the
      // accuracy claim gets a column rather than a footnote. Blank means the attack
      // claimed nothing, which is not the same as claiming zero.
      a.error === undefined ? '' : a.error.value === 0 ? 'exact' : `${mark(a.error.bound)}${a.error.value}`,
      a.memoryBytes === undefined ? '' : humanBytes(a.memoryBytes),
      a.seconds === undefined ? '' : humanTime(a.seconds),
    ]
  })
  const head = ['attack', 'answers', 'parameter / why not', 'error', 'memory', 'time']
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map(r => r[i]!.length)))
  const line = (cells: string[]): string =>
    cells.map((c, i) => i >= 4 ? c.padStart(widths[i]!) : c.padEnd(widths[i]!)).join('  ').trimEnd()

  const out = [
    `${report.qubits} qubits, ${report.gates} gates — budget ${humanBytes(report.budget.memoryBytes)} / ${humanTime(report.budget.seconds)}`,
    line(head),
    widths.map(w => '─'.repeat(w)).join('  '),
    ...rows.map(line),
    '',
    report.best
      ? `best: ${report.best.name} (${report.best.answers})` +
        (report.advantage ? ` — ${report.advantage.toExponential(1)}x cheaper than a statevector` : '')
      : 'no attack in this library landed within budget',
  ]
  // The cheapest attack and the cheapest *full* attack are often not the same
  // one, and reporting only the former would let a single amplitude stand in for
  // a distribution.
  if (report.best && report.bestFull && report.bestFull.name !== report.best.name) {
    out.push(`cheapest full distribution: ${report.bestFull.name}`)
  } else if (report.best && !report.bestFull) {
    out.push('no attack produced a full distribution within budget')
  }
  return out.join('\n')
}
