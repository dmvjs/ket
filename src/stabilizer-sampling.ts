/**
 * Drawing measurement shots from a stabilizer decomposition.
 *
 * Separated from `stabilizer-rank.ts` because it depends on the state only
 * through an amplitude oracle: an in-process decomposition and a pool of workers
 * each holding a slice both satisfy the same contract, so one implementation of
 * exact enumeration and one Metropolis chain serve both. Keeping them here stops
 * that seam from blurring back into the state representation.
 */

/** Basis states per batch when enumerating the exact distribution. */
const EXACT_CHUNK = 1 << 14
/** Tries at finding a non-zero-amplitude state to start the Metropolis chain. */
const SEED_ATTEMPTS = 64
/**
 * Metropolis steps below which incremental walkers are not worth building.
 *
 * A walker allocates per-term scratch, so it costs O(terms) up front and saves
 * O(terms) per step. Measured at n=100: 19× faster over ~40k steps (25.0 s →
 * 1.3 s). Over ~9 steps against a 148k-term decomposition the difference was
 * inside run-to-run variance — building the decomposition dominates there — so
 * this threshold exists to avoid allocating per-term scratch that cannot pay for
 * itself, not to fix a measured regression.
 */
const WALK_MIN_STEPS = 64

/**
 * Probes for other supported states when the Metropolis chain never moves.
 *
 * Each is one amplitude-oracle call, so this is cheap enough to run on every
 * frozen chain. Finding nothing is not proof the result is sound — only that no
 * counter-example turned up — so a frozen chain that survives all of them still
 * returns its samples.
 */
const FROZEN_PROBES = 64

export interface SampleOptions {
  /** Metropolis steps taken before the first sample is returned. */
  burnIn?: number
  /** Metropolis steps between successive samples. */
  thin?: number
  /**
   * Sampling strategy. `'exact'` enumerates all 2ⁿ amplitudes and draws from the
   * true distribution; `'metropolis'` runs the Markov chain of Section 4.2.
   * `'auto'` (default) picks exact whenever 2ⁿ·terms fits in `exactBudget`.
   */
  method?: 'auto' | 'exact' | 'metropolis'
  /** Amplitude-evaluation budget above which `'auto'` falls back to Metropolis. */
  exactBudget?: number
}

/**
 * @internal
 * Batched amplitude evaluator: given `count` basis states packed n bits each,
 * return ⟨x|ψ⟩ for every one.
 *
 * The seam between *where amplitudes come from* and *how shots are drawn*. An
 * in-process decomposition and a pool of workers each holding a slice both
 * satisfy it, which is what lets one sampler serve both.
 */
export type AmplitudeOracle = (basis: Uint8Array, count: number) => { re: Float64Array; im: Float64Array }

/**
 * @internal
 * Incremental probability evaluator for a single-bit walk.
 *
 * Successive Metropolis points differ in one bit, so rebuilding the amplitude
 * from scratch each step wastes the O(n) factor that Eq. (57) removes. Supplying
 * one makes a step cost O(⌈n/32⌉) per term instead of O(n·⌈n/32⌉).
 */
export interface WalkOracle {
  /** Move to x ⊕ e_j. Applying the same flip twice restores the previous point. */
  flip(j: number): void
  /** |⟨x|ψ⟩|² at the current point. */
  probability(): number
}

export interface OracleSampleOptions extends SampleOptions {
  /** Term count; only used to judge whether exact enumeration is affordable. */
  terms?: number
  /** Metropolis seed state. Defaults to uniformly random bits. */
  start?: (rand: () => number) => Uint8Array
  /** Incremental evaluator for the Metropolis path; falls back to `amp` without it. */
  walk?: (x: Uint8Array) => WalkOracle
}

/**
 * @internal
 * Draw `shots` basis states from P(x) ∝ |⟨x|ψ⟩|² through an amplitude oracle.
 *
 * Prefers exact enumeration, which is correct unconditionally, and falls back to
 * the Metropolis chain of Section 4.2 only when 2ⁿ amplitudes are out of reach.
 * That fallback is a genuine heuristic and can be *wrong*, not merely noisy: its
 * single-bit-flip proposals cannot cross a basis state of zero amplitude, so a
 * distribution whose support is not single-flip connected is sampled from only
 * one component. A Toffoli on |++0⟩ is already such a case — |111⟩ is two flips
 * from every other supported state — so treat Metropolis results as indicative
 * unless the support is known to be connected.
 */
export function sampleFromOracle(
  n: number,
  shots: number,
  rand: () => number,
  amp: AmplitudeOracle,
  { burnIn = 200, thin = 20, method = 'auto', exactBudget = 1 << 22, terms = 1, start, walk }: OracleSampleOptions = {},
): Uint8Array[] {
  const unpack = (v: number, into: Uint8Array, at = 0): Uint8Array => {
    for (let j = 0; j < n; j++) into[at + j] = (v >>> j) & 1
    return into
  }

  if (method === 'exact' || (method === 'auto' && n <= 30 && 2 ** n * terms <= exactBudget)) {
    const dim = 1 << n
    const cum = new Float64Array(dim)
    let acc = 0
    // Chunked so the packed basis stays small: a single 2ⁿ×n buffer would be
    // tens of MB at the budget ceiling, and the worker oracle clones it.
    const span = Math.min(dim, EXACT_CHUNK)
    const basis = new Uint8Array(span * n)
    for (let base = 0; base < dim; base += span) {
      const count = Math.min(span, dim - base)
      for (let i = 0; i < count; i++) unpack(base + i, basis, i * n)
      const { re, im } = amp(basis, count)
      for (let i = 0; i < count; i++) {
        acc += re[i]! * re[i]! + im[i]! * im[i]!
        cum[base + i] = acc
      }
    }
    if (acc <= 0) throw new RangeError('state has zero norm; nothing to sample')
    return Array.from({ length: shots }, () => {
      const target = rand() * acc
      let lo = 0, hi = dim - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (cum[mid]! < target) lo = mid + 1; else hi = mid
      }
      return unpack(lo, new Uint8Array(n))
    })
  }

  const seed = start ?? (r => Uint8Array.from({ length: n }, () => (r() < 0.5 ? 0 : 1)))
  const batchProb = (x: Uint8Array): number => {
    const { re, im } = amp(x, 1)
    return re[0]! * re[0]! + im[0]! * im[0]!
  }
  // The chain is only valid from a state of non-zero amplitude: at P(x) = 0 the
  // acceptance test P(y) >= P(x) passes unconditionally and the walk degenerates
  // into a uniform random walk that still returns plausible-looking bitstrings.
  // Callers should pass `start` drawing from a term's own support; give up loudly
  // rather than emit samples from a degenerate chain.
  const steps = burnIn + shots * thin
  const useWalk = steps >= WALK_MIN_STEPS ? walk : undefined
  let x = seed(rand)
  let walker = useWalk?.(x)
  const prob = (): number => walker ? walker.probability() : batchProb(x)
  let px = prob()
  for (let attempt = 1; px === 0 && attempt < SEED_ATTEMPTS; attempt++) {
    x = seed(rand)
    walker = useWalk?.(x)
    px = prob()
  }
  if (px === 0)
    throw new RangeError(
      `could not seed the Metropolis chain: no basis state with non-zero amplitude found in ${SEED_ATTEMPTS} attempts`)

  // Lazy chain: single-bit-flip proposals alone give a *periodic* walk, since
  // every accepted move flips Hamming parity. On a near-uniform distribution
  // almost everything is accepted, so an even `thin` would lock sampling to one
  // parity class and silently halve the support. Staying put with probability
  // 1/2 makes the chain aperiodic and leaves the stationary distribution alone.
  let accepted = 0
  const step = (): void => {
    if (rand() < 0.5) return
    const j = Math.floor(rand() * n)
    const flip = (): void => { x[j] = (x[j] ?? 0) ^ 1; walker?.flip(j) }
    flip()
    const py = prob()
    if (py >= px || rand() < py / px) { px = py; accepted++ }
    else flip()
  }

  for (let i = 0; i < burnIn; i++) step()
  const out = Array.from({ length: shots }, () => {
    for (let i = 0; i < thin; i++) step()
    return Uint8Array.from(x)
  })

  // A chain that never moved emits its seed `shots` times. That is correct for a
  // genuine point mass and badly wrong for a support that single-bit flips cannot
  // traverse — a GHZ state seeds at |1…1⟩, every neighbour has zero amplitude, and
  // the walk reports P(|1…1⟩) = 1 against a true 1/2.
  //
  // Having accepted nothing, the chain has proved every single-bit neighbour is
  // empty, so it cannot take even one step. Any other supported state anywhere is
  // therefore unreachable, and finding one is proof the samples are wrong rather
  // than evidence of it. Probing for that costs a handful of oracle calls; the
  // norm estimator would answer the same question thousands of times slower.
  if (accepted === 0) {
    const probe = new Uint8Array(n)
    for (let attempt = 0; attempt < FROZEN_PROBES; attempt++) {
      probe.set(x)
      // The complement first: it is the other half of every parity-split support,
      // GHZ included. Then random subsets, which cover less symmetric splits.
      if (attempt === 0) for (let j = 0; j < n; j++) probe[j] = (probe[j] ?? 0) ^ 1
      else {
        let flipped = 0
        for (let j = 0; j < n; j++) if (rand() < 0.5) { probe[j] = (probe[j] ?? 0) ^ 1; flipped++ }
        if (flipped === 0) continue
      }
      if (batchProb(probe) > 0)
        throw new RangeError(
          `Metropolis chain could not move: every single-bit neighbour of its starting state has zero ` +
          `amplitude, yet the state is not the whole distribution — the support is not connected under ` +
          `single-bit flips, so these samples would be wrong, not merely noisy. Use method: 'exact' ` +
          `(raising exactBudget if needed) for a correct distribution.`)
    }
  }
  return out
}

