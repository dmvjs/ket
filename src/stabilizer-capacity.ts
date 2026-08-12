/**
 * How far Clifford+T simulation reaches on a given machine.
 *
 * Everything here is an **empirical fit, not an identity**. The constants were
 * measured on one machine (Apple silicon, Node 24) and depend on the allocator
 * and GC as much as on the algorithm, so treat the output as a planning estimate
 * accurate to about ±1 T gate, not a guarantee. They live in their own module
 * precisely so that provenance stays visible instead of being mistaken for part
 * of the simulator's contract.
 *
 * The exact quantities are in `stabilizer-rank.ts`: `extent` and `termBudget`
 * are arithmetic, and carry no fitted constants.
 */

/** ζ = 1/cos²(π/8) — the stabilizer extent contributed by one T gate. */
const ZETA = 1 / Math.cos(Math.PI / 8) ** 2

/**
 * Peak resident memory as a multiple of the decomposition's steady-state size.
 *
 * A split allocates the 2m-term arrays while the m-term originals are still live,
 * and GC lags behind. Fitted to `process.resourceUsage().maxRSS` at n=100, δ=0.3,
 * one measurement per process:
 *
 *   t=50  0.20 GB resident  1.14 GB peak  ×5.71
 *   t=55  0.44 GB           1.99 GB       ×4.51
 *   t=60  0.97 GB           3.98 GB       ×4.09
 *   t=65  2.15 GB           8.21 GB       ×3.82
 *
 * The ratio falls as the decomposition outgrows Node's baseline footprint and
 * converges near 4, so 4 is the right asymptote and small runs are simply
 * dominated by fixed overhead.
 */
const PEAK_FACTOR = 4

/**
 * Resident bytes per stabilizer term at `n` qubits.
 *
 * Three bit-packed tableaus of n rows × ⌈n/32⌉ words (12·n·⌈n/32⌉ bytes), the
 * three length-n vectors γ, v, s (3n bytes), and fixed per-object overhead.
 * Fitted to 1.58 / 2.72 / 6.31 / 18.32 / 63.44 KB at n = 10 / 50 / 100 / 200 /
 * 400, reproduced to within 1.5% across the range.
 *
 * Packing is worth 4.9× at n=100 and 7.4× at n=400 — short of a full 8× because
 * ⌈n/32⌉ rounds up, so 100 qubits still occupy 128 bits per row.
 */
export function bytesPerTerm(qubits: number): number {
  return 12 * qubits * ((qubits + 31) >>> 5) + 3 * qubits + 1450
}

export interface MaxTGatesOptions {
  /** Circuit width. Memory per term grows as 3n²/8. */
  qubits: number
  /** Target ℓ₂ error δ. Looser tolerances buy T gates directly, since k ∝ δ⁻². */
  targetError: number
  /** Memory budget in bytes. Defaults to 4 GB. */
  memoryBytes?: number
}

/**
 * Estimated largest T-count that fits a memory budget — the point where ξ/δ²
 * terms stop being affordable.
 *
 * The ceiling is a property of the machine, not the algorithm: it moves with
 * qubit count, with the error tolerance, and with available RAM.
 *
 * **This is a memory bound, and on a large machine memory is not what stops you.**
 * Runtime grows faster than the term count, so the usable ceiling is well below
 * the number returned here. Measured at n=100, δ=0.3 on a 64 GB machine, one run
 * per process: t=60 took 67 s, t=65 took 228 s, t=70 took 27 min, and t=76 was
 * abandoned unfinished after 6 h — while this function reports 77 for that
 * machine, and memory never exceeded 8.2 GB of the 64 available.
 *
 * Treat the result as "you will not run out of RAM below this", not "this is
 * reachable". For what is reachable, measure: `benchmark/stabilizer-rank.ts`
 * takes one T-count per invocation. Do not extrapolate runtime from low points —
 * the growth exponent itself rises with t, and every such fit made during
 * development proved optimistic.
 *
 * @example
 * maxTGates({ qubits: 100, targetError: 0.3, memoryBytes: 64e9 })  // 77 — but ~70 is practical
 * maxTGates({ qubits: 400, targetError: 0.3, memoryBytes: 64e9 })  // 63
 */
export function maxTGates({ qubits, targetError, memoryBytes = 4e9 }: MaxTGatesOptions): number {
  if (!(targetError > 0)) throw new RangeError(`targetError must be positive, got ${targetError}`)
  if (!(memoryBytes > 0)) throw new RangeError(`memoryBytes must be positive, got ${memoryBytes}`)
  const affordable = (memoryBytes * targetError ** 2) / (PEAK_FACTOR * bytesPerTerm(qubits))
  if (affordable < 1) return 0
  return Math.max(0, Math.floor(Math.log(affordable) / Math.log(ZETA)))
}
