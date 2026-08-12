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
 * and GC lags behind. Fitted to `process.resourceUsage().maxRSS` at n=100, δ=0.3
 * with packed tableaus.
 *
 * Caveat worth knowing before trusting the high end: the fit is anchored on
 * points at t ≤ 60. Runs above that were originally measured with
 * `process.memoryUsage().rss` *after* the call returned, which reports post-GC
 * residual rather than peak and understates large runs. The two sources agree
 * closely at t ≤ 60; above it this constant is under-validated.
 */
const PEAK_FACTOR = 5

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
 * Memory is only half the story, and increasingly the less important half.
 * Runtime grows faster than the term count — a measured exponent of 1.74 between
 * t=60 and t=65, rising to 2.46 between t=65 and t=70 as the working set outruns
 * cache — so on a large machine the wall you hit first is time, and this function
 * will happily report a T-count that would take hours. Extrapolate runtime from
 * the two highest points you have actually measured; fits from lower pairs have
 * consistently proven optimistic.
 *
 * @example
 * maxTGates({ qubits: 100, targetError: 0.3, memoryBytes: 64e9 })  // 76
 * maxTGates({ qubits: 400, targetError: 0.3, memoryBytes: 64e9 })  // 61
 */
export function maxTGates({ qubits, targetError, memoryBytes = 4e9 }: MaxTGatesOptions): number {
  if (!(targetError > 0)) throw new RangeError(`targetError must be positive, got ${targetError}`)
  if (!(memoryBytes > 0)) throw new RangeError(`memoryBytes must be positive, got ${memoryBytes}`)
  const affordable = (memoryBytes * targetError ** 2) / (PEAK_FACTOR * bytesPerTerm(qubits))
  if (affordable < 1) return 0
  return Math.max(0, Math.floor(Math.log(affordable) / Math.log(ZETA)))
}
