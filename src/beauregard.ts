/**
 * Beauregard circuit for gate-decomposed Shor's algorithm.
 *
 * Implements modular arithmetic using only CNOT, CU1, and single-qubit
 * gates — no dense matrices. Scales to 40-bit semiprimes via MPS.
 *
 * Reference: Beauregard (2003), "Circuit for Shor's algorithm using 2n+3 qubits"
 * arXiv:quant-ph/0205095
 *
 * Build order (each layer is independently tested):
 *   1. phiAdd          — Draper phase adder: add classical constant to QFT register
 *   2. phiAddMod       — Modular adder in QFT basis (Beauregard §3)
 *   3. ccPhiAddMod     — Doubly-controlled modular adder
 *   4. cMultModAdd     — Controlled modular multiplier: |x⟩|b⟩ → |x⟩|b + ax mod N⟩
 *   5. beauregardU     — Single U_a application: |x⟩ → |ax mod N⟩
 *   6. shorBeauregard  — Full QPE Shor's circuit with retry loop
 */

import { Circuit } from './circuit.js'

// ── Qubit convention ───────────────────────────────────────────────────────────
//
// ket uses qubit 0 = LSB throughout.
//
// After ket's QFT (which ends with a bit-reversal swap), qubit j holds:
//   (|0⟩ + exp(2πi·x / 2^(n−j)) |1⟩) / √2
//
// Derivation: the main loop builds the standard QFT with qubit 0 = LSB-output
// (coarsest phase exp(2πix/2)), then the swap reverses the bit order so that:
//   qubit 0 (LSB, j=0)   → finest phase exp(2πix/2^n)
//   qubit n-1 (MSB, j=n-1) → coarsest phase exp(2πix/2)
//
// Wait — the swap sends qubit i to qubit n−1−i. Before the swap, qubit i holds
// exp(2πix/2^(i+1)). After swap, qubit i gets old qubit n−1−i, which held
// exp(2πix/2^(n−i)).  So:
//   after QFT, qubit j holds exp(2πix / 2^(n−j))
//
// To add classical constant a: need phase increment exp(2πia/2^(n−j)) on qubit j.
// Apply U1(2πa/2^(n−j)) = U1(2πa · 2^j / 2^n) to qubit j.  ✓

// ── Classical arithmetic helpers ─────────────────────────────────────────────

/** Reduce angle to (-π, π] to keep U1 rotations well-conditioned. */
function reduceAngle(theta: number): number {
  const TWO_PI = 2 * Math.PI
  let t = theta % TWO_PI
  if (t > Math.PI)  t -= TWO_PI
  if (t <= -Math.PI) t += TWO_PI
  return t
}

/** Modular exponentiation: base^exp mod m, using BigInt for exact arithmetic. */
export function modPow(base: bigint, exp: bigint, m: bigint): bigint {
  if (m === 1n) return 0n
  let result = 1n
  let b = base % m
  let e = exp
  while (e > 0n) {
    if (e & 1n) result = result * b % m
    b = b * b % m
    e >>= 1n
  }
  return result
}

/** Extended Euclidean algorithm: returns [g, x, y] with a·x + b·y = g = gcd(a,b). */
function extGcd(a: bigint, b: bigint): [bigint, bigint, bigint] {
  if (b === 0n) return [a, 1n, 0n]
  const [g, x, y] = extGcd(b, a % b)
  return [g, y, x - (a / b) * y]
}

/** Modular inverse of a mod m (requires gcd(a,m)=1). */
export function modInverse(a: bigint, m: bigint): bigint {
  const [, x] = extGcd(((a % m) + m) % m, m)
  return ((x % m) + m) % m
}

/** Greatest common divisor. */
export function gcd(a: bigint, b: bigint): bigint {
  while (b) { [a, b] = [b, a % b] }
  return a
}

/**
 * Continued-fractions approximation of phase s/2^t.
 * Returns the denominator r of the best rational approximation with r < N.
 * This is the period candidate in Shor's algorithm.
 */
export function continuedFractions(measured: number, precision: number, N: bigint): bigint {
  // Represent measured/2^precision as a continued fraction and find convergents.
  let num = BigInt(measured)
  let den = 1n << BigInt(precision)
  const convergents: [bigint, bigint][] = []
  let h0 = 0n, h1 = 1n, k0 = 1n, k1 = 0n
  while (den > 0n) {
    const a = num / den
    ;[h0, h1] = [h1, a * h1 + h0]
    ;[k0, k1] = [k1, a * k1 + k0]
    if (k1 > N) break
    convergents.push([h1, k1])
    ;[num, den] = [den, num - a * den]
  }
  // Return the largest denominator ≤ N
  for (let i = convergents.length - 1; i >= 0; i--) {
    const r = convergents[i]![1]
    if (r > 0n && r <= N) return r
  }
  return 1n
}

// ── Inline QFT / IQFT helpers (with qubit offset) ────────────────────────────
//
// These mirror src/algorithms.ts qft/iqft but accept an offset so they can be
// applied to a sub-register of a larger circuit without subcircuit indirection.

/** Apply QFT to n qubits starting at `offset`. */
export function applyQft(c: Circuit, n: number, offset: number): Circuit {
  for (let j = n - 1; j >= 0; j--) {
    c = c.h(offset + j)
    for (let k = j - 1; k >= 0; k--) {
      c = c.cu1(Math.PI / 2 ** (j - k), offset + k, offset + j)
    }
  }
  for (let i = 0; i < Math.floor(n / 2); i++) c = c.swap(offset + i, offset + n - 1 - i)
  return c
}

/** Apply IQFT to n qubits starting at `offset`. */
export function applyIqft(c: Circuit, n: number, offset: number): Circuit {
  for (let i = 0; i < Math.floor(n / 2); i++) c = c.swap(offset + i, offset + n - 1 - i)
  for (let j = 0; j < n; j++) {
    for (let k = j - 1; k >= 0; k--) c = c.cu1(-Math.PI / 2 ** (j - k), offset + k, offset + j)
    c = c.h(offset + j)
  }
  return c
}

// ── Draper phase adder ────────────────────────────────────────────────────────

/**
 * Draper phase adder: add classical constant `a` to an n-qubit register in QFT basis.
 *
 * Precondition:  qubits `offset`..`offset+n-1` are in the QFT basis (QFT already applied).
 * Postcondition: register encodes x+a (mod 2^n).
 *
 * In ket's QFT convention (qubit j=0 is LSB), qubit j holds
 * (|0⟩ + exp(2πix/2^(j+1))|1⟩)/√2, so adding a requires phase exp(2πia/2^(j+1)).
 *
 * O(n) gates, no ancilla, no entanglement.
 *
 * @param a may be negative (implements subtraction mod 2^n).
 */
export function phiAdd(c: Circuit, n: number, a: bigint, offset = 0): Circuit {
  // Reduce a mod 2^n to the canonical range [0, 2^n)
  const mod = 1n << BigInt(n)
  const aPos = ((a % mod) + mod) % mod

  for (let j = 0; j < n; j++) {
    // Phase on qubit j = 2π·a / 2^(n−j).  Reduce: take a mod 2^(n−j) first.
    const power = n - j
    const div   = 1n << BigInt(power)
    const aRed  = Number(aPos % div)
    const angle = reduceAngle(2 * Math.PI * aRed / Number(div))
    if (Math.abs(angle) > 1e-12) c = c.u1(angle, offset + j)
  }
  return c
}

/**
 * Controlled Draper adder: add `a` to QFT register, conditioned on `ctrl`.
 * Replaces each U1(θ) with CU1(θ, ctrl, qubit).
 */
export function cPhiAdd(c: Circuit, n: number, a: bigint, ctrl: number, offset = 0): Circuit {
  const mod = 1n << BigInt(n)
  const aPos = ((a % mod) + mod) % mod

  for (let j = 0; j < n; j++) {
    const power = n - j
    const div   = 1n << BigInt(power)
    const aRed  = Number(aPos % div)
    const angle = reduceAngle(2 * Math.PI * aRed / Number(div))
    if (Math.abs(angle) > 1e-12) c = c.cu1(angle, ctrl, offset + j)
  }
  return c
}

/**
 * Doubly-controlled Draper adder: add `a` to QFT register, conditioned on both `ctrl1` and `ctrl2`.
 *
 * Decomposes CCU1(θ) into 2×CNOT + 3×CU1 (no Toffoli required):
 *   CCU1(θ, c1, c2, t) = CU1(θ/2, c1, t) · CNOT(c1, c2) · CU1(-θ/2, c2, t)
 *                       · CNOT(c1, c2) · CU1(θ/2, c2, t)
 */
export function ccPhiAdd(
  c: Circuit, n: number, a: bigint,
  ctrl1: number, ctrl2: number, offset = 0,
): Circuit {
  const mod = 1n << BigInt(n)
  const aPos = ((a % mod) + mod) % mod

  for (let j = 0; j < n; j++) {
    const power = n - j
    const div   = 1n << BigInt(power)
    const aRed  = Number(aPos % div)
    const theta = reduceAngle(2 * Math.PI * aRed / Number(div))
    if (Math.abs(theta) <= 1e-12) continue
    const half = theta / 2
    const tgt = offset + j
    c = c.cu1(half,  ctrl1, tgt)
    c = c.cnot(ctrl1, ctrl2)
    c = c.cu1(-half, ctrl2, tgt)
    c = c.cnot(ctrl1, ctrl2)
    c = c.cu1(half,  ctrl2, tgt)
  }
  return c
}

// ── Modular phase adder ───────────────────────────────────────────────────────

/**
 * Beauregard modular adder: b → b + a mod N (in-place, QFT basis).
 *
 * Operates on an (n+1)-qubit b register in the QFT basis at `bOff`..`bOff+n`,
 * plus one clean ancilla qubit at `ancilla`.
 *
 * Invariant: 0 ≤ b < N and 0 ≤ a < N on entry; same on exit.
 *
 * Circuit structure (Beauregard 2003, Figure 5):
 *   phiAdd(a) · phiAdd(-N) · IQFT · CNOT(sign→anc) · QFT
 *   · cPhiAdd(N, anc) · phiAdd(-a) · IQFT
 *   · X(sign) · CNOT(sign→anc) · X(sign) · QFT · phiAdd(a)
 *
 * O(n²) gates (dominated by QFT/IQFT), 1 ancilla.
 */
export function phiAddMod(
  c: Circuit, n: number, a: bigint, N: bigint,
  bOff: number, ancilla: number,
): Circuit {
  const nb   = n + 1  // b register width including overflow bit
  const sign = bOff + nb - 1  // MSB = sign/overflow bit

  c = phiAdd(c,  nb, a,  bOff)   // add a
  c = phiAdd(c,  nb, -N, bOff)   // subtract N
  c = applyIqft(c, nb, bOff)
  c = c.cnot(sign, ancilla)       // copy sign to ancilla
  c = applyQft(c,  nb, bOff)
  c = cPhiAdd(c, nb, N, ancilla, bOff)  // conditional restore
  c = phiAdd(c,  nb, -a, bOff)   // subtract a (uncompute toward zero)
  c = applyIqft(c, nb, bOff)
  c = c.x(sign)
  c = c.cnot(sign, ancilla)       // uncompute ancilla
  c = c.x(sign)
  c = applyQft(c,  nb, bOff)
  c = phiAdd(c,  nb, a,  bOff)   // restore a
  return c
}

/**
 * Doubly-controlled modular adder: b → b + a mod N when both `ctrl1` and `ctrl2` are |1⟩.
 *
 * Same structure as {@link phiAddMod} but φADD(a) and φADD(−a) become doubly controlled.
 * φADD(−N) is unconditional (safe because b < N invariant ensures b − N < 0).
 *
 * One ancilla qubit required at `ancilla` (must start and end at |0⟩).
 */
export function ccPhiAddMod(
  c: Circuit, n: number, a: bigint, N: bigint,
  ctrl1: number, ctrl2: number,
  bOff: number, ancilla: number,
): Circuit {
  const nb   = n + 1
  const sign = bOff + nb - 1

  c = ccPhiAdd(c, nb,  a,  ctrl1, ctrl2, bOff)
  c = phiAdd(c,   nb, -N,  bOff)
  c = applyIqft(c, nb, bOff)
  c = c.cnot(sign, ancilla)
  c = applyQft(c,  nb, bOff)
  c = cPhiAdd(c,  nb,  N,  ancilla, bOff)
  c = ccPhiAdd(c, nb, -a,  ctrl1, ctrl2, bOff)
  c = applyIqft(c, nb, bOff)
  c = c.x(sign)
  c = c.cnot(sign, ancilla)
  c = c.x(sign)
  c = applyQft(c,  nb, bOff)
  c = ccPhiAdd(c, nb,  a,  ctrl1, ctrl2, bOff)
  return c
}

// ── Controlled modular multiplier ─────────────────────────────────────────────

/**
 * Controlled modular multiply-add: |ctrl⟩|x⟩|b⟩ → |ctrl⟩|x⟩|b + a·x mod N⟩.
 *
 * Qubit layout:
 *   ctrl  — single control qubit
 *   xOff..xOff+n-1   — n-qubit x register (read-only, each bit controls ccPhiAddMod)
 *   bOff..bOff+n     — (n+1)-qubit b register (accumulates product)
 *   ancilla           — single ancilla (must start and end at |0⟩)
 *
 * For each bit j of x: if x[j]=1 then b += a·2^j mod N.
 * Precomputes a·2^j mod N classically for each j.
 *
 * O(n²) doubly-controlled phase gates per bit, O(n³) total.
 */
export function cMultModAdd(
  c: Circuit, n: number, a: bigint, N: bigint,
  ctrl: number, xOff: number, bOff: number, ancilla: number,
): Circuit {
  c = applyQft(c, n + 1, bOff)

  let aShifted = a % N
  for (let j = 0; j < n; j++) {
    c = ccPhiAddMod(c, n, aShifted, N, ctrl, xOff + j, bOff, ancilla)
    aShifted = aShifted * 2n % N
  }

  c = applyIqft(c, n + 1, bOff)
  return c
}

// ── Single controlled-U_a application ────────────────────────────────────────

/**
 * Controlled modular multiplier gate U_a: |ctrl⟩|x⟩|0⟩ → |ctrl⟩|ax mod N⟩|0⟩.
 *
 * Implements one application of U_a for Shor's QPE using the Beauregard construction:
 *   1. cMultModAdd(a):    |x⟩|0⟩ → |x⟩|ax mod N⟩
 *   2. Controlled SWAP:   |x⟩|ax mod N⟩ → |ax mod N⟩|x⟩
 *   3. cMultModAdd(a⁻¹):  |ax mod N⟩|x⟩ → |ax mod N⟩|0⟩  (uncomputes b)
 *
 * Qubit layout: ctrl | x[0..n-1] | b[0..n] | ancilla
 *   ctrl     — control qubit (QPE counting qubit)
 *   xOff     — start of n-qubit x register
 *   bOff     — start of (n+1)-qubit b register (must be |0⟩ on entry and exit)
 *   ancilla  — single ancilla qubit (must be |0⟩)
 *
 * @param aInv  Modular inverse of a mod N (precomputed by caller).
 */
export function beauregardU(
  c: Circuit, n: number,
  a: bigint, aInv: bigint, N: bigint,
  ctrl: number, xOff: number, bOff: number, ancilla: number,
): Circuit {
  // Step 1: |x⟩|0⟩ → |x⟩|ax mod N⟩
  c = cMultModAdd(c, n, a, N, ctrl, xOff, bOff, ancilla)

  // Step 2: SWAP x ↔ b[0..n-1], conditioned on ctrl
  // (b has n+1 qubits but only n hold the value; bit n is the overflow and stays 0)
  for (let j = 0; j < n; j++) c = c.cswap(ctrl, xOff + j, bOff + j)

  // Step 3: uncompute b using inverse multiplier |b=old_x⟩ → |b=0⟩
  // Subtract: b → b − a⁻¹·(ax mod N) = b − x = 0
  c = cMultModAdd(c, n, N - aInv, N, ctrl, xOff, bOff, ancilla)

  return c
}

// ── Full Shor's QPE circuit ───────────────────────────────────────────────────

/**
 * Result of the Beauregard Shor's circuit.
 * `factor` is a non-trivial factor of N, or undefined if the run failed.
 * `attempts` is how many random bases were tried.
 */
export interface ShorResult {
  readonly factor:   bigint | undefined
  readonly factors:  [bigint, bigint] | undefined
  /** Base `a` that produced the successful period. */
  readonly a:        bigint
  /** Measured period `r` such that a^r ≡ 1 (mod N). */
  readonly period:   bigint | undefined
  readonly attempts: number
  readonly qubits:   number
}

/**
 * Full Shor's algorithm using the Beauregard gate-decomposed circuit.
 *
 * Builds the QPE circuit with an explicit quantum Fourier adder oracle —
 * no dense matrices, O(n³) gate count. Uses MPS simulation (required for n ≥ 7).
 *
 * @param N         Semiprime to factor (must be odd, not a prime power).
 * @param opts.a    Base for modular exponentiation (default: random coprime to N).
 * @param opts.precision  QPE counting qubits (default: 2n+1 where n=⌈log₂N⌉).
 * @param opts.shots      Shots per QPE run (default: 1).
 * @param opts.maxAttempts  Maximum random-base retries (default: 20).
 * @param opts.seed  PRNG seed for reproducibility.
 *
 * Circuit layout (total = precision + 2n + 2 qubits):
 *   0..precision-1       — counting register (QPE)
 *   precision..+n-1      — x register (starts |1⟩)
 *   precision+n..+n      — b register (n+1 qubits, ancilla for multiplier)
 *   precision+2n+1       — ancilla qubit
 */
/**
 * Factor N into two non-trivial factors using Shor's algorithm.
 *
 * Convenience wrapper around {@link shorBeauregard}: handles even N classically,
 * picks random coprime bases automatically, and retries on bad periods.
 *
 * @returns `[p, q]` with `p * q === N` and `1 < p, q < N`, or `undefined` if
 *          factoring failed (N is prime, or all random bases gave odd/trivial periods).
 *
 * @example
 * ```ts
 * factor(15n)  // → [3n, 5n]
 * factor(21n)  // → [3n, 7n]
 * factor(35n)  // → [5n, 7n]
 * ```
 */
export function factor(N: number | bigint): [bigint, bigint] | undefined {
  return shorBeauregard(BigInt(N)).factors
}

export function shorBeauregard(
  N: bigint,
  opts: {
    a?:          bigint
    precision?:  number
    shots?:      number
    maxAttempts?: number
    seed?:       number
    truncErr?:   number
  } = {},
): ShorResult {
  const n          = Math.ceil(Math.log2(Number(N)))
  const precision  = opts.precision  ?? 2 * n + 1
  const shots      = opts.shots      ?? 1
  const maxTries   = opts.maxAttempts ?? 20
  const truncErr   = opts.truncErr   ?? 0
  const totalQ     = precision + 2 * n + 2

  // Qubit offsets
  const xOff    = precision
  const bOff    = precision + n
  const ancilla = precision + 2 * n + 1

  // Quick classical checks
  if (N < 4n) throw new RangeError('N must be ≥ 4')
  if (N % 2n === 0n) return { factor: 2n, factors: [2n, N / 2n], a: 0n, period: undefined, attempts: 0, qubits: totalQ }

  const Nnum = Number(N)

  for (let attempt = 1; attempt <= maxTries; attempt++) {
    // Pick a random base a with gcd(a, N) = 1
    const aCand = opts.a ?? BigInt(2 + Math.floor(Math.random() * (Nnum - 3)))
    const g = gcd(aCand, N)
    if (g > 1n) {
      // Lucky: a shares a factor with N directly
      return { factor: g, factors: [g, N / g], a: aCand, period: undefined, attempts: attempt, qubits: totalQ }
    }

    const a    = aCand
    const aInv = modInverse(a, N)

    // Build QPE circuit
    let c = new Circuit(totalQ)

    // Initialise counting register to |+⟩^precision
    for (let k = 0; k < precision; k++) c = c.h(k)

    // Initialise x register to |1⟩ (qubit xOff = |1⟩, rest |0⟩)
    c = c.x(xOff)

    // Apply controlled-U_a^(2^k) for each counting qubit k
    // a^(2^k) mod N is computed classically; the circuit only uses phase rotations
    let ak = a % N
    for (let k = 0; k < precision; k++) {
      const akInv = modInverse(ak, N)
      c = beauregardU(c, n, ak, akInv, N, k, xOff, bOff, ancilla)
      ak = ak * ak % N  // a^(2^(k+1)) = (a^(2^k))^2
    }

    // Inverse QFT on counting register
    c = applyIqft(c, precision, 0)

    // Run via MPS (handles the entanglement efficiently)
    const dist = c.runMps({ shots, ...(opts.seed !== undefined && { seed: opts.seed }), truncErr })

    // Extract period candidates from all measurement outcomes
    const candidates = new Set<bigint>()
    for (const bs of Object.keys(dist.probs)) {
      // bs is qubit-0-first (LSB), convert to integer
      let val = 0
      for (let i = 0; i < precision; i++) if (bs[i] === '1') val |= 1 << i
      if (val === 0) continue
      const r = continuedFractions(val, precision, N)
      candidates.add(r)
    }

    for (const r of candidates) {
      if (r === 0n || r > N) continue
      if (modPow(a, r, N) !== 1n) continue       // verify period
      if (r % 2n !== 0n) continue                  // odd period — retry
      const halfPow = modPow(a, r / 2n, N)
      if (halfPow === N - 1n) continue             // a^(r/2) ≡ -1 mod N — retry
      const f1 = gcd(halfPow + 1n, N)
      const f2 = gcd(halfPow - 1n, N)
      for (const f of [f1, f2]) {
        if (f > 1n && f < N) {
          return { factor: f, factors: [f, N / f], a, period: r, attempts: attempt, qubits: totalQ }
        }
      }
    }
  }

  return { factor: undefined, factors: undefined, a: opts.a ?? 0n, period: undefined, attempts: maxTries, qubits: totalQ }
}
