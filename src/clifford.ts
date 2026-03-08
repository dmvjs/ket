/**
 * CHP (Aaronson-Gottesman 2004) Clifford stabilizer simulator — bit-packed.
 *
 * Maintains a (2n+1) × ⌈n/32⌉ word tableau:
 *   Rows 0..n-1:   destabilizer generators
 *   Rows n..2n-1:  stabilizer generators
 *   Row 2n:        scratch row for deterministic measurement
 *
 * 32 tableau bits are packed per Int32Array element.
 * `rowmul` phase and XOR are both O(n/32) via vectorized popcount and word XOR.
 * Single-qubit gates are O(n) over the 2n rows; measurement is O(n²) worst-case.
 * `?? 0` guards on `_r` reads are required by noUncheckedIndexedAccess.
 */

/** Hamming weight of a 32-bit integer (Knuth Vol. 4A §7.1.3). */
function popcount32(v: number): number {
  let x = v | 0
  x = x - ((x >>> 1) & 0x55555555)
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333)
  x = (x + (x >>> 4)) & 0x0f0f0f0f
  return Math.imul(x, 0x01010101) >>> 24
}

export class CliffordSim {
  readonly n: number
  private readonly W: number       // words per row = ⌈n/32⌉
  private readonly _x: Int32Array  // (2n+1)×W packed x-bits
  private readonly _z: Int32Array  // (2n+1)×W packed z-bits
  private readonly _r: number[]    // 2n+1 phase bits

  constructor(n: number) {
    this.n = n
    this.W = (n + 31) >> 5
    const W = this.W
    const rows = 2 * n + 1
    this._x = new Int32Array(rows * W)
    this._z = new Int32Array(rows * W)
    this._r = new Array<number>(rows).fill(0)
    // |0…0⟩: destabilizer i → X_i, stabilizer i → Z_i.
    // Direct write — Int32Array zero-initializes and each qubit maps to a unique slot.
    for (let i = 0; i < n; i++) {
      this._x[i * W + (i >> 5)]       = 1 << (i & 31)
      this._z[(i + n) * W + (i >> 5)] = 1 << (i & 31)
    }
  }

  /**
   * Tableau row multiply: row_i ← row_i · row_h.
   * Phase accumulated via vectorized popcount on packed Pauli word pairs (O(n/32)):
   *   +1 contributions: Y·Z, X·Y, Z·X
   *   -1 contributions: Y·X, X·Z, Z·Y
   * XOR update in the same pass to avoid re-reading words.
   */
  private rowmul(i: number, h: number): void {
    const { W, _x, _z, _r } = this
    const iW = i * W, hW = h * W
    let pos = 0, neg = 0
    for (let w = 0; w < W; w++) {
      const xh = _x[hW + w] ?? 0,  zh = _z[hW + w] ?? 0
      const xi = _x[iW + w] ?? 0,  zi = _z[iW + w] ?? 0
      pos += popcount32((xh & zh & ~xi & zi) | (xh & ~zh & xi & zi) | (~xh & zh & xi & ~zi))
      neg += popcount32((xh & zh & xi & ~zi) | (xh & ~zh & ~xi & zi) | (~xh & zh & xi &  zi))
      _x[iW + w] = xi ^ xh
      _z[iW + w] = zi ^ zh
    }
    const sum = 2 * ((_r[h] ?? 0) + (_r[i] ?? 0)) + pos - neg
    _r[i] = ((((sum % 4) + 4) % 4) >> 1) & 1
  }

  // ── Single-qubit gates ────────────────────────────────────────────────────

  /** H: swap x↔z for column a, r[i] ^= x[i,a] & z[i,a]. */
  h(a: number): void {
    const { n, W, _x, _z, _r } = this
    const w = a >> 5, sh = a & 31
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W
      const xv = _x[iW + w] ?? 0, zv = _z[iW + w] ?? 0
      const xi = (xv >>> sh) & 1,  zi = (zv >>> sh) & 1
      _r[i] = ((_r[i] ?? 0) ^ (xi & zi)) & 1
      const flip = (xi ^ zi) << sh  // branchless swap: flip both iff they differ
      _x[iW + w] = xv ^ flip
      _z[iW + w] = zv ^ flip
    }
  }

  /** S: z[i,a] ^= x[i,a], r[i] ^= x[i,a] & z_old[i,a]. */
  s(a: number): void {
    const { n, W, _x, _z, _r } = this
    const w = a >> 5, sh = a & 31
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W
      const xv = _x[iW + w] ?? 0, zv = _z[iW + w] ?? 0
      const xi = (xv >>> sh) & 1,  zi = (zv >>> sh) & 1
      _r[i] = ((_r[i] ?? 0) ^ (xi & zi)) & 1
      _z[iW + w] = zv ^ (xi << sh)
    }
  }

  /** S†: same z update as S, but r[i] ^= x[i,a] & ~z_old[i,a]. */
  si(a: number): void {
    const { n, W, _x, _z, _r } = this
    const w = a >> 5, sh = a & 31
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W
      const xv = _x[iW + w] ?? 0, zv = _z[iW + w] ?? 0
      const xi = (xv >>> sh) & 1,  zi = (zv >>> sh) & 1
      _r[i] = ((_r[i] ?? 0) ^ (xi & (zi ^ 1))) & 1
      _z[iW + w] = zv ^ (xi << sh)
    }
  }

  /** X: r[i] ^= z[i,a]. (X anticommutes with Z, commutes with X.) */
  x(a: number): void {
    const { n, W, _z, _r } = this
    const w = a >> 5, sh = a & 31
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W
      _r[i] = ((_r[i] ?? 0) ^ (((_z[iW + w] ?? 0) >>> sh) & 1)) & 1
    }
  }

  /** Y: r[i] ^= x[i,a] ^ z[i,a]. (Y anticommutes with X and Z, commutes with Y.) */
  y(a: number): void {
    const { n, W, _x, _z, _r } = this
    const w = a >> 5, sh = a & 31
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W
      const xi = ((_x[iW + w] ?? 0) >>> sh) & 1
      const zi = ((_z[iW + w] ?? 0) >>> sh) & 1
      _r[i] = ((_r[i] ?? 0) ^ (xi ^ zi)) & 1
    }
  }

  /** Z: r[i] ^= x[i,a]. (Z anticommutes with X, commutes with Z.) */
  z(a: number): void {
    const { n, W, _x, _r } = this
    const w = a >> 5, sh = a & 31
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W
      _r[i] = ((_r[i] ?? 0) ^ (((_x[iW + w] ?? 0) >>> sh) & 1)) & 1
    }
  }

  // ── Two-qubit gates ───────────────────────────────────────────────────────

  /** CNOT: x[i,b] ^= x[i,a], z[i,a] ^= z[i,b], r update per CHP §3. */
  cnot(a: number, b: number): void {
    const { n, W, _x, _z, _r } = this
    const wA = a >> 5, shA = a & 31
    const wB = b >> 5, shB = b & 31
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W
      const xvA = _x[iW + wA] ?? 0, xvB = _x[iW + wB] ?? 0
      const zvA = _z[iW + wA] ?? 0, zvB = _z[iW + wB] ?? 0
      const xa = (xvA >>> shA) & 1, xb = (xvB >>> shB) & 1
      const za = (zvA >>> shA) & 1, zb = (zvB >>> shB) & 1
      _r[i] = ((_r[i] ?? 0) ^ (xa & zb & ((xb ^ za ^ 1) & 1))) & 1
      _x[iW + wB] = xvB ^ (xa << shB)
      _z[iW + wA] = zvA ^ (zb << shA)
    }
  }

  /**
   * CZ: z[i,a] ^= x[i,b], z[i,b] ^= x[i,a], r[i] ^= x[i,a] & x[i,b] & (z[i,a] ^ z[i,b]).
   * Derived from H_b · CNOT(a,b) · H_b.
   * Same-word case (a and b share a 32-bit word) handled with a single combined write.
   */
  cz(a: number, b: number): void {
    const { n, W, _x, _z, _r } = this
    const wA = a >> 5, shA = a & 31
    const wB = b >> 5, shB = b & 31
    if (wA === wB) {
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W
        const xv = _x[iW + wA] ?? 0, zv = _z[iW + wA] ?? 0
        const xa = (xv >>> shA) & 1, xb = (xv >>> shB) & 1
        const za = (zv >>> shA) & 1, zb = (zv >>> shB) & 1
        _r[i] = ((_r[i] ?? 0) ^ (xa & xb & (za ^ zb))) & 1
        _z[iW + wA] = zv ^ (xb << shA) ^ (xa << shB)
      }
    } else {
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W
        const xvA = _x[iW + wA] ?? 0, xvB = _x[iW + wB] ?? 0
        const zvA = _z[iW + wA] ?? 0, zvB = _z[iW + wB] ?? 0
        const xa = (xvA >>> shA) & 1, xb = (xvB >>> shB) & 1
        const za = (zvA >>> shA) & 1, zb = (zvB >>> shB) & 1
        _r[i] = ((_r[i] ?? 0) ^ (xa & xb & (za ^ zb))) & 1
        _z[iW + wA] = zvA ^ (xb << shA)
        _z[iW + wB] = zvB ^ (xa << shB)
      }
    }
  }

  /** CY: S†_b · CNOT(a,b) · S_b. */
  cy(a: number, b: number): void { this.si(b); this.cnot(a, b); this.s(b) }

  /**
   * SWAP: swap columns a and b in _x and _z. No phase change.
   * XOR-swap trick handles both same-word and different-word cases correctly.
   */
  swap(a: number, b: number): void {
    const { n, W, _x, _z } = this
    const wA = a >> 5, shA = a & 31
    const wB = b >> 5, shB = b & 31
    if (wA === wB) {
      // Bits are in the same word: one read, one write per array
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W
        const xv = _x[iW + wA] ?? 0, zv = _z[iW + wA] ?? 0
        const xd = ((xv >>> shA) ^ (xv >>> shB)) & 1
        const zd = ((zv >>> shA) ^ (zv >>> shB)) & 1
        _x[iW + wA] = xv ^ ((xd << shA) | (xd << shB))
        _z[iW + wA] = zv ^ ((zd << shA) | (zd << shB))
      }
    } else {
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W
        const xvA = _x[iW + wA] ?? 0, xvB = _x[iW + wB] ?? 0
        const zvA = _z[iW + wA] ?? 0, zvB = _z[iW + wB] ?? 0
        const xd = ((xvA >>> shA) ^ (xvB >>> shB)) & 1
        const zd = ((zvA >>> shA) ^ (zvB >>> shB)) & 1
        _x[iW + wA] = xvA ^ (xd << shA)
        _x[iW + wB] = xvB ^ (xd << shB)
        _z[iW + wA] = zvA ^ (zd << shA)
        _z[iW + wB] = zvB ^ (zd << shB)
      }
    }
  }

  // ── Measurement ───────────────────────────────────────────────────────────

  /** Measure qubit a. rand uniform in [0,1). Returns 0 or 1. */
  measure(a: number, rand: number): number {
    const { n, W, _x, _z, _r } = this
    const aw = a >> 5, ash = a & 31

    // Find p ∈ [n, 2n) with x[p, a] = 1 → random outcome
    let p = -1
    for (let i = n; i < 2 * n; i++) {
      if (((_x[i * W + aw] ?? 0) >>> ash) & 1) { p = i; break }
    }

    if (p !== -1) {
      for (let i = 0; i < 2 * n; i++) {
        if (i !== p && ((_x[i * W + aw] ?? 0) >>> ash) & 1) this.rowmul(i, p)
      }
      // Copy row p → destabilizer row p-n
      const d = (p - n) * W, pW = p * W
      for (let w = 0; w < W; w++) {
        _x[d + w] = _x[pW + w] ?? 0
        _z[d + w] = _z[pW + w] ?? 0
      }
      _r[p - n] = _r[p] ?? 0
      // Set row p to Z_a
      _x.fill(0, pW, pW + W)
      _z.fill(0, pW, pW + W)
      _z[pW + aw] = 1 << ash
      _r[p] = rand < 0.5 ? 0 : 1
      return _r[p] ?? 0

    } else {
      // Deterministic: use scratch row 2n
      const sW = 2 * n * W
      _x.fill(0, sW, sW + W)
      _z.fill(0, sW, sW + W)
      _r[2 * n] = 0
      for (let i = 0; i < n; i++) {
        if (((_x[i * W + aw] ?? 0) >>> ash) & 1) this.rowmul(2 * n, i + n)
      }
      return _r[2 * n] ?? 0
    }
  }

  /**
   * Return the n stabilizer generators as signed Pauli strings, e.g. `['+XZZXI', '-IXZZX']`.
   * Rows n..2n-1 of the tableau; sign from the phase bit (0 → '+', 1 → '-').
   */
  stabilizerGenerators(): string[] {
    const { n, W, _x, _z, _r } = this
    const out: string[] = []
    for (let i = n; i < 2 * n; i++) {
      const iW = i * W
      let s = (_r[i] ?? 0) ? '-' : '+'
      for (let q = 0; q < n; q++) {
        const w = q >> 5, sh = q & 31
        const xb = ((_x[iW + w] ?? 0) >>> sh) & 1
        const zb = ((_z[iW + w] ?? 0) >>> sh) & 1
        s += xb && zb ? 'Y' : xb ? 'X' : zb ? 'Z' : 'I'
      }
      out.push(s)
    }
    return out
  }
}
