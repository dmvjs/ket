/**
 * CHP (Aaronson-Gottesman 2004) Clifford stabilizer simulator.
 *
 * Maintains a (2n+1) × n binary tableau:
 *   Rows 0..n-1:   destabilizer generators
 *   Rows n..2n-1:  stabilizer generators
 *   Row 2n:        scratch row for deterministic measurement
 *
 * Each row has x[row*n..row*n+n-1], z[row*n..row*n+n-1], and r[row].
 * Phase bit r=0 means +, r=1 means -.
 */
export class CliffordSim {
  readonly n: number
  // Using regular number arrays internally to avoid Uint8Array undefined indexing issues
  private readonly _x: number[]  // (2n+1)*n
  private readonly _z: number[]  // (2n+1)*n
  private readonly _r: number[]  // 2n+1

  constructor(n: number) {
    this.n  = n
    const rows = 2 * n + 1
    this._x = new Array<number>(rows * n).fill(0)
    this._z = new Array<number>(rows * n).fill(0)
    this._r = new Array<number>(rows).fill(0)

    // Initial state |0...0⟩:
    //   destabilizer i: x[i*n+i] = 1  (X_i generator)
    //   stabilizer i:   z[(i+n)*n+i] = 1  (Z_i generator)
    for (let i = 0; i < n; i++) {
      this._x[i * n + i]       = 1
      this._z[(i + n) * n + i] = 1
    }
  }

  /**
   * Phase contribution function g(x1,z1, x2,z2) used in rowmul.
   * Returns an integer in {-1, 0, 1} representing contributions to
   * the exponent of i (so 0 mod 4 → phase 0, 2 mod 4 → phase -1).
   */
  private g(x1: number, z1: number, x2: number, z2: number): number {
    if (x1 === 0 && z1 === 0) return 0
    if (x1 === 1 && z1 === 1) return z2 - x2          // can be -1, 0, or 1
    if (x1 === 1 && z1 === 0) return z2 * (2 * x2 - 1)  // 0 if x2=0 else z2*(2*0-1)=−z2... fix: z2*(2x2-1)
    // x1=0, z1=1
    return x2 * (1 - 2 * z2)
  }

  /**
   * Multiply row i (in place) by row h: tableau row i ← row_i · row_h.
   * Uses the phase function g to accumulate phase correctly.
   */
  private rowmul(i: number, h: number): void {
    const { n, _x, _z, _r } = this
    let sum = 2 * ((_r[h] ?? 0) + (_r[i] ?? 0))
    for (let j = 0; j < n; j++) {
      sum += this.g(_x[h * n + j] ?? 0, _z[h * n + j] ?? 0, _x[i * n + j] ?? 0, _z[i * n + j] ?? 0)
    }
    // sum mod 4: if 0 → r=0, if 2 → r=1 (sum is always even)
    _r[i] = ((((sum % 4) + 4) % 4) >> 1) & 1
    for (let j = 0; j < n; j++) {
      _x[i * n + j] = ((_x[i * n + j] ?? 0) ^ (_x[h * n + j] ?? 0)) & 1
      _z[i * n + j] = ((_z[i * n + j] ?? 0) ^ (_z[h * n + j] ?? 0)) & 1
    }
  }

  // ── Single-qubit gates ────────────────────────────────────────────────────

  /** Hadamard gate on qubit a. */
  h(a: number): void {
    const { n, _x, _z, _r } = this
    for (let i = 0; i < 2 * n; i++) {
      const xi = _x[i * n + a] ?? 0
      const zi = _z[i * n + a] ?? 0
      _r[i] = ((_r[i] ?? 0) ^ (xi & zi)) & 1
      _x[i * n + a] = zi
      _z[i * n + a] = xi
    }
  }

  /** Phase gate S on qubit a. */
  s(a: number): void {
    const { n, _x, _z, _r } = this
    for (let i = 0; i < 2 * n; i++) {
      const xi = _x[i * n + a] ?? 0
      const zi = _z[i * n + a] ?? 0
      _r[i]       = ((_r[i] ?? 0) ^ (xi & zi)) & 1
      _z[i * n + a] = (zi ^ xi) & 1
    }
  }

  /** S† (inverse phase gate) on qubit a — applied as S three times. */
  si(a: number): void {
    this.s(a); this.s(a); this.s(a)
  }

  /** Pauli X gate on qubit a. */
  x(a: number): void {
    // X = H S² H
    this.h(a); this.s(a); this.s(a); this.h(a)
  }

  /**
   * Pauli Y gate on qubit a.
   * Y anticommutes with X (x=1,z=0: r ^= 1) and Z (x=0,z=1: r ^= 1),
   * commutes with Y itself (x=1,z=1: r ^= 0). So: r[i] ^= x[i,a] XOR z[i,a].
   */
  y(a: number): void {
    const { n, _x, _z, _r } = this
    for (let i = 0; i < 2 * n; i++) {
      const xi = _x[i * n + a] ?? 0
      const zi = _z[i * n + a] ?? 0
      _r[i] = ((_r[i] ?? 0) ^ (xi ^ zi)) & 1
    }
  }

  /** Pauli Z gate on qubit a. Z anticommutes with X, commutes with Z. */
  z(a: number): void {
    const { n, _x, _r } = this
    for (let i = 0; i < 2 * n; i++) {
      _r[i] = ((_r[i] ?? 0) ^ (_x[i * n + a] ?? 0)) & 1
    }
  }

  // ── Two-qubit gates ───────────────────────────────────────────────────────

  /** CNOT gate: control qubit a, target qubit b. */
  cnot(a: number, b: number): void {
    const { n, _x, _z, _r } = this
    for (let i = 0; i < 2 * n; i++) {
      const xa = _x[i * n + a] ?? 0
      const xb = _x[i * n + b] ?? 0
      const za = _z[i * n + a] ?? 0
      const zb = _z[i * n + b] ?? 0
      _r[i]       = ((_r[i] ?? 0) ^ (xa & zb & ((xb ^ za ^ 1) & 1))) & 1
      _x[i * n + b] = (xb ^ xa) & 1
      _z[i * n + a] = (za ^ zb) & 1
    }
  }

  /** Controlled-Z gate on qubits a and b. */
  cz(a: number, b: number): void {
    this.h(b); this.cnot(a, b); this.h(b)
  }

  /** Controlled-Y gate on qubits a (control) and b (target). */
  cy(a: number, b: number): void {
    this.si(b); this.cnot(a, b); this.s(b)
  }

  /** SWAP gate on qubits a and b. */
  swap(a: number, b: number): void {
    this.cnot(a, b); this.cnot(b, a); this.cnot(a, b)
  }

  // ── Measurement ───────────────────────────────────────────────────────────

  /**
   * Measure qubit a. rand is a uniform random number in [0,1) used for random outcomes.
   * Returns 0 or 1.
   */
  measure(a: number, rand: number): number {
    const { n, _x, _z, _r } = this

    // Find p in stabilizer rows (n..2n-1) with x[p,a] = 1 → random outcome
    let p = -1
    for (let i = n; i < 2 * n; i++) {
      if ((_x[i * n + a] ?? 0) === 1) { p = i; break }
    }

    if (p !== -1) {
      // Random measurement outcome
      // Update all rows i (except p) where x[i,a] = 1
      for (let i = 0; i < 2 * n; i++) {
        if (i !== p && (_x[i * n + a] ?? 0) === 1) {
          this.rowmul(i, p)
        }
      }

      // Copy row p to row p-n (destabilizer gets old stabilizer)
      const destRow = p - n
      for (let j = 0; j < n; j++) {
        _x[destRow * n + j] = _x[p * n + j] ?? 0
        _z[destRow * n + j] = _z[p * n + j] ?? 0
      }
      _r[destRow] = _r[p] ?? 0

      // Set new stabilizer row p: all zero, z[p,a]=1, phase = measurement result
      for (let j = 0; j < n; j++) {
        _x[p * n + j] = 0
        _z[p * n + j] = 0
      }
      _z[p * n + a] = 1
      _r[p] = rand < 0.5 ? 0 : 1

      return _r[p] ?? 0

    } else {
      // Deterministic measurement: use scratch row 2n
      const scratch = 2 * n
      // Initialize scratch row to zero
      for (let j = 0; j < n; j++) {
        _x[scratch * n + j] = 0
        _z[scratch * n + j] = 0
      }
      _r[scratch] = 0

      // For each destabilizer row i where x[i][a] = 1, multiply scratch by
      // the corresponding stabilizer row i+n. This propagates the Z_a component.
      for (let i = 0; i < n; i++) {
        if ((_x[i * n + a] ?? 0) === 1) {
          this.rowmul(scratch, i + n)
        }
      }

      return _r[scratch] ?? 0
    }
  }
}
