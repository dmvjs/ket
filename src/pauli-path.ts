/**
 * Expectation values by Pauli-path propagation (Heisenberg picture).
 *
 * Instead of carrying a state forward, this carries the *observable* backward
 * through the circuit: ⟨ψ|O|ψ⟩ = ⟨0|U†OU|0⟩, so O is conjugated gate by gate in
 * reverse order and read off against |0…0⟩ at the end.
 *
 * The cost model is completely different from a statevector's. Clifford gates map
 * a Pauli to ±a Pauli — one term in, one term out, at any width. Only rotations
 * branch, and each branch carries a cos/sin factor, so terms decay geometrically
 * and can be truncated with a bound on the error introduced. Width is close to
 * free; what costs is the number of surviving Pauli terms.
 *
 * That makes this efficient exactly where a statevector is not: wide, shallow, or
 * heavily-Clifford circuits, which is what VQE and QAOA ansatze are. It is not a
 * general-purpose backend — it answers ⟨O⟩ for a Pauli observable, not a full
 * distribution, and a deep circuit of arbitrary rotations will still blow up.
 *
 * Truncation is reported rather than hidden: `droppedWeight` is the summed
 * magnitude of every discarded coefficient, an upper bound on the error in ⟨O⟩.
 *
 * Conjugation tables are derived numerically from the gate matrices at load
 * rather than written out by hand, because the sign conventions are exactly the
 * kind of thing that is easy to get subtly wrong and hard to notice.
 */

import type { Circuit } from './circuit.js'
import type { PauliTerm } from './algorithms.js'

// ── Tiny complex-matrix helpers, used only to derive the tables ───────────────

type M2 = [number, number, number, number, number, number, number, number]  // 2x2, re/im interleaved

const mul2 = (a: M2, b: M2): M2 => {
  const out = new Array(8).fill(0) as M2
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    let re = 0, im = 0
    for (let k = 0; k < 2; k++) {
      const ar = a[(i * 2 + k) * 2]!, ai = a[(i * 2 + k) * 2 + 1]!
      const br = b[(k * 2 + j) * 2]!, bi = b[(k * 2 + j) * 2 + 1]!
      re += ar * br - ai * bi
      im += ar * bi + ai * br
    }
    out[(i * 2 + j) * 2] = re
    out[(i * 2 + j) * 2 + 1] = im
  }
  return out
}
const dag2 = (a: M2): M2 => [a[0]!, -a[1]!, a[4]!, -a[5]!, a[2]!, -a[3]!, a[6]!, -a[7]!]

const PAULI: Record<string, M2> = {
  I: [1, 0, 0, 0, 0, 0, 1, 0],
  X: [0, 0, 1, 0, 1, 0, 0, 0],
  Y: [0, 0, 0, -1, 0, 1, 0, 0],
  Z: [1, 0, 0, 0, 0, 0, -1, 0],
}
const LETTERS = ['I', 'X', 'Y', 'Z'] as const
export type PauliLetter = (typeof LETTERS)[number]

/** Match a matrix to ±P, or return null when it is not a signed Pauli. */
function asSignedPauli(m: M2): { p: PauliLetter; sign: number } | null {
  for (const p of LETTERS) {
    for (const sign of [1, -1]) {
      const ref = PAULI[p]!
      let ok = true
      for (let i = 0; i < 8; i++) if (Math.abs(m[i]! - sign * ref[i]!) > 1e-9) { ok = false; break }
      if (ok) return { p, sign }
    }
  }
  return null
}

/** Conjugation table for a single-qubit Clifford: P ↦ G†PG. */
function singleTable(g: M2): Record<PauliLetter, { p: PauliLetter; sign: number }> {
  const out = {} as Record<PauliLetter, { p: PauliLetter; sign: number }>
  for (const p of LETTERS) {
    const r = asSignedPauli(mul2(mul2(dag2(g), PAULI[p]!), g))
    if (!r) throw new Error(`gate is not Clifford: ${p} does not map to a signed Pauli`)
    out[p] = r
  }
  return out
}

const S2 = Math.SQRT1_2
const CLIFFORD_1Q: Record<string, Record<PauliLetter, { p: PauliLetter; sign: number }>> = {
  h:   singleTable([S2, 0, S2, 0, S2, 0, -S2, 0]),
  x:   singleTable(PAULI['X']!),
  y:   singleTable(PAULI['Y']!),
  z:   singleTable(PAULI['Z']!),
  s:   singleTable([1, 0, 0, 0, 0, 0, 0, 1]),
  sdg: singleTable([1, 0, 0, 0, 0, 0, 0, -1]),
}

/**
 * CNOT conjugation, as a map from the (control, target) Pauli pair.
 *
 * Derived from the generator rules rather than a 4x4 matrix: X_c ↦ X_c X_t,
 * Z_t ↦ Z_c Z_t, with X_t and Z_c fixed. Y is i·X·Z, so the signs fall out of
 * composing the two, which is why this is computed rather than typed in.
 */
const CNOT_TABLE: Record<string, { c: PauliLetter; t: PauliLetter; sign: number }> = (() => {
  // A single-qubit Pauli as (x, z) bits, with Y = iXZ.
  const bits: Record<PauliLetter, [number, number]> = { I: [0, 0], X: [1, 0], Y: [1, 1], Z: [0, 1] }
  const fromBits = (x: number, z: number): PauliLetter => (x && z ? 'Y' : x ? 'X' : z ? 'Z' : 'I')
  const table: Record<string, { c: PauliLetter; t: PauliLetter; sign: number }> = {}

  for (const pc of LETTERS) for (const pt of LETTERS) {
    const [xc, zc] = bits[pc], [xt, zt] = bits[pt]
    // CNOT maps (xc,zc,xt,zt) -> (xc, zc^zt, xt^xc, zt); the sign flips when the
    // symplectic form picks up a term, which is the standard tableau rule.
    const nxc = xc, nzc = zc ^ zt, nxt = xt ^ xc, nzt = zt
    const sign = (xc && zt && ((xt ^ zc ^ 1) === 1)) ? -1 : 1
    table[`${pc}${pt}`] = { c: fromBits(nxc, nzc), t: fromBits(nxt, nzt), sign }
  }
  return table
})()

// ── Pauli terms ───────────────────────────────────────────────────────────────

/**
 * One Pauli string with a real coefficient.
 *
 * `ops` indexes by qubit directly — `ops[q]` acts on qubit q — which is the
 * reverse of the public {@link PauliTerm} convention and is converted at the
 * boundary. Coefficients stay real: Clifford conjugation contributes ±1, and a
 * rotation contributes cos/sin with the i from `iAP` cancelling the i in Y.
 */
interface Term { ops: PauliLetter[]; coeff: number }

const keyOf = (ops: PauliLetter[]): string => ops.join('')

/** Result of a Pauli-path expectation, including what truncation cost. */
export interface PauliPathResult {
  /** ⟨ψ|O|ψ⟩. */
  value: number
  /** Largest number of Pauli terms held at once. */
  peakTerms: number
  /** Summed magnitude of discarded coefficients — an upper bound on the error. */
  droppedWeight: number
  /** True when any term was discarded. */
  truncated: boolean
}

export interface PauliPathOptions {
  /** Discard terms whose coefficient falls below this magnitude. */
  threshold?: number
  /** Hard cap on simultaneous terms; the smallest are dropped first. */
  maxTerms?: number
  /**
   * Discard Pauli terms acting non-trivially on more than this many qubits.
   *
   * Weight is the other axis truncation runs along, and under noise it is the
   * principled one: a depolarizing channel damps a weight-w Pauli by a factor
   * that shrinks geometrically in w, so high-weight terms contribute
   * exponentially little and dropping them is cheap in accuracy.
   */
  maxWeight?: number
  /**
   * Depolarizing rates, matching `DmNoiseParams`: `p1` per single-qubit gate,
   * `p2` per two-qubit gate.
   *
   * Simulating the *noisy* circuit is both cheaper and a better model of the
   * hardware than simulating the ideal one. Each channel damps any term acting
   * non-trivially on the affected qubits — by 1−4p/3 for one qubit and 1−16p/15
   * for two, the same convention the density-matrix backend uses — so terms die
   * off and truncation costs progressively less.
   */
  noise?: { p1?: number; p2?: number }
}

/** Multiply single-qubit Paulis: returns the product and its power of i. */
function mulLetter(a: PauliLetter, b: PauliLetter): { p: PauliLetter; iPow: number } {
  if (a === 'I') return { p: b, iPow: 0 }
  if (b === 'I') return { p: a, iPow: 0 }
  if (a === b) return { p: 'I', iPow: 0 }
  const cyc: Record<string, PauliLetter> = { XY: 'Z', YZ: 'X', ZX: 'Y', YX: 'Z', ZY: 'X', XZ: 'Y' }
  const p = cyc[a + b]!
  // XY = iZ and YX = -iZ, and likewise round the cycle.
  const forward = (a + b) === 'XY' || (a + b) === 'YZ' || (a + b) === 'ZX'
  return { p, iPow: forward ? 1 : 3 }
}

/**
 * Compute ⟨0…0|U† O U|0…0⟩ for a Pauli observable, by propagating O backward.
 *
 * @param circuit    the circuit U, which must be pure (no measure/reset/if)
 * @param observable Pauli terms in the {@link PauliTerm} convention
 */
export function pauliPathExpectation(
  circuit: Circuit,
  observable: readonly PauliTerm[],
  { threshold = 1e-10, maxTerms = Infinity, maxWeight = Infinity, noise }: PauliPathOptions = {},
): PauliPathResult {
  const n = circuit.qubits
  const json = circuit.toJSON()

  // Seed with the observable, converted to qubit-indexed order.
  let terms = new Map<string, Term>()
  for (const { coeff, ops } of observable) {
    if (ops.length !== n) throw new TypeError(`observable '${ops}' must have one letter per qubit (${n})`)
    const letters: PauliLetter[] = []
    for (let q = 0; q < n; q++) {
      const ch = ops[n - 1 - q]!.toUpperCase()
      if (ch !== 'I' && ch !== 'X' && ch !== 'Y' && ch !== 'Z')
        throw new TypeError(`observable '${ops}' contains '${ch}', expected I, X, Y or Z`)
      letters.push(ch)
    }
    const k = keyOf(letters)
    const existing = terms.get(k)
    if (existing) existing.coeff += coeff
    else terms.set(k, { ops: letters, coeff })
  }

  let peakTerms = terms.size
  let droppedWeight = 0
  const p1 = noise?.p1 ?? 0, p2 = noise?.p2 ?? 0
  const damp1 = 1 - 4 * p1 / 3, damp2 = 1 - 16 * p2 / 15
  const weightOf = (o: PauliLetter[]): number => { let w = 0; for (const l of o) if (l !== 'I') w++; return w }

  /** Fold a produced term into the accumulator, cancelling where keys collide. */
  const add = (into: Map<string, Term>, ops: PauliLetter[], coeff: number): void => {
    const k = keyOf(ops)
    const hit = into.get(k)
    if (hit) hit.coeff += coeff
    else into.set(k, { ops, coeff })
  }

  // U = G_k···G_1, so U†OU conjugates by G_k first: walk the gate list backward.
  const ops = json.ops as Record<string, unknown>[]
  for (let i = ops.length - 1; i >= 0; i--) {
    const op = ops[i]!
    const kind = op['kind'] as string
    if (kind === 'barrier') continue
    if (kind === 'measure' || kind === 'reset' || kind === 'if')
      throw new TypeError('pauliPathExpectation requires a pure circuit — remove measure/reset/if ops')

    // ⟨O⟩ = Tr(O Λ(UρU†)) = Tr(U† Λ†(O) U ρ): walking backward, the channel that
    // followed a gate acts on the observable *before* that gate's conjugation.
    // Depolarizing is self-adjoint, so Λ† is the same damping.
    if (p1 > 0 || p2 > 0) {
      const q = op['q'] as number | undefined
      const a = (op['control'] ?? op['a'] ?? op['c1']) as number | undefined
      const b = (op['target'] ?? op['b'] ?? op['c2']) as number | undefined
      for (const term of terms.values()) {
        if (kind === 'single' && p1 > 0 && q !== undefined) {
          if (term.ops[q] !== 'I') term.coeff *= damp1
        } else if (p2 > 0 && a !== undefined && b !== undefined) {
          if (term.ops[a] !== 'I' || term.ops[b] !== 'I') term.coeff *= damp2
        }
      }
    }

    const next = new Map<string, Term>()

    if (kind === 'cnot' || (kind === 'controlled' && (op['meta'] as { name?: string })?.name === 'cx')) {
      const c = op['control'] as number, t = op['target'] as number
      for (const term of terms.values()) {
        const r = CNOT_TABLE[`${term.ops[c]!}${term.ops[t]!}`]!
        const o = term.ops.slice()
        o[c] = r.c; o[t] = r.t
        add(next, o, term.coeff * r.sign)
      }
    } else if (kind === 'single') {
      const q = op['q'] as number
      const meta = op['meta'] as { name: string; params?: number[] }
      const clifford = CLIFFORD_1Q[meta.name]
      if (clifford) {
        for (const term of terms.values()) {
          const r = clifford[term.ops[q]!]
          const o = term.ops.slice()
          o[q] = r.p
          add(next, o, term.coeff * r.sign)
        }
      } else {
        // Rotation about a Pauli axis: P ↦ cosθ·P + sinθ·(iAP) when P
        // anticommutes with A, and P is unchanged when it commutes.
        const rot = rotationOf(meta)
        if (!rot) throw new TypeError(
          `pauliPathExpectation does not support gate '${meta.name}'. ` +
          `Supported: h, x, y, z, s, sdg, t, tdg, rx, ry, rz, u1, p, cnot, cz, swap.`)
        const { axis, theta } = rot
        const cos = Math.cos(theta), sin = Math.sin(theta)
        for (const term of terms.values()) {
          const p = term.ops[q]!
          if (p === 'I' || p === axis) { add(next, term.ops.slice(), term.coeff); continue }
          // iAP: the i cancels against the i in the Pauli product, leaving ±1.
          const { p: prod, iPow } = mulLetter(axis, p)
          const realSign = iPow === 1 ? -1 : 1          // i·(i) = -1, i·(-i) = +1
          add(next, term.ops.slice(), term.coeff * cos)
          const o = term.ops.slice()
          o[q] = prod
          add(next, o, term.coeff * sin * realSign)
        }
      }
    } else if (kind === 'swap') {
      const a = op['a'] as number, b = op['b'] as number
      for (const term of terms.values()) {
        const o = term.ops.slice()
        o[a] = term.ops[b]!; o[b] = term.ops[a]!
        add(next, o, term.coeff)
      }
    } else if (kind === 'controlled' && (op['meta'] as { name?: string })?.name === 'cz') {
      // CZ = H_t · CNOT · H_t on the target.
      const c = op['control'] as number, t = op['target'] as number
      const h = CLIFFORD_1Q['h']!
      for (const term of terms.values()) {
        const o = term.ops.slice()
        let sign = 1
        const h1 = h[o[t]!]; o[t] = h1.p; sign *= h1.sign
        const r = CNOT_TABLE[`${o[c]!}${o[t]!}`]!
        o[c] = r.c; o[t] = r.t; sign *= r.sign
        const h2 = h[o[t]!]; o[t] = h2.p; sign *= h2.sign
        add(next, o, term.coeff * sign)
      }
    } else {
      throw new TypeError(`pauliPathExpectation does not support op '${kind}'`)
    }

    // Prune: drop what is below threshold, then the smallest if still over cap.
    terms = new Map()
    for (const [k, t] of next) {
      if (Math.abs(t.coeff) < threshold || weightOf(t.ops) > maxWeight) {
        droppedWeight += Math.abs(t.coeff)
        continue
      }
      terms.set(k, t)
    }
    if (terms.size > maxTerms) {
      const sorted = [...terms.entries()].sort((a, b) => Math.abs(b[1].coeff) - Math.abs(a[1].coeff))
      for (const [k, t] of sorted.slice(maxTerms)) { droppedWeight += Math.abs(t.coeff); terms.delete(k) }
    }
    peakTerms = Math.max(peakTerms, terms.size)
  }

  // ⟨0…0|P|0…0⟩ is 1 for a product of I and Z, and 0 as soon as any X or Y
  // appears, since those move the state off |0…0⟩.
  let value = 0
  for (const term of terms.values()) {
    if (term.ops.every(p => p === 'I' || p === 'Z')) value += term.coeff
  }

  return { value, peakTerms, droppedWeight, truncated: droppedWeight > 0 }
}

/** Interpret a gate as a rotation about a Pauli axis, if it is one. */
function rotationOf(meta: { name: string; params?: number[] }): { axis: PauliLetter; theta: number } | null {
  // Rx(a) = exp(-i a X / 2), and conjugation gives P -> cos(a)P + sin(a)(iXP):
  // the halves in the exponent cancel, so this is the full gate angle, not half
  // of it. Getting that wrong is invisible at a = 0 and wrong everywhere else.
  const p = meta.params?.[0] ?? 0
  switch (meta.name) {
    case 'rx': return { axis: 'X', theta: p }
    case 'ry': return { axis: 'Y', theta: p }
    case 'rz': return { axis: 'Z', theta: p }
    // u1/p are Rz up to a global phase, which conjugation cancels.
    case 'u1': case 'p': return { axis: 'Z', theta: p }
    case 't':   return { axis: 'Z', theta: Math.PI / 4 }
    case 'ti': case 'tdg': return { axis: 'Z', theta: -Math.PI / 4 }
    default: return null
  }
}
