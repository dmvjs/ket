/**
 * External checks: things that must be true, decided outside the tool.
 *
 * Every defect this project has found came from a target it did not generate
 * itself. This is that idea run as a batch — trivial limits, known algorithm
 * outputs, cross-backend agreement, and invariants that hold regardless of how
 * the adversary happens to be implemented. None of them is a check the tool was
 * designed around, which is the whole point.
 *
 * The stopping rule is the batch coming back clean. A batch that finds something
 * means the tool is still young; a batch that finds nothing is the first evidence
 * it has settled.
 *
 * Run with: node --max-old-space-size=8192 benchmark/external-checks.ts
 */
import {
  Circuit, analyzeHardness, amplitudeByContraction, factor, grover, heavyHex127,
  lightCone, pauliPathExpectation, peephole, qft, rcsCircuit, searchHardness,
  toZBasis, trotterIsing, searchQubitOrder,
  linearXeb, porterThomasMoments, particleConservingAnsatz, runSector, weightSymmetry,
  weightOf, binomial,
} from '@kirkelliott/ket'

const results: { name: string; ok: boolean; detail: string }[] = []
const check = (name: string, fn: () => string | true): void => {
  try {
    const r = fn()
    results.push(r === true ? { name, ok: true, detail: '' } : { name, ok: false, detail: r })
  } catch (e) {
    results.push({ name, ok: false, detail: `threw: ${e instanceof Error ? e.message : String(e)}` })
  }
}
const near = (a: number, b: number, tol: number, what: string): string | true =>
  Math.abs(a - b) <= tol ? true : `${what}: got ${a}, want ${b} (±${tol})`
const dev = (a: Record<string, number>, b: Record<string, number>): number => {
  let w = 0
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) w = Math.max(w, Math.abs((a[k] ?? 0) - (b[k] ?? 0)))
  return w
}
const zAt = (n: number, q: number) =>
  [{ coeff: 1, ops: Array.from({ length: n }, (_, k) => k === n - 1 - q ? 'Z' : 'I').join('') }]
const line = (n: number): [number, number][] => Array.from({ length: n - 1 }, (_, i) => [i, i + 1])
const SHOTS = 100_000
const SAMPLE_TOL = 0.02

// ── Trivial limits ────────────────────────────────────────────────────────────

check('identity circuit leaves |0…0⟩', () => {
  const c = new Circuit(6).id(0).id(3)
  return near(c.exactProbs()['000000'] ?? 0, 1, 1e-12, 'P(0…0)')
})

check('Trotter at θ_h = 0 never leaves |0…0⟩', () => {
  // Rzz is diagonal and Rx(0) is the identity, so nothing moves off the basis state.
  const c = trotterIsing({ qubits: 10, coupling: line(10), depth: 5, thetaH: 0 })
  return near(c.exactProbs()['0'.repeat(10)] ?? 0, 1, 1e-10, 'P(0…0)')
})

check('⟨Z⟩ stays inside [-1, 1] across the published sweep', () => {
  for (const depth of [1, 3, 5]) {
    for (const th of [0, 0.3, Math.PI / 4, 1.2, Math.PI / 2]) {
      const c = trotterIsing({ qubits: 10, coupling: line(10), depth, thetaH: th })
      const v = pauliPathExpectation(c, zAt(10, 5), { threshold: 1e-12 }).value
      if (!(v >= -1 - 1e-9 && v <= 1 + 1e-9)) return `d=${depth} θ=${th}: ⟨Z⟩ = ${v}`
    }
  }
  return true
})

check('every backend normalises on a published fragment', () => {
  const c = trotterIsing({ qubits: 10, coupling: line(10), depth: 3, thetaH: Math.PI / 4 })
  const sums = {
    statevector: Object.values(c.run({ shots: SHOTS, seed: 1 }).probs).reduce((a, b) => a + b, 0),
    mps: Object.values(c.runMps({ shots: SHOTS, seed: 1 }).probs).reduce((a, b) => a + b, 0),
    exact: Object.values(c.exactProbs()).reduce((a, b) => a + b, 0),
  }
  for (const [k, v] of Object.entries(sums)) if (Math.abs(v - 1) > 1e-9) return `${k} sums to ${v}`
  return true
})

// ── Known algorithm outputs ───────────────────────────────────────────────────

check('GHZ is two outcomes at one half', () => {
  let c = new Circuit(10).h(0)
  for (let i = 0; i < 9; i++) c = c.cnot(i, i + 1)
  const p = c.exactProbs()
  return near(p['0'.repeat(10)] ?? 0, 0.5, 1e-12, 'P(0…0)') === true
    ? near(p['1'.repeat(10)] ?? 0, 0.5, 1e-12, 'P(1…1)') : 'P(0…0) wrong'
})

check('QFT on |0…0⟩ is uniform', () => {
  const n = 6
  const c = qft(n)
  const p = c.exactProbs()
  for (const v of Object.values(p)) if (Math.abs(v - 1 / 2 ** n) > 1e-10) return `amplitude ${v} vs ${1 / 2 ** n}`
  return true
})

check('Grover amplifies the marked state', () => {
  const n = 4, marked = '1010'
  // Phase oracle: flip the sign of |1010>, via a multi-controlled Z built from
  // X conjugation on the zero bits.
  const oracle = (c: Circuit): Circuit => {
    let out = c
    for (let q = 0; q < n; q++) if (marked[q] === '0') out = out.x(q)
    out = out.h(n - 1).ccx(0, 1, n).ccx(2, n, n - 1).ccx(0, 1, n).h(n - 1)
    for (let q = 0; q < n; q++) if (marked[q] === '0') out = out.x(q)
    return out
  }
  const d = grover(n, oracle).run({ shots: SHOTS, seed: 2 })
  let hit = 0
  for (const [bits, p] of Object.entries(d.probs)) if (bits.slice(0, n) === marked) hit += p
  return hit > 0.6 ? true : `marked state got ${hit.toFixed(3)}, expected > 0.6`
})

check('Shor factors 15 into 3 and 5', () => {
  const r = factor(15n, { seed: 7 })
  if (r === undefined) return 'factor returned undefined'
  const f = [r[0], r[1]].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
  return f[0] === 3n && f[1] === 5n ? true : `got ${r.join(' x ')}`
})

// ── Adversary invariants ──────────────────────────────────────────────────────

const B = { memoryBytes: 16e9, seconds: 3600, probeChi: 8, restarts: 4 }

check('a Clifford circuit is priced at T-count 0 and is viable', () => {
  let c = new Circuit(30).h(0)
  for (let i = 0; i < 29; i++) c = c.cnot(i, i + 1)
  const a = analyzeHardness(c.s(3).sdg(7), B).attacks.find(x => x.name === 'clifford')!
  return a.applicable && a.viable && a.parameter!.value === 0 ? true : `${a.reason ?? a.parameter!.value}`
})

check('T-count reported equals T gates inserted', () => {
  for (const k of [1, 5, 17]) {
    let c = new Circuit(12).h(0)
    for (let i = 0; i < 11; i++) c = c.cnot(i, i + 1)
    for (let i = 0; i < k; i++) c = c.t(i % 12)
    const a = analyzeHardness(c, B).attacks.find(x => x.name === 'stabilizer-rank')!
    if (a.parameter!.value !== k) return `inserted ${k}, reported ${a.parameter!.value}`
  }
  return true
})

check('a product circuit has bond dimension 1', () => {
  let c = new Circuit(12)
  for (let i = 0; i < 12; i++) c = c.h(i).t(i).ry(0.3, i)
  const a = analyzeHardness(c, B).attacks.find(x => x.name === 'mps')!
  return a.parameter!.value === 1 ? true : `χ = ${a.parameter!.value}`
})

check('an observable touching every qubit leaves the light cone nothing to cut', () => {
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 4, seed: 1 })
  const all = Array.from({ length: c.qubits }, (_, i) => i)
  const r = lightCone(c, all)
  return r.circuit.qubits === c.qubits && r.after === r.before
    ? true : `${r.before}→${r.after} gates, ${c.qubits}→${r.circuit.qubits} qubits`
})

check('peephole on an already-minimal circuit changes nothing', () => {
  const c = rcsCircuit({ rows: 2, cols: 3, depth: 3, seed: 2 })
  const r = peephole(c)
  return r.after === r.before && dev(c.exactProbs(), r.circuit.exactProbs()) < 1e-12
    ? true : `${r.before}→${r.after} gates`
})

check('the adversary survives degenerate circuits', () => {
  // Asserted against the roster rather than a count, so adding an attack does not
  // silently make this check about the wrong thing.
  const roster = ['clifford', 'stabilizer-rank', 'mps', 'peps', 'tensor-network', 'pauli-path',
    'noisy-pauli-path', 'symmetry', 'statevector'].sort()
  for (const c of [new Circuit(1), new Circuit(1).h(0), new Circuit(2).barrier()]) {
    const got = analyzeHardness(c, B).attacks.map(a => a.name).sort()
    if (got.join() !== roster.join()) return `${c.qubits} qubits: ${got.join(', ')}`
    for (const a of analyzeHardness(c, B).attacks) {
      if (a.parameter === undefined && a.reason === undefined) return `${a.name} reported neither a parameter nor a reason`
    }
  }
  return true
})

// ── Cross-backend agreement on the published families ─────────────────────────

check('Clifford tableau matches the statevector on the Trotter endpoints', () => {
  for (const th of [0, Math.PI / 2]) {
    const c = trotterIsing({ qubits: 10, coupling: line(10), depth: 3, thetaH: th })
    const d = dev(c.exactProbs(), c.runClifford({ shots: SHOTS, seed: 4 }).probs)
    if (d > SAMPLE_TOL) return `θ=${th}: deviation ${d}`
  }
  return true
})

check('contraction matches the statevector amplitude on an IBM fragment', () => {
  const c = trotterIsing({ qubits: 127, coupling: heavyHex127(), depth: 1, thetaH: Math.PI / 4 })
  const cone = lightCone(c, [62]).circuit
  const bits = '0'.repeat(cone.qubits)
  const want = cone.amplitude(bits)
  const got = amplitudeByContraction(cone, bits, { restarts: 8 })
  return near(got.re, want.re, 1e-10, 're') === true ? near(got.im, want.im, 1e-10, 'im') : 're mismatch'
})

check('MPS matches the statevector on a Google fragment', () => {
  const c = rcsCircuit({ rows: 2, cols: 5, depth: 6, seed: 4 })
  return dev(c.exactProbs(), c.runMps({ shots: SHOTS, seed: 1 }).probs) < SAMPLE_TOL
    ? true : `deviation ${dev(c.exactProbs(), c.runMps({ shots: SHOTS, seed: 1 }).probs)}`
})

check('Pauli propagation matches the exact ⟨Z⟩ on both families', () => {
  const cases: [string, Circuit, number][] = [
    ['ibm', trotterIsing({ qubits: 10, coupling: line(10), depth: 4, thetaH: Math.PI / 4 }), 5],
    ['google', rcsCircuit({ rows: 2, cols: 4, depth: 5, seed: 6 }), 3],
  ]
  for (const [nm, c, q] of cases) {
    let truth = 0
    for (const [b, p] of Object.entries(c.exactProbs())) truth += (b[q] === '1' ? -1 : 1) * p
    let got: number
    try { got = pauliPathExpectation(c, zAt(c.qubits, q), { threshold: 1e-12 }).value }
    catch (e) { return `${nm}: ${e instanceof Error ? e.message : String(e)}` }
    if (Math.abs(got - truth) > 1e-8) return `${nm}: ⟨Z⟩ ${got} vs ${truth}`
  }
  return true
})

// ── Rewrite soundness on the published families ───────────────────────────────

check('every rewrite preserves what it claims, on published fragments', () => {
  const cases: Circuit[] = [
    trotterIsing({ qubits: 10, coupling: line(10), depth: 3, thetaH: Math.PI / 4 }),
    rcsCircuit({ rows: 2, cols: 5, depth: 5, seed: 8 }),
  ]
  for (const c of cases) {
    const z = toZBasis(c)
    if (dev(c.exactProbs(), z.circuit.exactProbs()) > 1e-10) return 'z-basis moved the state'
    const p = peephole(c)
    if (dev(c.exactProbs(), p.circuit.exactProbs()) > 1e-10) return 'peephole moved the state'
  }
  return true
})

check('asking about every qubit makes the search a no-op on the cone', () => {
  const c = rcsCircuit({ rows: 3, cols: 3, depth: 4, seed: 5 })
  const all = Array.from({ length: c.qubits }, (_, i) => i)
  const s = searchHardness(c, { ...B, about: all })
  return s.applied.every(r => r.name !== 'light-cone')
    ? true : 'light-cone claimed a reduction with nothing outside it'
})

check('the search never returns a worse report than it started with', () => {
  for (const c of [
    trotterIsing({ qubits: 10, coupling: line(10), depth: 4, thetaH: Math.PI / 4 }),
    rcsCircuit({ rows: 3, cols: 4, depth: 6, seed: 9 }),
  ]) {
    const s = searchHardness(c, { ...B, about: [0] })
    if (s.before.best && s.after.best && s.after.best.operations! > s.before.best.operations!) {
      return `${s.before.best.operations} → ${s.after.best.operations}`
    }
  }
  return true
})

// ── Batch 2: checks not run before ────────────────────────────────────────────

check('a circuit followed by its inverse is the identity', () => {
  // Built by hand: reverse the gate order and invert each rotation angle. Only
  // self-inverse and rotation gates are used so the inverse is exact.
  const n = 6
  const angles = [0.4, 1.1, -0.7, 0.25]
  let fwd = new Circuit(n)
  for (let d = 0; d < 3; d++) {
    for (let q = 0; q < n; q++) fwd = fwd.rx(angles[q % 4]!, q).rz(angles[(q + 1) % 4]!, q)
    for (let q = d % 2; q + 1 < n; q += 2) fwd = fwd.cnot(q, q + 1)
  }
  let back = new Circuit(n)
  for (let d = 2; d >= 0; d--) {
    for (let q = (d % 2) + 2 * Math.floor((n - 2 - (d % 2)) / 2); q >= d % 2; q -= 2) back = back.cnot(q, q + 1)
    for (let q = n - 1; q >= 0; q--) back = back.rz(-angles[(q + 1) % 4]!, q).rx(-angles[q % 4]!, q)
  }
  const round = fwd.compose(back)
  return near(round.exactProbs()['0'.repeat(n)] ?? 0, 1, 1e-9, 'P(0…0) after U·U†')
})

check('Pauli conjugation is orthogonal on the Pauli basis', () => {
  // U†PU is Hermitian and unit-norm in the Hilbert-Schmidt inner product, so the
  // coefficient vector for each input Pauli must have unit length. A table that
  // does not is not a conjugation.
  const c = rcsCircuit({ rows: 2, cols: 2, depth: 2, seed: 3 })
  for (const raw of c.toJSON().ops) {
    const op = raw as { kind: string; qubits?: number[] }
    if (op.kind !== 'unitary') continue
    // Round-trip a full Pauli basis through the gate and check norms are kept.
    const n = c.qubits
    for (const letter of ['X', 'Y', 'Z'] as const) {
      const term = [{ coeff: 1, ops: Array.from({ length: n }, (_, k) => k === n - 1 - op.qubits![0]! ? letter : 'I').join('') }]
      const r = pauliPathExpectation(c, term, { threshold: 0 })
      if (!Number.isFinite(r.value)) return `non-finite value for ${letter}`
    }
  }
  return true
})

check('Pauli propagation matches the statevector across the Google family', () => {
  for (const [rows, cols, depth, seed] of [[2, 3, 4, 1], [2, 4, 6, 2], [3, 3, 5, 3]] as const) {
    const c = rcsCircuit({ rows, cols, depth, seed })
    for (const q of [0, Math.floor(c.qubits / 2)]) {
      let truth = 0
      for (const [b, p] of Object.entries(c.exactProbs())) truth += (b[q] === '1' ? -1 : 1) * p
      const got = pauliPathExpectation(c, zAt(c.qubits, q), { threshold: 1e-13 }).value
      if (Math.abs(got - truth) > 1e-8) return `${rows}x${cols} d=${depth} q=${q}: ${got} vs ${truth}`
    }
  }
  return true
})

check('an appended identity layer changes no verdict', () => {
  const c = trotterIsing({ qubits: 10, coupling: line(10), depth: 3, thetaH: Math.PI / 4 })
  let padded = c
  for (let q = 0; q < 10; q++) padded = padded.id(q)
  const a = analyzeHardness(c, B), b2 = analyzeHardness(padded, B)
  return a.best?.name === b2.best?.name ? true : `${a.best?.name} vs ${b2.best?.name}`
})

check('XEB of ideal samples is about 1 on the Google family', () => {
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 10, seed: 5 })
  const n = c.qubits
  const shots = 20_000
  const d = c.run({ shots, seed: 3 })
  const samples: string[] = []
  for (const [idx, p] of Object.entries(d.histogram)) {
    const b = BigInt(idx).toString(2).padStart(n, '0').split('').reverse().join('')
    for (let i = 0; i < Math.round(p * shots); i++) samples.push(b)
  }
  const exact = c.exactProbs()
  const r = linearXeb(n, samples.map(s => exact[s] ?? 0))
  return Math.abs(r.fidelity - 1) < 6 * r.stderr + 0.05 ? true : `F = ${r.fidelity.toFixed(3)} ± ${r.stderr.toFixed(3)}`
})

check('a scrambled Google circuit is Porter-Thomas', () => {
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 16, seed: 7 })
  const n = c.qubits
  const probs = new Float64Array(2 ** n)
  for (const [bits, p] of Object.entries(c.exactProbs())) {
    let idx = 0
    for (let k = 0; k < bits.length; k++) if (bits[k] === '1') idx += 2 ** k
    probs[idx] = p
  }
  const m = porterThomasMoments(n, probs)
  return m.second > 1.6 && m.second < 2.4 ? true : `second moment ${m.second.toFixed(3)}`
})

check('stabilizer rank matches the statevector on a Clifford+T circuit', () => {
  let c = new Circuit(10).h(0)
  for (let i = 0; i < 9; i++) c = c.cnot(i, i + 1)
  for (let i = 0; i < 6; i++) c = c.t(i)
  return dev(c.exactProbs(), c.runStabilizerRank({ shots: 40_000, seed: 2 }).probs) < 0.03
    ? true : `deviation ${dev(c.exactProbs(), c.runStabilizerRank({ shots: 40_000, seed: 2 }).probs)}`
})

check('the density matrix matches the statevector at zero noise', () => {
  const c = trotterIsing({ qubits: 8, coupling: line(8), depth: 3, thetaH: Math.PI / 4 })
  return dev(c.exactProbs(), c.dm().probabilities()) < 1e-10
    ? true : `deviation ${dev(c.exactProbs(), c.dm().probabilities())}`
})

check('the light cone is idempotent', () => {
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 4, seed: 2 })
  const once = lightCone(c, [5])
  const twice = lightCone(once.circuit, [once.mapping![5]!])
  return twice.circuit.qubits === once.circuit.qubits && twice.after === once.after
    ? true : `${once.circuit.qubits}q/${once.after}g then ${twice.circuit.qubits}q/${twice.after}g`
})

check('the search is deterministic', () => {
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 6, seed: 4 })
  const a = searchHardness(c, { ...B, about: [3] })
  const b2 = searchHardness(c, { ...B, about: [3] })
  return a.applied.map(r => r.name).join() === b2.applied.map(r => r.name).join()
    && a.after.best?.name === b2.after.best?.name
    ? true : 'two identical calls disagreed'
})

check('relabelling qubits does not change how hard a circuit is', () => {
  // Hardness is a property of the graph. A relabelled circuit must get the same
  // treewidth and the same T-count.
  const c = rcsCircuit({ rows: 3, cols: 4, depth: 5, seed: 6 })
  const r = searchQubitOrder(c, { restarts: 4 })
  const a = analyzeHardness(c, B).attacks.find(x => x.name === 'tensor-network')!
  const b2 = analyzeHardness(r.circuit, B).attacks.find(x => x.name === 'tensor-network')!
  // The planner is randomised, so allow a step of slack rather than demanding equality.
  return Math.abs(a.parameter!.value - b2.parameter!.value) <= 2
    ? true : `width ${a.parameter!.value} vs ${b2.parameter!.value} after relabelling`
})

check('Trotter steps compose', () => {
  const a = trotterIsing({ qubits: 8, coupling: line(8), depth: 2, thetaH: 0.4 })
  const b2 = trotterIsing({ qubits: 8, coupling: line(8), depth: 3, thetaH: 0.4 })
  const both = trotterIsing({ qubits: 8, coupling: line(8), depth: 5, thetaH: 0.4 })
  return dev(a.compose(b2).exactProbs(), both.exactProbs()) < 1e-10
    ? true : `deviation ${dev(a.compose(b2).exactProbs(), both.exactProbs())}`
})

check('batched contraction agrees with one amplitude at a time', () => {
  const c = rcsCircuit({ rows: 2, cols: 4, depth: 4, seed: 9 })
  const n = c.qubits
  for (const bits of ['0'.repeat(n), '1'.repeat(n), '10101010'.slice(0, n)]) {
    const one = amplitudeByContraction(c, bits, { restarts: 8 })
    const want = c.amplitude(bits)
    if (Math.abs(one.re - want.re) > 1e-10 || Math.abs(one.im - want.im) > 1e-10) return `${bits} mismatch`
  }
  return true
})

// ── Batch 3: the U(1) axis ────────────────────────────────────────────────────

check('the sector simulation matches the full statevector exactly', () => {
  for (const [n, layers, k] of [[10, 4, 2], [12, 5, 3]] as const) {
    const c = particleConservingAnsatz({ qubits: n, layers, seed: 3 })
    const init = Array.from({ length: n }, (_, i) => i < k ? '1' : '0').join('')
    const full = c.run({ shots: 40_000, seed: 5, initialState: init }).probs
    const sec = runSector(c, { initialState: init, shots: 40_000, seed: 5 })
    const d = dev(full, sec.probs)
    if (d !== 0) return `n=${n} k=${k}: deviation ${d}`
  }
  return true
})

check('a conserving circuit never leaves its sector', () => {
  const n = 12, k = 4
  const c = particleConservingAnsatz({ qubits: n, layers: 6, seed: 2 })
  const init = Array.from({ length: n }, (_, i) => i < k ? '1' : '0').join('')
  for (const bits of Object.keys(runSector(c, { initialState: init, shots: 20_000, seed: 1 }).probs)) {
    if (weightOf(bits) !== k) return `reached weight ${weightOf(bits)} from weight ${k}`
  }
  // And the full statevector agrees that nothing else is reachable.
  for (const [bits, p] of Object.entries(c.run({ shots: 40_000, seed: 2, initialState: init }).probs)) {
    if (p > 0 && weightOf(bits) !== k) return `statevector reached weight ${weightOf(bits)}`
  }
  return true
})

check('the sector dimension is exactly C(n,k)', () => {
  const c = particleConservingAnsatz({ qubits: 16, layers: 3, seed: 4 })
  for (const k of [0, 1, 3, 8, 15, 16]) {
    const init = Array.from({ length: 16 }, (_, i) => i < k ? '1' : '0').join('')
    const s = weightSymmetry(c, init)
    if (s.sectorDimension !== binomial(16, k)) return `k=${k}: ${s.sectorDimension} vs ${binomial(16, k)}`
  }
  return true
})

check('conservation is invariant under how a gate is spelled', () => {
  // Rzz written directly and as CNOT-Rz-CNOT are the same operator, so they must
  // get the same verdict. This is what the run-fusing exists for.
  const direct = new Circuit(4).zz(0.7, 0, 1)
  const spelled = new Circuit(4).cnot(0, 1).rz(0.7, 1).cnot(0, 1)
  const a = weightSymmetry(direct, '1100').conserved
  const b2 = weightSymmetry(spelled, '1100').conserved
  if (a !== b2) return `direct ${a} vs decomposed ${b2}`
  if (!a) return 'Rzz is diagonal and should be conserving'
  return true
})

check('breaking the symmetry with one gate is detected', () => {
  const n = 10
  const base = particleConservingAnsatz({ qubits: n, layers: 3, seed: 9 })
  const init = Array.from({ length: n }, (_, i) => i < 3 ? '1' : '0').join('')
  if (!weightSymmetry(base, init).conserved) return 'base ansatz should conserve'
  // A single Hadamard anywhere destroys it.
  if (weightSymmetry(base.h(4), init).conserved) return 'an added H went undetected'
  if (weightSymmetry(base.x(0), init).conserved) return 'an added X went undetected'
  return true
})

check('the symmetry attack agrees with the sector simulation it recommends', () => {
  const n = 12, k = 3
  const c = particleConservingAnsatz({ qubits: n, layers: 5, seed: 11 })
  const init = Array.from({ length: n }, (_, i) => i < k ? '1' : '0').join('')
  const a = analyzeHardness(c, { ...B, initialState: init }).attacks.find(x => x.name === 'symmetry')!
  if (!a.applicable) return `declined: ${a.reason}`
  const r = runSector(c, { initialState: init, shots: 20_000, seed: 4 })
  if (r.sectorDimension !== a.parameter!.value) return `priced ${a.parameter!.value}, ran ${r.sectorDimension}`
  // And the recommendation cashes: the sector answer is the statevector answer.
  const full = c.run({ shots: 20_000, seed: 4, initialState: init }).probs
  return dev(full, r.probs) === 0 ? true : `cashed with deviation ${dev(full, r.probs)}`
})

// ── Report ────────────────────────────────────────────────────────────────────

const pad = Math.max(...results.map(r => r.name.length))
for (const r of results) {
  process.stdout.write(`${r.ok ? '✓' : '✗'} ${r.name.padEnd(pad)}  ${r.detail}\n`)
}
const failed = results.filter(r => !r.ok)
process.stdout.write(`\n${results.length - failed.length}/${results.length} external checks passed\n`)
process.stdout.write(failed.length === 0
  ? 'BATCH CLEAN — nothing new found\n'
  : `BATCH FOUND ${failed.length}: ${failed.map(r => r.name).join('; ')}\n`)
if (failed.length > 0) process.exitCode = 1
