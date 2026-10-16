/**
 * Does the adversary actually work? Randomised differential testing.
 *
 * Hand-written checks only test what the author thought of, which is why a
 * viable recommendation that returned an answer 0.28 off survived two "clean"
 * batches — nobody had written the one check that would have caught it. This
 * generates circuits at random, asks the adversary what it recommends, then
 * *runs every applicable attack* and compares against the exact statevector.
 *
 * The measurement is a defect rate, not a pass/fail. A tool whose recommendations
 * are wrong one time in a thousand is a different object from one that is never
 * wrong, and until this is run neither claim is supported.
 *
 * Circuits are drawn from the whole gate vocabulary rather than the published
 * families, because the failures so far have all come from combinations nobody
 * chose deliberately.
 *
 * Run with: node --max-old-space-size=8192 benchmark/fuzz-attacks.ts [trials]
 */
import {
  Circuit, amplitudeByContraction, analyzeHardness, pauliPathExpectation,
  runSector, weightSymmetry, type Attack,
} from '@kirkelliott/ket'

const TRIALS = Number(process.argv[2] ?? 300)
const say = (s: string): void => { process.stdout.write(`${s}\n`) }

function prng(seed: number): () => number {
  let s = (seed >>> 0) || 1
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x100000000 }
}

/** A random circuit, deliberately mixing gate families that are rarely combined. */
function randomCircuit(seed: number): { circuit: Circuit; initialState: string } {
  const rand = prng(seed)
  const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)]!
  const n = 4 + Math.floor(rand() * 7)                 // 4..10 qubits
  const depth = 1 + Math.floor(rand() * 14)
  // A style bias, so some circuits land in each attack's home territory rather
  // than every circuit being generic and hard for all of them.
  const style = pick(['clifford', 'diagonal', 'conserving', 'rotations', 'mixed', 'trotter', 'unitary'] as const)
  let c = new Circuit(n)

  // A Trotterised bond structure, which is the one shape the other styles cannot
  // produce: XX and YY each break conservation and their product does not. The
  // defect that motivated this fuzzer lived exactly there, so a generator that
  // cannot emit it proves nothing about that class.
  if (style === 'trotter') {
    for (let d = 0; d < Math.min(depth, 6); d++) {
      for (let q = d % 2; q + 1 < n; q += 2) {
        const th = 0.2 + rand() * 2
        c = c.xx(th, q, q + 1).yy(th, q, q + 1).zz(0.3 + rand() * 2, q, q + 1)
      }
      // Half the time add a transverse field, which must break the symmetry.
      if (rand() < 0.5) for (let q = 0; q < n; q++) c = c.rx(0.3 + rand(), q)
    }
    const kk = Math.floor(rand() * (n + 1))
    return { circuit: c, initialState: Array.from({ length: n }, (_, i) => i < kk ? '1' : '0').join('') }
  }

  for (let d = 0; d < depth; d++) {
    for (let q = 0; q < n; q++) {
      if (rand() < 0.4) continue
      const g = style === 'clifford' ? pick(['h', 'x', 'y', 'z', 's', 'sdg', 'v', 'vi'] as const)
        : style === 'diagonal' || style === 'conserving' ? pick(['z', 's', 't', 'sdg', 'tdg'] as const)
        : pick(['h', 'x', 'y', 'z', 's', 't', 'v', 'rx', 'ry', 'rz', 'u1'] as const)
      c = g === 'rx' ? c.rx(rand() * 6, q) : g === 'ry' ? c.ry(rand() * 6, q)
        : g === 'rz' ? c.rz(rand() * 6, q) : g === 'u1' ? c.u1(rand() * 6, q)
        : (c as unknown as Record<string, (q: number) => Circuit>)[g]!(q)
    }
    for (let q = rand() < 0.5 ? 0 : 1; q + 1 < n; q += 2) {
      if (rand() < 0.3) continue
      const a = q, b = q + 1
      const g = style === 'clifford' ? pick(['cnot', 'cz', 'swap'] as const)
        : style === 'diagonal' ? pick(['cz', 'zz'] as const)
        : style === 'conserving' ? pick(['iswap', 'srswap', 'xy', 'zz', 'swap'] as const)
        : style === 'unitary' ? pick(['u2q', 'iswap', 'cnot'] as const)
        : pick(['cnot', 'cz', 'swap', 'iswap', 'xy', 'zz', 'xx', 'yy', 'crx', 'cry', 'ccx', 'cswap'] as const)
      const third = (b + 1) % n === a ? (b + 2) % n : (b + 1) % n
      c = g === 'cnot' ? c.cnot(a, b) : g === 'cz' ? c.cz(a, b) : g === 'swap' ? c.swap(a, b)
        : g === 'iswap' ? c.iswap(a, b) : g === 'srswap' ? c.srswap(a, b)
        : g === 'xy' ? c.xy(rand() * 6, a, b) : g === 'zz' ? c.zz(rand() * 6, a, b)
        : g === 'xx' ? c.xx(rand() * 6, a, b) : g === 'yy' ? c.yy(rand() * 6, a, b)
        : g === 'crx' ? c.crx(rand() * 6, a, b) : g === 'cry' ? c.cry(rand() * 6, a, b)
        : g === 'cswap' ? c.cswap(third, a, b)
        : g === 'u2q' ? c.unitary(randomUnitary4(rand), a, b)
        : c.ccx(a, b, third)
    }
  }
  const k = Math.floor(rand() * (n + 1))
  const initialState = Array.from({ length: n }, (_, i) => i < k ? '1' : '0').join('')
  return { circuit: c, initialState }
}

/**
 * A Haar-ish random 2-qubit unitary, by Gram-Schmidt on a random complex matrix.
 *
 * `unitary` ops are their own code path in every backend — one of them refused
 * them outright for an entire circuit family — so the fuzzer has to emit them.
 */
function randomUnitary4(rand: () => number): { re: number; im: number }[][] {
  const n = 4
  const cols: { re: number[]; im: number[] }[] = []
  for (let j = 0; j < n; j++) {
    const re = Array.from({ length: n }, () => rand() * 2 - 1)
    const im = Array.from({ length: n }, () => rand() * 2 - 1)
    for (const prev of cols) {
      let dr = 0, di = 0
      for (let i = 0; i < n; i++) { dr += prev.re[i]! * re[i]! + prev.im[i]! * im[i]!; di += prev.re[i]! * im[i]! - prev.im[i]! * re[i]! }
      for (let i = 0; i < n; i++) {
        re[i] = re[i]! - (dr * prev.re[i]! - di * prev.im[i]!)
        im[i] = im[i]! - (dr * prev.im[i]! + di * prev.re[i]!)
      }
    }
    let norm = 0
    for (let i = 0; i < n; i++) norm += re[i]! * re[i]! + im[i]! * im[i]!
    norm = Math.sqrt(norm)
    for (let i = 0; i < n; i++) { re[i] = re[i]! / norm; im[i] = im[i]! / norm }
    cols.push({ re, im })
  }
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => ({ re: cols[j]!.re[i]!, im: cols[j]!.im[i]! })))
}

const deviation = (a: Record<string, number>, b: Record<string, number>): number => {
  let worst = 0
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    worst = Math.max(worst, Math.abs((a[k] ?? 0) - (b[k] ?? 0)))
  }
  return worst
}

const SHOTS = 60_000
/** 5 sigma at this shot count, with room for rounding. */
const SAMPLING = 5 * Math.sqrt(0.25 / SHOTS) + 0.004

/**
 * Run one attack for real and return how far off it was, or null if it has
 * nothing to cash here.
 */
function cash(c: Circuit, initialState: string, attack: Attack): { error: number; tol: number } | null {
  const prep = (): Circuit => {
    let out = new Circuit(c.qubits)
    for (let q = 0; q < c.qubits; q++) if (initialState[q] === '1') out = out.x(q)
    return out.compose(c)
  }
  const withState = prep()
  const exact = withState.exactProbs()
  switch (attack.name) {
    case 'clifford':
      return { error: deviation(exact, withState.runClifford({ shots: SHOTS, seed: 5 }).probs), tol: SAMPLING }
    case 'mps':
      return { error: deviation(exact, withState.runMps({ shots: SHOTS, seed: 5 }).probs), tol: SAMPLING }
    case 'statevector':
      return { error: deviation(exact, withState.run({ shots: SHOTS, seed: 5 }).probs), tol: SAMPLING }
    case 'stabilizer-rank':
      return { error: deviation(exact, withState.runStabilizerRank({ shots: SHOTS, seed: 5 }).probs), tol: SAMPLING }
    case 'symmetry': {
      const r = runSector(c, { initialState, shots: SHOTS, seed: 5 })
      return { error: deviation(exact, r.probs), tol: SAMPLING }
    }
    case 'tensor-network': {
      const bits = '0'.repeat(withState.qubits)
      const want = withState.amplitude(bits)
      const got = amplitudeByContraction(withState, bits, { restarts: 6 })
      return { error: Math.max(Math.abs(got.re - want.re), Math.abs(got.im - want.im)), tol: 1e-9 }
    }
    case 'pauli-path': {
      const n = withState.qubits
      const term = { coeff: 1, ops: `Z${'I'.repeat(n - 1)}` }
      let truth = 0
      for (const [b, p] of Object.entries(exact)) truth += (b[n - 1] === '1' ? -1 : 1) * p
      const got = pauliPathExpectation(withState, [term], { threshold: 1e-13 })
      return { error: Math.abs(got.value - truth), tol: 1e-7 }
    }
    default: return null
  }
}

interface Failure { seed: number; attack: string; error: number; tol: number; detail: string }
const failures: Failure[] = []
let cashed = 0, skipped = 0, errored = 0

for (let seed = 1; seed <= TRIALS; seed++) {
  let c: Circuit, initialState: string
  try { ({ circuit: c, initialState } = randomCircuit(seed)) } catch { errored++; continue }

  let report
  try {
    report = analyzeHardness(c, { memoryBytes: 16e9, seconds: 3600, probeChi: 16, restarts: 4, initialState })
  } catch (e) {
    failures.push({ seed, attack: 'analyzeHardness', error: NaN, tol: 0, detail: `threw: ${e instanceof Error ? e.message : String(e)}` })
    continue
  }

  // Cash *every* applicable attack, not just the winner: a wrong answer from a
  // non-winning attack is still a wrong answer waiting for a different budget.
  for (const attack of report.attacks) {
    if (!attack.applicable) continue
    let got: { error: number; tol: number } | null
    try { got = cash(c, initialState, attack) }
    catch (e) {
      failures.push({ seed, attack: attack.name, error: NaN, tol: 0, detail: `threw: ${e instanceof Error ? e.message : String(e)}` })
      continue
    }
    if (got === null) { skipped++; continue }
    cashed++
    if (!(got.error <= got.tol)) {
      failures.push({ seed, attack: attack.name, error: got.error, tol: got.tol, detail: `${c.qubits}q` })
    }
  }

  // The symmetry claim is checked separately: if it says conserved, no amount of
  // running the circuit should ever leave the sector.
  try {
    const sym = weightSymmetry(c, initialState)
    if (sym.conserved) {
      let out = new Circuit(c.qubits)
      for (let q = 0; q < c.qubits; q++) if (initialState[q] === '1') out = out.x(q)
      const probs = out.compose(c).run({ shots: 20_000, seed: 3 }).probs
      const k = [...initialState].filter(x => x === '1').length
      for (const [bits, p] of Object.entries(probs)) {
        const w = [...bits].filter(x => x === '1').length
        if (p > 1e-9 && w !== k) {
          failures.push({ seed, attack: 'symmetry-claim', error: p, tol: 0, detail: `claimed conserved, reached weight ${w} from ${k}` })
          break
        }
      }
    }
  } catch { /* covered above */ }

  if (seed % 50 === 0) say(`  … ${seed}/${TRIALS} circuits, ${cashed} attacks cashed, ${failures.length} failures`)
}

say('')
say(`${TRIALS} random circuits, ${cashed} attack executions cashed against the exact statevector`)
say(`${skipped} skipped (nothing to cash), ${errored} generation errors`)
if (failures.length === 0) {
  say('NO FAILURES — every applicable attack reproduced the exact answer')
} else {
  say(`${failures.length} FAILURES:`)
  const byAttack = new Map<string, number>()
  for (const f of failures) byAttack.set(f.attack, (byAttack.get(f.attack) ?? 0) + 1)
  for (const [a, n] of [...byAttack].sort((x, y) => y[1] - x[1])) say(`  ${a}: ${n}`)
  for (const f of failures.slice(0, 12)) {
    say(`  seed ${f.seed} ${f.attack}: error ${Number.isNaN(f.error) ? '—' : f.error.toExponential(2)} (tol ${f.tol.toExponential(1)}) ${f.detail}`)
  }
  process.exitCode = 1
}
