/**
 * Recovery of an RSA private exponent by Shor factorisation.
 *
 * Run:  node examples/node/rsa-shor.js
 *       RSA_P=3 RSA_Q=11 RSA_E=3 RSA_M=5 node examples/node/rsa-shor.js
 *
 * An RSA keypair is constructed from two small primes and used to encrypt a
 * plaintext. The modulus is then factored by quantum phase estimation over
 * modular exponentiation, using only the public parameters, and the private
 * exponent is reconstructed from the recovered factors. Correctness is asserted
 * at each stage; the program exits non-zero on any failed check.
 *
 * The modular exponentiation is the Beauregard construction, decomposed into
 * Toffoli and controlled-phase gates and simulated exactly. No oracle is
 * abstracted away, and no classical shortcut is accepted: runs terminating via
 * gcd or an even modulus are rejected and the next base is tried.
 *
 * Parameter sizes here are chosen for tractable simulation. Cost scales with the
 * bit length of N; factoring a 2048-bit modulus is estimated to require on the
 * order of 10^3 logical qubits, with error correction imposing a further
 * multiplicative overhead in physical qubits.
 */
import { shorBeauregard, shorCircuit, modPow, modInverse, gcd, DEVICES } from '../../dist/ket.js'

const P = BigInt(process.env.RSA_P ?? 3)
const Q = BigInt(process.env.RSA_Q ?? 11)
const E = BigInt(process.env.RSA_E ?? 3)
const M = BigInt(process.env.RSA_M ?? 5)
// Noisy trajectories re-simulate the whole circuit per shot, which costs
// minutes. The analytic estimate below is exact and instant, so sampling is
// opt-in and only confirms it.
const NOISY_SHOTS = Number(process.env.RSA_NOISY_SHOTS ?? 0)
const NOISE_DEVICE = process.env.RSA_NOISE ?? 'forte-1'

/** Abort with a non-zero status and a stated reason. */
const assert = (cond, reason) => {
  if (!cond) { console.error(`\nassertion failed: ${reason}`); process.exit(1) }
}
const isPrime = (v) => {
  if (v < 2n) return false
  for (let f = 2n; f * f <= v; f++) if (v % f === 0n) return false
  return true
}

console.log('Recovery of an RSA private exponent by Shor factorisation')
console.log('========================================================')

// ── 1. Key generation ─────────────────────────────────────────────────────────
assert(isPrime(P) && isPrime(Q), `p=${P} and q=${Q} must be prime`)
assert(P !== Q, 'p and q must be distinct')
const N = P * Q
const phi = (P - 1n) * (Q - 1n)
assert(gcd(E, phi) === 1n, `e=${E} must be coprime to phi(N)=${phi}`)
assert(M < N, `plaintext ${M} must be less than N=${N}`)
const d = modInverse(E, phi)

console.log('\n1. Key generation')
console.log(`   p = ${P}, q = ${Q}, N = ${N}, phi(N) = ${phi}`)
console.log(`   e = ${E}, d = e^-1 mod phi(N) = ${d}`)
if (d === E) {
  console.log(`   note: every unit of Z_${phi} is self-inverse at this size, so d = e.`)
  console.log('         set RSA_P=3 RSA_Q=11 RSA_E=3 for an instance where d != e.')
}

// ── 2. Encryption ─────────────────────────────────────────────────────────────
const c = modPow(M, E, N)
assert(modPow(c, d, N) === M, 'keypair fails its own encrypt/decrypt round trip')
console.log('\n2. Encryption')
console.log(`   m = ${M}, c = m^e mod N = ${c}`)

// ── 3. Adversary input ────────────────────────────────────────────────────────
console.log('\n3. Adversary input')
console.log(`   (N, e, c) = (${N}, ${E}, ${c}). The factorisation of N is not supplied.`)

// ── 4. Period finding ─────────────────────────────────────────────────────────
// Bases sharing a factor with N are excluded: gcd would resolve them classically.
const bases = []
for (let a = 2n; a < N - 1n; a++) if (gcd(a, N) === 1n) bases.push(a)

console.log('\n4. Period finding')
console.log(`   candidate bases coprime to N: ${bases.length}`)
// Bases are tried in order and every outcome is logged, so the number of
// attempts needed is visible rather than hidden behind a successful run.
const t0 = performance.now()
let res, tried = 0
for (const a of bases) {
  tried++
  const attempt = performance.now()
  const r = shorBeauregard(N, { a, shots: 64, seed: 7 })
  const dt = ((performance.now() - attempt) / 1000).toFixed(1)
  if (r.method === 'quantum' && r.factors) {
    console.log(`   a = ${String(a).padStart(3)}  r = ${String(r.period).padStart(3)}  ${dt.padStart(5)} s  quantum`)
    res = r
    break
  }
  console.log(`   a = ${String(a).padStart(3)}  ${''.padStart(9)}${dt.padStart(5)} s  ${r.failure ?? r.method ?? 'no period'}`)
}
const secs = (performance.now() - t0) / 1000
assert(res !== undefined, `no base produced a quantum factorisation in ${tried} attempts`)
assert(modPow(res.a, res.period, N) === 1n, `a^r != 1 mod N for a=${res.a}, r=${res.period}`)

console.log(`   succeeded on attempt ${tried} of ${bases.length}: a = ${res.a}, r = ${res.period}`)
console.log(`   a^r mod N = ${modPow(res.a, res.period, N)}`)
console.log(`   qubits = ${res.qubits}, total wall time = ${secs.toFixed(1)} s, method = ${res.method}`)

// ── 5. Factorisation ──────────────────────────────────────────────────────────
const [f1, f2] = res.factors
assert(f1 * f2 === N, `factors ${f1}, ${f2} do not multiply to ${N}`)
console.log('\n5. Factorisation')
console.log(`   N = ${f1} x ${f2} (product verified)`)

// ── 6. Key recovery ───────────────────────────────────────────────────────────
const phiR = (f1 - 1n) * (f2 - 1n)
const dR = modInverse(E, phiR)
assert(dR === d, `recovered d=${dR} differs from generated d=${d}`)
console.log('\n6. Key recovery')
console.log(`   phi(N) = (p-1)(q-1) = ${phiR}, d = e^-1 mod phi(N) = ${dR}`)

// ── 7. Decryption ─────────────────────────────────────────────────────────────
const m2 = modPow(c, dR, N)
assert(m2 === M, `decryption gave ${m2}, expected ${M}`)
console.log('\n7. Decryption')
console.log(`   c^d mod N = ${m2}`)

console.log('\nResult')
console.log('   The private exponent was reconstructed from public parameters alone.')
console.log(`   ${res.qubits} qubits, ${N.toString(2).length}-bit modulus, ${secs.toFixed(1)} s. All checks passed.`)

// ── 8. Hardware feasibility ───────────────────────────────────────────────────
// Sections 4-7 assume noiseless evolution. Reconstructing the phase-estimation
// circuit for the successful base allows the same question to be asked of real
// devices: not "is the algorithm correct" but "could a machine run it".
// The same construction shorBeauregard ran above, obtained from the library
// rather than rebuilt here, so this cannot drift from what was actually executed.
const qpe = shorCircuit(N, res.a)
const precision = qpe.qubits - 2 * Math.ceil(Math.log2(Number(N))) - 2

const { oneQubit: oneQ, twoQubit: twoQ } = qpe.gateCounts()

console.log('\n8. Hardware feasibility')
console.log(`   circuit for a = ${res.a}: ${qpe.qubits} qubits, depth ${qpe.depth()}`)
console.log(`   gate counts: ${oneQ} single-qubit, ${twoQ} two-qubit-equivalent`)
// IonQ systems currently accepting jobs. Per-gate error is what decides this,
// so the figures below stand in for any trapped-ion device of this generation.
console.log('   device                p1          p2          P(error-free run)')
for (const [name, dv] of Object.entries(DEVICES)) {
  if (dv.vendor !== 'IonQ' || dv.status !== 'available') continue
  const pOk = (1 - dv.noise.p1) ** oneQ * (1 - dv.noise.p2) ** twoQ
  console.log(`   ${name.padEnd(20)}  ${dv.noise.p1.toExponential(1).padEnd(10)}  ${dv.noise.p2.toExponential(1).padEnd(10)}  ${pOk.toExponential(2)}`)
}
console.log(`   the circuit uses ${qpe.qubits} qubits; several available devices exceed that.`)
console.log('   the binding constraint is gate count against per-gate error, not width.')
if (NOISY_SHOTS === 0) {
  console.log(`   set RSA_NOISY_SHOTS=16 to confirm this empirically (minutes).`)
}

if (NOISY_SHOTS > 0) {
  // Signal metric: the counting register should concentrate on multiples of
  // 2^precision / r. Measure how much weight survives there under noise.
  const step = 2 ** precision / Number(res.period)
  const onPeak = (dist) => {
    let w = 0
    for (const [bits, prob] of Object.entries(dist.probs)) {
      let v = 0
      for (let k = 0; k < precision; k++) if (bits[k] === '1') v |= 1 << k
      const nearest = Math.round(v / step) * step
      if (Math.abs(v - nearest) < 1) w += prob
    }
    return w
  }
  const ideal = onPeak(qpe.run({ shots: 256, seed: 5 }))
  const t1 = performance.now()
  const noisy = onPeak(qpe.run({ shots: NOISY_SHOTS, seed: 5, noise: NOISE_DEVICE }))
  const uniform = Number(res.period) / 2 ** precision * 100
  console.log(`\n   empirical, ${NOISY_SHOTS} shots under ${NOISE_DEVICE} depolarizing noise ` +
              `(${((performance.now() - t1) / 1000).toFixed(0)} s):`)
  console.log(`     weight on phase-estimation peaks: ${(noisy * 100).toFixed(1)} %`)
  console.log(`     noiseless: ${(ideal * 100).toFixed(1)} %    uniform guessing: ${uniform.toFixed(2)} %`)
  console.log(`   ${noisy <= ideal / 4 ? 'the period signal does not survive.' : 'residual signal survives at this sample size.'}`)
}
