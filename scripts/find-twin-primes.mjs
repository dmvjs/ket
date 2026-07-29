// Find twin prime semiprimes for Shor's demo targets.
// Run: node scripts/find-twin-primes.mjs <bits>
// Example: node scripts/find-twin-primes.mjs 1024
// Outputs a SHOR_TARGETS entry with BigInt literals.

const BITS = parseInt(process.argv[2] ?? '256')
const WITNESSES = [2n,3n,5n,7n,11n,13n,17n,19n,23n,29n,31n,37n,41n,43n,47n,53n,59n,61n,67n,71n]
const SMALL_PRIMES = []
{
  const sieve = new Uint8Array(100000).fill(1)
  sieve[0] = sieve[1] = 0
  for (let i = 2; i < 100000; i++) if (sieve[i]) { SMALL_PRIMES.push(BigInt(i)); for (let j = i*i; j < 100000; j += i) sieve[j] = 0 }
}

function modpow(b, e, m) {
  let r = 1n; b = b % m
  while (e > 0n) { if (e & 1n) r = r * b % m; e >>= 1n; b = b * b % m }
  return r
}

function millerRabin(n, rounds = 20) {
  if (n < 2n) return false
  if (n < 4n) return true
  if ((n & 1n) === 0n) return false
  let d = n - 1n, r = 0n
  while ((d & 1n) === 0n) { d >>= 1n; r++ }
  outer: for (const a of WITNESSES.slice(0, rounds)) {
    if (a >= n) continue
    let x = modpow(a, d, n)
    if (x === 1n || x === n - 1n) continue
    for (let i = 0n; i < r - 1n; i++) { x = x * x % n; if (x === n - 1n) continue outer }
    return false
  }
  return true
}

function sievePass(p) {
  for (const sp of SMALL_PRIMES) {
    if (sp * sp > p) break
    if (p % sp === 0n) return p === sp
  }
  return true
}

function isPrime(n) { return sievePass(n) && millerRabin(n) }

const start = Date.now()
let p = (1n << BigInt(Math.ceil(BITS / 2) - 1))
if ((p & 1n) === 0n) p++

let checked = 0, tested = 0
process.stderr.write(`Searching for ${BITS}-bit twin prime semiprime (p near 2^${Math.ceil(BITS/2)-1})...\n`)

while (true) {
  checked++
  // Quick sieve pass on p and p+2 before full Miller-Rabin
  if (sievePass(p) && sievePass(p + 2n)) {
    tested++
    if (millerRabin(p) && millerRabin(p + 2n)) {
      const N = p * (p + 2n)
      const bits = N.toString(2).length
      if (bits >= BITS) {
        const elapsed = ((Date.now() - start) / 1000).toFixed(1)
        const a = p + 1n
        process.stderr.write(`\nFound after ${checked} candidates, ${tested} full tests, ${elapsed}s\n`)
        console.log(`  { N: ${N}n, a: ${a}n, aSq: 1n, n: ${bits}, t: 1, sparse: true, label: "${N.toLocaleString()} [${bits}-bit]" },`)
        console.log(`  // p = ${p} (verify: isPrime(p) && isPrime(p+2))`)
        process.exit(0)
      }
    }
  }
  p += 2n
  if (checked % 50000 === 0) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(0)
    process.stderr.write(`  ${checked} candidates, ${tested} full tests, ${elapsed}s elapsed\n`)
  }
}
