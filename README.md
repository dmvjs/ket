# ket

[![Test](https://github.com/dmvjs/ket/actions/workflows/test.yml/badge.svg)](https://github.com/dmvjs/ket/actions/workflows/test.yml)
[![npm](https://img.shields.io/npm/v/@kirkelliott/ket)](https://www.npmjs.com/package/@kirkelliott/ket)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Quantum circuits in TypeScript — simulate locally, then run on real hardware.**
Immutable API, five backends, zero dependencies.

```typescript
import { Circuit } from '@kirkelliott/ket'

const bell = new Circuit(2).h(0).cnot(0, 1)

bell.draw()          // q0: ─H──●─
                     //          │
                     // q1: ─────⊕─

bell.stateAsString() // 0.7071|00⟩ + 0.7071|11⟩
bell.exactProbs()    // { '11': 0.4999999999999999, '00': 0.4999999999999999 }
```

Those are analytic probabilities, computed from the amplitudes rather than sampled
— the last digit is IEEE-754 rounding on 1/√2, not shot noise. Run it a thousand
times and it does not move.

**[Playground](https://dmvjs.com/ket/demo.html)** &nbsp;·&nbsp; **[Live demos](https://dmvjs.com/ket/)** &nbsp;·&nbsp; **[API docs](https://dmvjs.com/ket/docs.html)** &nbsp;·&nbsp; **[Full reference](docs/REFERENCE.md)**

## Install

```bash
npm install @kirkelliott/ket
```

No dependencies, no build step, no Python. Works in Node ≥ 22 and directly in the
browser, as a module or as a plain script tag:

```html
<script type="module">
  import { Circuit } from 'https://unpkg.com/@kirkelliott/ket/dist/ket.js'
</script>
```

```html
<script src="https://unpkg.com/@kirkelliott/ket"></script>
<script>
  const bell = new ket.Circuit(2).h(0).cnot(0, 1)
</script>
```

## Run on real quantum hardware

Design and simulate in TypeScript, then submit the same circuit to an IonQ QPU.
No Python, no SDK, no dependencies — the client uses the global `fetch`.

```typescript
import { Circuit, runIonQ, countsToProbs } from '@kirkelliott/ket'

const bell = new Circuit(2).h(0).cnot(0, 1)

bell.exactProbs()           // { '11': 0.4999999999999999, '00': 0.4999999999999999 }
bell.checkDevice('forte-1') // fail fast, before spending queue time

const { counts } = await runIonQ(bell.toIonQ(), {
  apiKey: process.env.IONQ_API_KEY,
  target: 'qpu.forte-1',   // or 'simulator' — free, no queue
  shots: 1024,
})

countsToProbs(counts, 2, 1024)
// { '00': 0.49, '11': 0.51 }   — from trapped ions
```

Try it against IonQ's free simulator in one command:

```bash
IONQ_API_KEY=... node examples/node/ionq.js
```

Algorithm circuits need one extra step. `toIonQ()` is a serializer and rejects
gates IonQ has no representation for; `toIonQBasis()` expands them. QFT and
modular-exponentiation circuits are built almost entirely from `cu1`, so this is
what makes them submittable:

```bash
IONQ_API_KEY=... node examples/node/ionq-shor.js
```

That runs Shor's algorithm on IonQ's simulator and factors 15. The counting
register returns 0, 2, 4, 6 at 25% each — period 4, giving 15 = 3 × 5.

`runIonQ` submits, polls and fetches in one call. Real jobs outlive the process
that started them, so the lifecycle is also available piecewise — the id is
enough to recover the work later:

```typescript
import { submitIonQ, getIonQJob, getIonQResults, cancelIonQJob } from '@kirkelliott/ket'

const { id } = await submitIonQ(circuit.toIonQBasis().toIonQ(), {
  apiKey, target: 'simulator',
  noise: { model: 'ideal' },     // say it — the server default is not guaranteed noiseless
})

const job = await getIonQJob(id, { apiKey })   // 'ready' means queued, not failed
await getIonQResults(job, { apiKey })          // results live behind results_url, not on the job
await cancelIonQJob(id, { apiKey })            // release the slot
```

The same `Circuit` object simulates in the browser and runs on the QPU from your
server. One representation, one set of types, no serialisation boundary between
two languages — write the frontend and the backend in the same code.

Hardware submission is server-side by necessity: an API key buys paid QPU time,
so it never belongs in a browser bundle. Point `endpoint` at your own route and
the browser half needs no key at all — see
[the proxy pattern](docs/REFERENCE.md#calling-from-a-browser-use-a-proxy).

## Factor a number with Shor's algorithm

Not an oracle mock-up — the full Beauregard circuit, modular exponentiation
decomposed into primitive gates, simulated exactly:

```typescript
import { shorBeauregard } from '@kirkelliott/ket'

const r = shorBeauregard(15n, { a: 7n })

r.factors  // [5n, 3n]
r.period   // 4n        — 7⁴ ≡ 1 (mod 15), recovered by phase estimation
r.method   // 'quantum' — the circuit really ran, not a gcd shortcut
r.qubits   // 19
```

Nineteen qubits, 0.4 seconds. N=21 takes 3.0s at 23 qubits; N=33, 22.4s at 27.
This does not reach cryptographic sizes — no classical simulator does. What you
get is the real circuit, exactly simulated, at sizes you can inspect by hand.

Two worked examples carry it through end to end, asserting at every step and
refusing any run a `gcd` shortcut resolved:

```bash
node examples/node/rsa-shor.js    # RSA private exponent, from the public key alone
node examples/node/dlog-shor.js   # discrete logarithm, the problem under ECDSA
```

Both close by reporting what the same circuit would do on hardware: 31,260
two-qubit gates for the RSA instance, a 10⁻⁶⁹ chance of an error-free run at
current trapped-ion error rates. That gap is the honest headline — the algorithm
is correct, and today's hardware cannot run it.

## Beyond the statevector

A statevector costs 2ⁿ and stops near 24 qubits. ket carries four other
representations and routes to the cheapest exact one automatically:

```typescript
circuit.simulate({ shots: 1024 })   // picks a backend; d.backend says which
```

| Backend | Cost | Reaches |
|---|---|---|
| Statevector | sparse → dense, automatic | ~20 qubits exactly |
| MPS / tensor network | O(n·χ²) | 127-qubit GHZ, 1024 shots, 10ms |
| Density matrix | O(4ⁿ) sparse | mixed states and noise, n≈12 |
| Clifford stabilizer | O(n²) | 1,024 qubits in milliseconds |
| Stabilizer rank | O(2^0.228ᵗ·n²/8) | 100 qubits with 50 T gates, 5.6s |

Stabilizer rank is the unusual one: its cost is exponential in the *non-Clifford*
gate count, not the width, so it goes where a statevector cannot. See
[the reference](docs/REFERENCE.md#simulation-backends) for bond dimensions, term
budgets, and the error accounting on approximate runs.

## Why you can build on it

- **Typed, and the types are the source.** Strict TypeScript, not JS with a `.d.ts`
  bolted on. Out-of-range qubit indices throw at gate-construction time, not 200
  gates later.
- **Immutable.** Every gate method returns a new `Circuit`.
- **BigInt state indices.** No 32-bit overflow at qubit 31.
- **2,165 tests.** Analytic correctness against known amplitudes — gate
  invertibility, backend cross-agreement, and full round-trips for every
  import/export format. Stabilizer backends are checked against the statevector
  kernel *including global phase*.
- **Zero dependencies.** 196 KB minified, total.

## Documentation

- **[Live demos](https://dmvjs.com/ket/)** — QAOA, Grover across hardware, 1,024-qubit GHZ, VQE
- **[Playground](https://dmvjs.com/ket/demo.html)** — run circuits in the browser, nothing to install
- **[API docs](https://dmvjs.com/ket/docs.html)** — every export
- **[Full reference](docs/REFERENCE.md)** — all gates, 14 import/export formats, noise
  models, device targeting, visualization, QEC
- **[Guide](ket-guide/)** — a 13-chapter Quarto book, from bits to backends, with
  runnable blocks. `quarto render` after `npm run build`
- **[Examples](examples/)** — Node scripts, a browser page, a Jupyter notebook, and
  the SVGs the docs use. `rsa-shor.js` and `dlog-shor.js` are the longest, and
  read as lab reports rather than snippets

## License

MIT
