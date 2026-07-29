# ket

[![Test](https://github.com/dmvjs/ket/actions/workflows/test.yml/badge.svg)](https://github.com/dmvjs/ket/actions/workflows/test.yml)
[![npm](https://img.shields.io/npm/v/@kirkelliott/ket)](https://www.npmjs.com/package/@kirkelliott/ket)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Quantum circuits in TypeScript.** Immutable API, four backends, zero dependencies.

```typescript
import { Circuit } from '@kirkelliott/ket'

const bell = new Circuit(2).h(0).cnot(0, 1)

bell.draw()          // q0: ─H──●─
                     //          │
                     // q1: ─────⊕─

bell.stateAsString() // 0.7071|00⟩ + 0.7071|11⟩
bell.exactProbs()    // { '00': 0.5, '11': 0.5 }
```

**[Playground](https://dmvjs.com/ket/demo.html)** &nbsp;·&nbsp; **[Guide](https://dmvjs.com/ket/)** &nbsp;·&nbsp; **[API docs](https://dmvjs.com/ket/docs.html)** &nbsp;·&nbsp; **[Full reference](docs/REFERENCE.md)**

## Install

```bash
npm install @kirkelliott/ket
```

No dependencies, no build step, no Python. Works in Node ≥ 22 and directly in the browser:

```html
<script type="module">
  import { Circuit } from 'https://unpkg.com/@kirkelliott/ket/dist/ket.js'
</script>
```

## Factor a number with Shor's algorithm

Not an oracle mock-up — the full Beauregard circuit, with modular exponentiation
decomposed into primitive gates, simulated exactly:

```typescript
import { shorBeauregard } from '@kirkelliott/ket'

const r = shorBeauregard(15n, { a: 7n })

r.factors  // [5n, 3n]
r.period   // 4n        — 7⁴ ≡ 1 (mod 15), recovered by phase estimation
r.method   // 'quantum' — the circuit really ran
r.qubits   // 19
```

Nineteen qubits, 0.4 seconds, six lines.

Shor's tries classical shortcuts first, and at these sizes they hit often — an even
N, or a base sharing a factor with N, is resolved by `gcd` without building a circuit.
`method` tells you which happened, so a demo can prove the quantum path ran:

```typescript
shorBeauregard(15n, { a: 3n }).method  // 'classical-gcd' — no circuit
shorBeauregard(15n, { a: 7n }).method  // 'quantum'
```

| N  | qubits | time  |
|----|--------|-------|
| 15 | 19     | 0.4s  |
| 21 | 23     | 3.0s  |
| 33 | 27     | 22.4s |
| 35 | 27     | 29.0s |

This does not reach cryptographic sizes — no classical simulator does. What you get
is the real circuit, exactly simulated, at sizes you can actually inspect.

## Pick a backend, or let ket pick

```typescript
circuit.simulate({ shots: 1024 })   // routes to the cheapest exact backend
```

| Backend | Memory | Best for |
|---|---|---|
| Statevector | sparse → dense, automatic | Exact simulation to ~20 qubits |
| MPS / tensor network | O(n·χ²), χ grows on demand | Low-entanglement circuits, 50+ qubits |
| Density matrix | sparse → dense, automatic | Mixed states and noise |
| Clifford stabilizer | O(n²) | Clifford circuits, QEC thresholds |

The statevector backend starts sparse and promotes itself to a contiguous
`Float64Array` once a state is more than ⅛ full, so sparse circuits stay cheap and
dense ones stop paying for a hash map. MPS bond dimension is exact by default —
`maxBond` is an initial allocation, not a cap, and χ grows as the circuit demands.

GHZ-50 runs in milliseconds on MPS at χ=2. A 127-qubit GHZ samples 1024 shots in 186ms.

## Why you can build on it

**It is typed, and the types are the source.** Written in strict TypeScript, not JavaScript
with a `.d.ts` bolted on afterward. Every qubit index is bounds-checked at gate-construction
time — an out-of-range index throws `RangeError` immediately instead of silently corrupting
state 200 gates later.

**Immutable.** Every gate method returns a new `Circuit`. Compose, branch, and reuse
without defensive copying.

**BigInt state indices.** No 32-bit overflow at qubit 31, the failure mode that silently
corrupts integer-indexed simulators.

**1,768 tests.** Analytic correctness against known amplitudes — not "doesn't crash."
Gate invertibility (U†U = I), backend cross-agreement, BigInt correctness at indices
30/31/40, and full round-trips for every supported import/export format. The sparse and
dense statevector kernels are differentially tested against each other gate by gate,
over every qubit ordering, so promotion can never change a result.

**Zero dependencies.** 147 KB minified, total. Nothing to audit but ket.

## Performance

ket matches its representation to the circuit instead of committing to one, so
the same API is efficient across shapes that usually need different tools.

A statevector starts as a sparse map and promotes itself to a flat
`Float64Array` once it is more than ⅛ full. A GHZ state holds two non-zero
amplitudes at any width, so it stays sparse and never touches the other million
slots; a depth-4 random circuit fills every amplitude in its first layer, so it
moves to the dense kernel once and runs the rest with no allocation at all. The
density matrix does the same at 1/32 fill, and MPS bond dimension grows on
demand rather than being capped up front.

Measured on Node 24 / Apple silicon, best of 5:

| Circuit | Representation | Time |
|---|---|---|
| GHZ-20, statevector | sparse | 5µs |
| QFT-16, statevector | dense | 10.5ms |
| random-16 depth 4, statevector | dense | 15.7ms |
| GHZ-50, MPS χ=2 | tensor network | milliseconds |
| GHZ-127, MPS, 1024 shots | tensor network | 186ms |
| 12-qubit noisy run, 1024 shots | dense | 0.69s |

None of this needs a flag — the thresholds are defaults, adjustable per call via
the `dense` option when you want to trade memory against speed.

## Documentation

- **[Guide](https://dmvjs.com/ket/)** — an interactive book, from bits to Shor's
- **[API docs](https://dmvjs.com/ket/docs.html)** — every export
- **[Full reference](docs/REFERENCE.md)** — all gates, 14 import/export formats, noise
  models, device targeting, visualization, QEC

## License

MIT
