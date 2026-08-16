# ket — complete reference

Every gate, backend, format, and option. For the short version, see the
[README](../README.md).

**[Playground](https://dmvjs.com/ket/demo.html)** &nbsp;·&nbsp; **[API docs](https://dmvjs.com/ket/docs.html)** &nbsp;·&nbsp; **[Live demos](https://dmvjs.com/ket/)** &nbsp;·&nbsp; **[npm](https://www.npmjs.com/package/@kirkelliott/ket)**

## Bitstring convention

All bitstrings in ket use **q0 leftmost** (standard convention, matching Qiskit, Cirq, and textbooks):

- `'10'` → q0=1, q1=0
- `amplitude('10')`, `exactProbs()['10']`, `Distribution.probs['10']` all use q0 leftmost
- `initialState: '10'` starts the circuit with q0=1, q1=0

This matches the convention used by every major quantum computing library and paper. q0 is the first (leftmost) character, just as the first qubit in a ket |q0 q1 q2⟩ is written leftmost.

## Why ket

- **Immutable by design** — every gate method returns a new `Circuit`. Safe to compose, branch, and reuse.
- **TypeScript-strict, zero runtime dependencies** — not a JavaScript library with bolted-on types.
- **BigInt state indices** — handles 30+ qubits without 32-bit integer overflow.
- **Bounds-checked** — every qubit index is validated at gate-construction time; out-of-range indices throw `RangeError` immediately rather than silently corrupting state.
- **Five simulation backends** — statevector, MPS/tensor network, exact density matrix, Clifford stabilizer, and stabilizer rank in one library.
- **WebGPU browser simulation** — the interactive guide (source in [`ket-guide/`](../ket-guide/), not yet published) runs statevector QPE on GPU compute shaders, holding the state in VRAM rather than the JS heap and so bypassing the tab memory limit. It reaches 29-bit state spaces, but only on deliberately favourable targets: N = p(p+2) for twin primes p, with base a = p+1 chosen so that a² ≡ 1 (mod N) and the period is always r = 2, which lets a single counting qubit (t = 1) suffice. The circuit itself is not hand-compiled — every amplitude is computed — but these are not arbitrary semiprimes. See [`ket-guide/book/08-shor.qmd`](../ket-guide/book/08-shor.qmd) for the full disclosure.
- **14 import/export formats** — more than any comparable JavaScript quantum library.
- **Algorithm library built-in** — QFT, Grover's search, QPE, VQE, Trotter simulation, QAOA, gradient (parameter shift rule), minimize, standard ansatz circuits, and Pauli operator algebra ship with the core.

## Install

```bash
npm install @kirkelliott/ket
```

Or load directly in a browser, as a module:

```html
<script type="module">
  import { Circuit } from 'https://unpkg.com/@kirkelliott/ket/dist/ket.js'

  const bell = new Circuit(2).h(0).cnot(0, 1)
  console.log(bell.stateAsString())  // 0.7071|00⟩ + 0.7071|11⟩
</script>
```

…or as a plain script tag, which needs no module setup and works in any page:

```html
<script src="https://unpkg.com/@kirkelliott/ket"></script>
<script>
  const bell = new ket.Circuit(2).h(0).cnot(0, 1)
</script>
```

### Which bundle

| File | Size | Format | For |
|---|---|---|---|
| `dist/ket.js` | 468 KB | ESM | bundlers that tree-shake and minify |
| `dist/ket.min.js` | 215 KB | ESM | `import` from a CDN |
| `dist/ket.global.js` | 216 KB | IIFE, `ket` global | `<script src>` — what `unpkg`/`jsdelivr` serve |
| `dist/compat.js` | 69 B | ESM | `@kirkelliott/ket/compat`, a re-export of `ket.js` |

The global build is a second format, not a second copy: `@kirkelliott/ket/compat`
re-exports the main bundle rather than bundling its own, so a `Circuit` from one
entry point is the same class as a `Circuit` from the other. No external
dependencies in any of them.

Requires Node.js ≥ 22 for server-side use. Worker-backed parallelism
(`workers: N`) additionally needs Node ≥ 22.3; below that it falls back to the
single-threaded path, as it does in a browser.

## Quick start

### Bell state — draw and run

```typescript
import { Circuit } from '@kirkelliott/ket'

const bell = new Circuit(2).h(0).cnot(0, 1)

console.log(bell.draw())
// q0: ─H──●─
//          │
// q1: ─────⊕─

console.log(bell.stateAsString())
// 0.7071|00⟩ + 0.7071|11⟩

console.log(bell.exactProbs())
// { '11': 0.4999999999999999, '00': 0.4999999999999999 }
// Analytic, not sampled: the final digit is IEEE-754 rounding on 1/√2, not shot
// noise. "Exact" here means free of sampling variance, not free of float error.

// Add measurement for shot-based sampling
const result = bell
  .creg('out', 2)
  .measure(0, 'out', 0)
  .measure(1, 'out', 1)
  .run({ shots: 1000, seed: 42 })
// result.probs → { '00': ~0.5, '11': ~0.5 }
```

### Clifford simulation

```typescript
import { Circuit } from '@kirkelliott/ket'

// runClifford accepts only Clifford gates: H, S, S†, X, Y, Z, CNOT, CZ, CY, SWAP
let ghz = new Circuit(5).h(0)
for (let i = 0; i < 4; i++) ghz = ghz.cnot(i, i + 1)
ghz = ghz.creg('out', 5)
for (let i = 0; i < 5; i++) ghz = ghz.measure(i, 'out', i)

const result = ghz.runClifford({ shots: 1024, seed: 42 })
// result.probs → { '00000': ~0.5, '11111': ~0.5 }

// Add noise — same interface as statevector/density matrix
ghz.runClifford({ shots: 10000, noise: 'forte-1' })
ghz.runClifford({ shots: 10000, noise: { p1: 0.001, p2: 0.005, pMeas: 0.004 } })
```

Non-Clifford gates throw at runtime:

```typescript
new Circuit(2).t(0).runClifford()
// TypeError: runClifford: gate 't' is not a Clifford gate
```

### Noise and density matrix

```typescript
import { Circuit } from '@kirkelliott/ket'

const circuit = new Circuit(2).h(0).cnot(0, 1)

// Run with a named device noise profile
const dm = circuit.dm({ noise: 'forte-1' })

console.log(dm.purity())     // < 1 under depolarizing noise
console.log(dm.entropy())    // von Neumann entropy in bits
console.log(dm.blochAngles(0))  // { theta, phi } for qubit 0
console.log(dm.probabilities()) // { '00': ..., '01': ..., ... }
```

## Simulation backends

### Auto-routing: `circuit.simulate()`

`simulate()` picks the cheapest exact backend automatically — zero cognitive overhead:

```typescript
const d = circuit.simulate({ shots: 1024, seed: 42 })
d.backend         // 'clifford' | 'statevector' | 'mps'
d.peakChi         // peak bond dimension χ used (MPS only)
d.representation  // 'sparse' | 'dense' — statevector only, undefined elsewhere
```

`simulate()` also forwards backend tuning: `dense` to the statevector path (see
[Tuning sparse → dense promotion](#tuning-sparse--dense-promotion)), and
`maxBond` / `truncErr` / `maxChi` to MPS. Either `truncErr` or `maxChi` will hold
χ down, which can keep a circuit on MPS that would otherwise fall back — at the
cost of making the simulation approximate.

### Bounding MPS bond dimension

Three MPS options sound similar and are not:

| Option | Default | Effect |
|---|---|---|
| `maxBond` | 64 | **Initial allocation only.** χ grows past it on demand, so this never changes how large χ becomes — raising it only avoids reallocation on circuits known to be highly entangled. |
| `truncErr` | 0 | Relative singular-value cutoff. Bounds the *error*; the resulting χ is whatever that implies. |
| `maxChi` | unbounded | **Hard ceiling on χ.** Bounds the *memory*, deterministically. The SVD keeps at most this many Schmidt values per bond and sets `Distribution.truncated`. |

`maxBond: 256` is a common mistake — it reads like a cap but is not one. Use
`maxChi` when the requirement is "do not exceed this much memory", and `truncErr`
when it is "do not exceed this much error". On a 14-qubit brickwork whose natural
χ is 46:

```typescript
k.runMps({ shots: 500 })                 // peakChi 46, truncated false — exact
k.runMps({ shots: 500, maxBond: 256 })   // peakChi 46, truncated false — unchanged
k.runMps({ shots: 500, maxChi: 16 })     // peakChi 16, truncated true
k.runMps({ shots: 500, maxChi: 4 })      // peakChi  4, truncated true
```

Both remain exact by default: without `truncErr` or `maxChi`, χ grows to whatever
the circuit needs.

Setting `maxChi` also switches off the entanglement fallback in `simulate()`. The
router abandons MPS when χ outgrows a budget, but a ceiling holds χ under that
budget by construction, so the check never fires — a bounded-memory MPS run is
what was asked for, and quietly swapping in a dense statevector would contradict
it. Leave `maxChi` unset if you want the router to choose.

`Distribution.representation` reports which statevector representation the run
finished on, and `DensityMatrix.representation` does the same for ρ. Both exist
so the effect of tuning `dense` is observable rather than inferred from timings:

```typescript
uniform12.run({ shots: 100 }).representation                        // 'dense'
uniform12.run({ shots: 100, dense: { maxQubits: 0 } }).representation // 'sparse'
ghz20.run({ shots: 100 }).representation                            // 'sparse' — never fills
ghz20.runClifford({ shots: 100 }).representation                    // undefined — n/a
```

Routing logic (in priority order):

| Condition | Backend chosen |
|---|---|
| All gates are Clifford (H, X, Y, Z, S, S†, CNOT, CX, CY, CZ, SWAP) | `clifford` |
| n ≤ `statevectorLimit` (default 20) | `statevector` |
| Larger, and entanglement stays bounded | `mps` |
| Larger, but entanglement outgrows MPS | `statevector` |
| Larger than the dense ceiling, or noisy / mid-circuit | `mps` |

The last three rows are decided by measurement, not by inspecting the gate list.
MPS is only the cheaper backend while bond dimension stays low, and whether it
does is a property of the circuit's entanglement rather than of its gates. So
`simulate()` runs the MPS forward and watches χ, abandoning it for a dense
statevector once χ passes the crossover at 2^(n/3) — the point where the MPS
bond update, ~O(χ³) per gate, costs more than an O(2ⁿ) statevector sweep.

Abandoning is cheap: χ never exceeds the budget before the check fires, so the
discarded work is a fraction of either alternative. On a 22-qubit brickwork of
generic two-qubit rotations, `simulate()` returns in 7.6s where forcing
`runMps()` takes 108s and reaches χ=500 — **14× faster for the same answer**. A
GHZ chain at any width still stays on MPS at χ=2.

The probe covers circuits both backends can build once and sample, which includes
terminal measurements. Noisy circuits and genuine mid-circuit feedback re-simulate
per shot and go straight to MPS, and circuits past the dense ceiling
(`dense.maxQubits`, 24 by default) skip it too — there is no statevector to fall
back to.

```typescript
ghz(50).simulate({ shots: 512 }).backend   // 'clifford' — GHZ is Clifford-only
new Circuit(5).t(0).cx(0,1).simulate().backend  // 'statevector' — T gate, n≤20
new Circuit(30).t(0).cx(0,1).simulate().backend  // 'mps' — T gate, n>20
new Circuit(30).h(0).measure(0,'c',0).simulate().backend  // 'clifford' — Clifford path handles mid-circuit

// Override threshold
circuit.simulate({ statevectorLimit: 30 })  // use statevector up to n=30
```

### Manual backend selection

| Backend | Method | Memory | Best for |
|---|---|---|---|
| Statevector | `circuit.run()` / `circuit.statevector()` | sparse → dense, automatic | Exact simulation, practical up to ~20 qubits |
| MPS / tensor network | `circuit.runMps({ shots, maxBond? })` | O(n·χ²), adaptive χ | Low-entanglement circuits, 50+ qubits |
| Exact density matrix | `circuit.dm({ noise? })` | sparse → dense, automatic | Mixed-state and noisy simulation |
| Clifford stabilizer | `circuit.runClifford({ shots, noise? })` | O(n²) | Clifford-only circuits, QEC threshold curves |
| Stabilizer rank | `circuit.runStabilizerRank({ shots, targetError? })` | O(2^0.228t · n²/8) | Clifford+T at 100+ qubits |

All five backends populate `Distribution.backend`. The MPS backend also sets `Distribution.peakChi` — the actual peak bond dimension used (not the allocation size), useful for profiling circuit entanglement:

```typescript
const d = myCircuit.runMps({ shots: 1024 })
console.log(`peak χ = ${d.peakChi}`)  // 2 for GHZ, larger for entangled circuits
```

MPS mid-circuit measurement projects the measured site and then restores Vidal canonical form. That second step is not optional: the bond lambdas either side of a projected site describe the pre-measurement Schmidt spectrum, and both the measurement marginal and sampling weight by them, so skipping it leaves the first measured qubit correct and biases every one after it. Re-canonicalisation is O(n·χ³) against O(χ²) for the projection, which is immaterial next to gate cost.

The MPS backend runs GHZ-50 in milliseconds at bond dimension χ=2. The density matrix backend runs to n=12 for probabilities, purity and Bloch angles. Its `entropy()` diagonalises the full 2ⁿ × 2ⁿ matrix by Householder tridiagonalisation plus implicitly-shifted QL — O(dim³) once, not per sweep. That is interactive to about n=9 (0.5s), 4.9s at n=10 and 56s at n=11; memory becomes the constraint past that, since the solver works on a real 2·dim × 2·dim embedding. The Clifford backend accepts only gates in {H, S, S†, X, Y, Z, CNOT, CZ, CY, SWAP} and throws if the circuit contains non-Clifford gates (T, Rx, etc.).

All backends accept an `initialState` option to start from an arbitrary computational basis state instead of |0...0⟩:

```typescript
// Start from |110⟩ (q0=1, q1=1, q2=0)
circuit.run({ initialState: '110' })
circuit.runMps({ shots: 1000, initialState: '110' })
circuit.statevector({ initialState: '110' })
```

### Clifford+T: stabilizer rank

`runStabilizerRank()` carries the state as a sum of stabilizer states,
|ψ⟩ = Σ c_α|φ_α⟩, each in CH-form. Clifford gates act on every term and leave the
count alone; each non-Clifford diagonal gate splits every term in two. Cost is
therefore exponential in *non-Clifford count* and only polynomial in width — the
inverse of the statevector trade, which is why it reaches circuits `run()` cannot
hold.

Accepts `h, x, y, z, s, sdg, t, tdg, rz, u1/p, r2/r4/r8, cx, cy, cz, swap, ccx`.
A Toffoli expands to its standard 7-T decomposition. Mid-circuit measurement is
not supported.

| Option | Default | Effect |
|---|---|---|
| `targetError` | unset | Target ℓ₂ error δ. Sets the term budget to ⌈ξ/δ²⌉ from the circuit's stabilizer extent. |
| `maxTerms` | `Infinity` | Hard cap on terms. Overrides `targetError`. |
| `workers` | 0 | Split the 2^t terms across worker threads. Requires the built bundle and an exact run. |
| `method` | `'auto'` | `'exact'` enumerates all 2ⁿ amplitudes; `'metropolis'` runs the chain of §4.2. `'auto'` picks exact when it fits `exactBudget`. |

Exact simulation costs 2^t terms and runs out near t = 18. `targetError` costs
2^0.228t/δ² instead — 30,495 terms for 50 T gates at δ=0.3, fewer than an exact
t=15 run.

Measured at n=100, δ=0.3 on a 64 GB machine, one run per process, **one shot** —
so these are the cost of building the decomposition, which is what grows with t.
Sampling is charged separately and is flat in t: at t=50 a shot costs ~43ms, so the
default 1024 shots turns the 5.6s below into about 49s.

| t | terms | time | peak RSS |
|---|---|---|---|
| 50 | 30,495 | 5.6s | 1.14 GB |
| 55 | 67,309 | 17.8s | 1.99 GB |
| 60 | 148,565 | 67s | 3.98 GB |
| 65 | 327,916 | 228s | 8.21 GB |
| 70 | 723,785 | 27min | — |

The ceiling is a property of the machine. Terms cost 3n²/8 bytes, so it moves
with width, tolerance and RAM:

```typescript
import { maxTGates, termBudget, extent } from '@kirkelliott/ket'

maxTGates({ qubits: 100, targetError: 0.3, memoryBytes: 64e9 })  // 77
maxTGates({ qubits: 400, targetError: 0.3, memoryBytes: 64e9 })  // 63
```

**`maxTGates` bounds memory, not time, and time is what binds first.** On the
machine above it reports 77, while t=76 was abandoned unfinished after six hours
and peak memory never exceeded 8.2 GB of the 64 available. Runtime grows faster
than the term count and the growth exponent itself rises with t, so do not
extrapolate from low points — measure with `benchmark/stabilizer-rank.ts`, which
takes one T-count per invocation for exactly this reason.

**Approximation is always reported.** Sparsification is unbiased but randomised,
and a streaming run applies it repeatedly, so the single-application error bound
does not certify the total. `Distribution.truncated` marks any approximate run,
`StabilizerRank.sparsifications` counts how often it fired, and `estimateNorm()`
measures the damage — an exact decomposition of a unitary circuit has ‖ψ‖² = 1,
so drift from 1 is what sparsification actually cost:

```typescript
const sr = new StabilizerRank(4, { maxTerms: 64 })
sr.h(0).t(0).cx(0, 1).t(1)
sr.estimateNorm({ epsilon: 0.1, delta: 0.05 })   // ≈ 1 if the run stayed faithful
```

`estimateNorm` uses the norm-estimation routine of §4.3 — inner products against
random equatorial stabilizer states, evaluated as quadratic-form exponential sums
in O(n³) rather than O(2ⁿ).

Metropolis sampling is a heuristic and can be *wrong*, not merely noisy: its
single-bit-flip proposals cannot cross a zero-amplitude basis state, so a
distribution whose support is not single-flip connected is sampled from one
component only. `'auto'` therefore prefers exact enumeration whenever 2ⁿ·terms
fits the budget.

Reference: Bravyi, Browne, Calpin, Campbell, Gosset, Howard, *Simulation of
quantum circuits by low-rank stabilizer decompositions*, Quantum 3, 181 (2019).

### Tuning sparse → dense promotion

The statevector and density-matrix backends each begin sparse and switch to a
contiguous `Float64Array` once the state fills in. Both thresholds are defaults,
not fixed limits — pass `dense` to override them per call:

```typescript
import type { DenseOptions } from '@kirkelliott/ket'

circuit.exactProbs({ dense: { maxQubits: 0 } })          // never promote
circuit.run({ shots: 1024, dense: { maxQubits: 20 } })   // lower the memory ceiling
circuit.dm({ noise: 'forte-1', dense: { fill: 64 } })    // promote sooner
```

| Field | Statevector default | Density matrix default | Meaning |
|---|---|---|---|
| `fill` | 64 | 32 | Promote once the state exceeds `1 / fill` of full occupancy. **Higher** values promote sooner. |
| `maxQubits` | 24 | 12 | Largest qubit count for which a dense buffer is allocated at all. Beyond it the sparse path is used however full the state gets. Both defaults correspond to a 256 MiB buffer. |

Accepted by `run()`, `simulate()`, `statevector()`, `exactProbs()` and `dm()`.
`runMps()` and `runClifford()` do not take it — neither uses this representation.
Invalid values (`fill ≤ 0`, negative or non-integer `maxQubits`) throw
`RangeError` at the call site. Disabling promotion for a state that then
fills up is caught rather than allowed to run the heap dry: `dense: { maxQubits:
0 }` on a 12-qubit density matrix would need 4¹² boxed map entries, so it throws a
`RangeError` naming the entry count and the ceiling to raise. The guard is narrow
— it needs the state past ~4M entries *and* a width the default ceiling would have
allowed, so small forced-sparse runs and circuits too wide for any dense buffer
are untouched.

**The choice never changes a result, only its cost.** A 14-qubit depth-4 circuit
takes 254 ms forced sparse against 13 ms on the default promoting path, with
bit-identical probabilities. The two reasons to reach for it are a constrained
environment that cannot afford the default ceiling, and a large machine where
paying more memory to go faster is the right trade.

## Gates

### Single-qubit

| Gate | Method | Description |
|---|---|---|
| H | `h(q)` | Hadamard |
| X | `x(q)` | Pauli-X (NOT) |
| Y | `y(q)` | Pauli-Y |
| Z | `z(q)` | Pauli-Z |
| S | `s(q)` | Phase (Rz(π/2)) |
| S† | `si(q)` / `sdg(q)` | S-inverse |
| T | `t(q)` | T gate (Rz(π/4)) |
| T† | `ti(q)` / `tdg(q)` | T-inverse |
| V | `v(q)` / `srn(q)` | √X |
| V† | `vi(q)` / `srndg(q)` | √X-inverse |
| Rx | `rx(θ, q)` | X-axis rotation |
| Ry | `ry(θ, q)` | Y-axis rotation |
| Rz | `rz(θ, q)` | Z-axis rotation |
| R2 | `r2(q)` | Rz(π/2) alias |
| R4 | `r4(q)` | Rz(π/4) alias |
| R8 | `r8(q)` | Rz(π/8) alias |
| U1 | `u1(λ, q)` / `p(λ, q)` | Phase gate (p = Qiskit 1.0+ name) |
| U2 | `u2(φ, λ, q)` | Two-parameter unitary |
| U3 | `u3(θ, φ, λ, q)` | General single-qubit unitary |
| VZ | `vz(θ, q)` | VirtualZ (Rz alias) |
| I | `id(q)` | Identity |

### Two-qubit

| Gate | Method | Description |
|---|---|---|
| CNOT | `cnot(c, t)` | Controlled-X |
| SWAP | `swap(q0, q1)` | SWAP |
| CX | `cx(c, t)` | Controlled-X (alias) |
| CY | `cy(c, t)` | Controlled-Y |
| CZ | `cz(c, t)` | Controlled-Z |
| CH | `ch(c, t)` | Controlled-H |
| CRx | `crx(θ, c, t)` | Controlled-Rx |
| CRy | `cry(θ, c, t)` | Controlled-Ry |
| CRz | `crz(θ, c, t)` | Controlled-Rz |
| CR2 | `cr2(c, t)` | Controlled-R2 |
| CR4 | `cr4(c, t)` | Controlled-R4 |
| CR8 | `cr8(c, t)` | Controlled-R8 |
| CU1 | `cu1(λ, c, t)` | Controlled-U1 |
| CU2 | `cu2(φ, λ, c, t)` | Controlled-U2 |
| CU3 | `cu3(θ, φ, λ, c, t)` | Controlled-U3 |
| CS | `cs(c, t)` | Controlled-S |
| CT | `ct(c, t)` | Controlled-T |
| CS† | `csdg(c, t)` | Controlled-S† |
| CT† | `ctdg(c, t)` | Controlled-T† |
| C√X | `csrn(c, t)` | Controlled-√NOT |
| XX | `xx(θ, q0, q1)` | Ising XX interaction |
| YY | `yy(θ, q0, q1)` | Ising YY interaction |
| ZZ | `zz(θ, q0, q1)` | Ising ZZ interaction |
| XY | `xy(θ, q0, q1)` | XY interaction |
| iSWAP | `iswap(q0, q1)` | iSWAP |
| √iSWAP | `srswap(q0, q1)` | Square-root iSWAP |

### Three-qubit

| Gate | Method | Description |
|---|---|---|
| CCX | `ccx(c0, c1, t)` | Toffoli |
| CSWAP | `cswap(c, q0, q1)` | Fredkin |
| C√SWAP | `csrswap(c, q0, q1)` | Controlled-√SWAP |

### Custom unitary gate

```typescript
import { Circuit } from '@kirkelliott/ket'
import type { Complex } from '@kirkelliott/ket'

// Real matrix (number[][])
const SWAP = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
circuit.unitary(SWAP, 0, 1)

// Complex matrix ({ re, im }[][])
const S: Complex[][] = [
  [{ re: 1, im: 0 }, { re: 0, im: 0 }],
  [{ re: 0, im: 0 }, { re: 0, im: 1 }],
]
circuit.unitary(S, 0)
```

`matrix` must be 2^N × 2^N where N is the number of qubits. The first qubit in the argument list is the MSB of the local state index — matching the convention of all other multi-qubit gates. Entries can be plain `number` (real part only) or `Complex` objects.

Supported in all backends: statevector, density matrix, and MPS (1 and 2-qubit only). Throws `TypeError` in `runClifford` (the simulator cannot verify Clifford membership from an arbitrary matrix).

JSON round-trip is lossless — the matrix is stored as `[[re, im], ...][]` in the serialized format.

### Scheduling

| Method | Description |
|---|---|
| `barrier(...qubits)` | Scheduling hint — no-op in simulation, emits `barrier` in QASM. No args = all qubits. |

### Native IonQ gates

| Gate | Method | Description |
|---|---|---|
| GPI | `gpi(φ, q)` | Single-qubit rotation on Bloch equator |
| GPI2 | `gpi2(φ, q)` | Half-angle GPI |
| MS | `ms(φ₀, φ₁, q0, q1)` | Mølmer-Sørensen entangling gate |

## Device targeting

ket ships noise profiles for IonQ, IBM, and Quantinuum hardware. All profiles are accessible via `DEVICES` and usable by name anywhere a `noise` option is accepted.

```typescript
import { DEVICES, IONQ_DEVICES, Circuit } from '@kirkelliott/ket'

// Query any device
const forte  = DEVICES['forte-1']       // { vendor, qubits, status, connectivity, nativeGates, noise, source }
const eagle  = DEVICES['ibm_brisbane']  // { qubits: 127, noise: {...} }
const helios = DEVICES['helios']        // { qubits: 98, noise: {...} }

// Retired machines stay queryable so old results reproduce; check before submitting
DEVICES['aria-1'].status                // 'retired'

// Use by name in any simulation method
circuit.run({ shots: 1000, noise: 'ibm_brisbane' })
circuit.runClifford({ shots: 10000, noise: 'forte-1' })
circuit.dm({ noise: 'h2-1' })

// Create a circuit sized for a specific device
const c = Circuit.device('forte-1')  // new Circuit(36)
```

**All devices** (`DEVICES`):

| Device | Vendor | Status | Qubits | Connectivity | p1 (1Q) | p2 (2Q) | pMeas |
|---|---|---|---|---|---|---|---|
| `forte-1` | IonQ | available | 36 | all-to-all | 0.027% | 0.49% | 0.20% |
| `forte-enterprise-1` | IonQ | available | 36 | all-to-all | 0.027% | 0.49% | 0.20% |
| `helios` | Quantinuum | available | 98 | all-to-all | 0.0025% | 0.079% | 0.10% |
| `h2-1` | Quantinuum | available | 56 | all-to-all | 0.0019% | 0.11% | 0.10% |
| `ibm_brisbane` | IBM | available | 127 | heavy-hex | 0.024% | 0.76% | 1.35% |
| `aria-1` | IonQ | **retired** | 25 | all-to-all | 0.05% | 1.33% | 0.40% |
| `aria-2` | IonQ | **retired** | 25 | all-to-all | 0.066% | 1.86% | 0.40% |
| `harmony` | IonQ | **retired** | 11 | all-to-all | 0.10% | 1.50% | 1.00% |
| `ibm_sherbrooke` | IBM | **retired** | 127 | heavy-hex | 0.024% | 0.74% | 1.35% |
| `ibm_torino` | IBM | **retired** | 133 | heavy-hex | 0.020% | 0.30% | 1.00% |
| `h1-1` | Quantinuum | **retired** | 20 | all-to-all | 0.0018% | 0.097% | 0.23% |

Every entry carries `vendor`, `status`, `connectivity`, a `source` string, and a
`confidence` field that is either `'vendor-published'` or `'estimated'`:

```typescript
DEVICES['forte-1'].confidence       // 'vendor-published' — IonQ's own r_1q/r_2q
DEVICES['ibm_brisbane'].confidence  // 'estimated' — representative Eagle-r3 profile
```

`confidence` is a field rather than a footnote because the distinction is
load-bearing. An earlier revision of this table carried IonQ rates **2.5× more
optimistic** than IonQ's published parameters, and nothing in the type made that
checkable. Only two vendors publish usable figures: IonQ's `r_1q`/`r_2q`, and
Quantinuum's Helios gate fidelities. IBM publishes per-device calibration only
behind an account, so **no IBM entry is vendor-attested** — those rows are
representative profiles for hardware of that class, not calibrations of the named
machine. Every `pMeas` is estimated; no vendor here publishes readout error. Retired machines are kept so historical
results stay reproducible — they will not accept new jobs.

**Connectivity is not cosmetic.** Trapped-ion machines (IonQ, Quantinuum) couple
any pair directly. IBM's heavy-hex lattice does not, so a circuit with distant
two-qubit gates needs SWAP networks, multiplying both gate count and error beyond
what the `p2` column alone suggests.

**These are published approximations, not live calibration.** IonQ's figures are
the `r_1q`/`r_2q` parameters from its noise-model documentation, which IonQ
explicitly says should not be compared directly to measured fidelities. IBM
recalibrates daily. Refresh IonQ entries with:

```bash
IONQ_API_KEY=... node scripts/refresh-ionq-devices.ts
```

Known gaps: IBM's current Heron r2/r3 and Nighthawk systems are absent — no
per-device error rates could be sourced without an IBM account. IonQ's post-2025-09
Forte noise model is unpublished, so `forte-1` carries the earlier parameters and
`forte-enterprise-1` reuses them as a stand-in.

**IonQ devices** (`IONQ_DEVICES`) additionally expose `nativeGates`, the vendor's native gate set. It is informational — IonQ accepts abstract gates and compiles them itself, so `checkDevice` validates against what IonQ's JSON format can express rather than against this list:

```typescript
import { IONQ_DEVICES, Circuit } from '@kirkelliott/ket'

const forte = IONQ_DEVICES['forte-1']
// { vendor: 'IonQ', qubits: 36, status: 'available', nativeGates: ['gpi','gpi2','zz'], ... }

// Validate before submitting
const circuit = new Circuit(2).h(0).cnot(0, 1)
circuit.checkDevice('forte-1')  // passes
circuit.toIonQ()                // safe to call

// checkDevice throws with all issues at once
new Circuit(30).cu1(Math.PI / 4, 0, 1).checkDevice('harmony')
// TypeError: Circuit is not compatible with harmony:
//   - circuit uses 30 qubits; harmony supports at most 11
//   - gate 'cu1' is not supported on harmony
//   Call toIonQBasis() to expand unsupported gates into IonQ ones.
```

### Expanding unsupported gates

`toIonQ()` is a serializer, not a compiler: it rejects any gate outside IonQ's
set rather than silently expanding it, so the gate count you inspect is the gate
count you send. `toIonQBasis()` is the explicit expansion step.

This matters for QFT and modular-exponentiation circuits, which are built almost
entirely from `cu1` and could not previously be exported at all:

```typescript
const shor = shorCircuit(15n, 7n, 3)
shor.toIonQ()                       // TypeError: 'cu1' is not serializable

const native = shor.toIonQBasis()   // cu1 -> rz + controlled rz; cswap -> controlled x
native.toIonQ()                     // 5,798 gates, submit-ready
```

Controlled gates are emitted natively, as IonQ's format expects: a base gate
plus a `controls` array, so `ccx` is `{gate:'x', controls:[a,b], target:t}`
rather than fifteen Clifford+T gates. What remains is still not free — `cu1`
becomes two gates and `cswap` three — so check `gateCounts()` before submitting.
It composes with `compile()` for hardware-native output:

```typescript
shor.toIonQBasis().compile('forte-1').toIonQ()   // rz, gpi2, gpi, ms
```

Each rule is exact up to global phase; `u1` and `cu1` expand to `rz`, which
differs by an unobservable phase of e^(−iθ/2).

### Running on IonQ hardware

`toIonQ()` produces the payload; `runIonQ` submits it, waits, and fetches the
results. No dependency is added — the client uses the global `fetch`.

```typescript
import { Circuit, runIonQ, countsToProbs } from '@kirkelliott/ket'

const bell = new Circuit(2).h(0).cnot(0, 1)
bell.checkDevice('forte-1')                      // fail fast, before spending queue time

const { job, counts } = await runIonQ(bell.toIonQ(), {
  apiKey: process.env.IONQ_API_KEY!,
  target: 'qpu.forte-1',                         // or 'simulator' — free, no queue
  shots: 1024,
  onPoll: j => console.log(j.status),
})

countsToProbs(counts, 2, 1024)                   // { '00': 0.49, '11': 0.51 }
```

| Function | Purpose |
|---|---|
| `runIonQ(circuit, opts)` | Submit, poll, fetch results, convert. The path that cannot be got wrong. |
| `submitIonQ(circuit, opts)` | Queue a job. Returns once accepted, not when it finishes. |
| `getIonQJob(id, opts)` | Fetch current state. **Does not include results.** |
| `awaitIonQJob(id, opts)` | Poll to a terminal state. **Throws** on `failed`/`canceled`, so a caller who forgets to check `status` cannot read a failure as empty results. |
| `getIonQResults(job, opts)` | Fetch the histogram, following the job's `results_url`. |
| `ionqSubmitRequest(circuit, opts)` | Build the request without sending it — for testing, or to route it yourself. |
| `ionqHistogramToCounts(histogram, shots)` | Probabilities → shot counts keyed by basis index. |
| `cancelIonQJob(id, opts)` | Cancel a queued or running job and release its slot. |

**State the noise model rather than inheriting it.** Omitting `noise` leaves the
choice to IonQ's server-side default, which is not guaranteed to be noiseless. At
any real depth a device model flattens the distribution completely, and a flat
distribution is indistinguishable from a wrong answer:

```typescript
await submitIonQ(circuit.toIonQ(), {
  apiKey, target: 'simulator',
  noise: { model: 'ideal' },        // or { model: 'forte-1', seed: 100 }
})
```

The job record reports the model actually used; check it before concluding a
circuit is wrong.

**Angle units differ between the two gate sets.** QIS gates (`rx`/`ry`/`rz`/
`xx`/`yy`/`zz`) take `rotation` in **radians** — IonQ's own example gives Rx(π/2)
as `rotation: 1.5708`. The native gates (`gpi`/`gpi2`/`ms`) take `phase` in
**turns**, where 1.0 = 2π. ket handles both, and the distinction is not
cosmetic: getting it wrong sends a valid circuit that computes something else,
and Clifford-only circuits such as a Bell pair are blind to the error.

**Results are not on the job object.** `GET /jobs/{id}` returns metadata and a
`results_url`; the histogram lives behind it. `runIonQ` and `getIonQResults`
follow that automatically — reading `job.data` will find nothing.

**Bit order matches, and that is verified rather than assumed.** IonQ histogram
keys are little-endian integers with qubit *i* at 2^*i*, which is exactly ket's
convention, so no conversion is applied. Submitting `x(0)` on two qubits returns
`{"1": 1.0}` and `x(0).x(2)` on four returns `{"5": 1.0}` — both the indices ket
assigns to those states. This is worth stating explicitly
because published summaries describe the keys as big-endian; they are not, and
reversing them corrupts every asymmetric result while leaving symmetric ones such
as a Bell state looking perfectly correct.

### Calling from a browser: use a proxy

`api.ionq.co` sends no `access-control-allow-origin`, so a browser cannot call it
directly. That is a feature: an IonQ API key buys paid hardware time, and a key
in a JS bundle is a key anyone can spend. Keep it server-side and forward.

The `endpoint` option exists for this — point it at your own route:

```typescript
// server — Express, Hono, Next route handler, whatever you already run
app.post('/api/ionq/jobs', async (req, res) => {
  const job = await submitIonQ(req.body.input, {
    apiKey: process.env.IONQ_API_KEY!,           // never leaves the server
    target: req.body.target,
    shots: req.body.shots,
  })
  res.json(job)
})

app.get('/api/ionq/jobs/:id', async (req, res) => {
  res.json(await getIonQJob(req.params.id, { apiKey: process.env.IONQ_API_KEY! }))
})
```

```typescript
// browser — same library, same types, no key
const { id } = await submitIonQ(circuit.toIonQ(), {
  endpoint: '/api/ionq', target: 'qpu.forte-1', shots: 1024,   // no apiKey: it lives on the server
})
const job = await awaitIonQJob(id, { endpoint: '/api/ionq' })
```

Both halves import the same `Circuit`, so the browser can simulate a circuit for
instant feedback and hand the identical object to the server to run on hardware —
one circuit representation, no serialisation boundary between two languages.

## Import / Export

| Format | Import | Export | Method(s) |
|---|---|---|---|
| OpenQASM 2.0 / 3.0 | ✓ | ✓ | `Circuit.fromQASM(s)` / `circuit.toQASM()` |
| IonQ JSON | ✓ | ✓ | `Circuit.fromIonQ(json)` / `circuit.toIonQ()` |
| Quil 2.0 | ✓ | ✓ | `Circuit.fromQuil(s)` / `circuit.toQuil()` |
| JSON (native) | ✓ | ✓ | `Circuit.fromJSON(json)` / `circuit.toJSON()` |
| Qiskit (Python) | ✓ | ✓ | `Circuit.fromQiskit(s)` / `circuit.toQiskit()` |
| Qiskit Qobj JSON | ✓ | — | `Circuit.fromQobj(json)` |
| Cirq (Python) | ✓ | ✓ | `Circuit.fromCirq(s)` / `circuit.toCirq()` |
| Q# | — | ✓ | `circuit.toQSharp()` |
| pyQuil | — | ✓ | `circuit.toPyQuil()` |
| Amazon Braket | — | ✓ | `circuit.toBraket()` |
| CudaQ | — | ✓ | `circuit.toCudaQ()` |
| TensorFlow Quantum | — | ✓ | `circuit.toTFQ()` |
| Quirk JSON | — | ✓ | `circuit.toQuirk()` |
| LaTeX (quantikz) | — | ✓ | `circuit.toLatex()` |

### OpenQASM coverage

`Circuit.fromQASM` parses OpenQASM 2.0 and the 3.0 subset that maps onto it,
auto-detecting the version.

**Supported:** any number of `qreg`/`creg` (or `qubit`/`bit`) declarations, laid
out in declaration order — `qreg a[2]; qreg b[2];` puts `b[0]` at qubit 2;
`gate` definitions, expanded inline, including parameterised and nested ones;
`opaque` declarations; `measure` in both the `->` and assignment forms; `reset`;
`barrier`; `if (creg == N) <statement>`; register-wide broadcast (`h q;`,
`cx a,b;`, `measure q -> c;`); `//` and `/* */` comments; the `U`/`CX` builtins
and all of qelib1.inc.

Angle expressions accept `pi`, `+ - * / ^`, parentheses, scientific notation,
`sin cos tan exp ln sqrt`, and — inside a gate body — that gate's own parameters.

**Rejected with a `TypeError`,** rather than silently mis-parsed: undeclared
registers, out-of-range indices, gate arity mismatches, registers that cannot be
broadcast together, gate modifiers (`ctrl @`, `inv @`, `pow @`), `gphase`,
`for`/`while`/`def`, `else`, register aliasing (`let`), and QASM 3 classical
types beyond `bit`.

## Mutable adapter

`@kirkelliott/ket/compat` exports a `QuantumCircuit` class with a mutable,
column-indexed API: gates are placed at an explicit column, the circuit grows to
fit any wire named, and `run()` stores state on the instance. It matches the API
shape used by the `quantum-circuit` package, so code written against that shape
runs unmodified.

It is an adapter, not a second API to build on. `toKet()` returns the equivalent
immutable `Circuit`, which is where the rest of this reference applies.

```javascript
import { QuantumCircuit } from '@kirkelliott/ket/compat'

const circuit = new QuantumCircuit(2)
circuit.addGate('h', 0, 0)
circuit.addGate('cx', 1, [0, 1])
circuit.addMeasure(0, 'c', 0)
circuit.run()

circuit.getCregValue('c')          // 0 or 1
circuit.measureAllMultishot(1024)  // { '00': 517, '11': 507 }
circuit.toKet()                    // → Circuit, for the rest of the library
```

| Area | Methods |
|---|---|
| Construction | `init`, `clearGates`, `resetState`, `addGate`, `appendGate`, `addMeasure`, `removeGate`, `appendCircuit` |
| Shape | `numQubits`, `numCols`, `numAmplitudes`, `getDepth`, `usedGates` |
| Classical registers | `createCreg`, `getCregs`, `getCregValue`, `getCregBit`, `setCregBit`, `cregsAsString` |
| Execution | `run(initialValues, options)`, `probabilities`, `probability`, `measure`, `measureAll`, `measureAllMultishot` |
| State | `stateAsString`, `print` |
| Custom gates | `registerGate`, `save`, `load` |
| Interchange | `importQASM`, `exportQASM`, `exportToQiskit`, `exportToCirq`, `exportToQuil`, `exportToPyquil`, `exportToQSharp`, `exportToTFQ`, `exportToBraket`, `exportToIonq`, `exportSVG` |
| Escape hatch | `toKet(initialValues?)` |

Every gate name this adapter accepts is a gate ket already implements under the
same name, so no translation table is involved.

**Bit order.** Strings returned by `stateAsString` and `measureAllMultishot` put
the highest wire leftmost — the opposite of ket's native wire-0-leftmost order,
and the convention this API shape expects. `probabilities()` and `measureAll()`
are indexed by wire.

**Measurement is non-destructive.** A `measure` gate writes to its classical
register but leaves the state alone, unless the circuit also contains a
classically-controlled gate or a reset, in which case it collapses.

**Seeding.** `run(initialValues, { seed })` makes a run reproducible. Unseeded
runs draw a fresh seed from the platform CSPRNG each time, so running the same
circuit in a loop gives independent results. This seeds a deterministic sampler —
for cryptographic randomness call `crypto.getRandomValues` directly.

**QASM output** follows ket's formatting: a blank line after the `include`, and no
space after the comma in `cx q[0],q[1];`.

## Amplitudes by tensor-network contraction

`amplitudeByContraction(circuit, bitstring)` computes ⟨x|U|0…0⟩ by treating the
circuit as a tensor network — every gate a tensor, every wire segment an index —
and contracting it to a scalar.

```typescript
import { amplitudeByContraction } from '@kirkelliott/ket'

const { re, im, width, cost } = amplitudeByContraction(circuit, '0'.repeat(400))
```

Cost has nothing to do with 2ⁿ. It is set by the **width** of the contraction —
the largest intermediate tensor — which is a property of the circuit's
connectivity rather than its qubit count. A statevector is simply the special case
of a contraction whose width is n.

| n, depth 4 | width | operations | time |
|---|---|---|---|
| 20 | 4 | 2.7e3 | 18 ms |
| 40 | 4 | 5.5e3 | 24 ms |
| 160 | 4 | 2.2e4 | 99 ms |
| 400 | 4 | 5.6e4 | 230 ms |

Width stays flat as n grows and rises with **depth** instead, which is where the
real limit is:

| n = 40 | width | operations | time |
|---|---|---|---|
| depth 4 | 4 | 5.5e3 | 24 ms |
| depth 8 | 7 | 6.8e4 | 51 ms |
| depth 12 | 13 | 2.2e6 | 87 ms |
| depth 16 | 17 | 1.2e8 | 282 ms |

So this wins decisively on wide shallow circuits and loses to a statevector once
depth pushes the width past n. A single call returns **one amplitude**, not a
distribution — but `amplitudeBatchByContraction` leaves chosen qubits open and
returns all 2^k at once, which is what makes sampling practical.

### The order is the algorithm

The same network contracted well or badly differs by orders of magnitude, so
`planContraction` is exposed separately and searched independently of the
contraction itself. It runs randomized greedy with restarts, minimising width
first — width sets memory, and memory is what makes a contraction impossible.

Planning dominates runtime at large n — contracting a 400-qubit depth-4 circuit is
5.6e4 operations, microseconds of arithmetic, and effectively all of the 230 ms
goes on choosing the order.

Three planners are available, and which one wins is a property of the network:

| Planner | Method |
|---|---|
| `planContraction` | Randomized greedy with restarts, minimising width then cost. |
| `planContractionPartitioned` | Recursive balanced bisection with Fiduccia–Mattheyses refinement. |
| `planBest` | Runs both, scores them with `evaluatePlan`, returns the better and which it was. |

**Neither planner dominates — they win on different networks.** Greedy is better
on small and mid-size circuits; recursive bisection pulls ahead as the width
grows, which is the regime where the order matters most:

| circuit | greedy | bisection |
|---|---|---|
| shallow n=14 d=8 | **width 7** | 8 |
| shallow n=30 d=6 | **width 5** | 6 |
| shallow n=40 d=12 | 13 | **width 12** |
| shallow n=50 d=14 | 15 | **width 13** (4× less memory) |
| shallow n=60 d=16 | 17 | **width 15** |
| long-range pairings | 4–10 | tie |

Greedy's quality plateaus: 64, 256 and 1,024 restarts all return the same width,
so extra search buys nothing past the default. Bisection's wins are quality greedy
cannot reach by repetition — but it costs seconds where greedy costs milliseconds,
and `planBest` is dominated by it. Pass `imbalance` or a low `restarts` to bound
that, or call `planContraction` directly when latency matters more than width.

Two things were needed to get there, and both were found by measurement rather
than assumed:

- **Multilevel coarsening.** Flat refinement cannot escape a local minimum unless
  some single-vertex move improves things, and on these graphs none does. The
  coarsened levels collapse clusters into single vertices, so one move relocates
  a whole region; the partition is projected back down and polished at each level.
- **Searching the balance tolerance.** A balanced split is the wrong shape for a
  circuit: the best order for a shallow circuit is a sweep, which is maximally
  lopsided. Holding halves cost 2–4 width; opening the tolerance recovers it, and
  the best value differs per circuit, so it is searched rather than fixed.

`planBest` exists so the choice does not have to be made in advance, and scoring
is cheap because `evaluatePlan` replays a plan over index sets without touching
tensor data.

### Slicing, when width is the wall

No contraction order helps once the width exceeds memory — 2^width is 2^width.
Slicing fixes a set of indices instead of summing over them, contracts once per
assignment, and adds the results:

```typescript
const { re, im, width, slices, overhead } =
  amplitudeBySlicedContraction(circuit, bitstring, { targetWidth: 24 })
```

Each sliced index halves the memory and doubles the number of contractions, and
the contractions are completely independent — this is the mechanism that let
petabyte-scale contractions run on ordinary hardware, by spreading the slices
across machines.

The doubling is a worst case, and the gap widens with the slice count, because a
slice is not merely a smaller copy of the same contraction — removing an index
also removes work that was being repeated inside it:

| circuit | width | sliced | slices | actual overhead |
|---|---|---|---|---|
| n=20 d=12 → target 9 | 12 → **9** | 5 | 32 | **4.7×** |
| n=20 d=12 → target 7 | 12 → **7** | 9 | 512 | **26.9×** |
| n=40 d=12 → target 10 | 13 → **11** | 3 | 8 | **4.8×** |

The middle row is the point: **32× less memory for 27× more work**, against a
naive expectation of 512×, and every one of those 512 contractions is independent.

Selection ranks candidates by how many of the widest intermediates carry them.
Width alone is the wrong signal — a peak held by six intermediates does not fall
when an index leaves five of them, so a width-only rule reports no progress and
stops before it starts. Emptying that peak set is the step before the width
moves, so progress is measured on `(width, count at that width)` and a slice is
taken only when one of them improves.

Because that comparison is between two *planned* contractions, it is only as
trustworthy as the planner is repeatable. Scoring candidates with too few
restarts measures planner variance rather than the slice, and the search stops
early — the same target that reaches width 7 with 24 trial restarts stalls at 11
with two. Planning is cheap enough now to buy that certainty.

Slicing stops at `maxSliced` (default 12) and reports the width it actually
reached rather than the width requested.

| Function | Purpose |
|---|---|
| `sliceContraction(indexSets, opts?)` | Choose slice indices and plan one slice. |
| `contractSliced(tensors, slicedPlan)` | Contract every slice and sum. |
| `projectTensor(tensor, fixed)` | Fix indices to values, dropping those axes. |
| `amplitudeBySlicedContraction(circuit, bitstring, opts?)` | The whole path in one call. |

### Many amplitudes at once

Contracting to a scalar answers one question: the amplitude of a single
bitstring. Sampling needs many, and running the contraction once per bitstring
pays the whole cost again each time.

Write `?` for a qubit to leave its wire open. The contraction then ends on a
tensor over those qubits instead of a scalar — 2^k amplitudes for roughly the
price of one:

```typescript
import { amplitudeBatchByContraction } from '@kirkelliott/ket'

const batch = amplitudeBatchByContraction(circuit, '?'.repeat(16) + '0'.repeat(44))
batch.open            // [0, 1, …, 15] — the qubits left open, ascending
batch.data            // 2^16 amplitudes, re/im interleaved
batch.at([1, 0, 1, …]) // → { re, im }, indexed in `open` order
```

At n=60 depth 10, 65,536 amplitudes take **76 ms** in one batched contraction.
One at a time, the same set is a projected ~3,299 s — the batch is not a constant
factor faster, it removes the repetition entirely.

The open indices widen every intermediate that carries them, so batch size trades
directly against contraction width. That is the same currency slicing spends,
which is why the two are normally used together: open enough qubits to make
sampling worthwhile, then slice the width back down to fit memory.

| Function | Purpose |
|---|---|
| `amplitudeBatchByContraction(circuit, pattern, opts?)` | Every amplitude matching a `?`-pattern, in one contraction. |

### The kernel

A contraction over shared indices is a matrix product once those indices are made
contiguous, so `contractPair` permutes both operands — shared indices trailing on
the left, leading on the right — and runs an ordinary GEMM. The alternative is to
walk output positions and re-derive each operand's offset a bit at a time, paying
that decomposition per element and reading memory in a stride the hardware cannot
prefetch.

| contraction | naive kernel | permute + GEMM |
|---|---|---|
| n=16 d=10 (width 9) | 1.0 ms | 0.7 ms |
| n=20 d=12 (width 13) | 8.7 ms | 3.2 ms |
| n=24 d=12 (width 13) | 11.6 ms | 3.1 ms |
| n=28 d=14 (width 13) | 77.5 ms | **16.7 ms** |

The advantage grows with size, which is the signature of removing per-element
overhead rather than shaving a constant. Two details carry most of it: operands
already in the right order skip the permutation entirely, which is about half of
them in a circuit network; and a zero row of the left operand skips a whole pass
over the right one, which matters because gate tensors are mostly zeros — a CNOT
has four non-zero entries out of sixteen.

### Planner cost

Candidate pairs are held in a min-heap keyed by score, with lazy invalidation by
version counter, so a contraction step costs the merged tensor's degree rather
than a scan of every remaining pair. Rescanning made planning O(tensors ×
candidates) per step and dominated everything else:

| network | scanning | heap |
|---|---|---|
| n=20 d=12, 654 tensors | 578 ms | **14 ms** |
| n=28 d=14, 1,057 tensors | 1,460 ms | **17 ms** |
| n=60 d=12, 1,974 tensors | 5,182 ms | **25 ms** |
| n=100 d=8, 2,296 tensors | 6,658 ms | **27 ms** |

Same restart count in both columns. The speedup is not free: freezing a
candidate's jittered score when it is pushed means each restart explores a little
less than rescoring everything every step did, so a single restart lands 1–2 wider.
Restarts are now cheap enough that spending the win on more of them more than
covers it — at 128 restarts the heap matches or beats the old planner's width on
most networks while still being 5–12× faster — so the default restart count is 64
rather than 24.

### Working with the network directly

The pieces are exposed so a network can be built, planned and contracted
separately — which is what you want when substituting a planner, or contracting
one network against many output bitstrings.

| Function | Purpose |
|---|---|
| `circuitNetwork(circuit, bitstring)` | Build the `Tensor[]` for ⟨bitstring\|U\|0…0⟩. A SWAP costs nothing — it is a relabelling, not a tensor. |
| `planContraction(indexSets, opts?)` | Search for an order. Takes only the index sets, so planning never touches the data. |
| `contractNetwork(tensors, plan)` | Execute a plan, returning the remaining tensor. |
| `contractPair(a, b)` | Contract two tensors over their shared indices. |
| `permuteTensor(tensor, order)` | Reorder indices, returning the tensor unchanged when it already matches. |
| `planContractionPartitioned(indexSets, opts?)` | Recursive bisection planner. |
| `planBest(indexSets, opts?)` | Best of the available planners, with `strategy` naming the winner. |
| `evaluatePlan(indexSets, steps)` | Score a plan's width and cost without contracting. |

```typescript
const net  = circuitNetwork(circuit, '0'.repeat(n))
const plan = planContraction(net.map(t => t.indices), { restarts: 64 })
const out  = contractNetwork(net, plan)   // out.indices === [] — a scalar
```

## Expectation values by Pauli path

`pauliPathExpectation(circuit, observable, options?)` computes ⟨ψ|O|ψ⟩ for a Pauli
observable without ever building a state. It propagates the observable *backward*
through the circuit in the Heisenberg picture — ⟨ψ|O|ψ⟩ = ⟨0|U†OU|0⟩ — and reads
the result off against |0…0⟩.

```typescript
import { pauliPathExpectation } from '@kirkelliott/ket'

const { value, peakTerms, droppedWeight, truncated } =
  pauliPathExpectation(ansatz, [{ coeff: 1, ops: 'ZZ' + 'I'.repeat(118) }])
```

The cost model is unlike the other backends. A Clifford gate maps one Pauli to
±one Pauli, at any width, so width is close to free; only rotations branch, and
each branch carries a cos/sin factor, so terms decay and can be truncated. What
costs is the number of surviving terms, which is set by the observable's light
cone rather than the qubit count:

| n | time | peak terms |
|---|---|---|
| 8 | 1.1 ms | 63 |
| 16 | 1.9 ms | 63 |
| 30 | 3.5 ms | 63 |
| 60 | 8.4 ms | 63 |
| 120 | 27.6 ms | 63 |

Two QAOA layers with a two-local observable. A statevector needs 127 ms at n=16
and is impossible past ~24.

**It answers ⟨O⟩, not a distribution.** There is no sampling and no state; for
shots use another backend. A deep circuit of arbitrary rotations still grows
exponentially in terms — this is efficient for wide, shallow or heavily-Clifford
circuits, which is what VQE and QAOA ansatze are.

| Option | Default | Meaning |
|---|---|---|
| `threshold` | `1e-10` | Discard terms with coefficient below this magnitude. |
| `maxTerms` | `Infinity` | Hard cap on simultaneous terms; smallest dropped first. |
| `maxWeight` | `Infinity` | Discard terms acting non-trivially on more than this many qubits. |
| `noise` | — | `{ p1, p2 }` depolarizing rates, as in `DmNoiseParams`. |

### Truncating

Term count grows roughly ten-fold per circuit layer, so the choice of truncation
is what decides whether a depth is reachable. Measured on a 60-qubit kicked-Ising
circuit with a single-site `Z` observable:

| depth 6, n = 60 | terms | time | ⟨Z⟩ |
|---|---|---|---|
| `threshold: 1e-8` | 62,626 | 8,665 ms | 0.0809 |
| `maxWeight: 6` | 5,087 | 1,186 ms | 0.0808 |
| `noise: { p1: 0.005, p2: 0.02 }`, `threshold: 1e-3` | 541 | 124 ms | 0.0523 |

Weight truncation is 7× faster here and agrees to the fourth decimal, because a
high-weight Pauli reaches the |0…0⟩ readout through more cancelling paths and
contributes little.

Modelling the device's **noise** is the bigger lever, and it is not an
approximation of the ideal circuit — it is a better model of the hardware. A
depolarizing channel damps a weight-w Pauli geometrically in w, so terms die off
instead of proliferating: 70× faster here, and the answer it gives (0.0523) is
what the noisy device would produce, not what a perfect one would.

Truncation is reported, not hidden: `droppedWeight` is the summed magnitude of
every discarded coefficient, an upper bound on the error in `value`.

Supported gates: `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`,
`u1`, `p`, `cnot`, `cz`, `swap`. Anything else is refused by name rather than
approximated.

## Algorithms

```typescript
import { Circuit, qft, iqft, grover, phaseEstimation, vqe, gradient, minimize,
         realAmplitudes, efficientSU2, PauliOp, trotter, qaoa, maxCutHamiltonian } from '@kirkelliott/ket'

// Quantum Fourier Transform
const qftCircuit = qft(4)
const iqftCircuit = iqft(4)

// Grover's search — find the marked state
const oracle = (c: Circuit) => c.cz(0, 1)  // mark |11⟩
const search = grover(2, oracle)

// Quantum Phase Estimation — estimates phase of T gate (φ = 1/8)
// T|1⟩ = e^{iπ/4}|1⟩; controlled-T^{2^k} = CU1(π·2^k/4)
// precision=3 counting qubits (q0–q2) + 1 target qubit (q3)
const qpe = phaseEstimation(3,
  (c, ctrl, pow, tgts) => c.cu1(Math.PI * pow / 4, ctrl, tgts[0]!), 1)
// Initialise target qubit (q3) to eigenstate |1⟩ via initialState (q0 leftmost → q3=1 at position 3)
const result = qpe.run({ shots: 1000, seed: 42, initialState: '0001' })
// Phase φ=1/8 → counting register = |001⟩ (q0=1) → dominant bitstring '1001'

// Pauli expectation value — ⟨ψ|P|ψ⟩ for a single Pauli string
// pauli[q] acts on qubit q (q0 leftmost). X → H rotation, Y → Rx(π/2), Z → identity.
new Circuit(1).h(0).expectation('X')                        // 1   (|+⟩ eigenstate of X)
new Circuit(2).h(0).cnot(0, 1).expectation('ZZ')           // 1   (Bell state ⟨ZZ⟩)
new Circuit(2).h(0).cnot(0, 1).expectation('XX')           // 1   (Bell state ⟨XX⟩)
new Circuit(2).h(0).cnot(0, 1).expectation('YY')           // -1  (Bell state ⟨YY⟩)

// Variational Quantum Eigensolver — exact statevector, no sampling noise
const H = [{ coeff: 0.5, ops: 'ZI' }, { coeff: 0.5, ops: 'IZ' }]
const energy = vqe(new Circuit(2).ry(Math.PI / 4, 0).cnot(0, 1), H)

// Standard ansatz circuits — paramCount tells you how many parameters to initialise
const ansatz = realAmplitudes(2, 2)  // Ry + CNOT layers, real amplitudes
ansatz.paramCount                    // 6 (= n × (reps + 1))

const ansatz2 = efficientSU2(2, 2)  // Ry·Rz + CNOT layers, full SU(2)
ansatz2.paramCount                   // 12 (= 2n × (reps + 1))

// Exact analytic gradient via parameter shift rule — 2N vqe() calls for N parameters
// ∂⟨H⟩/∂θᵢ = ½[⟨H⟩(θᵢ + π/2) − ⟨H⟩(θᵢ − π/2)]
const hamiltonian = [{ coeff: 1, ops: 'ZZ' }, { coeff: 0.5, ops: 'ZI' }]
const grad = gradient(ansatz, hamiltonian, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

// Gradient descent optimizer — converges when gradient L2 norm < tol
const { energy: groundEnergy, params, converged } = minimize(
  ansatz, hamiltonian, Array(ansatz.paramCount).fill(0.1), { lr: 0.2, steps: 500 },
)
// groundEnergy → −1.5  (ground state of ZZ + 0.5·ZI)

// Pauli operator algebra — compose Hamiltonians, check commutativity, compute products
const H1 = PauliOp.from([{ coeff: 1, ops: 'ZI' }, { coeff: 1, ops: 'IZ' }])
const H2 = PauliOp.from([{ coeff: 0.5, ops: 'XX' }])
vqe(new Circuit(2), H1.add(H2).toTerms())    // ⟨H1 + H2⟩
H1.scale(2).toTerms()                         // [{ coeff: 2, ops: 'ZI' }, ...]

const X = PauliOp.from([{ coeff: 1, ops: 'X' }])
const Y = PauliOp.from([{ coeff: 1, ops: 'Y' }])
X.mul(Y)           // iZ — product with phase tracking
X.commutator(Y)    // 2iZ — [X, Y] = XY − YX
// .toTerms() throws on non-Hermitian results (imaginary coefficients)

// Trotterized Hamiltonian simulation — e^{-iHt} ≈ (∏_j e^{-iH_j·t/r})^r
const Htrotter = [{ coeff: 1.0, ops: 'ZZ' }, { coeff: 0.5, ops: 'XX' }]
const evolution = trotter(2, Htrotter, Math.PI / 4, 4, 2)  // 4 steps, order 2

// QAOA Max-Cut — 4-cycle graph, p=1
const edges: [number, number][] = [[0,1],[1,2],[2,3],[3,0]]
const circuit = qaoa(4, edges, [Math.PI / 4], [0.15 * Math.PI])
vqe(circuit, maxCutHamiltonian(4, edges))  // → ~2.95  (random = 2.0, optimal = 4.0)
circuit.exactProbs()
// Top states: '1010': 0.265, '0101': 0.265  ← the two optimal bipartitions
```

`realAmplitudes(n, reps)` and `efficientSU2(n, reps)` return ansatz functions with a `.paramCount` property. Both use linear CNOT entanglement; `efficientSU2` adds Rz rotations for full SU(2) coverage per qubit.

`gradient(ansatz, hamiltonian, params)` computes exact analytic gradients via the parameter shift rule — not finite differences. The rule is exact for any gate of the form e^{−iθP/2} (Rx, Ry, Rz, and all standard rotation gates). `minimize(ansatz, hamiltonian, initialParams, options?)` runs gradient descent until convergence or step budget exhaustion, returning `{ params, energy, steps, converged }`.

`grover(n, oracle)` kicks the phase of the marked state; `groverAncilla(n, oracle)` instead gives the oracle an explicit ancilla qubit to flip, which is often the more natural way to express a Boolean predicate.

### VQE beyond statevector width

`gradientMps` and `minimizeMps` mirror `gradient` and `minimize` but evaluate on
the MPS backend, so an ansatz whose ground state obeys an area law stays cheap at
widths a statevector cannot hold:

```typescript
import { efficientSU2, minimizeMps } from '@kirkelliott/ket'

const ansatz = efficientSU2(12, 2)
const { energy, params } = minimizeMps(
  ansatz, heisenbergH, Array(ansatz.paramCount).fill(0.1),
  { lr: 0.12, steps: 80 }
)
```

Two circuit methods support the same workflow. `expectMps(hamiltonian, options?)`
returns `{ energy, truncated }` — the Pauli expectation value on MPS, with a flag
telling you whether truncation made it approximate. `bondEntropies(options?)`
returns the von Neumann entropy at every bond, which is the entanglement profile
across the chain:

```typescript
const ghz = new Circuit(6).h(0).cnot(0,1).cnot(1,2).cnot(2,3).cnot(3,4).cnot(4,5)

ghz.bondEntropies()  // [1, 1, 1, 1, 1] — every cut splits one shared bit
ghz.expectMps([{ coeff: 1, ops: 'ZZIIII' }])  // { energy: 1, truncated: false }
```

A product state is flat at zero; GHZ is flat at exactly 1 bit. A circuit whose
entropy grows toward the middle of the chain is one MPS will struggle with.

`PauliOp` supports full complex-coefficient arithmetic: `.add()`, `.scale()`, `.mul()` (with phase tracking), and `.commutator()`. `.toTerms()` converts back to `PauliTerm[]` for use with `vqe()`, `gradient()`, and `minimize()`, and throws if the operator is not Hermitian.

`trotter(n, hamiltonian, t, steps?, order?)` implements the Lie–Trotter product formula (`order=1`) and the symmetric Trotter–Suzuki decomposition (`order=2`). Error scales as O(t²/r) for order 1 and O(t³/r²) for order 2.

`qaoa(n, edges, gamma, beta)` builds the QAOA circuit (Farhi et al. 2014) for the Max-Cut problem. Each layer applies a cost unitary (ZZ rotation per edge) and a mixer unitary (Rx per qubit). `maxCutHamiltonian(n, edges)` returns the corresponding Pauli-string Hamiltonian for `vqe` to evaluate the expected cut value exactly.

QAOA p=1 on a 4-cycle — the two optimal bipartitions tower over all 16 possible outcomes:

![QAOA Max-Cut result](examples/svg/maxcut.svg)

### Shor's algorithm — Beauregard gate-decomposed circuit

`shorBeauregard(N, opts?)` implements Shor's factoring algorithm via the Beauregard (2003)
gate-decomposed circuit. Unlike the oracle-based demo, this decomposes the entire
modular exponentiation into primitive CNOT/Toffoli/phase gates — no classical oracle, no
exponentially large unitary matrix. The circuit uses O(n³) gates total where n = ⌈log₂ N⌉.

```typescript
import { shorBeauregard } from '@kirkelliott/ket'

const r = shorBeauregard(15n, { a: 7n })

r.factors  // [5n, 3n]     — 15 = 5 × 3
r.period   // 4n           — 7⁴ ≡ 1 (mod 15), recovered by QPE
r.method   // 'quantum'    — the circuit really ran (see below)
r.qubits   // 19           — full gate-level circuit
```

Nineteen qubits, complete modular arithmetic, **0.4 seconds**, zero dependencies.

```typescript
// Full result object
r.factors  // [bigint, bigint] | undefined — undefined if no factor found
r.factor   // one non-trivial factor | undefined
r.a        // base used for order-finding
r.period   // r such that a^r ≡ 1 (mod N) | undefined
r.method   // 'quantum' | 'classical-even' | 'classical-gcd' | undefined
r.failure  // 'bad-base' | 'no-period' — set only when the run failed
r.attempts // number of QPE runs before success
r.qubits   // total qubit count (precision + 2n + 2)
```

The stack is: Draper QFT adder (φADD) → controlled modular adder (φADD_mod) → controlled
modular multiplier (U_a) → quantum phase estimation. QPE uses 2n+1 counting qubits;
continued fractions extracts the period from the measurement.

#### Check `method` before you believe it

Shor's algorithm tries cheap classical shortcuts before reaching for a quantum
computer, and at these sizes they hit *often*. An even N returns immediately; a
random base that happens to share a factor with N is resolved by `gcd` alone.
Both give correct factors having never built a circuit:

```typescript
shorBeauregard(15n, { a: 3n }).method   // 'classical-gcd'  — gcd(3,15)=3, no circuit
shorBeauregard(14n).method              // 'classical-even' — no circuit
shorBeauregard(15n, { a: 7n }).method   // 'quantum'        — QPE actually ran
```

So `factor(15n)` on its own is not evidence of quantum factoring — roughly half
the random bases short-circuit classically. Assert on `method === 'quantum'` in
demos and benchmarks.

Not every coprime base works either, and that is the algorithm, not a bug. For
N=21, base 5 has period 6 but 5³ ≡ −1 (mod 21), which yields only trivial
factors. ket detects this and stops immediately rather than retrying a base that
cannot succeed:

```typescript
const bad = shorBeauregard(21n, { a: 5n })
bad.factors  // undefined
bad.failure  // 'bad-base'
bad.attempts // 1 — bailed out instead of burning all 20 retries
```

Leave `a` unset and it draws fresh bases until one works. Pass `seed` to make the
whole run reproducible, base selection included.

**Classical helpers** used internally, also exported for direct use:

```typescript
import { modPow, modInverse, gcd, continuedFractions, phiAdd, applyQft, applyIqft } from '@kirkelliott/ket'

modPow(7n, 4n, 15n)          // 1n   (7^4 mod 15 = 2401 mod 15 = 1)
modInverse(7n, 15n)           // 13n  (7·13 = 91 = 6·15 + 1)
gcd(12n, 8n)                  // 4n
continuedFractions(13n, 64n)  // convergents of 13/64 = [0, 4, 1, 3, ...]

// Build a QFT-basis adder manually
const c = new Circuit(4)
applyQft(c, 4, 0)         // QFT in place
phiAdd(c, 4, 5n, 0)       // add constant 5 in QFT basis
applyIqft(c, 4, 0)        // iQFT
```

**MPS simulation note**: The full QPE circuit generates significant intermediate entanglement
from the controlled-U_a gates. Empirically measured peak bond dimension χ:

| N    | n (bits) | qubits | peak χ | time    |
|------|----------|--------|--------|---------|
| 15   | 4        | 19     | 4      | 0.4s    |
| 21   | 5        | 23     | ~27    | 3.0s    |
| 33   | 6        | 27     | ~59    | 22.4s   |
| 35   | 6        | 27     | ~44    | 29.0s   |
| 77   | 7        | 31     | ~143   | ~907s   |

Read those χ values as approximate. Peak χ counts the singular values above the truncation
cutoff, and with the default `truncErr: 0` that cutoff is an absolute 1e-14. A QPE circuit
leaves a long tail of singular values down in that numerical-noise band, so the count moves
with floating-point details — the same circuit measures χ=27 on arm64 and χ=29 on x86, and
raising the cutoff to a relative 1e-12 drops it to 18. Only N=15, at χ=4, is insensitive to
the choice. The trend is the robust part, not the digits.

χ grows super-linearly with n — exact MPS simulation is **not** more efficient than
statevector for this circuit. `shorBeauregard` uses the MPS backend because the circuit
needs 4n+3 qubits, which exceeds statevector capacity for n ≥ 7. Simulation time scales
with χ, not qubit count.

**This does not scale to cryptographic sizes, and no classical simulator does.** N ≤ 35
runs in seconds; N=77 takes about 15 minutes. Factoring an RSA modulus this way is not
a matter of waiting longer. What ket gives you is the real circuit — every gate, exactly
simulated — at sizes where you can inspect and learn from it.

### Inspecting the Shor circuit directly

`shorBeauregard` runs the circuit and post-processes the measured phase.
`shorCircuit` returns the circuit itself, so it can be counted, drawn, or run
under noise without reimplementing the construction:

```typescript
import { shorCircuit, DEVICES } from '@kirkelliott/ket'

const c = shorCircuit(33n, 5n)          // modulus, base
c.qubits                                 // 27
c.depth()                                // 25868

const { oneQubit, twoQubit, byName } = c.gateCounts()
// 6163 single-qubit, 31260 two-qubit-equivalent (a Toffoli counts as 6)

// Chance of an error-free run. Two-qubit error dominates, but both terms count:
// dropping the single-qubit factor here overstates the odds by about 5x.
const { p1, p2 } = DEVICES['forte-1'].noise
(1 - p1) ** oneQubit * (1 - p2) ** twoQubit   // 8.45e-69
```

`gateCounts()` expands subcircuits first and excludes barriers, measurement and
reset, so the counts describe what would actually execute. `byName` gives finer
accounting — `byName['t']` is the T-count, which is what the stabilizer-rank
backend's cost depends on.

### Building blocks of the Beauregard circuit

The layers `shorCircuit` is assembled from are exported so a circuit can be
inspected, unit-tested, or rebuilt with different arithmetic. Each acts in place
on a `Circuit` and returns a new one.

| Function | Layer |
|---|---|
| `phiAdd(c, n, a, q0)` | add the constant `a` in the Fourier basis |
| `phiAddMod(c, n, a, N, ...)` | the same addition reduced mod `N` |
| `ccPhiAddMod(c, n, a, N, ctrl1, ctrl2, ...)` | doubly-controlled modular addition |
| `cMultModAdd(c, n, a, N, ctrl, ...)` | controlled modular multiply-accumulate |
| `beauregardU(c, n, a, aInv, N, ctrl, x, acc, anc)` | the controlled `U_a` used by phase estimation |
| `applyQft(c, n, off)` / `applyIqft(c, n, off)` | in-place QFT over a sub-register |

`examples/node/dlog-shor.js` builds the two-register discrete-logarithm circuit
directly from `beauregardU` and `applyIqft`, which is the shortest example of
composing them into something `shorCircuit` does not provide.

### Stabilizer internals

`StabilizerCH` is the phase-sensitive CH-form simulator underneath
`runStabilizerRank` (Bravyi et al., *Quantum* **3**, 181, 2019). Unlike the CHP
tableau in `CliffordSim` it tracks global phase, so `amplitude()` is exact rather
than correct-up-to-phase — which is what makes summing over Clifford terms valid.
`randomEquatorial(n, rand)` draws the random equatorial stabilizer states used by
the norm estimator. `bytesPerTerm(qubits)` gives the measured per-term memory
cost that `maxTGates` inverts.

### Worked examples: recovering a key end to end

Two example programs use the above to make a complete, self-checking argument.
Both are written as lab reports: numbered sections, an assertion at every stage,
and a non-zero exit on any failed check.

```bash
node examples/node/rsa-shor.js     # 27 qubits, ~57 s
node examples/node/dlog-shor.js    # 20 qubits, ~16 s
```

`rsa-shor.js` builds a genuine RSA keypair, encrypts a message, and then recovers
the private exponent from the public parameters alone. It enumerates every base
coprime to N and logs each outcome, **discarding runs that terminate classically**
via `gcd` or an even modulus, so `method === 'quantum'` is asserted rather than
assumed. It then verifies a^r ≡ 1 mod N, that the factors multiply back, that the
recovered d matches the generated one, and that decryption returns the plaintext.

`dlog-shor.js` solves the discrete logarithm underlying ECDSA. Given g of known
order r and h = g^x, it prepares |a⟩|b⟩|1⟩ → |a⟩|b⟩|gᵃhᵇ⟩ and applies an inverse
QFT to each exponent register; the result is supported entirely on pairs with
β ≡ xα (mod r), so x = βα⁻¹. The script **measures** the fraction of amplitude
violating that relation and asserts it is zero, which checks the circuit rather
than the derivation.

g must have power-of-two order so the transform is exact. That constrains g, not
p — every group of order 2^k·m contains an element of order 2^k — and if the
supplied generator is unsuitable the script computes a valid one and reports it.
Arbitrary orders would need an approximate transform with continued-fraction
post-processing, which is deliberately out of scope.

Both are parameterised (`RSA_P`, `RSA_Q`, `RSA_E`, `RSA_M`; `DLOG_P`, `DLOG_G`,
`DLOG_X`) and both close with a hardware-feasibility section: gate counts against
published per-gate error rates for IonQ systems currently accepting jobs. Setting
`RSA_NOISY_SHOTS` or `DLOG_NOISY_SHOTS` confirms the estimate by sampling noisy
trajectories, which takes minutes and is therefore opt-in.

The discrete-log figure is worth stating: noiseless, 100% of outcomes land on the
predicted support; under `forte-1` depolarizing noise, 6.3% — exactly the uniform
baseline for a 16-element group. The signal is not degraded but erased, which is
the concrete form of the error-correction argument.

## Visualization

### ASCII diagram

`circuit.draw()` renders a text-mode diagram suitable for terminals, notebooks, and log output.

```
q0: ─H──●──M─
         │
q1: ─────⊕──M─
```

Gates on non-conflicting qubits share a column. Parameterized gates display their angle: `Rx(π/4)`, `XX(π/2)`. Named sub-circuit gates show their registered name.

### SVG export

`circuit.toSVG()` returns a self-contained SVG string with no external fonts or stylesheets. The layout matches `draw()`: same column packing, rounded gate boxes, filled control dots, circle-cross CNOT targets, and × SWAP marks. Safe to write directly to `.svg` files or inline in HTML.

Bell state:

![Bell circuit](examples/svg/bell.svg)

4-qubit QFT:

![QFT circuit](examples/svg/qft4.svg)

### Measurement histogram

`result.toSVG()` returns a self-contained SVG bar chart of measurement outcomes — same visual style as the QAOA Max-Cut diagram above. Bars are sorted by bitstring; dominant peaks (≥ 80 % of the max probability) are highlighted in blue with a percentage label.

Bell state (1024 shots):

![Bell histogram](examples/svg/bell_histogram.svg)

QAOA Max-Cut standalone histogram:

![QAOA histogram](examples/svg/qaoa_histogram.svg)

```typescript
import fs from 'fs'
import { Circuit, qaoa } from '@kirkelliott/ket'

// Bell state
const result = new Circuit(2).h(0).cnot(0, 1)
  .creg('out', 2).measure(0, 'out', 0).measure(1, 'out', 1)
  .run({ shots: 1024, seed: 42 })
fs.writeFileSync('bell.svg', result.toSVG())

// Custom title and explicit highlight list
result.toSVG({ title: 'my experiment', highlight: ['00', '11'] })
```

### Bloch sphere

`circuit.blochSphere(q)` returns a self-contained SVG showing the single-qubit state for qubit `q` as an arrow on the Bloch sphere. Internally uses `blochAngles(q)`, which partial-traces the statevector over all other qubits.

|0⟩ state (north pole) and |+⟩ = H|0⟩ state (equator):

![Bloch |0⟩](examples/svg/bloch_zero.svg) ![Bloch |+⟩](examples/svg/bloch_plus.svg)

```typescript
// Write to file
import fs from 'fs'
fs.writeFileSync('state.svg', circuit.blochSphere(0))
```

### LaTeX

`circuit.toLatex()` emits a `quantikz` LaTeX environment with `\frac{\pi}{n}` angle formatting, proper `\ctrl{}`, `\targ{}`, `\swap{}`, `\gate[2]{}`, and `\meter{}` commands.

## Parametric circuits

Gate angle parameters can be symbolic strings, deferred until `.bind()` is called. This lets you build an ansatz once and evaluate it at many parameter values without reconstructing the circuit.

```typescript
import { Circuit } from '@kirkelliott/ket'

// Build once — 'theta' and 'phi' are symbolic
const ansatz = new Circuit(2)
  .ry('theta', 0)
  .rz('phi', 0)
  .cnot(0, 1)

ansatz.params  // ['phi', 'theta'] — sorted unbound names

// Evaluate at a specific point
const bound = ansatz.bind({ theta: Math.PI / 4, phi: 0.1 })
bound.params      // []
bound.statevector()  // runs normally

// VQE sweep — reuse the same ansatz object
for (const theta of [0, 0.1, 0.2, Math.PI / 4]) {
  const energy = vqe(ansatz.bind({ theta, phi: 0 }), hamiltonian)
}
```

Any gate with angle parameters accepts `number | string` for each angle: `rx`, `ry`, `rz`, `vz`, `u1`, `p`, `u2`, `u3`, `gpi`, `gpi2`, `xx`, `yy`, `zz`, `xy`, `ms`, `crx`, `cry`, `crz`, `cu1`, `cu2`, `cu3`.

Calling `statevector()`, `run()`, `toQASM()`, or any export on a circuit with unbound parameters throws a `TypeError` listing the missing names.

## Circuit composition

`circuit.compose(other)` concatenates two circuits of the same width, returning a new immutable circuit. Classical registers are merged (same name → larger size wins). Custom gate definitions are merged (`this` takes precedence on name conflicts).

```typescript
const state_prep = new Circuit(2).h(0).cnot(0, 1)
const rotation   = new Circuit(2).rz(Math.PI / 4, 0).rz(Math.PI / 4, 1)

const full = state_prep.compose(rotation)
// equivalent to: new Circuit(2).h(0).cnot(0,1).rz(π/4,0).rz(π/4,1)

full.statevector()   // runs the combined circuit
```

Mismatched qubit counts throw `TypeError` immediately.

## State inspection

```typescript
const circuit = new Circuit(2).h(0).cnot(0, 1)

circuit.statevector()           // Map<bigint, Complex> — full sparse amplitude map
circuit.amplitude('11')         // Complex — amplitude of |11⟩
circuit.probability('11')       // number — |amplitude|²
circuit.exactProbs()            // { bitstring: probability } — no sampling, no variance
circuit.marginals()             // [P(q0=1), P(q1=1)]
circuit.stateAsString()         // '0.7071|00⟩ + 0.7071|11⟩'
circuit.stateAsArray()          // [{ bitstring, re, im, prob, phase }, ...] sorted by prob
circuit.blochAngles(0)          // { theta, phi } via partial trace
circuit.expectation('ZZ')       // number — ⟨ψ|P|ψ⟩ for a Pauli string P
circuit.circuitMatrix()         // Complex[][] — the circuit's own 2ⁿ×2ⁿ unitary
```

`stateAsArray()` returns one entry per basis state with non-negligible amplitude (|a|² ≥ 1e-10), sorted by probability descending. Each entry carries the real and imaginary parts, the probability, and the phase angle `atan2(im, re)`. Throws on circuits with measurements or unbound parameters.

`circuitMatrix()` materializes the whole circuit as a single unitary, which is
useful for verifying a decomposition against a target gate. It requires a pure
circuit — `measure`, `reset`, or `if` throw `TypeError` — and refuses sizes whose
matrix would not fit in memory, throwing `RangeError` rather than attempting the
allocation.

## Classical control and named gates

```typescript
import { Circuit } from '@kirkelliott/ket'

// Classical registers, measurement, and reset
const c = new Circuit(2)
  .creg('out', 2)
  .h(0)
  .cnot(0, 1)
  .measure(0, 'out', 0)
  .measure(1, 'out', 1)
  .reset(0)

// Conditional gate application
const teleport = new Circuit(3)
  .if('out', 1, q => q.x(2))
  .if('out', 2, q => q.z(2))

// Named sub-circuit gates
const bell = new Circuit(2).h(0).cnot(0, 1)
const main = new Circuit(4)
  .defineGate('bell', bell)
  .gate('bell', 0, 1)
  .gate('bell', 2, 3)

main.decompose()  // inline all named gates back to primitives
```

## Noise models

All three stochastic backends accept a noise configuration with the same interface:

```typescript
// Named device profile (statevector, Clifford, density matrix)
circuit.run({ noise: 'forte-1' })
circuit.runClifford({ shots: 10000, noise: 'forte-1' })
circuit.dm({ noise: 'harmony' })

// Custom noise parameters
circuit.run({ noise: { p1: 0.001, p2: 0.005, pMeas: 0.004 } })
circuit.runClifford({ shots: 10000, noise: { p1: 0.001, p2: 0.005 } })
```

`p1` — single-qubit depolarizing error probability per gate. `p2` — two-qubit depolarizing probability. `pMeas` — bit-flip probability on each measured bit (SPAM error).

Named profiles cover all devices in the [device table](#device-targeting) — IonQ, IBM, and Quantinuum. The density matrix backend applies exact per-gate depolarizing channels (no Monte Carlo sampling). Noiseless circuits take the fast path — zero overhead.

### T1 / T2 relaxation

Beyond depolarizing noise, `gamma` applies amplitude damping (T1 relaxation) and
`lambda` pure dephasing (T2 beyond T1). Both are per-single-qubit-gate
probabilities, applied to every qubit after every gate, so convert from
coherence times:

```typescript
const tGate = 200e-9, T1 = 50e-6, T2 = 30e-6

circuit.run({ shots: 4096, noise: {
  gamma:  1 - Math.exp(-tGate / T1),
  lambda: 1 - Math.exp(-2 * tGate * (1 / T2 - 1 / (2 * T1))),
}})
```

### Custom Kraus channels

`kraus1` and `kraus2` take arbitrary Kraus operators — 2×2 matrices applied
after each single-qubit gate, 4×4 after each two-qubit gate. They must be
trace-preserving (Σ_k K_k† K_k = I); one operator is sampled per shot per gate.

```typescript
import { c } from '@kirkelliott/ket'

// Amplitude damping written out by hand, γ = 0.01
const g = 0.01
const K0 = [[c(1), c(0)], [c(0), c(Math.sqrt(1 - g))]]
const K1 = [[c(0), c(Math.sqrt(g))], [c(0), c(0)]]

circuit.run({ shots: 4096, noise: { kraus1: [K0, K1] } })
```

`kraus1` never applies after two-qubit gates and `kraus2` never after
single-qubit ones — neither leaks into the other. Both work in `run()` and
`dm()`, but **not** in `runMps()`.

### Readout error mitigation

Given a known symmetric bit-flip probability, `Distribution.mitigateReadout(p)`
inverts the readout confusion matrix and returns a corrected distribution:

```typescript
const noisy = circuit.run({ shots: 8192, noise: { pMeas: 0.02 } })
const fixed = noisy.mitigateReadout(0.02)
```

## Serialization

```typescript
// Lossless round-trip through JSON
const json = circuit.toJSON()
const restored = Circuit.fromJSON(json)

// Or pass a parsed object
const restored2 = Circuit.fromJSON(JSON.parse(json))
```

All operation types are preserved: gates, measure, reset, if, and named sub-circuits. Gate matrices are reconstructed from metadata on load.

## Performance

ket matches its representation to the circuit instead of committing to one, so
the same API is efficient across shapes that usually need different tools.

A statevector starts as a sparse map and promotes itself to a flat
`Float64Array` once it is more than 1/8 full. A GHZ state holds two non-zero
amplitudes at any width, so it stays sparse and never touches the other million
slots; a depth-4 random circuit fills every amplitude in its first layer, so it
moves to the dense kernel once and runs the rest with no allocation at all. The
density matrix does the same at 1/32 fill, and MPS bond dimension grows on
demand rather than being capped up front.

Measured on Node 24 / Apple silicon, best of 5:

| Circuit | Representation | Time |
|---|---|---|
| GHZ-20, statevector | sparse | 5us |
| QFT-16, statevector | dense | 10.5ms |
| random-16 depth 4, statevector | dense | 15.7ms |
| GHZ-50, MPS chi=2 | tensor network | milliseconds |
| GHZ-127, MPS, 1024 shots | tensor network | 10ms |
| 12-qubit noisy run, 1024 shots | dense | 0.69s |

None of this needs a flag — the thresholds are defaults, adjustable per call via
the `dense` option when you want to trade memory against speed.

### Continuous benchmarks

<!-- benchmark:start -->

Populated by CI on every push to main — run `node benchmark/run.ts | node benchmark/update-readme.ts` to regenerate locally.

<!-- benchmark:end -->

## How it works

The statevector backend is a hybrid of two representations, and switches between them on its own.

It starts sparse: a `Map<bigint, Complex>` holding only basis states with non-zero amplitude. Gate application iterates the entries present rather than allocating a full transformation matrix, so a GHZ chain costs two amplitudes per gate no matter how wide it is. BigInt keys eliminate the 32-bit overflow that silently corrupts state at qubit index 31 in integer-based simulators.

That representation stops paying once a state densifies — every gate then rebuilds a `Map`, allocates a `Set<bigint>` of visited keys, and boxes one `{re, im}` object per amplitude. So when the support exceeds 2ⁿ/8, the state is promoted once to a `DenseState`: a single contiguous `Float64Array` with real and imaginary parts interleaved, mutated in place. A gate becomes 2ⁿ unboxed f64 operations with no allocation, which V8 keeps in registers. Measured on a depth-4 random 16-qubit circuit, that is the difference between 1,254 ms and 16.5 ms.

Promotion is one-way — a dense state is never demoted, since the fill test would cost a full scan per gate to avoid work the dense kernel is already fast at. It is also capped by default at 24 qubits (2²⁴ amplitudes × 16 bytes = 256 MiB); above that the sparse path stays in charge regardless of fill, because a dense buffer would be a worse problem than a slow one. Both the fill fraction and the ceiling are adjustable per call — see [Tuning sparse → dense promotion](#tuning-sparse--dense-promotion). Permutation gates (CNOT, SWAP, Toffoli, CSWAP) skip the fill test entirely — they cannot change the size of the support.

The two kernels are differentially tested against each other in `src/hybrid.test.ts`, gate by gate across every qubit ordering, so which one runs is never observable in a result.

`run()` adds a second decision on top. A circuit whose measurements are all *terminal* — no `reset`, no `if`, and no gate touching a qubit after it is measured — does not need re-simulating per shot: measuring in the computational basis is a dephasing channel, and dephasing a qubit nothing else will touch cannot change the joint outcome distribution. Such a circuit is built once and sampled, with each measurement reading a bit straight out of the sampled index. Anything else (noise, mid-circuit feedback, a gate on a measured qubit) still runs one full simulation per shot, because there the later gates genuinely depend on the collapse.

`runMps()` applies the same rule, so an MPS circuit written that way is built once too — a 30-qubit depth-4 circuit went from 0.18s to 0.012s for 1024 shots, and 20,000 shots now costs 0.073s where the per-shot path scaled linearly.

This matters more than it sounds, because "apply gates, then measure everything" is how most circuits are written. On a depth-4 random 12-qubit circuit with all twelve qubits measured, 200 shots went from 8,652 ms to 5 ms; 20,000 shots now costs 8.4 ms, where the per-shot path scaled linearly with shot count. `src/terminal-measure.test.ts` pins both halves: that ineligible circuits keep the per-shot path, and that both paths agree on the ones that could take either.

The circuits that genuinely need one simulation per shot — noise, mid-circuit feedback — run on the hybrid representation too. Projection, renormalisation, amplitude and phase damping, custom Kraus channels and final sampling all have dense implementations alongside the sparse ones, so a noisy circuit densifies exactly as a pure one does. On the same 12-qubit circuit that is 42 ms per shot down to 0.9 ms, which takes the default 1024-shot noisy run from roughly 43 s to 0.69 s.

Sampling deserves one note: both representations walk basis indices in ascending order, so a given RNG draw selects the same outcome whichever one is live. Promotion cannot change a seeded result, and `src/hybrid.test.ts` asserts that directly over 200 draws.

The MPS backend represents state as a chain of tensors with an adaptive bond dimension χ. Memory is O(n·χ²) instead of O(2ⁿ), which makes circuits with limited entanglement — like GHZ, QFT, and most hardware-native gate sequences — practical at 50–100+ qubits. The bond dimension starts at `maxBond` (default 64) and grows automatically whenever a gate would require a larger χ, so simulation is always exact up to floating-point regardless of the starting value. For circuits with genuinely unbounded entanglement (deep random circuits), χ grows exponentially and memory eventually becomes the bottleneck — use `truncErr` to trade accuracy for a bounded χ when that matters. Each tensor is stored as a single contiguous `Float64Array` (interleaved re/im), eliminating per-element heap allocations and allowing V8 to JIT-compile the inner contraction loops as unboxed f64 operations. Mid-circuit measurement (`measure`), qubit reset (`reset`), and classical conditioning (`if`) are fully supported: measurement projects the site tensor in-place using the Vidal canonical form bond lambdas, restoring a normalised MPS without any SVD — O(χ²) per measurement. Each shot in a mid-circuit circuit runs as an independent trajectory with its own classical register state.

The density matrix backend tracks the full ρ = |ψ⟩⟨ψ| matrix as a sparse map, applying exact per-gate depolarizing channels without Monte Carlo sampling. Noiseless circuits take the fast path — zero overhead compared to the statevector backend.

The Clifford stabilizer backend implements the CHP algorithm (Aaronson & Gottesman 2004) with a bit-packed binary tableau. Each row stores 32 stabilizer bits per array element; row multiplication uses vectorized popcount for phase accumulation and word-level XOR for tableau update, both O(n/32). Gate application (H, S, CNOT, etc.) is O(n) over the 2n tableau rows. Measurement is O(n²) worst-case per qubit. The gate set is exactly the Clifford group — any non-Clifford gate raises a TypeError.

## Quantum error correction

`CliffordSim` is exported directly for researchers who need more control than `runClifford` provides — custom decoders, syndrome extraction, mid-circuit readout, threshold curve generation.

```typescript
import { CliffordSim } from '@kirkelliott/ket'

// Bell state as a minimal 2-qubit code: stabilizers XX and ZZ
const sim = new CliffordSim(2)
sim.h(0); sim.cnot(0, 1)

sim.stabilizerGenerators()  // → ['+XX', '+ZZ']

// Inject a bit-flip error on qubit 0
sim.x(0)

// Syndrome: ZZ flips sign, revealing the X error
sim.stabilizerGenerators()  // → ['+XX', '-ZZ']

// Measure qubit 0 to collapse the syndrome
const outcome = sim.measure(0, Math.random())
```

`stabilizerGenerators()` returns the current stabilizer generators as signed Pauli strings (`'+'` or `'-'` prefix, then one character per qubit: `I`, `X`, `Y`, `Z`). The sign encodes the ±1 eigenvalue. A sign flip on a generator is a syndrome bit — it identifies which error occurred without revealing the logical state. Call it after syndrome measurement to extract the full stabilizer state for soft-decision decoding.

For threshold curves, pass `noise` to `runClifford` and sweep the error rate:

ket ships no code constructors and no decoder, so a threshold study means building
the encoding circuit yourself. A three-qubit repetition code, which corrects any
single bit flip, is short enough to show the shape:

```typescript
import { Circuit } from '@kirkelliott/ket'

// Encode |psi> as |psi psi psi>, then sweep the physical error rate.
const encoded = new Circuit(3).cnot(0, 1).cnot(0, 2)

for (const p2 of [0.001, 0.005, 0.01, 0.02, 0.05]) {
  const d = encoded.runClifford({ shots: 10000, noise: { p2 } })
  console.log(p2, d.probs)   // weight away from the codespace is the logical error rate
}
```

A surface-code threshold curve needs the same loop over a stabilizer-measurement
circuit plus a matching decoder — the decoder being the part ket does not
provide. `stabilizerGenerators()` supplies the syndrome; pairing it with a
minimum-weight or union-find decoder is left to the caller.

## Testing

2,452 tests, ~40s (the Beauregard Shor's suite dominates). Run with:

```bash
npm test
```

The suite covers: analytic correctness (known complex amplitudes, not just "doesn't crash"), gate invertibility (U†U = I), math primitive unit tests (`add`, `mul`, `conj`, `norm2`, etc.), qubit index bounds checking, algorithm output correctness (QFT phase amplitudes, Grover, QPE, VQE, Pauli expectation values, gradient analytic match, minimize convergence), BigInt correctness at qubit indices 30/31/40, Clifford word-boundary correctness at n=33, probability normalization invariants, backend consistency (statevector vs density matrix), JSON round-trips for all 13 op kinds, serializer contract matrix (every op kind × every export format), and full import/export round-trips for all 14 supported formats.
