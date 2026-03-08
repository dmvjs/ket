# ket — Development Roadmap

A living document charting the path from a correct, minimal core to a complete, production-grade quantum circuit library for JavaScript and the browser.

---

## What exists today

**Core architecture**
- Sparse statevector over `Map<bigint, Complex>` — no 32-bit overflow at any qubit count
- IonQ QIS gate names as the primary API (`h`, `cnot`, `rx`, `xx`, …)
- Immutable circuit builder — every gate method returns a new `Circuit`
- Seeded PRNG for reproducible shot sampling
- `Distribution` with `probs`, `histogram`, `entropy`, `most`, `render()`
- ESM-native, TypeScript strict, zero runtime dependencies

**Gates implemented**
`h` `x` `y` `z` `s` `si` `t` `ti` `v` `vi` `rx` `ry` `rz` `cnot` `swap`
`r2` `r4` `r8` `u1` `u2` `u3`
`xx` `yy` `zz` `xy` `iswap` `srswap`
`cx` `cy` `cz` `ch` `crx` `cry` `crz` `cr2` `cr4` `cr8` `cu1` `cu3` `cs` `ct` `csdg` `ctdg`
`ccx` `cswap`
`gpi` `gpi2` `ms`

**Test suite (583 tests, ~200ms)**
- All single-qubit gates and their inverses
- All four Bell states
- Deutsch-Jozsa (constant and balanced oracle)
- Bernstein-Vazirani (hidden bitstring recovery)
- GHZ at n = 3, 5, 8, 12, 16, 20 qubits
- 8-qubit uniform superposition (all 256 outcomes present)
- Entropy equals n bits for H⊗n states (n = 1…4)
- BigInt correctness at qubits 30, 31, 40 — exactly where 32-bit operations fail
- Phase rotation and OpenQASM basis gates (U1/U2/U3)
- Two-qubit interaction gates (XX/YY/ZZ/XY/iSWAP/√iSWAP)
- Full controlled gate family with QFT integration test
- Toffoli and Fredkin truth tables, phase kickback, swap test
- Native IonQ gates: GPI Ramsey fringe, GPI2 inverse pair, MS Bell states
- Quantum teleportation: |0⟩, |1⟩, |+⟩, Rx(π/3)|0⟩
- OpenQASM 2.0: gate name mapping, angle notation, round-trip statevector fidelity
- ASCII `draw()`: column layout, all gate symbols, parameterized labels, multi-qubit connectors
- `toSVG()`: self-contained SVG, gate boxes, control dots, CNOT/SWAP marks, qubit labels
- `blochAngles(q)`: |0⟩/|1⟩ poles, equatorial states, mixed entangled qubits
- Export targets: Qiskit, Cirq, Q#, pyQuil — gate name mapping, angle formatting, throw-on-unsupported
- Noise models: depolarizing (p1/p2) + SPAM (pMeas); named device profiles; zero-overhead fast path preserved
- MPS backend: GHZ-50, BV-40, product-state-50, non-adjacent gates, statevector cross-check
- Algorithms: QFT n=2..4, IQFT round-trip, Grover n=2/3/4, QPE (T/S gate), VQE single/multi-qubit Hamiltonians
- `id` gate: no-op simulation, round-trip QASM, per-target export, `barrier` silently skipped in fromQASM
- `marginals()`: per-qubit P(q=1) from statevector; tested against known states and Bell/GHZ
- `cu2(φ,λ)`: completes cu1/cu2/cu3 family; QASM round-trip, Qiskit export with params
- `stateAsString()`: human-readable amplitude listing; handles real/imaginary/complex, omits near-zero terms
- `defineGate(name, sub)` / `gate(name, ...qubits)` / `decompose()`: named sub-circuit gates with qubit remapping; nested composition; auto-flattened for simulation; serialization via decompose()
- `toJSON()` / `Circuit.fromJSON(json)`: lossless versioned JSON round-trip; gate matrices reconstructed from meta on load; supports all op types including measure/reset/if/subcircuit; accepts string or parsed object

---

## Phase 1 — Gate completeness ✓

### 1a. Phase rotation gates ✓
`r2` (Rz(π/2) = S), `r4` (Rz(π/4) = T), `r8` (Rz(π/8))

### 1b. Parameterized unitaries ✓
`u1(λ)`, `u2(φ, λ)`, `u3(θ, φ, λ)` — OpenQASM basis gates used by IBM circuits.

### 1c. Two-qubit interaction gates ✓
`xx(θ)`, `yy(θ)`, `zz(θ)`, `xy(θ)`, `iswap`, `srswap`

### 1d. Controlled single-qubit gates ✓
`cx`/`cy`/`cz`/`ch`, `crx`/`cry`/`crz`, `cu1`/`cu3`, `cs`/`ct`/`csdg`/`ctdg`, `cr2`/`cr4`/`cr8`

### 1e. Three-qubit gates ✓
`ccx` (Toffoli), `cswap` (Fredkin)

### 1f. Native IonQ gates ✓
`gpi(φ)`, `gpi2(φ)`, `ms(φ₀, φ₁)`

---

## Phase 2 — Measurement and classical control ✓

### 2a. Classical registers ✓
`Circuit.creg(name, size)`, `Circuit.measure(qubit, creg, bit)`, `Circuit.reset(qubit)`

### 2b. Conditional gates ✓
`Circuit.if(creg, value, build)` — apply a gate sequence only if a classical register holds a value.

### 2c. Statevector inspection API ✓
`circuit.statevector()`, `circuit.amplitude(bitstring)`, `circuit.probability(bitstring)`

---

## Phase 3 — Import and export ✓

### 3a. IonQ JSON (import + export) ✓
`Circuit.fromIonQ(json)`, `circuit.toIonQ()`

### 3b. OpenQASM 2.0 (import + export) ✓
`Circuit.fromQASM(string)`, `circuit.toQASM()`

### 3c. Export targets ✓
| Target | Method | Use case |
|---|---|---|
| Qiskit (Python) | `circuit.toQiskit()` | IBM Quantum, Aer simulator |
| Cirq (Python) | `circuit.toCirq()` | Google Quantum AI |
| Q# | `circuit.toQSharp()` | Microsoft Azure Quantum |
| pyQuil | `circuit.toPyQuil()` | Rigetti hardware |
| OpenQASM 2.0 | `circuit.toQASM()` | Universal interchange format |
| IonQ JSON | `circuit.toIonQ()` | IonQ Cloud (via qsim or direct) |

---

## Phase 4 — Visualization ✓

### 4a. ASCII circuit diagram ✓
`circuit.draw()` — a text-mode circuit diagram for terminal and notebooks.

```
q0: ─H──●─
         │
q1: ─────⊕─
```

- Greedy column-packing: gates on non-conflicting qubits share a column
- Full qubit span blocked per op so vertical wires never overlap adjacent gate boxes
- Gate symbols: `H` `X` `Y` `Z` `S†` `T†` `●` `⊕` `╳` `M` `|0⟩` `Rx(π/4)` `XX(π/2)` …
- Named sub-circuit gates show their registered name

### 4b. SVG export ✓
`circuit.toSVG()` — a self-contained SVG string (no external fonts or stylesheets).

- Same column layout as `draw()`
- Rounded gate boxes, filled control dots, circle-cross CNOT target, × SWAP marks
- Embeds a monospace font stack; safe to write directly to `.svg` files or inline in HTML

### 4c. Bloch sphere coordinates ✓
`circuit.blochAngles(qubit)` — returns `{ theta, phi }` (θ ∈ [0,π], φ ∈ (-π,π]).

- Computes the reduced single-qubit density matrix via partial trace
- Extracts Bloch vector (rx, ry, rz) and converts to spherical coordinates
- Valid for pure product states (exact) and entangled qubits (mixed-state Bloch vector)

---

## Phase 5 — Advanced simulation ✓

### 5a. Noise models ✓
Depolarizing channel and SPAM readout errors matching IonQ's published characterization methodology.
`circuit.run({ noise: 'aria-1' | 'forte-1' | 'harmony' | NoiseParams })` — opt-in per run.
Named device profiles: `aria-1` (p1=0.03%, p2=0.5%, pMeas=0.4%), `forte-1`, `harmony`.

### 5b. Tensor network simulation ✓
`circuit.runMps({ shots, seed, maxBond? })` — MPS backend with configurable bond dimension χ (default 64).
Memory: O(n·χ²·2) vs O(2ⁿ) — GHZ-50 and BV-40 run in milliseconds with χ=2.

### 5c. Shots-free exact probabilities ✓
`circuit.exactProbs()` — returns `Readonly<Record<string, number>>` directly from the statevector; no sampling, no variance.

---

## Phase 6 — Ecosystem

### 6a. Quantum algorithms library ✓
- `qft(n)` / `iqft(n)` — n-qubit Quantum Fourier Transform and its inverse
- `grover(n, oracle, iterations?)` — Grover's search; ancilla-free for n ≤ 3, Barenco staircase for n ≥ 4
- `phaseEstimation(precision, unitary, targetQubits?)` — Quantum Phase Estimation
- `vqe(ansatz, hamiltonian)` — Variational Quantum Eigensolver; exact expectation via Pauli-string rotations

### 6b. npm publication
Publish as `ket` (or `@dmvjs/ket`). Semantic versioning from 0.1.0.

### 6c. Browser bundle
`dist/ket.min.js` — UMD bundle for direct `<script>` inclusion, built with esbuild.

---

## Phase 7 — Gate and API parity

Gaps identified against quantum-circuit (the leading JS quantum simulator), ordered by impact.

### 7a. Missing gates

**`id` — identity gate ✓**
Named no-op that survives round-trips through all import/export formats. `fromQASM` also silently skips `barrier` statements.

**`vz(θ)` — VirtualZ ✓**
A named Rz alias common in superconducting hardware native gate sets (IBM, Rigetti). Functionally identical to `rz`; serializes as `rz` in all targets (no hardware supports vz by name in their interchange formats).

**`cu2(φ, λ)` — controlled U2 ✓**
Completes the `cu1`/`cu2`/`cu3` family. Appears in IBM circuit exports.

**`csrswap` — controlled √SWAP ✓**
Completes the three-qubit gate set alongside `ccx` and `cswap`. If control=1, applies √iSWAP (SrSwap) to the two target qubits. Niche; throws in all serializers (no standard interchange representation).

### 7b. Per-qubit marginal probabilities ✓

`circuit.marginals()` — returns `number[]` where `result[q]` is `P(qubit q = |1⟩)`, summed over all basis states. Shares the pure-circuit restriction of `statevector()`.

### 7c. Reset to |1⟩ ✓

`circuit.reset(q, value?)` — `value` defaults to 0; `reset(q, 1)` resets to |1⟩ (equivalent to `reset(q).x(q)` but first-class API).

### 7d. Statevector pretty-print ✓

`circuit.stateAsString()` — human-readable amplitude listing, e.g. `0.7071|00⟩ + 0.7071|11⟩`.
Handles real, imaginary, and complex coefficients; omits near-zero terms; correct sign joining.

### 7e. Additional export targets

| Target | Method | Use case |
|---|---|---|
| Quil | `circuit.toQuil()` ✓ | Rigetti native format |
| Amazon Braket | `circuit.toBraket()` ✓ | AWS quantum hardware |
| CudaQ | `circuit.toCudaQ()` ✓ | NVIDIA GPU-accelerated simulation |
| TensorFlow Quantum | `circuit.toTFQ()` ✓ | Hybrid quantum-classical ML |
| Quirk JSON | `circuit.toQuirk()` ✓ | Browser-based circuit visualizer |

### 7f. Named sub-circuit gates ✓

`circuit.defineGate(name, subcircuit)` — register a reusable named gate from a `Circuit`.
`circuit.gate(name, ...qubits)` — apply a registered gate with qubit remapping.
`circuit.decompose()` — inline all named gates back to primitives.

Implemented as a new `SubcircuitOp` op type. Simulation auto-flattens subcircuits
(transparent to `.run()`, `.statevector()`, `.runMps()`, etc.). Nested named gates
(gates whose definition uses other named gates) work via recursive expansion.
Serialization requires an explicit `.decompose()` call; all serializers auto-flatten
named gates before emitting. Classical ops (measure/reset/if) inside gate definitions
are rejected at `defineGate()` time.

---

## Test coverage policy

Every feature ships with tests that verify:
1. **Correctness** — known analytic outputs (not just "doesn't crash")
2. **Invertibility** — U†U = I for every unitary gate
3. **Composition** — gate sequences that have known combined effects
4. **Edge cases** — zero rotation angles, large qubit indices, empty circuits

The full suite runs in < 200ms. No test may be slow-by-design; GHZ and other multi-qubit tests are parameterized and capped to keep CI fast.
