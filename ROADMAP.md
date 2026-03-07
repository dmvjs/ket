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

**Test suite (283 tests, ~115ms)**
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

---

## Phase 1 — Gate completeness

Fill out the standard gate set so any published circuit can be expressed.

### 1a. Phase rotation gates
`r2` (Rz(π/2) = S), `r4` (Rz(π/4) = T), `r8` (Rz(π/8))
Already expressible as `rz(Math.PI/N)` — add named aliases for readability and IonQ JSON compatibility.

### 1b. Parameterized unitaries
`u1(λ)`, `u2(φ, λ)`, `u3(θ, φ, λ)` — OpenQASM basis gates used by IBM circuits.

### 1c. Two-qubit interaction gates
`xx(θ)`, `yy(θ)`, `zz(θ)` — IonQ's native all-to-all interaction gates, fundamental to trapped-ion hardware.
`iswap`, `srswap` — common in superconducting and trapped-ion literature.
`xy(θ)` — XY interaction, used in quantum chemistry and QAOA.

### 1d. Controlled single-qubit gates
`cx`/`cy`/`cz`/`ch` — the standard controlled family.
`crx(θ)`, `cry(θ)`, `crz(θ)` — parameterized controlled rotations.
`cu1`, `cu3` — controlled parameterized unitaries.
`cs`, `ct`, `csdg`, `ctdg` — controlled phase gates.

### 1e. Three-qubit gates
`ccx` (Toffoli) — universal for classical reversible computation.
`cswap` (Fredkin) — controlled swap.

### 1f. Native IonQ gates ✓
`gpi(φ)`, `gpi2(φ)` — IonQ's hardware-native single-qubit gates.
`ms(φ₀, φ₁)` — Mølmer-Sørensen entangling gate, the native two-qubit operation on IonQ hardware.
These express circuits more compactly when targeting real IonQ devices.

Tests: unitarity (U†U = I), known analytic outputs (GPI(0)=X, Ramsey fringe, GPI2² = X), relationship to existing gates (MS(0,0) = XX(π/2), MS(π/2,π/2) = YY(π/2)).

---

## Phase 2 — Measurement and classical control

Today's `Distribution` is purely statistical (post-run probabilities). Real quantum programs need mid-circuit measurement.

### 2a. Classical registers ✓
`Circuit.creg(name, size)` — declare a classical register.
`Circuit.measure(qubit, creg, bit)` — collapse a qubit, store result classically.
`Circuit.reset(qubit)` — reset qubit to |0⟩ post-measurement.

### 2b. Conditional gates ✓
`Circuit.if(creg, value, build)` — apply a gate (or sequence) only if a classical register holds a value.
Required for teleportation, error correction, and mid-circuit feedback.

### 2c. Statevector inspection API ✓
`circuit.statevector()` — return the full sparse amplitude map after execution.
`circuit.amplitude(bitstring)` — return the complex amplitude for a specific basis state.
`circuit.probability(bitstring)` — return |amplitude|².
Needed for algorithms where you want exact amplitudes, not sampled counts.

---

## Phase 3 — Import and export

A circuit library only reaches its potential when it speaks other languages.

### 3a. IonQ JSON (import + export) ✓
The IonQ native circuit format (`ionq.circuit.v0`) — the format used by qsim and IonQ Cloud.
`Circuit.fromIonQ(json)` — parse an IonQ JSON object into a `Circuit`.
`circuit.toIonQ()` — serialize to IonQ JSON.
This closes the loop: write a circuit in ket, run it locally, submit the exact same JSON to IonQ hardware.

### 3b. OpenQASM 2.0 (import + export) ✓
The lingua franca of quantum circuits. All major platforms accept QASM.
`Circuit.fromQASM(string)` — parse OpenQASM 2.0.
`circuit.toQASM()` — emit valid OpenQASM 2.0.
Round-trip test: parse → emit → parse → compare unitary matrices.

### 3c. Export targets
Each target is a serializer; the core simulation is unaffected.
| Target | Use case |
|---|---|
| Qiskit (Python) | Run on IBM hardware or Aer locally |
| Cirq (Python) | Google hardware and noise simulation |
| Q# | Microsoft Azure Quantum |
| pyQuil / Quil | Rigetti hardware |
| AWS Braket | Amazon Braket service |
| IonQ JSON | IonQ Cloud (via qsim or direct) |

---

## Phase 4 — Visualization

### 4a. ASCII circuit diagram
`circuit.draw()` — a text-mode circuit diagram for terminal and notebooks.

```
q0: ─H──●──
         │
q1: ────X──
```

### 4b. SVG export
`circuit.toSVG()` — a scalable vector diagram suitable for documentation and web embedding.

### 4c. Bloch sphere coordinates
`circuit.blochAngles(qubit)` — return (θ, φ) for a single-qubit state, for visualization on the Bloch sphere.

---

## Phase 5 — Advanced simulation

### 5a. Noise models
Depolarizing channel, SPAM readout errors, and T1/T2 decay — matching IonQ's published characterization methodology (already implemented in qsim; can be ported directly).
`circuit.run({ noise: 'aria-1' | NoiseParams })` — opt-in per run.

### 5b. Tensor network simulation
For circuits with bounded entanglement (Bernstein-Vazirani, QAOA with low depth, product-state preparations), a matrix-product-state simulator can handle 50+ qubits in kilobytes.
Implemented as a second simulation backend; the `Circuit` API is unchanged.

### 5c. Shots-free exact probabilities
For small circuits where the user wants exact floating-point probabilities without sampling variance, expose a `circuit.exactProbs()` that returns the full probability map directly from the statevector.

---

## Phase 6 — Ecosystem

### 6a. Quantum algorithms library
Standard algorithms as composable circuit builders:
- `grover(oracle, n)` — Grover's search
- `qft(n)` — Quantum Fourier Transform
- `phaseEstimation(unitary, precision)` — Quantum Phase Estimation
- `vqe(ansatz, hamiltonian)` — Variational Quantum Eigensolver primitive

### 6b. npm publication
Publish as `ket` (or `@dmvjs/ket`). Semantic versioning from 0.1.0.

### 6c. Browser bundle
`dist/ket.min.js` — UMD bundle for direct `<script>` inclusion, built with esbuild.

---

## Test coverage policy

Every feature ships with tests that verify:
1. **Correctness** — known analytic outputs (not just "doesn't crash")
2. **Invertibility** — U†U = I for every unitary gate
3. **Composition** — gate sequences that have known combined effects
4. **Edge cases** — zero rotation angles, large qubit indices, empty circuits

The full suite runs in < 200ms. No test may be slow-by-design; GHZ and other multi-qubit tests are parameterized and capped to keep CI fast.
