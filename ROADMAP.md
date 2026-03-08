# ket — Roadmap

## What's built

**Simulation**
- Sparse statevector over `Map<bigint, Complex>` — no 32-bit overflow at any qubit count
- MPS / tensor network backend — O(n·χ²) memory, configurable bond dimension, GHZ-50 in milliseconds
- Exact density matrix — sparse mixed-state simulation, per-gate depolarizing channels, zero overhead on noiseless circuits
- Noise profiles: depolarizing (p1/p2) + SPAM readout errors; named profiles `aria-1`, `forte-1`, `harmony`
- Shots-free exact probabilities via `exactProbs()`

**Gates**
- Full single-qubit set: H, X, Y, Z, S/sdg, T/tdg, V/srn and inverses; Rx/Ry/Rz; U1/p, U2, U3; R2/R4/R8; VZ; identity
- Two-qubit: CNOT, SWAP, CX/CY/CZ/CH, CRx/CRy/CRz, CR2/CR4/CR8, CU1/CU2/CU3, CS/CT/CS†/CT†/C√X, XX/YY/ZZ/XY, iSWAP, √iSWAP
- Three-qubit: Toffoli (CCX), Fredkin (CSWAP), controlled-√SWAP
- Native IonQ gates: GPI, GPI2, MS
- Scheduling: barrier (no-op in simulation, emitted in QASM)
- Aliases match quantum-circuit and Qiskit naming conventions throughout

**Import / Export**
- OpenQASM 2.0 + 3.0: import + export (2.0); `fromQASM` auto-detects version
- IonQ JSON: import + export
- Quil 2.0: import + export
- Native JSON: lossless versioned round-trip (import + export)
- Export-only: Qiskit, Cirq, Q#, pyQuil, Amazon Braket, CudaQ, TensorFlow Quantum, Quirk JSON, LaTeX (quantikz)
- Qiskit/Cirq import: not supported (neither is quantum-circuit; future moat)

**Classical control**
- Classical registers, measurement, reset
- Conditional gate application (`circuit.if`)

**Named gates and composition**
- `defineGate` / `gate` / `decompose` — reusable sub-circuit gates with qubit remapping and recursive expansion

**State inspection and visualization**
- `statevector()`, `amplitude()`, `probability()`, `exactProbs()`, `marginals()`, `stateAsString()`
- `blochAngles(q)` — Bloch sphere (θ, φ) via partial trace
- `dm()` — `DensityMatrix` with `purity()`, `entropy()`, `blochAngles(q)`, `probabilities()`
- `draw()` — ASCII circuit diagram
- `toSVG()` — self-contained SVG
- `toLatex()` — quantikz LaTeX environment

**Algorithms**
- `qft(n)` / `iqft(n)` — Quantum Fourier Transform
- `grover(n, oracle)` — Grover's search
- `phaseEstimation(precision, unitary)` — Quantum Phase Estimation
- `vqe(ansatz, hamiltonian)` — Variational Quantum Eigensolver with Pauli string Hamiltonians

---

## Coming next

1. **npm publication** — publish as `ket` under semantic versioning from 0.1.0
2. **Qiskit / Cirq import** — parse Python SDK output formats directly; both ket and quantum-circuit are export-only today, so this is an open moat
3. **First-class IonQ hardware DX** — device-aware circuit validation against hardware gate sets; calibration data integration for realistic noise parameters

---

## Design principles

- **Immutable API** — every gate method returns a new `Circuit`; no mutation, safe to compose and branch
- **TypeScript-strict** — the library is written in TypeScript from the ground up, not typed after the fact
- **Zero runtime dependencies** — the entire simulator, all backends, and all serializers ship with no external packages
- **Tests verify correctness, not coverage** — every feature ships with tests against known analytic outputs; the full suite runs in under 200ms
