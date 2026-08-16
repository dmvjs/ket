# Changelog

## 0.9.2

### Added — exact shot sampling from a contraction

`sampleByContraction(circuit, { shots, blockSize })` samples bitstrings from a
circuit by contracting it against its own conjugate, resolving qubits a block at
a time. Closing a qubit's shared wire with |v⟩ conditions on it, an identity cap
leaves it open and returns ψ(x)·conj(ψ(x)) — the probability — and leaving it
alone marginalises it away, so the conditional chain is exact with no rejection
step.

Until now contraction returned one amplitude, which is no use for sampling: you
would need all 2ⁿ. This is what makes it a backend rather than a spot check.

| circuit | shots | width | time |
|---|---|---|---|
| 100 qubits, depth 4 | 1,000 | 7 | 1.3 s |
| 200 qubits, depth 4 | 1,000 | 9 | 4.3 s |
| **400 qubits, depth 4** | 1,000 | 9 | **19.5 s** |

Two things make it affordable. The network's structure does not depend on the
values sampled, so each block is planned once and replayed. And a block's
conditional can only depend on decided qubits in its **backward light cone**, so
the conditional cache is keyed on those bits rather than the whole prefix — on
200 qubits at depth 4 that is 385 contractions instead of 23,105, and 4.3 s
instead of 76 s, for bit-for-bit identical samples.

The limit is the doubling: joining ψ to its conjugate roughly doubles the
contraction width, so this is affordable exactly where contraction already wins.
Depth ends it — the cone widens, cache hits collapse, and the width doubles on
top. Past about depth 8 it is the wrong tool.

### Tests

2,462, up from 2,456.

## 0.9.1

### Fixed — stabilizer-rank sampling was wrong on any support single-bit flips cannot cross

`runStabilizerRank` returned a confident wrong distribution for a whole class of
states. A 26-qubit GHZ came back as `P(|1…1⟩) = 1` against a true 1/2 — on every
seed, with nothing to indicate a problem.

**Results from 0.9.0 and earlier are wrong for any circuit whose support is not
connected under single-bit flips, at n ≥ 23.** That class is not exotic: GHZ and
cat states, W states, and stabilizer-code states all have parity- or
weight-constrained support, and those are exactly what one points a stabilizer
simulator at. Clifford-only circuits, connected-support circuits, and anything
below n = 23 are unaffected — under that width the sampler enumerates exactly.

The decomposition was never at fault. Forcing `method: 'exact'` on the same
26-qubit state gave the right answer all along; only the Metropolis fallback was
broken. Its single-bit-flip proposals cannot cross a zero-amplitude basis state,
so the chain seeds on one component of the support and cannot leave it — for GHZ
every neighbour of the seed is empty, so it emitted its starting state for every
shot.

A chain that accepts no move now probes for supported states elsewhere. Having
accepted nothing it has proved every neighbour empty, so finding any other
supported state is *proof* the samples are wrong rather than evidence of it, and
it throws with an explanation instead of returning them. Detection costs 1–8 ms
even at n = 100. A genuine point mass, where the frozen chain is correct, still
returns normally.

Two tests were passing against the degenerate output — they asserted only that
shots came back and that samples lay in the GHZ support, both true of a single
repeated bitstring — and `benchmark/stabilizer-rank.ts` measured a run that never
sampled at all. All corrected.

### Fixed — MPS trajectories cleared the whole workspace on every shot

`MpsTrajectory.reset()` zeroed the entire preallocated workspace per shot. The
workspace is sized for `maxChi`, so at the default `maxBond = 64` each site holds
262 KB while a χ = 2 state needs 128 bytes: a 50-qubit run paid **13 MB of memset
per shot to clear 6 KB of live data**. It now clears only the extent the previous
shot used.

That was both a single-threaded tax and the reason worker runs stopped scaling —
every thread streamed those 13 MB at once and they contended on memory.

- Single-threaded: **3.9× faster** (2038 ms → 518 ms, 50 qubits, 32k shots)
- Worker scaling: **2.4× → 7.7×** on 16 cores
- End to end at 8 workers: 848 ms → 73 ms
- **127-qubit GHZ, 1024 shots: 186 ms → 10 ms**

### Changed — sparse states promote to dense at 1/64 fill instead of 1/8

The break-even between the sparse and dense kernels sits near 2ⁿ/100; promotion
at 2ⁿ/8 was twelve times later than that, so every amplitude past the crossover
paid a BigInt key, a boxed complex and a Map slot to build and was then copied
into the dense buffer anyway.

Measured across dense, sparse, partial-occupancy, QFT and Grover circuits at
n = 10…22, `fill: 64` is faster everywhere and slower nowhere — 4.9× on a
75%-occupancy state at n = 16, 4.2× at n = 22 (788 ms → 188 ms), 2.0× on QFT-16.
Genuinely sparse states still never promote. Override with `dense: { fill }`.

The `fill` option was also documented backwards: **higher** values promote sooner.

### Changed — sparse kernel applies gates in place

Diagonal gates (`z`, `s`, `t`, `rz`, `u1`, `p` and `cu1`) only scale amplitudes
and cannot change the support, so they now skip the pairing machinery entirely —
which per entry was a `Set` insert and lookup, three BigInt allocations and two
extra Map lookups. The general path drops that `Set` too, and gates mutate the
state rather than rebuilding the map, matching what the dense kernel already did.

Grover-12 **1.8× faster**; QFT-20 1.2×.

### Added — amplitudes by tensor-network contraction

`amplitudeByContraction(circuit, bitstring)` computes ⟨x|U|0…0⟩ by treating the
circuit as a tensor network and contracting it to a scalar. Cost is governed by
the contraction **width** — the largest intermediate tensor — which follows the
circuit's connectivity rather than its qubit count. A statevector is the special
case where the width is n.

A depth-4 circuit contracts at width 2 whether it is 20 qubits or 400, so a
400-qubit amplitude takes 362 ms where a statevector needs 2⁴⁰⁰. Width grows with
depth instead — 2, 5, 8, 11 at depths 4, 8, 12, 16 on 40 qubits — which is where
the real limit sits.

Planning is exposed separately, because the order *is* the algorithm: the same
network contracted well or badly differs by orders of magnitude. Three planners
ship — randomized greedy (`planContraction`), recursive bisection with
Fiduccia–Mattheyses refinement (`planContractionPartitioned`), and `planBest`,
which runs both and keeps the better.

Neither planner dominates. Greedy wins on smaller circuits (width 7 against 8 at
n=14, 5 against 6 at n=30); bisection pulls ahead as width grows, which is where
the order matters most — 12 against 13 at n=40 d=12, **13 against 15 at n=50 d=14**
(four times less memory), 15 against 17 at n=60 d=16. `planBest` runs both and
keeps the better, which is cheap because `evaluatePlan` scores a plan without
touching tensor data.

Bisection needed two things to become competitive, both found by measuring rather
than assuming. **Multilevel coarsening**: flat refinement cannot escape a local
minimum unless some single-vertex move improves matters, and on these graphs none
does; coarsening collapses clusters so one move relocates a whole region.
**Searching the balance tolerance**: balanced halves are the wrong shape for a
circuit, whose best order is a lopsided sweep, and forcing them cost 2–4 width.

### Changed — contraction planning is heap-driven

Candidate pairs now live in a min-heap with lazy invalidation, so each step costs
the merged tensor's degree instead of a scan over every remaining pair. Planning a
100-qubit network drops from 6,658 ms to 27 ms; a 60-qubit one from 5,182 ms to
25 ms.

Freezing a candidate's jittered score at push time means one restart explores
slightly less than rescoring everything each step, costing 1–2 width on its own.
Restarts are cheap now, so the default rises from 24 to 64, which recovers the
quality and is still several times faster than the old default.

### Changed — contraction kernel is a permute plus a matrix product

A contraction over shared indices *is* a matrix product once those indices are
contiguous, so both operands are permuted and multiplied rather than walked
position by position with the index decomposition repeated per element.

Between 1.4× and 4.6× faster, the advantage growing with contraction size:
77.5 ms to 16.7 ms on a 28-qubit depth-14 network. Operands already in the right
order skip the permutation, and a zero row of the left operand skips an entire
pass over the right — worth having, since gate tensors are mostly zeros.

Arithmetic is no longer the bottleneck: at that size the contraction is 17 ms
against 957 ms of planning.

### Added — contraction slicing

`amplitudeBySlicedContraction` fixes a set of indices rather than summing over
them, contracting once per assignment and adding the results. Each sliced index
halves the memory and doubles the number of contractions, and those contractions
are independent — the mechanism by which contractions too large for any machine
are spread across many.

The doubling is a worst case: one sliced index bought half the memory for
1.14–1.78× the work on the circuits measured, because removing an index also
removes work that was being repeated inside the contraction. Returns fall off
after a few indices — 9 sliced indices on a 20-qubit depth-12 network buy 8× the
memory for 168× the work.

Selection ranks candidates by how many of the widest intermediates carry them,
and measures progress on `(width, count of intermediates at that width)`. Width
alone is the wrong signal: a peak held by six intermediates does not fall when an
index leaves five of them, so a width-only rule stalls after a single slice.
Emptying the peak set is the step before the width moves.

That reaches real targets — width 12 down to 7 on a 20-qubit depth-12 circuit,
**32× less memory for 27× more work** against a naive 512×, with every slice
independent.

Candidates are scored by planning the reduced network, so the comparison is only
as trustworthy as the planner is repeatable: with too few restarts it measures
planner variance instead of the slice and stops early, reaching width 11 where 24
restarts reach 7. It stops at `maxSliced` and reports the width it reached rather
than the one requested.

### Changed — diagonal gates no longer cut their wires

Depth is what makes a contraction impossible, and most of the width building it
was bookkeeping. A gate diagonal in the computational basis does not mix basis
states, so it does not need to cut its wire and start a fresh index — it sits on
the index already there. The index is then held by three tensors or more, and a
layer of CZs stops doubling the index count.

Applies to `z`, `s`, `t`, `rz`, `u1`/`p` and inverses, and to any controlled
version — `cz`, `cs`, `cp`, `crz`.

| circuit | width before | width after | time before | time after |
|---|---|---|---|---|
| n=40 depth 12 | 13 | **8** | 87 ms | 130 ms |
| n=40 depth 16 | 17 | **11** | 282 ms | 185 ms |
| n=40 depth 20 | 23 | **14** | 5.4 s | **260 ms** |
| n=40 depth 24 | — | **17** | out of reach | **543 ms** |
| n=30 depth 30 | 27 | **20** | > 60 s | **2.8 s** |

Width falling by 6 is 64× less memory and 64× less arithmetic, which is why
depth 24 at 40 qubits went from unreachable to half a second.

An index still held by a third tensor cannot be summed when two of its holders
contract, so it survives as a **batch index**: both operands are addressed at the
same value and the result keeps it — a matrix product run once per assignment.
`contractPair` takes the indices to keep as a third argument; `planContraction`,
`evaluatePlan` and the slicing profiler apply the same rule when they replay a
plan over index sets.

Open wires get an identity cap so an index held once still means an output. A
trailing diagonal gate shares its wire's index rather than renaming it, and
without the cap `amplitudeBatchByContraction` would have summed an open wire away
instead of returning it.

The cost is planning time: a hyper-index makes every pair of tensors holding it a
candidate, so the candidate graph is denser and a 400-qubit depth-4 plan takes
362 ms against 230 ms — for a width of 2 instead of 4.

This moves the wall rather than removing it. Width still grows with depth — 20 at
depth 30, 26 at depth 40, 34 at depth 50 on 30 qubits.

Three slicing tests were re-aimed at deeper circuits. They had been asserting
that slicing was *needed* on networks that no longer need it, and one asserted
overhead is always above 1, which is no longer true — slicing a hyper-index
removes it from every tensor holding it at once.

### Added — many amplitudes from one contraction

`amplitudeBatchByContraction(circuit, pattern)` takes `?` for a qubit to leave
open. The wire is not closed off, so the contraction ends on a tensor over those
qubits rather than a scalar, and one contraction yields every amplitude matching
the pattern.

At 60 qubits and depth 10, **65,536 amplitudes in 76 ms** against a projected
~3,299 s one at a time. That is not a constant factor — contracting per bitstring
repeats the whole network each time, and the batch does it once.

This is what makes contraction usable for sampling rather than for spot checks.
The open indices widen every intermediate carrying them, so batch size trades
against contraction width — the same currency slicing spends, which is why the
two belong together.

### Added — expectation values by Pauli-path propagation

`pauliPathExpectation(circuit, observable)` computes ⟨ψ|O|ψ⟩ for a Pauli
observable without building a state, by carrying the observable backward through
the circuit in the Heisenberg picture.

The cost model is unlike the other backends: a Clifford gate maps one Pauli to
±one Pauli at any width, so width is nearly free, and only rotations branch. What
costs is the surviving term count, which is set by the observable's light cone
rather than the qubit count. Two QAOA layers with a two-local observable hold at
63 terms from 8 qubits to 120, so **120 qubits takes 28 ms** where a statevector
needs 127 ms at 16 and is out of reach past ~24.

It answers ⟨O⟩ only — no sampling, no state — which is what VQE and QAOA actually
want. Truncation is reported rather than hidden: `droppedWeight` bounds the error
in the returned value, and unsupported gates are refused by name.

Truncation runs along two axes, and both matter at depth. `maxWeight` discards
Pauli terms above a given weight; `noise: { p1, p2 }` damps terms by the same
depolarizing convention the density-matrix backend uses, validated against exact
`Tr(ρP)` at n ≤ 4. On a 60-qubit kicked-Ising circuit at depth 6, where an
exact-ish run holds 62,626 terms in 8.7 s, `maxWeight: 6` gets the same answer to
four decimals in 1.2 s, and modelling the device noise takes 124 ms — the latter
being a better model of real hardware rather than a worse one of ideal hardware.

### Added — one conformance battery for every backend

`src/conformance.test.ts` replaces eight hand-written per-backend test blocks
with a single battery that all six backends run through: agreement with the
analytic oracle, normalisation, seed determinism, and support containment.

It also carries invariants that hold **past statevector reach**, which is where
the stabilizer-rank bug lived and why nothing caught it — a GHZ state is two
outcomes at 1/2 each at any width, no oracle required. Adding a backend now means
declaring what it accepts and how it samples.

The contract it encodes: **a backend may decline a circuit; it may not answer one
wrongly.** A refusal passes, provided it explains itself.

### Changed — published figures corrected to measured values

Several numbers in the README, reference and site did not survive measurement:

- Multi-core speedup was quoted as 4× and had never been measured; it is 3.8× at
  4 workers and 7.1× at 8, on 16 cores.
- `exactProbs()` examples promised `{ '00': 0.5, '11': 0.5 }`; it returns
  `0.4999999999999999`, one ulp below, because 1/√2 is not representable in a
  double. The arithmetic is unchanged — rounding inside `exactProbs()` would hide
  real numerical error in the method people use to check correctness. "Exact"
  means free of sampling variance, not free of float error.
- The GHZ fidelity chart was labelled `P(|0…0⟩ + |11…1⟩)` but plots the
  probability of an error-free run (52%); landing on one of the two ideal
  outcomes is 46.9% ± 1.5% over 4096 shots. Both figures are now given.
- The stabilizer-rank headline is the cost of building the decomposition at one
  shot. Sampling is flat in t and charged separately: ~43 ms/shot at t = 50, so
  the default 1024 shots turns 5.6 s into about 49 s.
- The Clifford figure had the same problem: "GHZ-1024 — 3997 ms" was a 64-shot
  run with the shot count left off. Evolving the tableau takes 4 ms; each shot
  costs ~57 ms, because sampling 1,024 qubits is the expensive part. Both are now
  stated. The MPS, statevector and density-matrix rows were also re-measured.
- Bundle sizes in the reference were stale (429/195/196 KB → 431/196/197 KB).

### Fixed — option objects silently ignored unknown keys

Destructuring drops any key a method does not name, so a wrong option changed
behaviour without complaint. Passing `delta` to `runStabilizerRank`, whose option
is `targetError`, left the run exact rather than sparsified — a different
simulation, no warning, and plausible numbers out the other end. It is how a
benchmark in this very release cycle came to "disprove" a documented figure that
was in fact correct.

`run`, `runMps`, `runClifford`, `runStabilizerRank`, `simulate`, `statevector`
and `dm` now reject keys they do not read, listing the valid ones and suggesting
a near-miss where there is one:

    runMps: unknown option 'maxbond' — did you mean 'maxBond'? Valid options:
    shots, seed, maxBond, truncErr, maxChi, initialState, noise, workers.

### Tests

2,456, up from 2,154.

## 0.9.0

### Fixed — OpenQASM import mis-parsed multi-register programs, silently

`Circuit.fromQASM` was a set of regexes over `;`-split statements. It summed the
declared register sizes and then indexed by whatever number appeared in brackets,
so register *names* were discarded:

```
qreg a[2];
qreg b[2];
cx a[0], b[0];   // both resolved to qubit 0
```

That threw, which was the lucky case. The unlucky one was register-wide
application — `h q;` produced a gate on `q[undefined]`, building a corrupt
circuit with no error at all. Anything with a `gate` definition, an
`if (c == N)` guard, or a broadcast `measure q -> c;` failed outright.

Replaced with a real parser: brace-aware statement splitting, named registers
laid out in declaration order, `gate` definitions expanded inline (parameterised
and nested), `opaque`, `barrier`, `reset`, `if`, both `measure` forms, and QASM's
broadcast rule. Angle expressions gained `^`, scientific notation, the standard
functions, and the enclosing gate's parameters.

Everything unsupported — `ctrl @`, `gphase`, `for`/`while`/`def`, `else`, `let`,
QASM 3 classical types — now throws a `TypeError` naming the construct instead of
falling through to a partial parse. **No input should now produce a wrong circuit
without an error.**

### Fixed — `runMps({ workers })` left the host process running forever

The persistent worker pool never called `unref()`, so its threads kept Node's
event loop alive. The run itself was always correct and fast; the program simply
never exited, which reads as a hang at the end of `main`.

Workers also inherited the parent's `execArgv`. Under `node --input-type=module
-e '…'` that is invalid for a file-backed worker, which died at boot with
`ERR_INPUT_TYPE_NOT_ALLOWED` and surfaced five minutes later as a timeout on a
flag nothing would ever set. Workers are now created with `execArgv: []`.

There was no `error` listener on the pool at all, so *every* worker failure
became a silent five-minute wait. Failures now report what happened.

### Fixed — ket logged a CORS error on every page load in a browser

`src/worker-shim.ts` probed for `node:worker_threads` with a caught dynamic
import. A browser rejects that specifier at the network layer and logs an error
whether or not the rejection is caught, so the failure was never as silent as
intended. The lookup is now `process.getBuiltinModule`, which is synchronous and
asks for nothing on a non-Node host. On Node 22.0–22.2, which predates that API,
workers fall back to the single-threaded path.

### Added — `@kirkelliott/ket/compat`

A `QuantumCircuit` class with a mutable, column-indexed API: gates placed at an
explicit column, the circuit growing to fit any wire named, state stored on the
instance by `run()`. It covers the gate set, classical registers and conditions,
custom gates via `registerGate`, `save`/`load`, and QASM import/export.

`toKet()` returns the equivalent immutable `Circuit`, which is the way into the
rest of the library — a 200-qubit GHZ built through this API routes to the
Clifford backend and finishes in milliseconds.

### Added — global build for a plain `<script src>`

`dist/ket.global.js` is an IIFE bundle exposing one `ket` global, wired to the
`unpkg` and `jsdelivr` fields so the bare CDN URL serves it:

```html
<script src="https://unpkg.com/@kirkelliott/ket"></script>
<script>const bell = new ket.Circuit(2).h(0).cnot(0, 1)</script>
```

This format cannot contain a top-level await, which is what the worker-shim
change above removed. `import.meta` is empty here, so the two reads of
`import.meta.url` that detect a source-vs-bundle entry point are now guarded —
unguarded, they threw on every `runMps` call in the browser.

### Changed

- **Multi-core speedup is 2.2×, not 4×.** Measured with 4 workers on 16 cores
  across 50- and 100-qubit runs at 8k–32k shots. The previous figure was never
  measured. It is not the result transfer: a run with two distinct outcomes
  scales identically.
- **`exactProbs()` examples now show what it returns.** The docs promised
  `{ '00': 0.5, '11': 0.5 }`; the value is `0.4999999999999999`, one ulp below,
  because 1/√2 is not representable in a double. The arithmetic is unchanged —
  rounding inside `exactProbs()` would hide real numerical error in the method
  people reach for to check correctness. "Exact" means free of sampling
  variance, not free of float error.

### Tests

2,154, up from 2,012. New coverage where the suite was structurally blind:

- `src/dist.test.ts` asserts against the built artifacts — export parity across
  ESM and the global bundle, the global bundle evaluated as a classic script (a
  reintroduced top-level await becomes a `SyntaxError` there), no `node:`
  specifiers in the browser bundle, and the compat entry re-exporting rather
  than bundling a second copy of the engine.
- The `runMps` worker tests run in a child process against `dist/ket.js`.
  `runMps` checks `import.meta.url` and falls back to single-threaded under a
  `.ts` entry point, so a test calling it from vitest exercises the fallback and
  passes against a broken pool — which is how the pool bug survived.

## 0.8.0

### Fixed — IonQ rotation angles were sent π× too small

`toIonQ()` wrote `rotation: θ/π`, but IonQ's QIS gateset takes **radians** — its
own documentation gives Rx(π/2) as `rotation: 1.5708`. Every `rx`/`ry`/`rz`/
`xx`/`yy`/`zz` submitted through ket therefore described a different circuit than
the one you built, and `fromIonQ()` had the matching inverse error.

**If you submitted a circuit containing rotation gates to IonQ using 0.7.0 or
earlier, its results are wrong.** The job runs and returns a plausible
distribution, so there is nothing to alert you; a 12,248-gate Shor circuit came
back as 7,999 near-uniform outcomes that were easy to mistake for noise or for a
hardware limitation. Clifford-only circuits — Bell pairs, GHZ states, anything
without an angle parameter — are unaffected, which is why this survived so long.

Two properties hid it:

- `fromIonQ(toIonQ(c))` was exact, because both directions shared the same wrong
  scale factor. Round-trip tests cannot detect a convention error.
- The native gates `gpi`/`gpi2`/`ms` genuinely do take **turns**, and ket handled
  those correctly. The two conventions differ and must not be unified.

The wire format is now asserted against literal values from IonQ's published
example rather than only through round-trips.

### Added

- **`Circuit.toIonQBasis()`** — expands gates IonQ cannot represent. QFT and
  modular-exponentiation circuits are built almost entirely from `cu1` and
  previously could not be exported at all; `shorCircuit(15n, 7n).toIonQ()` threw.
  `toIonQ()` remains a serializer and still rejects what it cannot express, so
  the gate count you inspect is the gate count you send. Composes with
  `compile(device)` for hardware-native output.
- **Native controlled gates.** IonQ expresses these as a base gate plus a
  `controls` array — `{gate:'x', controls:[a,b], target:t}` is a Toffoli, not
  `ccx`. `toIonQ()` now emits that form and `fromIonQ()` parses it, so `ccx`,
  `cy`, `cz`, `ch`, `crx`/`cry`/`crz`, `cs`/`ct` and their daggered forms
  serialize directly instead of expanding to Clifford+T. `cu1` becomes
  `rz(0/2)` on the control plus a controlled `rz` — two gates rather than five —
  and `cswap` routes through a controlled `x` in three rather than seventeen.
  Shor at N=15 with full precision drops from 36,878 gates to 17,447.
  The emitted set is IonQ's published legal list; `swap` is excluded because the
  API rejects a controlled swap despite listing it.
- **`cancelIonQJob(id, opts)`** — cancel a queued or running job.
- **`noise` option on `submitIonQ`/`runIonQ`** — e.g. `{ model: 'ideal' }` or
  `{ model: 'forte-1', seed: 100 }`. Omitting it defers to IonQ's server-side
  default, which is not guaranteed to be noiseless; at any real depth a device
  model flattens the distribution into something indistinguishable from a wrong
  answer. State it explicitly when the distinction matters.

### Changed

- `checkDevice()` now points at `toIonQBasis()` for unsupported gates. It
  previously suggested `decompose()`, which only inlines named subcircuits and
  cannot rewrite a gate type.
- Device table: `aria-1`, `aria-2`, `harmony`, `h1-1`, `ibm_sherbrooke` and
  `ibm_torino` are marked `retired`; `helios` and `forte-enterprise-1` added.
  Retired entries are kept so historical results stay reproducible — they will
  not accept new jobs.

### Verified on hardware

Shor's algorithm factored 15 = 3 × 5 on IonQ's simulator: counting register at
0/2/4/6 with 25% each and exactly zero off-peak, work register on the orbit of 7
mod 15. Shor's two-register discrete logarithm recovered x = 3 for g = 2, p = 5,
with every outcome satisfying β ≡ xα (mod r).

These are toy sizes chosen so the answer is checkable by hand. Both example
scripts close by computing the probability of an error-free run on current
trapped-ion error rates; for the RSA instance it is 8×10⁻⁶⁹.
