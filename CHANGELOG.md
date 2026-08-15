# Changelog

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
