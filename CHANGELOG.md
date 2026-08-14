# Changelog

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
