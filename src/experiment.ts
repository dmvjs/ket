/**
 * Experiment records and parameter sweeps.
 *
 * A protocol answers one question about a device. An experiment is the loop
 * around that: vary something, run each point, keep every result with enough
 * provenance to reproduce or re-analyse it later. `sweep` runs the loop;
 * the `Experiment` it returns is plain JSON — circuits, parameters, outcomes,
 * backend, timing, library version — that `parseExperiment` reads back.
 *
 * The executor abstraction is shared with the protocols: any function from a
 * circuit to outcomes, synchronous or not, local or hardware.
 */

import { Circuit } from './circuit.js'
import type { CircuitJSON, Distribution } from './circuit.js'
import { VERSION } from './version.js'

// ── Executors ────────────────────────────────────────────────────────────────

/**
 * Measurement outcomes a protocol can analyse: a `Distribution` from any
 * backend, or a bitstring → probability map with qubit 0 leftmost, which is
 * what `countsToProbs` returns for a hardware job.
 */
export type Outcomes = Distribution | Readonly<Record<string, number>>

/**
 * Outcomes with provenance an executor wants kept: which backend or device ran
 * the circuit, the job id, anything else. Only `outcomes` is required.
 */
export interface RunReport {
  readonly outcomes: Outcomes
  /** Free-form backend label, e.g. `'ionq:qpu.forte-1'`. A `Distribution` supplies its own when this is absent. */
  readonly backend?: string
  /** Shots behind a probability map, so error bars can be computed later. A `Distribution` supplies its own. */
  readonly shots?: number
  readonly jobId?: string
  readonly meta?: Readonly<Record<string, unknown>>
}

export type ExecutorResult = Outcomes | RunReport

/**
 * Runs one circuit and returns its outcomes. Synchronous for a local backend
 * (`c => c.run({ shots })`), asynchronous for hardware. `index` is the
 * circuit's position in the batch, useful for per-circuit seeds.
 */
export type Executor = (circuit: Circuit, index: number) => ExecutorResult | Promise<ExecutorResult>

/**
 * Structural check rather than `instanceof`: a Distribution from a second copy
 * of the library (global bundle beside an ESM import) must still be recognised.
 */
export function isDistribution(o: Outcomes): o is Distribution {
  const d = o as { probs?: unknown; shots?: unknown }
  return typeof d.probs === 'object' && d.probs !== null && typeof d.shots === 'number'
}

/** Bitstrings contain only 0 and 1, so an `outcomes` key can only mean a report. */
function isReport(r: ExecutorResult): r is RunReport {
  return typeof r === 'object' && r !== null && 'outcomes' in r
}

/** The outcomes inside an executor's result, whichever shape it chose. */
export function resolveOutcomes(r: ExecutorResult): Outcomes {
  return isReport(r) ? r.outcomes : r
}

// ── Records ──────────────────────────────────────────────────────────────────

/** One executed circuit: what was run, with what parameters, and what came back. */
export interface ExperimentPoint<P = Record<string, unknown>> {
  /** Position in the sweep. */
  readonly index: number
  readonly params: P
  readonly circuit: CircuitJSON
  /** Bitstring → probability, qubit 0 leftmost. */
  readonly probs: Readonly<Record<string, number>>
  /** Shots behind `probs`, when known. */
  readonly shots: number | undefined
  /** Backend label: from the `RunReport`, else from the `Distribution`. */
  readonly backend: string | undefined
  readonly jobId: string | undefined
  readonly meta: Readonly<Record<string, unknown>> | undefined
  /** Wall-clock milliseconds the executor took. */
  readonly ms: number
}

/** A complete, serialisable experiment. `JSON.stringify` it; `parseExperiment` reads it back. */
export interface Experiment<P = Record<string, unknown>> {
  readonly format: 'ket-experiment'
  readonly formatVersion: 1
  /** Library version that produced the record. */
  readonly ket: string
  readonly name: string | undefined
  /** ISO timestamps. */
  readonly createdAt: string
  readonly finishedAt: string
  readonly meta: Readonly<Record<string, unknown>> | undefined
  readonly points: readonly ExperimentPoint<P>[]
}

export interface ExperimentOptions {
  readonly name?: string
  /** Anything worth keeping with the record: device calibration, operator, hypothesis. */
  readonly meta?: Readonly<Record<string, unknown>>
}

/**
 * Build one record point from a circuit and what running it returned. `sweep`
 * calls this for every point; call it yourself for results gathered out of band.
 */
export function recordPoint<P>(
  index: number, params: P, circuit: Circuit, result: ExecutorResult, ms = 0,
): ExperimentPoint<P> {
  const report = isReport(result) ? result : undefined
  const outcomes = resolveOutcomes(result)
  const dist = isDistribution(outcomes) ? outcomes : undefined
  const probs: Readonly<Record<string, number>> = dist ? dist.probs : (outcomes as Readonly<Record<string, number>>)
  return {
    index,
    params,
    circuit: circuit.toJSON(),
    probs: { ...probs },
    shots: report?.shots ?? dist?.shots,
    backend: report?.backend ?? dist?.backend,
    jobId: report?.jobId,
    meta: report?.meta,
    ms,
  }
}

/** Wrap points into an `Experiment`, stamping version and time. */
export function experiment<P>(
  points: readonly ExperimentPoint<P>[],
  options: ExperimentOptions & { createdAt?: string } = {},
): Experiment<P> {
  const now = new Date().toISOString()
  return {
    format: 'ket-experiment',
    formatVersion: 1,
    ket: VERSION,
    name: options.name,
    createdAt: options.createdAt ?? now,
    finishedAt: now,
    meta: options.meta,
    points: [...points],
  }
}

// ── Sweeps ───────────────────────────────────────────────────────────────────

/** A grid given as named axes: every combination is one point. */
export type GridSpec = Readonly<Record<string, readonly unknown[]>>

/** The parameter type a `GridSpec` produces: one value from each axis. */
export type GridPoint<S extends GridSpec> = { readonly [K in keyof S]: S[K][number] }

/**
 * Cartesian product of named axes, first axis slowest. An empty axis yields no
 * points.
 *
 * @example
 * cartesian({ theta: [0, 1], depth: [1, 2, 3] })
 * // [{ theta: 0, depth: 1 }, { theta: 0, depth: 2 }, …, { theta: 1, depth: 3 }]
 */
export function cartesian<S extends GridSpec>(spec: S): GridPoint<S>[] {
  const keys = Object.keys(spec) as (keyof S & string)[]
  let out: Record<string, unknown>[] = [{}]
  for (const k of keys) {
    const axis = spec[k]!
    const next: Record<string, unknown>[] = []
    for (const partial of out) for (const v of axis) next.push({ ...partial, [k]: v })
    out = next
  }
  return out as GridPoint<S>[]
}

export interface SweepOptions extends ExperimentOptions {
  /** Circuits in flight at once. Default 1 — one after another. Results keep grid order regardless. */
  readonly concurrency?: number
}

/**
 * Run a circuit family over a parameter grid and return the record.
 *
 * `grid` is either an explicit list of parameter objects or named axes for
 * `cartesian`. `build` turns one parameter object into a circuit; `execute`
 * runs it anywhere.
 *
 * @example
 * // Local: Ry rotation angle against measured ⟨Z⟩
 * const exp = await sweep(
 *   { theta: [0, Math.PI / 4, Math.PI / 2, Math.PI] },
 *   ({ theta }) => new Circuit(1).ry(theta, 0),
 *   c => c.run({ shots: 1000 }),
 *   { name: 'ry-scan' },
 * )
 * exp.points.map(p => [p.params.theta, p.probs['1'] ?? 0])
 *
 * // Hardware, four jobs at a time, job ids kept
 * await sweep(grid, build, async c => {
 *   const { id } = await submitIonQ(c.toIonQ(), { apiKey, target: 'qpu.forte-1', shots })
 *   const job = await awaitIonQJob(id, { apiKey })
 *   const counts = ionqHistogramToCounts(await getIonQResults(job, { apiKey }), shots)
 *   return { outcomes: countsToProbs(counts, c.qubits, shots), shots, backend: 'ionq:qpu.forte-1', jobId: id }
 * }, { concurrency: 4 })
 *
 * // Protocol circuits are a grid like any other
 * const seqs = rbSequences({ lengths: [1, 4, 16, 64], samples: 20 })
 * const rb = await sweep(seqs, s => s.circuit, execute)
 * rbFit(rb.points.map(p => ({ length: p.params.length, survival: rbSurvival(p.probs) })))
 */
export function sweep<S extends GridSpec>(
  grid: S, build: (params: GridPoint<S>, index: number) => Circuit, execute: Executor, options?: SweepOptions,
): Promise<Experiment<GridPoint<S>>>
export function sweep<P>(
  grid: readonly P[], build: (params: P, index: number) => Circuit, execute: Executor, options?: SweepOptions,
): Promise<Experiment<P>>
export async function sweep(
  grid: GridSpec | readonly unknown[],
  build: (params: never, index: number) => Circuit,
  execute: Executor,
  options: SweepOptions = {},
): Promise<Experiment<unknown>> {
  const params: readonly unknown[] = Array.isArray(grid) ? grid : cartesian(grid as GridSpec)
  const { concurrency = 1, name, meta } = options

  const createdAt = new Date().toISOString()
  const points: ExperimentPoint<unknown>[] = new Array(params.length)
  await pool(params.length, concurrency, 'sweep', async i => {
    const p = params[i]
    const circuit = build(p as never, i)
    if (!(circuit instanceof Circuit)) throw new TypeError(`sweep: build returned a non-Circuit at index ${i}`)
    const t = performance.now()
    const result = await execute(circuit, i)
    points[i] = recordPoint(i, p, circuit, result, performance.now() - t)
  })

  return experiment(points, {
    createdAt,
    ...(name !== undefined && { name }),
    ...(meta !== undefined && { meta }),
  })
}

// ── Resumable sweeps ─────────────────────────────────────────────────────────
//
// A hardware sweep outlives the process that started it: jobs sit in a queue
// for minutes or hours, and anything held only in memory is lost if the
// process dies. So the run is split. `submitSweep` sends every circuit and
// returns a record of job ids — store it. `collectSweep` fetches whatever has
// finished and returns the record with those results filled in — store that
// too, and call it again later. `finishSweep` turns a complete record into an
// ordinary `Experiment`.

/** Submits one circuit and returns its job id. */
export type Submitter = (circuit: Circuit, index: number) => string | Promise<string>

/**
 * Fetches the result of one submitted job. Return `undefined` when the job
 * has not finished; throw for a real failure.
 */
export type Collector = (jobId: string, circuit: Circuit, index: number) =>
  ExecutorResult | undefined | Promise<ExecutorResult | undefined>

export interface PendingPoint<P = Record<string, unknown>> {
  readonly index: number
  readonly params: P
  readonly circuit: CircuitJSON
  readonly jobId: string
  /** ISO timestamp. */
  readonly submittedAt: string
  /** Filled in by `collectSweep` once the job has a result. */
  readonly result: ExperimentPoint<P> | undefined
}

/** A sweep whose jobs are submitted but not all collected. Plain JSON, like `Experiment`. */
export interface PendingExperiment<P = Record<string, unknown>> {
  readonly format: 'ket-experiment-pending'
  readonly formatVersion: 1
  readonly ket: string
  readonly name: string | undefined
  readonly createdAt: string
  readonly meta: Readonly<Record<string, unknown>> | undefined
  readonly points: readonly PendingPoint<P>[]
}

/** Run `fn` over `n` indices with at most `concurrency` in flight. */
async function pool(n: number, concurrency: number, where: string, fn: (i: number) => Promise<void>): Promise<void> {
  if (!Number.isInteger(concurrency) || concurrency < 1)
    throw new RangeError(`${where}: concurrency must be a positive integer, got ${concurrency}`)
  let next = 0
  const worker = async (): Promise<void> => { while (next < n) await fn(next++) }
  await Promise.all(Array.from({ length: Math.min(concurrency, n) }, worker))
}

/**
 * Submit every circuit of a sweep and return the job ids as a record. Nothing
 * waits on a result; store the record and collect later.
 *
 * @example
 * const pending = await submitSweep(grid, build, async c => {
 *   const { id } = await submitIonQ(c.toIonQ(), { apiKey, target: 'qpu.forte-1', shots })
 *   return id
 * }, { name: 'ry-scan', concurrency: 4 })
 * await writeFile('ry-scan.pending.json', JSON.stringify(pending))
 */
export function submitSweep<S extends GridSpec>(
  grid: S, build: (params: GridPoint<S>, index: number) => Circuit, submit: Submitter, options?: SweepOptions,
): Promise<PendingExperiment<GridPoint<S>>>
export function submitSweep<P>(
  grid: readonly P[], build: (params: P, index: number) => Circuit, submit: Submitter, options?: SweepOptions,
): Promise<PendingExperiment<P>>
export async function submitSweep(
  grid: GridSpec | readonly unknown[],
  build: (params: never, index: number) => Circuit,
  submit: Submitter,
  options: SweepOptions = {},
): Promise<PendingExperiment<unknown>> {
  const params: readonly unknown[] = Array.isArray(grid) ? grid : cartesian(grid as GridSpec)
  const { concurrency = 1, name, meta } = options
  const createdAt = new Date().toISOString()
  const points: PendingPoint<unknown>[] = new Array(params.length)
  await pool(params.length, concurrency, 'submitSweep', async i => {
    const circuit = build(params[i] as never, i)
    if (!(circuit instanceof Circuit)) throw new TypeError(`submitSweep: build returned a non-Circuit at index ${i}`)
    const jobId = await submit(circuit, i)
    if (typeof jobId !== 'string' || jobId === '') throw new TypeError(`submitSweep: submit returned no job id at index ${i}`)
    points[i] = { index: i, params: params[i], circuit: circuit.toJSON(), jobId, submittedAt: new Date().toISOString(), result: undefined }
  })
  return { format: 'ket-experiment-pending', formatVersion: 1, ket: VERSION, name, createdAt, meta, points }
}

/**
 * Fetch results for every point that does not have one yet and return the
 * record with them filled in. A collector returning `undefined` leaves its
 * point pending; a collector that throws aborts the call, and nothing already
 * collected in this call is kept — the input record is never mutated.
 *
 * @example
 * let pending = parsePendingExperiment(await readFile('ry-scan.pending.json', 'utf8'))
 * pending = await collectSweep(pending, async (id, c) => {
 *   const job = await getIonQJob(id, { apiKey })
 *   if (job.status !== 'completed') return undefined
 *   const counts = ionqHistogramToCounts(await getIonQResults(job, { apiKey }), shots)
 *   return { outcomes: countsToProbs(counts, c.qubits, shots), shots, backend: 'ionq:qpu.forte-1', jobId: id }
 * })
 * pending.points.filter(p => p.result === undefined).length   // still queued
 */
export async function collectSweep<P>(
  pending: PendingExperiment<P>,
  collect: Collector,
  options: { concurrency?: number } = {},
): Promise<PendingExperiment<P>> {
  const { concurrency = 1 } = options
  const points = [...pending.points]
  const todo = points.map((p, i) => [p, i] as const).filter(([p]) => p.result === undefined)
  await pool(todo.length, concurrency, 'collectSweep', async k => {
    const [p, i] = todo[k]!
    const circuit = Circuit.fromJSON(p.circuit)
    const t = performance.now()
    const r = await collect(p.jobId, circuit, p.index)
    if (r === undefined) return
    const point = recordPoint(p.index, p.params, circuit, r, performance.now() - t)
    points[i] = { ...p, result: point.jobId === undefined ? { ...point, jobId: p.jobId } : point }
  })
  return { ...pending, points }
}

/**
 * Convert a fully collected record into an `Experiment`. Throws, naming the
 * indices still pending, if any point lacks a result.
 */
export function finishSweep<P>(pending: PendingExperiment<P>): Experiment<P> {
  const missing = pending.points.filter(p => p.result === undefined).map(p => p.index)
  if (missing.length)
    throw new TypeError(`finishSweep: ${missing.length} of ${pending.points.length} points still pending (indices ${missing.join(', ')})`)
  return experiment(pending.points.map(p => p.result!), {
    createdAt: pending.createdAt,
    ...(pending.name !== undefined && { name: pending.name }),
    ...(pending.meta !== undefined && { meta: pending.meta }),
  })
}

// ── Reading back ─────────────────────────────────────────────────────────────

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

/**
 * Validate a stored experiment — a JSON string or an already-parsed object —
 * and return it typed. Throws `TypeError` naming the first field that is wrong.
 * Circuits are checked for shape here and reconstructed on demand with
 * `Circuit.fromJSON(point.circuit)`.
 */
export function parseExperiment<P = Record<string, unknown>>(input: string | unknown): Experiment<P> {
  const raw: unknown = typeof input === 'string' ? JSON.parse(input) : input
  if (!isRecord(raw)) throw new TypeError('parseExperiment: expected an object')
  if (raw['format'] !== 'ket-experiment')
    throw new TypeError(`parseExperiment: format is ${JSON.stringify(raw['format'])}, expected 'ket-experiment'`)
  if (raw['formatVersion'] !== 1)
    throw new TypeError(`parseExperiment: unsupported formatVersion ${String(raw['formatVersion'])}`)
  const str = (k: 'ket' | 'createdAt' | 'finishedAt'): string => {
    const v = raw[k]
    if (typeof v !== 'string') throw new TypeError(`parseExperiment: ${k} must be a string`)
    return v
  }
  const ket = str('ket'), createdAt = str('createdAt'), finishedAt = str('finishedAt')
  if (raw['name'] !== undefined && typeof raw['name'] !== 'string')
    throw new TypeError('parseExperiment: name must be a string')
  if (raw['meta'] !== undefined && !isRecord(raw['meta']))
    throw new TypeError('parseExperiment: meta must be an object')
  if (!Array.isArray(raw['points'])) throw new TypeError('parseExperiment: points must be an array')

  const points = (raw['points'] as unknown[]).map((pt, i) => parsePoint<P>(pt, `parseExperiment: points[${i}]`))

  return {
    format: 'ket-experiment',
    formatVersion: 1,
    ket,
    name: raw['name'],
    createdAt,
    finishedAt,
    meta: raw['meta'] as Record<string, unknown> | undefined,
    points,
  }
}

function checkCircuitJSON(circuit: unknown, at: string): CircuitJSON {
  if (!isRecord(circuit) || circuit['ket'] !== 1 || typeof circuit['qubits'] !== 'number' || !Array.isArray(circuit['ops']))
    throw new TypeError(`${at}.circuit is not ket circuit JSON`)
  return circuit as unknown as CircuitJSON
}

function parsePoint<P>(pt: unknown, at: string): ExperimentPoint<P> {
  {
    if (!isRecord(pt)) throw new TypeError(`${at} must be an object`)
    if (typeof pt['index'] !== 'number') throw new TypeError(`${at}.index must be a number`)
    const circuit = checkCircuitJSON(pt['circuit'], at)
    const probs = pt['probs']
    if (!isRecord(probs)) throw new TypeError(`${at}.probs must be an object`)
    for (const [bits, v] of Object.entries(probs))
      if (typeof v !== 'number' || !/^[01]+$/.test(bits)) throw new TypeError(`${at}.probs['${bits}'] is not a bitstring probability`)
    if (pt['shots'] !== undefined && typeof pt['shots'] !== 'number') throw new TypeError(`${at}.shots must be a number`)
    if (pt['backend'] !== undefined && typeof pt['backend'] !== 'string') throw new TypeError(`${at}.backend must be a string`)
    if (pt['jobId'] !== undefined && typeof pt['jobId'] !== 'string') throw new TypeError(`${at}.jobId must be a string`)
    if (pt['meta'] !== undefined && !isRecord(pt['meta'])) throw new TypeError(`${at}.meta must be an object`)
    if (typeof pt['ms'] !== 'number') throw new TypeError(`${at}.ms must be a number`)
    return {
      index: pt['index'],
      params: pt['params'] as P,
      circuit,
      probs: probs as Record<string, number>,
      shots: pt['shots'],
      backend: pt['backend'],
      jobId: pt['jobId'],
      meta: pt['meta'] as Record<string, unknown> | undefined,
      ms: pt['ms'],
    }
  }
}

/** Validate a stored pending record — the output of `submitSweep` or `collectSweep`. */
export function parsePendingExperiment<P = Record<string, unknown>>(input: string | unknown): PendingExperiment<P> {
  const raw: unknown = typeof input === 'string' ? JSON.parse(input) : input
  if (!isRecord(raw)) throw new TypeError('parsePendingExperiment: expected an object')
  if (raw['format'] !== 'ket-experiment-pending')
    throw new TypeError(`parsePendingExperiment: format is ${JSON.stringify(raw['format'])}, expected 'ket-experiment-pending'`)
  if (raw['formatVersion'] !== 1)
    throw new TypeError(`parsePendingExperiment: unsupported formatVersion ${String(raw['formatVersion'])}`)
  const str = (k: 'ket' | 'createdAt'): string => {
    const v = raw[k]
    if (typeof v !== 'string') throw new TypeError(`parsePendingExperiment: ${k} must be a string`)
    return v
  }
  const ket = str('ket'), createdAt = str('createdAt')
  if (raw['name'] !== undefined && typeof raw['name'] !== 'string')
    throw new TypeError('parsePendingExperiment: name must be a string')
  if (raw['meta'] !== undefined && !isRecord(raw['meta']))
    throw new TypeError('parsePendingExperiment: meta must be an object')
  if (!Array.isArray(raw['points'])) throw new TypeError('parsePendingExperiment: points must be an array')

  const points = (raw['points'] as unknown[]).map((pt, i): PendingPoint<P> => {
    const at = `parsePendingExperiment: points[${i}]`
    if (!isRecord(pt)) throw new TypeError(`${at} must be an object`)
    if (typeof pt['index'] !== 'number') throw new TypeError(`${at}.index must be a number`)
    const circuit = checkCircuitJSON(pt['circuit'], at)
    if (typeof pt['jobId'] !== 'string' || pt['jobId'] === '') throw new TypeError(`${at}.jobId must be a non-empty string`)
    if (typeof pt['submittedAt'] !== 'string') throw new TypeError(`${at}.submittedAt must be a string`)
    return {
      index: pt['index'],
      params: pt['params'] as P,
      circuit,
      jobId: pt['jobId'],
      submittedAt: pt['submittedAt'],
      result: pt['result'] === undefined ? undefined : parsePoint<P>(pt['result'], `${at}.result`),
    }
  })

  return {
    format: 'ket-experiment-pending',
    formatVersion: 1,
    ket,
    name: raw['name'],
    createdAt,
    meta: raw['meta'] as Record<string, unknown> | undefined,
    points,
  }
}
