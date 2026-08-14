/**
 * IonQ Quantum Cloud client — submit circuits to real hardware.
 *
 * Uses the global `fetch`, so it adds no dependency and runs anywhere ket does.
 * Request construction is separated from transport (`ionqSubmitRequest` vs
 * `submitIonQ`) so the wire format can be tested without credentials or network,
 * and so callers can route through their own proxy or auth wrapper.
 *
 * Browser note: `api.ionq.co` does not send CORS headers, so a browser cannot
 * call it directly — point `endpoint` at your own proxy. Never ship an API key
 * to a browser regardless; it is a bearer credential for paid hardware time.
 *
 * Reference: https://docs.ionq.com/api-reference/v0.3/jobs/create-a-job
 */
import type { IonQCircuit } from './circuit.js'

const DEFAULT_ENDPOINT = 'https://api.ionq.co/v0.3'

/** Job lifecycle as reported by the API. */
export type IonQJobStatus =
  | 'ready' | 'submitted' | 'running' | 'completed' | 'failed' | 'canceled'

export interface IonQJob {
  id: string
  status: IonQJobStatus
  /** Where to fetch the histogram once complete. Results are not inlined. */
  results_url?: string
  failure?: { code?: string; error?: string }
  [k: string]: unknown
}

export interface IonQOptions {
  /**
   * IonQ API key, sent as `Authorization: apiKey <key>`.
   *
   * Required when talking to IonQ directly. Omit it when `endpoint` points at
   * your own proxy — the key belongs on the server, and the header is then left
   * off entirely rather than sent empty.
   */
  apiKey?: string
  /** Override the API root, e.g. to route through a CORS proxy. */
  endpoint?: string
  /** Injectable transport; defaults to the global `fetch`. */
  fetch?: typeof globalThis.fetch
}

export interface IonQSubmitOptions extends IonQOptions {
  /** `'simulator'` (default) or a QPU such as `'qpu.aria-1'`. */
  target?: string
  /** Shot count. IonQ defaults to 100. */
  shots?: number
  /** Optional job name, surfaced in the IonQ console. */
  name?: string
  /**
   * `'qis'` (default) treats the circuit as abstract gates; `'native'` says it
   * is already compiled to the target's native gate set.
   */
  gateset?: 'qis' | 'native'
  /**
   * Noise model for the `simulator` target, e.g. `{ model: 'ideal' }` or
   * `{ model: 'forte-1', seed: 100 }`.
   *
   * Omitting this leaves the choice to IonQ's server-side default, which is not
   * guaranteed to be noiseless. State it explicitly whenever the distinction
   * matters: a deep circuit run under a device model decoheres into a flat
   * distribution that is easily mistaken for a wrong answer.
   */
  noise?: { model: string; seed?: number }
}

/**
 * Build the HTTP request for a job submission without sending it.
 *
 * Exposed so the wire format is testable offline — the shape of this request is
 * the part that silently breaks when a vendor API drifts.
 */
export function ionqSubmitRequest(
  circuit: IonQCircuit,
  { apiKey, target = 'simulator', shots = 100, name, gateset, noise, endpoint = DEFAULT_ENDPOINT }: IonQSubmitOptions,
): { url: string; init: RequestInit } {
  if (!apiKey && endpoint === DEFAULT_ENDPOINT)
    throw new TypeError('an IonQ apiKey is required when calling api.ionq.co directly; omit it only when `endpoint` is your own proxy')
  if (!Number.isInteger(shots) || shots < 1) throw new RangeError(`shots must be a positive integer, got ${shots}`)
  const input: Record<string, unknown> = { ...circuit }
  if (gateset !== undefined) input['gateset'] = gateset
  const body: Record<string, unknown> = { target, shots, input }
  if (name !== undefined) body['name'] = name
  if (noise !== undefined) body['noise'] = noise
  return {
    url: `${endpoint}/jobs`,
    init: {
      method: 'POST',
      headers: {
        ...(apiKey ? { 'Authorization': `apiKey ${apiKey}` } : {}),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    },
  }
}

async function send(url: string, init: RequestInit, f: typeof globalThis.fetch): Promise<IonQJob> {
  const res = await f(url, init)
  const text = await res.text()
  if (!res.ok) throw new Error(`IonQ ${init.method ?? 'GET'} ${url} failed: ${res.status} ${res.statusText} — ${text}`)
  return JSON.parse(text) as IonQJob
}

/** Submit a circuit. Returns as soon as the job is queued, not when it finishes. */
export async function submitIonQ(circuit: IonQCircuit, opts: IonQSubmitOptions): Promise<IonQJob> {
  const { url, init } = ionqSubmitRequest(circuit, opts)
  return send(url, init, opts.fetch ?? globalThis.fetch)
}

/** Fetch a job's current state. */
export async function getIonQJob(id: string, opts: IonQOptions): Promise<IonQJob> {
  const endpoint = opts.endpoint ?? DEFAULT_ENDPOINT
  return send(`${endpoint}/jobs/${id}`, {
    method: 'GET',
    headers: opts.apiKey ? { 'Authorization': `apiKey ${opts.apiKey}` } : {},
  }, opts.fetch ?? globalThis.fetch)
}

/**
 * Cancel a queued or running job.
 *
 * Jobs are serialized per account, so one long-running submission holds up
 * everything behind it. Cancelling releases the slot; already-completed jobs
 * are unaffected. Returns the job in its post-cancellation state.
 */
export async function cancelIonQJob(id: string, opts: IonQOptions): Promise<IonQJob> {
  const endpoint = opts.endpoint ?? DEFAULT_ENDPOINT
  return send(`${endpoint}/jobs/${id}/status/cancel`, {
    method: 'PUT',
    headers: opts.apiKey ? { 'Authorization': `apiKey ${opts.apiKey}` } : {},
  }, opts.fetch ?? globalThis.fetch)
}

export interface IonQPollOptions extends IonQOptions {
  /** Seconds between polls. Default 2. */
  intervalSeconds?: number
  /** Give up after this many seconds. Default 3600 — QPU queues are long. */
  timeoutSeconds?: number
  /** Called after each poll, for progress reporting. */
  onPoll?: (job: IonQJob) => void
}

/**
 * Poll until the job reaches a terminal state.
 *
 * Rejects on `failed`/`canceled` rather than returning them, so a caller that
 * forgets to check `status` cannot mistake a failure for an empty result.
 */
export async function awaitIonQJob(
  id: string,
  { intervalSeconds = 2, timeoutSeconds = 3600, onPoll, ...opts }: IonQPollOptions,
): Promise<IonQJob> {
  const deadline = Date.now() + timeoutSeconds * 1000
  for (;;) {
    const job = await getIonQJob(id, opts)
    onPoll?.(job)
    if (job.status === 'completed') return job
    if (job.status === 'failed' || job.status === 'canceled') {
      throw new Error(`IonQ job ${id} ${job.status}: ${job.failure?.error ?? 'no reason given'}`)
    }
    if (Date.now() > deadline) throw new Error(`IonQ job ${id} still ${job.status} after ${timeoutSeconds}s`)
    await new Promise(r => setTimeout(r, intervalSeconds * 1000))
  }
}

/**
 * Fetch the probability histogram for a completed job.
 *
 * Results are not inlined on the job object — `GET /jobs/{id}` returns metadata
 * and a `results_url`, which this follows. Verified against the live v0.3 API.
 */
export async function getIonQResults(job: IonQJob, opts: IonQOptions): Promise<Record<string, number>> {
  const endpoint = opts.endpoint ?? DEFAULT_ENDPOINT
  const url = job.results_url
    ? (job.results_url.startsWith('http') ? job.results_url : `${endpoint.replace(/\/v\d+(\.\d+)?$/, '')}${job.results_url}`)
    : `${endpoint}/jobs/${job.id}/results`
  const res = await (opts.fetch ?? globalThis.fetch)(url, {
    headers: opts.apiKey ? { 'Authorization': `apiKey ${opts.apiKey}` } : {},
  })
  const text = await res.text()
  if (!res.ok) throw new Error(`IonQ GET ${url} failed: ${res.status} ${res.statusText} — ${text}`)
  return JSON.parse(text) as Record<string, number>
}

/**
 * Convert an IonQ probability histogram to shot counts keyed by basis index.
 *
 * **No bit reversal is applied, and that is deliberate.** IonQ keys are
 * little-endian integers with qubit i at 2^i, which is exactly ket's convention.
 * Verified empirically rather than from documentation: `x(0)` on two qubits
 * returns `{"1": 1.0}`, and `x(0).x(2)` on four returns `{"5": 1.0}` — both the
 * indices ket assigns. An earlier
 * version of this file reversed the bits on the strength of a secondary source
 * describing the keys as big-endian; that would have silently corrupted every
 * asymmetric result while leaving symmetric ones such as a Bell state correct.
 *
 * IonQ returns probabilities, so counts are reconstructed as `round(p · shots)`.
 */
export function ionqHistogramToCounts(histogram: Record<string, number>, shots: number): Map<bigint, number> {
  const out = new Map<bigint, number>()
  for (const [key, p] of Object.entries(histogram)) {
    const count = Math.round(p * shots)
    if (count > 0) out.set(BigInt(key), (out.get(BigInt(key)) ?? 0) + count)
  }
  return out
}

/** Render counts as ket-convention bitstrings (`bits[0]` is qubit 0). */
export function countsToProbs(counts: Map<bigint, number>, qubits: number, shots: number): Record<string, number> {
  const probs: Record<string, number> = {}
  for (const [idx, n] of counts) {
    const bits = idx.toString(2).padStart(qubits, '0').split('').reverse().join('')
    probs[bits] = n / shots
  }
  return probs
}

export interface RunIonQOptions extends IonQSubmitOptions, IonQPollOptions {}

/**
 * Submit, wait, and fetch results in one call.
 *
 * The three-step form exists for callers who want the job id or their own
 * polling; this is the path that cannot be got wrong — in particular it removes
 * the chance of reading results off the job object, which does not carry them.
 */
export async function runIonQ(
  circuit: IonQCircuit,
  opts: RunIonQOptions,
): Promise<{ job: IonQJob; histogram: Record<string, number>; counts: Map<bigint, number> }> {
  const shots = opts.shots ?? 100
  const queued = await submitIonQ(circuit, opts)
  const job = await awaitIonQJob(queued.id, opts)
  const histogram = await getIonQResults(job, opts)
  return { job, histogram, counts: ionqHistogramToCounts(histogram, shots) }
}
