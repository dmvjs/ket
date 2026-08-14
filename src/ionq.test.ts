import { describe, it, expect } from 'vitest'
import {
  ionqSubmitRequest, submitIonQ, getIonQJob, awaitIonQJob, cancelIonQJob,
  ionqHistogramToCounts, countsToProbs, getIonQResults, runIonQ,
} from './ionq.js'
import { Circuit } from './circuit.js'

const KEY = 'test-key-123'

/** A fetch stub that records calls and replays queued responses. */
function stubFetch(responses: (Partial<Response> & { json?: unknown })[]) {
  const calls: { url: string; init: RequestInit }[] = []
  let i = 0
  const f = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} })
    const r = responses[Math.min(i++, responses.length - 1)]!
    return {
      ok: r.ok ?? true,
      status: r.status ?? 200,
      statusText: r.statusText ?? 'OK',
      text: async () => JSON.stringify(r.json ?? {}),
    } as Response
  }) as unknown as typeof globalThis.fetch
  return { f, calls }
}

describe('ionqSubmitRequest — wire format', () => {
  const bell = new Circuit(2).h(0).cnot(0, 1).toIonQ()

  it('targets the jobs endpoint with the documented auth scheme', () => {
    const { url, init } = ionqSubmitRequest(bell, { apiKey: KEY })
    expect(url).toBe('https://api.ionq.co/v0.3/jobs')
    expect(init.method).toBe('POST')
    // IonQ uses `apiKey <key>`, not `Bearer <key>`.
    expect((init.headers as Record<string, string>)['Authorization']).toBe(`apiKey ${KEY}`)
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json')
  })

  it('nests the circuit under `input` and defaults target/shots', () => {
    const { init } = ionqSubmitRequest(bell, { apiKey: KEY })
    const body = JSON.parse(init.body as string)
    expect(body.target).toBe('simulator')
    expect(body.shots).toBe(100)
    expect(body.input.format).toBe('ionq.circuit.v0')
    expect(body.input.qubits).toBe(2)
    expect(body.input.circuit).toEqual([
      { gate: 'h', target: 0 },
      { gate: 'cnot', control: 0, target: 1 },
    ])
  })

  it('carries target, shots, name and gateset when given', () => {
    const { init } = ionqSubmitRequest(bell, {
      apiKey: KEY, target: 'qpu.aria-1', shots: 2048, name: 'bell', gateset: 'native',
    })
    const body = JSON.parse(init.body as string)
    expect(body.target).toBe('qpu.aria-1')
    expect(body.shots).toBe(2048)
    expect(body.name).toBe('bell')
    expect(body.input.gateset).toBe('native')
  })

  it('omits optional fields rather than sending undefined', () => {
    const body = JSON.parse(ionqSubmitRequest(bell, { apiKey: KEY }).init.body as string)
    expect('name' in body).toBe(false)
    expect('gateset' in body.input).toBe(false)
  })

  it('honours an endpoint override for proxying', () => {
    const { url } = ionqSubmitRequest(bell, { apiKey: KEY, endpoint: 'https://proxy.example/ionq' })
    expect(url).toBe('https://proxy.example/ionq/jobs')
  })

  it('allows an absent key when proxying, and omits the header', () => {
    // The documented browser pattern: the key lives on the server, so the
    // browser sends none. Requiring one here would break that pattern.
    const { init } = ionqSubmitRequest(bell, { apiKey: '', endpoint: '/api/ionq' })
    expect('Authorization' in (init.headers as Record<string, string>)).toBe(false)
    expect(() => ionqSubmitRequest(bell, { endpoint: '/api/ionq' })).not.toThrow()
  })

  it('rejects a missing key or nonsense shot count', () => {
    expect(() => ionqSubmitRequest(bell, { apiKey: '' })).toThrow(TypeError)
    expect(() => ionqSubmitRequest(bell, { apiKey: KEY, shots: 0 })).toThrow(RangeError)
    expect(() => ionqSubmitRequest(bell, { apiKey: KEY, shots: 1.5 })).toThrow(RangeError)
  })
})

describe('ionqSubmitRequest — noise model', () => {
  const bell = new Circuit(2).h(0).cnot(0, 1).toIonQ()
  const body = (opts: Parameters<typeof ionqSubmitRequest>[1]) =>
    JSON.parse(String(ionqSubmitRequest(bell, opts).init.body)) as Record<string, unknown>

  // `noise` is a sibling of target/shots, not part of `input`. Nesting it under
  // the circuit would be silently ignored by the server rather than rejected.
  it('sends noise at the top level of the job body', () => {
    const b = body({ apiKey: KEY, noise: { model: 'ideal' } })
    expect(b['noise']).toEqual({ model: 'ideal' })
    expect((b['input'] as Record<string, unknown>)['noise']).toBeUndefined()
  })

  it('carries a seed alongside the model', () => {
    expect(body({ apiKey: KEY, noise: { model: 'forte-1', seed: 100 } })['noise'])
      .toEqual({ model: 'forte-1', seed: 100 })
  })

  // Omission is meaningful: it defers to IonQ's server-side default, which is
  // not guaranteed to be noiseless. Sending an explicit null would not.
  it('omits the field entirely when unset', () => {
    expect('noise' in body({ apiKey: KEY })).toBe(false)
  })
})

describe('IonQ transport', () => {
  const bell = new Circuit(2).h(0).cnot(0, 1).toIonQ()

  it('returns the queued job on submit', async () => {
    const { f, calls } = stubFetch([{ json: { id: 'job-1', status: 'ready' } }])
    const job = await submitIonQ(bell, { apiKey: KEY, fetch: f })
    expect(job).toMatchObject({ id: 'job-1', status: 'ready' })
    expect(calls[0]!.url).toBe('https://api.ionq.co/v0.3/jobs')
  })

  it('fetches a job by id without a body', async () => {
    const { f, calls } = stubFetch([{ json: { id: 'job-1', status: 'running' } }])
    await getIonQJob('job-1', { apiKey: KEY, fetch: f })
    expect(calls[0]!.url).toBe('https://api.ionq.co/v0.3/jobs/job-1')
    expect(calls[0]!.init.method).toBe('GET')
    expect(calls[0]!.init.body).toBeUndefined()
  })

  it('surfaces HTTP errors with status and body', async () => {
    const { f } = stubFetch([{ ok: false, status: 401, statusText: 'Unauthorized', json: { error: 'bad key' } }])
    await expect(submitIonQ(bell, { apiKey: KEY, fetch: f })).rejects.toThrow(/401 Unauthorized/)
  })

  it('polls until completion', async () => {
    const { f, calls } = stubFetch([
      { json: { id: 'j', status: 'submitted' } },
      { json: { id: 'j', status: 'running' } },
      { json: { id: 'j', status: 'completed', data: { histogram: { '0': 1 } } } },
    ])
    const seen: string[] = []
    const job = await awaitIonQJob('j', {
      apiKey: KEY, fetch: f, intervalSeconds: 0, onPoll: j => seen.push(j.status),
    })
    expect(job.status).toBe('completed')
    expect(seen).toEqual(['submitted', 'running', 'completed'])
    expect(calls.length).toBe(3)
  })

  it('throws on a failed job rather than returning it', async () => {
    // A caller who forgets to check `status` must not read a failure as empty results.
    const { f } = stubFetch([{ json: { id: 'j', status: 'failed', failure: { error: 'too many qubits' } } }])
    await expect(awaitIonQJob('j', { apiKey: KEY, fetch: f, intervalSeconds: 0 }))
      .rejects.toThrow(/failed: too many qubits/)
  })

  it('times out instead of polling forever', async () => {
    const { f } = stubFetch([{ json: { id: 'j', status: 'running' } }])
    await expect(awaitIonQJob('j', { apiKey: KEY, fetch: f, intervalSeconds: 0, timeoutSeconds: -1 }))
      .rejects.toThrow(/still running/)
  })
})

describe('cancelIonQJob', () => {
  it('PUTs to the cancel path with auth and no body', async () => {
    const { f, calls } = stubFetch([{ json: { id: 'job-1', status: 'canceled' } }])
    const job = await cancelIonQJob('job-1', { apiKey: KEY, fetch: f })
    expect(job).toMatchObject({ status: 'canceled' })
    expect(calls[0]!.url).toBe('https://api.ionq.co/v0.3/jobs/job-1/status/cancel')
    expect(calls[0]!.init.method).toBe('PUT')
    expect(calls[0]!.init.body).toBeUndefined()
    expect((calls[0]!.init.headers as Record<string, string>)['Authorization']).toBe(`apiKey ${KEY}`)
  })

  it('honours an endpoint override and omits the header when proxying', async () => {
    const { f, calls } = stubFetch([{ json: { id: 'j', status: 'canceled' } }])
    await cancelIonQJob('j', { endpoint: '/api/ionq', fetch: f })
    expect(calls[0]!.url).toBe('/api/ionq/jobs/j/status/cancel')
    expect((calls[0]!.init.headers as Record<string, string>)['Authorization']).toBeUndefined()
  })

  it('surfaces an HTTP error rather than reporting success', async () => {
    const { f } = stubFetch([{ ok: false, status: 404, statusText: 'Not Found' }])
    await expect(cancelIonQJob('missing', { apiKey: KEY, fetch: f })).rejects.toThrow(/404/)
  })
})

describe('ionqHistogramToCounts — bit order', () => {
  // Settled empirically against the live v0.3 API, not from documentation:
  // submitting x(0) on two qubits returns {"1": 1.0}, and ket also indexes that
  // state as 1. The conventions match, so no reversal is applied. A previous
  // version reversed the bits on the strength of a secondary source; these tests
  // exist to stop that from coming back.
  it('leaves keys alone — IonQ is little-endian like ket', () => {
    // x(0): qubit 0 set. IonQ returns "1"; ket index is also 1.
    expect(ionqHistogramToCounts({ '1': 1 }, 1000).get(1n)).toBe(1000)
  })

  it('does not reverse an asymmetric outcome', () => {
    // The regression case. A reversal would send "1" to 4 on three qubits.
    const counts = ionqHistogramToCounts({ '1': 1 }, 500)
    expect(counts.get(1n)).toBe(500)
    expect(counts.get(4n)).toBeUndefined()
  })

  it('round-trips a Bell histogram to ket bitstrings', () => {
    const probs = countsToProbs(ionqHistogramToCounts({ '0': 0.5, '3': 0.5 }, 1024), 2, 1024)
    expect(probs['00']).toBeCloseTo(0.5, 6)
    expect(probs['11']).toBeCloseTo(0.5, 6)
  })

  it('renders bitstrings with qubit 0 first, matching Distribution', () => {
    expect(countsToProbs(new Map([[1n, 100]]), 2, 100)['10']).toBe(1)
  })

  it('drops outcomes that round to zero shots', () => {
    const counts = ionqHistogramToCounts({ '0': 0.9999, '1': 0.0001 }, 100)
    expect(counts.get(0n)).toBe(100)
    expect(counts.size).toBe(1)
  })
})

describe('getIonQResults', () => {
  it('follows results_url rather than reading the job object', async () => {
    // Results are not inlined on the job; assuming they were is what crashed the
    // first real submission.
    const { f, calls } = stubFetch([{ json: { '1': 1 } }])
    const hist = await getIonQResults(
      { id: 'j', status: 'completed', results_url: '/v0.3/jobs/j/results' },
      { apiKey: KEY, fetch: f })
    expect(hist).toEqual({ '1': 1 })
    expect(calls[0]!.url).toBe('https://api.ionq.co/v0.3/jobs/j/results')
  })

  it('falls back to the conventional path when results_url is absent', async () => {
    const { f, calls } = stubFetch([{ json: { '0': 1 } }])
    await getIonQResults({ id: 'j', status: 'completed' }, { apiKey: KEY, fetch: f })
    expect(calls[0]!.url).toBe('https://api.ionq.co/v0.3/jobs/j/results')
  })
})

describe('runIonQ', () => {
  it('submits, polls and fetches results in one call', async () => {
    const { f, calls } = stubFetch([
      { json: { id: 'j', status: 'ready' } },
      { json: { id: 'j', status: 'completed', results_url: '/v0.3/jobs/j/results' } },
      { json: { '0': 0.5, '3': 0.5 } },
    ])
    const { counts } = await runIonQ(new Circuit(2).h(0).cnot(0, 1).toIonQ(),
      { apiKey: KEY, fetch: f, shots: 1024, intervalSeconds: 0 })
    expect(counts.get(0n)).toBe(512)
    expect(counts.get(3n)).toBe(512)
    expect(calls.map(c => c.init.method ?? 'GET')).toEqual(['POST', 'GET', 'GET'])
  })
})
