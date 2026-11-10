import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { Circuit } from './circuit.js'
import { cartesian, collectSweep, experiment, finishSweep, parseExperiment, parsePendingExperiment, recordPoint, submitSweep, sweep } from './experiment.js'
import type { Experiment, Outcomes, PendingExperiment } from './experiment.js'
import { rbFit, rbSequences, rbSurvival, runChsh, runRb } from './protocols.js'
import { VERSION } from './version.js'

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as { version: string }

describe('VERSION', () => {
  it('matches package.json', () => {
    expect(VERSION).toBe(pkg.version)
  })
})

describe('cartesian', () => {
  it('enumerates every combination, first axis slowest', () => {
    expect(cartesian({ a: [0, 1], b: ['x', 'y', 'z'] })).toEqual([
      { a: 0, b: 'x' }, { a: 0, b: 'y' }, { a: 0, b: 'z' },
      { a: 1, b: 'x' }, { a: 1, b: 'y' }, { a: 1, b: 'z' },
    ])
  })

  it('handles one axis, an empty axis, and no axes', () => {
    expect(cartesian({ t: [1, 2, 3] })).toEqual([{ t: 1 }, { t: 2 }, { t: 3 }])
    expect(cartesian({ t: [1, 2], u: [] })).toEqual([])
    expect(cartesian({})).toEqual([{}])
  })
})

describe('sweep', () => {
  const ry = ({ theta }: { theta: number }) => new Circuit(1).ry(theta, 0)

  it('runs a grid in order and records what each point was and returned', async () => {
    const thetas = [0, Math.PI / 2, Math.PI]
    const exp = await sweep({ theta: thetas }, ry, c => c.run({ shots: 2000, seed: 1 }), { name: 'ry', meta: { by: 'test' } })
    expect(exp.format).toBe('ket-experiment')
    expect(exp.formatVersion).toBe(1)
    expect(exp.ket).toBe(VERSION)
    expect(exp.name).toBe('ry')
    expect(exp.meta).toEqual({ by: 'test' })
    expect(Date.parse(exp.createdAt)).toBeLessThanOrEqual(Date.parse(exp.finishedAt))
    expect(exp.points.map(p => p.params.theta)).toEqual(thetas)
    expect(exp.points.map(p => p.index)).toEqual([0, 1, 2])
    for (const p of exp.points) {
      expect(p.shots).toBe(2000)
      expect(p.backend).toBe('statevector')
      expect(p.ms).toBeGreaterThanOrEqual(0)
      expect(p.jobId).toBeUndefined()
      expect(Circuit.fromJSON(p.circuit).exactProbs()).toEqual(ry(p.params).exactProbs())
    }
    expect(exp.points[0]!.probs['0']).toBe(1)
    expect(exp.points[2]!.probs['1']).toBe(1)
    expect(Math.abs(exp.points[1]!.probs['1']! - 0.5)).toBeLessThan(0.05)
  })

  it('accepts an explicit parameter list of any type', async () => {
    const exp = await sweep([2, 3, 4], n => new Circuit(n).h(0), c => c.exactProbs())
    expect(exp.points.map(p => p.params)).toEqual([2, 3, 4])
    expect(exp.points.map(p => p.circuit.qubits)).toEqual([2, 3, 4])
    for (const p of exp.points) { expect(p.shots).toBeUndefined(); expect(p.backend).toBeUndefined() }
  })

  it('keeps provenance from a RunReport', async () => {
    const exp = await sweep([1], () => new Circuit(1).x(0), (c, i) => ({
      outcomes: c.exactProbs(), backend: 'ionq:simulator', jobId: `job-${i}`, meta: { shots: 100 },
    }))
    const p = exp.points[0]!
    expect(p.probs).toEqual({ '1': 1 })
    expect(p.backend).toBe('ionq:simulator')
    expect(p.jobId).toBe('job-0')
    expect(p.meta).toEqual({ shots: 100 })
    expect(p.shots).toBeUndefined()
  })

  it('takes shots from a RunReport so error bars survive a probability map', async () => {
    const exp = await sweep([1], () => new Circuit(1).h(0), c => ({ outcomes: c.run({ shots: 300 }).probs, shots: 300 }))
    expect(exp.points[0]!.shots).toBe(300)
    expect(exp.points[0]!.backend).toBeUndefined()
  })

  it('a RunReport backend label overrides the Distribution backend', async () => {
    const exp = await sweep([1], () => new Circuit(1), c => ({ outcomes: c.run({ shots: 10 }), backend: 'custom' }))
    expect(exp.points[0]!.backend).toBe('custom')
    expect(exp.points[0]!.shots).toBe(10)
  })

  it('runs up to `concurrency` circuits at once and still returns grid order', async () => {
    let inFlight = 0, peak = 0
    const exp = await sweep([5, 1, 3, 2, 4, 6], n => new Circuit(1).ry(n, 0), async c => {
      inFlight++; peak = Math.max(peak, inFlight)
      await new Promise(r => setTimeout(r, 5))
      inFlight--
      return c.exactProbs()
    }, { concurrency: 3 })
    expect(peak).toBe(3)
    expect(exp.points.map(p => p.params)).toEqual([5, 1, 3, 2, 4, 6])
    expect(exp.points.map(p => p.index)).toEqual([0, 1, 2, 3, 4, 5])
  })

  it('defaults to one at a time', async () => {
    let inFlight = 0, peak = 0
    await sweep([1, 2, 3], () => new Circuit(1), async c => {
      inFlight++; peak = Math.max(peak, inFlight)
      await new Promise(r => setTimeout(r, 2))
      inFlight--
      return c.exactProbs()
    })
    expect(peak).toBe(1)
  })

  it('propagates executor failures and rejects bad builders and options', async () => {
    await expect(sweep([1, 2], () => new Circuit(1), (_, i) => { if (i === 1) throw new Error('boom'); return { '0': 1 } }))
      .rejects.toThrow('boom')
    await expect(sweep([1], () => 'nope' as unknown as Circuit, c => c.exactProbs())).rejects.toThrow(TypeError)
    await expect(sweep([1], () => new Circuit(1), c => c.exactProbs(), { concurrency: 0 })).rejects.toThrow(RangeError)
  })

  it('handles an empty grid', async () => {
    const exp = await sweep([], () => new Circuit(1), c => c.exactProbs())
    expect(exp.points).toEqual([])
  })
})

describe('serialisation', () => {
  it('round-trips through JSON and parseExperiment', async () => {
    const exp = await sweep({ theta: [0.3, 1.1], q: [1, 2] },
      ({ theta, q }) => new Circuit(q).ry(theta, 0), c => c.run({ shots: 500, seed: 2 }), { name: 'rt', meta: { k: [1, 2] } })
    const text = JSON.stringify(exp)
    const back = parseExperiment<{ theta: number; q: number }>(text)
    expect(back).toEqual(JSON.parse(text))
    expect(back.points).toHaveLength(4)
    expect(back.name).toBe('rt')
    expect(back.meta).toEqual({ k: [1, 2] })
    for (const [i, p] of back.points.entries()) {
      expect(p.params).toEqual(exp.points[i]!.params)
      expect(p.probs).toEqual(exp.points[i]!.probs)
      expect(Circuit.fromJSON(p.circuit).exactProbs()).toEqual(Circuit.fromJSON(exp.points[i]!.circuit).exactProbs())
    }
    // Optional fields that JSON dropped come back as undefined, not missing-and-different.
    expect(back.points[0]!.jobId).toBeUndefined()
    expect(parseExperiment(back)).toEqual(back)
  })

  it('reports the first wrong field by path', async () => {
    const good = JSON.parse(JSON.stringify(await sweep([1], () => new Circuit(1), c => c.exactProbs()))) as Record<string, unknown>
    const pt = () => (structuredClone(good) as { points: Record<string, unknown>[] })
    expect(() => parseExperiment('[]')).toThrow(/expected an object/)
    expect(() => parseExperiment({ ...good, format: 'x' })).toThrow(/format is "x"/)
    expect(() => parseExperiment({ ...good, formatVersion: 2 })).toThrow(/formatVersion 2/)
    expect(() => parseExperiment({ ...good, ket: 1 })).toThrow(/ket must be a string/)
    expect(() => parseExperiment({ ...good, points: {} })).toThrow(/points must be an array/)
    let bad = pt(); bad.points[0]!['circuit'] = { ket: 2 }
    expect(() => parseExperiment(bad)).toThrow(/points\[0\]\.circuit/)
    bad = pt(); bad.points[0]!['probs'] = { '0x': 1 }
    expect(() => parseExperiment(bad)).toThrow(/probs\['0x'\]/)
    bad = pt(); bad.points[0]!['probs'] = { '0': '1' }
    expect(() => parseExperiment(bad)).toThrow(/probs\['0'\]/)
    bad = pt(); bad.points[0]!['shots'] = '10'
    expect(() => parseExperiment(bad)).toThrow(/shots must be a number/)
    bad = pt(); delete bad.points[0]!['ms']
    expect(() => parseExperiment(bad)).toThrow(/ms must be a number/)
    expect(() => parseExperiment('not json')).toThrow(SyntaxError)
  })
})

describe('recordPoint and experiment', () => {
  it('build a record from results gathered out of band', () => {
    const c = new Circuit(2).h(0).cnot(0, 1)
    const fromHardware: Outcomes = { '00': 0.49, '11': 0.51 }
    const p = recordPoint(0, { device: 'forte-1' }, c, { outcomes: fromHardware, backend: 'ionq:qpu.forte-1', jobId: 'abc' }, 12.5)
    const exp: Experiment<{ device: string }> = experiment([p], { name: 'bell-on-forte' })
    expect(exp.points[0]!.probs).toEqual(fromHardware)
    expect(exp.points[0]!.ms).toBe(12.5)
    expect(exp.points[0]!.jobId).toBe('abc')
    expect(exp.ket).toBe(VERSION)
    expect(parseExperiment(JSON.stringify(exp)).points[0]!.params).toEqual({ device: 'forte-1' })
  })

  it('copies probabilities so later mutation of the source does not change the record', () => {
    const src: Record<string, number> = { '0': 1 }
    const p = recordPoint(0, null, new Circuit(1), src)
    src['0'] = 0.5
    expect(p.probs['0']).toBe(1)
  })
})

describe('protocols on top of sweep', () => {
  it('RB sequences are a grid; the record feeds rbFit', async () => {
    const seqs = rbSequences({ lengths: [1, 4, 16, 64], samples: 6, seed: 3 })
    const exp = await sweep(seqs, s => s.circuit, (c, i) => c.runClifford({ shots: 500, seed: i + 1, noise: { p1: 0.003 } }), { name: 'rb-q0' })
    expect(exp.points).toHaveLength(24)
    expect(exp.points[5]!.params.length).toBe(1)
    const fit = rbFit(exp.points.map(p => ({ length: p.params.length, survival: rbSurvival(p.probs) })), { bootstrap: 0 })
    expect(fit.errorPerClifford).toBeGreaterThan(0.002)
    expect(fit.errorPerClifford).toBeLessThan(0.012)
    // And the record is still plain JSON with the sequence labels inside.
    const back = parseExperiment<{ length: number; sample: number }>(JSON.stringify(exp))
    expect(back.points[23]!.params).toMatchObject({ length: 64, sample: 5 })
  })

  it('protocol runners accept executors that return RunReports', async () => {
    const chsh = await runChsh(c => ({ outcomes: c.exactProbs(), backend: 'exact' }))
    expect(chsh.S).toBeCloseTo(2 * Math.SQRT2, 10)
    const rb = await runRb(c => ({ outcomes: c.exactProbs() }), { lengths: [1, 2, 4], samples: 2, bootstrap: 0 })
    expect(rb.errorPerClifford).toBeLessThan(1e-9)
  })
})

describe('resumable sweeps', () => {
  const grid = { theta: [0.2, 0.7, 1.4, 2.5] }
  const build = ({ theta }: { theta: number }) => new Circuit(2).ry(theta, 0).cnot(0, 1)

  /** An in-memory queue standing in for a hardware vendor. */
  function fakeQueue() {
    const jobs = new Map<string, { circuit: Circuit; done: boolean }>()
    let n = 0
    return {
      jobs,
      submit: (c: Circuit) => { const id = `job-${n++}`; jobs.set(id, { circuit: c, done: false }); return id },
      finish: (pred: (id: string) => boolean) => { for (const [id, j] of jobs) if (pred(id)) j.done = true },
      collect: (id: string) => {
        const j = jobs.get(id)
        if (!j) throw new Error(`unknown job ${id}`)
        return j.done ? { outcomes: j.circuit.run({ shots: 200, seed: 1 }), backend: 'fake' } : undefined
      },
    }
  }

  it('submitSweep records a job id per point and nothing else', async () => {
    const q = fakeQueue()
    const pending = await submitSweep(grid, build, q.submit, { name: 'resume', meta: { k: 1 } })
    expect(pending.format).toBe('ket-experiment-pending')
    expect(pending.ket).toBe(VERSION)
    expect(pending.name).toBe('resume')
    expect(pending.points.map(p => p.jobId)).toEqual(['job-0', 'job-1', 'job-2', 'job-3'])
    expect(pending.points.map(p => p.params.theta)).toEqual(grid.theta)
    for (const p of pending.points) {
      expect(p.result).toBeUndefined()
      expect(Date.parse(p.submittedAt)).not.toBeNaN()
      expect(Circuit.fromJSON(p.circuit).exactProbs()).toEqual(build(p.params).exactProbs())
    }
    expect(parsePendingExperiment(JSON.stringify(pending))).toEqual(JSON.parse(JSON.stringify(pending)))
  })

  it('collectSweep fills in finished jobs, leaves the rest, and finishSweep refuses until all are in', async () => {
    const q = fakeQueue()
    const p0 = await submitSweep(grid, build, q.submit, { name: 'resume' })
    const stored = JSON.stringify(p0)

    q.finish(id => id === 'job-1' || id === 'job-3')
    const p1 = await collectSweep(parsePendingExperiment<{ theta: number }>(stored), q.collect)
    expect(p1.points.map(p => p.result !== undefined)).toEqual([false, true, false, true])
    expect(p1.points[1]!.result!.backend).toBe('fake')
    expect(p1.points[1]!.result!.jobId).toBe('job-1')   // job id carried into the result
    expect(p1.points[1]!.result!.shots).toBe(200)
    expect(p0.points[1]!.result).toBeUndefined()         // input untouched
    expect(() => finishSweep(p1)).toThrow(/2 of 4 points still pending \(indices 0, 2\)/)

    q.finish(() => true)
    const p2 = await collectSweep(parsePendingExperiment<{ theta: number }>(JSON.stringify(p1)), q.collect)
    expect(p2.points.every(p => p.result !== undefined)).toBe(true)
    const exp = finishSweep(p2)
    expect(exp.format).toBe('ket-experiment')
    expect(exp.name).toBe('resume')
    expect(exp.createdAt).toBe(p0.createdAt)
    expect(exp.points.map(p => p.index)).toEqual([0, 1, 2, 3])
    expect(exp.points.map(p => p.params.theta)).toEqual(grid.theta)
    expect(exp.points[0]!.probs).toEqual(build({ theta: 0.2 }).run({ shots: 200, seed: 1 }).probs)
    expect(parseExperiment(JSON.stringify(exp))).toEqual(JSON.parse(JSON.stringify(exp)))
  })

  it('a collector that throws aborts the call and the record is unchanged', async () => {
    const q = fakeQueue()
    const pending = await submitSweep(grid, build, q.submit)
    await expect(collectSweep(pending, () => { throw new Error('vendor down') })).rejects.toThrow('vendor down')
    expect(pending.points.every(p => p.result === undefined)).toBe(true)
  })

  it('submits and collects concurrently, keeping order', async () => {
    let inFlight = 0, peak = 0
    const tick = async () => { inFlight++; peak = Math.max(peak, inFlight); await new Promise(r => setTimeout(r, 3)); inFlight-- }
    const pending = await submitSweep([1, 2, 3, 4, 5], n => new Circuit(1).ry(n, 0), async (_, i) => { await tick(); return `j${i}` }, { concurrency: 2 })
    expect(peak).toBe(2)
    expect(pending.points.map(p => p.jobId)).toEqual(['j0', 'j1', 'j2', 'j3', 'j4'])
    peak = 0
    const done = await collectSweep(pending, async (_, c) => { await tick(); return c.exactProbs() }, { concurrency: 3 })
    expect(peak).toBe(3)
    expect(finishSweep(done).points.map(p => p.params)).toEqual([1, 2, 3, 4, 5])
  })

  it('validates submissions and stored pending records', async () => {
    await expect(submitSweep([1], () => new Circuit(1), () => '')).rejects.toThrow(/no job id/)
    await expect(submitSweep([1], () => 'x' as unknown as Circuit, () => 'j')).rejects.toThrow(TypeError)
    await expect(submitSweep([1], () => new Circuit(1), () => 'j', { concurrency: 0 })).rejects.toThrow(RangeError)
    const good = JSON.parse(JSON.stringify(await submitSweep([1], () => new Circuit(1), () => 'j'))) as PendingExperiment
    expect(() => parsePendingExperiment({ ...good, format: 'ket-experiment' })).toThrow(/format is "ket-experiment"/)
    const bad = structuredClone(good) as { points: Record<string, unknown>[] }
    bad.points[0]!['jobId'] = ''
    expect(() => parsePendingExperiment(bad)).toThrow(/jobId/)
    const badResult = structuredClone(good) as { points: Record<string, unknown>[] }
    badResult.points[0]!['result'] = { index: 0 }
    expect(() => parsePendingExperiment(badResult)).toThrow(/points\[0\]\.result\.circuit/)
  })
})
