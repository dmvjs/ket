/**
 * Refresh ket's IonQ device entries from IonQ's own API.
 *
 * Run:  IONQ_API_KEY=... node scripts/refresh-ionq-devices.ts
 *
 * Emits the `DEVICES` block to paste into `src/circuit.ts`, so the table is
 * regenerated from the vendor rather than hand-maintained and left to rot.
 *
 * Two endpoints are used:
 *   GET /v0.3/backends                    — public: fleet, status, native gates
 *   GET /v0.3/characterizations/backends  — authenticated: measured error rates
 *
 * Without a key it still reports fleet and status, which is what goes stale
 * fastest — devices retire, and a table that lists a retired device sends users
 * at hardware that will reject them.
 */
const API = 'https://api.ionq.co/v0.3'
const apiKey = process.env.IONQ_API_KEY

interface Backend {
  backend: string
  status: string
  degraded: boolean
  qubits: number
  last_updated: number
  location?: string
  supported_native_gates?: string[]
  average_queue_time?: number
}

/** Characterization payloads vary by device; only the fields ket needs are typed. */
interface Characterization {
  backend?: string
  qubits?: number
  fidelity?: { '1q'?: { mean?: number }; '2q'?: { mean?: number }; spam?: { mean?: number } }
  timing?: Record<string, number>
  date?: number
}

const get = async <T>(path: string, auth: boolean): Promise<T | null> => {
  const res = await fetch(`${API}${path}`, {
    headers: auth && apiKey ? { Authorization: `apiKey ${apiKey}` } : {},
  })
  if (!res.ok) {
    console.error(`  ! ${path} → ${res.status} ${res.statusText}`)
    return null
  }
  return res.json() as Promise<T>
}

const backends = (await get<Backend[]>('/backends', false)) ?? []
const qpus = backends.filter(b => b.backend.startsWith('qpu.'))

console.log(`IonQ fleet as of ${new Date().toISOString().slice(0, 10)}\n`)
for (const b of qpus) {
  const flag = b.status === 'available' ? (b.degraded ? 'DEGRADED' : 'ok') : b.status.toUpperCase()
  console.log(
    `  ${b.backend.padEnd(24)} ${flag.padEnd(10)} ${String(b.qubits).padStart(3)}q  ` +
    `native=[${(b.supported_native_gates ?? []).join(', ')}]  ${b.location ?? ''}`)
}

if (!apiKey) {
  console.log('\nSet IONQ_API_KEY to also pull measured error rates.')
} else {
  console.log('\nCharacterizations:')
  const chars = await get<Characterization[]>('/characterizations/backends', true)
  if (chars) {
    for (const c of chars) {
      const f = c.fidelity ?? {}
      console.log(
        `  ${(c.backend ?? '?').padEnd(24)} ` +
        `1q=${f['1q']?.mean ?? '?'}  2q=${f['2q']?.mean ?? '?'}  spam=${f.spam?.mean ?? '?'}`)
    }
    console.log('\nRaw (first entry), so unmapped fields are visible:')
    console.log(JSON.stringify(chars[0], null, 2).slice(0, 1200))
  }
}

console.log(`
Note: ket's noise params are depolarizing error *rates* (p = 1 - fidelity).
Only devices reporting status 'available' belong in DEVICES; retired entries
point users at hardware that will reject their jobs.`)
