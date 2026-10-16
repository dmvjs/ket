import {
  Circuit, analyzeHardness, qaoa, realAmplitudes, efficientSU2, trotter,
  particleConservingAnsatz, trotterIsing, rcsCircuit, type PauliTerm,
} from '@kirkelliott/ket'

const prng = (s: number) => () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff

const line = (n: number): [number, number][] =>
  Array.from({ length: n - 1 }, (_, i) => [i, i + 1] as [number, number])
const ring = (n: number): [number, number][] => [...line(n), [n - 1, 0] as [number, number]]
const regular3 = (n: number, seed: number): [number, number][] => {
  const r = prng(seed), out = new Set<string>()
  for (const [u, v] of ring(n)) out.add(`${u},${v}`)
  for (let k = 0; k < n / 2; k++) {
    const u = Math.floor(r() * n); let v = Math.floor(r() * n)
    if (u !== v) out.add(`${Math.min(u, v)},${Math.max(u, v)}`)
  }
  return [...out].map(s => s.split(',').map(Number) as [number, number])
}
const randomHamiltonian = (n: number, terms: number, seed: number): PauliTerm[] => {
  const r = prng(seed), out: PauliTerm[] = []
  for (let k = 0; k < terms; k++) {
    const ops = Array<string>(n).fill('I')
    const a = Math.floor(r() * n); let b = Math.floor(r() * n)
    if (b === a) b = (a + 1) % n
    const p = ['X', 'Y', 'Z'] as const
    ops[a] = p[Math.floor(r() * 3)]!; ops[b] = p[Math.floor(r() * 3)]!
    out.push({ coeff: r() * 2 - 1, ops: ops.join('') })
  }
  return out
}
const params = (k: number, seed: number): number[] => {
  const r = prng(seed); return Array.from({ length: k }, () => r() * 2 * Math.PI)
}

type Case = { family: string; label: string; build: () => Circuit }
const cases: Case[] = []
for (const n of [32, 36, 40, 48]) {
  for (const seed of [1, 2, 3]) {
    for (const reps of [2, 4, 8]) {
      cases.push({ family: 'realAmplitudes', label: `n=${n} reps=${reps} s=${seed}`,
        build: () => realAmplitudes(n, reps)(params(n * (reps + 1), seed)) })
      cases.push({ family: 'efficientSU2', label: `n=${n} reps=${reps} s=${seed}`,
        build: () => efficientSU2(n, reps)(params(2 * n * (reps + 1), seed)) })
      cases.push({ family: 'particleConserving', label: `n=${n} layers=${reps} s=${seed}`,
        build: () => particleConservingAnsatz({ qubits: n, layers: reps, seed }) })
    }
    for (const p of [1, 2, 4]) {
      cases.push({ family: 'qaoa-3regular', label: `n=${n} p=${p} s=${seed}`,
        build: () => qaoa(n, regular3(n, seed), params(p, seed), params(p, seed + 7)) })
      cases.push({ family: 'qaoa-ring', label: `n=${n} p=${p} s=${seed}`,
        build: () => qaoa(n, ring(n), params(p, seed), params(p, seed + 7)) })
    }
    for (const steps of [2, 4, 8]) {
      cases.push({ family: 'trotter-random', label: `n=${n} steps=${steps} s=${seed}`,
        build: () => trotter(n, randomHamiltonian(n, n, seed), 0.5, steps) })
    }
    for (const depth of [4, 8, 16]) {
      cases.push({ family: 'trotterIsing', label: `n=${n} d=${depth} s=${seed}`,
        build: () => trotterIsing({ qubits: n, coupling: line(n), depth, thetaH: Math.PI / 4 }) })
    }
  }
}
// Controls: circuits that must NOT land.
for (const [rows, cols, depth] of [[6, 6, 14], [5, 8, 12], [6, 8, 20]] as const) {
  cases.push({ family: 'RCS-control', label: `${rows}x${cols} d=${depth}`,
    build: () => rcsCircuit({ rows, cols, depth, seed: 1 }) })
}

const t0 = performance.now()
type Row = { family: string; label: string; qubits: number; attack: string; bound: string; adv: number | null }
const landed: Row[] = []
let analysed = 0, errored = 0
const byFamily = new Map<string, { n: number; hit: number }>()

for (const c of cases) {
  let circuit: Circuit
  try { circuit = c.build() } catch { errored++; continue }
  const obs: PauliTerm[] = [{ coeff: 1, ops: 'Z' + 'I'.repeat(circuit.qubits - 1) }]
  let r
  try { r = analyzeHardness(circuit, { memoryBytes: 8e9, seconds: 3600, observable: obs }) }
  catch { errored++; continue }
  analysed++
  const f = byFamily.get(c.family) ?? { n: 0, hit: 0 }
  f.n++
  if (r.best) {
    f.hit++
    landed.push({ family: c.family, label: c.label, qubits: circuit.qubits,
      attack: r.best.name, bound: r.best.parameter?.bound ?? '—', adv: r.advantage })
  }
  byFamily.set(c.family, f)
}

console.log(`\nanalysed ${analysed} circuits (${errored} build/analysis errors) in ${((performance.now() - t0) / 1000).toFixed(1)}s\n`)
console.log('family                 cases  landed')
for (const [k, v] of [...byFamily].sort()) console.log(`${k.padEnd(22)} ${String(v.n).padStart(5)}  ${String(v.hit).padStart(6)}`)

const exact = landed.filter(r => r.bound === 'exact')
console.log(`\n${landed.length} circuits had a viable attack; ${exact.length} of those with an EXACT bound`)
const shown = exact.sort((a, b) => (b.adv ?? 0) - (a.adv ?? 0)).slice(0, 25)
if (shown.length) {
  console.log('\nEXACT-bound landings, biggest advantage first:')
  console.log('family                 case                    n   attack           advantage')
  for (const r of shown) {
    console.log(`${r.family.padEnd(22)} ${r.label.padEnd(22)} ${String(r.qubits).padStart(3)}  ${r.attack.padEnd(15)}  ${r.adv === null ? '—' : r.adv.toExponential(2)}`)
  }
}
const byAttack = new Map<string, number>()
for (const r of landed) byAttack.set(r.attack, (byAttack.get(r.attack) ?? 0) + 1)
console.log('\nwinning attack, where anything landed:')
for (const [a, n] of [...byAttack].sort((x, y) => y[1] - x[1])) console.log(`  ${a.padEnd(18)} ${n}`)
