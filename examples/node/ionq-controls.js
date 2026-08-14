/**
 * Probe whether IonQ accepts natively-controlled gates.
 *
 * Run:  IONQ_API_KEY=... node examples/node/ionq-controls.js
 *
 * IonQ's circuit format expresses controlled gates as a base gate plus a
 * `controls` array — `{gate: 'x', controls: [0, 1], target: 2}` is a Toffoli.
 * ket currently expands those into Clifford+T instead, turning one gate into
 * fifteen and manufacturing CNOTs the compiler then has to re-analyse.
 *
 * Each case sends a known input and asserts the one basis state that may come
 * back, so a pass means the semantics match, not merely that the job ran.
 * Cases are submitted separately so a failure identifies which form is at fault.
 */
import { submitIonQ, awaitIonQJob, getIonQResults } from '../../dist/ket.js'

const apiKey = process.env.IONQ_API_KEY
if (!apiKey) {
  console.error('Set IONQ_API_KEY. Get one at https://cloud.ionq.com/settings/keys')
  process.exit(1)
}

// Keys are little-endian: qubit i contributes 2^i.
const CASES = [
  {
    name: 'ccx via controls',
    detail: '|110> -> |111>',
    qubits: 3,
    circuit: [
      { gate: 'x', target: 0 },
      { gate: 'x', target: 1 },
      { gate: 'x', controls: [0, 1], target: 2 },
    ],
    expect: '7',
  },
  {
    name: 'ccx control unsatisfied',
    detail: '|100> unchanged — guards against a control that is silently ignored',
    qubits: 3,
    circuit: [
      { gate: 'x', target: 0 },
      { gate: 'x', controls: [0, 1], target: 2 },
    ],
    expect: '1',
  },
  {
    name: 'cswap via controls',
    detail: '|101> -> |110>',
    qubits: 3,
    circuit: [
      { gate: 'x', target: 0 },
      { gate: 'x', target: 2 },
      { gate: 'swap', targets: [1, 2], controls: [0] },
    ],
    expect: '3',
  },
  {
    name: 'controlled rz via controls',
    detail: 'H·cRz(pi)·H with control set -> |01>; exercises a rotation under control',
    qubits: 2,
    circuit: [
      { gate: 'x', target: 0 },
      { gate: 'h', target: 1 },
      { gate: 'rz', controls: [0], target: 1, rotation: Math.PI },
      { gate: 'h', target: 1 },
    ],
    expect: '3',
  },
]

let allOk = true
for (const c of CASES) {
  process.stdout.write(`${c.name.padEnd(28)} `)
  try {
    const { id } = await submitIonQ(
      { format: 'ionq.circuit.v0', qubits: c.qubits, circuit: c.circuit },
      { apiKey, target: 'simulator', shots: 1024, noise: { model: 'ideal' }, name: `ket probe: ${c.name}` },
    )
    const job = await awaitIonQJob(id, { apiKey, timeoutSeconds: 900 })
    const hist = await getIonQResults(job, { apiKey })
    const top = Object.entries(hist).sort((a, b) => b[1] - a[1])[0]
    const ok = top?.[0] === c.expect && top[1] > 0.99
    allOk &&= ok
    console.log(`${ok ? 'OK  ' : 'FAIL'}  got ${top?.[0]}:${((top?.[1] ?? 0) * 100).toFixed(0)}%  expected ${c.expect}:100%   (${c.detail})`)
  } catch (e) {
    allOk = false
    console.log(`REJECTED  ${String(e.message).slice(0, 120)}`)
  }
}

console.log(allOk
  ? '\nNative controls accepted. toIonQBasis() should emit them instead of expanding.'
  : '\nNative controls not usable as tested — keep the Clifford+T expansion.')
