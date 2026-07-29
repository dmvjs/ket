/**
 * SVG chart renderer for benchmark data.
 * Generates dual-panel charts (time | memory) with log-scale y-axes.
 *
 * Usage: import { renderChart } from './chart.js'
 */

const C_TIME = '#4f86f7'
const C_MEM  = '#e05c5c'

function logTicks(yMin: number, yMax: number): number[] {
  const result: number[] = []
  const eMin = Math.floor(Math.log10(yMin))
  const eMax = Math.ceil(Math.log10(yMax))
  for (let e = eMin; e <= eMax; e++) {
    const v = 10 ** e
    if (v >= yMin * 0.4 && v <= yMax * 2.5) result.push(v)
  }
  return result.length ? result : [yMin, yMax]
}

function fmtTime(ms: number): string {
  if (ms < 0.001) return '<1µs'
  if (ms < 1)     return `${(ms * 1000).toFixed(0)}µs`
  if (ms < 10)    return `${ms.toFixed(1)}ms`
  if (ms < 1000)  return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function fmtMem(mb: number): string {
  if (mb < 1e-3) return `${(mb * 1024 * 1024).toFixed(0)}B`
  if (mb < 1)    return `${(mb * 1024).toFixed(0)}KB`
  if (mb < 10)   return `${mb.toFixed(1)}MB`
  return `${Math.round(mb)}MB`
}

/** One (x, y) series; y is null when the measurement is missing. */
interface Point { x: number; y: number | null }

interface PanelOpts {
  x0: number
  plotW: number
  plotH: number
  mt: number
  ml: number
  color: string
  fmt: (v: number) => string
  yLabel: string
}

function renderPanel(data: readonly Point[], { x0, plotW, plotH, mt, ml, color, fmt, yLabel }: PanelOpts): string {
  const valid = data.filter((d): d is { x: number; y: number } => d.y != null && d.y > 0)
  if (valid.length < 2) return ''

  const xs = data.map(d => d.x)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yVals = valid.map(d => d.y)
  const yMin = Math.min(...yVals)
  const yMax = Math.max(...yVals)

  // Expand range for flat lines so they center nicely
  const yLo = yMax / yMin < 4 ? yMin * 0.25 : yMin
  const yHi = yMax / yMin < 4 ? yMax * 4    : yMax
  const logMin = Math.log10(yLo)
  const logMax = Math.log10(yHi)

  const xS = (x: number): number => x0 + ml + ((x - xMin) / (xMax - xMin)) * plotW
  const yS = (y: number): number => mt + plotH - ((Math.log10(Math.max(y, 1e-20)) - logMin) / (logMax - logMin)) * plotH

  const pathD = valid.map((d, i) =>
    `${i === 0 ? 'M' : 'L'}${xS(d.x).toFixed(1)},${yS(d.y).toFixed(1)}`
  ).join(' ')

  const step = Math.max(1, Math.floor((xMax - xMin) / 8))
  const xTicks = xs.filter(x => (x - xMin) % step === 0 || x === xMax)
  const yTicks = logTicks(yLo, yHi)

  const els: string[] = []

  // Background
  els.push(`<rect x="${(x0 + ml).toFixed(0)}" y="${mt}" width="${plotW.toFixed(0)}" height="${plotH}" fill="#f8f9fa" rx="2"/>`)

  // Grid
  for (const y of yTicks)
    els.push(`<line x1="${(x0 + ml).toFixed(0)}" y1="${yS(y).toFixed(1)}" x2="${(x0 + ml + plotW).toFixed(0)}" y2="${yS(y).toFixed(1)}" stroke="#e8e8e8" stroke-width="1"/>`)

  // Axes
  els.push(`<line x1="${(x0 + ml).toFixed(0)}" y1="${mt}" x2="${(x0 + ml).toFixed(0)}" y2="${mt + plotH}" stroke="#bbb" stroke-width="1"/>`)
  els.push(`<line x1="${(x0 + ml).toFixed(0)}" y1="${mt + plotH}" x2="${(x0 + ml + plotW).toFixed(0)}" y2="${mt + plotH}" stroke="#bbb" stroke-width="1"/>`)

  // X tick labels
  for (const x of xTicks)
    els.push(`<text x="${xS(x).toFixed(1)}" y="${mt + plotH + 13}" text-anchor="middle" font-size="8" fill="#999" font-family="monospace">${x}</text>`)

  // Y tick labels
  for (const y of yTicks)
    els.push(`<text x="${(x0 + ml - 4).toFixed(0)}" y="${(yS(y) + 3).toFixed(1)}" text-anchor="end" font-size="8" fill="#999" font-family="monospace">${fmt(y)}</text>`)

  // Axis labels
  els.push(`<text transform="translate(${(x0 + ml - 34).toFixed(0)},${(mt + plotH / 2).toFixed(0)}) rotate(-90)" text-anchor="middle" font-size="9" fill="#666" font-family="sans-serif">${yLabel}</text>`)
  els.push(`<text x="${(x0 + ml + plotW / 2).toFixed(0)}" y="${mt + plotH + 25}" text-anchor="middle" font-size="9" fill="#666" font-family="sans-serif">qubits</text>`)

  // Line + dots
  els.push(`<path d="${pathD}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`)
  for (const d of valid)
    els.push(`<circle cx="${xS(d.x).toFixed(1)}" cy="${yS(d.y).toFixed(1)}" r="2.5" fill="${color}" stroke="white" stroke-width="1"/>`)

  return els.join('\n  ')
}

/** One benchmark measurement: qubit count, wall time, and estimated memory. */
export interface ChartPoint {
  n: number
  ms: number
  memMB: number
}

/** Renders a dual-panel SVG chart (time | memory) for benchmark data. */
export function renderChart(
  title: string,
  data: readonly ChartPoint[],
  { width = 700, height = 220 }: { width?: number; height?: number } = {},
): string {
  const gap = 20
  const ml = 48, mr = 10, mt = 32, mb = 32
  const plotW = (width - ml - mr - gap) / 2
  const plotH = height - mt - mb

  const timeData = data.map(d => ({ x: d.n, y: d.ms }))
  const memData  = data.map(d => ({ x: d.n, y: d.memMB }))

  const p1 = renderPanel(timeData, { x0: 0,           plotW, plotH, mt, ml, color: C_TIME, fmt: fmtTime, yLabel: 'time' })
  const p2 = renderPanel(memData,  { x0: plotW + gap, plotW, plotH, mt, ml, color: C_MEM,  fmt: fmtMem,  yLabel: 'memory' })

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">
  <rect width="${width}" height="${height}" fill="white"/>
  <text x="${width / 2}" y="20" text-anchor="middle" font-size="13" font-weight="600" fill="#222" font-family="sans-serif">${title}</text>
  ${p1}
  ${p2}
</svg>`
}
