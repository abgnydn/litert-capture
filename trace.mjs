// Drives www/trace.html, then reconstructs: GPU memory footprint, the per-token
// dispatch sequence (layer structure), matrix shapes of the matvec kernels,
// weight bytes read per token, and achieved bandwidth (joined with
// out/kernel-profile.json).

import { createReadStream, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { provenance } from './provenance.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const WWW = join(HERE, 'www')
const OUT = join(HERE, 'out')
const PORT = 8917
const TIMEOUT_MS = 20 * 60 * 1000

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' }
const server = createServer((req, res) => {
  const file = join(WWW, normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)))
  if (!file.startsWith(WWW)) return res.writeHead(403).end()
  let size
  try { size = statSync(file).size } catch { return res.writeHead(404).end() }
  res.writeHead(200, {
    'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
    'Content-Length': size,
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  })
  createReadStream(file).pipe(res)
})
await new Promise((r) => server.listen(PORT, '127.0.0.1', r))

const browser = await puppeteer.launch({
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist', '--enable-features=Vulkan'],
  protocolTimeout: TIMEOUT_MS,
})

const BPT = { rgba8uint: 4, rgba8unorm: 4, rgba8sint: 4, rgba16float: 8, rgba16uint: 8, rgba32uint: 16, rgba32float: 16, r32uint: 4, r32float: 4, rg32uint: 8, rg32float: 8, r16float: 2, rg16float: 4, r8uint: 1 }
const MB = (b) => (b / 1048576).toFixed(0)
const f = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')

let exitCode = 0
try {
  const page = await browser.newPage()
  page.on('console', (m) => { const t = m.text(); if (t.startsWith('[trace]')) console.log(t) })
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
  await page.goto(`http://127.0.0.1:${PORT}/trace.html`, { waitUntil: 'load' })
  await page.waitForFunction('window.__traceDone === true', { timeout: TIMEOUT_MS, polling: 1000 })
  const R = await page.evaluate(() => window.__trace)
  R.provenance = await provenance({ browser, adapter: R.adapter, pkg: R.version })
  mkdirSync(OUT, { recursive: true })
  writeFileSync(join(OUT, 'trace.json'), JSON.stringify(R))
  if (R.error) { console.log(`PAGE ERROR: ${R.error}`); exitCode = 2 }

  const texBytes = (t) => t.size[0] * t.size[1] * (t.size[2] ?? 1) * (BPT[t.format] ?? 0)
  const unknownFmt = [...new Set(R.textures.filter((t) => !BPT[t.format]).map((t) => t.format))]

  console.log(`\n== resources | @litert-lm/core ${R.version} | ${JSON.stringify(R.adapter)}`)
  const byFmt = {}
  for (const t of R.textures) { const k = t.format; byFmt[k] ??= { n: 0, bytes: 0 }; byFmt[k].n++; byFmt[k].bytes += texBytes(t) }
  const texTotal = R.textures.reduce((s, t) => s + texBytes(t), 0)
  const bufTotal = R.buffers.reduce((s, b) => s + b.size, 0)
  console.log(`textures: ${R.textures.length}, ${MB(texTotal)} MB` + (unknownFmt.length ? ` (unknown formats: ${unknownFmt.join(', ')})` : ''))
  for (const [k, v] of Object.entries(byFmt).sort((a, b) => b[1].bytes - a[1].bytes)) console.log(`  ${k.padEnd(14)} ${String(v.n).padStart(5)}  ${MB(v.bytes).padStart(6)} MB`)
  console.log(`buffers:  ${R.buffers.length}, ${MB(bufTotal)} MB (created during load: ${MB(R.buffers.filter((b) => b.phase === 'load').reduce((s, b) => s + b.size, 0))} MB)`)
  console.log(`resident GPU total: ${MB(texTotal + bufTotal)} MB`)

  // profile join
  let prof = null
  try { prof = JSON.parse(readFileSync(join(OUT, 'kernel-profile.json'), 'utf8')) } catch {}
  const usByPid = new Map(prof ? prof.rows.map((r) => [r.pid, r.us]) : [])
  const shaderOf = (pid) => R.pipelines[pid]?.shaderId ?? -1

  // one token's dispatch sequence
  const chunks = [...new Set(R.dispatches.map((d) => d.chunk))]
  const token = R.dispatches.filter((d) => d.chunk === (chunks[1] ?? chunks[0]))
  const seq = token.map((d) => shaderOf(d.pid))
  console.log(`\n== decode tape: ${token.length} dispatches in traced token, ${chunks.length} tokens traced`)
  let best = { p: 0, score: 0 }
  for (let p = 8; p <= 120; p++) {
    let hit = 0
    for (let i = 0; i + p < seq.length; i++) if (seq[i] === seq[i + p]) hit++
    const score = hit / (seq.length - p)
    if (score > best.score + 1e-9) best = { p, score }
  }
  console.log(`best repeating period: ${best.p} dispatches (${f(100 * best.score, 1)}% of positions match one period later)`)
  const period = best.p
  // find where the periodic region starts and ends
  let start = 0
  while (start + period < seq.length && seq[start] !== seq[start + period]) start++
  let end = seq.length - 1
  while (end - period >= 0 && seq[end] !== seq[end - period]) end--
  const nLayers = Math.round((end - start + 1) / period)
  console.log(`periodic region: dispatches ${start}..${end} = ${nLayers} repeats; prologue ${start} dispatches, epilogue ${seq.length - 1 - end}`)
  const layer = seq.slice(start, start + period)
  const rle = []
  for (const s of layer) { if (rle.length && rle[rle.length - 1].s === s) rle[rle.length - 1].n++; else rle.push({ s, n: 1 }) }
  console.log(`one layer (${period} dispatches), shader ids with repeat counts:`)
  console.log('  ' + rle.map((r) => (r.n > 1 ? `${String(r.s).padStart(4, '0')}x${r.n}` : String(r.s).padStart(4, '0'))).join(' '))
  // layer variants: compare each repeat to the first
  const variants = new Map()
  for (let L = 0; L < nLayers; L++) {
    const key = seq.slice(start + L * period, start + (L + 1) * period).join(',')
    variants.set(key, (variants.get(key) ?? 0) + 1)
  }
  console.log(`distinct layer variants: ${variants.size} (${[...variants.values()].join(' + ')})`)
  console.log(`prologue shaders: ${seq.slice(0, start).map((s) => String(s).padStart(4, '0')).join(' ')}`)
  console.log(`epilogue shaders: ${seq.slice(end + 1).map((s) => String(s).padStart(4, '0')).join(' ')}`)

  // matvec kernels: shapes and bytes.
  // These kernel indices were read off @litert-lm/core 0.17.1 with the Gemma 4
  // E2B web bundle; they are positions in that build's shader creation order and
  // will differ for any other package version or model bundle.
  const MATVEC = [99, 105, 112, 113]
  console.log(`\n== weight matrices read per token`)
  let totalBytes = 0, totalUs = 0
  const rows = []
  for (const sid of [...MATVEC, 134]) {
    const ds = token.filter((d) => shaderOf(d.pid) === sid)
    if (!ds.length) continue
    const shapes = new Map()
    let bytes = 0
    for (const d of ds) {
      const tex = d.res.find((r) => r.kind === 'texture' && r.binding === 2)
      const wbuf = d.res.find((r) => r.kind === 'buffer' && r.binding === 2)
      const uni = d.res.find((r) => r.kind === 'buffer' && r.uniform)
      const t = tex ? R.textures[tex.tex] : null
      const b = t ? texBytes(t) : (wbuf?.size ?? R.buffers[wbuf?.buf]?.size ?? 0)
      bytes += b
      const key = `${t ? `${t.size[0]}x${t.size[1]} ${t.format}` : `buffer ${MB(b)} MB`} | U.i0=${JSON.stringify(uni?.uniform?.slice(0, 4))} | wg=${JSON.stringify(d.wg)}`
      shapes.set(key, (shapes.get(key) ?? 0) + 1)
    }
    const us = (usByPid.get(ds[0].pid) ?? NaN) * ds.length
    totalBytes += bytes; totalUs += us || 0
    rows.push({ sid, n: ds.length, bytes, us })
    console.log(`shader ${String(sid).padStart(4, '0')}: ${ds.length} dispatches, ${MB(bytes)} MB of weights, ${f(us / 1000, 2)} ms GPU -> ${f(bytes / (us * 1e-6) / 1e9, 0)} GB/s`)
    for (const [k, n] of [...shapes.entries()].sort((a, b) => b[1] - a[1])) console.log(`    ${String(n).padStart(3)}x  ${k}`)
  }
  console.log(`total: ${MB(totalBytes)} MB of weights in ${f(totalUs / 1000, 2)} ms of matvec GPU time -> ${f(totalBytes / (totalUs * 1e-6) / 1e9, 0)} GB/s achieved on these kernels`)
  console.log(`(kernel times are medians from out/kernel-profile.json, split-pass measurement, so bandwidth is a lower bound)`)

  // prefill
  const prefillOnly = Object.keys(R.prefillPipelines).map(Number).filter((pid) => !usByPid.has(pid))
  const prefillDisp = Object.values(R.prefillPipelines).reduce((s, n) => s + n, 0)
  console.log(`\n== prefill (cold message): ${prefillDisp} dispatches over ${Object.keys(R.prefillPipelines).length} pipelines, ${prefillOnly.length} of them never used in decode`)
  const topPrefill = Object.entries(R.prefillPipelines).sort((a, b) => b[1] - a[1]).slice(0, 8)
  console.log('  most dispatched in prefill: ' + topPrefill.map(([pid, n]) => `${String(shaderOf(+pid)).padStart(4, '0')}x${n}`).join(' '))
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally {
  await browser.close()
  server.close()
}
process.exit(exitCode)
