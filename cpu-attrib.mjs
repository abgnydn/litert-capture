// CAPTURE-ROADMAP 13 (cpu-attrib): CPU-side per-pass attribution.
//
// Extends the timing.mjs bare/gpu/kernel split (CPU-side vs waiting-on-GPU
// from the one mapAsync per token, GPU busy vs span from timestamped passes,
// per-dispatch medians from the split-kernel run) to a per-pass table: one
// row per warm decode chunk of a fresh short bare run, with wall ms split
// into CPU-side ms and GPU-wait ms from that run's own readback map timings,
// joined by median against the committed timing-gpu.json (GPU busy per token)
// and out/kernel-profile.json (split per-dispatch sum per token) measured on
// the same machine. Modes come from separate runs, so the join is by median,
// not same-token; the record says so instead of pretending otherwise.
//
// Usage: node cpu-attrib.mjs [chunks]   (default 16)
// Writes out/cpu-attrib.json. Committed out/timing-*.json are read, never written.

import { createReadStream, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { provenance } from './provenance.mjs'
import { manifestSha256 } from './shader-sha.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const WWW = join(HERE, 'www')
const OUT = join(HERE, 'out')
const PORT = 8921
const TIMEOUT_MS = 20 * 60 * 1000
const CHUNKS = Number(process.argv[2] ?? 16)
const STEADY_FROM = 8

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

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN }
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * p))] : NaN }
const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')
const readJson = (name) => JSON.parse(readFileSync(join(OUT, name), 'utf8'))

const once = async () => {
  const browser = await puppeteer.launch({
    headless: false,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist',
      '--disable-dawn-features=timestamp_quantization',
      '--enable-features=Vulkan',
    ],
    protocolTimeout: TIMEOUT_MS,
  })
  try {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
    await page.goto(`http://127.0.0.1:${PORT}/timing.html?mode=bare&chunks=${CHUNKS}`, { waitUntil: 'load' })
    await page.waitForFunction('window.__timingDone === true', { timeout: TIMEOUT_MS, polling: 1000 })
    const T = await page.evaluate(() => window.__timing)
    const prov = await provenance({ browser, adapter: T.adapter, pkg: T.version })
    return { T, prov }
  } finally { await browser.close() }
}

let exitCode = 0
try {
  let { T, prov } = await once()
  // R5: one re-run max, then keep what was seen.
  if (T.error) {
    console.log(`run errored: ${T.error}; re-running once`)
    ;({ T, prov } = await once())
  }
  if (T.error) throw new Error(`bare run failed twice: ${T.error}`)

  const chunkT = T.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
  // Per-pass (per-chunk) attribution: wall time of chunk i is split into
  // GPU-wait (sum of readback resolve-call spans stamped with that chunk)
  // and CPU-side (the remainder). Same maps split as timing.mjs, cut per chunk.
  const perChunk = []
  for (let i = STEADY_FROM; i < chunkT.length - 1; i++) {
    const wall = chunkT[i + 1] - chunkT[i]
    const maps = T.maps.filter((m) => m.chunk === i && m.resolve > 0)
    const wait = maps.reduce((s, m) => s + (m.resolve - m.call), 0)
    perChunk.push({ chunk: i, wallMs: wall, waitMs: wait, cpuMs: wall - wait, readbacks: maps.length })
  }
  const cpu = perChunk.map((r) => r.cpuMs)
  const wait = perChunk.map((r) => r.waitMs)

  // Committed joins, same machine: GPU busy per token (unsplit passes) and
  // the split per-dispatch sum per token, both as medians over full tokens.
  const G = readJson('timing-gpu.json')
  const byChunkG = new Map()
  for (const p of G.passes) {
    if (p.beginNs == null) continue
    if (!byChunkG.has(p.chunk)) byChunkG.set(p.chunk, [])
    byChunkG.get(p.chunk).push(p)
  }
  const gGroups = [...byChunkG.values()]
  const gFull = gGroups.filter((g) => g.length === median(gGroups.map((x) => x.length)))
  const gpuBusy = gFull.map((g) => g.reduce((s, p) => s + (p.endNs - p.beginNs), 0) / 1e6)
  const KP = readJson('kernel-profile.json')

  const width = (a) => (pct(a, 0.9) - pct(a, 0.1)) / median(a)
  const summary = {
    n: perChunk.length,
    cpuMs: { median: median(cpu), p10: pct(cpu, 0.1), p90: pct(cpu, 0.9) },
    waitMs: { median: median(wait), p10: pct(wait, 0.1), p90: pct(wait, 0.9) },
    gpuBusyMsUnsplit: { median: median(gpuBusy), p10: pct(gpuBusy, 0.1), p90: pct(gpuBusy, 0.9) },
    splitSumMsPerToken: KP.totalMs,
    residualWaitMinusGpuMs: median(wait) - median(gpuBusy),
  }
  // R5: widths committed as seen, not re-run until pretty.
  const note = `per-chunk CPU-side median ${f(median(cpu))} ms vs GPU-wait median ${f(median(wait))} ms (wall ${f(median(cpu) + median(wait))} ms); unsplit GPU busy median ${f(median(gpuBusy))} ms, split per-dispatch sum ${f(KP.totalMs)} ms. p90-p10 widths: cpu ${f(100 * width(cpu), 1)}%, wait ${f(100 * width(wait), 1)}%. Join is by median across separate runs (bare/gpu/kernel), not same-token.`
  console.log(note)
  const record = {
    date: new Date().toISOString(),
    method: 'fresh bare run (maps split per timing.mjs) cut per chunk, joined by median with committed timing-gpu.json + kernel-profile.json from the same machine',
    chunks: CHUNKS,
    steadyFrom: STEADY_FROM,
    summary,
    perChunk,
    note,
    sources: ['out/timing-bare.json', 'out/timing-gpu.json', 'out/kernel-profile.json'],
    manifest_sha256: manifestSha256(join(OUT, 'manifest.json')),
    provenance: prov,
  }
  mkdirSync(OUT, { recursive: true })
  writeFileSync(join(OUT, 'cpu-attrib.json'), JSON.stringify(record, null, 2))
  console.log(`wrote out/cpu-attrib.json (${perChunk.length} per-pass rows)`)
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally { server.close() }
process.exit(exitCode)
