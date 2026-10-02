// CAPTURE-ROADMAP 04 (load-path): Engine.create spread plus first-token
// latency across fresh-browser bare runs. Each run launches a fresh Chrome,
// loads www/timing.html?mode=bare and records engine-create time
// (import-done -> engine-created) and first-token time (cold:send ->
// cold:chunk-0) with per-run provenance (power, thermal, browser).
//
// The weight-convert vs warm-up split inside Engine.create is not observable
// from outside the engine, so the record reports the total plus first-token
// and notes that split as a limitation instead of inventing it.
//
// Usage: node load-path.mjs [runs] [chunks]   (defaults 5, 16)
// Writes out/load-path.json.

import { createReadStream, mkdirSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { provenance } from './provenance.mjs'
import { manifestSha256 } from './shader-sha.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const WWW = join(HERE, 'www')
const OUT = join(HERE, 'out')
const PORT = 8918
const TIMEOUT_MS = 20 * 60 * 1000
const RUNS = Number(process.argv[2] ?? 5)
const CHUNKS = Number(process.argv[3] ?? 16)

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
const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')

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
    const mark = (n) => T.marks.find((m) => m.name === n)?.t
    const prov = await provenance({ browser, adapter: T.adapter, pkg: T.version })
    return {
      error: T.error ?? null,
      engineMs: mark('engine-created') - mark('import-done'),
      coldFirstMs: mark('cold:chunk-0') - mark('cold:send'),
      warmFirstMs: mark('warm:chunk-0') - mark('warm:send'),
      coldChunks: T.cold?.chunks ?? null,
      warmChunks: T.warm?.chunks ?? null,
      provenance: prov,
    }
  } finally { await browser.close() }
}

const runs = []
let exitCode = 0
let failure = null
try {
  for (let r = 0; r < RUNS; r++) {
    let R = await once()
    // R5: one re-run max, then keep what was seen.
    if (R.error) {
      console.log(`run ${r + 1} errored: ${R.error}; re-running once`)
      R = await once()
    }
    runs.push({ run: r + 1, ...R })
    if (R.error) failure = `run ${r + 1} failed twice: ${R.error}`
    else console.log(`run ${r + 1}: engine ${f(R.engineMs, 0)} ms, cold first token ${f(R.coldFirstMs, 0)} ms, power ${R.provenance.power.source} ${R.provenance.power.percent}%`)
    if (failure) break
    mkdirSync(OUT, { recursive: true })
    writeFileSync(join(OUT, 'load-path.json'), JSON.stringify({ runs, failure: failure ?? 'pending' }, null, 2))
  }

  const good = runs.filter((R) => !R.error)
  const spread = (xs) => ({ n: xs.length, median: median(xs), min: Math.min(...xs), max: Math.max(...xs) })
  const engine = spread(good.map((R) => R.engineMs))
  const coldFirst = spread(good.map((R) => R.coldFirstMs))
  const summary = { engineMs: engine, coldFirstMs: coldFirst }
  const width = (s) => (s.max - s.min) / s.median
  const note = `weight-convert vs warm-up split inside Engine.create is not observable from outside the engine; record reports totals. p90-p10-style width (max-min)/median: engine ${f(100 * width(engine), 1)}%, cold-first ${f(100 * width(coldFirst), 1)}%.`
  console.log(note)
  const record = {
    date: new Date().toISOString(),
    runsRequested: RUNS,
    chunks: CHUNKS,
    failure,
    summary,
    note,
    manifest_sha256: manifestSha256(join(OUT, 'manifest.json')),
    runs,
  }
  writeFileSync(join(OUT, 'load-path.json'), JSON.stringify(record, null, 2))
  if (failure || good.length < 3) { console.log(failure ?? `only ${good.length} good runs`); exitCode = 2 }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally { server.close() }
process.exit(exitCode)
