// CAPTURE-ROADMAP 06 (firstslot): cold-slot effect across repeated
// fresh-browser bare runs. Each run records cold first-chunk latency
// (cold:send -> cold:chunk-0), warm first-chunk latency, engine-create time
// and per-run provenance, isolating the first-message cold-slot cost from the
// steady state.
//
// Usage: node firstslot.mjs [runs] [chunks]   (defaults 5, 16)
// Writes out/firstslot.json.

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
const PORT = 8920
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
      coldFirstMs: mark('cold:chunk-0') - mark('cold:send'),
      warmFirstMs: mark('warm:chunk-0') - mark('warm:send'),
      engineMs: mark('engine-created') - mark('import-done'),
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
    else console.log(`run ${r + 1}: cold-first ${f(R.coldFirstMs, 0)} ms, warm-first ${f(R.warmFirstMs, 0)} ms, engine ${f(R.engineMs, 0)} ms`)
    if (failure) break
    mkdirSync(OUT, { recursive: true })
    writeFileSync(join(OUT, 'firstslot.json'), JSON.stringify({ runs, failure: failure ?? 'pending' }, null, 2))
  }

  const good = runs.filter((R) => !R.error)
  const cold = good.map((R) => R.coldFirstMs)
  const warm = good.map((R) => R.warmFirstMs)
  // Cold-slot effect, two cuts: cold vs warm first-chunk within each fresh
  // browser, and the first run of the sweep vs later runs (cold machine slot).
  const ratios = good.map((R) => R.coldFirstMs / R.warmFirstMs)
  const summary = {
    n: good.length,
    coldFirstMs: { median: median(cold), min: Math.min(...cold), max: Math.max(...cold) },
    warmFirstMs: { median: median(warm), min: Math.min(...warm), max: Math.max(...warm) },
    coldOverWarm: { median: median(ratios), min: Math.min(...ratios), max: Math.max(...ratios) },
    run1ColdMs: cold[0] ?? null,
    laterColdMedianMs: median(cold.slice(1)),
  }
  const note = `cold first-chunk is ${f(median(ratios), 2)}x warm first-chunk (median); run 1 cold ${f(cold[0], 0)} ms vs later-run cold median ${f(median(cold.slice(1)), 0)} ms. Widths committed as seen per R5, not re-run until pretty.`
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
  writeFileSync(join(OUT, 'firstslot.json'), JSON.stringify(record, null, 2))
  if (failure || good.length < 3) { console.log(failure ?? `only ${good.length} good runs`); exitCode = 2 }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally { server.close() }
process.exit(exitCode)
