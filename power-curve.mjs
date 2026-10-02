// CAPTURE-ROADMAP 05 (power-curve): steady-state decode speed across repeated
// fresh-browser bare runs, joined with the per-run power and thermal fields
// from provenance.mjs. On a single power state (this machine runs on AC) the
// curve is flat by construction; the record reports that flatness with the
// power fields rather than manufacturing variance.
//
// Usage: node power-curve.mjs [runs] [chunks]   (defaults 5, 40)
// Writes out/power-curve.json.

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
const PORT = 8919
const TIMEOUT_MS = 20 * 60 * 1000
const RUNS = Number(process.argv[2] ?? 5)
const CHUNKS = Number(process.argv[3] ?? 40)
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
    const chunkT = T.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
    const gaps = chunkT.slice(STEADY_FROM).map((t, i) => t - chunkT[STEADY_FROM + i - 1]).slice(1)
    const prov = await provenance({ browser, adapter: T.adapter, pkg: T.version })
    return {
      error: T.error ?? null,
      tokPerSec: 1000 / median(gaps),
      gapMs: { median: median(gaps), n: gaps.length },
      power: prov.power,
      thermal: prov.thermal,
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
    else console.log(`run ${r + 1}: ${f(R.tokPerSec, 1)} tok/s, power ${R.power.source} ${R.power.percent}%, thermal notes ${R.thermal.length}`)
    if (failure) break
    mkdirSync(OUT, { recursive: true })
    writeFileSync(join(OUT, 'power-curve.json'), JSON.stringify({ runs, failure: failure ?? 'pending' }, null, 2))
  }

  const good = runs.filter((R) => !R.error)
  const toks = good.map((R) => R.tokPerSec)
  const states = [...new Set(good.map((R) => `${R.power.source}/${R.power.percent}%${R.power.low_power_mode ? '/lpm' : ''}`))]
  const note = states.length === 1
    ? `all ${good.length} runs on a single power state (${states[0]}); no battery/thermal variance was available to correlate, so the curve is flat by construction. Power source cannot be switched by the harness; a battery-vs-AC curve needs a human to unplug.`
    : `power states seen: ${states.join(', ')}.`
  console.log(note)
  const record = {
    date: new Date().toISOString(),
    runsRequested: RUNS,
    chunks: CHUNKS,
    failure,
    summary: { n: toks.length, tokPerSec: { median: median(toks), min: Math.min(...toks), max: Math.max(...toks) }, powerStates: states },
    note,
    manifest_sha256: manifestSha256(join(OUT, 'manifest.json')),
    runs,
  }
  writeFileSync(join(OUT, 'power-curve.json'), JSON.stringify(record, null, 2))
  if (failure || good.length < 3) { console.log(failure ?? `only ${good.length} good runs`); exitCode = 2 }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally { server.close() }
process.exit(exitCode)
