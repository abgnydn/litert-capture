// In-model A/B: runs www/patch.html for each patch set in a fresh Chrome,
// reports steady-state decode time and checks the output text is identical
// to the unpatched run. Usage: node patch.mjs [repeats]   (default 2)

import { createReadStream, mkdirSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'

const HERE = dirname(fileURLToPath(import.meta.url))
const WWW = join(HERE, 'www')
const OUT = join(HERE, 'out')
const PORT = 8917
const TIMEOUT_MS = 20 * 60 * 1000
const STEADY_FROM = 8
const REPEATS = Number(process.argv[2] ?? 2)
const SETS = ['none', '0112', '0112,0105', 'all']

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.wasm': 'application/wasm', '.wgsl': 'text/plain' }
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname))
  const file = path.startsWith('/out/') ? join(OUT, path.slice(5)) : join(WWW, path)
  if (!file.startsWith(WWW) && !file.startsWith(OUT)) return res.writeHead(403).end()
  let size
  try { size = statSync(file).size } catch { return res.writeHead(404).end() }
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] ?? 'application/octet-stream', 'Content-Length': size, 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' })
  createReadStream(file).pipe(res)
})
await new Promise((r) => server.listen(PORT, '127.0.0.1', r))

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN }
const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')

const once = async (set) => {
  const browser = await puppeteer.launch({ headless: false, args: ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist', '--enable-features=Vulkan'], protocolTimeout: TIMEOUT_MS })
  try {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
    await page.goto(`http://127.0.0.1:${PORT}/patch.html?patch=${encodeURIComponent(set)}`, { waitUntil: 'load' })
    await page.waitForFunction('window.__patchDone === true', { timeout: TIMEOUT_MS, polling: 1000 })
    const P = await page.evaluate(() => window.__patch)
    const mark = (n) => P.marks.find((m) => m.name === n)?.t
    const chunkT = P.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
    const gaps = chunkT.slice(STEADY_FROM).map((t, i) => t - chunkT[STEADY_FROM + i - 1]).slice(1)
    return { set, error: P.error, gpuErrors: P.gpuErrors, patched: P.patched, scaled: P.scaledDispatches, unscaled: P.unscaledDispatches, cold: P.cold, warm: P.warm, msPerToken: median(gaps), n: gaps.length, firstChunkMs: mark('warm:chunk-0') - mark('warm:send'), engineMs: mark('engine-created') - mark('import-done') }
  } finally { await browser.close() }
}

const results = []
let exitCode = 0
try {
  for (let r = 0; r < REPEATS; r++) for (const set of SETS) {
    const R = await once(set)
    results.push(R)
    if (R.error) { console.log(`patch=${set}: PAGE ERROR ${R.error}`); exitCode = 2; continue }
    console.log(`patch=${set.padEnd(10)} run ${r + 1}: ${f(R.msPerToken)} ms/token = ${f(1000 / R.msPerToken, 1)} tok/s over ${R.n} tokens | patched ${R.patched.length} pipelines, ${R.scaled} scaled dispatches | GPU errors ${R.gpuErrors.length} | engine ${f(R.engineMs, 0)} ms`)
  }
  mkdirSync(OUT, { recursive: true })
  writeFileSync(join(OUT, 'patch.json'), JSON.stringify(results, null, 2))
  const base = results.find((R) => R.set === 'none' && !R.error)
  console.log('\n== summary (median over runs; output compared with the unpatched run)')
  for (const set of SETS) {
    const rs = results.filter((R) => R.set === set && !R.error)
    if (!rs.length) continue
    const ms = median(rs.map((R) => R.msPerToken))
    const same = rs.every((R) => R.warm.text === base.warm.text && R.cold.text === base.cold.text)
    console.log(`  ${set.padEnd(10)} ${f(ms)} ms/token  ${f(1000 / ms, 1)} tok/s  ${base ? `x${f(base.msPerToken / ms, 3)} vs unpatched` : ''}  output identical: ${same}  scaled dispatches/run: ${rs.map((R) => R.scaled).join('/')}`)
  }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally { server.close() }
process.exit(exitCode)
