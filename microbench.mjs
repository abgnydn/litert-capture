// Drives www/microbench.html (the 2-bit kernels 0112 and 0113 in isolation) and
// www/microbench4.html (the 4-bit kernels 0105 and 0099 in isolation) and
// saves a dated record under out/. Serves out/ as /out/ so the page can load the
// captured shader text verbatim.
//
// Usage: node microbench.mjs [page|--kernel=ID|ID]
//   node microbench.mjs                            kernel 0112 (the default page)
//   node microbench.mjs 0113                       same as 'microbench.html?kernel=0113'
//   node microbench.mjs --kernel=0113              same as above
//   node microbench.mjs 0105                       same as 'microbench4.html?kernel=0105'
//   node microbench.mjs 0099                       same as 'microbench4.html?kernel=0099'
//   node microbench.mjs 'microbench.html?kernel=0113'   full page path still works
//   node microbench.mjs 'microbench4.html?kernel=0105'  full page path still works
// PUPPETEER_EXECUTABLE_PATH picks the Chrome build, which is the whole point of
// the 131-vs-146 pair: the record then carries that build in `provenance.browser`.
//
// ROADMAP 10: K-split search per kernel shape. Sweep ks with the matching wgX
// so the workgroup stays 64 invocations (2-bit 0112/0113 in www/microbench.html,
// 4-bit 0105/0099 in www/microbench4.html) or 256 invocations, dispatchX =
// OUT_SLICES / wgX; each variant is timed forward+reverse pooled (576 cold +
// 24 hot samples) against a CPU reference and written to a dated
// out/microbench-<kernel>-<stamp>.json record. Committed best splits by cold
// median: 0112 2x32 (36.8 us, out/microbench-0112-2026-09-24T07-36-19-655Z.json),
// 0105 2x32 (28.1 us, out/microbench-0105-2026-09-22T05-39-22-096Z.json), 0099
// 2x128 (24.2 us, out/microbench-0099-2026-09-22T05-39-24-280Z.json), 0113 1x256
// (37.0 us, out/microbench-0113-2026-09-24T07-36-44-034Z.json). Documented from
// the existing records; no bench run here, defaults unchanged.
//
// ROADMAP 11: Sweep workgroup sizes (X) per kernel at capture time. The
// 0112 wgX 16 vs 64 control already exists as wg256-same (same 12,288
// threads, 64 vs 256 invocations); generalize per kernel at fixed ks/threads
// (dispatchX = OUT_SLICES / wgX): 0112 16 vs 64 (ks 4), 0113 2 vs 8 (ks 32),
// 0105 16 vs 64 (ks 4), 0099 4 vs 16 (ks 16). Current VARIANTS tables
// (www/microbench.html, www/microbench4.html) vary wgX together with ks;
// default `node microbench.mjs` runs all variants unchanged (optional --wg
// passthrough not wired).
//
// ROADMAP 12: Record hot/cold timings separately. Cold rotates through 24
// matrices (NMAT_COLD, 108 MiB, 576 samples = 2 x ROUNDS 12 x 24 pooled
// forward+reverse); hot reuses 1 matrix (24 samples = 2 x 12 x 1). Both are
// already recorded as cold/hot medians (median/p10/p90/n) per variant in
// www/microbench.html; this split is explicit and the default runs both
// (optional --mode=cold|hot passthrough not wired).
//
// ROADMAP 13: forward+reverse order cancels linear clock drift. The pages time
// each variant forward then again in reverse ([...timed, ...[...timed].reverse()])
// and pool both positions' cold+hot samples before the median; R.order records
// the per-position medians so the two directions can be compared. Always on,
// no flag; default unchanged.
//
// ROADMAP 14: CPU reference per kernel shape. Each page checks every variant
// against a JS CPU reference for matrix 0 and records maxAbsErr alongside the
// GPU medians (2-bit 0112/0113 in www/microbench.html: 2-bit dequant, zero
// point 2, TOL 0.05 / REL_FLOOR 0.1, mismatches + maxRelErr; 4-bit 0105/0099
// in www/microbench4.html: nibble dequant, zero point 8, maxAbsErr only).
// Optional ?f32=1 runs an f32 transcription on adapters without shader-f16
// (same structure, not Google's kernel verbatim); default is the f16 kernel
// as shipped. Always on, no flag for the check itself; defaults unchanged.

import { createReadStream, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { provenance } from './provenance.mjs'
import { manifestSha256 } from './shader-sha.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const WWW = join(HERE, 'www')
const OUT = join(HERE, 'out')
const PORT = 8917

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wgsl': 'text/plain' }
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname))
  const file = path.startsWith('/out/') ? join(OUT, path.slice(5)) : join(WWW, path)
  if (!file.startsWith(WWW) && !file.startsWith(OUT)) return res.writeHead(403).end()
  let size
  try { size = statSync(file).size } catch { return res.writeHead(404).end() }
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] ?? 'application/octet-stream', 'Content-Length': size })
  createReadStream(file).pipe(res)
})
await new Promise((r) => server.listen(PORT, '127.0.0.1', r))

const browser = await puppeteer.launch({
  headless: false,
  executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist', '--disable-dawn-features=timestamp_quantization', '--enable-features=Vulkan'],
})
let exitCode = 0
try {
  const page = await browser.newPage()
  page.on('console', (m) => { const t = m.text(); if (t.startsWith('[mb]')) console.log(t) })
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
  // ROADMAP 09: per-kernel flags. 2-bit kernels live in microbench.html,
  // 4-bit kernels in microbench4.html; bare IDs and --kernel= map to those
  // existing ?kernel= paths so the default stays kernel 0112.
  const KERNEL_PAGES = {
    '0112': 'microbench.html',
    '0113': 'microbench.html?kernel=0113',
    '0105': 'microbench4.html?kernel=0105',
    '0099': 'microbench4.html?kernel=0099',
  }
  const resolvePage = (arg) => {
    if (arg === undefined) return 'microbench.html'
    const m = /^(?:--kernel=|\?kernel=)?(0112|0113|0105|0099)$/.exec(arg.trim())
    if (m) return KERNEL_PAGES[m[1]]
    return arg
  }
  // default page is the 2-bit kernel 0112; pass e.g. 0113 or --kernel=0105
  const PAGE = resolvePage(process.argv[2])
  await page.goto(`http://127.0.0.1:${PORT}/${PAGE}`, { waitUntil: 'load' })
  await page.waitForFunction('window.__mbDone === true', { timeout: 5 * 60 * 1000, polling: 500 })
  const R = await page.evaluate(() => window.__mb)
  // The page runs the captured shader on its own, with no @litert-lm/core and no
  // model bundle, so package is null and model is whatever bundle is on disk.
  R.provenance = await provenance({ browser, adapter: R.adapter, pkg: null })
  // ROADMAP 08: tie this record to the capture manifest it joins against.
  R.manifest_sha256 = manifestSha256(join(OUT, 'manifest.json'))
  R.date = new Date().toISOString()
  // every run is kept in its own dated file; nothing is overwritten
  const stamp = R.date.replace(/[:.]/g, '-')
  const name = `microbench${R.kernel ? '-' + R.kernel : ''}-${stamp}.json`
  writeFileSync(join(OUT, name), JSON.stringify(R, null, 2))
  console.log(`saved out/${name}`)
  if (R.error || R.gpuError) exitCode = 2
  console.log(`\nadapter ${JSON.stringify(R.adapter)}, kernel source ${R.baseBytes} bytes, ${(R.weightBytes / 1048576).toFixed(2)} MB of weights per dispatch`)
  // the committed metal-3 records are Chrome 146 (recorded or inferred); the committed Chrome 131 record reports common-3 and inverts the result, see README "Browser version"
  if (R.adapter?.vendor === 'apple' && R.adapter?.architecture !== 'metal-3') console.log(`WARNING: adapter architecture is ${R.adapter.architecture}, not metal-3; single-dispatch timings are not comparable with the committed metal-3 records`)
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally {
  await browser.close()
  server.close()
}
process.exit(exitCode)
