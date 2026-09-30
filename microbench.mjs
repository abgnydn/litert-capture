// Drives www/microbench.html (the 2-bit kernels 0112 and 0113 in isolation) and
// saves a dated record under out/. Serves out/ as /out/ so the page can load the
// captured shader text verbatim.
//
// Usage: node microbench.mjs [page]
//   node microbench.mjs                            kernel 0112 (the default page)
//   node microbench.mjs 'microbench.html?kernel=0113'
//   node microbench.mjs 'microbench4.html?kernel=0105'
// PUPPETEER_EXECUTABLE_PATH picks the Chrome build, which is the whole point of
// the 131-vs-146 pair: the record then carries that build in `provenance.browser`.

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
  // default page is the 2-bit kernel 0112; pass e.g. 'microbench.html?kernel=0113'
  const PAGE = process.argv[2] ?? 'microbench.html'
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
