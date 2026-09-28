// Drives www/capture.html in a real Chrome with WebGPU and writes what the
// hooks recorded to out/. Reuses zero-tvm's puppeteer install and the desktop
// flag set from zero-tvm/bench/run.mjs.

import { createHash } from 'node:crypto'
import { createReadStream, mkdirSync, statSync, writeFileSync } from 'node:fs'
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

// Model bundle pinned to Hugging Face revision b3ca0d2f (see README "Run").
// Bytes + sha256 come from out/model.json via provenance when present; the
// constants are that pinned bundle's values.
const BUNDLE_REVISION = 'b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1'
const BUNDLE_BYTES = 2008432640
const BUNDLE_SHA256 = '3a08e8d94e23b814ae5414469c370c503813949acb8ceaa17e4ebf8a35af35b5'

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' }

const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname))
  const file = join(WWW, path)
  if (!file.startsWith(WWW)) return res.writeHead(403).end()
  let size
  try { size = statSync(file).size } catch { return res.writeHead(404).end() }
  res.writeHead(200, {
    'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
    'Content-Length': size,
    // Same isolation headers the official chat demo's vite config sets, so the
    // threaded wasm build is the one that loads.
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  })
  createReadStream(file).pipe(res)
})
await new Promise((r) => server.listen(PORT, '127.0.0.1', r))

const browser = await puppeteer.launch({
  headless: false,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist',
    '--enable-features=Vulkan',
  ],
  protocolTimeout: TIMEOUT_MS,
})

let exitCode = 0
try {
  const page = await browser.newPage()
  page.on('console', (m) => console.log(`[page] ${m.text()}`))
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
  page.on('requestfailed', (r) => console.log(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`))

  const query = process.argv[2] ?? ''
  await page.goto(`http://127.0.0.1:${PORT}/capture.html${query}`, { waitUntil: 'load' })
  await page.waitForFunction('window.__captureDone === true', { timeout: TIMEOUT_MS, polling: 1000 })

  const cap = await page.evaluate(() => {
    const { adapter, ...rest } = window.__capture
    return { adapter, ...rest }
  })

  // out/ is not cleared: it holds committed records and the shader LICENSE and NOTICE
  mkdirSync(join(OUT, 'shaders'), { recursive: true })
  for (const s of cap.shaders) {
    const safe = (s.label || 'unlabeled').replace(/[^A-Za-z0-9_.-]+/g, '_').slice(0, 80)
    writeFileSync(join(OUT, 'shaders', `${String(s.id).padStart(4, '0')}_${safe}.wgsl`), s.code)
  }
  // sha256 is the shader's identity for cross-run joins; bytes is kept because
  // the committed records join on it. The page hashes the source it saw; if
  // crypto.subtle was unavailable there the same text is hashed here instead.
  const { shaders, ...manifest } = cap
  manifest.shaders = shaders.map(({ id, label, code, shaderSha256 }) => ({
    id,
    label,
    bytes: code.length,
    sha256: shaderSha256 ?? createHash('sha256').update(code).digest('hex'),
  }))
  manifest.provenance = await provenance({ browser, adapter: cap.adapter, pkg: cap.version })
  manifest.bundle = {
    revision: BUNDLE_REVISION,
    bytes: manifest.provenance.model?.bytes ?? BUNDLE_BYTES,
    sha256: manifest.provenance.model?.sha256 ?? BUNDLE_SHA256,
  }
  writeFileSync(join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2))
  if (cap.shaderSha256Unavailable) console.log('note: crypto.subtle was unavailable in the page; shader hashes were computed in the driver from the captured source')

  console.log(`\nshaders: ${shaders.length}  pipelines: ${cap.pipelines.length}`)
  console.log(`counters: ${JSON.stringify(cap.counters)}`)
  if (cap.error) { console.log(`PAGE ERROR: ${cap.error}`); exitCode = 2 }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally {
  await browser.close()
  server.close()
}
process.exit(exitCode)
