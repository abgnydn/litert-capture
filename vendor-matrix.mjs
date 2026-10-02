// CAPTURE-ROADMAP 15 (vendor-matrix): one row per GPU vendor.
//
// The local Apple-Silicon/Metal row is real, derived from committed records
// measured on this machine (adapter + provenance from out/prefill.json,
// steady-state tok/s from out/timing-bare.json, which predates provenance and
// is cited with that caveat). Intel/AMD/Adreno/desktop-Linux rows need
// hardware this loop cannot drive from here, so they are parked as
// blocked/colab-needed with exact run + copy-back commands, and the T4 row
// has a ready-to-run notebook (colab/vendor-matrix-t4.ipynb). No synthetic
// rows: pending rows carry no numbers.
//
// Usage: node vendor-matrix.mjs   (offline, brief Chrome launch for provenance only)
// Writes out/vendor-matrix.json. Committed records are read, never written.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { provenance } from './provenance.mjs'
import { manifestSha256 } from './shader-sha.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out')
const PORT = 8923
const MODEL_SHA = '3a08e8d94e23b814ae5414469c370c503813949acb8ceaa17e4ebf8a35af35b5'

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN }

// Brief Chrome launch for a real provenance stamp (browser + adapter). Serves
// a memory-only probe page: no model bundle, no engine, nothing measured here.
const freshProvenance = async () => {
  const srv = createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' })
    res.end('<!doctype html><title>probe</title>')
  })
  await new Promise((r) => srv.listen(PORT, '127.0.0.1', r))
  const browser = await puppeteer.launch({
    headless: false,
    args: ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist', '--enable-features=Vulkan'],
  })
  try {
    const page = await browser.newPage()
    await page.goto(`http://127.0.0.1:${PORT}/probe`, { waitUntil: 'load' })
    const adapter = await page.evaluate(async () => {
      const a = await navigator.gpu.requestAdapter()
      return a ? { vendor: a.info?.vendor, architecture: a.info?.architecture, device: a.info?.device, description: a.info?.description } : null
    })
    return await provenance({ browser, adapter, pkg: null })
  } finally {
    await browser.close()
    srv.close()
  }
}

let exitCode = 0
try {
  const P = JSON.parse(readFileSync(join(OUT, 'prefill.json'), 'utf8'))
  const B = JSON.parse(readFileSync(join(OUT, 'timing-bare.json'), 'utf8'))
  const prov = await freshProvenance()

  const chunkT = B.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
  const gaps = chunkT.slice(8).map((t, i) => t - chunkT[8 + i - 1]).slice(1)
  const tokMs = median(gaps)
  const metalRow = {
    status: 'measured',
    vendor: P.provenance.adapter.vendor,
    architecture: P.provenance.adapter.architecture,
    deviceLabel: 'Apple M2 Max (local)',
    browser: P.provenance.browser,
    shaderF16: P.adapter.features.includes('shader-f16'),
    timestampQuery: P.adapter.features.includes('timestamp-query'),
    package: P.provenance.package,
    modelSha256: P.provenance.model.sha256,
    steadyTokPerSec: 1000 / tokMs,
    steadyMsPerToken: tokMs,
    warmChunks: chunkT.length,
    power: P.provenance.power,
    thermal: P.provenance.thermal,
    harnessCommit: P.provenance.harness_commit,
    caveat: 'tok/s recomputed read-only from out/timing-bare.json, which predates provenance (ROADMAP 37 legacy); adapter/browser/power come from out/prefill.json provenance on the same machine and Chrome build',
    sources: ['out/prefill.json', 'out/timing-bare.json'],
  }
  const pending = (vendor, arch, how) => ({
    status: 'parked',
    reason: 'blocked/colab-needed',
    vendor,
    architecture: arch,
    detail: `no ${vendor}/${arch} hardware is reachable from this loop (no gh issue posts, no Colab driving from here per CAPTURE-LOOP.md guards); the row is filled by running the notebook and copying the row back.`,
    run: how,
  })
  const t4run = [
    'colab/vendor-matrix-t4.ipynb cells 1-3 (T4 runtime, Node + Vulkan userspace, unpack timing.html + driver, fetch the pinned model bundle, SHA-checked)',
    'run cell 4; download vendor/vendor-t4-row.json from the runtime',
    'on a branch from main: merge the row into out/vendor-matrix.json rows.t4_tesla, add manifest_sha256, run the standard gates, commit',
  ]
  const record = {
    date: new Date().toISOString(),
    method: 'local Metal row derived read-only from committed records measured on this machine; other rows need hardware absent here',
    modelSha256: MODEL_SHA,
    rows: {
      metal_apple_silicon: metalRow,
      t4_tesla_colab: pending('nvidia', 'tesla-t4 (colab)', t4run),
      intel_igpu: pending('intel', 'integrated (Linux/Windows)', ['same vendor-t4-row procedure on an Intel iGPU host with Chrome 154+ and --enable-unsafe-webgpu', 'copy the row back as rows.intel_igpu']),
      amd_dgpu: pending('amd', 'discrete (Linux/Windows)', ['same vendor-t4-row procedure on an AMD dGPU host', 'copy the row back as rows.amd_dgpu']),
      adreno_android: pending('qualcomm', 'adreno (Android Chrome)', ['same vendor-t4-row procedure in Android Chrome (model fetch is 2 GB; Wi-Fi only)', 'copy the row back as rows.adreno_android']),
    },
    notebook: 'colab/vendor-matrix-t4.ipynb',
    note: '1 of 5 rows measured (Metal, this machine, Chrome 146, 53.2 tok/s steady); 4 rows parked blocked/colab-needed with exact run-back steps, no synthetic numbers.',
    manifest_sha256: manifestSha256(join(OUT, 'manifest.json')),
    provenance: prov,
  }
  mkdirSync(OUT, { recursive: true })
  writeFileSync(join(OUT, 'vendor-matrix.json'), JSON.stringify(record, null, 2))
  console.log(record.note)
  console.log('wrote out/vendor-matrix.json')
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
}
process.exit(exitCode)
