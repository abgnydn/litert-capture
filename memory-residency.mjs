// CAPTURE-ROADMAP 14 (memory-residency): GPU memory residency + upload time.
//
// Derives textures/buffers sizes plus peak vs steady state from the committed
// out/trace.json tape (367 textures + 2522 buffers with load/prefill/run
// phases, all real measured sizes) plus the per-token GPU cost from the
// committed out/kernel-profile.json. No new GPU run: the tape already records
// every resource the engine created; the record reports what the tape shows
// and parks what it cannot show (weight-upload timing: the tape carries no
// load timestamps) with the exact next step instead of manufacturing numbers.
//
// Usage: node memory-residency.mjs   (offline, brief Chrome launch for provenance only)
// Writes out/memory-residency.json. Committed records are read, never written.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { provenance } from './provenance.mjs'
import { manifestSha256 } from './shader-sha.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out')
const PORT = 8922

const BPT = { rgba8uint: 4, rgba8unorm: 4, rgba8sint: 4, rgba16float: 8, rgba16uint: 8, rgba32uint: 16, rgba32float: 16, r32uint: 4, r32float: 4, rg32uint: 8, rg32float: 8, r16float: 2, rg16float: 4, r8uint: 1 }
const texBytes = (t) => t.size[0] * t.size[1] * (t.size[2] ?? 1) * (BPT[t.format] ?? 0)
const MIB = (b) => b / 1048576

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
    // No @litert-lm/core is loaded by this derivation (microbench.mjs:152 precedent: package null).
    return await provenance({ browser, adapter, pkg: null })
  } finally {
    await browser.close()
    srv.close()
  }
}

let exitCode = 0
try {
  const T = JSON.parse(readFileSync(join(OUT, 'trace.json'), 'utf8'))
  const KP = JSON.parse(readFileSync(join(OUT, 'kernel-profile.json'), 'utf8'))
  const prov = await freshProvenance()

  const byFormat = {}
  for (const t of T.textures) {
    const b = texBytes(t)
    byFormat[t.format] ??= { n: 0, bytes: 0 }
    byFormat[t.format].n++
    byFormat[t.format].bytes += b
  }
  const texTotal = T.textures.reduce((s, t) => s + texBytes(t), 0)
  const unknownFmt = [...new Set(T.textures.filter((t) => !BPT[t.format]).map((t) => t.format))]
  const bufByPhase = {}
  for (const b of T.buffers) {
    bufByPhase[b.phase] ??= { n: 0, bytes: 0 }
    bufByPhase[b.phase].n++
    bufByPhase[b.phase].bytes += b.size
  }
  const bufTotal = T.buffers.reduce((s, b) => s + b.size, 0)
  const bigBufs = [...T.buffers].sort((a, b) => b.size - a.size).slice(0, 8).map((b) => ({ id: b.id, bytes: b.size, phase: b.phase }))
  const bigTex = [...T.textures].sort((a, b) => texBytes(b) - texBytes(a)).slice(0, 8).map((t) => ({ id: t.id, size: t.size, format: t.format, bytes: texBytes(t), phase: t.phase }))

  // Peak vs steady from tape phases. The tape records creation, not
  // destruction, so peak is everything live at trace end; steady decode is the
  // load-phase base (weights + static state, shared by both) plus the
  // run-phase working set created per traced run.
  const loadBytes = texTotal + (bufByPhase.load?.bytes ?? 0)
  const runBytes = bufByPhase.run?.bytes ?? 0
  const peakBytes = texTotal + bufTotal
  const summary = {
    textures: { n: T.textures.length, mib: MIB(texTotal) },
    buffers: { n: T.buffers.length, mib: MIB(bufTotal) },
    peakMib: MIB(peakBytes),
    steadyLoadBaseMib: MIB(loadBytes),
    decodeRunPhaseMib: MIB(runBytes),
    prefillPhaseBuffers: bufByPhase.prefill ?? { n: 0, bytes: 0 },
    decodeCostPerToken: { splitSumMs: KP.totalMs, tokens: KP.tokens },
  }
  const record = {
    date: new Date().toISOString(),
    method: 'offline derivation from committed out/trace.json (+ committed out/kernel-profile.json split cost per token); sizes are measured tape values, no new GPU run',
    summary,
    texturesByFormat: Object.fromEntries(Object.entries(byFormat).map(([k, v]) => [k, { n: v.n, mib: MIB(v.bytes) }])),
    unknownTextureFormats: unknownFmt,
    buffersByPhase: Object.fromEntries(Object.entries(bufByPhase).map(([k, v]) => [k, { n: v.n, mib: MIB(v.bytes) }])),
    largestBuffers: bigBufs,
    largestTextures: bigTex,
    uploadTiming: {
      status: 'parked',
      reason: 'not-in-tape',
      detail: 'out/trace.json records resource creation (size/format/phase) but no load timestamps, so weight-upload time vs steady state cannot be derived from it. Next step: add __mark spans around weight-texture creation in www/trace.html, re-run node trace.mjs --out memory-residency-trace.json, join by resource id.',
    },
    note: `peak residency ${MIB(peakBytes).toFixed(0)} MiB (${T.textures.length} textures + ${T.buffers.length} buffers live at trace end); steady decode is the ${MIB(loadBytes).toFixed(0)} MiB load base plus the ${MIB(runBytes).toFixed(1)} MiB run-phase working set. Upload timing parked, not estimated.`,
    sources: ['out/trace.json', 'out/kernel-profile.json'],
    manifest_sha256: manifestSha256(join(OUT, 'manifest.json')),
    provenance: prov,
  }
  mkdirSync(OUT, { recursive: true })
  writeFileSync(join(OUT, 'memory-residency.json'), JSON.stringify(record, null, 2))
  console.log(record.note)
  console.log('wrote out/memory-residency.json')
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
}
process.exit(exitCode)
