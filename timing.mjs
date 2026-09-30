// Drives www/timing.html and reports where a decoded token's time goes.
//   node timing.mjs bare   -> tok/s, time to first token, CPU-side vs waiting-on-GPU split
//   node timing.mjs gpu    -> GPU execution time per token from timestamp queries
//
// ROADMAP 16: timestamp-query path when available for GPU timing. Bare uses
// no GPU timestamps (mapAsync CPU split only); gpu requests timestamp-query
// when the adapter exposes it and injects timestampWrites per compute pass,
// kernel additionally splits every dispatch into its own timestamped pass;
// both resolve via resolveQuerySet + copyBufferToBuffer and record gpuNote
// when unavailable. Documented from existing practice in www/timing.html;
// optional ?v= / ?chunks= passthrough not wired here; defaults unchanged.

import { createReadStream, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { provenance } from './provenance.mjs'
import { manifestSha256, shaderShaById } from './shader-sha.mjs'

const MODE = process.argv[2] ?? 'bare'
const HERE = dirname(fileURLToPath(import.meta.url))
const WWW = join(HERE, 'www')
const OUT = join(HERE, 'out')
const PORT = 8917
const TIMEOUT_MS = 20 * 60 * 1000
const STEADY_FROM = 8 // ignore the first chunks of the warm message

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

const browser = await puppeteer.launch({
  headless: false,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist',
    // Chrome rounds timestamp queries to 100 µs by default; passes here are shorter.
    '--disable-dawn-features=timestamp_quantization',
    '--enable-features=Vulkan',
  ],
  protocolTimeout: TIMEOUT_MS,
})

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN }
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * p))] : NaN }
const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')

let exitCode = 0
try {
  const page = await browser.newPage()
  page.on('console', (m) => { const t = m.text(); if (t.startsWith('[timing]')) console.log(t) })
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
  await page.goto(`http://127.0.0.1:${PORT}/timing.html?mode=${MODE}`, { waitUntil: 'load' })
  await page.waitForFunction('window.__timingDone === true', { timeout: TIMEOUT_MS, polling: 1000 })
  const T = await page.evaluate(() => window.__timing)
  T.provenance = await provenance({ browser, adapter: T.adapter, pkg: T.version })
  // ROADMAP 08: tie this record to the capture manifest it joins against.
  T.manifest_sha256 = manifestSha256(join(OUT, 'manifest.json'))

  mkdirSync(OUT, { recursive: true })
  writeFileSync(join(OUT, `timing-${MODE}.json`), JSON.stringify(T, null, 2))
  if (T.error) { console.log(`PAGE ERROR: ${T.error}`); exitCode = 2 }

  const mark = (n) => T.marks.find((m) => m.name === n)?.t
  console.log(`\n== ${MODE} | @litert-lm/core ${T.version} | ${JSON.stringify(T.adapter)}`)
  console.log(`engine create: ${f(mark('engine-created') - mark('import-done'), 0)} ms`)
  for (const label of ['cold', 'warm']) {
    console.log(`${label}: first chunk after ${f(mark(`${label}:chunk-0`) - mark(`${label}:send`), 0)} ms, ${T[label]?.chunks} chunks`)
  }

  const chunkT = T.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
  const gaps = chunkT.slice(STEADY_FROM).map((t, i) => t - chunkT[STEADY_FROM + i - 1]).slice(1)
  console.log(`warm steady state: median ${f(median(gaps))} ms per chunk (p10 ${f(pct(gaps, 0.1))}, p90 ${f(pct(gaps, 0.9))}) = ${f(1000 / median(gaps), 1)} chunks/s over ${gaps.length} chunks`)

  // CPU-side vs waiting-on-GPU, from the one mapAsync per token.
  const t0 = chunkT[STEADY_FROM]
  const t1 = mark('warm:done')
  const maps = T.maps.filter((m) => m.call >= t0 && m.call <= t1 && m.resolve > 0)
  const cpu = []
  const wait = []
  for (let i = 1; i < maps.length; i++) {
    cpu.push(maps[i].call - maps[i - 1].resolve)
    wait.push(maps[i].resolve - maps[i].call)
  }
  console.log(`readbacks in window: ${maps.length} (${f(maps.length / Math.max(1, gaps.length + 1), 2)} per chunk)`)
  console.log(`  CPU side, readback resolved -> next readback requested: median ${f(median(cpu))} ms (p10 ${f(pct(cpu, 0.1))}, p90 ${f(pct(cpu, 0.9))})`)
  console.log(`  waiting for the GPU, readback requested -> resolved:    median ${f(median(wait))} ms (p10 ${f(pct(wait, 0.1))}, p90 ${f(pct(wait, 0.9))})`)

  if (MODE === 'gpu') {
    console.log(`gpu: ${T.gpuNote}`)
    const byChunk = new Map()
    for (const p of T.passes) {
      if (p.beginNs == null) continue
      if (!byChunk.has(p.chunk)) byChunk.set(p.chunk, [])
      byChunk.get(p.chunk).push(p)
    }
    const groups = [...byChunk.values()]
    const full = groups.filter((g) => g.length === median(groups.map((x) => x.length)))
    const busy = full.map((g) => g.reduce((s, p) => s + (p.endNs - p.beginNs), 0) / 1e6)
    const span = full.map((g) => (Math.max(...g.map((p) => p.endNs)) - Math.min(...g.map((p) => p.beginNs))) / 1e6)
    const disp = full.map((g) => g.reduce((s, p) => s + p.dispatches, 0))
    console.log(`  tokens with a full set of passes: ${full.length} (passes per token ${full[0]?.length}, dispatches per token ${median(disp)})`)
    console.log(`  GPU execution per token, sum of pass durations: median ${f(median(busy))} ms (p10 ${f(pct(busy, 0.1))}, p90 ${f(pct(busy, 0.9))})`)
    console.log(`  GPU first-pass-start to last-pass-end per token:  median ${f(median(span))} ms`)
    const negative = T.passes.filter((p) => p.beginNs != null && p.endNs < p.beginNs).length
    if (negative) console.log(`  WARNING: ${negative} passes have end < begin; timestamps unreliable`)
  }

  if (MODE === 'kernel') {
    console.log(`gpu: ${T.gpuNote}`)
    console.log(`validation errors: ${T.gpuErrors.length}${T.gpuErrors.length ? ' -> ' + T.gpuErrors[0] : ''}; debug-group calls: ${T.debugGroupCalls}`)
    let reference = null
    try { reference = JSON.parse(readFileSync(join(OUT, 'timing-bare.json'), 'utf8')).warm.text } catch {}
    console.log(`warm output identical to the unmodified (bare) run: ${reference == null ? 'no bare run to compare' : reference === T.warm.text}`)

    // pipeline ids are creation order; check they line up with the capture manifest.
    // A pipeline is matched to the shader it was built from by the SHA-256 of the
    // WGSL text when both records carry it. Byte length is not an identity (16
    // pairs of distinct shaders in this capture share a length), so it is only
    // the fallback for records written before the hash was recorded, and the
    // printed note says which join was used.
    let manifest = null
    try { manifest = JSON.parse(readFileSync(join(OUT, 'manifest.json'), 'utf8')) } catch {}
    const fileSha = shaderShaById(join(OUT, 'shaders'))
    const capSha = (id) => manifest?.shaders[id]?.sha256 ?? fileSha.get(id) ?? null
    const byHash = !!manifest && T.pipelines.every((p) => typeof p.shaderSha256 === 'string') &&
      manifest.shaders.every((s) => capSha(s.id))
    const aligned = manifest && manifest.pipelines.length === T.pipelines.length &&
      T.pipelines.every((p, i) => (byHash
        ? capSha(manifest.pipelines[i].shaderId) === p.shaderSha256
        : manifest.shaders[manifest.pipelines[i].shaderId]?.bytes === p.shaderBytes))
    console.log(`pipeline ids line up with out/manifest.json: ${aligned} (joined by ${byHash ? 'shader SHA-256' : 'WGSL byte length: this record carries no shader hash, and byte length is not an identity'})`)

    const byChunk = new Map()
    for (const p of T.passes) {
      if (p.beginNs == null) continue
      if (!byChunk.has(p.chunk)) byChunk.set(p.chunk, [])
      byChunk.get(p.chunk).push(p)
    }
    const tokens = [...byChunk.values()].filter((g) => g.length === 1304 && g.every((p) => p.dispatches === 1))
    console.log(`fully split tokens: ${tokens.length} (of ${byChunk.size} chunk groups: ${[...byChunk.values()].map((g) => g.length).join(', ')})`)
    if (tokens.length) {
      const perToken = tokens.map((g) => g.reduce((s, p) => s + (p.endNs - p.beginNs), 0) / 1e6)
      console.log(`sum of per-dispatch GPU time per token: median ${f(median(perToken))} ms (unsplit measurement was ~16.0 ms)`)

      const pids = [...new Set(tokens.flatMap((g) => g.map((p) => p.pid)))]
      const rows = pids.map((pid) => {
        const totals = tokens.map((g) => g.filter((p) => p.pid === pid).reduce((s, p) => s + (p.endNs - p.beginNs), 0) / 1e6)
        const each = tokens.flatMap((g) => g.filter((p) => p.pid === pid).map((p) => (p.endNs - p.beginNs) / 1e3))
        const n = tokens[0].filter((p) => p.pid === pid).length
        const wg = tokens[0].find((p) => p.pid === pid)?.wg
        const shaderId = aligned ? manifest.pipelines[pid]?.shaderId : -1
        return { pid, n, ms: median(totals), us: median(each), usMax: Math.max(...each), wg, shaderId, bytes: T.pipelines[pid]?.shaderBytes }
      }).sort((a, b) => b.ms - a.ms)
      const total = rows.reduce((s, r) => s + r.ms, 0)
      writeFileSync(join(OUT, 'kernel-profile.json'), JSON.stringify({ tokens: tokens.length, totalMs: total, rows, provenance: T.provenance, manifest_sha256: T.manifest_sha256 }, null, 2))

      console.log(`\n  ms/token  share   cum   count  median us  shader file            bytes  first dispatch size`)
      let cum = 0
      for (const r of rows.slice(0, 20)) {
        cum += r.ms
        console.log(`  ${f(r.ms, 3).padStart(8)} ${f((100 * r.ms) / total, 1).padStart(5)}% ${f((100 * cum) / total, 0).padStart(4)}% ${String(r.n).padStart(6)} ${f(r.us, 1).padStart(10)}  ${String(r.shaderId).padStart(4, '0')}_unlabeled.wgsl ${String(r.bytes).padStart(7)}  ${JSON.stringify(r.wg)}`)
      }
      console.log(`  (${rows.length} kernels in decode; full table in out/kernel-profile.json)`)
    }
  }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally {
  await browser.close()
  server.close()
}
process.exit(exitCode)
