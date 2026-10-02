// CAPTURE-ROADMAP 16 (correctness-gate): token-diff across prompts.
//
// Generalizes the patch.mjs near-tie analysis (baseline vs patched text
// identical cold, one-token warm difference) from the single committed prompt
// pair to three prompt pairs: the original thirty-count plus a ten-count and
// a weekday list, each run unpatched (patch=none) and fully patched
// (patch=all) in a fresh Chrome. Word-level diff per prompt plus a gate
// verdict: cold identical on every prompt, warm at most one hunk, patched
// pipelines complete on every run. One re-run max per cell (R5).
//
// Usage: node correctness-gate.mjs   (6 fresh-Chrome model runs, ~10 min)
// Writes out/correctness-gate.json. Committed out/*.json are read, never written.

import { execFileSync } from 'node:child_process'
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
const PORT = 8924
const TIMEOUT_MS = 20 * 60 * 1000
const STEADY_FROM = 8

const PROMPTS = [
  { id: 'thirty', cold: 'Count from one to thirty in English words, separated by commas.', warm: 'Count down from thirty to one in English words, separated by commas.' },
  { id: 'ten', cold: 'Count from one to ten in English words, separated by commas.', warm: 'Count down from ten to one in English words, separated by commas.' },
  { id: 'weekdays', cold: 'List the days Monday to Sunday in order, separated by commas.', warm: 'List the days Sunday to Monday in reverse order, separated by commas.' },
]
const CONDITIONS = [
  { condition: 'baseline', patch: 'none' },
  { condition: 'all-four', patch: 'all' },
]

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

// Word-level diff, same construction as patch.mjs tokenDiff: whole-string
// equality says only that two messages differ; this says which tokens and
// where, which is what a near-tie in f16 rounding shows up as.
const tokenize = (s) => s.match(/[\w'’-]+|[^\s\w'’-]+/g) ?? []
const tokenDiff = (baseText, runText) => {
  const a = tokenize(baseText)
  const b = tokenize(runText)
  const L = Array.from({ length: a.length + 1 }, () => new Int32Array(b.length + 1))
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1])
  }
  const hunks = []
  let i = 0
  let j = 0
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) { i++; j++; continue }
    const at = { baselineIndex: i, runIndex: j, removed: [], added: [] }
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) break
      if (j >= b.length || (i < a.length && L[i + 1][j] >= L[i][j + 1])) at.removed.push(a[i++])
      else at.added.push(b[j++])
    }
    hunks.push(at)
  }
  return { identical: hunks.length === 0, baselineTokens: a.length, runTokens: b.length, hunks }
}

// A GPU run on battery is not comparable with one on mains, so the sweep
// checks before every run (patch.mjs precedent).
const powerOk = () => {
  const batt = execFileSync('pmset', ['-g', 'batt'], { encoding: 'utf8' })
  const pct = batt.match(/(\d+)%/)
  return { source: /AC Power/.test(batt) ? 'AC' : /Battery Power/.test(batt) ? 'battery' : null, percent: pct ? Number(pct[1]) : null }
}

const once = async ({ condition, patch, prompt }) => {
  const browser = await puppeteer.launch({ headless: false, args: ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist', '--enable-features=Vulkan'], protocolTimeout: TIMEOUT_MS })
  try {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
    const q = `patch=${encodeURIComponent(patch)}&cold=${encodeURIComponent(prompt.cold)}&warm=${encodeURIComponent(prompt.warm)}`
    await page.goto(`http://127.0.0.1:${PORT}/patch.html?${q}`, { waitUntil: 'load' })
    await page.waitForFunction('window.__patchDone === true', { timeout: TIMEOUT_MS, polling: 1000 })
    const P = await page.evaluate(() => window.__patch)
    const chunkT = P.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
    const gaps = chunkT.slice(STEADY_FROM).map((t, i) => t - chunkT[STEADY_FROM + i - 1]).slice(1)
    const prov = await provenance({ browser, adapter: P.adapter, pkg: P.version })
    return { condition, patch, promptId: prompt.id, error: P.error ?? null, gpuErrors: P.gpuErrors, patched: P.patched, cold: P.cold, warm: P.warm, msPerToken: median(gaps), provenance: prov }
  } finally { await browser.close() }
}

const runs = []
let exitCode = 0
let failure = null
try {
  for (const prompt of PROMPTS) {
    for (const C of CONDITIONS) {
      const pw = powerOk()
      if (pw.source !== 'AC') {
        failure = `prompt ${prompt.id} ${C.condition} NOT RUN: power is ${pw.source}, the sweep needs AC`
        break
      }
      let R = await once({ ...C, prompt })
      // R5: one re-run max, then keep what was seen.
      if (R.error) {
        console.log(`${prompt.id}/${C.condition} errored: ${String(R.error).slice(0, 160)}; re-running once`)
        R = await once({ ...C, prompt })
      }
      runs.push(R)
      const missed = (C.patch === 'all' ? [99, 105, 112, 113] : []).filter((id) => !R.patched.some((p) => p.id === id))
      if (R.error) failure = `prompt ${prompt.id} ${C.condition} FAILED twice: ${String(R.error).slice(0, 200)}`
      else if (R.gpuErrors.length) failure = `prompt ${prompt.id} ${C.condition} FAILED: ${R.gpuErrors.length} GPU validation error(s)`
      else if (missed.length) failure = `prompt ${prompt.id} ${C.condition} FAILED: kernels never patched: ${missed.join(',')}`
      else console.log(`${prompt.id}/${C.condition}: ${f(R.msPerToken)} ms/token, cold ${R.cold.chunks} chunks, warm ${R.warm.chunks} chunks, patched ${R.patched.length}`)
      if (failure) break
    }
    if (failure) break
  }

  if (failure) throw new Error(failure)

  // Gate: per prompt, baseline vs all-four token diff; verdict generalizes the
  // committed near-tie (cold identical, warm <= 1 hunk).
  const perPrompt = PROMPTS.map((prompt) => {
    const base = runs.find((R) => R.promptId === prompt.id && R.condition === 'baseline')
    const patched = runs.find((R) => R.promptId === prompt.id && R.condition === 'all-four')
    const cold = tokenDiff(base.cold.text, patched.cold.text)
    const warm = tokenDiff(base.warm.text, patched.warm.text)
    return {
      prompt: prompt.id,
      baselineMsPerToken: base.msPerToken,
      patchedMsPerToken: patched.msPerToken,
      speedupVsBaseline: base.msPerToken / patched.msPerToken,
      cold,
      warm,
      pass: cold.identical && warm.hunks.length <= 1,
    }
  })
  const verdict = perPrompt.every((p) => p.pass)
    ? 'PASS: cold identical and warm within one hunk on all 3 prompt pairs; the near-tie generalizes beyond the committed thirty-count pair'
    : 'FAIL: token difference exceeds the near-tie bound on at least one prompt pair'
  for (const p of perPrompt) {
    console.log(`${p.prompt}: cold ${p.cold.identical ? 'identical' : `${p.cold.hunks.length} hunks`} (${p.cold.baselineTokens} vs ${p.cold.runTokens} tokens), warm ${p.warm.identical ? 'identical' : `${p.warm.hunks.length} hunks`} (${p.warm.baselineTokens} vs ${p.warm.runTokens} tokens) -> ${p.pass ? 'pass' : 'FAIL'}`)
  }
  console.log(verdict)
  const record = {
    date: new Date().toISOString(),
    method: 'fresh baseline vs all-four patch.html runs on 3 prompt pairs (?cold= ?warm=), word-level token diff per pair',
    prompts: PROMPTS.map((p) => p.id),
    perPrompt,
    verdict,
    gate: 'cold identical on every prompt AND warm hunks <= 1 on every prompt',
    manifest_sha256: manifestSha256(join(OUT, 'manifest.json')),
    runs,
  }
  // Provenance lives per run (fresh Chrome each); the record carries the first.
  record.provenance = runs[0].provenance
  mkdirSync(OUT, { recursive: true })
  writeFileSync(join(OUT, 'correctness-gate.json'), JSON.stringify(record, null, 2))
  console.log('wrote out/correctness-gate.json')
  if (!verdict.startsWith('PASS')) exitCode = 2
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally { server.close() }
process.exit(exitCode)
