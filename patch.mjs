// In-model A/B: runs www/patch.html in a fresh Chrome for each of six patch
// conditions — unpatched, all four matrix-vector kernels split deeper, and the
// four leave-one-out sets — and reports steady-state decode time. The condition
// order is reshuffled independently inside every repetition, so drift over the
// sweep (thermals, page cache, GPU clocks) cannot line up with one condition;
// the realized order is recorded. The greedy-decoded text of every run is
// diffed against the unpatched run at token granularity, not compared for
// equality, so a difference is reported as the tokens that differ and where.
//
// ROADMAP 13: the reshuffle above is the in-model analogue of the microbench
// forward+reverse pooling (each variant timed forward then reverse, both positions
// pooled before the median, R.order records per-position medians); both cancel
// linear clock drift. Always on, no flag; default unchanged.
//
// ROADMAP 19: 6 conditions x 5 reps = 30 runs (default `node patch.mjs 5`).
// Each repetition Fisher-Yates reshuffles CONDITIONS with one recorded seed
// (out/patch-ab.json shuffleSeed 1381548739, orders[] holds the realized
// order); each run launches a fresh Chrome in once() and closes it after.
// Documented from existing practice; defaults unchanged.
//
// ROADMAP 20: scaled dispatches hold split wins at size. Every dispatch of a
// patched pipeline is scaled oldWgX/newWgX so the same output slices are
// covered (www/patch.html dispatchWorkgroups hook; P.scaledDispatches vs
// P.unscaledDispatches in the record). Documented from existing practice;
// defaults unchanged.
//
// ROADMAP 23: note output near-tie where patched vs baseline outputs match
// within tolerance. Cold texts are identical (60 tokens); warm texts differ
// by one token ("twenty," at index 20, 58 vs 60 tokens) in out/patch-ab.json
// runs. Documented from existing practice; defaults unchanged.
//
// Usage: node patch.mjs [repeats]   (default 5, i.e. 30 runs)
// Writes out/patch-ab.json. out/patch.json, the record of the earlier
// four-set, two-repetition run, is not touched.

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
const PORT = 8917
const TIMEOUT_MS = 20 * 60 * 1000
const STEADY_FROM = 8
const REPEATS = Number(process.argv[2] ?? 5)
// name -> the ?patch= list www/patch.html parses. 'all' is all four; a comma
// list selects exactly those captured shader ids, so each leave-one-out set
// names the three kernels it keeps.
const CONDITIONS = [
  { condition: 'baseline', patch: 'none' },
  { condition: 'all-four', patch: 'all' },
  { condition: 'no-0099', patch: '105,112,113' },
  { condition: 'no-0105', patch: '99,112,113' },
  { condition: 'no-0112', patch: '99,105,113' },
  { condition: 'no-0113', patch: '99,105,112' },
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

// Each repetition gets its own shuffle from one recorded seed, so the realized
// orders can be reproduced from the record.
const SEED = Date.now() % 2 ** 31
let seed = SEED >>> 0
const rnd = () => {
  seed = (seed * 1664525 + 1013904223) >>> 0
  return seed / 2 ** 32
}
const shuffle = (a) => {
  const x = [...a]
  for (let i = x.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    const t = x[i]
    x[i] = x[j]
    x[j] = t
  }
  return x
}

// A GPU run on battery is not comparable with one on mains, so the sweep checks
// before every run rather than discovering it afterwards in the provenance.
const powerState = () => {
  const batt = execFileSync('pmset', ['-g', 'batt'], { encoding: 'utf8' })
  const pct = batt.match(/(\d+)%/)
  return { source: /AC Power/.test(batt) ? 'AC' : /Battery Power/.test(batt) ? 'battery' : null, percent: pct ? Number(pct[1]) : null }
}

// Word-level diff. Whole-string equality says only that two messages differ;
// this says which tokens and where, which is what a near-tie in f16 rounding
// shows up as. Hyphenated words stay one token ("twenty-one").
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
const diffLine = (d) => (d.identical ? 'identical' : d.hunks.map((h) => `@${h.baselineIndex} -[${h.removed.join(' ')}] +[${h.added.join(' ')}]`).join('; '))

const once = async ({ condition, patch }) => {
  const browser = await puppeteer.launch({ headless: false, args: ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist', '--enable-features=Vulkan'], protocolTimeout: TIMEOUT_MS })
  try {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
    await page.goto(`http://127.0.0.1:${PORT}/patch.html?patch=${encodeURIComponent(patch)}`, { waitUntil: 'load' })
    await page.waitForFunction('window.__patchDone === true', { timeout: TIMEOUT_MS, polling: 1000 })
    const P = await page.evaluate(() => window.__patch)
    const mark = (n) => P.marks.find((m) => m.name === n)?.t
    const chunkT = P.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
    const gaps = chunkT.slice(STEADY_FROM).map((t, i) => t - chunkT[STEADY_FROM + i - 1]).slice(1)
    const prov = await provenance({ browser, adapter: P.adapter, pkg: P.version })
    const msPerToken = median(gaps)
    return { condition, patch, want: P.want, error: P.error, gpuErrors: P.gpuErrors, patched: P.patched, scaled: P.scaledDispatches, unscaled: P.unscaledDispatches, cold: P.cold, warm: P.warm, msPerToken, tokPerSec: 1000 / msPerToken, n: gaps.length, firstChunkMs: mark('warm:chunk-0') - mark('warm:send'), engineMs: mark('engine-created') - mark('import-done'), provenance: prov }
  } finally { await browser.close() }
}

const runs = []
const orders = []
let exitCode = 0
// The record is rewritten after every run: a sweep is half an hour of GPU time
// and a crash at run 29 should not lose the first 28.
const save = (extra = {}) => {
  mkdirSync(OUT, { recursive: true })
  // ROADMAP 08: tie this record to the capture manifest it joins against.
  writeFileSync(join(OUT, 'patch-ab.json'), JSON.stringify({ date: new Date().toISOString(), repeats: REPEATS, steadyFrom: STEADY_FROM, conditions: CONDITIONS, shuffleSeed: SEED, manifest_sha256: manifestSha256(join(OUT, 'manifest.json')), orders, failure: null, summary: [], ...extra, runs }, null, 2))
}
// A run that errored, raised a GPU validation error or did not patch every
// kernel it was asked to makes every comparison in the sweep meaningless, so
// the sweep stops there rather than averaging it in silently.
let failure = null
try {
  sweep: for (let r = 0; r < REPEATS; r++) {
    const order = shuffle(CONDITIONS)
    orders.push({ rep: r + 1, order: order.map((c) => c.condition) })
    console.log(`\n== repetition ${r + 1} of ${REPEATS}, order: ${order.map((c) => c.condition).join(' -> ')}`)
    for (let k = 0; k < order.length; k++) {
      const C = order[k]
      const pw = powerState()
      if (pw.source !== 'AC' || !(pw.percent >= 30)) {
        failure = `rep ${r + 1} position ${k + 1} (${C.condition}) NOT RUN: power is ${pw.source} at ${pw.percent}%, the sweep needs AC and 30% or more`
        break sweep
      }
      const R = { rep: r + 1, position: k + 1, ...(await once(C)) }
      runs.push(R)
      save()
      const missed = (R.want ?? []).filter((id) => !R.patched.some((p) => p.id === id))
      if (R.error) failure = `rep ${r + 1} position ${k + 1} condition ${C.condition} (patch=${C.patch}) FAILED, the run did not complete cleanly: ${R.error}`
      else if (R.gpuErrors.length) failure = `rep ${r + 1} position ${k + 1} condition ${C.condition} (patch=${C.patch}) FAILED, ${R.gpuErrors.length} GPU validation error(s): ${R.gpuErrors.join(' | ')}`
      else if (missed.length) failure = `rep ${r + 1} position ${k + 1} condition ${C.condition} (patch=${C.patch}) FAILED, requested kernels were never patched: ${missed.join(', ')}`
      if (failure) break sweep
      console.log(`  rep ${r + 1} pos ${k + 1} ${C.condition.padEnd(9)} ${f(R.msPerToken)} ms/token = ${f(R.tokPerSec, 1)} tok/s over ${R.n} tokens | patched ${R.patched.length} pipelines, ${R.scaled} scaled dispatches | GPU errors ${R.gpuErrors.length} | engine ${f(R.engineMs, 0)} ms | chunks cold ${R.cold.chunks} warm ${R.warm.chunks}`)
    }
  }

  // Text is diffed against the first completed unpatched run.
  const ref = runs.find((R) => R.condition === 'baseline' && !R.error && R.cold && R.warm)
  if (ref) for (const R of runs) {
    if (R.cold && R.warm) R.diffVsBaseline = { cold: tokenDiff(ref.cold.text, R.cold.text), warm: tokenDiff(ref.warm.text, R.warm.text) }
  }

  // Only a sweep that ran clean to the end gets a summary; a stopped sweep is
  // written out raw so the failed run can be read, and nothing is averaged.
  const summary = []
  if (!failure) {
    const baseMs = median(runs.filter((R) => R.condition === 'baseline').map((R) => R.msPerToken))
    for (const { condition, patch } of CONDITIONS) {
      const rs = runs.filter((R) => R.condition === condition)
      if (!rs.length) continue
      const ms = rs.map((R) => R.msPerToken)
      const med = median(ms)
      const min = Math.min(...ms)
      const max = Math.max(...ms)
      summary.push({
        condition, patch, n: rs.length,
        msPerToken: { median: med, min, max },
        tokPerSec: { median: 1000 / med, min: 1000 / max, max: 1000 / min },
        speedupVsBaseline: baseMs / med,
        scaledDispatches: [...new Set(rs.map((R) => R.scaled))],
        patchedPipelines: [...new Set(rs.map((R) => R.patched.length))],
        coldChunks: [...new Set(rs.map((R) => R.cold.chunks))],
        warmChunks: [...new Set(rs.map((R) => R.warm.chunks))],
        coldTextIdenticalToBaseline: rs.every((R) => R.diffVsBaseline?.cold.identical === true),
        warmTextIdenticalToBaseline: rs.every((R) => R.diffVsBaseline?.warm.identical === true),
      })
    }
  }
  save({ failure, summary })

  if (failure) {
    console.log(`\n${failure}`)
    console.log('no summary is comparable: the sweep stopped at the first failed run; out/patch-ab.json holds what was collected')
    exitCode = 2
  } else {
    console.log(`\n== summary over ${REPEATS} repetitions (text diffed against the unpatched run at token granularity)`)
    for (const s of summary) {
      console.log(`  ${s.condition.padEnd(9)} ${f(s.msPerToken.median)} ms/token (${f(s.msPerToken.min)}-${f(s.msPerToken.max)})  ${f(s.tokPerSec.median, 1)} tok/s (${f(s.tokPerSec.min, 1)}-${f(s.tokPerSec.max, 1)})  x${f(s.speedupVsBaseline, 3)} vs unpatched  n=${s.n}  scaled dispatches ${s.scaledDispatches.join('/')}`)
    }
    console.log('\n== token diff against the unpatched run')
    for (const s of summary) {
      const rs = runs.filter((R) => R.condition === s.condition)
      const cold = [...new Set(rs.map((R) => diffLine(R.diffVsBaseline.cold)))]
      const warm = [...new Set(rs.map((R) => diffLine(R.diffVsBaseline.warm)))]
      console.log(`  ${s.condition.padEnd(9)} cold: ${cold.join(' || ')}`)
      console.log(`  ${''.padEnd(9)} warm: ${warm.join(' || ')}`)
    }
  }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`)
  exitCode = 1
} finally { server.close() }
process.exit(exitCode)
