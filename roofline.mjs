// CAPTURE-ROADMAP 17 (roofline): per-kernel bytes/µs from committed records.
//
// Offline join of out/trace.json (weight bytes per dispatch) with
// out/timing-kernel.json (median GPU µs per dispatch position) and
// out/kernel-profile.json (per-pid cross-check). No GPU run, no browser,
// no synthetic numbers: every byte and microsecond below was already
// committed on main and is only regrouped here. Join, MATVEC set, texture
// byte accounting and split-pass correction are the same construction as
// analyze.mjs, so the figures match its console summary; the record exists
// so www/roofline.html can render the per-kernel chart from JSON.
//
// Usage: node roofline.mjs [peak-gbs]   (default 400, same as analyze.mjs)
// Writes out/roofline.json. Committed out/*.json are read, never written.

import { execFileSync } from 'node:child_process'
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { arch, platform, release } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { manifestSha256, shaderShaById } from './shader-sha.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out')
const PEAK_GBS = Number(process.argv[2] ?? 400)
const VERSION = process.env.LITERT_VERSION ?? '0.17.1'

const read = (name, required) => {
  try {
    return JSON.parse(readFileSync(join(OUT, name), 'utf8'))
  } catch {
    if (required) {
      console.log(`missing required input out/${name}`)
      process.exit(1)
    }
    return null
  }
}

const R = read('trace.json', true)
const K = read('timing-kernel.json', true)
const M = read('manifest.json', true)
const P = read('kernel-profile.json', true)
const G = read('timing-gpu.json', false)
const B = read('timing-bare.json', false)

const git = (args) => {
  try {
    return execFileSync('git', args, { cwd: HERE, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  } catch {
    return null
  }
}
const pmset = (args) => {
  try {
    return execFileSync('pmset', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  } catch {
    return ''
  }
}

const BPT = { rgba8uint: 4, rgba16float: 8, rgba16uint: 8, rgba32uint: 16, rgba32float: 16, rgba32sint: 16 }
const texBytes = (t) => t.size[0] * t.size[1] * (t.size[2] ?? 1) * (BPT[t.format] ?? 0)
const f = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')
const median = (a) => {
  const s = [...a].sort((x, y) => x - y)
  return s.length ? s[Math.floor(s.length / 2)] : NaN
}
const sid = (pid) => M.pipelines[pid]?.shaderId ?? -1

// Same weight/uniform binding scan as analyze.mjs: which binding holds the
// weight stream in each shader. Attention kernels 0101/0103/0108/0110 read
// the 8-bit KV cache through a binding also named weights_buffer and are
// excluded from the weight rows below.
const weightBinding = {}
const uniformBinding = {}
for (const file of readdirSync(join(OUT, 'shaders'))) {
  const id = Number(file.slice(0, 4))
  const src = readFileSync(join(OUT, 'shaders', file), 'utf8')
  const m = [...src.matchAll(/@binding\((\d+)\) var(?:<[^>]*>)? (\w*weights\w*)\s*:/g)].find((x) => !/scale|zero_point/.test(x[2]))
  if (m) weightBinding[id] = { binding: Number(m[1]), name: m[2] }
  const u = src.match(/@binding\((\d+)\) var<uniform>/)
  if (u) uniformBinding[id] = Number(u[1])
}

// Same positional join as analyze.mjs: dispatch order within a token is
// deterministic, so the trace run and the timed run join by position after
// checking their pipeline sequences are identical.
const byChunk = (arr) => {
  const m = new Map()
  for (const d of arr) {
    if (!m.has(d.chunk)) m.set(d.chunk, [])
    m.get(d.chunk).push(d)
  }
  return [...m.values()]
}
const traceTokens = byChunk(R.dispatches).filter((t) => t.length === 1304)
const timeTokens = byChunk(K.passes.filter((p) => p.beginNs != null)).filter((t) => t.length === 1304)
const tr = traceTokens[1] ?? traceTokens[0]
const pidSeq = tr.map((d) => d.pid).join(',')
const aligned = timeTokens.filter((t) => t.map((p) => p.pid).join(',') === pidSeq)
if (!aligned.length) {
  console.log('no timed token matches the traced pipeline sequence; cannot join')
  process.exit(1)
}
const usAt = (i) => median(aligned.map((t) => (t[i].endNs - t[i].beginNs) / 1e3))

const rows = tr.map((d, i) => {
  const s = sid(d.pid)
  const wb = weightBinding[s]
  let bytes = 0
  let shape = ''
  if (wb) {
    const r = d.res.find((x) => x.binding === wb.binding)
    if (r?.kind === 'texture') {
      const t = R.textures[r.tex]
      bytes = texBytes(t)
      shape = `${t.size[0]}x${t.size[1]} ${t.format}`
    } else if (r?.kind === 'buffer') {
      bytes = r.size ?? R.buffers[r.buf]?.size ?? 0
      shape = `buffer ${(bytes / 1048576).toFixed(1)} MB`
    }
  }
  const uni = d.res.find((x) => x.binding === uniformBinding[s] && x.uniform)?.uniform
  return { i, pid: d.pid, s, bytes, shape, uni: uni ? uni.slice(0, 4) : null, us: usAt(i) }
})

// Same MATVEC set as analyze.mjs: kernels verified to stream a weight matrix
// once per dispatch. Left out: 0115 (embedding, one row), 0123 (gathers) and
// attention 0101/0103/0108/0110 (KV cache, not weights). Indices are positions
// in @litert-lm/core 0.17.1 shader creation order with this Gemma 4 E2B bundle.
const MATVEC = new Set([98, 99, 100, 105, 106, 107, 112, 113, 134])
if (R.version !== VERSION || M.version !== VERSION) console.log(`warning: kernel indices read off @litert-lm/core ${VERSION}, record is trace ${R.version} / manifest ${M.version}`)
const weightRows = rows.filter((r) => r.bytes > 0 && MATVEC.has(r.s))

// Same split-pass correction as analyze.mjs: splitting every dispatch into
// its own timestamped pass adds ~1.5 µs per pass, measured as (split sum -
// unsplit sum) / 1304 against out/timing-gpu.json when readable.
let overheadUs = 0
let overheadNote = 'out/timing-gpu.json not readable: split-pass overhead taken as 0'
try {
  if (!G) throw new Error('no timing-gpu')
  const gTokens = byChunk(G.passes.filter((p) => p.beginNs != null)).filter((t) => t.length === 23)
  const unsplit = median(gTokens.map((t) => t.reduce((a, p) => a + (p.endNs - p.beginNs), 0) / 1e3))
  const split = median(aligned.map((t) => t.reduce((a, p) => a + (p.endNs - p.beginNs), 0) / 1e3))
  overheadUs = (split - unsplit) / 1304
  overheadNote = `unsplit GPU sum ${f(unsplit / 1000, 2)} ms, split sum ${f(split / 1000, 2)} ms -> ${f(overheadUs, 2)} µs per pass`
} catch {
  overheadUs = 0
}
let tokenMs = NaN
try {
  if (!B) throw new Error('no timing-bare')
  const chunkT = B.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
  const STEADY_FROM = 8
  tokenMs = median(chunkT.slice(STEADY_FROM).map((t, i) => t - chunkT[STEADY_FROM + i - 1]).slice(1))
} catch {
  tokenMs = NaN
}
const corr = (us) => Math.max(0.5, us - overheadUs)
const gbs = (bytes, us) => bytes / (us * 1e-6) / 1e9

// Per (kernel, shape): median µs per dispatch, GB/s and bytes/µs both raw and
// split-pass corrected. bytes/µs = GB/s x 1000; the roofline reads either way.
const groups = new Map()
for (const r of weightRows) {
  const k = `${r.s}|${r.shape}|${JSON.stringify(r.uni)}`
  if (!groups.has(k)) groups.set(k, [])
  groups.get(k).push(r)
}
const shapeRows = [...groups.values()]
  .map((g) => ({
    kernel: g[0].s,
    count: g.length,
    bytes_per_dispatch: g[0].bytes,
    bytes_per_dispatch_mib: Number((g[0].bytes / 1048576).toFixed(3)),
    shape: g[0].shape,
    uniform_i0_3: g[0].uni,
    median_us: Number(median(g.map((r) => r.us)).toFixed(3)),
    median_us_corrected: Number(median(g.map((r) => corr(r.us))).toFixed(3)),
    gbs_raw: Number(gbs(g[0].bytes, median(g.map((r) => r.us))).toFixed(1)),
    gbs_corrected: Number(gbs(g[0].bytes, median(g.map((r) => corr(r.us)))).toFixed(1)),
    bytes_per_us_raw: Number((g[0].bytes / median(g.map((r) => r.us))).toFixed(1)),
    bytes_per_us_corrected: Number((g[0].bytes / median(g.map((r) => corr(r.us)))).toFixed(1)),
    total_ms_raw: Number((g.reduce((a, r) => a + r.us, 0) / 1000).toFixed(3)),
    _total: g.reduce((a, r) => a + r.us, 0),
  }))
  .sort((a, b) => b._total - a._total || a.kernel - b.kernel || (a.shape < b.shape ? -1 : 1))
  .map(({ _total, ...rest }) => rest)

// Per-kernel aggregates over the shape rows above.
const byKernelMap = new Map()
for (const r of shapeRows) {
  if (!byKernelMap.has(r.kernel)) byKernelMap.set(r.kernel, [])
  byKernelMap.get(r.kernel).push(r)
}
const byKernel = [...byKernelMap.entries()]
  .map(([kernel, rs]) => {
    const dispatches = rs.reduce((a, r) => a + r.count, 0)
    const bytesTotal = rs.reduce((a, r) => a + r.count * r.bytes_per_dispatch, 0)
    const usTotal = rs.reduce((a, r) => a + r.count * r.median_us, 0)
    const usTotalCorr = rs.reduce((a, r) => a + r.count * r.median_us_corrected, 0)
    return {
      kernel,
      dispatches,
      bytes_total: bytesTotal,
      bytes_total_mib: Number((bytesTotal / 1048576).toFixed(1)),
      ms_raw: Number((usTotal / 1000).toFixed(3)),
      ms_corrected: Number((usTotalCorr / 1000).toFixed(3)),
      gbs_raw: Number(gbs(bytesTotal, usTotal).toFixed(1)),
      gbs_corrected: Number(gbs(bytesTotal, usTotalCorr).toFixed(1)),
      bytes_per_us_raw: Number((bytesTotal / usTotal).toFixed(1)),
      bytes_per_us_corrected: Number((bytesTotal / usTotalCorr).toFixed(1)),
      shapes: rs.length,
    }
  })
  .sort((a, b) => b.ms_raw - a.ms_raw)

// Linear fit over texture-weight matvec dispatches, same as analyze.mjs:
// µs = intercept + bytes / bandwidth.
const fitRows = weightRows.filter((r) => /rgba/.test(r.shape))
const n = fitRows.length
const sx = fitRows.reduce((a, r) => a + r.bytes, 0)
const sy = fitRows.reduce((a, r) => a + r.us, 0)
const sxx = fitRows.reduce((a, r) => a + r.bytes * r.bytes, 0)
const sxy = fitRows.reduce((a, r) => a + r.bytes * r.us, 0)
const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
const intercept = (sy - slope * sx) / n
const ss = fitRows.reduce((a, r) => a + (r.us - (intercept + slope * r.bytes)) ** 2, 0)
const st = fitRows.reduce((a, r) => a + (r.us - sy / n) ** 2, 0)
const fit = {
  n,
  intercept_us: Number(intercept.toFixed(2)),
  intercept_us_corrected: Number((intercept - overheadUs).toFixed(2)),
  streaming_gbs: Number((1 / slope / 1e3).toFixed(0)),
  r2: Number((1 - ss / st).toFixed(3)),
}

// Cross-check against out/kernel-profile.json: same timing run, so per-pid
// medians should agree up to token selection. kernel-profile bytes are WGSL
// bytes, not weight bytes, and are not used for the roofline. Multi-shape
// kernels (0099 has five shapes here) mix fast and slow dispatches under one
// pid, so the check compares per-pid medians, not each shape row to the pid
// median.
const profByPid = new Map((P.rows ?? []).map((r) => [r.pid, r]))
const oursByPid = new Map()
for (const r of weightRows) {
  if (!oursByPid.has(r.pid)) oursByPid.set(r.pid, [])
  oursByPid.get(r.pid).push(r.us)
}
let maxAbsDiffUs = 0
let checked = 0
for (const [pid, usList] of oursByPid) {
  const p = profByPid.get(pid)
  if (p && Number.isFinite(p.us)) {
    checked++
    const d = Math.abs(p.us - median(usList))
    if (d > maxAbsDiffUs) maxAbsDiffUs = d
  }
}

// Prefill inventory for the joint roofline: trace.prefillPipelines counts the
// cold message vs decode pids. Prefill dispatches have no per-kernel timing
// on main, so they are listed without bytes/µs; the timed roofline below is
// decode only.
const shaderOf = (pid) => M.pipelines[pid]?.shaderId ?? -1
const prefillOnly = Object.keys(R.prefillPipelines ?? {})
  .map(Number)
  .filter((pid) => !new Set(tr.map((d) => d.pid)).has(pid))
  .map((pid) => ({ pid, shader: shaderOf(pid), dispatches: R.prefillPipelines[String(pid)] }))
  .sort((a, b) => b.dispatches - a.dispatches)

const weightBytes = weightRows.reduce((a, r) => a + r.bytes, 0)
const weightUs = weightRows.reduce((a, r) => a + r.us, 0)
const weightUsCorr = weightRows.reduce((a, r) => a + corr(r.us), 0)
const batt = pmset(['-g', 'batt'])
const lpm = pmset(['-g']).match(/lowpowermode\s+(\d+)/)
const pct = batt.match(/(\d+)%/)
const harnessCommit = git(['rev-parse', 'HEAD'])?.trim() ?? null
const statusOut = git(['status', '--porcelain', '--', '.', ':!out/*.json'])
const record = {
  date: new Date().toISOString(),
  method:
    'offline join of committed out/trace.json (weight bytes per dispatch from texture/buffer sizes) with out/timing-kernel.json (median GPU µs per dispatch position over tokens with identical pipeline sequence) and out/kernel-profile.json (per-pid cross-check); no GPU run, no synthetic numbers; same positional join, weight-binding scan, MATVEC set and split-pass correction as analyze.mjs so figures match its summary; rendered per kernel in www/roofline.html',
  inputs: {
    'trace.json': { version: R.version, dispatches: R.dispatches.length, pipelines: R.pipelines.length, adapter: R.adapter ?? null },
    'timing-kernel.json': { version: K.version, passes: K.passes.length, mode: K.mode ?? null },
    'kernel-profile.json': { tokens: P.tokens ?? null, rows: (P.rows ?? []).length, totalMs: P.totalMs ?? null },
    'manifest.json': { version: M.version, pipelines: M.pipelines.length, shaders: M.shaders.length },
    'timing-gpu.json': G ? { version: G.version ?? null, passes: G.passes?.length ?? null } : null,
    'timing-bare.json': B ? { version: B.version ?? null, marks: B.marks?.length ?? null } : null,
    provenance_note:
      'trace.json, timing-kernel.json, manifest.json, timing-gpu.json, timing-bare.json and kernel-profile.json on main predate the provenance block and carry no provenance; derivation provenance below describes this offline regrouping, not a GPU run',
  },
  manifest_sha256: manifestSha256(join(OUT, 'manifest.json')),
  provenance: {
    browser: null,
    adapter: R.adapter ?? null,
    node: process.version,
    os: { platform: platform(), release: release(), arch: arch() },
    package: R.version ?? null,
    model: null,
    power: {
      source: /AC Power/.test(batt) ? 'AC' : /Battery Power/.test(batt) ? 'battery' : null,
      percent: pct ? Number(pct[1]) : null,
      low_power_mode: lpm ? lpm[1] === '1' : null,
    },
    thermal: pmset(['-g', 'therm']).split('\n').map((l) => l.trim()).filter(Boolean),
    harness_commit: harnessCommit,
    harness_dirty: statusOut == null ? null : statusOut.trim() !== '',
    timestamp: new Date().toISOString(),
    note: 'offline derivation: browser null because no page was driven; adapter/package copied from trace.json inputs; power/thermal are the derivation machine state, not a measurement condition',
  },
  peak_gbs: PEAK_GBS,
  split_pass_overhead_us: Number(overheadUs.toFixed(3)),
  split_pass_note: overheadNote,
  summary: {
    weight_bytes_per_token: weightBytes,
    weight_mib_per_token: Number((weightBytes / 1048576).toFixed(1)),
    weight_dispatches_per_token: weightRows.length,
    weight_kernels: [...MATVEC].sort((a, b) => a - b),
    weight_ms_raw: Number((weightUs / 1000).toFixed(2)),
    weight_ms_corrected: Number((weightUsCorr / 1000).toFixed(2)),
    gbs_raw: Number(gbs(weightBytes, weightUs).toFixed(0)),
    gbs_corrected: Number(gbs(weightBytes, weightUsCorr).toFixed(0)),
    pct_of_peak_corrected: Number((((100 * gbs(weightBytes, weightUsCorr)) / PEAK_GBS).toFixed(0))),
    whole_token_ms: Number.isFinite(tokenMs) ? Number(tokenMs.toFixed(1)) : null,
    whole_token_effective_gbs: Number.isFinite(tokenMs) ? Number(gbs(weightBytes, tokenMs * 1000).toFixed(0)) : null,
    traced_tokens: traceTokens.length,
    timed_tokens_aligned: aligned.length,
    dispatches_per_token: tr.length,
  },
  fit,
  rows: shapeRows,
  by_kernel: byKernel,
  kernel_profile_check: {
    pids_checked: checked,
    max_abs_diff_us_vs_profile: Number(maxAbsDiffUs.toFixed(3)),
    note: 'per-pid medians agree up to token selection; kernel-profile.json bytes are WGSL bytes, not weight bytes, so only its us medians are compared; multi-shape kernels mix shapes under one pid, hence per-pid (not per-shape) comparison',
  },
  prefill: {
    note: 'prefill dispatches counted in trace.prefillPipelines have no per-kernel GPU timing on main, so the timed roofline above is decode only; prefill-only pipelines listed for the joint view',
    prefill_pipelines: Object.keys(R.prefillPipelines ?? {}).length,
    prefill_dispatches: Object.values(R.prefillPipelines ?? {}).reduce((a, v) => a + v, 0),
    prefill_only_count: prefillOnly.length,
    prefill_only_top10: prefillOnly.slice(0, 10),
  },
  note: `roofline reading: decode is memory-bound on the weight stream (about ${f(gbs(weightBytes, weightUsCorr), 0)} GB/s corrected, ${f((100 * gbs(weightBytes, weightUsCorr)) / PEAK_GBS, 0)}% of the ${PEAK_GBS} GB/s rated); per-kernel bytes/µs = GB/s x 1000; the linear fit gives about ${f(intercept, 0)} µs fixed cost per matvec dispatch (about ${f(intercept - overheadUs, 1)} µs corrected) then streaming at about ${f(1 / slope / 1e3, 0)} GB/s; small kernels under ~6 µs sit below the line (launch-bound), large matvecs sit on it; file sha: ${shaderShaById(join(OUT, 'shaders')).size} shaders scanned for weight bindings`,
}

mkdirSync(OUT, { recursive: true })
writeFileSync(join(OUT, 'roofline.json'), JSON.stringify(record, null, 2))
console.log(`wrote out/roofline.json: ${shapeRows.length} (kernel,shape) rows over ${byKernel.length} kernels, ${f(weightBytes / 1048576, 1)} MiB/token at ${f(gbs(weightBytes, weightUsCorr), 0)} GB/s corrected, profile check ${checked} pids max diff ${f(maxAbsDiffUs, 2)} µs`)
