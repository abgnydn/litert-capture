// Offline analysis joining out/trace.json (resources and shapes per dispatch),
// out/timing-kernel.json (GPU time per dispatch) and out/shaders/*.wgsl.
// Dispatch order within a token is deterministic, so the two runs are joined
// by position after checking that their pipeline sequences are identical.
//
// ROADMAP 29: summarize top kernels with analyze.mjs 400-char text summary
// (`package.json:13`: `node analyze.mjs 400`). The 400 is the rated memory
// bandwidth in GB/s for this machine (PEAK_GBS, argv[2] ?? 400), used for the
// "% of rated" figures; text summary sections are weights per token, achieved
// GB/s, weight kernels by shape sorted by total (top kernels), linear fit,
// and layers. Documented from existing practice; default unchanged.
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { shaderShaById } from './shader-sha.mjs'

const OUT = new URL('./out/', import.meta.url).pathname
const R = JSON.parse(readFileSync(join(OUT, 'trace.json'), 'utf8'))
const K = JSON.parse(readFileSync(join(OUT, 'timing-kernel.json'), 'utf8'))
const M = JSON.parse(readFileSync(join(OUT, 'manifest.json'), 'utf8'))
const optional = (name) => { try { return JSON.parse(readFileSync(join(OUT, name), 'utf8')) } catch { return null } }
const G = optional('timing-gpu.json')
const B = optional('timing-bare.json')
const PEAK_GBS = Number(process.argv[2] ?? 400) // rated memory bandwidth of this machine
const VERSION = process.env.LITERT_VERSION ?? '0.17.1'

// What produced each record this join reads. Records written before the field
// existed say so instead of being passed off as belonging to this machine.
const READ = [['trace.json', R], ['timing-kernel.json', K], ['manifest.json', M], ['timing-gpu.json', G], ['timing-bare.json', B]].filter(([, rec]) => rec)
for (const [name, rec] of READ) {
  const p = rec.provenance
  if (!p) continue
  const a = p.adapter ?? {}
  console.log(`${name} provenance: ${p.browser ?? '?'}, ${a.vendor ?? '?'}/${a.architecture ?? '?'}, @litert-lm/core ${p.package ?? 'n/a'}, node ${p.node ?? '?'} ${p.os?.platform ?? '?'}-${p.os?.arch ?? '?'}, model ${p.model ? `${p.model.sha256.slice(0, 12)}…` : 'none'}, harness ${p.harness_commit ? p.harness_commit.slice(0, 8) : 'unknown'}, ${p.timestamp ?? '?'}`)
  const pw = p.power
  console.log(`  power: ${pw ? `${pw.source ?? '?'}${pw.percent == null ? '' : ` ${pw.percent}%`}, low power mode ${pw.low_power_mode == null ? '?' : pw.low_power_mode ? 'on' : 'off'}` : 'not recorded'}; thermal: ${p.thermal?.length ? p.thermal.join(' | ') : 'not recorded'}`)
}
const noProv = READ.filter(([, rec]) => !rec.provenance).map(([name]) => name)
if (noProv.length) console.log(`provenance: not recorded in ${noProv.join(', ')}`)

const BPT = { rgba8uint: 4, rgba16float: 8, rgba16uint: 8, rgba32uint: 16, rgba32float: 16, rgba32sint: 16 }
const texBytes = (t) => t.size[0] * t.size[1] * (t.size[2] ?? 1) * (BPT[t.format] ?? 0)
//
// ROADMAP 31: use MiB vs MB consistently and report effective GB/s for each
// kernel. MB() divides by 1048576 so its "MB" values are MiB (2^20 bytes);
// GB/s divides by 1e9 (decimal); effective means weight bytes read divided
// by kernel time, not DRAM traffic (a cache-served byte counts the same).
// Documented from existing practice; default unchanged.
const MB = (b) => (b / 1048576).toFixed(1)
const f = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN }
const sid = (pid) => M.pipelines[pid]?.shaderId ?? -1

// The trace run and the capture run are joined by pipeline id, which is creation
// order in both. Check the two runs really built the same pipelines by comparing
// the shader each was built from: by SHA-256 when both records carry it, else by
// WGSL byte length, which is not an identity (16 pairs of distinct shaders in
// this capture share a length, 0099/0100 among them) and so is only a weak
// check. The capture side is hashed from out/shaders/ when the record predates
// the field; a trace record that predates it has no source to hash.
const fileSha = shaderShaById(join(OUT, 'shaders'))
const capSha = (id) => M.shaders[id]?.sha256 ?? fileSha.get(id) ?? null
const joinByHash = R.shaders.every((s) => typeof s.sha256 === 'string') && M.shaders.every((s) => capSha(s.id))
const joinOk = M.pipelines.length === R.pipelines.length && R.pipelines.every((p, i) => (joinByHash
  ? R.shaders[p.shaderId]?.sha256 === capSha(M.pipelines[i]?.shaderId)
  : R.shaders[p.shaderId]?.bytes === M.shaders[M.pipelines[i]?.shaderId]?.bytes))
console.log(`trace/capture pipeline join: ${R.pipelines.length} pipelines, shaders agree: ${joinOk} (by ${joinByHash ? 'SHA-256' : 'WGSL byte length: out/trace.json carries no shader hash, and byte length is not an identity'})`)

// static: which binding holds weights, and which the uniform block, in each shader.
// Shaders 0101, 0103, 0108 and 0110 are attention kernels reading the 8-bit KV
// cache through a binding also named weights_buffer (0108/0110 take the sequence
// length from params_buffer; 0101/0103 are bounded by fixed uniforms); they are
// not weight streams and are left out of weightRows.
const weightBinding = {}
const uniformBinding = {}
const ATTENTION = new Set()
for (const file of readdirSync(join(OUT, 'shaders'))) {
  const id = Number(file.slice(0, 4))
  const src = readFileSync(join(OUT, 'shaders', file), 'utf8')
  const m = [...src.matchAll(/@binding\((\d+)\) var(?:<[^>]*>)? (\w*weights\w*)\s*:/g)].find((x) => !/scale|zero_point/.test(x[2]))
  if (m) weightBinding[id] = { binding: Number(m[1]), name: m[2] }
  const u = src.match(/@binding\((\d+)\) var<uniform>/)
  if (u) uniformBinding[id] = Number(u[1])
  if (/params_buffer\.data\[/.test(src)) ATTENTION.add(id)
}

// per-token dispatch lists from both runs
const byChunk = (arr) => { const m = new Map(); for (const d of arr) { if (!m.has(d.chunk)) m.set(d.chunk, []); m.get(d.chunk).push(d) } return [...m.values()] }
const traceTokens = byChunk(R.dispatches).filter((t) => t.length === 1304)
const timeTokens = byChunk(K.passes.filter((p) => p.beginNs != null)).filter((t) => t.length === 1304)
const tr = traceTokens[1] ?? traceTokens[0]
const pidSeq = tr.map((d) => d.pid).join(',')
const aligned = timeTokens.filter((t) => t.map((p) => p.pid).join(',') === pidSeq)
console.log(`tokens: ${traceTokens.length} traced, ${timeTokens.length} timed, ${aligned.length} with identical pipeline sequence (joined by position)`)
const usAt = (i) => median(aligned.map((t) => (t[i].endNs - t[i].beginNs) / 1e3))

// per dispatch: bytes of weights, time
const rows = tr.map((d, i) => {
  const s = sid(d.pid)
  const wb = weightBinding[s]
  let bytes = 0, shape = ''
  if (wb) {
    const r = d.res.find((x) => x.binding === wb.binding)
    if (r?.kind === 'texture') { const t = R.textures[r.tex]; bytes = texBytes(t); shape = `${t.size[0]}x${t.size[1]} ${t.format}` }
    else if (r?.kind === 'buffer') { bytes = r.size ?? R.buffers[r.buf]?.size ?? 0; shape = `buffer ${MB(bytes)} MB` }
  }
  const uni = d.res.find((x) => x.binding === uniformBinding[s] && x.uniform)?.uniform
  return { i, s, bytes, shape, uni: uni ? uni.slice(0, 4) : null, us: usAt(i) }
})
const totalUs = rows.reduce((a, r) => a + r.us, 0)
// Kernels verified (by reading their WGSL and the trace) to stream a weight
// matrix once per dispatch. Left out: 0115 (embedding table, one row read),
// 0123 (per-layer gathers), and the attention kernels 0101/0103 (local, over a
// 131 KB 8-bit KV window) and 0108/0110 (global, loop bound from params_buffer)
// whose binding is also called weights_buffer but holds the KV cache.
// These kernel indices were read off @litert-lm/core VERSION with the Gemma 4
// E2B web bundle; they are positions in that build's shader creation order and
// will differ for any other package version or model bundle.
const MATVEC = new Set([98, 99, 100, 105, 106, 107, 112, 113, 134])
if (R.version !== VERSION || M.version !== VERSION) console.log(`warning: kernel indices read off @litert-lm/core ${VERSION}, record is trace ${R.version} / manifest ${M.version}`)
const weightRows = rows.filter((r) => r.bytes > 0 && MATVEC.has(r.s))
console.log(`weight kernels counted: ${[...MATVEC].map((s) => String(s).padStart(4, '0')).join(' ')}; attention kernels seen (params_buffer bound): ${[...ATTENTION].filter((s) => [101, 103, 108, 110].includes(s)).map((s) => String(s).padStart(4, '0')).join(' ')} plus 0101 0103 by structure`)

// Splitting every dispatch into its own pass adds a per-pass cost. Measure it
// as (split per-token sum - unsplit per-token sum) / 1304 and report corrected
// figures alongside the raw ones.
//
// ROADMAP 30: apply split-pass correction (~1.5us/pass) to K-split timings.
// Measured about 1.52 us per pass here, subtracted via corr() for corrected
// figures alongside the raw ones. Documented from existing practice; default
// unchanged.
let overheadUs = 0
try {
  if (!G) throw new Error('out/timing-gpu.json not readable')
  const gTokens = byChunk(G.passes.filter((p) => p.beginNs != null)).filter((t) => t.length === 23)
  const unsplit = median(gTokens.map((t) => t.reduce((a, p) => a + (p.endNs - p.beginNs), 0) / 1e3))
  const split = median(aligned.map((t) => t.reduce((a, p) => a + (p.endNs - p.beginNs), 0) / 1e3))
  overheadUs = (split - unsplit) / 1304
  console.log(`split-pass overhead: unsplit GPU sum ${f(unsplit / 1000, 2)} ms, split sum ${f(split / 1000, 2)} ms -> about ${f(overheadUs, 2)} µs per pass; corrected figures below subtract it`)
} catch {
  console.log('out/timing-gpu.json not readable: split-pass overhead taken as 0, so "corrected" figures below equal the raw ones')
}
// wall-clock ms per token: median warm steady-state chunk gap, as timing.mjs computes it
let tokenMs = NaN
try {
  if (!B) throw new Error('out/timing-bare.json not readable')
  const chunkT = B.marks.filter((m) => m.name.startsWith('warm:chunk-')).map((m) => m.t)
  const STEADY_FROM = 8
  tokenMs = median(chunkT.slice(STEADY_FROM).map((t, i) => t - chunkT[STEADY_FROM + i - 1]).slice(1))
} catch {
  console.log('out/timing-bare.json not readable: whole-token figures below are n/a')
}
const corr = (us) => Math.max(0.5, us - overheadUs)
const weightBytes = weightRows.reduce((a, r) => a + r.bytes, 0)
const weightUs = weightRows.reduce((a, r) => a + r.us, 0)

const weightUsCorr = weightRows.reduce((a, r) => a + corr(r.us), 0)
console.log(`\n== weights per token: ${MB(weightBytes)} MB read by ${weightRows.length} dispatches over ${new Set(weightRows.map((r) => r.s)).size} kernels, ${f(weightUs / 1000, 2)} ms of GPU time (split-pass), ${f(weightUsCorr / 1000, 2)} ms corrected`)
console.log(`achieved on weight kernels: ${f(weightBytes / (weightUs * 1e-6) / 1e9, 0)} GB/s raw, ${f(weightBytes / (weightUsCorr * 1e-6) / 1e9, 0)} GB/s corrected = ${f((100 * weightBytes / (weightUsCorr * 1e-6) / 1e9) / PEAK_GBS, 0)}% of the ${PEAK_GBS} GB/s rated`)
const small = rows.filter((r) => r.us < 6)
console.log(`kernels under 6 µs: ${new Set(small.map((r) => r.s)).size} kernels, ${small.length} dispatches, ${f((100 * small.reduce((a, r) => a + r.us, 0)) / totalUs, 1)}% of GPU time raw, ${f((100 * small.reduce((a, r) => a + corr(r.us), 0)) / rows.reduce((a, r) => a + corr(r.us), 0), 1)}% corrected`)
const fq = rows.filter((r) => r.s === 119)
console.log(`fake-quant 0119: ${fq.length} dispatches, ${f(fq.reduce((a, r) => a + r.us, 0) / 1000, 2)} ms raw, ${f(fq.reduce((a, r) => a + corr(r.us), 0) / 1000, 2)} ms corrected per token`)
console.log(`whole token: ${MB(weightBytes)} MB in ${f(tokenMs, 1)} ms = ${f(weightBytes / (tokenMs * 1e-3) / 1e9, 0)} GB/s effective; at ${PEAK_GBS} GB/s the weights alone would take ${f(weightBytes / (PEAK_GBS * 1e9) * 1e3, 2)} ms`)
console.log(`non-weight kernels: ${f((totalUs - weightUs) / 1000, 2)} ms of ${f(totalUs / 1000, 2)} ms (split-pass measurement, ~12% inflated)`)

// per (kernel, shape)
console.log(`\n== weight kernels by shape (median GPU µs per dispatch, GB/s)`)
console.log(`  kernel  count  weights/dispatch          shape                       U.i0             µs    GB/s`)
const groups = new Map()
for (const r of weightRows) { const k = `${r.s}|${r.shape}|${JSON.stringify(r.uni)}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r) }
const shapeRows = [...groups.values()].map((g) => ({ s: g[0].s, n: g.length, bytes: g[0].bytes, shape: g[0].shape, uni: g[0].uni, us: median(g.map((r) => r.us)), total: g.reduce((a, r) => a + r.us, 0) }))
  .sort((a, b) => b.total - a.total)
for (const r of shapeRows) console.log(`  ${String(r.s).padStart(4, '0')}    ${String(r.n).padStart(3)}   ${(MB(r.bytes) + ' MB').padStart(10)}   ${r.shape.padEnd(26)} ${JSON.stringify(r.uni).padEnd(18)} ${f(r.us, 1).padStart(6)}  ${f(r.bytes / (r.us * 1e-6) / 1e9, 0).padStart(5)}`)

// fit: us = a + bytes / BW, over texture-weight matvec dispatches only
const fitRows = weightRows.filter((r) => /rgba/.test(r.shape))
const n = fitRows.length, sx = fitRows.reduce((a, r) => a + r.bytes, 0), sy = fitRows.reduce((a, r) => a + r.us, 0)
const sxx = fitRows.reduce((a, r) => a + r.bytes * r.bytes, 0), sxy = fitRows.reduce((a, r) => a + r.bytes * r.us, 0)
const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx), intercept = (sy - slope * sx) / n
const ss = fitRows.reduce((a, r) => a + (r.us - (intercept + slope * r.bytes)) ** 2, 0), st = fitRows.reduce((a, r) => a + (r.us - sy / n) ** 2, 0)
console.log(`\n== linear fit over ${n} texture-weight dispatches: µs = ${f(intercept, 1)} + bytes / ${f(1 / slope / 1e3, 0)} GB/s  (R² ${f(1 - ss / st, 2)})`)
console.log(`   read: about ${f(intercept, 0)} µs fixed cost per matvec kernel (about ${f(intercept - overheadUs, 1)} µs after the split-pass overhead), then streaming at about ${f(1 / slope / 1e3, 0)} GB/s`)

// layers: segment the tape at FFN-down dispatches (input 3072 or 1536/1024 slices back to 384)
// FFN down-projection: 6144 -> 1536 (0099) or 12288 -> 1536 (0113). The
// attention output projections (2048 or 4096 -> 1536) are not cuts.
//
// ROADMAP 32: map dispatches to 35 layers via layer map. One FFN-down cut
// per layer (uni[0] 384, uni[1] 1536/3072, bytes > 0), 35 cuts here;
// prologue folds into layer 1, tail after last cut is epilogue.
// Documented from existing practice; default unchanged.
const isFfnDown = (r) => r.uni && r.uni[0] === 384 && (r.uni[1] === 1536 || r.uni[1] === 3072) && r.bytes > 0
const cuts = rows.filter(isFfnDown).map((r) => r.i)
console.log(`\n== layers: ${cuts.length} FFN-down dispatches per token -> ${cuts.length} layers`)
const segs = []
let prev = -1
for (const c of cuts) { segs.push(rows.slice(prev + 1, c + 1)); prev = c }
const tail = rows.slice(prev + 1)
console.log(`prologue (before layer 1): ${segs[0].length - (segs[1]?.length ?? 0) > 0 ? 'folded into layer 1 below' : 'none'}; epilogue after last layer: ${tail.length} dispatches, ${f(tail.reduce((a, r) => a + r.us, 0) / 1000, 2)} ms (final norm, vocab projection ${MB(tail.find((r) => r.bytes > 50e6)?.bytes ?? 0)} MB, sampling)`)
const sig = (seg) => seg.filter((r) => r.bytes > 0).map((r) => `${String(r.s).padStart(4, '0')}:${r.uni?.[0]}x${r.uni?.[1]}`).join(' ')
const kinds = new Map()
segs.forEach((seg, L) => { const k = sig(seg); if (!kinds.has(k)) kinds.set(k, { layers: [], us: [] }); kinds.get(k).layers.push(L + 1); kinds.get(k).us.push(seg.reduce((a, r) => a + r.us, 0)) })
console.log(`layer types by weight-kernel signature (kernel:out_slices x in_slices, slices of 4):`)
for (const [k, v] of [...kinds.entries()].sort((a, b) => a[1].layers[0] - b[1].layers[0])) {
  console.log(`  layers ${v.layers.join(',')}\n    ${k}\n    dispatches ${segs[v.layers[0] - 1].length}, ${f(median(v.us) / 1000, 2)} ms each`)
}
console.log(`\nd_model = ${384 * 4} (384 slices of 4). dims seen: ${[...new Set(weightRows.flatMap((r) => r.uni ? [r.uni[0] * 4, r.uni[1] * 4] : []))].sort((a, b) => a - b).join(', ')}`)
