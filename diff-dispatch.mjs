// Diff per-token dispatch lists between two trace runs (ROADMAP point 3).
// Usage: node diff-dispatch.mjs <old-trace.json> <new-trace.json>
// Prints added/removed/renamed kernels by pipeline index + shape + bytes.
// A dispatch key is pid | shader WGSL bytes | resource shape | workgroup
// counts. Uniform contents are left out of the key on purpose: they can
// carry tensor data, not just shapes, and would differ run to run. Exit 0
// on a completed diff (even when the lists differ), 1 on usage/read errors.
import { readFileSync } from 'node:fs'

const [oldPath, newPath] = process.argv.slice(2)
if (!oldPath || !newPath) {
  console.error('usage: node diff-dispatch.mjs <old-trace.json> <new-trace.json>')
  process.exit(1)
}

const load = (path) => {
  try {
    return JSON.parse(readFileSync(path, 'utf8'))
  } catch (e) {
    console.error(`cannot read ${path}: ${e.message}`)
    process.exit(1)
  }
}

// One decode token per trace: the modal chunk length wins, ties broken the
// way trace.mjs reads them (second chunk, else first).
const tokenOf = (R) => {
  const byChunk = new Map()
  for (const d of R.dispatches ?? []) {
    if (!byChunk.has(d.chunk)) byChunk.set(d.chunk, [])
    byChunk.get(d.chunk).push(d)
  }
  const chunks = [...byChunk.values()]
  if (!chunks.length) return { chunk: null, token: [] }
  const counts = new Map()
  for (const c of chunks) counts.set(c.length, (counts.get(c.length) ?? 0) + 1)
  const mode = [...counts.entries()].sort((a, b) => b[1] - a[1] || b[0] - a[0])[0][0]
  const sized = chunks.filter((c) => c.length === mode)
  const token = sized[1] ?? sized[0]
  return { chunk: token[0].chunk, token }
}

const shapeOf = (R, d) => {
  const tex = (d.res ?? []).find((r) => r.kind === 'texture')
  if (tex) {
    const t = R.textures[tex.tex]
    if (!t) return `texture#${tex.tex}`
    const depth = (t.size[2] ?? 1) > 1 ? `x${t.size[2]}` : ''
    return `${t.size[0]}x${t.size[1]}${depth} ${t.format}`
  }
  const bufs = (d.res ?? []).filter((r) => r.kind === 'buffer')
  if (bufs.length) {
    const size = (r) => r.size ?? R.buffers[r.buf]?.size ?? 0
    const bytes = size(bufs.reduce((a, r) => (size(r) > size(a) ? r : a)))
    return `buffer ${(bytes / 1048576).toFixed(1)} MB`
  }
  return 'no-resource'
}

const keysOf = (R, token) => {
  const bytesById = new Map((R.shaders ?? []).map((s) => [s.id, s.bytes]))
  const pipeToShader = new Map((R.pipelines ?? []).map((p) => [p.id, p.shaderId]))
  return token.map((d) => {
    const shaderId = pipeToShader.get(d.pid) ?? -1
    const bytes = bytesById.get(shaderId) ?? -1
    const shape = shapeOf(R, d)
    const wg = (d.wg ?? []).join('x')
    return { pid: d.pid, shaderId, bytes, shape, wg, key: `pid ${d.pid} | shader ${shaderId} ${bytes}B | ${shape} | wg ${wg}` }
  })
}

const countBy = (keys) => {
  const m = new Map()
  for (const k of keys) {
    if (!m.has(k.key)) m.set(k.key, { fields: k, n: 0 })
    m.get(k.key).n++
  }
  return m
}

const O = load(oldPath)
const N = load(newPath)
const ot = tokenOf(O)
const nt = tokenOf(N)
if (!ot.token.length || !nt.token.length) {
  console.error('no decode token found in one of the traces')
  process.exit(1)
}
console.log(`old: ${oldPath} (chunk ${ot.chunk}, ${ot.token.length} dispatches)`)
console.log(`new: ${newPath} (chunk ${nt.chunk}, ${nt.token.length} dispatches)`)

const ok = countBy(keysOf(O, ot.token))
const nk = countBy(keysOf(N, nt.token))
const removed = new Map()
const added = new Map()
for (const [key, { fields, n }] of ok) {
  const d = n - (nk.get(key)?.n ?? 0)
  if (d > 0) removed.set(key, { fields, n: d })
}
for (const [key, { fields, n }] of nk) {
  const d = n - (ok.get(key)?.n ?? 0)
  if (d > 0) added.set(key, { fields, n: d })
}

// Renamed: pair a removed key with an added key that shares everything but
// the pipeline index (moved) or everything but the byte count (edited).
const renamed = []
const pairBy = (restFn, label) => {
  const rs = new Map()
  for (const [key, e] of removed) {
    const k = restFn(e.fields)
    if (!rs.has(k)) rs.set(k, [])
    rs.get(k).push(key)
  }
  for (const [key, e] of [...added]) {
    const k = restFn(e.fields)
    const match = (rs.get(k) ?? []).find((rk) => removed.has(rk))
    if (!match) continue
    const r = removed.get(match)
    const n = Math.min(r.n, e.n)
    renamed.push({ label, n, from: r.fields, to: e.fields })
    r.n -= n
    e.n -= n
    if (r.n === 0) removed.delete(match)
    if (e.n === 0) added.delete(key)
  }
}
pairBy((f) => `shader ${f.shaderId} ${f.bytes}B | ${f.shape} | wg ${f.wg}`, 'moved')
pairBy((f) => `pid ${f.pid} | ${f.shape} | wg ${f.wg}`, 'edited')

const show = (title, m) => {
  const total = [...m.values()].reduce((a, e) => a + e.n, 0)
  console.log(`${title} (${m.size} keys, ${total} dispatches):`)
  for (const [, e] of [...m.entries()].sort((a, b) => b[1].n - a[1].n)) console.log(`  ${String(e.n).padStart(4)}x ${e.fields.key}`)
}
show('removed', removed)
show('added', added)
console.log(`renamed (${renamed.length} pairings):`)
for (const r of renamed) {
  const detail = r.label === 'moved'
    ? `${r.from.key}  ->  pid ${r.to.pid}`
    : `pid ${r.from.pid} | ${r.from.shape} | wg ${r.from.wg}: ${r.from.bytes}B  ->  ${r.to.bytes}B`
  console.log(`  ${String(r.n).padStart(4)}x ${r.label}: ${detail}`)
}

// Order check over the shared prefix: same multiset in a different order
// still reads as a reorder, not as identical.
const oldKeys = keysOf(O, ot.token).map((k) => k.key)
const newKeys = keysOf(N, nt.token).map((k) => k.key)
const shared = Math.min(oldKeys.length, newKeys.length)
let diffPos = 0
let first = -1
for (let i = 0; i < shared; i++) {
  if (oldKeys[i] !== newKeys[i]) {
    if (first < 0) first = i
    diffPos++
  }
}
console.log(`order: ${diffPos} of ${shared} positions differ${diffPos ? ` (first at ${first}: old ${oldKeys[first]} vs new ${newKeys[first]})` : ''}`)
if (!removed.size && !added.size && !renamed.length && !diffPos && ot.token.length === nt.token.length) {
  console.log('identical: no added/removed/renamed kernels, same order')
}
