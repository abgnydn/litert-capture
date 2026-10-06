// Backfill a provenance index without touching the frozen records.
//
// Scans out/*.json (excluding the generated out/index.json itself). For each
// record it emits { file, sha256, has_provenance, browser, adapter,
// harness_commit, date } with nulls where a field is absent and no invented
// values (no browser inferred from the adapter string, no date parsed from
// the filename, no manifest hash synthesized). Writes only out/index.json;
// the BENCH.md appendix lines (cold median per variant plus provenance plus
// manifest_sha256 for each dated microbench record) go to stdout only.
//
// Field sources follow provenance.mjs:100-123 (provenance() shape: browser,
// adapter, harness_commit live under top-level rec.provenance; per-run blocks
// under rec.runs[].provenance are not lifted to the top level -- nulls stay
// nulls, never invented), the dated-write pattern in
// microbench.mjs:155-164 (writeFileSync(join(OUT, name),
// JSON.stringify(R, null, 2))), and the READ join plus noProv warning first
// in analyze.mjs:31-43.
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, '..', 'out')

const provenanceOf = (rec) => {
  if (rec !== null && typeof rec === 'object' && rec.provenance !== null && typeof rec.provenance === 'object') {
    return rec.provenance
  }
  return null
}

const adapterString = (adapter) => {
  if (adapter === null || adapter === undefined) {
    return 'null'
  }
  if (typeof adapter === 'object' && adapter.vendor !== undefined && adapter.architecture !== undefined) {
    return `${adapter.vendor ?? '?'}/${adapter.architecture ?? '?'}`
  }
  return JSON.stringify(adapter)
}

const files = readdirSync(OUT)
  .filter((f) => f.endsWith('.json'))
  .filter((f) => f !== 'index.json')
  .sort()

const index = []
const byFile = new Map()
for (const file of files) {
  const raw = readFileSync(join(OUT, file))
  const sha256 = createHash('sha256').update(raw).digest('hex')
  let rec = null
  try {
    rec = JSON.parse(raw.toString('utf8'))
  } catch {
    rec = null
  }
  const prov = rec === null ? null : provenanceOf(rec)
  const entry = {
    file,
    sha256,
    has_provenance: prov !== null,
    browser: prov?.browser ?? null,
    adapter: prov?.adapter ?? rec?.adapter ?? null,
    harness_commit: prov?.harness_commit ?? null,
    date: rec !== null && typeof rec.date === 'string' ? rec.date : null,
  }
  index.push(entry)
  byFile.set(file, rec)
}

writeFileSync(join(OUT, 'index.json'), JSON.stringify(index, null, 2))

const noProv = index.filter((e) => !e.has_provenance).map((e) => e.file)
if (noProv.length > 0) {
  console.log(`provenance: not recorded in ${noProv.join(', ')}`)
}

const datedMicrobench = files.filter((file) => {
  if (!file.startsWith('microbench-')) {
    return false
  }
  const rec = byFile.get(file)
  return rec !== null && typeof rec === 'object' && typeof rec.date === 'string' && rec.variants !== null && typeof rec.variants === 'object'
})

for (const file of datedMicrobench) {
  const rec = byFile.get(file)
  const prov = provenanceOf(rec)
  const medians = Object.entries(rec.variants)
    .map(([name, v]) => {
      const median = v !== null && typeof v === 'object' && v.cold !== null && typeof v.cold === 'object' ? v.cold.median : null
      return `${name}=${median === null || median === undefined ? 'null' : median}`
    })
    .join(' ')
  const kernel = typeof rec.kernel === 'string' ? rec.kernel : 'unknown'
  const manifest = rec.manifest_sha256 ?? null
  console.log(`- ${file} kernel=${kernel} date=${rec.date} cold_median_us ${medians} | provenance browser=${prov?.browser ?? 'null'} adapter=${adapterString(prov?.adapter ?? rec.adapter ?? null)} harness_commit=${prov?.harness_commit ?? 'null'} | manifest_sha256=${manifest ?? 'null'}`)
}
