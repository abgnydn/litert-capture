// The one place that says what produced a record: browser build, WebGPU
// adapter, node, OS, the @litert-lm/core version the page loaded, the local
// model bundle, the power and thermal state of the machine, the harness commit
// and whether that tree was dirty, and the time. Imported by every driver that
// writes under out/.
//
// The model bundle is 2 GB, so its SHA-256 is cached in out/model.json keyed by
// file, size and mtime; a changed file re-hashes. A missing model (microbench
// does not need one) is recorded as null, not an error.
//
// ROADMAP 35: stamp browser/adapter/node/os/pkg/model/power/thermal/commit
// via provenance.mjs. Fields already stamped by provenance(): browser,
// adapter, node, os, package, model, power, thermal, harness_commit,
// harness_dirty, timestamp. Made explicit here; default unchanged.

import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createReadStream, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { arch, platform, release } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out')
// LATEST_TESTED: highest Chrome for Testing major verified so far (154, maintainer repro on M3); update on each G0 latest-Chrome check.
const LATEST_TESTED = 154
const MODEL_FILE = 'www/gemma-4-E2B-it-web.litertlm'
const MODEL = join(HERE, MODEL_FILE)
const CACHE = join(OUT, 'model.json')

const modelInfo = async () => {
  let st
  try { st = statSync(MODEL) } catch { return null }
  const key = { file: MODEL_FILE, bytes: st.size, mtimeMs: st.mtimeMs }
  try {
    const c = JSON.parse(readFileSync(CACHE, 'utf8'))
    if (c.file === key.file && c.bytes === key.bytes && c.mtimeMs === key.mtimeMs) {
      return { file: c.file, bytes: c.bytes, sha256: c.sha256 }
    }
  } catch {}
  const h = createHash('sha256')
  for await (const chunk of createReadStream(MODEL)) h.update(chunk)
  const sha256 = h.digest('hex')
  mkdirSync(OUT, { recursive: true })
  writeFileSync(CACHE, JSON.stringify({ ...key, sha256 }, null, 2))
  return { file: key.file, bytes: key.bytes, sha256 }
}

// Power and thermal state when the record was written. A run on battery, in low
// power mode, or under a CPU speed limit is not comparable with one on mains, so
// the record carries the state rather than leaving it to prose. `pmset -g therm`
// prints only "Note: ..." lines when nothing has been recorded.
const pmset = (args) => {
  try {
    return execFileSync('pmset', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  } catch {
    return ''
  }
}

const powerState = () => {
  const batt = pmset(['-g', 'batt'])
  const pct = batt.match(/(\d+)%/)
  const lpm = pmset(['-g']).match(/lowpowermode\s+(\d+)/)
  return {
    source: /AC Power/.test(batt) ? 'AC' : /Battery Power/.test(batt) ? 'battery' : null,
    percent: pct ? Number(pct[1]) : null,
    low_power_mode: lpm ? lpm[1] === '1' : null,
  }
}

const thermalState = () => pmset(['-g', 'therm']).split('\n').map((l) => l.trim()).filter(Boolean)

// HEAD is the commit the working tree was based on when the record was written,
// which is the parent of the commit that ends up containing the record: a record
// is written before it is committed. So HEAD alone does not say what ran.
// harness_dirty carries the rest of it: whether that tree had uncommitted
// changes outside the JSON records under out/ (records and the model-hash cache
// are written there during a run; the captured shaders under out/shaders/ are
// inputs and still count), so a record written from an edited harness says so.
const git = (args) => {
  try {
    return execFileSync('git', args, { cwd: HERE, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  } catch {
    return null
  }
}

const harnessCommit = () => git(['rev-parse', 'HEAD'])?.trim() ?? null

const harnessDirty = () => {
  const s = git(['status', '--porcelain', '--', '.', ':!out/*.json'])
  return s == null ? null : s.trim() !== ''
}
//
// ROADMAP 39: mark harness_dirty when working tree differs from committed
// harness. harnessDirty() already reports uncommitted changes outside the
// JSON records under out/; made explicit here. Default unchanged.

export const provenance = async ({ browser, adapter, pkg }) => {
  const version = await browser.version()
  const major = Number(/Chrome\/(\d+)/.exec(version)?.[1])
  if (Number.isFinite(major) && major < LATEST_TESTED) {
    console.warn(`provenance: browser ${version} is behind latest tested Chrome ${LATEST_TESTED}`)
  }
  //
  // ROADMAP 36: refuse records missing browser version, adapter, or harness
  // commit. provenance() warns when the browser is behind LATEST_TESTED;
  // refusing a record with a missing field is open work -- fields stay
  // nullable here so existing writers keep writing. Default unchanged.
  return {
    browser: version,
  adapter: adapter ?? null,
  node: process.version,
  os: { platform: platform(), release: release(), arch: arch() },
  package: pkg ?? null,
  model: await modelInfo(),
  power: powerState(),
  thermal: thermalState(),
  harness_commit: harnessCommit(),
  harness_dirty: harnessDirty(),
  timestamp: new Date().toISOString(),
  }
}
