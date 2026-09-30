// SHA-256 of each captured shader, by shader id, read from out/shaders/NNNN_*.wgsl.
// Cross-run joins match a pipeline to the shader it was built from. Byte length
// is not an identity for that: in this capture 16 pairs of distinct shaders
// share a length (0099/0100 and 0106/0107 among them). Records written before
// the hash was recorded carry only lengths, but the shader files they were
// written from are still on disk, so the capture side of a join can always be
// hashed even when the record cannot.

import { createHash } from 'node:crypto'
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

export const shaderShaById = (shaderDir) => {
  const byId = new Map()
  let files
  try { files = readdirSync(shaderDir) } catch { return byId }
  for (const f of files) {
    if (!f.endsWith('.wgsl')) continue
    const id = Number(f.slice(0, 4))
    if (!Number.isInteger(id)) continue
    byId.set(id, createHash('sha256').update(readFileSync(join(shaderDir, f))).digest('hex'))
  }
  return byId
}

// ROADMAP 08: SHA-256 of out/manifest.json, stored alongside out/*.json
// records so a record joins unambiguously to the capture it was measured
// against. Null when the manifest is unreadable; records written before the
// field existed read back as null rather than a made-up hash.
export const manifestSha256 = (manifestPath) => {
  try { return createHash('sha256').update(readFileSync(manifestPath)).digest('hex') } catch { return null }
}
