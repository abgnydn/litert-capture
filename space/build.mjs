// Wraps www/kernel-0112.html (an HTML fragment that begins with <title>)
// into a complete document for
// the static Hugging Face Space. Run: node space/build.mjs
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const fragment = readFileSync(join(HERE, '..', 'www', 'kernel-0112.html'), 'utf8')
const doc = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Runs one of LiteRT-LM's WebGPU matrix-vector kernels on your GPU against a control and two deeper work splits, and explains what it measures and how.">
${fragment.replace(/^<title>/, '<title>').replace(/<meta charset="utf-8">\n/, '')}
</html>
`
writeFileSync(join(HERE, 'index.html'), doc)
console.log(`wrote space/index.html (${Buffer.byteLength(doc)} bytes)`)
