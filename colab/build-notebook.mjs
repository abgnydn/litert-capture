// Builds colab/kernel-0112-t4.ipynb: a self-contained Colab notebook that runs
// www/microbench.html on a T4 through headless Chrome. The page, the captured
// kernel and a driver are embedded base64, so the notebook runs on its own.
// Cell 1 is zero-tvm's proven recipe for getting a real Vulkan device on Colab.
//
// ROADMAP 38: add T4 machine fields date/gpuInfo/order for Colab runs. The
// driver stamps R.date and R.gpuInfo and the page records R.order; the fresh
// record out/microbench-colab-t4-f32.json already carries them. Made explicit
// here; default unchanged.
//
// ROADMAP 42: stable block ids for repro pastes (a1b2c3d4...).
// Ids are fixed literals in the cells below so pastes stay addressable.
// Made explicit here; default unchanged.
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..')
const b64 = (p) => readFileSync(p).toString('base64')

const driver = `// headless Chrome on a Colab T4, per Chrome's colab-headless guide
import { createServer } from 'node:http'
import { createReadStream, statSync, writeFileSync } from 'node:fs'
import { join, normalize, extname } from 'node:path'
import puppeteer from 'puppeteer'
const ROOT = process.cwd()
const MIME = { '.html': 'text/html; charset=utf-8', '.wgsl': 'text/plain' }
const server = createServer((req, res) => {
  const f = join(ROOT, normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)))
  let s; try { s = statSync(f).size } catch { return res.writeHead(404).end() }
  res.writeHead(200, { 'Content-Type': MIME[extname(f)] ?? 'application/octet-stream', 'Content-Length': s })
  createReadStream(f).pipe(res)
})
await new Promise((r) => server.listen(8917, '127.0.0.1', r))
const browser = await puppeteer.launch({ headless: 'new', args: [
  '--no-sandbox', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface',
  '--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist',
  '--disable-dawn-features=timestamp_quantization', '--ignore-gpu-blocklist' ], protocolTimeout: 600000 })
const page = await browser.newPage()
page.on('console', (m) => { const t = m.text(); if (t.startsWith('[mb]')) console.log(t) })
// Chrome on Linux/NVIDIA (Vulkan) may withhold shader-f16 even when the driver has it;
// the page then runs an f32 transcription of the same kernel (?f32=1) and says so.
let R
for (const q of ['', '?f32=1']) {
  await page.goto('http://127.0.0.1:8917/microbench.html' + q, { waitUntil: 'load' })
  await page.waitForFunction('window.__mbDone === true', { timeout: 600000, polling: 500 })
  R = await page.evaluate(() => window.__mb)
  if (!(R.error && /lacks shader-f16/.test(R.error))) break
  console.log('no shader-f16 on this adapter; rerunning as f32')
}
R.date = new Date().toISOString()
R.gpuInfo = await page.evaluate(async () => { const a = await navigator.gpu.requestAdapter(); return { vendor: a?.info?.vendor, architecture: a?.info?.architecture, device: a?.info?.device, description: a?.info?.description, features: [...(a?.features ?? [])] } })
writeFileSync('microbench-colab-t4-f32.json', JSON.stringify(R, null, 2))
console.log('ADAPTER', JSON.stringify(R.gpuInfo))
if (R.error) console.log('PAGE ERROR', R.error)
console.log('saved microbench-colab-t4-f32.json; copy it back into out/ of the litert-capture folder')
await browser.close(); server.close()
`

const files = {
  'microbench.html': b64(join(ROOT, 'www/microbench.html')),
  'out/shaders/0112_unlabeled.wgsl': b64(join(ROOT, 'out/shaders/0112_unlabeled.wgsl')),
  'run.mjs': Buffer.from(driver).toString('base64'),
}

const cell = (type, id, source) => ({ cell_type: type, id, metadata: {}, source: source.split('\n').map((l, i, a) => (i < a.length - 1 ? l + '\n' : l)), ...(type === 'code' ? { execution_count: null, outputs: [] } : {}) })

const nb = {
  nbformat: 4, nbformat_minor: 5,
  metadata: { accelerator: 'GPU', colab: { gpuType: 'T4' }, kernelspec: { name: 'python3', display_name: 'Python 3' } },
  cells: [
    cell('markdown', 'a1b2c3d4', `# LiteRT-LM kernel 0112 on a Colab T4

**First: Runtime → Change runtime type → T4 GPU.**

Runs the captured LiteRT-LM WebGPU matrix-vector kernel 0112 (Gemma 4 E2B web, @litert-lm/core 0.17.1) verbatim (or, where Chrome offers no shader-f16, an f32 transcription of it, which the driver announces) and with deeper K splits, through headless Chrome on the T4, as \`node microbench.mjs\` does on the Mac. Cell 1 installs the NVIDIA Vulkan userspace Chrome needs (zero-tvm's recipe, after Chrome's colab-headless guide); if it prints \`llvmpipe\` instead of \`Tesla T4\`, stop and report the driver version from \`nvidia-smi\`.

Nothing is downloaded from Hugging Face: the page generates its own random weights.

Kernel 0112 is emitted at run time by \`@litert-lm/core\` 0.17.1 (Google LLC) and is reproduced here under the Apache License 2.0; provenance and the license text are in \`out/shaders/NOTICE\` and \`out/shaders/LICENSE\` in the repository, https://github.com/abgnydn/litert-capture, which also holds the method, the harness and the records. Independent work, not affiliated with Google.`),
    cell('code', 'e5f6a7b8', `# 1) Node 22 + the NVIDIA Vulkan userspace matching the driver
import subprocess
drv = subprocess.check_output('nvidia-smi --query-gpu=driver_version --format=csv,noheader', shell=True).decode().split('.')[0].strip()
print('NVIDIA driver major:', drv, '-> installing libnvidia-gl-' + drv)
!curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1
!apt-get -qq update > /dev/null
# libasound2 is libasound2t64 on Colab's Ubuntu 24.04; one missing package name aborts the whole install
!apt-get -qq install -y nodejs vulkan-tools libnvidia-gl-{drv} libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libgbm1 libasound2t64 libxcomposite1 libxdamage1 libxrandr2 libxkbcommon0 libxfixes3 libdrm2 libpango-1.0-0 libcairo2 libatspi2.0-0 fonts-liberation > /dev/null
!rm -f /usr/share/vulkan/icd.d/lvp_icd*.json
print('--- Vulkan device (want Tesla T4, not llvmpipe): ---')
!vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName' || echo 'NO VULKAN DEVICE'`),
    cell('code', 'c9d0e1f2', `# 2) Unpack the page, the captured kernel and the driver; install puppeteer 25.11.0 (downloads Chrome for Testing 153, which made the committed T4 record)
import base64, os, json
files = ${JSON.stringify(files)}
os.makedirs('kernel0112/out/shaders', exist_ok=True)
for name, data in files.items():
    with open(os.path.join('kernel0112', name), 'wb') as f:
        f.write(base64.b64decode(data))
!cd kernel0112 && npm init -y > /dev/null && npm i --silent puppeteer@25.11.0 > /dev/null && echo installed`),
    cell('code', 'a3b4c5d6', `# 3) Run. Prints one line per variant; the JSON is the record to copy back.
!cd kernel0112 && node run.mjs
print(open('kernel0112/microbench-colab-t4-f32.json').read()[:600])`),
    cell('markdown', 'e7f8a9b0', `## Reading the result

\`orig\` is Google's kernel as shipped (16 × 4 workgroups, 12,288 threads). \`wg256-same\` changes only the workgroup size (control). \`split16-64\` and \`split32-64\` keep 64-thread workgroups and dispatch 4x and 8x the threads. The GB/s on each printed line is effective weight-streaming bandwidth: the 4.5 MiB (4,718,592 bytes) of weights the kernel reads per unit of kernel time, which is not a measurement of memory traffic, since a byte served from cache counts the same. Byte counts in this notebook are MiB (2^20 bytes) and GB/s is decimal. On the M2 Max: orig 60.9 µs, control 60.6, split32 36.8 (1.65x, in two runs whose browser, Chrome 146, is inferred from the adapter string, confirmed by provenanced out/microbench-0112-2026-09-24T07-36-19-655Z.json at 1.638x; same single-dispatch harness reads 0.84x in one recorded Chrome 131 run, 78.12 vs 93.12 us, on the same machine). Whether the T4 shows the same shape is the question this notebook answers. Download \`kernel0112/microbench-colab-t4-f32.json\` and drop it into \`out/\`.`),
  ],
}
writeFileSync(join(HERE, 'kernel-0112-t4.ipynb'), JSON.stringify(nb, null, 1))
console.log('wrote colab/kernel-0112-t4.ipynb')
