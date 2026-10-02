// Driver for CAPTURE-ROADMAP 12 (generic WebGPU matmul microbench).
//
// Drives engines/matmul/matmul.html (f32 C = A @ B, 512^3, no model) and
// writes engines/matmul/matmul-micro.json. Port of root microbench.mjs:
// same puppeteer WebGPU flags, same provenance + manifest_sha256 stamping,
// same dated-file discipline (writes a dated file per run, never
// overwrites; root out/*.json untouched).
//
// Usage: node engines/matmul/microbench.mjs
// 30-min R1 time-box is enforced by the page wait timeout below.

import { createReadStream, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";
import { provenance } from "../../provenance.mjs";
import { manifestSha256 } from "../../shader-sha.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const PORT = 8924;

const MIME = { ".html": "text/html", ".js": "text/javascript" };
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
  const file = join(HERE, path.slice(1));
  if (!file.startsWith(HERE)) return res.writeHead(403).end();
  let size;
  try {
    size = statSync(file).size;
  } catch {
    return res.writeHead(404).end();
  }
  res.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream", "Content-Length": size });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(PORT, "127.0.0.1", r));

const browser = await puppeteer.launch({
  headless: false,
  executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist",
    "--disable-dawn-features=timestamp_quantization",
    "--enable-features=Vulkan",
  ],
});
let exitCode = 0;
try {
  const page = await browser.newPage();
  page.on("console", (m) => {
    const t = m.text();
    if (t.startsWith("[matmul]")) console.log(t);
  });
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto(`http://127.0.0.1:${PORT}/matmul.html`, { waitUntil: "load" });
  await page.waitForFunction("window.__mbDone === true", { timeout: 30 * 60 * 1000, polling: 500 });
  const R = await page.evaluate(() => window.__mb);
  R.provenance = await provenance({ browser, adapter: R.adapter, pkg: null });
  R.manifest_sha256 = manifestSha256(join(ROOT, "out", "manifest.json"));
  R.date = new Date().toISOString();
  const stamp = R.date.replace(/[:.]/g, "-");
  const name = `matmul-micro-${stamp}.json`;
  writeFileSync(join(HERE, name), JSON.stringify(R, null, 2));
  writeFileSync(join(HERE, "matmul-micro.json"), JSON.stringify(R, null, 2));
  console.log(`saved engines/matmul/${name} + engines/matmul/matmul-micro.json`);
  if (R.error || R.gpuError) exitCode = 2;
  console.log(`adapter ${JSON.stringify(R.adapter)}, base ${R.baseBytes} bytes`);
  if (R.adapter?.vendor === "apple" && R.adapter?.architecture !== "metal-3") {
    console.log(`WARNING: adapter architecture is ${R.adapter.architecture}, not metal-3`);
  }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`);
  exitCode = 1;
} finally {
  await browser.close();
  server.close();
}
process.exit(exitCode);
