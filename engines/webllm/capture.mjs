// Driver for CAPTURE-ROADMAP 09 (WebLLM TVM shaders).
//
// Serves engines/webllm/ + a local MLC model mirror, installs hook.js before
// importing @mlc-ai/web-llm, runs one chat completion, and writes
// engines/webllm/webllm-tvm.json (shaders + pipelines + dispatch counts +
// provenance + manifest_sha256). Per-kernel table + K-search live in
// analyze.mjs (offline join, same construction as root analyze.mjs).
//
// Model gate FIRST (CAPTURE-LOOP.md R3, other-engine rule): the MLC bundle
// is ~2 GB and is NOT downloaded silently. If engines/webllm/model/ has no
// mlc-chat-config.json + params_shard_*.bin, print the exact hf download +
// harness commands and exit 3 (park, never fail, never synthetic).
//
// Usage:
//   node engines/webllm/capture.mjs
//
// Parked harness (after download):
//   node engines/webllm/capture.mjs
//   node engines/webllm/analyze.mjs 400

import { createReadStream, readdirSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";
import { provenance } from "../../provenance.mjs";
import { manifestSha256 } from "../../shader-sha.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const MODEL_DIR = join(HERE, "model");
const PORT = 8921;
const TIMEOUT_MS = 30 * 60 * 1000;
const PEAK_GBS = Number(process.argv[2] ?? 400);
void PEAK_GBS;

const modelPresent = (() => {
  try {
    const files = readdirSync(MODEL_DIR);
    return files.includes("mlc-chat-config.json") && files.some((f) => /^params_shard_/.test(f));
  } catch {
    return false;
  }
})();

if (!modelPresent) {
  console.log("PARKED model-missing: engines/webllm/model/ has no MLC bundle.");
  console.log("Do NOT download silently (multi-GB). To unpark, run:");
  console.log("  hf download mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC --local-dir engines/webllm/model");
  console.log("  node engines/webllm/capture.mjs");
  console.log("  node engines/webllm/analyze.mjs 400");
  process.exit(3);
}

const MIME = { ".html": "text/html", ".js": "text/javascript", ".bin": "application/octet-stream", ".json": "application/json" };
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
  const file = path.startsWith("/model/") ? join(MODEL_DIR, path.slice(7)) : join(HERE, path.slice(1));
  if (!file.startsWith(HERE) && !file.startsWith(MODEL_DIR)) return res.writeHead(403).end();
  let size;
  try {
    size = statSync(file).size;
  } catch {
    return res.writeHead(404).end();
  }
  res.writeHead(200, {
    "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
    "Content-Length": size,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
  });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(PORT, "127.0.0.1", r));

const browser = await puppeteer.launch({
  headless: false,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist",
    "--disable-dawn-features=timestamp_quantization",
    "--enable-features=Vulkan",
  ],
  protocolTimeout: TIMEOUT_MS,
});

let exitCode = 0;
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log(`[page] ${m.text()}`));
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto(`http://127.0.0.1:${PORT}/capture.html`, { waitUntil: "load" });
  await page.waitForFunction("window.__webllmDone === true", { timeout: TIMEOUT_MS, polling: 1000 });
  const cap = await page.evaluate(() => {
    const { adapter, ...rest } = window.__webllmCapture;
    return { adapter, ...rest, output: window.__webllmOutput ?? null, version: window.__webllmVersion ?? null };
  });
  const record = {
    ...cap,
    provenance: await provenance({ browser, adapter: cap.adapter, pkg: cap.version }),
    manifest_sha256: manifestSha256(join(ROOT, "out", "manifest.json")),
    date: new Date().toISOString(),
  };
  writeFileSync(join(HERE, "webllm-tvm.json"), JSON.stringify(record, null, 2));
  console.log(`saved engines/webllm/webllm-tvm.json (${cap.shaders.length} shaders, ${cap.pipelines.length} pipelines)`);
  if (cap.error) {
    console.log(`PAGE ERROR: ${cap.error}`);
    exitCode = 2;
  }
} catch (e) {
  console.log(`DRIVER ERROR: ${e?.stack ?? e}`);
  exitCode = 1;
} finally {
  await browser.close();
  server.close();
}
process.exit(exitCode);
