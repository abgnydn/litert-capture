// Driver for CAPTURE-ROADMAP 11 (wllama / llama.cpp WebGPU).
//
// Serves engines/wllama/ + a local GGUF mirror, installs hook.js before the
// wllama WebGPU backend compiles anything, runs one short completion, and
// writes engines/wllama/wllama.json (WGSL dump + pipeline labels + wall
// time + provenance + manifest_sha256). Same capture+trace pipeline shape
// as the root harness; the trace join lives in analyze.mjs.
//
// Model gate FIRST (CAPTURE-LOOP.md R3, other-engine rule): the GGUF is
// ~2.2 GB and is NOT downloaded silently. If engines/wllama/model/ holds
// no *.gguf, print the exact hf download + harness commands and exit 3.
//
// Usage: node engines/wllama/capture.mjs

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
const PORT = 8923;
const TIMEOUT_MS = 30 * 60 * 1000;

const modelPresent = (() => {
  try {
    return readdirSync(MODEL_DIR).some((f) => f.endsWith(".gguf"));
  } catch {
    return false;
  }
})();

if (!modelPresent) {
  console.log("PARKED model-missing: engines/wllama/model/ holds no *.gguf.");
  console.log("Do NOT download silently (multi-GB). To unpark, run:");
  console.log("  hf download microsoft/Phi-3-mini-4k-instruct-gguf --include 'Phi-3-mini-4k-instruct-q4.gguf' --local-dir engines/wllama/model");
  console.log("  node engines/wllama/capture.mjs");
  console.log("  node engines/wllama/analyze.mjs 400");
  process.exit(3);
}

const MIME = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".gguf": "application/octet-stream", ".wasm": "application/wasm" };
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
  await page.waitForFunction("window.__wllamaDone === true", { timeout: TIMEOUT_MS, polling: 1000 });
  const cap = await page.evaluate(() => {
    const { adapter, ...rest } = window.__wllamaCapture;
    return { adapter, ...rest, output: window.__wllamaOutput ?? null, wallMs: window.__wllamaWallMs ?? null, version: window.__wllamaVersion ?? null };
  });
  const record = {
    ...cap,
    provenance: await provenance({ browser, adapter: cap.adapter, pkg: cap.version }),
    manifest_sha256: manifestSha256(join(ROOT, "out", "manifest.json")),
    date: new Date().toISOString(),
  };
  writeFileSync(join(HERE, "wllama.json"), JSON.stringify(record, null, 2));
  console.log(`saved engines/wllama/wllama.json (${cap.shaders.length} shaders, ${cap.pipelines.length} pipelines)`);
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
