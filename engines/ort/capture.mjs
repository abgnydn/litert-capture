// Driver for CAPTURE-ROADMAP 10 (ORT-Web / transformers.js).
//
// Serves engines/ort/ + a local ONNX mirror, installs hook.js before the
// ORT WebGPU EP compiles anything, runs one generation, and writes
// engines/ort/ort-web.json (WGSL dump + pipeline labels + ORT wall time +
// provenance + manifest_sha256). Per-operator join lives in analyze.mjs.
//
// Model gate FIRST (CAPTURE-LOOP.md R3, other-engine rule): the ONNX q4f16
// bundle is ~2.2 GB and is NOT downloaded silently. If
// engines/ort/model/onnx/model_q4f16.onnx (+ .onnx_data) is absent, print
// the exact hf download + harness commands and exit 3.
//
// Usage: node engines/ort/capture.mjs

import { createReadStream, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";
import { provenance } from "../../provenance.mjs";
import { manifestSha256 } from "../../shader-sha.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const MODEL_ONNX = join(HERE, "model", "onnx", "model_q4f16.onnx");
const PORT = 8922;
const TIMEOUT_MS = 30 * 60 * 1000;

const modelPresent = (() => {
  try {
    statSync(MODEL_ONNX);
    return true;
  } catch {
    return false;
  }
})();

if (!modelPresent) {
  console.log("PARKED model-missing: engines/ort/model/onnx/model_q4f16.onnx absent.");
  console.log("Do NOT download silently (multi-GB). To unpark, run:");
  console.log("  hf download onnx-community/Phi-3-mini-4k-instruct-ONNX --include 'onnx/model_q4f16.onnx*' --local-dir engines/ort/model");
  console.log("  node engines/ort/capture.mjs");
  console.log("  node engines/ort/analyze.mjs 400");
  process.exit(3);
}

const MIME = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".onnx": "application/octet-stream" };
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
  const file = path.startsWith("/model/") ? join(HERE, "model", path.slice(7)) : join(HERE, path.slice(1));
  if (!file.startsWith(HERE)) return res.writeHead(403).end();
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
  await page.waitForFunction("window.__ortDone === true", { timeout: TIMEOUT_MS, polling: 1000 });
  const cap = await page.evaluate(() => {
    const { adapter, ...rest } = window.__ortCapture;
    return { adapter, ...rest, output: window.__ortOutput ?? null, wallMs: window.__ortWallMs ?? null, version: window.__ortVersion ?? null };
  });
  const record = {
    ...cap,
    provenance: await provenance({ browser, adapter: cap.adapter, pkg: cap.version }),
    manifest_sha256: manifestSha256(join(ROOT, "out", "manifest.json")),
    date: new Date().toISOString(),
  };
  writeFileSync(join(HERE, "ort-web.json"), JSON.stringify(record, null, 2));
  console.log(`saved engines/ort/ort-web.json (${cap.shaders.length} shaders, ${cap.pipelines.length} pipelines)`);
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
