// Offline join for CAPTURE-ROADMAP 10: WGSL dump x ORT profiling times.
//
// Reads engines/ort/ort-web.json (hook dump + ORT wall time from a real
// session) and, when an ORT profiling sidecar is present
// (engines/ort/ort-profile.json, per-operator ms exported from the session),
// joins the two by pipeline/dispatch position -- the same position-join
// construction as root analyze.mjs (trace.json x timing-kernel.json).
// ORT profiling gives TIMES only, so without the hook there is no WGSL to
// attach them to; without profiling there are no per-op times, only the
// wall total. Both halves are required for the per-operator table.
//
// No GPU run here, no synthetic numbers: absent record = exit 1 with the
// unpark commands.
//
// Usage: node engines/ort/analyze.mjs [peak-gbs]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PEAK_GBS = Number(process.argv[2] ?? 400);
void PEAK_GBS;

let R;
try {
  R = JSON.parse(readFileSync(join(HERE, "ort-web.json"), "utf8"));
} catch {
  console.log("no engines/ort/ort-web.json: parked on model-missing, no synthetic table.");
  console.log("To unpark:");
  console.log("  hf download onnx-community/Phi-3-mini-4k-instruct-ONNX --include 'onnx/model_q4f16.onnx*' --local-dir engines/ort/model");
  console.log("  node engines/ort/capture.mjs");
  console.log("  node engines/ort/analyze.mjs 400");
  process.exit(1);
}

let P = null;
try {
  P = JSON.parse(readFileSync(join(HERE, "ort-profile.json"), "utf8"));
} catch {
  P = null;
}

const rows = (R.pipelines ?? []).map((p) => {
  const s = (R.shaders ?? [])[p.shaderId];
  return {
    pid: p.id,
    label: (p.label || s?.label || "unlabeled").slice(0, 60),
    bytes: s?.code?.length ?? 0,
    dispatches: R.dispatchByPipeline?.[p.id] ?? 0,
    ms: Array.isArray(P?.perOperator) ? (P.perOperator[p.id]?.ms ?? null) : null,
  };
}).sort((a, b) => b.dispatches - a.dispatches);

console.log(`ort-web: ${R.shaders.length} shaders, ${R.pipelines.length} pipelines, wall ${R.wallMs ?? "n/a"} ms`);
if (!P) console.log("no ort-profile.json sidecar: per-operator ms unavailable (wall total only); export ORT session profiling to join times by position.");
for (const r of rows.slice(0, 20)) {
  console.log(`  pid ${String(r.pid).padStart(3)} dispatches ${String(r.dispatches).padStart(6)} bytes ${String(r.bytes).padStart(6)} ms ${r.ms ?? "n/a"}  ${r.label}`);
}
