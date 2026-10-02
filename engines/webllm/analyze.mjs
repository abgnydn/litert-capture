// Offline per-kernel table + K-search plan for CAPTURE-ROADMAP 09.
//
// Reads engines/webllm/webllm-tvm.json (written by capture.mjs from a real
// WebLLM session) and prints a per-kernel-family table (dispatch count share,
// WGSL bytes, sha) plus the K-search worklist for the fused_dequantize*
// matmul family (ks/wgX sweep ported from www/microbench.html variantSource,
// dispatchX = OUT_SLICES / wgX). Same join construction as root analyze.mjs:
// pipelines joined to shaders by shaderId, family derived from label (never
// from byte length alone).
//
// No GPU run here, no synthetic numbers: if the record is absent (parked on
// model-missing) this exits 1 with the unpark commands instead of printing
// a made-up table.
//
// Usage: node engines/webllm/analyze.mjs [peak-gbs]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PEAK_GBS = Number(process.argv[2] ?? 400);
void PEAK_GBS;

let R;
try {
  R = JSON.parse(readFileSync(join(HERE, "webllm-tvm.json"), "utf8"));
} catch {
  console.log("no engines/webllm/webllm-tvm.json: parked on model-missing, no synthetic table.");
  console.log("To unpark:");
  console.log("  hf download mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC --local-dir engines/webllm/model");
  console.log("  node engines/webllm/capture.mjs");
  console.log("  node engines/webllm/analyze.mjs 400");
  process.exit(1);
}

const family = (label) => {
  const l = label ?? "";
  const m = /fused_dequantize\d+[^\s]*_matmul\d*/.exec(l) ?? /batch_(prefill|decode)[^\s]*/.exec(l) ?? /(rms_norm|rope|softmax|gather|take|copy|reshape|compact)[^\s]*/.exec(l);
  return m ? m[0] : (l || "unlabeled").slice(0, 48);
};

const byFamily = new Map();
for (const p of R.pipelines ?? []) {
  const s = (R.shaders ?? [])[p.shaderId];
  const f = family(p.label || s?.label);
  if (!byFamily.has(f)) byFamily.set(f, { family: f, pipelines: 0, dispatches: 0, bytes: 0 });
  const row = byFamily.get(f);
  row.pipelines++;
  row.dispatches += R.dispatchByPipeline?.[p.id] ?? 0;
  row.bytes += s?.code?.length ?? 0;
}

const rows = [...byFamily.values()].sort((a, b) => b.dispatches - a.dispatches);
console.log(`webllm-tvm: ${(R.shaders ?? []).length} shaders, ${(R.pipelines ?? []).length} pipelines`);
console.log("per-kernel-family table (dispatch share, WGSL bytes):");
for (const r of rows.slice(0, 20)) {
  console.log(`  ${r.family.slice(0, 52).padEnd(52)} pipelines ${String(r.pipelines).padStart(3)} dispatches ${String(r.dispatches).padStart(6)} bytes ${r.bytes}`);
}
console.log("K-search worklist: fused_dequantize* matmul family, sweep ks/wgX at fixed workgroup budget (dispatchX = OUT_SLICES / wgX), forward+reverse pooled medians + CPU reference, as in www/microbench.html variantSource.");
