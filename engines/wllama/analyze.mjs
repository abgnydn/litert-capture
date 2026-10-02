// Offline join for CAPTURE-ROADMAP 11: same capture+trace pipeline shape.
//
// Reads engines/wllama/wllama.json (hook dump + wall time from a real
// wllama session) and prints the per-pipeline table (dispatch share, WGSL
// bytes, sha) that the trace half (per-dispatch shapes/times by position)
// joins against -- same position-join as root analyze.mjs. No GPU run here,
// no synthetic numbers: absent record = exit 1 with the unpark commands.
//
// Usage: node engines/wllama/analyze.mjs [peak-gbs]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PEAK_GBS = Number(process.argv[2] ?? 400);
void PEAK_GBS;

let R;
try {
  R = JSON.parse(readFileSync(join(HERE, "wllama.json"), "utf8"));
} catch {
  console.log("no engines/wllama/wllama.json: parked on model-missing, no synthetic table.");
  console.log("To unpark:");
  console.log("  hf download microsoft/Phi-3-mini-4k-instruct-gguf --include 'Phi-3-mini-4k-instruct-q4.gguf' --local-dir engines/wllama/model");
  console.log("  node engines/wllama/capture.mjs");
  console.log("  node engines/wllama/analyze.mjs 400");
  process.exit(1);
}

const rows = (R.pipelines ?? []).map((p) => {
  const s = (R.shaders ?? [])[p.shaderId];
  return {
    pid: p.id,
    label: (p.label || s?.label || "unlabeled").slice(0, 60),
    bytes: s?.code?.length ?? 0,
    sha: s?.shaderSha256?.slice(0, 12) ?? "n/a",
    dispatches: R.dispatchByPipeline?.[p.id] ?? 0,
  };
}).sort((a, b) => b.dispatches - a.dispatches);

console.log(`wllama: ${R.shaders.length} shaders, ${R.pipelines.length} pipelines, wall ${R.wallMs ?? "n/a"} ms`);
for (const r of rows.slice(0, 20)) {
  console.log(`  pid ${String(r.pid).padStart(3)} dispatches ${String(r.dispatches).padStart(6)} bytes ${String(r.bytes).padStart(6)} sha ${r.sha}  ${r.label}`);
}
