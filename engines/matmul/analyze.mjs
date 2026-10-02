// Offline K-search ranking + bytes/us roofline for CAPTURE-ROADMAP 12.
//
// Reads engines/matmul/matmul-micro.json (real GPU record from
// microbench.mjs) and prints the per-variant table (cold median us,
// bytes/us, GB/s, % of rated) plus the committed best split by cold
// median. Same construction as root analyze.mjs (bytes read / kernel
// time = effective GB/s). No GPU run, no synthetic numbers: absent record
// = exit 1.
//
// Usage: node engines/matmul/analyze.mjs [peak-gbs]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PEAK_GBS = Number(process.argv[2] ?? 400);

let R;
try {
  R = JSON.parse(readFileSync(join(HERE, "matmul-micro.json"), "utf8"));
} catch {
  console.log("no engines/matmul/matmul-micro.json: run node engines/matmul/microbench.mjs first (no synthetic table).");
  process.exit(1);
}

const bytes = R.bytesPerDispatch;
console.log(`matmul ${R.shape.M}x${R.shape.N}x${R.shape.K}: ${bytes} bytes/dispatch`);
console.log("variant  cold-us  GB/s  %rated  hot-us  max|err|");
const ranked = Object.entries(R.variants)
  .map(([name, v]) => ({ name, ...v }))
  .sort((a, b) => a.cold.median - b.cold.median);
for (const v of ranked) {
  const gbs = bytes / (v.cold.median * 1e-6) / 1e9;
  console.log(
    `${v.name.padEnd(8)} ${v.cold.median.toFixed(1).padStart(8)} ${gbs.toFixed(1).padStart(5)} ${(100 * gbs / PEAK_GBS).toFixed(0).padStart(6)}% ${v.hot.median.toFixed(1).padStart(7)}  ${v.maxAbsErr.toExponential(2)}`,
  );
}
const best = ranked[0];
console.log(`best split by cold median: ${best.name} (${best.cold.median.toFixed(1)} us, ${(bytes / (best.cold.median * 1e-6) / 1e9).toFixed(1)} GB/s)`);
console.log(`provenance: ${R.provenance?.browser ?? "?"}, ${R.provenance?.adapter?.vendor ?? "?"}/${R.provenance?.adapter?.architecture ?? "?"}, harness ${R.provenance?.harness_commit?.slice(0, 8) ?? "unknown"}`);
