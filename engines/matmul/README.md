# engines/matmul — CAPTURE-ROADMAP 12 (generic WebGPU matmul/conv)

Status: done (real local GPU run, no model needed).

## What this is

Port of the `microbench.mjs` pattern to a model-free kernel: f32
`C = A @ B` at 512^3 (3 MiB per dispatch), four variants (baseline `orig`
8x8, `wg256` 16x16 workgroup control, `ksplit2`/`ksplit4` K-split depths),
timestamp-query per-dispatch timing, forward+reverse pooled medians (96
cold + 12 hot per variant), JS CPU reference per variant (tol 1e-3),
bytes/us roofline + K-search ranking.

## Files

- `matmul.html` -- the bench page (serves itself only; no `out/`, no
  bundle). Smaller sample budget than `www/microbench.html` (ROUNDS 6 x
  NMAT_COLD 8) because each 512^3 dispatch costs milliseconds, not
  microseconds.
- `microbench.mjs` -- puppeteer driver (same flags as root `microbench.mjs`),
  stamps `provenance` + `manifest_sha256`, writes a dated
  `matmul-micro-<stamp>.json` plus the `matmul-micro.json` tip the analyzer
  reads. Root `out/*.json` untouched.
- `analyze.mjs` -- offline K-search ranking + roofline (`[peak-gbs]`,
  default 400 like root `analyze.mjs`).
- `matmul-micro.json` (+ dated sibling) -- real record from this machine.

## Run

```sh
node engines/matmul/microbench.mjs
node engines/matmul/analyze.mjs 400
```
