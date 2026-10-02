# engines/wllama — CAPTURE-ROADMAP 11 (wllama / llama.cpp WebGPU)

Status: parked on `model-missing` (no GGUF in this repo; see
`ledger/capture-11-parked.json`). No synthetic `wllama.json` written.

## Pipeline

Same capture+trace shape as the root harness, pointed at wllama: WGSL dump
via `hook.js` (capture half), per-pipeline table via `analyze.mjs`, trace
half (per-dispatch shapes/times by position) joins like root `analyze.mjs`.

Reference: `~/dev/zero-tvm/src/wllama-bench/main.ts` (`@wllama/wllama`,
GGUF from the local mirror, `?model=phi3|qwen3|qwen35`, `?ngl` GPU-layer
override, same prompt/protocol as the sibling benches). wllama cannot do a
same-bytes comparison (GGUF vs MLC `q4f16_1`: different block layout/group
size/rounding), so any future number is a runtime+quantization comparison --
stated wherever published.

## Files

- `hook.js` -- prototype-level WGSL dump + dispatch counters.
- `capture.html` -- `@wllama/wllama@3.5.1` (CDN esm), GGUF at `/model/`,
  full offload (`n_gpu_layers: 99999`; `?ngl=0` analogue is the CPU
  control), one short completion.
- `capture.mjs` -- serves `engines/wllama/` + `/model/`, puppeteer driver,
  writes `engines/wllama/wllama.json` (+ provenance + `manifest_sha256`).
  Model gate first: no `*.gguf` = exit 3 with exact commands.
- `analyze.mjs` -- offline per-pipeline table (dispatch share, bytes, sha).

## Unpark (needs explicit human go, ~2.2 GB)

```sh
hf download microsoft/Phi-3-mini-4k-instruct-gguf --include 'Phi-3-mini-4k-instruct-q4.gguf' --local-dir engines/wllama/model
node engines/wllama/capture.mjs
node engines/wllama/analyze.mjs 400
```

Same-file mirrors already on this machine
(`~/dev/zero-tvm/.weights-local/gguf/phi3-q4.gguf` 2.2 GB,
`qwen3-4b-q4km.gguf` 2.3 GB, `qwen35-4b-q4km.gguf` 2.6 GB) are not copied
silently.
