# engines/webllm — CAPTURE-ROADMAP 09 (WebLLM TVM shaders)

Status: parked on `model-missing` (no MLC bundle in this repo; see
`ledger/capture-09-parked.json`). No synthetic `webllm-tvm.json` written.

## What this captures

Same `capture.mjs` + `trace.mjs` pipeline as the LiteRT engine, pointed at
WebLLM (TVM) instead: WGSL dump via `hook.js`, per-kernel-family table plus
K-search via `analyze.mjs` (offline join, same construction as root
`analyze.mjs`).

## Hook

`hook.js` is a prototype-level wrapper (`GPUDevice.createShaderModule`,
`createComputePipeline` + `createComputePipelineAsync`, dispatch counters),
installed before `@mlc-ai/web-llm` is imported. Prototype-level (like
`www/capture.html`) rather than device-level because WebLLM's TVM runtime
creates its own device internally. TVM uses `createComputePipelineAsync`
almost exclusively; both paths are recorded.

Proved reference: `~/dev/zero-tvm/src/_archived/capture.ts:124`
(`patchForCapture`, device-level) plus the whole-session WGSL dump in
`~/dev/zero-tvm/src/tvm-shaders/` (~60 files: `fused_dequantize*_NT_matmul*`
family, `batch_prefill` / `batch_decode` paged-KV, `rms_norm` /
`fused_add_norm`, rope, chunked softmax, gather/take/copy/reshape).

Adapted to `@mlc-ai/web-llm` 0.2.85 (npm latest at time of writing; zero-tvm
pins 0.2.84). The hook is version-agnostic; the `model_lib` wasm URL segment
(`v0_2_84` vs `v0_2_85`) and the loaded package version are recorded in
provenance.

## Per-kernel table + K-search

`analyze.mjs` groups pipelines by kernel family (from label, never byte
length) and prints dispatch share + WGSL bytes. K-search worklist: the
`fused_dequantize*` matmul family, sweeping `ks`/`wgX` at fixed workgroup
budget (`dispatchX = OUT_SLICES / wgX`, forward+reverse pooled medians + CPU
reference), ported from `www/microbench.html` `variantSource`.

## Unpark (needs explicit human go, ~2 GB)

```sh
hf download mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC --local-dir engines/webllm/model
node engines/webllm/capture.mjs
node engines/webllm/analyze.mjs 400
```

Smaller alternative: `mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC` (~1.3 GB).
A whole-session reference already exists on this machine at
`~/dev/zero-tvm/.weights-local/Qwen3.5-4B-q4f16_1-MLC` (2.2 GB) plus the
committed `~/dev/zero-tvm/src/tvm-shaders/` dump, but neither is wired into
this repo (no GBs copied silently).
