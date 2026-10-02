# engines/ort — CAPTURE-ROADMAP 10 (Transformers.js / ORT-Web)

Status: parked on `model-missing` (no ONNX bundle in this repo; see
`ledger/capture-10-parked.json`). No synthetic `ort-web.json` written.

## Gap this closes

ORT WebGPU profiling gives per-operator TIMES only -- no WGSL source.
`hook.js` supplies the missing half: a prototype-level WGSL dump
(`createShaderModule` + sync/async pipelines + dispatch counters),
installed before `onnxruntime-web` compiles anything. `analyze.mjs` joins
the dump with ORT profiling times by pipeline/dispatch position, the same
position-join as root `analyze.mjs` (`trace.json` x `timing-kernel.json`).

Proved pattern: `www/capture.html` (prototype-level, covers any device) +
`engines/webllm/hook.js` (same shape for TVM). ORT reference: the
`tjs-bench` harness in `~/dev/zero-tvm/src/tjs-bench/main.ts`
(`@huggingface/transformers` on the WebGPU EP, `env.localModelPath` mirror,
`?model=phi3|qwen3|qwen35`).

## Files

- `hook.js` -- WGSL dump (times absent by design; times come from ORT).
- `capture.html` -- loads `@huggingface/transformers@4.2.0` from CDN,
  `env.localModelPath = <origin>/model/`, `device: 'webgpu'`, one
  32-token generation; records wall ms + hook dump.
- `capture.mjs` -- serves `engines/ort/` + `/model/` mirror, puppeteer
  driver, writes `engines/ort/ort-web.json` (+ provenance +
  `manifest_sha256`). Model gate first: absent ONNX = exit 3 with exact
  commands, never silent download.
- `analyze.mjs` -- offline join with `ort-profile.json` sidecar
  (per-operator ms by position) into a per-operator table.

## Unpark (needs explicit human go, ~2.2 GB)

```sh
hf download onnx-community/Phi-3-mini-4k-instruct-ONNX --include 'onnx/model_q4f16.onnx*' --local-dir engines/ort/model
node engines/ort/capture.mjs
node engines/ort/analyze.mjs 400
```

A same-quant mirror already exists on this machine at
`~/dev/zero-tvm/.weights-local/onnx/Phi-3-mini-4k-instruct-ONNX/onnx`
(`model_q4f16.onnx` 291 KB + `model_q4f16.onnx_data` 2.0 GB +
`model_q4f16.onnx_data_1` 188 MB) but is not copied silently.
