# CAPTURE-ROADMAP.md — LiteRT WebGPU capture targets

Scope: LiteRT 0.17.1 on same M2 Max machine first; no new infra for A.
Done per item: commit `out/<name>.json` + `provenance.mjs` fields; gate per `LOOP.md` (`biome`, `node --check`, `analyze.mjs`, `www/` builds, `gitleaks`).

## A. Same-engine unmeasured (LiteRT 0.17.1, no new infra)

1. Prefill (37 decode-absent pipelines): long-prompt capture via `capture.mjs` / `trace.mjs`, prefill tok/s + dominant kernels. → `out/prefill.json`
2. KV-cache layout 0094/0095/0096 (3 MiB/dispatch): fill context to 2k tokens, watch shape growth in `trace.mjs`. → `out/kv-layout.json`
3. Fake-quant 0119 fusion (322 dispatches/token, 1.15 ms): `patch.mjs` method-fusion test, before/after via `diff-dispatch.mjs`. → `out/fakequant-0119.json`
4. Load path (Engine.create 2.65–30.1 s spread, 23 s first token): split weight-convert vs warm-up with `timing.mjs` + `provenance.mjs`. → `out/load-path.json`
5. Power/thermal curve: `provenance.mjs` power fields vs throttled <1.0x readings, repeated runs. → `out/power-curve.json`
6. First-slot 17% slow + 6.5% baseline gap: repeated bare runs via `timing.mjs`, isolate cold-slot effect. → `out/firstslot.json`
7. Native ratio: M2 Max native vs web on same machine (existing card numbers not comparable), same prompts. → `out/native-ratio.json`
8. Non-f16 path: what engine does where shader-f16 missing (beyond f32 note), capture fallback shaders via `capture.mjs` + `shader-sha.mjs`. → `out/nonf16.json`

## B. Same-technique other engines (new `out/` per engine)

9. WebLLM TVM shaders: capture + per-kernel table + K-search (zero-tvm proved hook via `patch.mjs`). → `out/webllm-tvm.json`
10. Transformers.js / ORT-Web: add WGSL dump where ORT profiling gives times only, join with `analyze.mjs`. → `out/ort-web.json`
11. wllama / webllm-bench models: same `capture.mjs` + `trace.mjs` pipeline on their model/shader sets. → `out/wllama.json`
12. Generic WebGPU matmul/conv: `microbench.mjs` pattern port, K-search + bytes/µs roofline. → `out/matmul-micro.json`

## C. New metric types (harness extensions)

13. CPU-side per-pass attribution: extend `timing.mjs` bare/gpu/kernel split to per-pass table. → `out/cpu-attrib.json`
14. Memory residency + upload time: textures/buffers sizes + weight-upload callback, peak vs steady state. → `out/memory-residency.json`
15. Vendor matrix (Intel/AMD/Adreno/Metal/desktop-Linux): `colab/` notebook template (T4 row exists), one row per vendor. → `out/vendor-matrix.json`
16. Correctness gate: token-diff across prompts as standard, generalize near-tie analysis. → `out/correctness-gate.json`
17. Prefill/decode joint roofline: bytes/µs already computed in `analyze.mjs`, render per-kernel chart in `www/`. → `out/roofline.json`
18. Timestamp-quantization + throttling matrix: browser flags x power states, resolution floor per setting. → `out/quant-throttle.json`

## D. Order + done-definition

Order: A first (same machine, items 1–8), then C extensions (items 13–18), then B engines (items 9–12, forces new repo per `ROADMAP.md` split rule).
Done: each item commits `out/<name>.json` + provenance; gate is `biome`, `node --check`, `analyze.mjs`, `www/` builds, `gitleaks` per `LOOP.md`.
