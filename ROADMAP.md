# litert-capture roadmap

Status: v0.1.0, 8 scripts (`package.json:9-17`), live at `78ba31f`, Space `abgunaydin/kernel-0112`, upstream issue #3775 OPEN.

1. Parameterize VERSION in `analyze.mjs:101` and `trace.mjs:114` instead of hard-coded 0.17.1.
2. Re-run `capture.mjs` per release to track kernel drift across versions.
3. Diff dispatch lists between releases to flag added/removed/renamed kernels.
4. Pin bundle commit `b3ca0d2f` in manifest for reproducible captures.
5. Record model SHA in each capture manifest.
6. Note CDN vs tarball source in manifest (URL plus byte size).
7. Wire `capture.html` hooks for headless capture start/stop.
8. Store manifest SHA alongside `out/*.json` records.
9. Add per-kernel flags for 0105/0099/0113 in `microbench.mjs` (see `out/microbench-0105-*.json`, `out/microbench-0099-*.json`).
10. Search K-split per kernel shape and commit best split as record.
11. Sweep workgroup sizes (X) per kernel at capture time.
12. Record hot/cold timings separately in microbench runs.
13. Run forward+reverse order to cancel clock drift.
14. Capture CPU ref timings for each kernel shape.
15. Collect 576 timed samples plus 24 warmup/discard samples per config.
16. Use timestamp query path when available for GPU timing.
17. Fall back to cpu-batch timing when timestamp queries are missing.
18. Write dated `out/microbench-*.json` per run with full flags.
19. Run `patch.mjs` 6x5 reshuffled with seed `1381548739`.
20. Test scaled dispatches to check split wins hold at size.
21. Evaluate LOO sets to isolate per-kernel contribution.
22. Report paired medians alongside ratio-of-medians.
23. Note output near-tie where patched vs baseline outputs match within tolerance.
24. Repeat in-model run on Chrome 131 to check version inversion.
25. Run 0112-alone reshuffled to confirm single-kernel effect.
26. Analyze overlap between patched kernels to rule out double-counting.
27. Report timing in bare/gpu/kernel modes via `timing.mjs`.
28. Dump full dispatch tape with `trace.mjs` to `trace.json`.
29. Summarize top kernels with `analyze.mjs` 400-char text summary (`package.json:13`).
30. Apply split-pass correction (~1.5us/pass) to K-split timings.
31. Use MiB vs MB consistently and report effective GB/s for each kernel.
32. Map dispatches to 35 layers via layer map.
33. Cover 37 prefill pipelines never used in decode (`README:919`).
34. Render `trace.json` as timeline viewer beyond text summary.
35. Stamp browser/adapter/node/os/pkg/model/power/thermal/commit via `provenance.mjs`.
36. Refuse records missing browser version, adapter, or harness commit (`provenance.mjs:89-101`).
37. Backfill 13 legacy records with missing provenance fields.
38. Add T4 machine fields date/gpuInfo/order for Colab runs (`out/microbench-colab-t4-f32.json`).
39. Mark `harness_dirty` when working tree differs from committed harness.
40. Append each run median plus provenance to BENCH.md log.
41. Wrap results with `space/build.mjs` (`package.json:16`) for Space deploy.
42. Build notebook uploader with stable block ids for repro pastes.
43. Link records via blob/main URLs for versioned shader plus JSON replay.
44. Add inference badge with ref-note explaining measurement host.
45. Add glossary for split/K-split/workgroup/paired-median terms.
46. Add second Run CTA plus mobile polish for Space page.
47. Run nightly CI chain `pnpm capture -> timing -> trace -> analyze -> microbench -> patch` (`package.json:10-15`).
48. Fill vendor matrix Intel/AMD/Adreno/Metal/Linux plus f16 toggle trial.
49. Generalize `capture.mjs` / `trace.mjs` to WebLLM, ONNX Runtime Web, and Transformers.js.
50. Grow versioned corpus (`out/shaders/` plus `out/*.json` plus replay) and triage Dawn/Chrome inversions via linked repros.
