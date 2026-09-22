# LiteRT-LM web shader capture

What Google's LiteRT-LM web engine (`@litert-lm/core` 0.17.1, Gemma 4 E2B)
runs on the GPU, captured from the shipped npm package by hooking
the WebGPU API before the engine loads; where each decoded token's time
goes; and one tuning change, measured in isolation on two GPU vendors and
inside the running model. Started 2026-09-21. Independent work, not
affiliated with Google. The 153 WGSL files under `out/shaders/` are
Google's, redistributed under Apache-2.0 (`out/shaders/NOTICE`); everything
else in this repository is MIT.

**Findings, in one paragraph.** Decode runs 1,304 GPU dispatches per token
and spends 16.0 of its 18.8 ms with the GPU running kernels, so it is
GPU-execution-bound. Nine weight-streaming kernels read about 741 MiB per token
at 81 GB/s after the split-pass correction (78 raw), a fifth of the M2 Max's
rated 400 GB/s
([Apple](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/)),
and four
quantized matrix-vector kernels carry 48.5% of GPU time while dispatching
only 6,144 to 12,288 threads each. Splitting each dot product 16 to 32 ways,
which multiplies the thread count, makes kernel 0112, the busiest 2-bit
matrix-vector kernel (13.7% of GPU time in the profile run), 1.65x faster on
an Apple M2 Max in Chrome 146 (control: a bigger workgroup at the same
thread count does nothing; the isolated measurement inverts on Chrome 131,
see "Browser version"), 1.53x on an NVIDIA T4 in one run of an f32
transcription of the same kernel, copied by hand from the notebook output,
and takes the running model from 53 to 67
tokens per second, with output identical but for one near-tie token
([Google's model card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
reports 73 tokens per second for this same web bundle on a newer M4 Max,
over 256 decode tokens after a 1,024-token prefill with a context length of
2,048 tokens; its 160.2 tok/s native figure is for the 2,583 MB general
bundle). The captured kernels all begin with
`enable f16`, and Chrome exposed no `shader-f16` on a Colab T4 through
headless Chrome for Testing, so that kernel set could not compile there. The
architecture of the model as executed (35 layers, KV sharing in layers 16
to 35, 2-bit FFN in the second half, 8-bit KV cache) was read off the
dispatch tape. The 0112 isolation and the in-model results reproduced in two
runs each; the capture, the timing and trace runs, the per-kernel profile,
the two 4-bit isolation runs and the T4 run are single runs and say so.

Try the isolated kernel on your own GPU:
https://huggingface.co/spaces/abgunaydin/kernel-0112

## Run

```bash
pnpm install                 # puppeteer 24.40, which downloads Chrome for Testing 146 and its headless shell (about 177 MB + 96 MB, roughly 550 MB on disk); see "Browser version" below
# 1. the model, 2,008,432,640 bytes, ungated, Apache-2.0
curl -L -C - -o www/gemma-4-E2B-it-web.litertlm \
  https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm
# 2. the experiments, each opens a Chrome window
node capture.mjs             # 1: record every shader and count dispatches   -> out/shaders, out/manifest.json
node timing.mjs bare         # 2: ms per token, CPU side vs waiting on GPU    -> out/timing-bare.json
node timing.mjs gpu          #    GPU time per pass                            -> out/timing-gpu.json
node timing.mjs kernel       # 3: GPU time per kernel                          -> out/timing-kernel.json, out/kernel-profile.json
node trace.mjs               # 4: shapes, bytes, memory, dispatch tape         -> out/trace.json
node analyze.mjs 400         #    joins 3 and 4 offline (400 = rated GB/s of this machine)
node microbench.mjs          # 5: kernel 0112 in isolation, controls included  -> out/microbench-<date>.json
node microbench.mjs 'microbench4.html?kernel=0105'   # the 4-bit kernels (also ?kernel=0099)
node patch.mjs 2             # 6: the change patched into the running model    -> out/patch.json
node colab/build-notebook.mjs   # 7: self-contained notebook for a Colab T4
node space/build.mjs         # the public page as a complete document, for the Hugging Face Space
```

Every experiment needs the model except 5 and 7. `capture.mjs`, `timing.mjs`,
`trace.mjs` and `patch.mjs` overwrite their records under `out/` (`out/shaders/`
included), so commit or copy `out/` before re-running them; `microbench.mjs`
keeps every run in a dated file. Chrome flags used by the drivers:
`--enable-unsafe-webgpu`, `--enable-features=Vulkan` (no effect on Metal),
`--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist` and,
for the timing harnesses, `--disable-dawn-features=timestamp_quantization`.
The first launch after `pnpm install` can time out at puppeteer's 30 s
connection deadline while macOS verifies the newly downloaded Chrome; the
retry works.

**Browser version.** All Mac records were made with Chrome for Testing
146.0.7680.153 (puppeteer 24.40; the adapter reports `apple` /
`metal-3`), which `pnpm install` now pins. The records do not store the
browser version; the `metal-3` adapter string is the only in-record proxy,
and `out/patch.json` and `out/kernel-profile.json` do not carry even that.
Chrome for Testing 131 (puppeteer 23's bundle; the adapter reports `apple` /
`common-3`) reverses the isolated result of experiment 5 on the same
machine: original 78.9 µs, 32-way split 93.1 µs, in three runs, while
Chrome 146 the same minute gave 61.0 and 36.7 µs (console numbers with no
committed record; a later run on Chrome 131 gave 81.1 and 94.0 µs). The
public page, which times batches of
48 dispatches rather than one timestamped pass per dispatch, still shows
about 1.5 to 1.6x across clicks on Chrome 131, also a console observation
with no committed record. So
the single-dispatch timestamp measurement is not portable across Chrome
versions on this adapter, and the ratio is; why is not established. The
T4 record was made with Chrome for Testing 153.

Output of experiment 1: `out/shaders/NNNN_<label>.wgsl` and
`out/manifest.json` (pipelines, per-pipeline dispatch counts, counter
snapshots at every streamed chunk).

## Result of the first run

`@litert-lm/core` 0.17.1, Gemma 4 E2B web bundle, Chrome for Testing
146.0.7680.153 via puppeteer, Apple M2 Max (adapter `apple` / `metal-3`),
cross-origin isolated. One cold run with the hooks installed, so timings are
indicative only.

- 153 shader modules and 153 compute pipelines; the harness logged all of
  them as already created when `Engine.create` returned (a console
  observation, not in the record). 652,895 bytes of WGSL. None of them carries a label.
- Every decoded token: exactly 1,304 dispatches, 23 compute passes, 24 submits,
  71 `writeBuffer` calls, 1 `mapAsync`. No variation over 47 tokens.
  19.1 ms median per token.
- 65 of the 153 pipelines are used in decode.
- The most-dispatched kernel (`0119`, 322 of the 1,304 dispatches, about 9 per
  layer over 35 layers) is a standalone clamp, quantize, round, dequantize on a
  texture: an activation fake-quant run as its own dispatch each time.
- All 153 shaders start with `enable f16`. 135 of 153 touch a texture; 114 of
  those write `rgba16float` storage textures. None uses subgroups or subgroup
  matrices, although the adapter offers both. The engine logs "Subgroups
  Enabled!" during startup; what that log refers to is not established here.
- `Engine.create` returned in 3.6 s with zero dispatches; the first message
  then took 23.2 s (6,076 dispatches, 433 `writeBuffer`). Cold, single run.

Reference points from
[zero-tvm's RESEARCH.md](https://github.com/abgnydn/zero-tvm/blob/main/RESEARCH.md)
(Phi-3, 32 layers, not the same model, measured on an M2 Pro): WebLLM 342
dispatches per token, zero-tvm 228.

## Where a token's time goes (second experiment, 2026-09-21)

`node timing.mjs bare` and `node timing.mjs gpu`. Same setup, fresh Chrome per
run, second ("warm") message, steady state from chunk 8, 67 tokens. Raw data in
`out/timing-bare.json` and `out/timing-gpu.json`.

| Per decoded token | Median |
|---|---|
| Total | 18.8 ms (53.2 tok/s); 18.9 ms with GPU timestamps on |
| CPU side in the page, issuing all 1,304 dispatches | 1.25 ms |
| Page waiting for the readback | 17.5 ms |
| GPU executing compute passes (timestamp queries) | 16.0 ms |
| GPU idle gap per token, between the first pass and the rest | 2.2 ms |

- **Decode is GPU-execution-bound on this machine.** 16.0 of the 18.8 ms of
  each token is the GPU running kernels. The hypothesis that browser per-dispatch overhead
  explains the web-vs-native gap is not supported here: the CPU side is 1.25 ms
  and the GPU bubble 2.2 ms.
- Dispatches are batched 64 per compute pass (19 full passes, then 39, plus
  two 1-dispatch passes and a 47-dispatch pass that is the slowest per
  dispatch at 24.5 µs). 19 of the 22 gaps between passes are about 0.001 ms;
  the first is 2.2 ms, and one adjacent pair overlaps by about 0.4 ms (0.02
  to 0.95 ms across tokens).
- Mean GPU time is 12.3 µs per dispatch (16.03 ms / 1,304). The two single-dispatch
  passes take about 4 µs, a rough floor for a tiny dispatch. At that floor the
  322 fake-quant dispatches cost roughly 1.3 ms per token, so fusing them is
  worth up to about 8%. Estimate; the per-kernel measurement below
  gives 0.65 ms after correcting for its own overhead.
- The counting hooks of the first experiment cost about 0.3 ms per token
  (19.1 vs 18.8 ms), so its numbers stand.
- First-message latency: 265 ms and 150 ms in these fresh-profile runs, against
  23.2 s in the first-ever run. All 153 pipelines are created with the
  synchronous `createComputePipeline`. Consistent with a one-time compile that
  an OS-level Metal shader cache then absorbs; not verified by clearing it.
- Observation only: the warm countdown skipped "twenty" in both runs
  ("twenty-one, nineteen"). The sixth experiment shows it is a near-tie that
  flips with f16 accumulation order.

This is one machine, with Metal and unified memory. On discrete GPUs with faster kernels and
costlier dispatch paths (Vulkan, D3D12) the split may differ.

## GPU time per kernel (third experiment, 2026-09-21)

`node timing.mjs kernel`. For 7 consecutive tokens of the warm message every
dispatch is moved into its own timestamped compute pass, with pipeline and bind
groups replayed. Full table in `out/kernel-profile.json`.

Validity checks, all passed: 0 WebGPU validation errors, warm output
byte-identical to the unmodified run, pipeline ids line up with
`out/manifest.json`, all 7 tokens split into exactly 1,304 single-dispatch
passes. Splitting inflates the per-token GPU sum from about 16.0 to 18.0 ms,
and the inflation is per pass, about 1.5 µs each (2.0 ms / 1,304): 3% on a
59 µs kernel, 44% on a 3.5 µs one. The table below is raw, so small-kernel
shares are overstated; `node analyze.mjs 400` prints corrected figures
(fake-quant 0119: 1.15 ms raw and 0.65 ms corrected per token, which is 4%
of the 16.03 ms token; the kernels under 6 µs: 9.7% of
GPU time against 14.2% raw; the 24-or-25 count of those kernels is explained under
the table).

| Kernel (shader file) | What it is | Per token | Median each | Share |
|---|---|---|---|---|
| 0099 | 4-bit weight matrix times vector, 16-way workgroup split | 84 | 23.7 µs | 15.2% |
| 0112 | 2-bit weight matrix times vector, 4-way split | 40 | 58.9 µs | 13.7% |
| 0105 | 4-bit weight matrix times vector, 4-way split | 37 | 57.4 µs | 12.5% |
| 0113 | 2-bit weight matrix times vector, 32-way split | 20 | 60.6 µs | 7.1% |
| 0119 | activation fake-quant, elementwise | 322 | 3.5 µs | 6.9% |
| 0134 | vocabulary projection, 2-bit, 262,144 outputs | 1 | 645 µs | 3.7% |

- **The four quantized matrix-vector kernels are 48.5% of decode GPU time in
  181 dispatches** (2-bit 20.8%, 4-bit 27.7%). These four kernels hold
  enough of the token that a web-to-native comparison would have to measure
  them; no native comparison was run here.
- **Dispatch fusion has a low ceiling here.** The 24 kernels with a median
  under 6 µs account for 624 of the 1,304 dispatches but only 14.4% of GPU
  time raw. (Grouped per kernel median; `analyze.mjs` groups per dispatch
  position and prints 25 kernels, 625 dispatches, 14.2% raw and 9.7%
  corrected.) After the split-pass correction the share falls to about 9.7%,
  the per-dispatch figure that `analyze.mjs` prints. The fake-quant kernel
  measures 1.15 ms per token raw, 0.65 ms corrected.
- Shares in the table are of the 17.40 ms split-pass profile total in
  `out/kernel-profile.json` (the sum of the per-pipeline median totals); the
  18.0 ms split-pass sum quoted above is the median per-token sum from the
  same run, a different aggregate, so the two do not divide into each other.
- How the matrix-vector kernels are written (read from `0112`):
  - Weights live in a `texture_2d<u32>`, activations in `rgba16float`
    textures; both are fetched with `textureLoad` in the inner loop.
  - 2-bit fields (0112, 0113) are unpacked with f16 arithmetic (multiply by
    0.25, `floor`, subtract, multiply by 4); the 4-bit kernels (0099, 0105)
    use integer shifts and masks (`f16((w.y >> 4u) & 15u)`).
  - Per-output-channel scale, symmetric zero point (2 for 2-bit, 8 for 4-bit).
  - Each output's dot product is split across 4, 16 or 32 invocations of a
    workgroup and summed with a `workgroupBarrier()` reduction in
    `var<workgroup>` memory. The kernels use no subgroup operations.
  LiteRT's GPU backend targets mobile GPUs as well as desktop; these shapes
  are consistent with a tuner whose search space is shared across those
  targets. Not established.

## Shapes, bytes and bandwidth (fourth experiment, 2026-09-22)

`node trace.mjs` records every texture and buffer, each bind group's
resources, the latest uniform contents per location and, for three decode
tokens, the full dispatch sequence with the resources each dispatch saw.
`node analyze.mjs 400` joins that by position with the per-kernel timings
(the pipeline sequence is identical across runs, checked) and the shader
sources. Raw: `out/trace.json`.

Consistency check (from `out/trace.json`; not printed by a script): the
4-bit weight textures resident on the GPU total 335 MB and the per-token
4-bit reads sum to the same 335 MB, all 145 textures read exactly once;
2-bit weights 270 MB resident against 270 MB read (the other 7 MB of
`rgba8uint` textures are attention masks and quantized K/V staging, not
weights); 32-bit-packed 39 against 39 MB. Every weight texture is read
exactly once per token.

**Bandwidth.** Nine kernels stream a weight matrix once per dispatch (0098,
0099, 0100, 0105, 0106, 0107, 0112, 0113, 0134): **740.6 MB per token in 277
dispatches, 10.0 ms of split-pass GPU time (9.6 ms after the per-pass
correction): 78 GB/s raw, 81 GB/s corrected, about 20% of the M2 Max's
rated 400 GB/s** ([Apple's M2 Pro and M2 Max announcement](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/);
this is a 30-core M2 Max).
Units, here and throughout: byte counts are MiB (2^20 bytes, as
`analyze.mjs` prints them; 740.6 MB is 776,601,600 bytes) and GB/s is
decimal (10^9 bytes per second), so 740.6 MB in 9.57 ms is 81 GB/s. The
download sizes in the Run section, and the bundle sizes quoted from Google's
[model card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
(2,008 MB and 2,583 MB, the card's own on-disk figures), are decimal MB.
The best kernel measured here reaches 195 GB/s in one run, 49% of rated
(0099 at 2 × 128, fifth experiment), 2.4x the weight-kernel rate on this
hardware; whether a different kernel structure can approach the rated
figure is not established. Over the whole 18.8 ms token that is 41 GB/s
effective; at the rated bandwidth the weights would take 1.9 ms. A linear
fit over the 276 texture-weight dispatches gives
`µs = 6.2 + bytes / (89 GB/s)`, R² 0.98, but it pools 0.2 to 0.4 MB kernels
at 20 to 45 GB/s with 4.5 MB kernels at 80, and about 1.5 µs of the intercept is
measurement overhead. The fit describes matrices of 1.5 MB and up: about
4.7 µs of fixed cost, then about 89 GB/s. It does not extend to the 0.2 to
0.4 MB kernels the pooling hides.

Four other kernels bind a large buffer and are attention kernels over the
KV cache (0101, 0103, 0108, 0110): all four use zero point 128, and
the binding named `weights_buffer` holds the 8-bit KV cache (0101/0103: a
128 KB (131,072-byte) window of 512 positions × 256 channels for local
attention; 0108/0110: 2 MB slices of an 18 MB buffer for global attention).
0110's loop bound, and 0108's output bound, are the current sequence
length read from `params_buffer`, so those two read only the slice the
sequence selects; 0101 and 0103 are bounded by fixed uniforms and cover
their whole 128 KB window each dispatch (7.0 MiB per token for the two
together). None is in the weight set because the binding holds the KV cache
rather than model weights. Only the vocabulary projection (0134, 96 MB from a storage
buffer, 65k threads, 156 GB/s) exceeds 91 GB/s. Texture-weight kernels of
1.5 MB and up sit at 57 to 91 GB/s with 4k to 12k threads; the 0.2 to
0.4 MB ones at 20 to 45 GB/s. The fifth experiment tests thread count directly, on one of these kernels.

For comparison, zero-tvm's hand-written FFN kernels on this machine reach
higher bandwidth at larger shapes (Qwen3.5-9B) in
[zero-tvm's BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md).
Its numbers are not quoted here: the error notes in
[BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md) give a
160 GB/s figure for `ffn_down` as a warm-cache reading and its published
`ffn_down` row carries the same figure after the rotation fix, so which
protocol that number belongs to is not clear from the file. That file also
records MLX's kernels beating
zero-tvm's on every shape. The finding below does not depend on it.

**Per-shape table** (median GPU µs per dispatch, split-pass measurement):

| Kernel | Per token (this shape) | Weights each | Matrix (in → out) | µs | GB/s |
|---|---|---|---|---|---|
| 0112 | 40 | 4.5 MB | 1536 → 12288, 2-bit | 58.9 | 80 |
| 0105 | 30 | 4.5 MB | 1536 → 6144, 4-bit | 58.0 | 81 |
| 0113 | 20 | 4.5 MB | 12288 → 1536, 2-bit | 60.5 | 78 |
| 0099 | 15 | 4.5 MB | 6144 → 1536, 4-bit | 59.3 | 80 |
| 0099 | 28 | 1.5 MB | 1536 → 2048 (q), 4-bit | 20.4 | 77 |
| 0099 | 28 | 1.5 MB | 2048 → 1536 (o), 4-bit | 23.8 | 66 |
| 0134 | 1 | 96 MB | 1536 → 262144 vocab, 2-bit, buffer | 645 | 156 |
| 0098 | 1 | 13.1 MB | 1536 → 8960 (35 × 256), 32-bit packed | 151 | 91 |

**Architecture as executed.** Google documents the mixed 2/4/8-bit scheme on
the model card and the Gemma 4 architecture in its own materials. What
follows is what the dispatch tape shows (model dims from the uniforms;
slices are groups of 4 channels), which agrees with those where they
overlap; the value is the cross-check on the capture method, and the details
the tape adds: per-layer time, the 512-token local window, and which layers
compute K and V. 35 layers, d_model 1536, one FFN-down per layer confirms
the count.

- Layers 1 to 15, FFN gate and up 1536 → 6144 at 4-bit, down at 4-bit.
  Layers 16 to 35, FFN gate and up 1536 → 12288 at 2-bit, down at 2-bit.
  Wider and lower-precision in the second half; same 4.5 MB per matrix.
- Local attention (all layers except every fifth): q 1536 → 2048 (8 heads of
  256), k and v 1536 → 256 each (one KV head), o 2048 → 1536.
- Global attention at layers 5, 10, 15, 20, 25, 30, 35: q 1536 → 4096, k and
  v 1536 → 512 each, o 4096 → 1536, plus two attention kernels (0108, 0110)
  over the 8-bit KV cache that read the current sequence length from
  `params_buffer` (0110 as its loop bound, 0108 as its output bound).
- KV sharing: layers 16 to 35 have no k or v projection at all. Only
  layers 1 to 15 compute K and V; the rest reuse cached KV from earlier
  layers. 24 k/v dispatches per token (12 local layers × 2) plus 6 (3 global
  × 2) confirms it.
- Per-layer embeddings: once per token a 1536 → 8960 projection (35 × 256),
  35 small gathers, then a 1536 → 256 gate (0106) and a 256 → 1536 projection
  (0107), 35 pairs per token; one pair sits in the epilogue segment rather
  than in layer 1's.
- Two attention kernels per local layer (0101, 0103; 28 of each per token)
  over a 128 KB (131,072-byte) window of the 8-bit KV cache (512 positions × 256 channels):
  the local sliding-window attention, scores and weighted sum. So the KV
  cache is stored at 8 bits with a 512-token local window.
- Embedding table 262144 × 1536 at 2-bit, 96 MB, with a separate zero-point
  buffer (0115). The output head reads a 96 MB matrix (0134), the same size,
  so probably tied.
- Time per layer (split-pass): local own-KV 0.44 ms, global own-KV 0.53,
  local shared-KV 0.39, global shared-KV 0.46; layer 1 with prologue 0.93;
  epilogue (final norm, vocab projection, sampling) 1.34.
- Resident GPU memory 1,192 MB: 1,001 MB in 367 textures (350 MB rgba16float
  activations; 335 MB 4-bit weights; 270 MB 2-bit weights; 39 MB 32-bit
  packed; 7 MB attention masks and K/V staging) and
  190 MB in 2,522 buffers, which is where the 8-bit KV cache lives.

Prefill (cold message, about 20 prompt tokens): 6,076 dispatches over 102
pipelines, 37 of which never run in decode. Not profiled.

## Kernel 0112 in isolation: thread count is the variable that moves it (fifth experiment, 2026-09-22)

`node microbench.mjs`. Runs the captured 2-bit matrix-vector kernel 0112
verbatim on its real decode shape (1536 → 12288, 4.5 MB packed weights,
workgroups of 16 × 4, dispatch 192) and against minimal text edits of the
same source. Every variant is checked against a CPU reference (max error
6e-3 on outputs of magnitude 2, shrinking to 2e-3 as the split widens, which
is f16 accumulation behaviour) and against each other. "Cold" rotates
through 24 different 4.5 MB matrices (108 MB) so weights come from DRAM;
"hot" reuses one.

Two controls keep workgroup size and total thread count apart, since a
variant that changes both at once cannot say which one moved the number:
64 × 4 (256 invocations, the same 12,288 threads) isolates workgroup size,
and 2 × 32 (64 invocations, 98,304 threads) isolates thread count. Whatever
is timed first reads about 17% slow (70.7 µs for the first-timed variant in
`out/microbench-run3.json` against 60 µs warmed), so the harness runs a
discarded warm-up variant, times every variant forward
and again in reverse, pools the samples, and writes every run to a
timestamped file (`out/microbench-<date>.json`). Raw: two runs,
`out/microbench-2026-09-22T03-06-35-136Z.json` and `...03-06-37-804Z.json`.

| Variant | Workgroup | Workgroups | Threads | Cold µs, run 1 / 2 | GB/s | Hot µs |
|---|---|---|---|---|---|---|
| orig, verbatim | 16 × 4 = 64 | 192 | 12,288 | 60.5 / 61.3 | 77 | 43 to 46 |
| wg256-same, control | 64 × 4 = 256 | 48 | 12,288 | 61.2 / 60.0 | 78 | 45 to 48 |
| split16-64 | 4 × 16 = 64 | 768 | 49,152 | 40.1 / 40.2 | 117 | 33 |
| split32-64 | 2 × 32 = 64 | 1,536 | 98,304 | 36.8 / 36.7 | 128 | 34 to 35 |
| split16-256 | 16 × 16 = 256 | 192 | 49,152 | 37.8 / 37.7 | 125 | 34 to 35 |
| buf-4, weights from a storage buffer | 16 × 4 = 64 | 192 | 12,288 | 55.4 / 55.0 | 85 | 42 to 44 |

GB/s is computed from the two-run mean of the cold medians rounded to 0.1 µs
(4,718,592 bytes over 60.9 µs → 77 GB/s for orig), the same convention as
the public page's reference panel. Per-position medians agree forward and reverse to within 2.3 µs, with no
consistent sign (orig reads 0.6 µs slow in one run and 2.2 µs fast in the
other), so the systematic first-slot penalty is gone; the residual is
run-to-run noise of the same size as the 1 µs spread between the two whole
runs. 576 cold and 24 hot single-dispatch samples per number.

- **The kernel dispatches 12,288 threads for a 4.5 MB read. Splitting each
  output's dot product 16 or 32 ways multiplies the thread count by 4 or 8
  and takes it from 77 to 117 to 128 GB/s (from the two-run mean µs): 1.65x
  (60.9 → 36.8 µs), output
  identical to f16 rounding.** The control at the same thread count does not
  change the time; 256 invocations per workgroup on its own changes nothing.
  The fastest configuration, 2 × 32, keeps the 64-invocation workgroup the
  tuner already picks and is the workgroup shape of Google's own kernel 0113.
- At the same 49,152 threads, 16 × 16 (256 invocations) is about 6% faster
  than 4 × 16 (64 invocations): a small secondary effect of workgroup size.
- The texture fetch path is secondary: a storage buffer helps by about 10%
  at 12,288 threads and, in `out/microbench-run3.json` (a record from an
  earlier version of this harness with buffer variants at higher thread
  counts; that variant set is not in this repository and `microbench.mjs`
  no longer reproduces it), made no difference once the thread count was
  raised.
- The in-model data agrees. The four 4.5 MB kernels run 4-, 4-, 16- and
  32-way splits (0112 16 × 4, 0105 16 × 4, 0099 4 × 16, 0113 2 × 32) and all
  land at 57 to 61 µs, because each dispatches only 6,144 (0099, 0105: 96
  workgroups) or 12,288 (0112, 0113: 192 workgroups) threads. 0113
  (12288 → 1536) already splits 32 ways and needs a 256-way split to reach
  98k threads.
- **The 4-bit kernels, in isolation** (`node microbench.mjs
  'microbench4.html?kernel=0105'` and `?kernel=0099`, same discipline, CPU
  reference from the nibble layout read off the WGSL: 16-bit component k of
  a texel holds four nibbles, nibble c is output channel c, zero point 8):

  | Kernel | Variant | Workgroup | Threads | Cold µs | GB/s | max err vs f32 ref |
  |---|---|---|---|---|---|---|
  | 0105 (1536 → 6144) | as shipped | 16 × 4 | 6,144 | 59.0 | 80 | 1.1e-2 |
  | 0105 | split32-64 | 2 × 32 | 49,152 | 28.1 | 168 | 4.0e-3 |
  | 0105 | split16-256 | 16 × 16 | 24,576 | 28.8 | 164 | 4.5e-3 |
  | 0099 (6144 → 1536) | as shipped | 4 × 16 | 6,144 | 61.9 | 76 | 1.9e-2 |
  | 0099 | split32-64 | 2 × 32 | 12,288 | 39.2 | 120 | 1.3e-2 |
  | 0099 | split64-256 | 4 × 64 | 24,576 | 29.4 | 160 | 6.8e-3 |
  | 0099 | split128-256 | 2 × 128 | 49,152 | 24.2 | 195 | 6.0e-3 |

  Both pass (limit 0.05; reference maxima 2.6 and 5.3): in one interleaved
  run each, 2.1x for 0105 at 2 × 32 and 2.55x for 0099 at 2 × 128. The
  deeper splits are closer to the f32 reference than the shipped kernels,
  because f16 partial sums are shorter; that is the sign of the one-token
  difference seen in the model. In that same single run, 0099 at 2 × 128
  reaches 195 GB/s, the highest of any kernel here, so the deeper reduction
  tree costs little. One interleaved run per kernel, so this table is a
  single run each. Raw:
  `out/microbench-0105-*.json`, `out/microbench-0099-*.json`.
- Warmed and interleaved, the original measures 60 to 61 µs, the same as in
  the running model (58.9 µs split-pass).
- Hot and cold converge at 98k threads (34 vs 37 µs): the 128 GB/s of the
  32-way split is where this kernel structure stops gaining from thread
  count on this GPU. Nothing measured here goes past it without changing the
  kernel structure.
- Estimated effect on a token if the in-model gain matched isolation: the
  four 4.5 MB kernels are about 6.2 ms per token; at 1.65x that saves about
  2.4 ms of 18.8 (53 → about 61 tok/s), more if the 1.5 MB q and o kernels
  gain similarly. Estimate only.

The finding is reproducible with `node microbench.mjs`: their kernel,
arithmetic unchanged, with a deeper K split and the matching loop stride and
reduction depth, runs 1.65x faster on an Apple M2 Max, with the control that
rules out workgroup size. The sixth experiment measures it inside the
running model and the seventh on a second GPU vendor.

## In the running model (sixth experiment, 2026-09-22)

`node patch.mjs 2`. `www/patch.html` hooks `createComputePipeline`: when the
engine creates a pipeline from one of the four matrix-vector shaders (matched
by exact source text), the page rewrites the K split in the WGSL, swaps the
workgroup constants, and scales every `dispatchWorkgroups` of that pipeline
by oldWgX / newWgX so the same output slices are covered. Arithmetic
untouched. Four patch sets, each in a fresh Chrome, twice, same two prompts
as the timing harness, steady state from token 8 of the second message.
Raw: `out/patch.json`.

| Patched kernels | New workgroup (threads) | ms per token, run 1 / 2 | tok/s | Scaled dispatches per run |
|---|---|---|---|---|
| none | – | 18.36 / 18.90 | 53 to 54 | 0 |
| 0112 | 2 × 32 (98k) | 17.67 / 18.02 | 56 to 57 | 6,320 |
| 0112, 0105 | 2 × 32 (98k, 49k) | 16.35 / 16.64 | 60 to 61 | 12,320 |
| all four | + 0099 2 × 128 (49k), 0113 1 × 256 (98k) | 14.85 / 14.68 | 67 to 68 | 28,960 |

- **Decode goes from 53 to 67 tokens per second, about 1.26x (1.24 to 1.29x
  across the two run pairings; 18.4 to 18.9 → 14.7 to 14.9 ms per token),
  with all four kernels split
  deeper.** Zero GPU validation errors. The per-kernel prediction held:
  patching 0112 alone saved 0.7 to 0.9 ms against a predicted 40 × 22 µs =
  0.9 ms; adding 0105 saved a further 1.3 to 1.4 ms against 37 dispatches × the
  isolated saving of 59.0 − 28.1 = 31 µs, 1.1 ms.
- The estimate of "53 → about 61" covered only the four 4.5 MB shapes; the
  0099 and 0113 pipelines also serve the 1.5 MB q/o and down projections,
  which gained too.
- **Output.** The cold message is byte-identical for every patch set. The
  warm message is identical with 0112 patched; with 0105 or more patched it
  differs in exactly one token: the unpatched model skips "twenty"
  ("twenty-one, nineteen"), the patched one does not. Everything else in the
  unpatched message's 268 characters is identical. This is f16 accumulation order at a near-tie
  (a 32-way partial sum rounds differently from a 4-way one; in isolation
  the three rewrites used here differ from the shipped kernels by up to
  1.9e-2 on outputs of magnitude 2 to 5 (the largest disagreement of any
  measured rewrite is 2.1e-2, for 0099's 4 × 64 split, which the patch does
  not use), `vsOrig` in the `out/microbench-*.json` records, while each is
  closer to the f32 CPU reference than the kernel it replaces). A kernel
  with an arithmetic error would produce outputs unrelated to the
  reference. A near-tie can fall either way, so this is not
  evidence that the patch improves quality.
- Reference level: 0099 and 0105 rewrites were then checked in isolation
  against a CPU reference and pass (table in the fifth experiment). 0113's
  1 × 256 rewrite has not been; its evidence is the identical cold text and
  the same one-token warm difference as the 0105 set.

## On an NVIDIA T4 (seventh experiment, 2026-09-22)

`colab/kernel-0112-t4.ipynb`, run on a Colab Pro T4 through headless Chrome
for Testing 153 (Vulkan 1.4, NVIDIA driver 580.82.7), driven from my
desktop Chrome. Record: `out/microbench-colab-t4-f32.json`, transcribed by
hand from the notebook's printed output (the notebook itself writes
`microbench-colab.json` with the driver's field names; that file was not
copied back). Three things had to be fixed on the way
and are now in the notebook builder and the page: Colab's Ubuntu 24.04 names
the ALSA package `libasound2t64` (one wrong name aborts the whole apt
install, so no Vulkan userspace and "NO VULKAN DEVICE"); headless Chrome on
Linux returns null from `requestAdapter({powerPreference:
'high-performance'})` while the plain call works; and one warm-up round is
not enough on a T4 (first-measured variants read 400 µs against 70 µs once
the clocks are up), so the harness now warms up for ten rounds. The
engine's own device setup, `createDefaultWebGpuDevice()` in
`dist/litertlm_web.js` (lines 50 to 57 of 0.17.1), makes that same
high-performance request and throws `No GPU adapter found.` when it returns
null, with no fallback to the plain call; on this configuration the engine
would stop there, before the `shader-f16` question arises. Not tested with
the engine itself.

**Finding 1: no `shader-f16` on this adapter.** Chrome 153 exposes
`subgroups`, `subgroup-size-control`,
`chromium-experimental-subgroup-matrix`, `timestamp-query` and 20 other
features on the T4 (24 in all, listed in the record), but not `shader-f16`;
Chrome 131 also withheld `shader-f16` (its full feature list was not
recorded). This although Vulkan reports
`shaderFloat16`, `storageBuffer16BitAccess` and
`uniformAndStorageBuffer16BitAccess` all true. Every captured LiteRT-LM
web kernel begins with `enable f16`, and the shipped package requests
`shader-f16` and `subgroups` only when the adapter offers them. From
`dist/litertlm_web.js` in the `@litert-lm/core` 0.17.1 tarball, lines 46 to
49 and 64 to 69 (the jsdelivr `+esm` build the harness imports minifies the
same code):

```js
const DESIRED_WEBGPU_FEATURES = [
    'shader-f16',
    'subgroups',
];
```

```js
    const requiredFeatures = [];
    for (const feature of DESIRED_WEBGPU_FEATURES) {
        if (adapter.features.has(feature)) {
            requiredFeatures.push(feature);
        }
    }
```

So on this configuration the captured kernel set cannot compile; whether the
engine has another shader path for such adapters is not established here.
Related to
[google-ai-edge/LiteRT-LM#1881](https://github.com/google-ai-edge/LiteRT-LM/issues/1881),
an open issue on their repo reported by another user (FP16 shaders on
Pascal, in the WebGPU backend of the native binary): both paths reach the
GPU through Dawn, there the native accelerator emitting f16 shaders for an
adapter that reports `shaderFloat16` false, here Chrome withholding the
feature for an adapter that reports it true. The run launched Chrome
with `--enable-unsafe-webgpu` and
`--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist`, so
the adapter blocklist was already off. Scope: Colab's container and
headless Chrome for Testing;
whether desktop Linux Chrome on NVIDIA behaves the same is not established.
Dawn's Vulkan backend has a documented gate for exactly this (Dawn main at
commit 42a517fc, read 2026-09-22): its adapter feature validation refuses `shader-f16` on
NVIDIA adapters unless the adapter-stage toggle
`vulkan_enable_f16_on_nvidia` ("Enables F16 on Nvidia GPUs with Vulkan") is
enabled, with the message "Feature %s is not yet supported on Nvidia GPUs"
and a TODO referencing crbug.com/42251215 that reads "Investigate f16 CTS
test failures to enable on Nvidia" (the gate and the TODO in
`src/dawn/native/vulkan/PhysicalDeviceVk.cpp`; the toggle's registration in
`src/dawn/native/Toggles.cpp`). This run
did not set that toggle, so the missing feature is consistent with that
gate (Dawn's Vulkan enable path for the feature also requires
`shaderInt16`, which this run did not record). Dawn main defaults the
toggle off and refuses `shader-f16` on NVIDIA
unless something enables it, so on the Dawn side the toggle is the control
point. Issue #1881 names the same toggle from the native side and gives the toggle
being hardcoded on for all NVIDIA GPUs as the likely cause, while its own
notes record that patching the toggle string has no effect because Dawn
registers toggles by enum, and that the driver already reports
`shaderFloat16` false. On that GPU, a GTX 1060, the issue's conclusion
holds by a different route: Dawn's enable path never reaches the toggle:
`PhysicalDeviceVk.cpp` enables `ShaderF16` only when `shaderFloat16`,
`shaderInt16` and `storageBuffer16BitAccess` are all true, and the NVIDIA
toggle check sits in the later validation path. On Dawn's WebGPU adapter
path the feature is therefore unavailable on that GPU regardless of the
toggle, which is consistent with the issue's conclusion that f16 shaders
reach a GPU that cannot run them without a capability check; the
accelerator in the issue reaches Dawn by another route, which this
repository did not inspect, and whether anything enables the toggle there
is not established here.

**Finding 2: the thread-count change also speeds the kernel up on Turing.** Because f16 is
unavailable, this run used an f32 transcription of kernel 0112 (`f16` →
`f32` throughout, scales uploaded as f32; the structure is unchanged and
the bytes are a transcription rather than Google's). CPU reference passes at
1.0e-3 on outputs of magnitude 2.2.

| Variant | Workgroup | Threads | Cold µs (p10 to p90) | GB/s | Forward / reverse position |
|---|---|---|---|---|---|
| orig | 16 × 4 = 64 | 12,288 | 71.8 (61 to 74) | 66 | 73.7 / 61.9 |
| wg256-same, control | 64 × 4 = 256 | 12,288 | 78.0 (68 to 80) | 61 | 79.9 / 69.0 |
| split16-64 | 4 × 16 = 64 | 49,152 | 49.5 (43 to 52) | 95 | 51.3 / 44.4 |
| split32-64 | 2 × 32 = 64 | 98,304 | 46.9 (39 to 49) | 101 | 48.0 / 39.8 |
| split16-256 | 16 × 16 = 256 | 49,152 | 46.0 (42 to 53) | 103 | 52.5 / 42.7 |
| buf-4 | 16 × 4 = 64 | 12,288 | 66.3 (65 to 70) | 71 | 67.1 / 65.7 |

- 32-way split: **1.53x** pooled, 1.54x forward, 1.56x reverse, from this
  single Colab run of the f32 transcription, copied by hand from the
  notebook output. The bigger
  workgroup at the same thread count reads 78 µs against the original's 72.
  Same shape as the M2 Max result, on a different vendor and API.
- 66 → 101 GB/s in this single f32 run, copied by hand from the notebook
  output, against the T4's rated 300 GB/s
  ([NVIDIA's T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf),
  "16 GB GDDR6" / "300 GB/sec"): 22% → 34%. The T4 is a
  70 W part and keeps speeding up through the run (five of the six variants
  read 13 to 19% faster in the reverse position; the storage-buffer variant
  only 2%), so the pooled medians are conservative; the ratios are stable
  across positions.
- Storage-buffer weights help by 8% at 12k threads here (66 vs 72 µs).
- Caveats: f32 not f16; one Colab session; Chrome for Testing, headless, in a
  container. A desktop NVIDIA run with f16 is the next check.

## Public page (2026-09-22)

`www/kernel-0112.html` runs the same experiment in any visitor's browser (no
download: it generates 24 random 4.5 MB matrices locally, computes a CPU
reference, times batches of 48 dispatches with GPU timestamps or 480 from
JavaScript, runs a discarded warm-up and then every variant forward and in
reverse) and explains what is measured and how, with the kernel source shown
under Apache-2.0 attribution and a link to the license. Variants: original,
the 64 × 4 control, 16-way and 32-way splits. Public at
https://huggingface.co/spaces/abgunaydin/kernel-0112 (a static Space built
by `space/build.mjs`); results are copied as JSON by the visitor, not
stored. A later version on gpubench.dev will collect them. Runs on the M2
Max through
the page: about 1.5 to 1.8x across sessions on Chrome 146, with the control
tracking the original to within a few percent either way, and about 1.5 to
1.6x across clicks in Chrome 131, both console observations with no
committed record, all variants
matching the reference, GPU timestamps available without flags. They move with other
tabs, thermal state and power mode, as the page's own note says: on a
contended or battery-powered GPU the page reads lower, sometimes close to
1.0x, with every variant still matching the reference. The page requests a
fresh adapter for every run, since a `GPUAdapter` creates one device only.

## Not established

- Whether the fake-quant dispatches could be fused into neighbouring kernels.
- Whether the K-split gain holds with f16 on NVIDIA (the T4 run was f32
  because Chrome exposed no shader-f16 there), on Intel and on AMD, and on
  desktop Linux Chrome rather than headless in a container. On the T4 in f32
  it does (seventh experiment); in the running model on this Mac it does
  (sixth).
- Whether `--enable-dawn-features=vulkan_enable_f16_on_nvidia` restores
  `shader-f16` on the T4 adapter, and whether Finding 2 then holds in f16
  rather than f32. The run did not set that toggle.
- A CPU-reference check in isolation for the rewritten 0113 (2-bit,
  12288 → 1536, 1 × 256); 0099 and 0105 are done.
- The best split per kernel: in one run each, 0099 gains from 2 × 32
  (120 GB/s) through 2 × 128 (195 GB/s), so the in-model patch's 2 × 128
  was a good choice;
  finer sweeps per shape are untried.
- Why the first timed slot reads about 17% slow (GPU clock ramp is the
  guess). The harness and page avoid it; the cause is not characterised.
- What a zero-tvm-style kernel (integer unpack, subgroup reduction, no
  textures) reaches on the same shape; the harness has the data layout and
  reference to drop one in.
- The rest of the KV-cache layout: 0094, 0095, 0096 read 3 MB textures per
  dispatch and are not identified.
- Prefill kernels (37 pipelines never used in decode).
- Native LiteRT-LM GPU speed on this same M2 Max, for a real web-to-native
  ratio. [Google's model card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
  gives 160.2 native vs 73 web tok/s on an M4 Max, over 256 decode tokens
  after a 1,024-token prefill with a context length of 2,048 tokens. The
  web row is this bundle (the card's
  footnote links it, 2,008 MB, text-only); the native row is the 2,583 MB
  general bundle, so the two rows are not the same model.
- Whether a non-f16 shader path exists for adapters without `shader-f16`.
- Load time: `Engine.create` took 3.6 s in the capture run and 2.6 to 3.2 s
  in every later fresh run (`timing-bare` 3.05 s, `timing-gpu` 2.82 s,
  `timing-kernel` 2.79 s, the eight `out/patch.json` runs 2.65 to 3.16 s);
  whether the 23 s first message is weight conversion, pipeline warm-up or
  both.

## Publishing

`@litert-lm/core` declares Apache-2.0 on npm (the repository is Apache-2.0
too), so the license permits redistribution with attribution and a copy of
the license text.
The tarball ships no LICENSE or NOTICE file, so the license text is
reproduced next to the shaders. The kernel text is emitted at run time by the engine;
the page says so rather than asserting a license on generated output.

Running `timing.mjs` overwrites `out/timing-*.json`; back up before
re-running. `microbench.mjs` writes only dated files and keeps every run;
`out/microbench-run3.json` is a record from an earlier version of the
harness whose variant set `microbench.mjs` no longer reproduces, kept as
the record behind the storage-buffer note in the fifth experiment.
