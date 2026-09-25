---
title: Kernel 0112
emoji: 🧮
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Run one LiteRT-LM WebGPU kernel on your GPU, with a control
tags:
  - webgpu
  - benchmark
  - gpu
---

# Kernel 0112

One of the 153 GPU programs Google's LiteRT-LM web engine (`@litert-lm/core` 0.17.1)
compiles to run Gemma 4 in the browser. This page runs it on your GPU as shipped,
then with a bigger workgroup (a control) and with each dot product split 16 and 32
ways, and checks every variant against a CPU reference. On an Apple M2 Max the
32-way split is 1.65x faster in Chrome 146, a browser inferred from the
adapter string (the two records behind that figure predate the provenance
field and carry only the adapter string `apple`/`metal-3`; the
provenanced `out/microbench-0112-2026-09-24T07-36-19-655Z.json` gives 1.638x for
the same kernel on the same machine with `provenance.browser`
`Chrome/146.0.7680.153`), and in Chrome 131 on the same machine the
single-dispatch harness measures the split as slower (0.84x, one run); both records
are committed. On a Colab T4 it is
1.53x, from a single run of an f32 transcription of the kernel, copied by hand from
the notebook output, because Chrome exposed no 16-bit float shaders there. In the
running model on the M2 Max, with this kernel and the three other quantized
matrix-vector kernels split deeper, decode goes from 57 to 71 tokens per second, 1.20x
as the median of the per-repetition pairings (1.18 to 1.33x) and 1.25x as the
ratio of the condition medians. Leaving one kernel out at a time puts the
largest marginal contributions within the four-kernel patch on the two 4-bit
kernels 0099 and 0105 and smaller ones on this one and 0113, which cannot be
ordered against each other because which of them ranks last follows from the
choice of aggregate. Leaving this one out of the four-kernel patch costs 0.240
ms per token by the condition medians, while patched alone, in an earlier
two-repetition sweep with a fixed condition order (`out/patch.json`, not
directly comparable), it saved 0.690 and 0.885 ms per token, near the roughly
0.96 ms its isolated gain predicts over its 40 dispatches per token; with the
other three patched its marginal contribution is smaller than its
stand-alone saving in that earlier sweep; whether that reflects overlapping
gains or the difference between the two sweeps is not established.

Needs WebGPU with 16-bit float shaders (`shader-f16`). Tested in Chrome. The page
checks on load and tells you if your browser or GPU does not offer it. No model
weights and no dataset are downloaded; the page loads its fonts from Google
Fonts. No result leaves your browser; copy the JSON if you want to share it. The copied JSON contains your browser's user-agent string and GPU name.

Method, harness and full record: https://github.com/abgnydn/litert-capture

The kernel source shown on the page was emitted by `@litert-lm/core` (Google LLC,
Apache License 2.0) and is reproduced with attribution; it is not covered by the
MIT license declared above for this Space. The Apache-2.0 license text is in
`LICENSE-Apache-2.0` and the kernel's provenance in `NOTICE`, both in this Space, and the page links the
license. Independent work, not affiliated with Google.
