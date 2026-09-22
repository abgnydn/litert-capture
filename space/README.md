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
models:
  - litert-community/gemma-4-E2B-it-litert-lm
---

# Kernel 0112

One of the 153 GPU programs Google's LiteRT-LM web engine (`@litert-lm/core` 0.17.1)
compiles to run Gemma 4 in the browser. This page runs it on your GPU as shipped,
then with a bigger workgroup (a control) and with each dot product split 16 and 32
ways, and checks every variant against a CPU reference. On an Apple M2 Max the
32-way split is 1.65x faster in Chrome 146; the single-dispatch harness
inverts on Chrome 131, and this page's batched method still favours the split
there, as a console observation with no committed record. On a Colab T4 it is
1.53x, from a single run of an f32 transcription of the kernel, copied by hand from
the notebook output, because Chrome exposed no 16-bit float shaders there. In the
running model on the M2 Max, decode goes from 53 to 67 tokens per second.

Needs WebGPU with 16-bit float shaders (`shader-f16`). Tested in Chrome. The page
checks on load and tells you if your browser or GPU does not offer it. No model
weights and no dataset are downloaded; the page loads its fonts from Google
Fonts. No result leaves your browser; copy the JSON if you want to share it.

Method, harness and full record: https://github.com/abgnydn/litert-capture

The kernel source shown on the page was emitted by `@litert-lm/core` (Google LLC,
Apache License 2.0) and is reproduced with attribution; it is not covered by the
MIT license declared above for this Space. Independent work, not affiliated with
Google.
