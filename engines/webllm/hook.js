// Browser-side WebGPU capture hook for WebLLM (TVM) sessions.
//
// Prototype-level wrappers so every GPUDevice WebLLM's TVM runtime creates
// is covered, however it is created. Same construction as www/capture.html
// in this repo; the device-level variant is zero-tvm's patchForCapture
// (src/_archived/capture.ts:124), which proved the interception point but
// only covers the one device it is handed. TVM creates pipelines almost
// exclusively via createComputePipelineAsync, so both sync and async paths
// are recorded.
//
// Reference shader set: ~/dev/zero-tvm/src/tvm-shaders/ (a whole WebLLM
// session dump, ~60 WGSL files): fused_dequantize{1,2,3,4,5}*_NT_matmul*
// (the dequantize-matmul family this item K-searches), batch_prefill /
// batch_decode paged-KV kernels, rms_norm / fused_add_norm, rope, softmax
// with chunked sum, gather/argsort/take/copy utilities. Kernel family for
// the per-kernel table is derived from the pipeline label / shader label,
// not from byte length (distinct TVM shaders share lengths).
//
// Version note: adapted against @mlc-ai/web-llm 0.2.85 (npm latest at time
// of writing; zero-tvm pins 0.2.84). The hook itself is version-agnostic
// (WebGPU prototype level), but the model_lib wasm URL segment carries the
// version (v0_2_84 vs v0_2_85) and the record must carry the loaded package
// version in provenance, like capture.mjs does for @litert-lm/core.
//
// Install BEFORE importing @mlc-ai/web-llm, then await window.__wlSettle()
// before declaring done. No model bytes are embedded here.
(() => {
  const cap = (window.__webllmCapture = {
    shaders: [],
    pipelines: [],
    dispatchByPipeline: {},
    counters: { dispatch: 0, dispatchIndirect: 0, passes: 0, submits: 0 },
  });

  const shaderIds = new WeakMap();
  const pipelineIds = new WeakMap();
  const passPipeline = new WeakMap();
  const hashJobs = [];
  const hashInto = (rec, text) => {
    if (!globalThis.crypto?.subtle) {
      cap.shaderSha256Unavailable = true;
      return;
    }
    hashJobs.push(
      crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)).then((d) => {
        rec.shaderSha256 = [...new Uint8Array(d)]
          .map((b) => b.toString(16).padStart(2, "0"))
          .join("");
      }),
    );
  };
  window.__wlSettle = () => Promise.all(hashJobs);

  const wrap = (proto, name, fn) => {
    const orig = proto[name];
    proto[name] = function (...args) {
      return fn.call(this, orig, args);
    };
  };

  wrap(GPUDevice.prototype, "createShaderModule", function (orig, args) {
    const mod = orig.apply(this, args);
    const id = cap.shaders.length;
    const rec = { id, label: args[0]?.label ?? "", code: args[0]?.code ?? "", shaderSha256: null };
    cap.shaders.push(rec);
    hashInto(rec, rec.code);
    shaderIds.set(mod, id);
    return mod;
  });

  const recordPipeline = (pipeline, desc, isAsync) => {
    const id = cap.pipelines.length;
    cap.pipelines.push({
      id,
      shaderId: shaderIds.get(desc?.compute?.module) ?? -1,
      entryPoint: desc?.compute?.entryPoint ?? "",
      label: desc?.label ?? "",
      constants: desc?.compute?.constants ?? null,
      async: isAsync,
    });
    pipelineIds.set(pipeline, id);
  };
  wrap(GPUDevice.prototype, "createComputePipeline", function (orig, args) {
    const p = orig.apply(this, args);
    recordPipeline(p, args[0], false);
    return p;
  });
  wrap(GPUDevice.prototype, "createComputePipelineAsync", function (orig, args) {
    return orig.apply(this, args).then((p) => {
      recordPipeline(p, args[0], true);
      return p;
    });
  });

  wrap(GPUCommandEncoder.prototype, "beginComputePass", function (orig, args) {
    cap.counters.passes++;
    return orig.apply(this, args);
  });
  wrap(GPUComputePassEncoder.prototype, "setPipeline", function (orig, args) {
    passPipeline.set(this, pipelineIds.get(args[0]) ?? -1);
    return orig.apply(this, args);
  });
  const countDispatch = (pass, key) => {
    cap.counters[key]++;
    const pid = passPipeline.get(pass) ?? -1;
    cap.dispatchByPipeline[pid] = (cap.dispatchByPipeline[pid] ?? 0) + 1;
  };
  wrap(GPUComputePassEncoder.prototype, "dispatchWorkgroups", function (orig, args) {
    countDispatch(this, "dispatch");
    return orig.apply(this, args);
  });
  wrap(GPUComputePassEncoder.prototype, "dispatchWorkgroupsIndirect", function (orig, args) {
    countDispatch(this, "dispatchIndirect");
    return orig.apply(this, args);
  });
  wrap(GPUQueue.prototype, "submit", function (orig, args) {
    cap.counters.submits++;
    return orig.apply(this, args);
  });
})();
