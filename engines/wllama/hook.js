// Browser-side WebGPU capture hook for wllama (llama.cpp WASM + WebGPU).
//
// Prototype-level wrappers so every GPUDevice wllama's WebGPU backend
// (ngxson/wllama, shipped in v3.1) creates is covered. Same construction as
// www/capture.html, engines/webllm/hook.js and engines/ort/hook.js: the
// capture half of the capture+trace pipeline (WGSL text + pipeline labels
// + dispatch counters). The trace half (per-dispatch shapes/times) joins by
// pipeline/dispatch position like root analyze.mjs.
//
// Install BEFORE constructing the Wllama instance, then await
// window.__wllamaSettle() before declaring done.
(() => {
  const cap = (window.__wllamaCapture = {
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
  window.__wllamaSettle = () => Promise.all(hashJobs);

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
