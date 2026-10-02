# capture loop — non-stop plan for CAPTURE-ROADMAP 1-18

Status: `main @ fd7b402`, 18 targets, issue #3775 OPEN. Companion to `LOOP.md`
(same gates G0-G6) plus GPU-run rules below. Batches run back-to-back with no
human stop until every item is done or parked with a logged reason.

## 0. One-shot before loop
- `CAPTURE-ROADMAP.md` + `CAPTURE-LOOP.md` (this file) committed on `main`.
- Model present: `www/gemma-4-E2B-it-web.litertlm` (~2 GB, gitignored). If
  missing, GPU-measurement items park immediately — no 2 GB download inside
  the loop without explicit human go.
- Chrome builds present: 131 / 146 / 154 in puppeteer cache. Missing build
  for a required version = park, not fail.

## 1. Loop shape — one target at a time
- Queue: `CAPTURE-ROADMAP.md` lines `^(\d+)\. `, expect 01..18. Count != 18
  = stop.
- Branch `capture/<nn>-<slug>` from `main`, implement, gate, push branch.
  Merge batch to `main` when the batch is fully done-or-parked, then next
  batch. Never push `main` except batch merges. Never auto-PR.
- Docs/provenance items behave like `LOOP.md` points (comment-only allowed).
- Measurement items MUST produce a real `out/<name>.json` from a real run
  (never synthetic, never hand-written numbers) or park.

## 2. GPU-run rules (in addition to LOOP.md G0-G6)
- R1 time-box: single harness run max 30 min (`TIMEOUT_MS` respected); hang
  past timeout = kill, log tail, park.
- R2 backups: `cp out/<name>.json out/<name>-<date>-bak.json` before any
  overwrite; new captures write dated files where the harness supports it.
- R3 model check first: `ls -lh www/gemma-4-E2B-it-web.litertlm` + `shasum`
  vs `out/model.json`; mismatch = park, never re-download silently.
- R4 provenance required: every new record must carry `provenance` (browser,
  adapter, harness commit) + `manifest_sha256`, else park.
- R5 no thermal lies: if p90-p10 width >25% or forward/reverse ramp >15%,
  note it in the record (like the old T4 note) rather than re-running until
  pretty. One re-run max, then commit-what-you-see or park.

## 3. Guards (add to LOOP.md §3)
- `out/shaders/*.wgsl` frozen (153). `out/*.json` existing records:
  read-only unless the item names them; new files only (`out/prefill.json`,
  `out/kv-layout.json`, …) plus dated backups.
- No `gh issue` comments/posts, no `hf upload`, no Colab driving from here.
  Vendor-matrix rows that need Colab/T4 produce a ready-to-run notebook +
  instructions instead, then park as `blocked/colab-needed`.
- Other-engine items (9-12): capture code goes under `engines/<name>/`, never
  mixed into root `out/`; needs model download = park with exact commands.

## 4. Park policy (done = merged; parked = logged, never silent)
- Park reasons: `no-gpu`, `timeout`, `model-missing`, `blocked/colab-needed`,
  `needs-download`, `flaky>25pct`, `gate-fail-3x`.
- Each parked item writes `ledger/capture-<nn>-parked.json` with reason +
  logs, committed on its branch, branch pushed. Batch merge includes parked
  branches (docs/ledger only, no fake records).
- 18/18 done-or-parked = loop complete. Report table: done vs parked+reason.

## 5. Batches (run back-to-back, no stop between)
- A same-engine 1-8: prefill, kv-layout, fakequant-0119, load-path,
  power-curve, firstslot, native-ratio, nonf16.
- C extensions 13-18: cpu-attrib, memory-residency, vendor-matrix,
  correctness-gate, roofline, quant-throttle.
- B other-engines 9-12: webllm-tvm, ort-web, wllama, matmul-micro.
