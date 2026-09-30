# litert-capture loop — non-stop plan for ROADMAP 1-50

Status: `main @ 0a5590a`, `ROADMAP.md` 50 points, `out/shaders/*.wgsl = 153`, issue #3775 OPEN.

## 0. One-shot before loop
- Commit `ROADMAP.md` (currently `??`). Loop fails closed until tracked.
- `git checkout main && git pull --ff-only`, clean tree. Tools: `node pnpm biome gitleaks`.
- Create `LOOP.md` (this file), `STATE.md`, `scripts/loop.mjs`, `ledger/`.

## 1. Loop shape — one point at a time
- Parse queue: `ROADMAP.md` lines `^(\d+)\. `, expect 01..50. Count != 50 = stop.
- Slug: lowercase, `[^a-z0-9]+ -> -`, 40 chars. Ex `01-parameterize-version`.
- Resume from `STATE.md`: `done -> +1`, `parked|in-progress -> re-verify or fresh branch`.
- Per point: `checkout main && pull --ff-only && checkout -b loop/<nn>-<slug>`. Scratch `/tmp/litert-loop-<nn>/` (wiped). Implement only that scope.
- Max 5 points per batch, then stop for human review/merge. Never push `main`. Never merge. Never auto-PR.

## 2. Gates — all green or no push
Run from root, logs to `/tmp/litert-loop-<nn>/gate-<name>.log`:
- G0 latest Chrome check: compare puppeteer-pinned Chrome for Testing vs latest (`npx @puppeteer/browsers install chrome@stable --dry-run`, or `gh api` / registry check); if behind >1 major, note in record + re-run key microbench on latest before posting.
- G1 builds byte-identical: `node space/build.mjs && node colab/build-notebook.mjs`, sha both, rerun both, `sha256sum -c`. Differ = red.
- G2 lint: `pnpm exec biome check .` exit 0.
- G3 syntax: `for f in *.mjs colab/*.mjs space/*.mjs scripts/*.mjs; do node --check "$f"; done` exit 0.
- G4 analyze: `node analyze.mjs 400` exit 0.
- G5 status set: `git status --porcelain` equals expected set only (touched + `STATE.md` + `ledger/<nn>-*.json` + declared `space/index.html` / `colab/kernel-0112-t4.ipynb`). Any `out/shaders/*.wgsl`, `docs/`, `node_modules/`, undeclared `out/*.json` = red.
- G6 secrets: `gitleaks git --staged --redact --no-banner` exit 0.
- Push branch only if G0-G6 green. Record exits + SHAs in ledger.

## 3. Guards — hard deny
- `out/shaders/*.wgsl` frozen. Touched = abort point.
- `out/*.json`: copy dated backup first, write dated files only (point 18 pattern).
- No `gh issue create / gh pr create / hf upload` inside loop. Space upload manual after branch verify.
- Only `git push origin loop/<nn>-<slug>`. `docs/` stays gitignored (`.gitignore:5`). Never `git add -f docs/`.

## 4. State
- `STATE.md`: `current_point, status (done|parked|awaiting-review), branch, last_sha, last_verify_utc, gates, parked[], next_batch_review_after`.
- `ledger/<nn>-<slug>.json`: `{point, slug, branch, commit_short, timestamp, gates:{g1..g6}, files_changed[], notes}`. One per point, committed on its branch.
- Scratch `/tmp/litert-loop-<nn>/` wiped each start. `--resume` re-checkouts branch or restarts from `main`.

## 5. Stop conditions
- Gate fail 3x: park point, write `ledger/<nn>-parked.json`, `checkout main`, next point. Never force-green.
- Stop file: `./.loop-stop` or `/tmp/litert-loop.STOP` exists = finish gate, update STATE, exit 0.
- Remote `main` moved: `fetch + rev-parse main vs origin/main` differ = abort, `checkout main && pull --ff-only`, human decides. No auto-rebase.
- 5 done = stop, print `git log main..loop/* --oneline` + ledger summary, wait.

## 6. Batches
- P1 capture 1-8, P2 isolate 9-18, P3 in-model 19-26, P4 analyze 27-34, P5 provenance 35-40, P6 publish 41-50 (nightly/vendor/multi-engine/corpus last).
- Scripts: `package.json:10-15` chain, `provenance.mjs:89-101` stamps, `lefthook.yml:8-23` hooks stay on push path.
