#!/bin/sh
# Daily watch for google-ai-edge/LiteRT-LM#3775. Appends to user-local log.
# Installed daily 09:00 via com.litert.issue3775.plist. Manual: sh scripts/check-issue.sh
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
LOG="$HOME/.claude/litert-issue-watch.log"
mkdir -p "$(dirname "$LOG")"
TS="$(date -u +%FT%TZ)"
META="$(gh issue view 3775 -R google-ai-edge/LiteRT-LM --json number,state,updatedAt --jq '"\(.number) \(.state) updated=\(.updatedAt)"' 2>&1)"
COUNT="$(gh issue view 3775 -R google-ai-edge/LiteRT-LM --comments --json comments --jq '.comments | length' 2>&1)"
LATEST="$(gh issue view 3775 -R google-ai-edge/LiteRT-LM --comments --json comments --jq '.comments[-1] | "\(.author.login) @ \(.createdAt)"' 2>&1)"
printf '%s %s comments=%s latest=%s\n' "$TS" "$META" "$COUNT" "$LATEST" >> "$LOG"
tail -n 1 "$LOG"
