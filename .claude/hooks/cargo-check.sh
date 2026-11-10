#!/usr/bin/env bash
# Claude Code PostToolUse hook (Edit|Write): after a Rust source file changes, run `cargo check`
# and feed warnings/errors back to the model (CLAUDE.md: warnings = errors). Reads the hook
# JSON on stdin; does nothing for non-.rs files. Skips when another cargo process is running
# (parallel agents) to avoid piling up on the target-dir lock.
set -u
input=$(cat)
file=$(printf '%s' "$input" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print(''); sys.exit(0)
ti = d.get('tool_input') or {}
tr = d.get('tool_response') or {}
print(ti.get('file_path') or (tr.get('filePath') if isinstance(tr, dict) else '') or '')
")
case "$file" in
  *.rs) ;;
  *) exit 0 ;;
esac
if tasklist 2>/dev/null | grep -qi "cargo.exe"; then exit 0; fi
cd "$(dirname "$0")/../.." || exit 0
out=$(cargo check --all-targets --message-format short 2>&1 \
  | grep -E '^(src|tests|plugins)[^:]*:[0-9]+:[0-9]+: (warning|error)' | head -20)
if [ -n "$out" ]; then
  printf '%s' "$out" | python -c "
import sys, json
msg = 'cargo check after editing ' + sys.argv[1] + ' reported:\n' + sys.stdin.read()
print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': msg}}))
" "$file"
fi
exit 0
