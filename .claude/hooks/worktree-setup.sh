#!/bin/bash
# WorktreeCreate hook: install dependencies in new worktrees.
# Called by Claude Code with JSON on stdin: {"name": "...", "session_id": "...", "cwd": "..."}
# Must print the worktree path to stdout (required by Claude Code).
set -e

HOOK_INPUT=$(cat)
WORKTREE_PATH=$(echo "$HOOK_INPUT" | python3 -c "import sys, json; print(json.load(sys.stdin)['cwd'])")

cd "$WORKTREE_PATH" || exit 1

# Install Python dependencies
if [ -f "pyproject.toml" ] && command -v uv &> /dev/null; then
    uv sync --quiet 2>/dev/null || true
fi

# Required: output the worktree path
echo "$WORKTREE_PATH"
