#!/bin/bash
# WorktreeCreate hook: creates a git worktree and installs dependencies.
# This hook REPLACES Claude Code's default git worktree creation.
# Called with JSON on stdin: {"name": "...", "session_id": "...", "cwd": "..."}
# Must print the absolute worktree path to stdout (all other output to stderr).
set -e

HOOK_INPUT=$(cat)
REPO_ROOT=$(echo "$HOOK_INPUT" | python3 -c "import sys, json; print(json.load(sys.stdin)['cwd'])")
WT_NAME=$(echo "$HOOK_INPUT" | python3 -c "import sys, json; print(json.load(sys.stdin)['name'])")

WT_PATH="$REPO_ROOT/.claude/worktrees/$WT_NAME"
BRANCH="worktree-$WT_NAME"

# Create the worktree (all diagnostic output to stderr)
mkdir -p "$(dirname "$WT_PATH")"
git -C "$REPO_ROOT" worktree add -b "$BRANCH" "$WT_PATH" main >&2

# Install Python dependencies
cd "$WT_PATH"
if [ -f "pyproject.toml" ] && command -v uv &> /dev/null; then
    uv sync --quiet >&2 || echo "Warning: uv sync failed" >&2
fi

# Required: print the absolute worktree path (only stdout output)
echo "$WT_PATH"
