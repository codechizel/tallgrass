# ADR-0077: Worktree CWD Death Prevention

**Date:** 2026-03-02
**Status:** Accepted

## Context

Git worktrees provide branch isolation for Claude Code sessions, but a recurring failure mode — "CWD death" — has caused multiple session crashes:

1. `just wt-done` is called from inside a worktree directory
2. The recipe fast-forwards main, then removes the worktree directory
3. The calling shell's CWD now points to a deleted directory
4. All subsequent Bash/Glob commands fail — the session is unrecoverable

This is a POSIX limitation: no subprocess can change the parent shell's CWD. The `cd` in the old recipe only changed the subshell's CWD, not Claude Code's persistent Bash CWD.

A second bug compounded the problem: `git fetch . HEAD:main` (used to fast-forward main without checkout) is blocked by Git 2.35+ when main is already checked out in the primary worktree. This caused `wt-done` to fail mid-recipe, sometimes leaving the worktree in a partially-cleaned state.

## Decision

Three changes:

### 1. Replace `git fetch . branch:main` with `git update-ref`

```bash
git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD
```

Works on all Git versions. The `merge-base` check ensures fast-forward safety; `update-ref` directly moves the ref without checkout.

### 2. `wt-done` accepts optional worktree name

```bash
# From inside worktree (interactive/human use — CWD dies after):
just wt-done

# From main repo (Claude Code — no CWD death):
just wt-done feature-name
```

When a name is provided, the recipe resolves the worktree path (`.claude/worktrees/<name>/`) and branch (`worktree-<name>`) from the name. All `git` commands use `-C "$WT_PATH"` so they work regardless of the caller's CWD.

### 3. Split into forwarder + implementation

`wt-done` is a thin forwarder that delegates to `_wt-done-impl` in the main repo's Justfile via `just --justfile "$MAIN_ROOT/Justfile"`. This ensures worktrees created before a Justfile fix still pick up the latest logic.

### 4. WorktreeCreate hook creates the worktree

`.claude/hooks/worktree-setup.sh` now creates the git worktree + branch (not just installing dependencies). All diagnostic output goes to stderr; only the worktree path goes to stdout (required by Claude Code's hook protocol).

### 5. Rules updated

`worktree-workflow.md` and `CLAUDE.md` now state that Claude Code sessions **must** use `just wt-done <name>` from the main repo. The auto-detect form (`just wt-done` from inside the worktree) remains for interactive human use.

## Consequences

**Positive:**
- CWD death is structurally prevented for Claude Code sessions — the shell CWD is always the main repo when the worktree is removed
- `git update-ref` works on Git 2.35+ (macOS ships 2.39+)
- Justfile fixes are always picked up by worktrees (forwarder pattern)
- `wt-done` no longer auto-pushes — push is a separate explicit step

**Negative:**
- Claude Code must remember to use the named form — enforced by rules in `worktree-workflow.md`
- Human users calling `just wt-done` from inside a worktree will still experience CWD death in their terminal (they need to `cd` to main repo afterward) — acceptable since interactive users can simply open a new terminal or `cd`

**Recovery from CWD death (if it still happens):**
1. Start a new Claude Code session from the main repo
2. `git worktree prune -v` to clean stale registrations
3. `git branch -d worktree-<name>` to delete orphan branches
4. `git reset --hard HEAD` if working tree is out of sync (save uncommitted improvements first)
