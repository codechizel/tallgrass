# Worktree Workflow

Git worktrees provide branch isolation without cloning. Claude Code creates them at `.claude/worktrees/<name>/` with branch `worktree-<name>`.

## Rules (Non-Negotiable)

1. **Never `git checkout main` from inside a worktree.** Main is checked out in the primary repo — git prevents the same branch in two worktrees. Use `git fetch . HEAD:main` to fast-forward main without checkout.

2. **Never remove a worktree while CWD is inside it.** The shell dies and all subsequent commands fail (known Claude Code bug #29653). Always `cd` to the main repo first, or use `just wt-done` which handles this.

3. **Never `git push origin branch:main` as a merge strategy.** It bypasses the local main ref, creating a diverged state. Use `git fetch . branch:main` (updates local main) then `git push origin main`.

## Lifecycle

### Creating a worktree

Use `just wt-new <name>` or Claude Code's built-in `--worktree` flag. Both create `.claude/worktrees/<name>/` with a new branch.

### Working in a worktree

Normal workflow — commit with version tags, run tests, update docs. The worktree is a full repo checkout with shared `.git` state.

### Merging and cleanup (single command)

From inside the worktree, run `just wt-done`. This:
1. Verifies all changes are committed
2. Fast-forwards `main` to the worktree branch via `git fetch . branch:main`
3. Changes CWD to the main repo root (prevents CWD death)
4. Removes the worktree directory
5. Deletes the local branch
6. Pushes updated main to remote

If main has diverged (non-fast-forward), `wt-done` fails safely. Rebase onto main first: `git rebase main` in the worktree, then retry.

### Manual merge (when wt-done can't fast-forward)

```bash
# In the worktree — rebase onto main first
git fetch origin main
git rebase origin/main

# Then wt-done works
just wt-done
```

## Key Primitives

| Want to... | Command | Why |
|------------|---------|-----|
| Update main without checkout | `git fetch . HEAD:main` | Treats local repo as remote; fast-forward only |
| Check if fast-forward is possible | `git merge-base --is-ancestor main HEAD` | Exit 0 = yes |
| List worktrees | `git worktree list` | Shows all linked worktrees |
| Clean stale admin | `git worktree prune` | After manual directory deletion |
