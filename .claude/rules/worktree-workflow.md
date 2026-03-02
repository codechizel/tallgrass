# Worktree Workflow

Git worktrees provide branch isolation without cloning. Claude Code creates them at `.claude/worktrees/<name>/` with branch `worktree-<name>`.

## Rules (Non-Negotiable)

1. **Never `git checkout main` from inside a worktree.** Main is checked out in the primary repo — git prevents the same branch in two worktrees. Use `git update-ref refs/heads/main HEAD` (after `merge-base --is-ancestor` check) to fast-forward main without checkout.

2. **Never remove a worktree while CWD is inside it.** The shell dies and all subsequent commands fail (known Claude Code bug #29653). Always `cd` to the main repo first, or use `just wt-done` which handles this.

3. **Never `git push origin branch:main` as a merge strategy.** It bypasses the local main ref, creating a diverged state. Use `git update-ref refs/heads/main HEAD` (updates local main) then `git push origin main`.

## Lifecycle

### Creating a worktree

Use `just wt-new <name>` or Claude Code's built-in `--worktree` flag. Both create `.claude/worktrees/<name>/` with a new branch.

### Working in a worktree

Normal workflow — commit with version tags, run tests, update docs. The worktree is a full repo checkout with shared `.git` state.

### Merging and cleanup

Two calling patterns — both do the same thing:

```bash
# From inside the worktree (interactive use):
just wt-done

# From the main repo (Claude Code — prevents CWD death):
just wt-done feature-name
```

**Claude Code sessions MUST use the second form.** Running `just wt-done` from inside the worktree deletes the CWD, killing the shell. Call from the main repo with the worktree name instead.

What it does:
1. Verifies all changes are committed
2. Fast-forwards `main` to the worktree branch via `git update-ref`
3. Removes the worktree directory
4. Deletes the local branch

Push separately with `git push origin main` when on a network that supports it.

If main has diverged (non-fast-forward), `wt-done` fails safely. Rebase onto main first: `git rebase main` in the worktree, then retry.

### Manual merge (when wt-done can't fast-forward)

```bash
# In the worktree — rebase onto main first
git fetch origin main
git rebase origin/main

# Then from main repo:
just wt-done feature-name
```

## Key Primitives

| Want to... | Command | Why |
|------------|---------|-----|
| Update main without checkout | `git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD` | Fast-forward only; works with Git 2.35+ (which blocks `git fetch . branch:main` when main is checked out) |
| Check if fast-forward is possible | `git merge-base --is-ancestor main HEAD` | Exit 0 = yes |
| List worktrees | `git worktree list` | Shows all linked worktrees |
| Clean stale admin | `git worktree prune` | After manual directory deletion |
