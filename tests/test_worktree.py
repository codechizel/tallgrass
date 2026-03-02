"""Tests for worktree workflow: hook script and recipe guard logic."""

import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
class TestWorktreeHook:
    """Test the WorktreeCreate hook script contract."""

    def _init_repo(self, tmp_path):
        """Create a minimal git repo with one commit on main."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo, check=True, capture_output=True,
        )
        (repo / "README.md").write_text("init")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo, check=True, capture_output=True,
        )
        return repo

    def test_hook_creates_worktree(self, tmp_path):
        """Hook creates worktree at .claude/worktrees/<name>/ with correct branch."""
        repo = self._init_repo(tmp_path)
        hook_path = str(Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "worktree-setup.sh")

        hook_input = json.dumps({
            "session_id": "test-session",
            "cwd": str(repo),
            "hook_event_name": "WorktreeCreate",
            "name": "test-feature",
        })

        result = subprocess.run(
            ["bash", hook_path],
            input=hook_input, capture_output=True, text=True, cwd=str(repo),
        )

        assert result.returncode == 0, f"Hook failed: {result.stderr}"

        # stdout must be exactly the worktree path (Claude Code reads this)
        expected_path = str(repo / ".claude" / "worktrees" / "test-feature")
        assert result.stdout.strip() == expected_path

        # Worktree directory must exist
        wt_dir = repo / ".claude" / "worktrees" / "test-feature"
        assert wt_dir.is_dir()

        # Branch must be worktree-<name>
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=wt_dir, capture_output=True, text=True, check=True,
        )
        assert branch_result.stdout.strip() == "worktree-test-feature"

    def test_hook_rejects_duplicate_name(self, tmp_path):
        """Hook fails if worktree branch already exists."""
        repo = self._init_repo(tmp_path)
        hook_path = str(Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "worktree-setup.sh")

        hook_input = json.dumps({
            "session_id": "test-session",
            "cwd": str(repo),
            "hook_event_name": "WorktreeCreate",
            "name": "dupe-test",
        })

        # First call succeeds
        r1 = subprocess.run(
            ["bash", hook_path],
            input=hook_input, capture_output=True, text=True, cwd=str(repo),
        )
        assert r1.returncode == 0

        # Second call with same name fails (branch already exists)
        r2 = subprocess.run(
            ["bash", hook_path],
            input=hook_input, capture_output=True, text=True, cwd=str(repo),
        )
        assert r2.returncode != 0

    def test_hook_stdout_is_only_path(self, tmp_path):
        """Hook must not pollute stdout — only the path, nothing else."""
        repo = self._init_repo(tmp_path)
        hook_path = str(Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "worktree-setup.sh")

        hook_input = json.dumps({
            "session_id": "test-session",
            "cwd": str(repo),
            "hook_event_name": "WorktreeCreate",
            "name": "stdout-test",
        })

        result = subprocess.run(
            ["bash", hook_path],
            input=hook_input, capture_output=True, text=True, cwd=str(repo),
        )
        assert result.returncode == 0

        lines = result.stdout.strip().splitlines()
        assert len(lines) == 1, f"stdout must be exactly one line, got: {lines}"


@pytest.mark.integration
class TestWtDoneGuards:
    """Test wt-done safety guards (without running the full recipe)."""

    def _init_repo_with_worktree(self, tmp_path):
        """Create repo + worktree, return (repo, worktree_path, branch)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo, check=True, capture_output=True,
        )
        (repo / "README.md").write_text("init")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo, check=True, capture_output=True,
        )

        wt_path = repo / "worktrees" / "test-wt"
        subprocess.run(
            ["git", "worktree", "add", "-b", "worktree-test-wt", str(wt_path), "main"],
            cwd=repo, check=True, capture_output=True,
        )
        return repo, wt_path, "worktree-test-wt"

    def test_rejects_on_main_branch(self, tmp_path):
        """wt-done guard rejects execution when on main."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)

        # Inline the guard logic
        guard = 'BRANCH=$(git branch --show-current); [ "$BRANCH" != "main" ]'
        result = subprocess.run(
            ["bash", "-c", guard], cwd=repo, capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_rejects_uncommitted_changes(self, tmp_path):
        """wt-done guard rejects execution with dirty working tree."""
        _, wt_path, _ = self._init_repo_with_worktree(tmp_path)

        # Dirty the worktree
        (wt_path / "dirty.txt").write_text("uncommitted")
        subprocess.run(["git", "add", "dirty.txt"], cwd=wt_path, check=True, capture_output=True)

        guard = "git diff --quiet && git diff --cached --quiet"
        result = subprocess.run(
            ["bash", "-c", guard], cwd=wt_path, capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_fast_forward_works(self, tmp_path):
        """merge-base + update-ref fast-forwards main when worktree is ahead."""
        repo, wt_path, branch = self._init_repo_with_worktree(tmp_path)

        # Make a commit in the worktree
        (wt_path / "feature.txt").write_text("new feature")
        subprocess.run(["git", "add", "."], cwd=wt_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add feature"],
            cwd=wt_path, check=True, capture_output=True,
        )

        # Verify fast-forward is possible
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "main", "HEAD"],
            cwd=wt_path, capture_output=True, text=True,
        )
        assert is_ancestor.returncode == 0

        # Fast-forward main via update-ref (works with Git 2.35+ unlike git fetch)
        result = subprocess.run(
            ["git", "update-ref", "refs/heads/main", "HEAD"],
            cwd=wt_path, capture_output=True, text=True,
        )
        assert result.returncode == 0

        # main should now include the feature commit
        log = subprocess.run(
            ["git", "log", "main", "--oneline", "-1"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        assert "add feature" in log.stdout

    def test_fast_forward_rejects_diverged(self, tmp_path):
        """merge-base --is-ancestor fails when main has diverged."""
        repo, wt_path, branch = self._init_repo_with_worktree(tmp_path)

        # Commit in worktree
        (wt_path / "feature.txt").write_text("feature")
        subprocess.run(["git", "add", "."], cwd=wt_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "worktree commit"],
            cwd=wt_path, check=True, capture_output=True,
        )

        # Commit on main (diverges)
        (repo / "main-change.txt").write_text("main diverged")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "main commit"],
            cwd=repo, check=True, capture_output=True,
        )

        # merge-base should reject (main is NOT an ancestor of HEAD)
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "main", "HEAD"],
            cwd=wt_path, capture_output=True, text=True,
        )
        assert result.returncode != 0
