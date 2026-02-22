"""
Tests for analysis run context infrastructure in analysis/run_context.py.

Covers _TeeStream output capture, session normalization, git hash retrieval,
and RunContext lifecycle (directory creation, log capture, symlinks).

Run: uv run pytest tests/test_run_context.py -v
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.run_context import (
    RunContext,
    _git_commit_hash,
    _normalize_session,
    _TeeStream,
)

# ── _TeeStream ───────────────────────────────────────────────────────────────


class TestTeeStream:
    """Duplicates output to original stream and internal buffer."""

    def test_write_returns_length(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        assert tee.write("hello") == 5

    def test_write_goes_to_original(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.write("hello")
        assert original.getvalue() == "hello"

    def test_write_goes_to_buffer(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.write("hello")
        assert tee.getvalue() == "hello"

    def test_multiple_writes_accumulate(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.write("hello ")
        tee.write("world")
        assert tee.getvalue() == "hello world"
        assert original.getvalue() == "hello world"

    def test_flush_does_not_raise(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.flush()  # should not raise

    def test_empty_write(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        assert tee.write("") == 0
        assert tee.getvalue() == ""


# ── _normalize_session() ─────────────────────────────────────────────────────


class TestNormalizeSession:
    """Convert session shorthand to biennium directory format."""

    def test_dash_two_digit(self):
        """'2025-26' → '91st_2025-2026'."""
        assert _normalize_session("2025-26") == "91st_2025-2026"

    def test_dash_four_digit(self):
        """'2025-2026' → '91st_2025-2026'."""
        assert _normalize_session("2025-2026") == "91st_2025-2026"

    def test_underscore_two_digit(self):
        """'2025_26' → '91st_2025-2026' (underscore normalized to dash)."""
        assert _normalize_session("2025_26") == "91st_2025-2026"

    def test_historical_session(self):
        """'2023-24' → '90th_2023-2024'."""
        assert _normalize_session("2023-24") == "90th_2023-2024"

    def test_special_session_passthrough(self):
        """'2024s' passes through unchanged."""
        assert _normalize_session("2024s") == "2024s"

    def test_already_normalized_underscore_becomes_dash(self):
        """Underscore in biennium format is normalized to dash."""
        assert _normalize_session("91st_2025-2026") == "91st-2025-2026"

    def test_plain_year_passthrough(self):
        """Bare year with no separator passes through."""
        assert _normalize_session("2025") == "2025"


# ── _git_commit_hash() ───────────────────────────────────────────────────────


class TestGitCommitHash:
    """Get current git commit hash."""

    def test_returns_string(self):
        result = _git_commit_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_hex_or_unknown(self):
        """Result is either a 40-char hex hash or 'unknown'."""
        result = _git_commit_hash()
        if result != "unknown":
            assert len(result) == 40
            assert all(c in "0123456789abcdef" for c in result)


# ── RunContext ────────────────────────────────────────────────────────────────


class TestRunContext:
    """Context manager for structured analysis output."""

    def test_setup_creates_directories(self, tmp_path):
        ctx = RunContext(
            session="2025-26",
            analysis_name="test_analysis",
            results_root=tmp_path,
        )
        ctx.setup()
        assert ctx.plots_dir.exists()
        assert ctx.data_dir.exists()
        assert ctx.run_dir.exists()
        # Restore stdout since setup replaces it
        ctx.finalize()

    def test_session_normalized(self, tmp_path):
        ctx = RunContext(
            session="2025-26",
            analysis_name="test",
            results_root=tmp_path,
        )
        assert ctx.session == "91st_2025-2026"

    def test_finalize_writes_run_info(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            params={"flag": True},
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        info_path = ctx.run_dir / "run_info.json"
        assert info_path.exists()
        data = json.loads(info_path.read_text())
        assert data["analysis"] == "test"
        assert data["session"] == "2024s"
        assert data["params"]["flag"] is True
        assert "git_commit" in data
        assert "python_version" in data

    def test_finalize_writes_run_log(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        print("test log line")
        ctx.finalize()
        log_path = ctx.run_dir / "run_log.txt"
        assert log_path.exists()
        assert "test log line" in log_path.read_text()

    def test_finalize_creates_latest_symlink(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        latest = tmp_path / "2024s" / "test" / "latest"
        assert latest.is_symlink()

    def test_context_manager_protocol(self, tmp_path):
        """__enter__ returns self, __exit__ calls finalize."""
        with RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        ) as ctx:
            assert isinstance(ctx, RunContext)
            assert ctx.plots_dir.exists()
        # After exit, run_info should exist
        assert (ctx.run_dir / "run_info.json").exists()

    def test_primer_written(self, tmp_path):
        primer_text = "# Test Analysis\nThis is a primer."
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            primer=primer_text,
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        readme = tmp_path / "2024s" / "test" / "README.md"
        assert readme.exists()
        assert readme.read_text() == primer_text

    def test_no_primer_no_readme(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        readme = tmp_path / "2024s" / "test" / "README.md"
        assert not readme.exists()

    def test_stdout_restored_after_finalize(self, tmp_path):
        original = sys.stdout
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        assert sys.stdout is not original  # tee is active
        ctx.finalize()
        assert sys.stdout is original  # restored

    def test_default_params_empty_dict(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        assert ctx.params == {}

    def test_consecutive_runs_update_latest(self, tmp_path):
        """Second run overwrites the latest symlink."""
        ctx1 = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx1.setup()
        ctx1.finalize()

        ctx2 = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx2.setup()
        ctx2.finalize()

        latest = tmp_path / "2024s" / "test" / "latest"
        assert latest.is_symlink()
