"""Tests for experiment monitoring infrastructure.

Verifies PlatformCheck validation, write_status() atomic writes,
ExperimentLifecycle PID locking and cleanup, and monitoring callback creation.

Run: uv run pytest tests/test_experiment_monitor.py -v
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.experiment_monitor import (
    PID_PATH,
    STATUS_PATH,
    ExperimentLifecycle,
    PlatformCheck,
    create_monitoring_callback,
    write_status,
)

# ── PlatformCheck ────────────────────────────────────────────────────────────


class TestPlatformCheck:
    """PlatformCheck validates Apple Silicon MCMC constraints."""

    def test_frozen(self):
        pc = PlatformCheck(6, 6, 12, 0, "/usr/bin/clang++")
        with pytest.raises(AttributeError):
            pc.omp_threads = 8  # type: ignore[misc]

    def test_valid_config_no_warnings(self):
        pc = PlatformCheck(6, 6, 12, 0, "/usr/bin/clang++")
        assert pc.validate(4) == []

    def test_missing_compiler_fatal(self):
        pc = PlatformCheck(6, 6, 12, 0, "")
        warnings = pc.validate(4)
        assert len(warnings) >= 1
        assert "FATAL" in warnings[0]
        assert "C++ compiler" in warnings[0]

    def test_omp_threads_zero_warns(self):
        pc = PlatformCheck(0, 6, 12, 0, "/usr/bin/clang++")
        warnings = pc.validate(4)
        assert any("OMP_NUM_THREADS=0" in w for w in warnings)

    def test_omp_threads_too_high_warns(self):
        pc = PlatformCheck(12, 6, 12, 0, "/usr/bin/clang++")
        warnings = pc.validate(4)
        assert any("OMP_NUM_THREADS=12" in w for w in warnings)

    def test_openblas_zero_warns(self):
        pc = PlatformCheck(6, 0, 12, 0, "/usr/bin/clang++")
        warnings = pc.validate(4)
        assert any("OPENBLAS_NUM_THREADS=0" in w for w in warnings)

    def test_active_processes_warns(self):
        pc = PlatformCheck(6, 6, 12, 2, "/usr/bin/clang++")
        warnings = pc.validate(4)
        assert any("2 other MCMC" in w for w in warnings)

    def test_too_many_chains_warns(self):
        pc = PlatformCheck(6, 6, 12, 0, "/usr/bin/clang++")
        warnings = pc.validate(8)
        assert any("n_chains=8" in w for w in warnings)

    def test_all_bad_multiple_warnings(self):
        pc = PlatformCheck(0, 0, 12, 1, "")
        warnings = pc.validate(8)
        assert len(warnings) >= 4  # compiler + omp + openblas + concurrent + chains

    def test_current_classmethod(self):
        """current() returns a PlatformCheck with real values."""
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6"}):
            pc = PlatformCheck.current()
        assert isinstance(pc, PlatformCheck)
        assert pc.cpu_count > 0


# ── write_status ─────────────────────────────────────────────────────────────


class TestWriteStatus:
    """Atomic status file writes."""

    def test_roundtrip(self, tmp_path: Path):
        status_file = tmp_path / "status.json"
        data = {"experiment": "test", "draw": 100, "phase": "sampling"}
        write_status(data, status_file)
        loaded = json.loads(status_file.read_text())
        assert loaded == data

    def test_overwrites_existing(self, tmp_path: Path):
        status_file = tmp_path / "status.json"
        write_status({"draw": 1}, status_file)
        write_status({"draw": 2}, status_file)
        loaded = json.loads(status_file.read_text())
        assert loaded["draw"] == 2

    def test_creates_parent_dirs(self, tmp_path: Path):
        status_file = tmp_path / "sub" / "dir" / "status.json"
        write_status({"ok": True}, status_file)
        assert status_file.exists()

    def test_valid_json(self, tmp_path: Path):
        status_file = tmp_path / "status.json"
        write_status({"nested": {"a": 1}, "list": [1, 2, 3]}, status_file)
        loaded = json.loads(status_file.read_text())
        assert loaded["nested"]["a"] == 1
        assert loaded["list"] == [1, 2, 3]


# ── Monitoring Callback ─────────────────────────────────────────────────────


class TestMonitoringCallback:
    """Monitoring callback for PyMC sampling."""

    def test_callback_callable(self):
        cb = create_monitoring_callback("test-exp", "baseline", "house", 2000, 1500)
        assert callable(cb)

    def test_callback_writes_status_on_draw_0(self, tmp_path: Path):
        status_file = tmp_path / "status.json"
        cb = create_monitoring_callback(
            "test-exp", "lognormal", "senate", 2000, 1500, status_path=status_file
        )
        # Simulate a draw at index 0
        draw = MagicMock()
        draw.tuning = False
        draw.draw_idx = 0
        draw.chain = 0
        cb(None, draw)
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["experiment"] == "test-exp"
        assert data["variant"] == "lognormal"
        assert data["chamber"] == "senate"
        assert data["phase"] == "sampling"

    def test_callback_skips_non_50_draws(self, tmp_path: Path):
        status_file = tmp_path / "status.json"
        cb = create_monitoring_callback(
            "test-exp", "baseline", "house", 2000, 1500, status_path=status_file
        )
        draw = MagicMock()
        draw.tuning = True
        draw.draw_idx = 7
        draw.chain = 0
        cb(None, draw)
        assert not status_file.exists()

    def test_callback_writes_on_multiples_of_50(self, tmp_path: Path):
        status_file = tmp_path / "status.json"
        cb = create_monitoring_callback(
            "test-exp", "baseline", "house", 2000, 1500, status_path=status_file
        )
        draw = MagicMock()
        draw.tuning = False
        draw.draw_idx = 150
        draw.chain = 2
        cb(None, draw)
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["draw"] == 150
        assert data["chain"] == 2

    def test_callback_tuning_phase(self, tmp_path: Path):
        status_file = tmp_path / "status.json"
        cb = create_monitoring_callback(
            "test-exp", "baseline", "house", 2000, 1500, status_path=status_file
        )
        draw = MagicMock()
        draw.tuning = True
        draw.draw_idx = 50
        draw.chain = 0
        cb(None, draw)
        data = json.loads(status_file.read_text())
        assert data["phase"] == "tuning"
        assert data["total"] == 1500


# ── ExperimentLifecycle ─────────────────────────────────────────────────────


class TestExperimentLifecycle:
    """ExperimentLifecycle context manager for PID locking and cleanup."""

    def test_creates_pid_file(self):
        with ExperimentLifecycle("test-lifecycle"):
            assert PID_PATH.exists()
            pid_content = PID_PATH.read_text()
            assert pid_content == str(os.getpid())

    def test_cleans_up_pid_file(self):
        with ExperimentLifecycle("test-lifecycle"):
            pass
        assert not PID_PATH.exists()

    def test_cleans_up_status_file(self):
        write_status({"test": True})
        with ExperimentLifecycle("test-lifecycle"):
            pass
        assert not STATUS_PATH.exists()

    def test_context_returns_self(self):
        lifecycle = ExperimentLifecycle("test-lifecycle")
        with lifecycle as ctx:
            assert ctx is lifecycle
            assert ctx.experiment_name == "test-lifecycle"
