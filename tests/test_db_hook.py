"""
Tests for the scraper post-hook that loads CSVs into PostgreSQL.

All subprocess calls are mocked — no real database or Django needed.

Run: uv run pytest tests/test_db_hook.py -v
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tallgrass.db_hook import _run_manage, try_load_alec, try_load_session

pytestmark = pytest.mark.scraper


# ── _run_manage internals ───────────────────────────────────────────────────


class TestRunManage:
    """Low-level subprocess invocation for Django management commands."""

    def test_calls_subprocess_with_correct_args(self):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_manage(["load_session", "91st_2025-2026"])
            args = mock_run.call_args[0][0]
            assert "src/web/manage.py" in args
            assert "load_session" in args
            assert "91st_2025-2026" in args

    def test_sets_django_settings_module(self):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_manage(["load_session", "91st_2025-2026"])
            env = mock_run.call_args[1]["env"]
            assert env["DJANGO_SETTINGS_MODULE"] == "tallgrass_web.settings.local"

    def test_sets_pythonpath(self):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_manage(["load_session", "91st_2025-2026"])
            env = mock_run.call_args[1]["env"]
            assert env["PYTHONPATH"] == "src/web"

    def test_uses_uv_run_group_web(self):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_manage(["load_session", "91st_2025-2026"])
            args = mock_run.call_args[0][0]
            # Check that --group web is in the args
            assert "--group" in args
            idx = args.index("--group")
            assert args[idx + 1] == "web"

    def test_timeout_is_300_seconds(self):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_manage(["load_session", "91st_2025-2026"])
            assert mock_run.call_args[1]["timeout"] == 300

    def test_captures_output(self):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_manage(["load_session", "91st_2025-2026"])
            assert mock_run.call_args[1]["capture_output"] is True
            assert mock_run.call_args[1]["text"] is True

    def test_prints_success_on_zero_returncode(self, capsys):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run_manage(["load_session", "91st_2025-2026"])
            output = capsys.readouterr().out
            assert "complete" in output.lower()

    def test_prints_warning_on_nonzero_returncode(self, capsys):
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Connection refused")
            _run_manage(["load_session", "91st_2025-2026"])
            output = capsys.readouterr().out
            assert "WARNING" in output
            assert "Connection refused" in output

    def test_does_not_raise_on_nonzero_returncode(self):
        """Fail soft — non-zero exit should not propagate as an exception."""
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="DB down")
            _run_manage(["load_session", "91st_2025-2026"])  # Should not raise

    def test_handles_file_not_found_gracefully(self, capsys):
        """If uv binary is not found, print warning instead of crashing."""
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("No such file: uv")
            _run_manage(["load_session", "91st_2025-2026"])
            output = capsys.readouterr().out
            assert "WARNING" in output
            assert "uv not found" in output

    def test_handles_timeout_gracefully(self, capsys):
        """If subprocess times out, print warning instead of crashing."""
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="uv", timeout=300)
            _run_manage(["load_session", "91st_2025-2026"])
            output = capsys.readouterr().out
            assert "WARNING" in output
            assert "timed out" in output

    def test_shows_last_3_lines_of_stderr(self, capsys):
        stderr = "line1\nline2\nline3\nline4\nline5"
        with patch("tallgrass.db_hook.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr=stderr)
            _run_manage(["load_session", "91st_2025-2026"])
            output = capsys.readouterr().out
            assert "line3" in output
            assert "line4" in output
            assert "line5" in output
            assert "line1" not in output


# ── try_load_session ────────────────────────────────────────────────────────


class TestTryLoadSession:
    """Public API for loading a session into PostgreSQL."""

    def test_calls_load_session_command(self):
        with patch("tallgrass.db_hook._run_manage") as mock:
            try_load_session("91st_2025-2026")
            mock.assert_called_once_with(["load_session", "91st_2025-2026"])

    def test_passes_skip_bill_text_flag(self):
        with patch("tallgrass.db_hook._run_manage") as mock:
            try_load_session("91st_2025-2026", skip_bill_text=True)
            mock.assert_called_once_with(
                ["load_session", "91st_2025-2026", "--skip-bill-text"]
            )

    def test_no_skip_bill_text_by_default(self):
        with patch("tallgrass.db_hook._run_manage") as mock:
            try_load_session("91st_2025-2026")
            args = mock.call_args[0][0]
            assert "--skip-bill-text" not in args

    def test_prints_session_name(self, capsys):
        with patch("tallgrass.db_hook._run_manage"):
            try_load_session("91st_2025-2026")
            output = capsys.readouterr().out
            assert "91st_2025-2026" in output


# ── try_load_alec ───────────────────────────────────────────────────────────


class TestTryLoadAlec:
    """Public API for loading the ALEC corpus into PostgreSQL."""

    def test_calls_load_alec_command(self):
        with patch("tallgrass.db_hook._run_manage") as mock:
            try_load_alec()
            mock.assert_called_once_with(["load_alec"])

    def test_prints_alec_message(self, capsys):
        with patch("tallgrass.db_hook._run_manage"):
            try_load_alec()
            output = capsys.readouterr().out
            assert "ALEC" in output


# ── CLI integration: tallgrass ──────────────────────────────────────────────


class TestMainCliAutoLoad:
    """The --auto-load flag on the main tallgrass CLI."""

    @pytest.fixture
    def mock_scraper(self, monkeypatch):
        instances = []

        class FakeScraper:
            def __init__(self, session, output_dir=None, delay=0.15):
                self.session = session
                self.output_dir = output_dir
                self.delay = delay
                instances.append(self)

            def clear_cache(self):
                pass

            def run(self, enrich=True):
                pass

        monkeypatch.setattr("tallgrass.cli.KSVoteScraper", FakeScraper)
        return instances

    def test_auto_load_flag_parsed(self, mock_scraper):
        """--auto-load is accepted without error."""
        from tallgrass.cli import main

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["2025", "--auto-load"])
            mock_load.assert_called_once()

    def test_auto_load_calls_try_load_session(self, mock_scraper):
        from tallgrass.cli import main

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["2025", "--auto-load"])
            call_args = mock_load.call_args[0]
            assert "91st_2025-2026" in call_args[0]

    def test_no_auto_load_by_default(self, mock_scraper):
        from tallgrass.cli import main

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["2025"])
            mock_load.assert_not_called()

    def test_list_sessions_skips_auto_load(self, mock_scraper, capsys):
        """--list-sessions exits before --auto-load would trigger."""
        from tallgrass.cli import main

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["--list-sessions", "--auto-load"])
            mock_load.assert_not_called()


# ── CLI integration: tallgrass-text ─────────────────────────────────────────


class TestTextCliAutoLoad:
    """The --auto-load flag on the tallgrass-text CLI."""

    def test_auto_load_flag_parsed(self, monkeypatch):
        """--auto-load is accepted without error."""
        from tallgrass.text.cli import main

        # Mock the entire fetch pipeline
        monkeypatch.setattr(
            "tallgrass.text.cli.KansasAdapter",
            MagicMock(return_value=MagicMock(
                discover_bills=MagicMock(return_value=[]),
            )),
        )
        # discover_bills returns empty → early exit, no auto-load
        main(["2025", "--auto-load"])

    def test_auto_load_calls_try_load_session(self, monkeypatch):
        from tallgrass.text.cli import main

        refs = [MagicMock()]
        texts = [MagicMock()]

        adapter = MagicMock()
        adapter.discover_bills.return_value = refs
        adapter.cache_dir.return_value = "/tmp/cache"
        adapter.data_dir.return_value = "/tmp/data"
        monkeypatch.setattr("tallgrass.text.cli.KansasAdapter", MagicMock(return_value=adapter))

        fetcher = MagicMock()
        fetcher.fetch_all.return_value = texts
        monkeypatch.setattr("tallgrass.text.cli.BillTextFetcher", MagicMock(return_value=fetcher))
        monkeypatch.setattr("tallgrass.text.cli.save_bill_texts", MagicMock())

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["2025", "--auto-load"])
            mock_load.assert_called_once()

    def test_no_auto_load_by_default(self, monkeypatch):
        from tallgrass.text.cli import main

        monkeypatch.setattr(
            "tallgrass.text.cli.KansasAdapter",
            MagicMock(return_value=MagicMock(discover_bills=MagicMock(return_value=[]))),
        )
        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["2025"])
            mock_load.assert_not_called()


# ── CLI integration: tallgrass-kanfocus ─────────────────────────────────────


class TestKanfocusCliAutoLoad:
    """The --auto-load flag on the tallgrass-kanfocus CLI."""

    @pytest.fixture
    def mock_kanfocus(self, monkeypatch):
        fetcher = MagicMock()
        fetcher.fetch_biennium.return_value = [MagicMock()]
        monkeypatch.setattr(
            "tallgrass.kanfocus.cli.KanFocusFetcher", MagicMock(return_value=fetcher)
        )
        monkeypatch.setattr("tallgrass.kanfocus.cli.load_existing_slugs", MagicMock(return_value={}))
        monkeypatch.setattr(
            "tallgrass.kanfocus.cli.convert_to_standard",
            MagicMock(return_value=(["v"], ["r"], ["l"])),
        )
        monkeypatch.setattr("tallgrass.kanfocus.cli.save_full", MagicMock())
        monkeypatch.setattr("tallgrass.kanfocus.cli._archive_cache", MagicMock())

    def test_auto_load_calls_try_load_session(self, mock_kanfocus):
        from tallgrass.kanfocus.cli import main

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["2025", "--auto-load"])
            mock_load.assert_called_once()

    def test_no_auto_load_by_default(self, mock_kanfocus):
        from tallgrass.kanfocus.cli import main

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["2025"])
            mock_load.assert_not_called()

    def test_list_sessions_skips_auto_load(self, mock_kanfocus, capsys):
        from tallgrass.kanfocus.cli import main

        with patch("tallgrass.db_hook.try_load_session") as mock_load:
            main(["--list-sessions", "--auto-load"])
            mock_load.assert_not_called()
