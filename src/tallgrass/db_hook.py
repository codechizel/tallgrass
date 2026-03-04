"""Post-scrape hook to load CSVs into PostgreSQL via Django management commands.

Invoked via subprocess so the core scraper remains Django-free.
Fails soft — prints a warning if Django or PostgreSQL is unavailable.
"""

import subprocess
import sys


def _run_manage(args: list[str]) -> None:
    """Run a Django management command via uv, printing a warning on failure."""
    cmd = [
        sys.executable,
        "-m",
        "uv",
        "run",
        "--group",
        "web",
        "python",
        "src/web/manage.py",
        *args,
    ]
    env = {
        "DJANGO_SETTINGS_MODULE": "tallgrass_web.settings.local",
        "PYTHONPATH": "src/web",
    }
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env={**__import__("os").environ, **env},
        )
    except FileNotFoundError:
        print("  [auto-load] WARNING: uv not found — skipping database load")
        return
    except subprocess.TimeoutExpired:
        print("  [auto-load] WARNING: database load timed out (300s) — skipping")
        return

    if result.returncode == 0:
        print("  [auto-load] Database load complete")
    else:
        stderr = result.stderr.strip()
        print(f"  [auto-load] WARNING: database load failed (exit {result.returncode})")
        if stderr:
            # Show last 3 lines of stderr for diagnostics
            lines = stderr.splitlines()[-3:]
            for line in lines:
                print(f"    {line}")


def try_load_session(session_name: str, *, skip_bill_text: bool = False) -> None:
    """Attempt to load a session into PostgreSQL via Django management command.

    Fails soft — prints a warning if Django or PostgreSQL is unavailable.
    """
    print(f"\n  [auto-load] Loading {session_name} into PostgreSQL...")
    args = ["load_session", session_name]
    if skip_bill_text:
        args.append("--skip-bill-text")
    _run_manage(args)


def try_load_alec() -> None:
    """Attempt to load ALEC corpus into PostgreSQL. Fails soft."""
    print("\n  [auto-load] Loading ALEC corpus into PostgreSQL...")
    _run_manage(["load_alec"])
