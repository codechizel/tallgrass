"""Command-line interface for KanFocus vote data retrieval."""

import argparse
import shutil
from pathlib import Path

from tallgrass.kanfocus.fetcher import DEFAULT_DELAY, KanFocusFetcher
from tallgrass.kanfocus.output import convert_to_standard, merge_gap_fill, save_full
from tallgrass.kanfocus.session import session_id_for_biennium
from tallgrass.kanfocus.slugs import load_existing_slugs
from tallgrass.session import KSSession

# Raw HTML cache is precious — takes hours to rebuild per biennium.
# Archive here after each successful run.
ARCHIVE_DIR = Path("data/kanfocus_archive")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tallgrass-kanfocus",
        description="Scrape roll call vote data from KanFocus (kanfocus.com). "
        "Produces the same CSV format as the main tallgrass scraper.",
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        help="Biennium start year (odd), e.g. 1999, 2009, 2011",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "gap-fill", "crossval"],
        default="full",
        help="full: scrape all votes (default). gap-fill: fill missing votes in existing CSVs. "
        "crossval: compare cached KF data against JE CSVs (read-only diagnostic).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between requests (default: {DEFAULT_DELAY}). "
        "KanFocus is a shared paid service — be conservative.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached pages before running.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Show KanFocus session ID mapping and exit.",
    )
    parser.add_argument(
        "--auto-load",
        action="store_true",
        help="After scraping, load CSVs into PostgreSQL (requires web dependencies + running DB)",
    )

    args = parser.parse_args(argv)

    if args.list_sessions:
        _print_sessions()
        return

    if args.year is None:
        parser.error("year is required (unless using --list-sessions)")

    session = KSSession.from_year(args.year)
    cache_dir = session.data_dir / ".cache" / "kanfocus"

    if args.mode == "crossval":
        _run_crossval(session, cache_dir)
        return

    fetcher = KanFocusFetcher(cache_dir=cache_dir, delay=args.delay)

    if args.clear_cache:
        # Refuse to clear if there's no archive — raw HTML takes hours to rebuild
        archive = ARCHIVE_DIR / session.output_name
        if not archive.exists():
            print(f"  ERROR: no archive at {archive}. Run without --clear-cache first")
            print("  to build and archive the cache, then clear it if needed.")
            return
        fetcher.clear_cache()

    print("=" * 60)
    print(f"KanFocus Vote Scraper: {session.label}")
    print(f"Mode: {args.mode}")
    print(f"Delay: {args.delay}s between requests")
    print("=" * 60)

    # Load existing slugs for cross-referencing
    existing_slugs = load_existing_slugs(session.data_dir, session.output_name)
    if existing_slugs:
        print(f"\n  Loaded {len(existing_slugs)} existing slugs for cross-reference")

    # Fetch votes
    print("\nStep 1: Fetching votes from KanFocus...")
    records = fetcher.fetch_biennium(session.start_year)

    if not records:
        print("\nNo votes found. Exiting.")
        return

    # Convert to standard format
    print("\nStep 2: Converting to standard tallgrass format...")
    votes, rollcalls, legislators = convert_to_standard(records, session.label, existing_slugs)
    print(
        f"  {len(votes)} individual votes, {len(rollcalls)} rollcalls, "
        f"{len(legislators)} legislators"
    )

    # Save
    print("\nStep 3: Saving CSV files...")
    if args.mode == "gap-fill":
        merge_gap_fill(session.data_dir, session.output_name, votes, rollcalls, legislators)
    else:
        save_full(session.data_dir, session.output_name, votes, rollcalls, legislators)

    # Archive raw HTML cache — this data takes hours to rebuild
    _archive_cache(cache_dir, session.output_name)

    print(f"\nDone: {len(rollcalls)} rollcalls saved to {session.data_dir}")

    if args.auto_load:
        from tallgrass.db_hook import try_load_session

        try_load_session(session.output_name)


def _run_crossval(session: KSSession, cache_dir: Path) -> None:
    """Run cross-validation between KanFocus cache and JE CSVs."""
    from tallgrass.kanfocus.crossval import format_report, run_crossval

    print("=" * 60)
    print(f"KanFocus Cross-Validation: {session.label}")
    print("=" * 60)

    existing_slugs = load_existing_slugs(session.data_dir, session.output_name)
    if existing_slugs:
        print(f"\n  Loaded {len(existing_slugs)} existing slugs for cross-reference")

    report = run_crossval(
        session_label=session.label,
        start_year=session.start_year,
        data_dir=session.data_dir,
        cache_dir=cache_dir,
        existing_slugs=existing_slugs,
    )

    # Write report
    report_path = session.data_dir / "crossval_report.md"
    md = format_report(report)
    report_path.write_text(md, encoding="utf-8")
    print(f"\n  Report written to {report_path}")

    # Print summary
    print(f"\n  Summary: {report.matched_rollcalls} matched rollcalls")
    if report.matched_rollcalls:
        print(f"    Tally exact match: {report.tally_perfect}")
        print(f"    Tally compatible (ANV/NV): {report.tally_compatible}")
        print(f"    Tally mismatch: {report.tally_mismatch}")
        print(f"    Passed agree: {report.passed_agree}")
        print(f"    Passed disagree: {report.passed_disagree}")
        print(f"    Individual votes perfect: {report.individual_perfect}")
        print(f"    Individual votes compatible: {report.individual_compatible}")
        print(f"    Individual votes mismatch: {report.individual_mismatch}")
        print(f"    Genuine individual mismatches: {report.total_genuine_mismatches}")


def _archive_cache(cache_dir: Path, output_name: str) -> None:
    """Copy raw HTML cache to a permanent archive location.

    The cache takes hours to rebuild per biennium. This ensures the raw data
    survives ``--clear-cache`` and accidental deletion.
    """
    archive = ARCHIVE_DIR / output_name
    html_files = list(cache_dir.glob("*.html"))
    if not html_files:
        return

    archive.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in html_files:
        dest = archive / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            copied += 1

    total = len(list(archive.glob("*.html")))
    if copied:
        print(f"\n  Archived {copied} new pages to {archive} ({total} total)")
    else:
        print(f"\n  Archive up to date: {archive} ({total} pages)")


def _print_sessions() -> None:
    """Print KanFocus session ID mapping table."""
    print("KanFocus Session ID Mapping:")
    print()
    print(f"  {'Session ID':<12} {'Legislature':<14} {'Years'}")
    print(f"  {'─' * 12} {'─' * 14} {'─' * 11}")
    for start in range(1999, 2027, 2):
        session = KSSession.from_year(start)
        sid = session_id_for_biennium(start)
        print(f"  {sid:<12} {session.legislature_name:<14} {start}-{start + 1}")
