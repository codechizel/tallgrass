"""Command-line interface for bill text retrieval."""

import argparse

from tallgrass.config import BASE_URL
from tallgrass.session import CURRENT_BIENNIUM_START, SPECIAL_SESSION_YEARS, KSSession
from tallgrass.text.fetcher import BillTextFetcher
from tallgrass.text.kansas import KansasAdapter
from tallgrass.text.output import save_bill_texts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tallgrass-text",
        description="Download and extract bill text from the Kansas Legislature website.",
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        default=CURRENT_BIENNIUM_START,
        help=f"Session start year, e.g. 2025, 2023, 2021 (default: {CURRENT_BIENNIUM_START})",
    )
    parser.add_argument(
        "--special",
        action="store_true",
        help="Fetch text for a special session (e.g., 2024 special session)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default="introduced,supp_note",
        help="Comma-separated document types to fetch (default: introduced,supp_note)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached downloads before running",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List known session years and exit",
    )

    args = parser.parse_args(argv)

    if args.list_sessions:
        print("Known Kansas Legislature sessions:")
        print()
        print("  Regular sessions:")
        for start in range(CURRENT_BIENNIUM_START, 2010, -2):
            s = KSSession.from_year(start)
            print(f"    {s.label:22s}  {BASE_URL}{s.bills_path}")
        print()
        print("  Special sessions:")
        for year in SPECIAL_SESSION_YEARS:
            s = KSSession(start_year=year, special=True)
            print(f"    {s.label:22s}  {BASE_URL}{s.li_prefix}/")
        return

    # Build session string from year + special flag
    if args.special:
        session_id = f"{args.year}s"
    else:
        session = KSSession.from_year(args.year)
        session_id = f"{session.start_year}-{session.end_year % 100:02d}"

    document_types = [t.strip() for t in args.types.split(",")]
    adapter = KansasAdapter(document_types=document_types)

    session = KSSession.from_session_string(session_id)
    cache_dir = adapter.cache_dir(session_id)

    fetcher = BillTextFetcher(cache_dir=cache_dir)

    if args.clear_cache:
        fetcher.clear_cache()

    print("=" * 60)
    print(f"Bill Text Retrieval: {session.label}")
    print(f"Document types: {', '.join(document_types)}")
    print("=" * 60)

    # Step 1: Discover bills
    print("\nStep 1: Discovering bills...")
    refs = adapter.discover_bills(session_id, get_fn=fetcher.get_html)
    print(f"  Found {len(refs)} document references")

    if not refs:
        print("\nNo bills found. Exiting.")
        return

    # Step 2: Download and extract
    print("\nStep 2: Downloading and extracting text...")
    bill_texts = fetcher.fetch_all(refs)

    # Step 3: Save CSV
    print("\nStep 3: Saving CSV...")
    data_dir = adapter.data_dir(session_id)
    save_bill_texts(data_dir, session.output_name, bill_texts)

    print(f"\nDone: {len(bill_texts)} bill texts saved to {data_dir}")
