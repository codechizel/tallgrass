"""CSV output for scraped vote data."""

import csv
from dataclasses import asdict, fields
from pathlib import Path

from ks_vote_scraper.models import IndividualVote, RollCall


def save_csvs(
    output_dir: Path,
    output_name: str,
    individual_votes: list[IndividualVote],
    rollcalls: list[RollCall],
    legislators: dict[str, dict],
) -> None:
    """Save all collected data to CSV files."""
    print("\n" + "=" * 60)
    print("Saving CSV files...")
    print("=" * 60)

    # Individual votes (deduplicate by legislator_slug + vote_id — ODT sessions
    # can produce duplicates when the same vote page is linked from multiple bills)
    votes_file = output_dir / f"{output_name}_votes.csv"
    seen_vote_keys: set[tuple[str, str]] = set()
    deduped_votes: list[IndividualVote] = []
    for iv in individual_votes:
        key = (iv.legislator_slug, iv.vote_id)
        if key not in seen_vote_keys:
            seen_vote_keys.add(key)
            deduped_votes.append(iv)
    if len(deduped_votes) < len(individual_votes):
        print(
            f"  Deduplicated votes: {len(individual_votes)} → {len(deduped_votes)}"
            f" ({len(individual_votes) - len(deduped_votes)} duplicates removed)"
        )
    with open(votes_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [fld.name for fld in fields(IndividualVote)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for iv in deduped_votes:
            writer.writerow(asdict(iv))
    print(f"  {votes_file} ({len(deduped_votes)} rows)")

    # Roll call summaries
    rollcalls_file = output_dir / f"{output_name}_rollcalls.csv"
    with open(rollcalls_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [fld.name for fld in fields(RollCall)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rc in rollcalls:
            writer.writerow(asdict(rc))
    print(f"  {rollcalls_file} ({len(rollcalls)} rows)")

    # Legislators
    legislators_file = output_dir / f"{output_name}_legislators.csv"
    with open(legislators_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "full_name", "slug", "chamber", "party", "district", "member_url"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for slug in sorted(legislators.keys()):
            row = {k: legislators[slug].get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"  {legislators_file} ({len(legislators)} rows)")
