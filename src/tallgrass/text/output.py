"""CSV export for bill text data."""

import csv
from pathlib import Path

from tallgrass.text.models import BillText

# Column order for bill_texts.csv
FIELDNAMES = [
    "session",
    "bill_number",
    "document_type",
    "version",
    "text",
    "page_count",
    "source_url",
]


def save_bill_texts(
    output_dir: Path,
    output_name: str,
    bill_texts: list[BillText],
) -> Path:
    """Save extracted bill texts to CSV.

    Writes ``{output_name}_bill_texts.csv`` into ``output_dir``,
    alongside existing vote CSV files.

    Returns the path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_name}_bill_texts.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for bt in bill_texts:
            writer.writerow(
                {
                    "session": bt.session,
                    "bill_number": bt.bill_number,
                    "document_type": bt.document_type,
                    "version": bt.version,
                    "text": bt.text,
                    "page_count": bt.page_count,
                    "source_url": bt.source_url,
                }
            )

    print(f"  {csv_path} ({len(bill_texts)} rows)")
    return csv_path
