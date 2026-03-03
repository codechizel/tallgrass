#!/usr/bin/env bash
# Fetch all KanFocus bienniums, most recent first.
# Gap-fill for sessions with existing data (84th-91st),
# full mode for new sessions (78th-83rd).
#
# Usage: bash scripts/kanfocus_all.sh [--delay SECONDS]
#
# Safe to interrupt and restart — cache ensures already-fetched pages are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

DELAY="${1:-12}"
if [[ "$1" == "--delay" ]]; then
    DELAY="${2:-12}"
fi

echo "=== KanFocus full backfill ==="
echo "Delay: ${DELAY}s between requests"
echo "Started: $(date)"
echo ""

# 91st-84th: gap-fill (existing kslegislature.gov data)
for YEAR in 2025 2023 2021 2019 2017 2015 2013 2011; do
    echo "--- $(date '+%H:%M') Starting ${YEAR} (gap-fill) ---"
    uv run tallgrass-kanfocus "$YEAR" --mode gap-fill --delay "$DELAY" || {
        echo "WARNING: ${YEAR} failed, continuing..."
    }
    echo ""
done

# 83rd-78th: full mode (no existing data)
for YEAR in 2009 2007 2005 2003 2001 1999; do
    echo "--- $(date '+%H:%M') Starting ${YEAR} (full) ---"
    uv run tallgrass-kanfocus "$YEAR" --mode full --delay "$DELAY" || {
        echo "WARNING: ${YEAR} failed, continuing..."
    }
    echo ""
done

echo "=== All bienniums complete: $(date) ==="
