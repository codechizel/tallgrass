"""KanFocus vote data adapter — scrapes kanfocus.com vote tally pages.

Produces the same 3 CSV files as the main tallgrass scraper (votes, rollcalls,
legislators) so the analysis pipeline works unchanged. Covers 78th-91st
legislatures (1999-2026).
"""

# Cross-validation (read-only diagnostic)
from tallgrass.kanfocus.crossval import CrossValReport as CrossValReport
from tallgrass.kanfocus.crossval import normalize_bill_number as normalize_bill_number
from tallgrass.kanfocus.crossval import run_crossval as run_crossval
from tallgrass.kanfocus.fetcher import KanFocusFetcher as KanFocusFetcher
from tallgrass.kanfocus.models import KanFocusLegislator as KanFocusLegislator
from tallgrass.kanfocus.models import KanFocusVoteRecord as KanFocusVoteRecord
from tallgrass.kanfocus.parser import parse_vote_page as parse_vote_page
from tallgrass.kanfocus.session import generate_vote_id as generate_vote_id
from tallgrass.kanfocus.session import session_id_for_biennium as session_id_for_biennium
from tallgrass.kanfocus.session import vote_tally_url as vote_tally_url
from tallgrass.kanfocus.slugs import generate_slug as generate_slug
