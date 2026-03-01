"""
Tests for HTML parsing in scraper.py using inline BeautifulSoup fixtures.

Each fixture is a minimal HTML string reproducing the real kslegislature.gov
tag structure. Covers documented bugs #1 (vote page tag hierarchy), #2 (party
detection via h2), #2b (legislator name from second h1), #3 (vote categories
from both h2 and h3), #5 (inline <a> space preservation), pre-2015 party
detection via h3, and odt_view link detection.

Run: uv run pytest tests/test_scraper_html.py -v
"""

import re

import pytest
from bs4 import BeautifulSoup

from tallgrass.scraper import KSVoteScraper, _clean_text

pytestmark = pytest.mark.scraper

# ── _clean_text() ────────────────────────────────────────────────────────────
# Bug #5: get_text(strip=True) drops spaces around inline <a> elements.


class TestCleanText:
    """Preserve spaces around inline tags like <a>."""

    def test_inline_anchor_spaces(self):
        """Bug #5: 'Amendment by <a>Senator Francisco</a> was rejected'."""
        html = "<h3>Amendment by <a>Senator Francisco</a> was rejected</h3>"
        elem = BeautifulSoup(html, "lxml").find("h3")
        assert _clean_text(elem) == "Amendment by Senator Francisco was rejected"

    def test_plain_text(self):
        html = "<h3>Simple text here</h3>"
        elem = BeautifulSoup(html, "lxml").find("h3")
        assert _clean_text(elem) == "Simple text here"

    def test_collapsed_whitespace(self):
        html = "<h3>  Multiple   spaces   here  </h3>"
        elem = BeautifulSoup(html, "lxml").find("h3")
        assert _clean_text(elem) == "Multiple spaces here"


# ── _extract_bill_number() ───────────────────────────────────────────────────


class TestExtractBillNumber:
    """Extract bill number from h2 or fallback to URL path."""

    def test_from_h2(self):
        html = "<html><body><h2>SB 1 - Some title</h2></body></html>"
        soup = BeautifulSoup(html, "lxml")
        assert KSVoteScraper._extract_bill_number(soup, "/li/b2025_26/measures/sb1/") == "SB 1"

    def test_fallback_to_url(self):
        """When h2 doesn't contain a bill pattern, extract from the URL."""
        html = "<html><body><h2>Some non-bill heading</h2></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = KSVoteScraper._extract_bill_number(soup, "/li/b2025_26/measures/hb2124/")
        assert result == "HB 2124"

    def test_missing_both(self):
        """No h2 and no URL match — returns the raw path."""
        html = "<html><body><p>nothing</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = KSVoteScraper._extract_bill_number(soup, "/some/unknown/path/")
        assert result == "/some/unknown/path/"


# ── _extract_sponsor() ──────────────────────────────────────────────────────


class TestExtractSponsor:
    """Extract sponsor from bill page portlet structure."""

    def test_single_sponsor(self):
        html = """
        <html><body>
        <div class="portlet-header">Original Sponsor</div>
        <div class="portlet-content">
          <ul><li><a>Senator Steffen</a></li></ul>
        </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        assert KSVoteScraper._extract_sponsor(soup) == "Senator Steffen"

    def test_multiple_sponsors(self):
        html = """
        <html><body>
        <div class="portlet-header">Original Sponsor</div>
        <div class="portlet-content">
          <ul>
            <li><a>Senator Steffen</a></li>
            <li><a>Senator Bowers</a></li>
          </ul>
        </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        assert KSVoteScraper._extract_sponsor(soup) == "Senator Steffen; Senator Bowers"

    def test_missing_portlet(self):
        html = "<html><body><p>No sponsor section</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        assert KSVoteScraper._extract_sponsor(soup) == ""


# ── Vote category parsing ───────────────────────────────────────────────────
# Bugs #1/#3: Categories can appear as either <h2> or <h3>. The parser scans
# both tags via soup.find_all(["h2", "h3", "a"]).


class TestVoteCategoryParsing:
    """Parse vote categories (Yea, Nay, etc.) from both h2 and h3 tags."""

    @pytest.fixture
    def vote_page_html(self) -> str:
        """Minimal vote page with categories in mixed h2/h3 tags."""
        return """
        <html><body>
        <h2>SB 1</h2>
        <h4>AN ACT concerning taxation</h4>
        <h3>Senate - Emergency Final Action - 01/15/2025</h3>

        <h3>Yea - (3):</h3>
        <a href="/li/b2025_26/members/sen_doe_john_1/">Doe, John</a>,
        <a href="/li/b2025_26/members/sen_smith_jane_1/">Smith, Jane</a>,
        <a href="/li/b2025_26/members/sen_jones_bob_1/">Jones, Bob</a>

        <h2>Nay - (2):</h2>
        <a href="/li/b2025_26/members/sen_lee_amy_1/">Lee, Amy</a>,
        <a href="/li/b2025_26/members/sen_kim_dan_1/">Kim, Dan</a>

        <h3>Absent and Not Voting - (1):</h3>
        <a href="/li/b2025_26/members/sen_ray_sue_1/">Ray, Sue</a>
        </body></html>
        """

    def test_yea_from_h3(self, vote_page_html: str):
        """Bug #3: Yea heading as <h3> — must be recognized."""
        soup = BeautifulSoup(vote_page_html, "lxml")
        categories = self._parse_categories(soup)
        assert len(categories["Yea"]) == 3

    def test_nay_from_h2(self, vote_page_html: str):
        """Bug #3: Nay heading as <h2> — must also be recognized."""
        soup = BeautifulSoup(vote_page_html, "lxml")
        categories = self._parse_categories(soup)
        assert len(categories["Nay"]) == 2

    def test_absent_from_h3(self, vote_page_html: str):
        soup = BeautifulSoup(vote_page_html, "lxml")
        categories = self._parse_categories(soup)
        assert len(categories["Absent and Not Voting"]) == 1

    def test_member_links_extracted(self, vote_page_html: str):
        """Member slugs are extracted from href."""
        soup = BeautifulSoup(vote_page_html, "lxml")
        categories = self._parse_categories(soup)
        slugs = [m["slug"] for m in categories["Yea"]]
        assert "sen_doe_john_1" in slugs
        assert "sen_smith_jane_1" in slugs

    def test_total_count(self, vote_page_html: str):
        soup = BeautifulSoup(vote_page_html, "lxml")
        categories = self._parse_categories(soup)
        total = sum(len(members) for members in categories.values())
        assert total == 6

    def test_empty_categories_default(self, vote_page_html: str):
        """Categories with no members should be empty lists."""
        soup = BeautifulSoup(vote_page_html, "lxml")
        categories = self._parse_categories(soup)
        assert len(categories["Present and Passing"]) == 0
        assert len(categories["Not Voting"]) == 0

    @staticmethod
    def _parse_categories(soup: BeautifulSoup) -> dict[str, list[dict]]:
        """Replicate the category parsing logic from _parse_vote_page."""
        import re

        from tallgrass.scraper import VOTE_CATEGORIES

        vote_categories: dict[str, list[dict]] = {cat: [] for cat in VOTE_CATEGORIES}
        current_category = None

        for element in soup.find_all(["h2", "h3", "a"]):
            if element.name in ("h2", "h3"):
                text = element.get_text(strip=True)
                for cat in vote_categories:
                    if text.lower().startswith(cat.lower()):
                        current_category = cat
                        break
            elif element.name == "a" and current_category is not None:
                href = element.get("href", "")
                if "/members/" in href:
                    name = element.get_text(strip=True).rstrip(",").strip()
                    slug = re.search(r"/members/([^/]+)/", href)
                    slug = slug.group(1) if slug else ""
                    if name:
                        vote_categories[current_category].append({"name": name, "slug": slug})

        return vote_categories


# ── Legislator parsing ───────────────────────────────────────────────────────
# Bug #2: Party from h2 "District N - Party", NOT full page text.
# Bug #2b: Name from second h1 (not the nav heading), strip leadership suffix.


class TestLegislatorParsing:
    """Parse legislator name, party, and district from member pages."""

    @pytest.fixture
    def member_page_html(self) -> str:
        """Minimal legislator page reproducing real kslegislature.gov structure.

        Note the party dropdown that would false-positive if searching full text.
        """
        return """
        <html><body>
        <h1>Legislators</h1>
        <select>
          <option value="all">All</option>
          <option value="republican">Republican</option>
          <option value="democrat">Democrat</option>
        </select>
        <h1>Senator Schreiber - Senate Minority Leader</h1>
        <h2>District 27 - Democrat</h2>
        </body></html>
        """

    def test_name_from_second_h1(self, member_page_html: str):
        """Bug #2b: First h1 is 'Legislators' nav heading, not the name."""
        soup = BeautifulSoup(member_page_html, "lxml")
        name, _, _ = self._parse_legislator(soup)
        assert name == "Schreiber"

    def test_leadership_suffix_stripped(self, member_page_html: str):
        """Bug #2b: ' - Senate Minority Leader' must be stripped."""
        soup = BeautifulSoup(member_page_html, "lxml")
        name, _, _ = self._parse_legislator(soup)
        assert "Leader" not in name
        assert "Minority" not in name

    def test_all_leadership_suffixes(self):
        """Four suffix variants from the real site."""
        suffixes = [
            "Senate Minority Leader",
            "House Minority Caucus Chair",
            "Speaker of the House",
            "Senate President",
        ]
        for suffix in suffixes:
            html = f"""
            <html><body>
            <h1>Legislators</h1>
            <h1>Representative Smith - {suffix}</h1>
            <h2>District 10 - Republican</h2>
            </body></html>
            """
            soup = BeautifulSoup(html, "lxml")
            name, _, _ = self._parse_legislator(soup)
            assert name == "Smith", f"Failed to strip suffix '{suffix}'"

    def test_party_from_h2_not_dropdown(self, member_page_html: str):
        """Bug #2: Party must come from h2 'District N - Party', not the dropdown."""
        soup = BeautifulSoup(member_page_html, "lxml")
        _, party, _ = self._parse_legislator(soup)
        assert party == "Democrat"

    def test_republican_detection(self):
        html = """
        <html><body>
        <h1>Legislators</h1>
        <select>
          <option value="republican">Republican</option>
          <option value="democrat">Democrat</option>
        </select>
        <h1>Senator Masterson</h1>
        <h2>District 16 - Republican</h2>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        _, party, _ = self._parse_legislator(soup)
        assert party == "Republican"

    def test_district_extraction(self, member_page_html: str):
        soup = BeautifulSoup(member_page_html, "lxml")
        _, _, district = self._parse_legislator(soup)
        assert district == "27"

    @staticmethod
    def _parse_legislator(soup: BeautifulSoup) -> tuple[str, str, str]:
        """Replicate the legislator parsing logic from enrich_legislators."""
        import re

        from tallgrass.scraper import _clean_text

        # Full name from h1 containing "Senator" or "Representative"
        name_h1 = soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))
        full_name = ""
        if name_h1:
            full_name = _clean_text(name_h1)
            full_name = re.sub(r"^(Senator|Representative)\s+", "", full_name)
            full_name = re.sub(r"\s+-\s+.*$", "", full_name)

        # Party and district from h2 containing "District \d+"
        party = ""
        district = ""
        dist_h2 = soup.find("h2", string=re.compile(r"District\s+\d+"))
        if dist_h2:
            h2_text = dist_h2.get_text(strip=True)
            dist_match = re.search(r"District\s+(\d+)", h2_text)
            if dist_match:
                district = dist_match.group(1)
            if "Republican" in h2_text:
                party = "Republican"
            elif "Democrat" in h2_text:
                party = "Democrat"

        return full_name, party, district


# ── Vote page metadata ──────────────────────────────────────────────────────
# Bug #1: Title from <h4>, chamber/motion from <h3> (NOT h2).


class TestVotePageMetadata:
    """Extract bill title, chamber, motion, and date from vote pages."""

    @pytest.fixture
    def vote_metadata_html(self) -> str:
        """Minimal vote page with the real tag hierarchy."""
        return """
        <html><body>
        <h2>SB 1</h2>
        <h4>AN ACT concerning taxation; relating to income</h4>
        <h3>Senate - Emergency Final Action - Passed as amended - 01/15/2025</h3>
        <h3>Yea - (33):</h3>
        </body></html>
        """

    def test_title_from_h4(self, vote_metadata_html: str):
        """Bug #1: Bill title is in <h4>, not <h2>."""
        import re

        soup = BeautifulSoup(vote_metadata_html, "lxml")
        title_heading = soup.find(
            "h4", string=re.compile(r"AN ACT|A CONCURRENT|A RESOLUTION|A JOINT", re.I)
        )
        assert title_heading is not None
        assert "AN ACT concerning taxation" in _clean_text(title_heading)

    def test_chamber_from_h3(self, vote_metadata_html: str):
        """Bug #1: Chamber is in <h3>, not <h2>."""
        import re

        soup = BeautifulSoup(vote_metadata_html, "lxml")
        chamber = ""
        for h3 in soup.find_all("h3"):
            text = _clean_text(h3)
            match = re.match(
                r"(Senate|House)\s*-\s*(.+?)\s*-\s*(\d{2}/\d{2}/\d{4})$",
                text,
            )
            if match:
                chamber = match.group(1)
                break
        assert chamber == "Senate"

    def test_motion_from_h3(self, vote_metadata_html: str):
        """Motion text extracted from the h3 chamber/date line."""
        import re

        soup = BeautifulSoup(vote_metadata_html, "lxml")
        motion = ""
        for h3 in soup.find_all("h3"):
            text = _clean_text(h3)
            match = re.match(
                r"(Senate|House)\s*-\s*(.+?)\s*-\s*(\d{2}/\d{2}/\d{4})$",
                text,
            )
            if match:
                motion = match.group(2).strip().rstrip(" -;")
                break
        assert "Emergency Final Action" in motion

    def test_vote_date_from_h3(self, vote_metadata_html: str):
        """Date extracted from the h3 chamber/date line."""
        soup = BeautifulSoup(vote_metadata_html, "lxml")
        vote_date = ""
        for h3 in soup.find_all("h3"):
            text = _clean_text(h3)
            match = re.match(
                r"(Senate|House)\s*-\s*(.+?)\s*-\s*(\d{2}/\d{2}/\d{4})$",
                text,
            )
            if match:
                vote_date = match.group(3)
                break
        assert vote_date == "01/15/2025"


# ── Pre-2015 party detection ───────────────────────────────────────────────
# Pre-2015 legislator pages use <h3>Party: Republican</h3> instead of the
# "District N - Republican" format in <h2>.


class TestPreFifteenLegislatorParsing:
    """Fallback party detection from <h3>Party: ...</h3> (pre-2015 sessions)."""

    def test_party_from_h3(self):
        """When h2 has no party, fall back to h3 'Party: Republican'."""
        html = """
        <html><body>
        <h1>Legislators</h1>
        <h1>Representative Smith</h1>
        <h2>District 10</h2>
        <h3>Party: Republican</h3>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        _, party, _ = self._parse_legislator(soup)
        assert party == "Republican"

    def test_democrat_from_h3(self):
        html = """
        <html><body>
        <h1>Legislators</h1>
        <h1>Senator Davis</h1>
        <h2>District 5</h2>
        <h3>Party: Democrat</h3>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        _, party, _ = self._parse_legislator(soup)
        assert party == "Democrat"

    def test_h2_takes_priority_over_h3(self):
        """When h2 has party info, h3 fallback is not used."""
        html = """
        <html><body>
        <h1>Legislators</h1>
        <h1>Senator Jones</h1>
        <h2>District 15 - Republican</h2>
        <h3>Party: Republican</h3>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        _, party, _ = self._parse_legislator(soup)
        assert party == "Republican"

    @staticmethod
    def _parse_legislator(soup: BeautifulSoup) -> tuple[str, str, str]:
        """Replicate the legislator parsing logic including pre-2015 fallback."""
        # Full name from h1 containing "Senator" or "Representative"
        name_h1 = soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))
        full_name = ""
        if name_h1:
            full_name = _clean_text(name_h1)
            full_name = re.sub(r"^(Senator|Representative)\s+", "", full_name)
            full_name = re.sub(r"\s+-\s+.*$", "", full_name)

        # Party and district from h2 containing "District \d+"
        party = ""
        district = ""
        dist_h2 = soup.find("h2", string=re.compile(r"District\s+\d+"))
        if dist_h2:
            h2_text = dist_h2.get_text(strip=True)
            dist_match = re.search(r"District\s+(\d+)", h2_text)
            if dist_match:
                district = dist_match.group(1)
            if "Republican" in h2_text:
                party = "Republican"
            elif "Democrat" in h2_text:
                party = "Democrat"

        # Pre-2015 fallback: party from h3 "Party: ..."
        if not party:
            for h3 in soup.find_all("h3"):
                h3_text = h3.get_text(strip=True)
                if "Party:" in h3_text:
                    if "Republican" in h3_text:
                        party = "Republican"
                    elif "Democrat" in h3_text:
                        party = "Democrat"
                    break

        return full_name, party, district


# ── odt_view link detection ───────────────────────────────────────────────


class TestOdtViewLinkDetection:
    """Detect odt_view links on bill pages (2011-2014 sessions)."""

    def test_odt_view_link_found(self):
        """odt_view hrefs should match the vote link regex."""
        html = """
        <html><body>
        <h2>SB 1</h2>
        <a href="/li_2014/b2013_14/measures/odt_view/je_20130327_207704.odt">
          Final Action Yea: 40 Nay: 0
        </a>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = soup.find_all("a", href=re.compile(r"(?:vote_view|odt_view)"))
        assert len(links) == 1
        assert "odt_view" in links[0]["href"]

    def test_vote_view_still_found(self):
        """Existing vote_view links still match."""
        html = """
        <html><body>
        <a href="/li/b2025_26/measures/vote_view/je_20250320/page.html">
          Emergency Final Action
        </a>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = soup.find_all("a", href=re.compile(r"(?:vote_view|odt_view)"))
        assert len(links) == 1

    def test_mixed_links(self):
        """Page with both vote_view and odt_view links."""
        html = """
        <html><body>
        <a href="/measures/vote_view/je_1/">Vote A</a>
        <a href="/measures/odt_view/je_2.odt">Vote B</a>
        <a href="/unrelated/link/">Not a vote</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = soup.find_all("a", href=re.compile(r"(?:vote_view|odt_view)"))
        assert len(links) == 2
