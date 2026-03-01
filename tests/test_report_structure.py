"""
Structural tests for the HTML report pipeline.

Tests the report *skeleton* (section ordering, TOC, numbering, tag hierarchy)
without snapshotting full HTML. This catches regressions in the template and
CSS embedding without brittleness from great_tables inline CSS changes.

Run: uv run pytest tests/test_report_structure.py -v
"""

import re
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.report import (
    REPORT_CSS,
    FigureSection,
    ReportBuilder,
    TableSection,
    TextSection,
    make_gt,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_report(*sections: TableSection | FigureSection | TextSection) -> str:
    """Build a report from sections and return rendered HTML."""
    report = ReportBuilder(title="Test Report", session="91st_2025-2026")
    for section in sections:
        report.add(section)
    return report.render()


# ── TOC and Section Structure ────────────────────────────────────────────────


class TestReportStructure:
    """Structural assertions on the rendered HTML skeleton."""

    @pytest.fixture
    def three_section_html(self) -> str:
        """Report with one of each section type."""
        return _build_report(
            TableSection(id="overview", title="Session Overview", html="<table></table>"),
            FigureSection(id="margins", title="Vote Margins", image_data="AAAA"),
            TextSection(id="notes", title="Methodology Notes", html="<p>Notes here.</p>"),
        )

    def test_toc_count_matches_sections(self, three_section_html):
        toc_items = re.findall(r'<li><a href="#[\w-]+">', three_section_html)
        assert len(toc_items) == 3

    def test_toc_anchors_match_section_ids(self, three_section_html):
        toc_hrefs = re.findall(r'<li><a href="#([\w-]+)">', three_section_html)
        section_ids = re.findall(
            r'<section class="report-section" id="([\w-]+)">',
            three_section_html,
        )
        assert toc_hrefs == section_ids

    def test_section_numbering_sequential(self, three_section_html):
        numbers = re.findall(r'<span class="section-number">(\d+)\.</span>', three_section_html)
        assert numbers == ["1", "2", "3"]

    def test_all_container_types_present(self, three_section_html):
        assert 'class="table-container"' in three_section_html
        assert 'class="figure-container"' in three_section_html
        assert 'class="text-container"' in three_section_html

    def test_title_in_h1(self, three_section_html):
        assert "<h1>Test Report</h1>" in three_section_html

    def test_title_in_html_title(self, three_section_html):
        assert "<title>Test Report</title>" in three_section_html

    def test_session_in_header_meta(self, three_section_html):
        assert "91st_2025-2026" in three_section_html

    def test_footer_contains_title(self, three_section_html):
        footer_match = re.search(r"<footer>(.*?)</footer>", three_section_html, re.DOTALL)
        assert footer_match is not None
        assert "Test Report" in footer_match.group(1)

    def test_timestamp_present(self, three_section_html):
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", three_section_html)


# ── Section Ordering ────────────────────────────────────────────────────────


class TestReportSectionOrdering:
    """Sections render in insertion order."""

    def test_insertion_order_preserved(self):
        html = _build_report(
            TextSection(id="alpha", title="Alpha", html="<p>A</p>"),
            TextSection(id="beta", title="Beta", html="<p>B</p>"),
            TextSection(id="gamma", title="Gamma", html="<p>C</p>"),
        )
        assert html.index('id="alpha"') < html.index('id="beta"') < html.index('id="gamma"')

    def test_toc_order_matches_content_order(self):
        html = _build_report(
            TextSection(id="first", title="First Section", html="<p>1</p>"),
            TextSection(id="second", title="Second Section", html="<p>2</p>"),
        )
        toc_hrefs = re.findall(r'<li><a href="#([\w-]+)">', html)
        section_ids = re.findall(r'<section class="report-section" id="([\w-]+)">', html)
        assert toc_hrefs == section_ids


# ── Duplicate IDs ────────────────────────────────────────────────────────────


class TestReportDuplicateIds:
    """Duplicate section IDs don't crash rendering."""

    def test_duplicate_ids_render_without_error(self):
        html = _build_report(
            TextSection(id="dup", title="First", html="<p>1</p>"),
            TextSection(id="dup", title="Second", html="<p>2</p>"),
        )
        # Both sections present even with duplicate IDs
        assert html.count('id="dup"') >= 2

    def test_duplicate_ids_toc_has_both_entries(self):
        html = _build_report(
            TextSection(id="dup", title="First", html="<p>1</p>"),
            TextSection(id="dup", title="Second", html="<p>2</p>"),
        )
        toc_items = re.findall(r'<li><a href="#dup">', html)
        assert len(toc_items) == 2


# ── Empty Report ─────────────────────────────────────────────────────────────


class TestReportEmpty:
    """Report with no sections still renders valid HTML shell."""

    def test_empty_report_renders(self):
        report = ReportBuilder(title="Empty Report")
        html = report.render()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_empty_report_has_no_sections(self):
        report = ReportBuilder(title="Empty Report")
        html = report.render()
        assert '<section class="report-section"' not in html

    def test_empty_report_has_empty_toc(self):
        report = ReportBuilder(title="Empty Report")
        html = report.render()
        toc_items = re.findall(r"<li><a href=", html)
        assert len(toc_items) == 0

    def test_empty_report_has_header(self):
        report = ReportBuilder(title="Empty Report")
        html = report.render()
        assert "<h1>Empty Report</h1>" in html


# ── CSS Embedding ────────────────────────────────────────────────────────────


class TestReportCSS:
    """REPORT_CSS is embedded in <style> tag."""

    def test_css_in_style_tag(self):
        html = _build_report(
            TextSection(id="s1", title="S", html="<p>X</p>"),
        )
        assert "<style>" in html
        assert "</style>" in html

    def test_css_contains_key_selectors(self):
        html = _build_report(
            TextSection(id="s1", title="S", html="<p>X</p>"),
        )
        # Verify key CSS selectors from REPORT_CSS are embedded
        assert ".table-container" in html
        assert ".figure-container" in html
        assert ".text-container" in html
        assert "nav.toc" in html

    def test_report_css_constant_is_nonempty(self):
        assert len(REPORT_CSS) > 100


# ── make_gt Integration ─────────────────────────────────────────────────────


class TestMakeGtIntegration:
    """make_gt() output produces valid HTML within a full report."""

    def test_gt_table_inside_report(self):
        df = pl.DataFrame({"legislator": ["Smith", "Jones"], "score": [0.85, 0.72]})
        gt_html = make_gt(df, title="Test Scores")
        html = _build_report(
            TableSection(id="scores", title="Scores", html=gt_html),
        )
        # GT output is embedded inside the report
        assert "Test Scores" in html
        assert '<div class="table-container" id="scores">' in html
        # Report structure still intact
        assert '<a href="#scores">' in html
        assert '<span class="section-number">1.</span>' in html

    def test_gt_table_with_formatting_inside_report(self):
        df = pl.DataFrame({"name": ["A"], "value": [1.23456]})
        gt_html = make_gt(
            df,
            title="Formatted",
            number_formats={"value": ".2f"},
            source_note="Test data",
        )
        html = _build_report(
            TableSection(id="fmt", title="Formatted Table", html=gt_html),
        )
        assert "Formatted" in html
        assert "Test data" in html
