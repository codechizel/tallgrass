"""
Tests for new report section types added in the report enhancement (WU0).

Covers InteractiveTableSection, InteractiveSection, KeyFindingsSection,
make_interactive_table() helper, and ReportBuilder integration.

Run: uv run pytest tests/test_report_sections.py -v
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.report import (
    InteractiveSection,
    InteractiveTableSection,
    KeyFindingsSection,
    ReportBuilder,
    TextSection,
    make_interactive_table,
)

# ── InteractiveTableSection ──────────────────────────────────────────────────


class TestInteractiveTableSection:
    """Searchable/sortable table section (ITables)."""

    def test_render_basic(self):
        section = InteractiveTableSection(id="it1", title="Scores", html="<table>data</table>")
        html = section.render()
        assert '<div class="interactive-table-container" id="it1">' in html
        assert "<table>data</table>" in html
        assert "</div>" in html

    def test_render_with_caption(self):
        section = InteractiveTableSection(
            id="it1", title="Scores", html="<table></table>", caption="All rows shown"
        )
        html = section.render()
        assert '<p class="caption">All rows shown</p>' in html

    def test_render_without_caption(self):
        section = InteractiveTableSection(id="it1", title="Scores", html="<table></table>")
        html = section.render()
        assert "caption" not in html

    def test_frozen(self):
        section = InteractiveTableSection(id="it1", title="Scores", html="<table></table>")
        with pytest.raises(AttributeError):
            section.id = "it2"  # type: ignore[misc]


# ── InteractiveSection ───────────────────────────────────────────────────────


class TestInteractiveSection:
    """Raw HTML fragment for interactive content (Plotly/PyVis)."""

    def test_render_basic(self):
        section = InteractiveSection(id="plotly1", title="Scatter", html="<div>plotly chart</div>")
        html = section.render()
        assert '<div class="interactive-container" id="plotly1">' in html
        assert "<div>plotly chart</div>" in html

    def test_render_with_caption(self):
        section = InteractiveSection(
            id="p1", title="Chart", html="<div></div>", caption="Hover for details"
        )
        html = section.render()
        assert '<p class="caption">Hover for details</p>' in html

    def test_render_without_caption(self):
        section = InteractiveSection(id="p1", title="Chart", html="<div></div>")
        html = section.render()
        assert "caption" not in html

    def test_frozen(self):
        section = InteractiveSection(id="p1", title="Chart", html="<div></div>")
        with pytest.raises(AttributeError):
            section.title = "New"  # type: ignore[misc]


# ── KeyFindingsSection ───────────────────────────────────────────────────────


class TestKeyFindingsSection:
    """Bullet-point key findings rendered above the TOC."""

    def test_render_basic(self):
        section = KeyFindingsSection(findings=["Finding 1", "Finding 2"])
        html = section.render()
        assert '<div class="key-findings">' in html
        assert "<h2>Key Findings</h2>" in html
        assert "<li>Finding 1</li>" in html
        assert "<li>Finding 2</li>" in html

    def test_render_single_finding(self):
        section = KeyFindingsSection(findings=["Only one finding."])
        html = section.render()
        assert "<li>Only one finding.</li>" in html

    def test_render_empty_findings(self):
        section = KeyFindingsSection(findings=[])
        html = section.render()
        assert '<div class="key-findings">' in html
        assert "<ul>" in html

    def test_frozen(self):
        section = KeyFindingsSection(findings=["A"])
        with pytest.raises(AttributeError):
            section.findings = ["B"]  # type: ignore[misc]


# ── make_interactive_table() ─────────────────────────────────────────────────


class TestMakeInteractiveTable:
    """ITables-powered interactive table helper."""

    def test_returns_html_string(self):
        df = pl.DataFrame({"name": ["Smith", "Jones"], "score": [0.85, 0.72]})
        result = make_interactive_table(df, title="Test Table")
        assert isinstance(result, str)
        assert "<" in result

    def test_rejects_non_polars(self):
        with pytest.raises(TypeError, match="polars DataFrame"):
            make_interactive_table({"a": [1]}, title="Bad")

    def test_title_rendered_as_h4(self):
        df = pl.DataFrame({"x": [1]})
        html = make_interactive_table(df, title="My Title")
        assert "<h4>My Title</h4>" in html

    def test_no_title_omits_h4(self):
        df = pl.DataFrame({"x": [1]})
        html = make_interactive_table(df)
        assert "<h4>" not in html

    def test_column_labels_applied(self):
        df = pl.DataFrame({"slug": ["smith"], "xi_mean": [0.5]})
        html = make_interactive_table(
            df, column_labels={"slug": "Legislator", "xi_mean": "Ideal Point"}
        )
        assert "Legislator" in html
        assert "Ideal Point" in html

    def test_caption_rendered(self):
        df = pl.DataFrame({"x": [1]})
        html = make_interactive_table(df, caption="Source: test data")
        assert '<p class="caption">Source: test data</p>' in html

    def test_contains_script_tag(self):
        """connected=False should inline the DataTables JS."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        html = make_interactive_table(df)
        assert "<script" in html

    def test_number_formats_applied(self):
        df = pl.DataFrame({"value": [1.23456]})
        html = make_interactive_table(df, number_formats={"value": ".2f"})
        assert "1.23" in html


# ── ReportBuilder with Key Findings ──────────────────────────────────────────


class TestReportBuilderKeyFindings:
    """KeyFindingsSection appears before the TOC in rendered output."""

    def test_key_findings_before_toc(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["Finding A", "Finding B"]))
        report.add(TextSection(id="s1", title="Section 1", html="<p>Content</p>"))
        html = report.render()
        kf_pos = html.index("Key Findings")
        toc_pos = html.index("Table of Contents")
        assert kf_pos < toc_pos

    def test_key_findings_not_in_toc(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["Finding A"]))
        report.add(TextSection(id="s1", title="Section 1", html="<p>Content</p>"))
        html = report.render()
        # Key findings should NOT have a TOC entry
        assert 'href="#key-findings"' not in html

    def test_key_findings_not_numbered(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["Finding A"]))
        report.add(TextSection(id="s1", title="Section 1", html="<p>Content</p>"))
        html = report.render()
        # Section numbering should start at 1 for the first real section
        assert '<span class="section-number">1.</span> Section 1' in html

    def test_no_key_findings_still_works(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert "Key Findings" not in html

    def test_has_sections_ignores_key_findings(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["A"]))
        assert report.has_sections is False  # only KF, no numbered sections


# ── ReportBuilder with InteractiveTableSection ───────────────────────────────


class TestReportBuilderInteractive:
    """InteractiveTableSection is properly rendered within a report."""

    def test_interactive_table_in_report(self):
        report = ReportBuilder(title="Test")
        report.add(InteractiveTableSection(id="it1", title="Scores", html="<table>data</table>"))
        html = report.render()
        assert '<div class="interactive-table-container" id="it1">' in html
        assert '<a href="#it1">' in html

    def test_interactive_section_in_report(self):
        report = ReportBuilder(title="Test")
        report.add(InteractiveSection(id="plotly1", title="Scatter", html="<div>chart</div>"))
        html = report.render()
        assert '<div class="interactive-container" id="plotly1">' in html


# ── CSS Styles ───────────────────────────────────────────────────────────────


class TestReportCSSNewStyles:
    """New CSS classes are present in rendered output."""

    def test_interactive_table_css(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".interactive-table-container" in html

    def test_interactive_container_css(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".interactive-container" in html

    def test_key_findings_css(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".key-findings" in html
