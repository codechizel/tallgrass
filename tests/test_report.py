"""
Tests for HTML report system in analysis/report.py.

Covers section rendering (Table, Figure, Text), format parsing, ReportBuilder
assembly, and the make_gt helper.

Run: uv run pytest tests/test_report.py -v
"""

import base64
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.report import (
    FigureSection,
    ReportBuilder,
    TableSection,
    TextSection,
    _decimals_from_fmt,
    make_gt,
)

# ── _decimals_from_fmt() ─────────────────────────────────────────────────────


class TestDecimalsFromFmt:
    """Extract decimal count from format spec."""

    def test_three_decimals(self):
        assert _decimals_from_fmt(".3f") == 3

    def test_one_decimal_with_comma(self):
        assert _decimals_from_fmt(",.1f") == 1

    def test_zero_decimals(self):
        assert _decimals_from_fmt(".0f") == 0

    def test_no_match_returns_zero(self):
        assert _decimals_from_fmt("d") == 0

    def test_plain_format(self):
        assert _decimals_from_fmt(".2f") == 2


# ── TableSection ─────────────────────────────────────────────────────────────


class TestTableSection:
    """Pre-rendered HTML table section."""

    def test_render_basic(self):
        section = TableSection(id="t1", title="Title", html="<table></table>")
        html = section.render()
        assert '<div class="table-container" id="t1">' in html
        assert "<table></table>" in html
        assert "</div>" in html

    def test_render_with_caption(self):
        section = TableSection(
            id="t1", title="Title", html="<table></table>", caption="Note"
        )
        html = section.render()
        assert '<p class="caption">Note</p>' in html

    def test_render_without_caption(self):
        section = TableSection(id="t1", title="Title", html="<table></table>")
        html = section.render()
        assert "caption" not in html

    def test_frozen(self):
        section = TableSection(id="t1", title="Title", html="<table></table>")
        with pytest.raises(AttributeError):
            section.id = "t2"  # type: ignore[misc]


# ── FigureSection ────────────────────────────────────────────────────────────


class TestFigureSection:
    """Base64-embedded PNG figure section."""

    def test_render_basic(self):
        section = FigureSection(id="f1", title="Plot", image_data="AAAA")
        html = section.render()
        assert '<div class="figure-container" id="f1">' in html
        assert "data:image/png;base64,AAAA" in html
        assert 'alt="Plot"' in html

    def test_render_with_caption(self):
        section = FigureSection(
            id="f1", title="Plot", image_data="AAAA", caption="Figure 1"
        )
        html = section.render()
        assert '<p class="caption">Figure 1</p>' in html

    def test_render_without_caption(self):
        section = FigureSection(id="f1", title="Plot", image_data="AAAA")
        html = section.render()
        assert "caption" not in html

    def test_from_file(self, tmp_path):
        """Read a PNG file and base64-encode it."""
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        png_path = tmp_path / "test.png"
        png_path.write_bytes(png_data)
        section = FigureSection.from_file("f1", "Plot", png_path)
        expected_b64 = base64.b64encode(png_data).decode("ascii")
        assert section.image_data == expected_b64
        assert section.id == "f1"
        assert section.title == "Plot"

    def test_from_file_with_caption(self, tmp_path):
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"\x89PNG")
        section = FigureSection.from_file("f1", "Plot", png_path, caption="Cap")
        assert section.caption == "Cap"

    def test_frozen(self):
        section = FigureSection(id="f1", title="Plot", image_data="AAAA")
        with pytest.raises(AttributeError):
            section.title = "New"  # type: ignore[misc]


# ── TextSection ──────────────────────────────────────────────────────────────


class TestTextSection:
    """Raw HTML text block."""

    def test_render_basic(self):
        section = TextSection(id="x1", title="Note", html="<p>Hello</p>")
        html = section.render()
        assert '<div class="text-container" id="x1">' in html
        assert "<p>Hello</p>" in html

    def test_render_with_caption(self):
        section = TextSection(
            id="x1", title="Note", html="<p>Hello</p>", caption="Note 1"
        )
        html = section.render()
        assert '<p class="caption">Note 1</p>' in html

    def test_frozen(self):
        section = TextSection(id="x1", title="Note", html="<p>Hello</p>")
        with pytest.raises(AttributeError):
            section.html = "<p>New</p>"  # type: ignore[misc]


# ── ReportBuilder ────────────────────────────────────────────────────────────


class TestReportBuilder:
    """Assembles sections into a single HTML file."""

    def test_has_sections_empty(self):
        report = ReportBuilder(title="Test")
        assert report.has_sections is False

    def test_has_sections_after_add(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S1", html="<p>Hi</p>"))
        assert report.has_sections is True

    def test_render_contains_title(self):
        report = ReportBuilder(title="My Report", session="2025-2026")
        report.add(TextSection(id="s1", title="Section One", html="<p>Content</p>"))
        html = report.render()
        assert "My Report" in html

    def test_render_contains_session(self):
        report = ReportBuilder(title="Test", session="91st_2025-2026")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert "91st_2025-2026" in html

    def test_render_contains_toc(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="intro", title="Introduction", html="<p>Hi</p>"))
        report.add(TextSection(id="body", title="Body", html="<p>Main</p>"))
        html = report.render()
        assert 'href="#intro"' in html
        assert 'href="#body"' in html
        assert "Introduction" in html
        assert "Body" in html

    def test_render_contains_section_content(self):
        report = ReportBuilder(title="Test")
        report.add(TableSection(id="t1", title="Table", html="<table>DATA</table>"))
        html = report.render()
        assert "<table>DATA</table>" in html

    def test_render_section_ordering(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="a", title="First", html="<p>1</p>"))
        report.add(TextSection(id="b", title="Second", html="<p>2</p>"))
        html = report.render()
        assert html.index("First") < html.index("Second")

    def test_render_includes_git_hash(self):
        report = ReportBuilder(title="Test", git_hash="abcdef1234567890" * 2 + "abcdef12")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert "abcdef12" in html

    def test_render_includes_elapsed_display(self):
        report = ReportBuilder(title="Test", elapsed_display="2m 15s")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert "Runtime: 2m 15s" in html

    def test_render_no_runtime_when_empty(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert "Runtime:" not in html

    def test_render_no_git_hash_when_unknown(self):
        report = ReportBuilder(title="Test", git_hash="unknown")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert "Git:" not in html

    def test_write_creates_file(self, tmp_path):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        path = tmp_path / "report.html"
        report.write(path)
        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_render_valid_html_structure(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html


# ── make_gt() ────────────────────────────────────────────────────────────────


class TestMakeGt:
    """great_tables helper for APA-style tables."""

    def test_returns_html_string(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = make_gt(df, title="Test Table")
        assert isinstance(result, str)
        assert "<" in result  # contains HTML tags

    def test_rejects_non_polars(self):
        with pytest.raises(TypeError, match="polars DataFrame"):
            make_gt({"a": [1]}, title="Bad")

    def test_title_in_output(self):
        df = pl.DataFrame({"x": [1]})
        html = make_gt(df, title="My Title")
        assert "My Title" in html

    def test_subtitle_in_output(self):
        df = pl.DataFrame({"x": [1]})
        html = make_gt(df, title="T", subtitle="Sub")
        assert "Sub" in html

    def test_source_note_in_output(self):
        df = pl.DataFrame({"x": [1]})
        html = make_gt(df, title="T", source_note="Source: test data")
        assert "Source: test data" in html

    def test_no_title_still_works(self):
        df = pl.DataFrame({"x": [1]})
        html = make_gt(df)
        assert isinstance(html, str)
        assert "<" in html

    def test_number_formats_applied(self):
        df = pl.DataFrame({"value": [1.23456]})
        html = make_gt(df, number_formats={"value": ".2f"})
        assert isinstance(html, str)
