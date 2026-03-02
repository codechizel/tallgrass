"""Component-based HTML report system for analysis output.

Produces a self-contained HTML file with SPSS/APA-style tables, embedded plots,
and a navigable table of contents. Each analysis phase adds sections independently.

Six section types:
  - TableSection: Pre-rendered HTML (from great_tables). The caller builds the GT
    object via make_gt() and passes the HTML string.
  - FigureSection: Base64-embedded PNG. Classmethods for on-disk and in-memory figures.
  - TextSection: Raw HTML block.
  - InteractiveTableSection: Searchable/sortable table via ITables (offline mode).
  - InteractiveSection: Raw HTML fragment (Plotly/PyVis output).
  - KeyFindingsSection: Bullet-point summary rendered above the TOC.

ReportBuilder assembles sections into a single HTML file via a Jinja2 template.

Usage:
    from analysis.report import ReportBuilder, TableSection, FigureSection, make_gt

    report = ReportBuilder(title="EDA Report", session="2025-2026")
    report.add(TableSection(id="overview", title="Session Overview", html=make_gt(...)))
    report.add(FigureSection.from_file("margins", "Vote Margins", path))
    report.write(Path("report.html"))
"""

import base64
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment, Template

# ── Section Types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TableSection:
    """A table section containing pre-rendered HTML (typically from great_tables)."""

    id: str
    title: str
    html: str
    caption: str | None = None

    def render(self) -> str:
        parts = [f'<div class="table-container" id="{self.id}">']
        parts.append(self.html)
        if self.caption:
            parts.append(f'<p class="caption">{self.caption}</p>')
        parts.append("</div>")
        return "\n".join(parts)


@dataclass(frozen=True)
class FigureSection:
    """A figure section with a base64-embedded PNG image."""

    id: str
    title: str
    image_data: str  # base64-encoded PNG
    caption: str | None = None

    @classmethod
    def from_file(
        cls,
        id: str,
        title: str,
        path: Path,
        caption: str | None = None,
    ) -> FigureSection:
        """Create a FigureSection from a PNG file on disk."""
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return cls(id=id, title=title, image_data=b64, caption=caption)

    @classmethod
    def from_figure(
        cls,
        id: str,
        title: str,
        fig: object,
        caption: str | None = None,
        dpi: int = 150,
    ) -> FigureSection:
        """Create a FigureSection from an in-memory matplotlib Figure."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")  # type: ignore[union-attr]
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return cls(id=id, title=title, image_data=b64, caption=caption)

    def render(self) -> str:
        parts = [f'<div class="figure-container" id="{self.id}">']
        parts.append(f'<img src="data:image/png;base64,{self.image_data}" alt="{self.title}" />')
        if self.caption:
            parts.append(f'<p class="caption">{self.caption}</p>')
        parts.append("</div>")
        return "\n".join(parts)


@dataclass(frozen=True)
class TextSection:
    """A raw HTML text block."""

    id: str
    title: str
    html: str
    caption: str | None = None

    def render(self) -> str:
        parts = [f'<div class="text-container" id="{self.id}">']
        parts.append(self.html)
        if self.caption:
            parts.append(f'<p class="caption">{self.caption}</p>')
        parts.append("</div>")
        return "\n".join(parts)


@dataclass(frozen=True)
class InteractiveTableSection:
    """A searchable/sortable table rendered by ITables (offline mode)."""

    id: str
    title: str
    html: str  # pre-rendered ITables HTML
    caption: str | None = None

    def render(self) -> str:
        parts = [f'<div class="interactive-table-container" id="{self.id}">']
        parts.append(self.html)
        if self.caption:
            parts.append(f'<p class="caption">{self.caption}</p>')
        parts.append("</div>")
        return "\n".join(parts)


@dataclass(frozen=True)
class InteractiveSection:
    """A raw HTML fragment for interactive content (Plotly/PyVis output)."""

    id: str
    title: str
    html: str
    caption: str | None = None

    def render(self) -> str:
        parts = [f'<div class="interactive-container" id="{self.id}">']
        parts.append(self.html)
        if self.caption:
            parts.append(f'<p class="caption">{self.caption}</p>')
        parts.append("</div>")
        return "\n".join(parts)


@dataclass(frozen=True)
class KeyFindingsSection:
    """Bullet-point key findings rendered above the TOC."""

    findings: list[str]

    def render(self) -> str:
        items = "\n".join(f"  <li>{f}</li>" for f in self.findings)
        return f'<div class="key-findings">\n<h2>Key Findings</h2>\n<ul>\n{items}\n</ul>\n</div>'


SectionType = (
    TableSection | FigureSection | TextSection | InteractiveTableSection | InteractiveSection
)


# ── make_gt Helper ────────────────────────────────────────────────────────────


def make_gt(
    df: object,
    title: str | None = None,
    subtitle: str | None = None,
    column_labels: dict[str, str] | None = None,
    number_formats: dict[str, str] | None = None,
    source_note: str | None = None,
) -> str:
    """Build a great_tables GT object with SPSS/APA styling and return HTML string.

    Args:
        df: A polars DataFrame to display.
        title: Table title (bold, above table).
        subtitle: Subtitle (below title, smaller).
        column_labels: Mapping of column name -> display label.
        number_formats: Mapping of column name -> Python format spec (e.g. ".3f", ",.0f").
        source_note: Footnote text below the table.

    Returns:
        HTML string with inline CSS (self-contained).
    """
    import great_tables as gt_mod
    import polars as pl

    if not isinstance(df, pl.DataFrame):
        msg = f"make_gt expects a polars DataFrame, got {type(df).__name__}"
        raise TypeError(msg)

    tbl = gt_mod.GT(df)

    if title:
        tbl = tbl.tab_header(title=title, subtitle=subtitle)

    if column_labels:
        tbl = tbl.cols_label(**column_labels)

    if number_formats:
        for col_name, fmt in number_formats.items():
            if col_name in df.columns:
                tbl = tbl.fmt_number(
                    columns=col_name,
                    decimals=_decimals_from_fmt(fmt),
                    use_seps="," in fmt,
                )

    if source_note:
        tbl = tbl.tab_source_note(source_note)

    # APA/SPSS-style borders
    tbl = tbl.tab_options(
        table_border_top_style="solid",
        table_border_top_width="2px",
        table_border_top_color="#000000",
        table_border_bottom_style="solid",
        table_border_bottom_width="2px",
        table_border_bottom_color="#000000",
        column_labels_border_bottom_style="solid",
        column_labels_border_bottom_width="1px",
        column_labels_border_bottom_color="#000000",
        table_body_border_bottom_style="solid",
        table_body_border_bottom_width="1px",
        table_body_border_bottom_color="#000000",
        table_width="100%",
        table_font_size="14px",
        heading_title_font_size="16px",
        heading_subtitle_font_size="13px",
        source_notes_font_size="11px",
    )

    return tbl.as_raw_html(inline_css=True)


def _decimals_from_fmt(fmt: str) -> int:
    """Extract decimal count from a format spec like '.3f' or ',.1f'."""
    m = re.search(r"\.(\d+)f", fmt)
    return int(m.group(1)) if m else 0


# ── make_interactive_table Helper ────────────────────────────────────────────


def make_interactive_table(
    df: object,
    title: str | None = None,
    column_labels: dict[str, str] | None = None,
    number_formats: dict[str, str] | None = None,
    caption: str | None = None,
) -> str:
    """Build a searchable/sortable table via ITables and return HTML string.

    Uses ITables in offline (connected=False) mode for self-contained reports.
    All rows shown (paging=False). Client-side search and sort.

    Args:
        df: A polars DataFrame to display.
        title: Table title (rendered as <h4> above the table).
        column_labels: Mapping of column name -> display label.
        number_formats: Mapping of column name -> Python format spec (e.g. ".3f").
        caption: Caption text below the table.

    Returns:
        HTML string with embedded DataTable (self-contained).
    """
    import polars as pl
    from itables import to_html_datatable

    if not isinstance(df, pl.DataFrame):
        msg = f"make_interactive_table expects a polars DataFrame, got {type(df).__name__}"
        raise TypeError(msg)

    # Apply number formatting by creating string columns
    formatted = df
    if number_formats:
        for col_name, fmt in number_formats.items():
            if col_name in formatted.columns:
                formatted = formatted.with_columns(
                    pl.col(col_name)
                    .map_elements(
                        lambda v, f=fmt: f"{v:{f}}" if v is not None else "",
                        return_dtype=pl.String,
                    )
                    .alias(col_name)
                )

    # Rename columns for display
    if column_labels:
        rename_map = {k: v for k, v in column_labels.items() if k in formatted.columns}
        if rename_map:
            formatted = formatted.rename(rename_map)

    html_parts = []
    if title:
        html_parts.append(f"<h4>{title}</h4>")

    table_html = to_html_datatable(
        formatted,
        paging=False,
        connected=False,
        showIndex=False,
    )
    html_parts.append(table_html)

    if caption:
        html_parts.append(f'<p class="caption">{caption}</p>')

    return "\n".join(html_parts)


def _itables_init_html() -> str:
    """Return ITables offline init HTML for injection into <head>.

    With ITables >= 2.6 and connected=False, each to_html_datatable() call
    embeds the DataTables JS inline. This helper returns an empty string since
    no separate initialization is needed — the JS is bundled per-table.
    """
    return ""


# ── ReportBuilder ─────────────────────────────────────────────────────────────


@dataclass
class ReportBuilder:
    """Assembles report sections into a single self-contained HTML file."""

    title: str = "Analysis Report"
    session: str = ""
    git_hash: str = ""
    elapsed_display: str = ""
    _sections: list[tuple[str, SectionType]] = field(default_factory=list)
    _key_findings: KeyFindingsSection | None = field(default=None)

    def add(self, section: SectionType | KeyFindingsSection) -> None:
        """Append a titled section to the report.

        KeyFindingsSection is stored separately and rendered above the TOC.
        """
        if isinstance(section, KeyFindingsSection):
            self._key_findings = section
        else:
            self._sections.append((section.title, section))

    @property
    def has_sections(self) -> bool:
        return len(self._sections) > 0

    def render(self) -> str:
        """Render all sections into a complete HTML document."""
        toc_items = []
        rendered_sections = []

        for i, (title, section) in enumerate(self._sections, 1):
            toc_items.append({"number": i, "id": section.id, "title": title})
            rendered_sections.append(
                {
                    "number": i,
                    "id": section.id,
                    "title": title,
                    "content": section.render(),
                }
            )

        # Key findings rendered above the TOC
        key_findings_html = self._key_findings.render() if self._key_findings else ""

        # Inject ITables init JS if any InteractiveTableSection is present
        has_itables = any(isinstance(s, InteractiveTableSection) for _, s in self._sections)
        extra_head = _itables_init_html() if has_itables else ""

        now = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M %Z")
        template = _get_template()
        return template.render(
            title=self.title,
            session=self.session,
            git_hash=self.git_hash,
            elapsed_display=self.elapsed_display,
            generated_at=now,
            toc_items=toc_items,
            sections=rendered_sections,
            css=REPORT_CSS,
            key_findings_html=key_findings_html,
            extra_head=extra_head,
        )

    def write(self, path: Path) -> None:
        """Render and write the HTML report to disk."""
        html = self.render()
        path.write_text(html, encoding="utf-8")


# ── Template & CSS ────────────────────────────────────────────────────────────


REPORT_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 32px;
  color: #1a1a1a;
  background: #ffffff;
  line-height: 1.5;
}
header {
  border-bottom: 3px solid #1a1a1a;
  padding-bottom: 12px;
  margin-bottom: 24px;
}
header h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
header .meta { font-size: 13px; color: #555; }
header .meta span { margin-right: 16px; }
nav.toc {
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 16px 20px;
  margin-bottom: 32px;
}
nav.toc h2 { font-size: 15px; margin-bottom: 8px; color: #333; }
nav.toc ol {
  padding-left: 20px;
  column-count: 2;
  column-gap: 24px;
}
nav.toc li { font-size: 13px; margin-bottom: 4px; }
nav.toc a { color: #0066cc; text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
section.report-section {
  margin-bottom: 36px;
  page-break-inside: avoid;
}
section.report-section h2 {
  font-size: 18px;
  font-weight: 600;
  border-bottom: 2px solid #333;
  padding-bottom: 4px;
  margin-bottom: 16px;
}
.section-number {
  color: #888;
  font-weight: 400;
  margin-right: 6px;
}
.table-container {
  overflow-x: auto;
  margin-bottom: 12px;
}
.figure-container {
  text-align: center;
  margin: 12px 0;
  padding: 8px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: #fafafa;
}
.figure-container img {
  max-width: 100%;
  height: auto;
}
.text-container { margin-bottom: 12px; }
.interactive-table-container {
  overflow-x: auto;
  margin-bottom: 12px;
}
.interactive-container {
  margin: 12px 0;
  padding: 8px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: #fafafa;
}
.key-findings {
  background: #f0f7ff;
  border: 1px solid #b3d4fc;
  border-radius: 6px;
  padding: 16px 24px;
  margin-bottom: 24px;
}
.key-findings h2 {
  font-size: 17px;
  font-weight: 600;
  color: #1a3a5c;
  margin-bottom: 10px;
}
.key-findings ul {
  padding-left: 20px;
  margin: 0;
}
.key-findings li {
  font-size: 15px;
  line-height: 1.6;
  margin-bottom: 4px;
  color: #1a1a1a;
}
.caption {
  font-size: 12px;
  color: #666;
  font-style: italic;
  margin-top: 6px;
  text-align: center;
}
footer {
  margin-top: 48px;
  padding-top: 12px;
  border-top: 1px solid #ccc;
  font-size: 11px;
  color: #888;
  text-align: center;
}
@media print {
  body { max-width: none; padding: 12px; }
  nav.toc { display: none; }
  section.report-section { page-break-inside: avoid; }
}"""

REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <style>{{ css }}</style>
  {{ extra_head }}
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <div class="meta">
      {% if session %}<span>Session: <strong>{{ session }}</strong></span>{% endif %}
      <span>Generated: {{ generated_at }}</span>
      {% if elapsed_display %}<span>Runtime: {{ elapsed_display }}</span>{% endif %}
      {% if git_hash and git_hash != "unknown" %}\
<span>Git: <code>{{ git_hash[:8] }}</code></span>{% endif %}
    </div>
  </header>

  {{ key_findings_html }}

  <nav class="toc">
    <h2>Table of Contents</h2>
    <ol>
      {% for item in toc_items %}
      <li><a href="#{{ item.id }}">{{ item.title }}</a></li>
      {% endfor %}
    </ol>
  </nav>

  {% for section in sections %}
  <section class="report-section" id="{{ section.id }}">
    <h2>\
<span class="section-number">{{ section.number }}.</span> {{ section.title }}</h2>
    {{ section.content }}
  </section>
  {% endfor %}

  <footer>
    {{ title }} &mdash; {{ generated_at }}\
{% if git_hash and git_hash != "unknown" %} &mdash; {{ git_hash[:8] }}{% endif %}
  </footer>
</body>
</html>"""


def _get_template() -> Template:
    """Return a compiled Jinja2 Template."""
    env = Environment(autoescape=False)
    return env.from_string(REPORT_TEMPLATE)
