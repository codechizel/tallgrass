"""BillTextFetcher — concurrent document download and text extraction.

State-agnostic orchestrator: takes ``BillDocumentRef`` objects from any
adapter, downloads documents, extracts text, and returns ``BillText`` objects.
Follows the same concurrency pattern as ``KSVoteScraper`` (ThreadPoolExecutor
with rate limiting) but is self-contained.
"""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from tallgrass.config import MAX_RETRIES, MAX_WORKERS, REQUEST_DELAY, REQUEST_TIMEOUT, USER_AGENT
from tallgrass.text.extractors import clean_legislative_text, extract_pdf_text, strip_line_numbers
from tallgrass.text.models import BillDocumentRef, BillText


@dataclass
class _HtmlResult:
    """Lightweight result compatible with ``discover_bill_urls()`` interface."""

    ok: bool
    html: str | None = None


class BillTextFetcher:
    """Downloads bill documents and extracts text.

    Reuses the same concurrency patterns as the vote scraper:
    - ThreadPoolExecutor for concurrent downloads
    - Thread-safe rate limiting via Lock
    - SHA-256-keyed file cache for raw downloads

    State-agnostic: works with any ``BillDocumentRef`` regardless of source state.
    """

    def __init__(
        self,
        cache_dir: Path,
        delay: float = REQUEST_DELAY,
        max_workers: int = MAX_WORKERS,
    ):
        self.cache_dir = cache_dir
        self.delay = delay
        self.max_workers = max_workers

        self.http = requests.Session()
        self.http.headers.update({"User-Agent": USER_AGENT})
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=max_workers)
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self) -> None:
        """Thread-safe rate limiting."""
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_request_time = time.monotonic()

    def get_html(self, url: str) -> _HtmlResult:
        """Fetch an HTML page. Compatible with ``discover_bill_urls()`` interface.

        Used during bill discovery to fetch listing pages. Separate from
        ``_download()`` which handles binary PDF content.
        """
        try:
            self._rate_limit()
            resp = self.http.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return _HtmlResult(ok=True, html=resp.text)
        except requests.RequestException:
            return _HtmlResult(ok=False)

    def _download(self, url: str) -> bytes | None:
        """Download a document, using cache when available.

        Returns raw bytes on success, None on failure.
        """
        cache_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{cache_hash}.pdf"

        if cache_file.exists():
            return cache_file.read_bytes()

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self.http.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()

                content = resp.content
                # Detect HTML error pages served with 200 for PDF URLs
                if content[:5].startswith(b"<html") or content[:5].startswith(b"<HTML"):
                    return None

                try:
                    cache_file.write_bytes(content)
                except OSError:
                    pass
                return content

            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                continue

        return None

    def _process_ref(self, ref: BillDocumentRef) -> BillText | None:
        """Download and extract text from a single document reference."""
        pdf_bytes = self._download(ref.url)
        if pdf_bytes is None:
            return None

        raw_text, page_count = extract_pdf_text(pdf_bytes)
        if not raw_text:
            return None

        text = clean_legislative_text(raw_text)
        text = strip_line_numbers(text)

        return BillText(
            bill_number=ref.bill_number,
            document_type=ref.document_type,
            version=ref.version,
            session=ref.session,
            text=text,
            page_count=page_count,
            source_url=ref.url,
            extraction_method="pdfplumber",
        )

    def fetch_all(self, refs: list[BillDocumentRef]) -> list[BillText]:
        """Download and extract text from all document references.

        Uses concurrent downloads with rate limiting. Returns only
        successfully extracted documents (failures are silently skipped).
        """
        if not refs:
            return []

        results: list[BillText] = []
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ref = {executor.submit(self._process_ref, ref): ref for ref in refs}
            for future in tqdm(
                as_completed(future_to_ref),
                total=len(refs),
                desc="Extracting bill texts",
            ):
                try:
                    bill_text = future.result()
                    if bill_text is not None:
                        results.append(bill_text)
                    else:
                        failed += 1
                except Exception:
                    failed += 1

        # Sort by bill number for deterministic output
        results.sort(key=lambda bt: bt.bill_number)

        print(f"  Extracted {len(results)} documents ({failed} failed/empty)")
        return results

    def clear_cache(self) -> None:
        """Remove all cached downloads."""
        if self.cache_dir.exists():
            for f in self.cache_dir.iterdir():
                if f.is_file():
                    f.unlink()
            print(f"  Cleared cache: {self.cache_dir}")
