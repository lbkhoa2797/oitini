"""Build a local arXiv corpus for RAG.

Two decoupled stages so a failure in one never forces redoing the other:

  1. fetch_metadata()  -- query the arXiv API ONCE and cache the results to
                          disk. Reruns load the cache instead of re-querying,
                          which is what keeps you from tripping the rate limit.
  2. download_pdfs()   -- download PDFs for the cached metadata. Resumable:
                          already-downloaded (and valid) PDFs are skipped.

Run:
    python scripts/01_download.py                # uses cached metadata if present
    python scripts/01_download.py --refresh      # force a fresh API query
    python scripts/01_download.py --max-results 200
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import arxiv
import requests
from slugify import slugify
from tqdm import tqdm

from config import PAPERS_DIR, ARXIV_QUERY, ARXIV_MAX_RESULTS, USER_AGENT

# --- constants -------------------------------------------------------------
META_CACHE = PAPERS_DIR.parent / "metadata.json"        # stage 1 output
INDEX_PATH = PAPERS_DIR.parent / "corpus_index.json"    # stage 2 output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("corpus")


# --- stage 1: metadata -----------------------------------------------------
def fetch_metadata(
    query: str = ARXIV_QUERY,
    max_results: int = ARXIV_MAX_RESULTS,
    refresh: bool = False,
) -> list[dict]:
    """Query arXiv once and cache metadata to disk. Loads cache on rerun."""
    if META_CACHE.exists() and not refresh:
        cached = json.loads(META_CACHE.read_text())
        # An empty cache is almost always the residue of a rate-limited run that
        # gathered nothing. Treating it as a valid cache would wedge every future
        # run ("0 papers") until calling --refresh, so ignore it and re-query instead.
        if cached:
            log.info("Loading %d cached metadata records from %s", len(cached), META_CACHE)
            return cached
        log.info("Cached metadata at %s is empty; ignoring it and re-querying.", META_CACHE)

    # fewer requests = fewer chances to hit an empty page
    # delay_seconds=3 according to arXiv's requested minimum spacing.
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    # Identify ourselves on the metadata requests
    session = getattr(client, "_session", None)
    if session is not None:
        session.headers.update({"User-Agent": USER_AGENT})

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    # The arxiv client retries internally, but only re-spaces by delay_seconds
    # (~3s) and ignores Retry-After, so a 429 exhausts all its retries within
    # seconds and raises. An arXiv 429 cooldown is on the order of minutes, so we
    # wrap the whole fetch in our own loop that actually waits it out before
    # re-querying from the top.
    records: list[dict] = []
    max_429_retries = 5
    for attempt in range(max_429_retries):
        records = []
        try:
            for r in tqdm(client.results(search), total=max_results, desc="metadata"):
                records.append(
                    {
                        "arxiv_id": r.get_short_id(),
                        "pdf_url": r.pdf_url,
                        "title": r.title,
                        "authors": [a.name for a in r.authors],
                        "abstract": r.summary,
                        "year": r.published.year,
                        "primary_category": r.primary_category,
                        "categories": r.categories,
                        "doi": r.doi,
                    }
                )
            break  # iterated to completion without error
        except KeyboardInterrupt:
            raise
        except arxiv.HTTPError as e:
            if e.status == 429 and attempt < max_429_retries - 1:
                wait = min(60 * 2 ** attempt, 900)  # 60s, 120s, 240s, ... capped at 15min
                log.warning(
                    "arXiv rate-limited us (HTTP 429). Waiting %ds before retry %d/%d.",
                    wait, attempt + 1, max_429_retries - 1,
                )
                time.sleep(wait)
                continue
            log.warning("Metadata fetch failed (%s). Keeping %d records gathered.", e, len(records))
            break
        except Exception as e:  # UnexpectedEmptyPageError, network, parse, ...
            log.warning(
                "Metadata fetch interrupted (%s). Keeping %d records gathered so far.",
                e,
                len(records),
            )
            break

    # Never overwrite a (possibly good) cache with nothing: a 0-record result is
    # a failure, not a corpus. Leave any existing cache untouched and bail.
    if not records:
        log.error(
            "Fetched 0 metadata records (likely rate-limited). Not writing cache; "
            "wait a few minutes and rerun."
        )
        return []

    META_CACHE.write_text(json.dumps(records, indent=2))
    log.info("Cached %d metadata records to %s", len(records), META_CACHE)
    return records


# --- stage 2: PDFs ---------------------------------------------------------
def _looks_like_pdf(path: Path) -> bool:
    """A real PDF starts with the %PDF- magic bytes. arXiv sometimes returns
    an HTML error/processing page with a 200, so check before trusting it."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def download_pdf(url: str, path: Path, max_retries: int = 5) -> bool:
    """Download one PDF with backoff, written atomically. Returns True on success."""
    headers = {"User-Agent": USER_AGENT}
    tmp = path.with_suffix(path.suffix + ".tmp")

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=(10, 120), headers=headers)
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("network error on %s: %s -- retrying in %ss", url, e, wait)
            time.sleep(wait)
            continue

        # Rate limited: honor Retry-After, else exponential backoff.
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** attempt * 5))
            log.warning("429 rate limited; waiting %ss", wait)
            time.sleep(wait)
            continue

        # Temporary server-side conditions: back off and retry.
        if resp.status_code in (403, 503):
            wait = 2 ** attempt * 10
            log.warning("HTTP %s from arXiv; backing off %ss", resp.status_code, wait)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            log.warning("unexpected HTTP %s for %s", resp.status_code, url)
            return False

        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        if not _looks_like_pdf(tmp):
            log.warning("downloaded file is not a valid PDF: %s", url)
            tmp.unlink(missing_ok=True)
            return False

        tmp.replace(path)  # atomic rename: a crash never leaves a half-file at `path`
        return True

    log.warning("gave up on %s after %d retries", url, max_retries)
    return False


def download_pdfs(metadata: list[dict], delay: float = 3.0) -> list[dict]:
    """Download PDFs for each metadata record. Skips ones already on disk.

    Writes the index on exit (including on Ctrl-C) so progress is never lost.
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    def flush() -> None:
        INDEX_PATH.write_text(json.dumps(index, indent=2))

    try:
        for rec in tqdm(metadata, desc="pdfs"):
            arxiv_id = rec["arxiv_id"]
            pdf_path = PAPERS_DIR / f"{slugify(arxiv_id)}.pdf"

            # Resume: trust an existing file only if it's a valid PDF.
            if pdf_path.exists() and _looks_like_pdf(pdf_path):
                index.append({**rec, "pdf_path": str(pdf_path)})
                continue

            url = rec.get("pdf_url")
            if not url:
                log.info("skip %s: no PDF URL", arxiv_id)
                continue

            if download_pdf(url, pdf_path):
                index.append({**rec, "pdf_path": str(pdf_path)})
                time.sleep(delay)  # be polite between downloads
            else:
                log.info("skip %s: download failed", arxiv_id)
    finally:
        flush()

    return index


# --- orchestration ---------------------------------------------------------
def build_corpus(
    query: str = ARXIV_QUERY,
    max_results: int = ARXIV_MAX_RESULTS,
    refresh: bool = False,
) -> list[dict]:
    metadata = fetch_metadata(query, max_results, refresh=refresh)
    if not metadata:
        log.error("No metadata fetched. Check the query, or wait out a rate limit.")
        return []

    index = download_pdfs(metadata)
    log.info("Done: %d/%d papers downloaded, index at %s",
             len(index), len(metadata), INDEX_PATH)
    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-query the arXiv API, ignoring cached metadata",
    )
    parser.add_argument(
        "--max-results", type=int, default=ARXIV_MAX_RESULTS,
        help="max papers to fetch metadata for",
    )
    args = parser.parse_args()

    build_corpus(max_results=args.max_results, refresh=args.refresh)