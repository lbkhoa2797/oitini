"""Build a local arXiv corpus for RAG.

Two decoupled stages so a failure in one never forces redoing the other:

  1. fetch_metadata()  -- query the arXiv API and cache the results to disk.
                          Reruns load the cache instead of re-querying, which is
                          what keeps you from tripping the rate limit. Resumable:
                          an interrupted fetch resumes from its offset, and the
                          cache is reused only when it matches the current
                          ARXIV_QUERY *and* recorded that it completed.
  2. download_pdfs()   -- download PDFs for the cached metadata. Resumable:
                          already-downloaded (and valid) PDFs are skipped.

Run:
    python scripts/01_download.py                  # uses cached metadata if present
    python scripts/01_download.py --refresh        # force a fresh API query
    python scripts/01_download.py --max-results 200
    python scripts/01_download.py --metadata-only  # hit count only, no PDFs
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
def _read_cache() -> tuple[list[dict], dict]:
    """Return (records, provenance). Tolerates the legacy bare-list cache format."""
    raw = json.loads(META_CACHE.read_text())
    if isinstance(raw, list):
        return raw, {}          # written before provenance was tracked
    return raw.get("records", []), raw


def _write_cache(records: list[dict], query: str, max_results: int,
                 total_results: int | None, complete: bool) -> None:
    """Cache records alongside enough provenance to later tell whether they are
    still valid: which query produced them, and whether that fetch finished."""
    META_CACHE.write_text(json.dumps({
        "query": query,
        "max_results": max_results,
        "total_results": total_results,
        "complete": complete,
        "records": records,
    }, indent=2))


def _total_results(client, search) -> int | None:
    """Best-effort read of arXiv's reported match count, so we know up front how
    many papers actually exist for the query. Private client API — returns None
    if those internals move."""
    try:
        feed = client._parse_feed(
            client._format_url(search, 0, client.page_size), first_page=True)
        return feed.header.total_results
    except Exception as e:
        log.debug("could not read total_results: %s", e)
        return None


def fetch_metadata(
    query: str = ARXIV_QUERY,
    max_results: int = ARXIV_MAX_RESULTS,
    refresh: bool = False,
) -> list[dict]:
    """Query arXiv and cache metadata to disk, resuming an interrupted fetch.

    The cache is reused only when it provably matches the current query AND the
    fetch that produced it ran to completion. A stale or truncated cache is
    silently wrong for every downstream stage, which costs far more than the
    re-query it saves.
    """
    resume: list[dict] = []
    if META_CACHE.exists() and not refresh:
        cached, prov = _read_cache()
        if not cached:
            # An empty cache is almost always the residue of a rate-limited run that
            # gathered nothing. Treating it as valid would wedge every future run
            # ("0 papers") until --refresh, so ignore it and re-query instead.
            log.info("Cached metadata at %s is empty; ignoring it and re-querying.", META_CACHE)
        elif not prov:
            log.warning(
                "Cache at %s records no query (written by an older version); re-querying so "
                "it cannot silently belong to a previous ARXIV_QUERY.", META_CACHE)
        elif prov.get("query") != query:
            log.warning("Cache was built from a different ARXIV_QUERY; re-querying.")
        elif not prov.get("complete"):
            resume = cached
            log.warning("Cache holds a PARTIAL fetch (%d of %s); resuming from record %d.",
                        len(cached), prov.get("total_results", "?"), len(cached))
        else:
            log.info("Loading %d cached metadata records from %s", len(cached), META_CACHE)
            return cached

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

    total = _total_results(client, search)
    target = min(max_results, total) if total is not None else max_results
    if total is not None:
        log.info("arXiv reports %d matches; fetching up to %d.", total, target)
        if total < max_results:
            log.warning(
                "Query yields only %d papers, fewer than ARXIV_MAX_RESULTS=%d. Broaden "
                "ARXIV_QUERY if you need more.", total, max_results)

    # Two distinct faults truncate a fetch, and the client survives neither:
    # a 429 (its retries re-space by ~3s and ignore Retry-After, while an arXiv
    # cooldown runs minutes) and an intermittently empty mid-pagination page
    # (raises UnexpectedEmptyPageError). Catch both here and RESUME from the
    # offset already reached — restarting from zero re-walks pages that worked,
    # and giving up caches a partial corpus as if it were the whole thing.
    records: list[dict] = list(resume)
    seen: set[str] = {r["arxiv_id"] for r in records}
    complete = False
    max_attempts = 5

    try:
        for attempt in range(max_attempts):
            try:
                for r in tqdm(client.results(search, offset=len(records)),
                              total=max(target - len(records), 0), desc="metadata"):
                    sid = r.get_short_id()
                    if sid in seen:
                        continue    # a resumed offset can re-serve a boundary record
                    seen.add(sid)
                    records.append(
                        {
                            "arxiv_id": sid,
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
                complete = True
                break  # generator exhausted without error
            except KeyboardInterrupt:
                raise
            except arxiv.HTTPError as e:
                reason = f"HTTP {e.status}"
            except Exception as e:  # UnexpectedEmptyPageError, network, parse, ...
                reason = f"{type(e).__name__}: {e}"

            # Only reachable on failure. If we already hold everything available,
            # the failure was on a page past the end — that counts as done.
            if len(records) >= target:
                complete = True
                break
            if attempt == max_attempts - 1:
                log.warning(
                    "Fetch still failing (%s) after %d attempts; keeping %d records. An arXiv "
                    "429 can persist far longer than this backoff — rerun later to resume.",
                    reason, max_attempts, len(records))
                break
            wait = min(60 * 2 ** attempt, 900)  # 60s, 120s, 240s, ... capped at 15min
            log.warning(
                "Fetch interrupted at %d/%d records (%s). Waiting %ds, then resuming (attempt %d/%d).",
                len(records), target, reason, wait, attempt + 2, max_attempts,
            )
            time.sleep(wait)
    finally:
        # Persist whatever we hold, including on Ctrl-C, so an interrupted fetch
        # is never discarded — the resume branch above picks it up next run.
        # Never overwrite a (possibly good) cache with nothing, though: a
        # 0-record result is a failure, not a corpus.
        if records:
            _write_cache(records, query, max_results, total, complete)

    if not records:
        log.error(
            "Fetched 0 metadata records (likely rate-limited). Not writing cache; "
            "wait a few minutes and rerun."
        )
        return []

    log.info("Cached %d metadata records to %s (complete=%s)", len(records), META_CACHE, complete)
    if not complete:
        log.warning("This fetch is INCOMPLETE — rerun 01_download.py to resume from record %d.",
                    len(records))
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
        expected = written = None
        try:
            resp = requests.get(url, stream=True, timeout=(10, 120), headers=headers)

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

            # stream=True means requests.get returns once the HEADERS arrive; the
            # body is pulled here. A connection dropped mid-transfer therefore
            # raises from THIS loop, not from the get() above, so it has to stay
            # inside the try -- otherwise ChunkedEncodingError escapes the retry
            # logic and aborts the entire batch on one flaky download.
            # Content-Length is only comparable when the body isn't re-encoded.
            expected = (None if resp.headers.get("Content-Encoding")
                        else resp.headers.get("Content-Length"))
            written = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    written += len(chunk)
        except requests.RequestException as e:
            tmp.unlink(missing_ok=True)   # never leave a partial file behind
            wait = 2 ** attempt
            log.warning("network error on %s: %s -- retrying in %ss", url, e, wait)
            time.sleep(wait)
            continue

        # A truncated transfer that does NOT raise still begins with %PDF- and
        # would sail past the magic-byte check, entering the corpus as a
        # half-parsed paper. Check the declared length before trusting it.
        try:
            short = expected is not None and written != int(expected)
        except ValueError:
            short = False             # unparseable Content-Length: nothing to compare
        if short:
            tmp.unlink(missing_ok=True)
            wait = 2 ** attempt
            log.warning("short read on %s (%d of %s bytes) -- retrying in %ss",
                        url, written, expected, wait)
            time.sleep(wait)
            continue

        if not _looks_like_pdf(tmp):
            log.warning("downloaded file is not a valid PDF: %s", url)
            tmp.unlink(missing_ok=True)
            return False

        tmp.replace(path)  # atomic rename: a crash never leaves a half-file at `path`
        return True

    tmp.unlink(missing_ok=True)
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
            else:
                log.info("skip %s: download failed", arxiv_id)
            # Delay after failures too: a failure is usually a rate-limit signal,
            # so skipping the pause would hammer arXiv exactly when it asked us not to.
            time.sleep(delay)
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