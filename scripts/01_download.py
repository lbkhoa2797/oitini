import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.corpus import build_corpus, fetch_metadata
from config import PAPERS_DIR, ARXIV_MAX_RESULTS

parser = argparse.ArgumentParser(
    description="Fetch arXiv metadata and PDFs for the RAG corpus.")
parser.add_argument(
    "--refresh", action="store_true",
    help="re-query the arXiv API, ignoring any cached metadata",
)
parser.add_argument(
    "--max-results", type=int, default=ARXIV_MAX_RESULTS,
    help=f"max papers to fetch metadata for (default: ARXIV_MAX_RESULTS={ARXIV_MAX_RESULTS})",
)
parser.add_argument(
    "--metadata-only", action="store_true",
    help="stop after the metadata query — check how many papers a new ARXIV_QUERY "
         "actually returns before committing to a long PDF download",
)
args = parser.parse_args()

if args.metadata_only:
    records = fetch_metadata(max_results=args.max_results, refresh=args.refresh)
    print(f"\n{len(records)} metadata records fetched (no PDFs downloaded).")
else:
    # fetches metadata (cached) + downloads PDFs + writes the index
    records = build_corpus(max_results=args.max_results, refresh=args.refresh)
    print(f"\n{len(records)} papers placed in {PAPERS_DIR}.")
