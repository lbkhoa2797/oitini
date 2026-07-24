# Rebuild corpus_index.json from the PDFs on disk, sourcing every field from the
# local metadata cache (data/metadata.json) instead of re-querying arXiv.
#
# Why off metadata.json and not the arXiv API: the metadata fetch caches all 400
# records up front; only some PDF *downloads* can fail (e.g. a 404 on an old
# paper). So the authoritative record for any PDF already lives on disk. Matching
# by slugify(arxiv_id) also sidesteps the old-vs-new arXiv ID problem -- there's
# no need to reconstruct "cond-mat/9910131v1" from a filename, which is lossy.
#
# Run:  python utils/rebuild_index.py   (works from any directory)
import json
import sys
from pathlib import Path

from slugify import slugify

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAPERS_DIR
from rag.corpus import META_CACHE, INDEX_PATH, _looks_like_pdf

# --- load the metadata cache (source of truth for record fields) -----------
if not META_CACHE.exists():
    sys.exit(f"No metadata cache at {META_CACHE}. Run scripts/01_download.py first.")

metadata = json.loads(META_CACHE.read_text())
# slug -> record, deduplicated by arxiv_id (first occurrence wins).
meta_by_slug = {}
for rec in metadata:
    slug = slugify(rec["arxiv_id"])
    meta_by_slug.setdefault(slug, rec)

# --- scan PDFs on disk, keeping only valid ones ----------------------------
pdfs_on_disk = sorted(PAPERS_DIR.glob("*.pdf"))
valid_pdfs = [p for p in pdfs_on_disk if _looks_like_pdf(p)]
corrupt = [p for p in pdfs_on_disk if p not in valid_pdfs]

# --- match each PDF to its metadata record ---------------------------------
index = []
orphans = []  # a PDF on disk with no matching record in metadata.json
for pdf in valid_pdfs:
    rec = meta_by_slug.get(pdf.stem)
    if rec is None:
        orphans.append(pdf.stem)
        continue
    index.append({**rec, "pdf_path": str(pdf)})

indexed_slugs = {pdf.stem for pdf in valid_pdfs if pdf.stem in meta_by_slug}
missing_pdfs = [slug for slug in meta_by_slug if slug not in indexed_slugs]

# --- report ----------------------------------------------------------------
print(f"Metadata records:        {len(meta_by_slug)}")
print(f"PDFs on disk:            {len(pdfs_on_disk)}")
if corrupt:
    print(f"  ! corrupt/invalid:     {len(corrupt)} (skipped)")
    for stem in (p.stem for p in corrupt):
        print(f"      {stem}")
if orphans:
    print(f"  ! no metadata match:   {len(orphans)} (skipped -- not in metadata.json)")
    for stem in orphans:
        print(f"      {stem}")
print(f"Indexed papers:          {len(index)}")
print(f"Metadata without a PDF:  {len(missing_pdfs)}")
for slug in missing_pdfs:
    print(f"      {slug}  ({meta_by_slug[slug].get('pdf_url', 'no url')})")

# --- write the rebuilt index -----------------------------------------------
INDEX_PATH.write_text(json.dumps(index, indent=2))
print(f"\nDone. Wrote {len(index)} records to {INDEX_PATH}.")
