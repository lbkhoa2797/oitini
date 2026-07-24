"""Parse PDFs into structured documents with Docling."""
import json
from tqdm import tqdm

from docling.document_converter import DocumentConverter

from config import PAPERS_DIR, PARSED_DIR

def parse_all() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    for pdf in tqdm(pdfs, desc="docling"):
        out = PARSED_DIR / f"{pdf.stem}.json"
        if out.exists():
            continue
        try:
            result = converter.convert(str(pdf))
            doc = result.document
            # We use Docling to parse a structured rep. from a scientific paper
            payload = {
                "arxiv_id_slug": pdf.stem,
                "markdown": doc.export_to_markdown(),
                "structure": doc.export_to_dict(),   # full structured tree
            }
            out.write_text(json.dumps(payload, indent=2, default=str))
        except Exception as e:
            print(f"skip {pdf.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parse_all()
    parsed = list(PARSED_DIR.glob("*.json"))
    print(f"\n{len(parsed)} parsed documents")
