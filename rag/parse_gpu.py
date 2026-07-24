"""Parse PDFs into structured documents with Docling. GPU-aware."""
import json
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

from config import PAPERS_DIR, PARSED_DIR, USE_GPU

def _make_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA if USE_GPU else AcceleratorDevice.CPU,
    )
    # Enable layout-aware extraction; skip OCR unless the PDF is image-based
    # (Docling auto-decides when to invoke it).
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

def parse_all_gpu() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    converter = _make_converter()

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    for pdf in tqdm(pdfs, desc=f"docling ({'gpu' if USE_GPU else 'cpu'})"):
        out = PARSED_DIR / f"{pdf.stem}.json"
        if out.exists():
            continue
        try:
            result = converter.convert(str(pdf))
            doc = result.document
            payload = {
                "arxiv_id_slug": pdf.stem,
                "markdown": doc.export_to_markdown(),
                "structure": doc.export_to_dict(),
            }
            out.write_text(json.dumps(payload, indent=2, default=str))
        except Exception as e:
            print(f"skip {pdf.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parse_all_gpu()
    print(f"\n{len(list(PARSED_DIR.glob('*.json')))} parsed documents")
