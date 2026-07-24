import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.corpus import build_corpus
from config import PAPERS_DIR

# fetches metadata (cached) + downloads PDFs + writes the index
records = build_corpus()
print(f"\n{len(records)} papers placed in {PAPERS_DIR}.")