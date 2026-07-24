import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunk import chunk_all
from rag.index import build_indices
from rag.index_gpu import build_indices_gpu

from config import USE_GPU

chunk_all()

if USE_GPU:
   print("Indexing with GPU.")
   build_indices_gpu()
else:
   print("Indexing with CPU.")
   build_indices()
