import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.parse import parse_all
from rag.parse_gpu import parse_all_gpu

from config import USE_GPU

if USE_GPU:
   print("Parsing with GPU.")
   parse_all_gpu()
else:
   print("Parsing with CPU.")
   parse_all()