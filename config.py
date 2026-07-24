# config.py
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# =========================================================================
# USER SETTINGS — edit these for your machine / project.
# everything in section 1–3 must be set before the pipeline can run;
# sections 4–5 have working defaults
# =========================================================================

# --- 1. DFT toolchain paths (REQUIRED — replace the placeholders) --------
# Modify QE, Pseudopotentials, Wannier90, Perturbo to your system path
QE_BIN = "path/to/qe/bin"
QE_PSEUDOS_DIR = "path/to/your/pseudo-potentials/dir/"
W90_DIR = "path/to/your/wannier90/bin/"
PERTURBO_BIN = "/path/to/your/perturbo/bin/"

# --- 2. Machine resources (should match to your hardware) ------------
MPI_NPROCS = 12

# --- 3. API credentials & contact (REQUIRED for Materials Project) -------
# Set MP_API_KEY in your .env file (or export it in your shell)
MP_API_KEY = os.environ.get("MP_API_KEY", "")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")   # your email, used in the API user agent

# --- 4. Research topic (defines what the pipeline studies) ---------------
# ARXIV_QUERY = (
#     'abs:"chromium sulfur bromide" OR abs:"CrSBr" '
#     'OR abs:"van der Waals" OR abs:"magnetic semiconductor" '
#     'OR abs:"exciton-phonon" OR abs:"exciton-magnon" '
#     'OR abs:"magnon-phonon" OR abs:"coherent phonon" '
#     'OR abs:"coherent magnon" OR abs:"coupling"'
# )

ARXIV_QUERY = (
   'abs:"silicon" OR abs:"mobility" OR abs:"transport" '
   'OR abs:"electron-phonon" OR abs:"thermal conductivity" '
   'OR abs:"Perturbo" OR abs:"EPW" '
   'OR abs:"Wannier function" OR abs:"first-principles"'
)

# ARXIV_QUERY = (
#     '(abs:"wearable" OR abs:"flexible electronics" OR abs:"stretchable" '
#     'OR abs:"electronic skin" OR abs:"e-skin" OR abs:"epidermal electronics" '
#     'OR abs:"e-textile" OR abs:"smart textile") '
#     'AND '
#     '(abs:"optical" OR abs:"transparent" OR abs:"transmittance" '
#     'OR abs:"conductivity" OR abs:"carrier mobility" OR abs:"charge transport" '
#     'OR abs:"sheet resistance" OR abs:"piezoelectric" OR abs:"thermoelectric" '
#     'OR abs:"triboelectric" OR abs:"dielectric")'
# )

ARXIV_MAX_RESULTS = 400

# --- 5. Optional tuning (defaults are fine for most users) ---------------
USE_LLM_CHUNK_CLASSIFIER = False   # True = Haiku classifies sections; False = keyword rules only
MAX_DISCOVERY_ITERATIONS = 5

# Model tiers — match cost to role
PLANNER_MODEL = "claude-opus-4-7"           # We want to use the best one for planner
GENERATOR_MODEL = "claude-sonnet-4-6"       # creative role; balanced
SIMULATOR_MODEL = "claude-sonnet-4-6"       # mostly orchestrates the tool
CRITIC_MODEL = "claude-haiku-4-5"           # skeptical, narrow, cheap
REPORTER_MODEL = "claude-sonnet-4-6"        # writes the final document

# # Local LLM judge config (used in 6.5). Comment out if you'll stick with Anthropic.
# LOCAL_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# LOCAL_JUDGE_QUANTIZATION = "8bit"   # "none" | "8bit" | "4bit" — pick by VRAM

# =========================================================================
#  INTERNAL — derived settings; normally no need to edit below this line
# =========================================================================

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "data" / "papers"
PARSED_DIR = ROOT / "data" / "parsed"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"
DFT_RUNS_DIR = ROOT / "data" / "dft_runs"
W90_RUNS_DIR = ROOT / "data" / "w90_runs"
PH_RUNS_DIR = ROOT / "data" / "ph_runs"
QE2PERT_RUNS_DIR = ROOT / "data" / "qe2pert_runs"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"        # default for cpu
RERANK_MODEL = "BAAI/bge-reranker-base"       # base version of bge

# --- LangGraph configuration ---------------------------------------------
GRAPH_DB_PATH = INDEX_DIR / "checkpoints.sqlite"

# --- Materials Project API ------------------------------------------------
USER_AGENT = (
    f"oitoni/1.0 (mailto:{CONTACT_EMAIL})" if CONTACT_EMAIL
    else "oitoni/1.0"
)

# --- GPU configuration ----------------------------------------------------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = DEVICE == "cuda"

# On GPU, upgrade to the larger BGE-M3 model. Modify if OOM error
if USE_GPU:
    EMBED_MODEL = "BAAI/bge-m3"
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
    EMBED_BATCH_SIZE = 32
    RERANK_BATCH_SIZE = 32
else:
    # (leave the CPU defaults from earlier in this file untouched above)
    EMBED_BATCH_SIZE = 64
    RERANK_BATCH_SIZE = 16
