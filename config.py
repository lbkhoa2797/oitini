# =========================================================================
# USER SETTINGS — edit these for your machine / project.
# everything in this section must be set before the pipeline can run;
# =========================================================================
#
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# 1. Research topic - (REQUIRED for any grounded task)
# For example:
ARXIV_QUERY = (
    # Clause 1 — material-class anchor. This is what the old query lacked
    # entirely: every paper in the corpus must actually be about an oxide
    # semiconductor / transparent conductor, not merely mention one.
    '(abs:"oxide semiconductor" OR abs:"transparent conducting oxide" '
    'OR abs:"transparent conductive oxide" OR abs:"transparent oxide" '
    'OR abs:"amorphous oxide semiconductor" OR abs:"metal oxide thin film" '
    'OR abs:"oxide thin film" OR abs:"wide band gap oxide" '
    'OR abs:"wide bandgap oxide" OR abs:"ternary oxide" OR abs:"binary oxide") '
    'AND '
    # Clause 2 — the evidence the run must cite: measured or computed
    # optoelectronic properties, or a transparent/flexible device demonstration.
    '(abs:"band gap" OR abs:"bandgap" OR abs:"optical transmittance" '
    'OR abs:"visible transparency" OR abs:"optical absorption edge" '
    'OR abs:"carrier mobility" OR abs:"electron mobility" '
    'OR abs:"carrier concentration" OR abs:"sheet resistance" '
    'OR abs:"first-principles" OR abs:"density functional theory" '
    'OR abs:"electronic structure" OR abs:"thin-film transistor" '
    'OR abs:"transparent electrode" OR abs:"flexible substrate" '
    'OR abs:"flexible electronics" OR abs:"low-temperature deposition")'
)

# 2. Max number of papers to pull from the matching results
ARXIV_MAX_RESULTS = 350

# 3. Machine resources (should match to your hardware)
MPI_NPROCS = 12

# 4. Optional tuning (defaults are fine for generally easy task)
USE_LLM_CHUNK_CLASSIFIER = False   # True = Haiku classifies sections; False = keyword rules only
MAX_DISCOVERY_ITERATIONS = 5

# 5. Model tiers — match cost to role
PLANNER_MODEL = "claude-opus-4-7"           # We want to use the best one for planner
GENERATOR_MODEL = "claude-sonnet-4-6"       # creative role; balanced
SIMULATOR_MODEL = "claude-sonnet-4-6"       # mostly orchestrates the tool
CRITIC_MODEL = "claude-haiku-4-5"           # skeptical, narrow, cheap
REPORTER_MODEL = "claude-sonnet-4-6"        # writes the final document

# 6. Audit every [chunk: ...] marker in the final report against the chunk it cites.
# Costs one judge call per citation; turn off for quick iteration, not for a run
# whose citations you intend to believe.
RUN_CITATION_AUDIT = True

# =========================================================================
#  INTERNAL — derived settings; normally no need to edit below this line
# =========================================================================

# DFT toolchain paths (REQUIRED to be set in .env)
QE_BIN = os.environ.get("QE_BIN", "")
QE_PSEUDOS_DIR = os.environ.get("QE_PSEUDOS_DIR", "")
W90_DIR = os.environ.get("W90_DIR", "")
PERTURBO_BIN = os.environ.get("PERTURBO_BIN", "")

# API credentials & contact (REQUIRED for Materials Project) -------
MP_API_KEY = os.environ.get("MP_API_KEY", "")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")   # your email, used in the API user agent

# Data paths
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
