<h1 align="center">Oitini</h1>

<p align="center">
  <strong>A closed-loop agentic system for first-principles materials discovery.</strong><br>
</p>

<p align="center">
  <a href="#license"><img alt="License: GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <a href="https://lbkhoa2797.github.io/Oitini/generate_Si_epr.html"><img alt="Live sample runs" src="https://img.shields.io/badge/sample%20runs-live-brightgreen.svg"></a>
</p>

<p align="center">
  <a href="https://github.com/user-attachments/assets/30531e33-c0a8-456c-9fc6-6c4f8988e2bd">
    <img src="https://github.com/user-attachments/assets/30531e33-c0a8-456c-9fc6-6c4f8988e2bd" width="850" alt="Rendered execution trace of an Oitini run">
  </a>
</p>

---

## Motivation
Agent-generated data is only worth having if it can be checked. Here, we puts physics-based
guardrails on the LLM and keeps every step cached and auditable, so a run reproduces almost
exactly — the bar this data has to clear before it belongs in a paper.

## Overview
The project has two main components designed to work together:
1. **Perturbo Workflow Agent** — An assistant to run [Perturbo](https://perturbo-code.github.io/) code workflows, with knowledge grounded in literature, incorporating:
    * RAG pipeline over arXiv papers
    * [Materials Project](https://next-gen.materialsproject.org/) for material query
    * LLM tool-use API (DFT, DFPT, Perturbo,...)
2. **Closed-Loop Discovery System** — A multi-agent pipeline for autonomous materials discovery.

## Architecture
 
```
                    ┌───────────┐
   discovery goal ─►│  Planner  │   target properties, search strategy, budget
                    └─────┬─────┘  + toolchain preflight (stop if required tools are not satisfied)
                          ▼     
                    ┌───────────┐   ◄── RAG: hybrid BM25 + BGE over an arXiv corpus
              ┌────►│ Generator │
              │     └─────┬─────┘
              │           ▼
              │     ┌───────────┐   ◄── Materials Project: structure, E_hull, reference gap
              │     │ Simulator │───► Quantum ESPRESSO, Wannier90, PERTURBO
              │     └─────┬─────┘
              │           │           
              │           ▼
              │     ┌───────────┐   ◄── retrieval per candidate, kept only if the chunk
              │     │ Evidence  │       actually names that material
              │     └─────┬─────┘
              │           ▼
              │     ┌───────────┐
              │     │  Critic   │   physics sanity + goal check
              │     └─────┬─────┘  
              │           │
    continue ─┘           └─ conclude ─►┌───────────┐
                                        │ Reporter  │  cites only the chunks it was given
                                        └─────┬─────┘
                                              ▼
                                        ┌───────────┐
                                        │ Verifier  │  re-reads every [chunk: …] marker
                                        └───────────┘  → faithfulness_rate
```

Routing after the Critic is deterministic Python over a structured verdict. States are checkpointed, so runs are resumable and open to human intervention mid-loop.

Simulation and retrieval tools are additionally exposed through an MCP server. (experimental 🧪)

---
## See it work

Two complete runs, rendered from real execution traces:

| Task | What the agent did | Output |
|---|---|---|
| **CrX₃ Curie-temperature trends** | Retrieved from the arXiv corpus, reasoned over the halide series, produced a cited comparison | [View trace &rarr;](https://lbkhoa2797.github.io/Oitini/CrX3_Tc_trends.html) |
| **Silicon electron&ndash;phonon database** | SCF (`pw.x`) &rarr; NSCF &rarr; Wannier90 &rarr; DFPT (`ph.x`) &rarr; `qe2pert.x`, producing `Si_epr.h5` (e-ph in Wannier gauge for PERTURBO) | [View trace &rarr;](https://lbkhoa2797.github.io/Oitini/generate_Si_epr.html) |

## Quickstart

```bash
git clone https://github.com/lbkhoa2797/oitini.git && cd oitini
pip install -r requirements.txt
cp env.sample .env          # add the API keys and your QE / Wannier90 / PERTURBO paths
./setup.sh                  # create the QE / Wannier90 / PERTURBO output directories
```

**Build a retrieval corpus.** Any literature-grounded run (the discovery loop, and literature
questions to the workflow agent) reads from a local index, so build one first.

1. In `config.py`, set `ARXIV_QUERY` to your relevant keywords (other parameters are optional).
2. Run the files in `scripts/` in order to download, parse, chunk, and index the corpus.

```bash
# Download the corpus from arXiv
python scripts/01_download.py
# Parse the pdfs into structured JSON files
python scripts/02_parse.py
# Chunk + classify each section, then index with Chroma vector store (BGE embeddings, cosine) and a BM25 pickle (bm25.pkl)
python scripts/03_chunk_and_index.py 
# 04 is only for benchmarking and can be safely ignored
```

Then run your discovery task. For example,

```bash
python -m graph.run "Find a 2D ferromagnetic semiconductor with Curie temperature > 30 K \
  and direct bandgap > 1 eV. Cite literature and verify with DFT."
```

**Requirements:** 
* `QE`, `Wannier90`, and `PERTURBO` must be installed, with their paths set in `.env`
(see `env.sample`). They live there rather than in `config.py` so that a tracked file never
carries machine-local paths. Check them up front with:
```bash
python -c "from tools.dft import preflight_dft; preflight_dft()"
```
The discovery graph runs the same check before its first model call and names whichever key is wrong.
* The API keys from [Claude](https://platform.claude.com/) and [Materials Project](https://next-gen.materialsproject.org/materials). (also through `.env`)

### (Optional) Tracing with LangSmith
The discovery graph can stream full traces per-node runs, LLM calls with token/cost usage, RAG and DFT tool spans to [LangSmith](https://smith.langchain.com).

Opt-in by modifying the `.env` file with:
```
LANGSMITH_TRACING=   # Set to true to enable LangSmith tracing
LANGSMITH_API_KEY=   # API key obtained from LangSmith
LANGSMITH_PROJECT=   # Name of your awesome project
```
Leave them unset and the workflow behaves identically.

When the run finishes, the CLI prints a link to the trace; you can also find it on your LangSmith project page.

## Features and Usage

### 1. Perturbo-robot (Perturbot 🤖) Research Assistant

```bash
python agent.py "Compare the Curie temperature of CrX3 family. Explain the trend and cite relevant literature."
# or
python agent.py "Help me compute the Silicon electron-phonon database from qe2pert.x for processing with the Perturbo code."
```

Traces land in `traces/`; render them as standalone HTML with `utils/trace2html.py`. For example:
<img width="800" height="550" alt="trace_demo" src="https://github.com/user-attachments/assets/70665036-68cf-4a8d-9100-b3bbabf1368b" />

### 2. Closed-Loop Materials Discovery Agent

A LangGraph staged multi-agent system, featuring:
- A **Planner → Generator → Simulator → Evidence → Critic → Reporter → Verifier** state graph
- Real first-principles calculations
- Per-candidate literature retrieval, so a citation is tied to the material it describes
- An automatic citation audit: every `[chunk: …]` marker in the report is re-checked against
  the chunk it cites to inspect model faithfulness

```bash
python -m graph.run "Find a 2D ferromagnetic semiconductor with Curie temperature > 30 K and direct bandgap > 1 eV. Cite literature for any property values."
```
As mentioned, the trace can be visualized with [LangSmith](https://smith.langchain.com), or you can track the agent's thoughs with `show_thoughts.py`:
```
python -m graph.show_thoughts <run_id> # --help for options
```
<img width="800" height="480" alt="showthoughts" src="https://github.com/user-attachments/assets/2fafe505-5ea5-4085-a15c-e59db736d7dc" />


## Repository layout

```
agent.py           ReAct workflow agent (Perturbot)
graph/             LangGraph closed-loop discovery system
tools/             Tool implementations: DFT, Wannier90, DFPT, qe2pert, retrieval, MP lookup
rag/               Retrieval pipeline: chunking, embedding, hybrid search
mcp_server/        MCP server fronting the simulation and retrieval tools
scripts/           Corpus build: download → parse → chunk & index
eval/              Evaluation tasks and harness
utils/             Trace rendering (trace2html.py) and helpers
docs/              Published sample outputs
config.py          Corpus settings, model tiers, tuning (machine paths live in .env)
```

## Key dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API client |
| `langgraph` | Multi-agent state machine |
| `sentence-transformers` | BGE embeddings and reranking |
| `chromadb` | Vector store |
| `rank-bm25` | Sparse retrieval |
| `docling` | PDF parsing with structure extraction |
| `arxiv` | arXiv API client |
| `mp-api` | Materials Project structures and properties |
| `ase` | Structure handling and the Quantum ESPRESSO bridge |

## About the name

It's `Initio` being spelled backward. I think it is funny for a robot that can do `ab initio` calculations.

## Citation

If you use PERTURBO, please cite:

> J.-J. Zhou, J. Park, I-T. Lu, I. Maliyov, X. Tong, and M. Bernardi,
> *PERTURBO: A software package for ab initio electron–phonon interactions, charge transport and
> ultrafast dynamics*,
> [Comput. Phys. Commun. **264**, 107970 (2021)](https://doi.org/10.1016/j.cpc.2021.107970).

## License

GPL-3.0. See [LICENSE](LICENSE).