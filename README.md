# Oitini - an agent equipped with first-principles tools

This repo is an AI research assistant for computational materials science, incorporating:
* LLM tool-use API
* RAG pipeline over arXiv papers
* [Materials Project](https://next-gen.materialsproject.org/) for material query

Therefore, the Anthropic API key and the MP API key are necessary to run the agent.

The project has two main components designed to work together:

1. **Perturbo Workflow Agent** — Runs end-to-end [Perturbo](https://perturbo-code.github.io/) code workflows for first-principles calculations.
2. **Closed-Loop Discovery System** — A multi-agent pipeline for autonomous material discovery.

The idea is to enforce Physics guardrails to the LLM and produce retraceable results.

Each step is cached for auditing. Thus the reproducibility is nearly 1-to-1, in case we want to publish the data from the agent.

---
### Before running:
* Have `QE`, `Wannier90` and `Perturbo` installed and their paths specified in `config.py`.
* Create the `.env` file with the `env.sample` as template in the root folder:
```bash
cp env.sample .env
```
Then enter your Anthropic API key and MP API keys:
```
ANTHROPIC_API_KEY=<your-anthropic-key>
MP_API_KEY=<your-mp-key>
```
Please navigate to [Claude](https://platform.claude.com/) and [Materials Project](https://next-gen.materialsproject.org/materials) website for how to obtain these.

#### Tracing with LangSmith
The discovery graph can stream full traces — per-node runs, LLM calls with token/cost usage, RAG and DFT tool spans — to [LangSmith](https://smith.langchain.com).
Opt-in by modifying the `.env` file with:
```
LANGSMITH_TRACING=   # Set to true to enable LangSmith tracing
LANGSMITH_API_KEY=   # API key obtained from LangSmith
LANGSMITH_PROJECT=   # Name of your awesome project
```
With the variables unset, the workflow runs exactly the same. At the end of the CLI, the link to LangSmith will be printed out or you can just search your LangSmith page.


## Features & Usage

### 1. Build an arXiv Corpus for RAG with ReAct Loop

1. In `config.py`, set `ARXIV_QUERY` to your relevant keywords (other parameters are optional).
2. Run the scripts in `scripts/` in order to download, parse, chunk, and index the corpus.
```bash
# Create the data directories where the actual QE, Wannier90 and Perturbo outputs will be written
./setup.sh
# Download the corpus from arXiv
python scripts/01_download.py
# Parse the pdf to structured JSON files
python scripts/02_parse.py
# Chunk classify each section and index with Chroma vector store (BGE embeddings, cosine) and a BM25 pickle (bm25.pkl)
python scripts/03_chunk_and_index.py 
# 04 is only for benchmarking and can be safely ignored
```

### 2. Perturbo-robot (Perturbot 🤖) Research Agent

```bash
python agent.py "Compare the Curie temperature of CrX3 family. Explain the trend and cite relevant literatures."
# or
python agent.py "Help me compute the Silicon electron-phonon database from qe2pert.x for processing with the Perturbo code."
```

Execution traces are written to `traces/`. Use `utils/trace2html.py` to render them as HTML reports. For examples:
<img width="800" height="700" alt="trace_demo" src="https://github.com/user-attachments/assets/2158a985-a57c-4d6b-9666-b17ceea423fd" />

For actual sample outputs, see the `docs/` directory or use the following links:
* [CrX3 Curie temperature](https://lbkhoa2797.github.io/Oitini/CrX3_Tc_trends.html)
* [Generating `Si_epr.h5` for Perturbo](https://lbkhoa2797.github.io/Oitini/generate_Si_epr.html)

### 3. Closed-Loop Material Discovery

A multi-agent system built with LangGraph, featuring:
- A **Planner → Generator → Simulator → Critic → Reporter** state graph
- Real first-principles calculations via the ASE/Quantum ESPRESSO bridge
- An MCP server fronting simulation and retrieval tools (currently experimental)

```bash
python -m graph.run "Find a 2D ferromagnetic semiconductor with Curie temperature > 30 K and direct bandgap > 1 eV. Cite literature for any property values."
```
As mentioned, the trace can be visualize with [LangSmith](https://smith.langchain.com):

<img width="800" height="450" alt="LangSmith_tracing" src="https://github.com/user-attachments/assets/30531e33-c0a8-456c-9fc6-6c4f8988e2bd" />

Or you can print the thoughts of the discovery agent with `show_thoughts.py`:
```
python -m graph.show_thoughts <run_id>

# For details, please use: python -m graph.show_thoughts --help
```
<img width="800" height="480" alt="showthoughts" src="https://github.com/user-attachments/assets/2fafe505-5ea5-4085-a15c-e59db736d7dc" />

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API client |
| `sentence-transformers` | BGE embeddings and reranking |
| `chromadb` | Vector database |
| `rank-bm25` | BM25 sparse retrieval |
| `docling` | PDF parsing with structure extraction |
| `arxiv` | arXiv API client |
| `mp-api` | Materials project API for crystal stuctures and relevant data |
---

Well, about the name, it's `Initio` spelling backward. I thought it sounds cool for a robot that can do `ab initio` calculations.

Later, I found out it's more like an Italian food or a cocktail (which is also cool and I like them both so I just keep it).

## Citation

If you use Perturbo, please cite:

> Jin-Jian Zhou, Jinsoo Park, I-Te Lu, Ivan Maliyov, Xiao Tong, Marco Bernardi,
> *"PERTURBO: A software package for ab initio electron–phonon interactions, charge transport and ultrafast dynamics."*
> [Comput. Phys. Commun. 264, 107970 (2021)](https://doi.org/10.1016/j.cpc.2021.107970)
