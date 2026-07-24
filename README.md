# Materials Agent for Perturbo

An AI research assistant for computational materials science, built with Claude's tool-use API and a RAG pipeline over arXiv papers.

The project has two main components designed to work together:

1. **Perturbo Workflow Agent** — Runs end-to-end [Perturbo](https://perturbo-code.github.io/) code workflows for first-principles calculations.
2. **Closed-Loop Discovery System** — A multi-agent pipeline for autonomous material discovery.

---

## Features & Usage

### Build an arXiv Corpus

1. In `config.py`, set `ARXIV_QUERY` to your relevant keywords (other parameters are optional).
2. Run the scripts in `scripts/` in order to download, parse, chunk, and index the corpus.

### Perturbo Research Agent

```bash
python agent.py "What are the magnetic moments of each species in CrI3? Run a DFT calculation to check."
# or
python agent.py "Run a Wannier90 calculation for Si for the bands relevant to transport."
```

Execution traces are written to `traces/`. Use `utils/trace2html.py` to render them as HTML reports.

### Closed-Loop Material Discovery

A multi-agent system built with LangGraph, featuring:
- A **Planner → Generator → Simulator → Critic → Reporter** state graph
- Real first-principles calculations via the ASE/Quantum ESPRESSO bridge
- An MCP server fronting simulation and retrieval tools

```bash
python -m graph.run "Find a 2D ferromagnetic semiconductor with Curie temperature > 30 K and direct bandgap > 1 eV. Cite literature for any property values."
```

#### Tracing with LangSmith

The discovery graph can stream full traces — per-node runs, LLM calls with
token/cost usage, RAG and DFT tool spans — to [LangSmith](https://smith.langchain.com).
Tracing is opt-in: copy `.env.example` to `.env`, set `LANGSMITH_TRACING=true`,
`LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`, then run `python -m graph.run` as
usual — the CLI prints a trace URL at the end of the run. Runs resumed with
`--thread-id` are grouped in the project's **Threads** view. With the variables
unset, the workflow runs exactly as before.

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

## Citation

If you use Perturbo, please cite:

> Jin-Jian Zhou, Jinsoo Park, I-Te Lu, Ivan Maliyov, Xiao Tong, Marco Bernardi,
> *"PERTURBO: A software package for ab initio electron–phonon interactions, charge transport and ultrafast dynamics."*
> [Comput. Phys. Commun. 264, 107970 (2021)](https://doi.org/10.1016/j.cpc.2021.107970)