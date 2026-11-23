<h1 align="center">Oitini</h1>

<p align="center">
  <strong>A closed-loop agentic system for first-principles materials discovery.</strong><br>
</p>

<p align="center">
<em>Note: This public demo is an early release. The latest version, with additional features, is available on request.</em>
</p>

<p align="center">
  <a href="#license"><img alt="License: GPL-3.0"
    src="https://img.shields.io/badge/license-GPL--3.0-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <a href="https://lbkhoa2797.github.io/oitini/Si.html"><img alt="Live sample runs"
    src="https://img.shields.io/badge/sample%20runs-live-brightgreen.svg"></a>
</p>

<p align="center">
  <a href="https://github.com/user-attachments/assets/30531e33-c0a8-456c-9fc6-6c4f8988e2bd">
    <img src="https://github.com/user-attachments/assets/30531e33-c0a8-456c-9fc6-6c4f8988e2bd"
      width="850" alt="Rendered execution trace of an Oitini run">
  </a>
</p>

---

## Motivation
Agent-generated data is only worth having if it can be checked. Here, we put physics-based
guardrails on the LLM and keep every step cached and auditable, so a run reproduces almost exactly —
this is the bar that this agent has to clear before it belongs in a publication.

## Overview
The project has two main components designed to work together:
1. **Perturbo Workflow Agent** — An assistant to run [Perturbo](https://perturbo-code.github.io/)
   code workflows, with knowledge grounded in literature, incorporating:
    * RAG pipeline over arXiv papers
    * [Materials Project](https://next-gen.materialsproject.org/) for material query
    * LLM tool-use API (DFT, DFPT, Perturbo,...)
2. **Closed-Loop Discovery System** — A multi-agent pipeline for autonomous materials discovery.

The repo is co-authored by: Anthropic Claude.

## Architecture
 
```
                        ┌───────────┐
      A user request ─► │  Planner  │   target properties, search strategy, budget
                        └─────┬─────┘   + toolchain preflight (stop if a tool is missing)
                              ▼
                        ┌───────────┐
   ┌───────────────────►│ Generator │   ◄── RAG: hybrid BM25 + BGE over an arXiv corpus
   │                    └─────┬─────┘
   │                          ▼
   │                    ┌───────────┐   
   │    ┌──────────────►│ Protocol  │   ──► one resolved Protocol per candidate, with   
   │    │               └─────┬─────┘       per-field provenance; bad proposals rejected
   │    │                     ▼
   │    │               ┌───────────┐   
   │    │               │ Simulator │   ◄── Materials Project: structure, E_hull, ref. gap
   │    │               └─────┬─────┘   ──► Quantum ESPRESSO `pw.x` — zero model calls,
   │    │                     ▼             parameters arrive already resolved
   │    │               ┌───────────┐   
   │    │               │  EphPlan  │   ──► LLM call to judge nbnd, num_wann, windows, ...
   │    │               └─────┬─────┘       from the scf results
   │    │                     ▼
   │    │               ┌───────────┐   
   │    │               │ Transport │   ──► Wannier90 → `ph.x` → `qe2pert.x` → PERTURBO.
   │    │               └─────┬─────┘       Model-free, blocks for hours; every stage
   │    │                     ▼             caches, so a killed chain resumes
   │    │               ┌───────────┐
   │    │               │ Evidence  │   ◄── retrieval per candidate, kept only if the chunk       
   │    │               └─────┬─────┘       actually names that material
   │    │                     ▼
   │    │               ┌───────────┐
   │    │               │  Critic   │   physics sanity, goal check, promote / demote
   │    │               └─────┬─────┘
   │    │  promote            │
   │    └─────────────────────┤   re-run the survivors one accuracy tier up
   │       continue           │
   └──────────────────────────┤             ┌───────────┐
                              └─ conclude ─►│ Reporter  │  cites only the chunks it got
                                            └─────┬─────┘
                                                  ▼
                                            ┌───────────┐
                                            │ Verifier  │  re-reads every [chunk: …] marker
                                            └───────────┘  → faithfulness_rate
```
## Design
**One protocol sets every parameter.** All Quantum ESPRESSO inputs come from a single protocol,
built in three layers: an accuracy tier, then material rules (cutoffs from the pseudopotentials,
k-grids from the cell), then literature values, which are accepted only if whitelisted and within
bounds. Every field records which layer set it.

**Each stage builds on a verified record of the last.** The chain runs SCF → NSCF and phonons →
Wannierization → electron–phonon interpolation. Each stage records the protocol and environment
that produced it, so changing one setting re-runs only the stages it affects. A record that can't
be verified counts as a cache miss, so stale results are never reused.

**The output is transport coefficients.** Depending on the request, the agent can drive Perturbo
to solve the Boltzmann transport equation for conductivity, mobility, Seebeck coefficient and
electronic thermal conductivity.

**Transport settings are derived per material.** The band range, energy window and transport
k-grid are always computed fresh, never copied from papers.

**Every SCF starts from a relaxed cell.** Structures from Materials Project are relaxed with the
pseudopotentials' functional, so the geometry and results are consistent. Spin-polarised systems
aren't relaxed yet.

**Three accuracy tiers keep sweeps affordable.** Screening is about 8× cheaper and skips
electron–phonon work. Production and converged add accuracy but more computationally demanding.
Promotions and batch sizes are capped, and every truncation is recorded.

**Cost limits are reported as cost decisions.** An optional cap on cell size can stop expensive
electron–phonon chains. The workflow reports this as a cost decision, not a verdict on the
candidate. This prevents the agent got stuck too long in an expensive candidate while high-throughput
screening.

**The Perturbo chain, stage by stage.**

At the end of each stage, the results are recorded and can be reused between runs if the request is
determined to be the same. Each folder is tagged with one or two hashkeys: its own and its
precursor.

- **DFT (`run_dft`)**
  - The SCF outputs and a `tmp/` folder: QE's XML, charge density and wavefunctions.
- **Wannier90 (`run_wannier90`)**
  - An NSCF run folder: wavefunctions on the full grid, for the e-ph matrices.
  - A Wannier90 run folder: the centres and rotation matrices (`_u_dis.mat` only with
    disentanglement), used to rotate the e-ph matrices to Wannier gauge.
- **DFPT (`create_ph_save`)**
  - A `save/` directory collected for `qe2pert.x`: `<prefix>.phsave/` with the displacement
    patterns, the dynamical matrices, and the phonon perturbation potentials
  - A `tmp/_ph0/` directory, ph.x's checkpoint, so a run stopped at its wall-clock limit resumes
    instead of restarting.
  - **Note**: The ph.x k-mesh is the finest whose projected cost fits `PH_BUDGET_S`; when none fits,
    the stage returns `status='over_budget'` instead of running.
- **qe2pert (`run_qe2pert`)**
  - Its inputs are symbolic links, not copied: the NSCF wavefunctions, the Wannier rotation matrices
    and centres, and the DFPT `save/` directory.
  - `<prefix>_epr.h5`, the e-ph matrices in Wannier gauge: the electron-phonon database that
    `perturbo.x` reads. The acoustic sum rule is re-imposed here (`crystal` for bulk, `all` for a
    monolayer), since ph.x imposes none.
  - A new Wannier gauge under the same cache key is a cache miss, so a re-Wannierisation never gets
    an old `<prefix>_epr.h5`.
- **Transport (`run_perturbo`)**
  - Its input, `<prefix>_epr.h5` from `run_qe2pert` (the e-ph matrices in Wannier gauge), is linked
    in rather than copied.
  - A `setup/` directory: `<prefix>_tet.h5`, the interpolation grid and tetrahedra inside the energy
    window, linked into every solve that reuses that window.
  - A `trans/` directory: `<prefix>_trans-rta.yml` or `<prefix>_trans-ita.yml`, from which the
    conductivity, mobility, Seebeck and thermal-conductivity tensors are read.
- **Retrieval (`literature_search`)**
  - Nothing is written per run. It reads the corpus index in `data/corpus_<label>/index/`: `chroma/`
    (BGE embeddings) and `bm25.pkl` (keywords), built once with `scripts/`.
  - Each hit carries a `chunk_id`, `arxiv_id`, title, year, section and score. A literature value
    passed on to `run_dft` or `run_perturbo` must cite its `chunk_id` with a verbatim quote, which
    is checked against the corpus.

Simulation and retrieval tools are additionally exposed through an MCP server. (experimental 🧪)


---
## See it work

Four complete runs, rendered from real execution traces:

| Task | What the agent did | Output |
|---|---|---|
| **CrX₃ Curie-temperature trends** | Retrieved from the arXiv corpus, reasoned over the halide series, produced a cited comparison | [View trace &rarr;](https://lbkhoa2797.github.io/oitini/CrX3_Tc_trends.html) |
| **Silicon electron mobility** (300 K, 10¹⁶ cm⁻³) | SCF (`pw.x`) &rarr; NSCF &rarr; Wannier90 &rarr; DFPT (`ph.x`) &rarr; `qe2pert.x` &rarr; `perturbo.x`: 1371 cm²/V/s (ITA), compared at matched conditions with a cited ab initio 1381 cm²/V/s | [View trace &rarr;](https://lbkhoa2797.github.io/oitini/Si.html) |
| **Aluminium resistivity** (300 K) | The same chain on the metal path (μ fixed at the DFT Fermi level): 2.46 μΩ·cm, compared with a cited EPW 2.73 μΩ·cm and a study of thermal-expansion effects | [View trace &rarr;](https://lbkhoa2797.github.io/oitini/Al.html) |
| **Graphene electron mobility** (monolayer, 300 K, 10¹² cm⁻²) | The same chain on the semimetal path (electron band only): 2.04 × 10⁵ cm²/V/s (ITA), compared with a cited ab initio intrinsic 1.67 × 10⁵ cm²/V/s | [View trace &rarr;](https://lbkhoa2797.github.io/oitini/graphene.html) |

## Quickstart

```bash
git clone https://github.com/lbkhoa2797/oitini.git && cd oitini
pip install -r requirements.txt
cp env.sample .env          # add the API keys and your QE / Wannier90 / PERTURBO paths
```

**Build a retrieval corpus.** Any literature-grounded run (the discovery loop, and literature
questions to the workflow agent) reads from a local index, so build one first.

1. In `config.py`, set `CORPUS_LABEL` (a name for the corpus) and `ARXIV_QUERY` (its keywords);
   other parameters are optional. `python scripts/00_check_corpus_query.py` checks the query against
   arXiv before anything is downloaded. The corpus lands in `data/corpus_<label>/`, and switching
   corpora later is a change of label.
2. Run the files in `scripts/` in order to download, parse, chunk, and index the corpus.

```bash
# Download the corpus from arXiv
python scripts/01_download.py
# Parse the pdfs into structured JSON files
python scripts/02_parse.py
# Chunk + classify each section, then index with a Chroma vector store (BGE embeddings, cosine)
# and a BM25 pickle (bm25.pkl)
python scripts/03_chunk_and_index.py 
```

Then run your discovery task. For example,

```bash
python -m graph.run "Find a 2D ferromagnetic semiconductor with Curie temperature > 30 K \
  and direct bandgap > 1 eV. Cite literature and verify with DFT."
```

See more in "Features and Usage" below.

**Requirements:**
* `QE`, `Wannier90`, and `PERTURBO` must be installed, with their paths set in `.env` (see
  `env.sample`). They live there rather than in `config.py` so that a tracked file never carries
  machine-local paths. Check them up front with:
```bash
python -c "from tools.dft import preflight_dft; preflight_dft()"
```
The discovery graph runs the same check before its first model call and names whichever key is
wrong.
* The API keys from [Claude](https://platform.claude.com/) and
  [Materials Project](https://next-gen.materialsproject.org/materials). (also through `.env`)

### Running on a cluster

Every MPI job — `pw.x`, `ph.x`, `projwfc.x`, `pw2wannier90.x`, `qe2pert.x` — goes through
`modules/launcher.py`, which picks the launcher, sizes the job and enforces a wall-clock limit. The
defaults are right for both a workstation and a single allocated node, so the same checkout runs
unedited in either place:

```
LAUNCHER=auto        # srun inside a SLURM allocation, mpirun otherwise
MPI_NPROCS=          # blank = every rank the allocation granted
OMP_NUM_THREADS_QE=1 # threads per rank
```

Start the agent **inside** the allocation, not on the login node:

```bash
salloc -N1 -n64 -t 8:00:00
python agent.py            # preflight prints the resolved launcher and rank count
```

Three things worth knowing:

- `LAUNCHER=auto` uses `srun` only when `SLURM_JOB_ID` is set. Plain `mpirun` inside an allocation
  escapes SLURM's cgroup and ignores the cpuset it was handed, which is how a 64-rank job ends up
  crammed onto one core.
- Keep `OMP_NUM_THREADS_QE=1` for a threaded QE build. The OpenMP default is one thread per core
  *per rank* — 64 ranks × 64 threads thrashes hard enough to look like a hang.
- `-nk` is derived per job from the k-point count, so ranks split over k-points first and plane
  waves second. Without it, 64-way FFT parallelization on a small oxide cell stalls in all-to-all.

Tool calls are executed one at a time, so each job gets the whole allocation. Requesting
`run_wannier90` and `create_ph_save` together is fine — they consume the same SCF independently —
but they run back to back, not concurrently.

### (Optional) Tracing with LangSmith
The discovery graph can stream full traces per-node runs, LLM calls with token/cost usage, RAG and
DFT tool spans to [LangSmith](https://smith.langchain.com).

Opt-in by modifying the `.env` file with:
```
LANGSMITH_TRACING=   # Set to true to enable LangSmith tracing
LANGSMITH_API_KEY=   # API key obtained from LangSmith
LANGSMITH_PROJECT=   # Name of your awesome project
```
Leave them unset and the workflow behaves identically.

When the run finishes, the CLI prints a link to the trace; you can also find it on your LangSmith
project page.

## Features and Usage

### 1. Perturbo-robot (Perturbot 🤖) Research Assistant

```bash
python agent.py "Compare the Curie temperature of CrX3 family. \
  Explain the trend and cite relevant literature."
# or
python agent.py "Help me compute the Silicon electron-phonon database from qe2pert.x \
  for processing with the Perturbo code."
```

For a magnetic material the agent first calls `classify_magnetic_order` (the magnetic-ordering
literature corpus, Materials Project and screening SCFs), runs the ground-state SCF in the
classified cell through `run_dft(magnetic_state=...)`, and continues the electron-phonon chain one
spin channel at a time (spin-up by default; one NSCF and one DFPT serve both channels).

**Note**: spin–orbit coupling and non-collinear magnetism are not supported yet.


Traces land in `traces/`; render them as standalone HTML with `utils/trace2html.py`. For example:
<img width="800" height="550" alt="trace_demo"
  src="https://github.com/user-attachments/assets/70665036-68cf-4a8d-9100-b3bbabf1368b" />

### 2. Closed-Loop Materials Discovery Agent

A LangGraph staged multi-agent system, featuring:
- A **Planner → Generator → Protocol → Simulator → EphPlan → Transport → Evidence → Critic →
  Reporter → Verifier** state graph
- Real first-principles calculations
- Per-candidate literature retrieval, so a citation is tied to the material it describes
- An automatic citation audit: every `[chunk: …]` marker in the report is re-checked against the
  chunk it cites to inspect model faithfulness

```bash
python -m graph.run "Find a 2D ferromagnetic semiconductor with Curie temperature > 30 K \
  and direct bandgap > 1 eV. Cite literature for any property values."
```
As mentioned, the trace can be visualized with [LangSmith](https://smith.langchain.com), or you can
track the agent's thoughts with `show_thoughts.py`:
```
python -m graph.show_thoughts <run_id> # --help for options
```
<img width="800" height="480" alt="showthoughts"
  src="https://github.com/user-attachments/assets/2fafe505-5ea5-4085-a15c-e59db736d7dc" />


## Repository layout

```
agent.py           ReAct workflow agent (Perturbot)
graph/             LangGraph closed-loop discovery system
tools/             The agent's tools, one module per registry.py entry: DFT, Wannier90, DFPT,
                   qe2pert, perturbo, retrieval, MP lookup, magnetic order
modules/           Modules that the tools are built on
rag/               Retrieval pipeline: chunking, embedding, hybrid search
mcp_server/        MCP server fronting the simulation and retrieval tools (Experimental)
scripts/           Corpus build: query check → download → parse → chunk & index
                   (build_corpora.sh runs them end to end)
benchmark_scripts/ Development and benchmark tooling, not needed for a run
  run_checks.py    Runs the offline check_*.py suite (no pw.x run, no network, no API key);
                   check_deployment.py also vets a checkout before a graded run
tests/golden/      Regression fixtures: Si chain (si_v1 inputs, si_v2 numbers, si_relax), GaAs,
                   Fe LSDA inputs, graphene flexural dyn files, the magnetic-ordering replay
eval/              FINDINGS.md: what the 2026 benchmarks measured (the harness is being redesigned)
utils/             Trace rendering (trace2html.py) and helpers
docs/              Design notes (pipeline-and-graph.md) and published sample outputs
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

It's `Initio` being spelled backward. I just think it's a fun name for an agent that can do `ab initio`
calculations.

## Citation

If you use PERTURBO, please cite:

> J.-J. Zhou, J. Park, I-T. Lu, I. Maliyov, X. Tong, and M. Bernardi,
> *PERTURBO: A software package for ab initio electron–phonon interactions, charge transport and
> ultrafast dynamics*,
> [Comput. Phys. Commun. **264**, 107970 (2021)](https://doi.org/10.1016/j.cpc.2021.107970).

## License

GPL-3.0. See [LICENSE](LICENSE).