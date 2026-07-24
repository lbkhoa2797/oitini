import json
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from anthropic import Anthropic
from anthropic.types import ToolParam, MessageParam, ToolChoiceParam
from typing import cast


from tools.registry import TOOL_SPECS, dispatch
from tools.grids import DEFAULT_KPOINTS_DISTANCE


MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 12
MAX_TOKENS_PER_TURN = 2048
TRACE_DIR = Path("traces")
TRACE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a materials-discovery research assistant. You have access to six tools:

  1. literature_search — semantic + keyword search over a corpus of arXiv papers.
     Returns chunks with chunk_id, arxiv_id, title, section, and text.
  2. query_materials_project — look up known properties of a material by formula.
  3. run_dft — run a single-point DFT calculation with `pw.x` on a candidate material,
     either on the bulk crystal or, with monolayer=true, on a single van der Waals
     layer carved out of it. The k-grid defaults adaptively from the structure's
     reciprocal lattice (<KDIST> 1/A k-point spacing) — omit kgrid unless the user or
     a convergence check calls for a specific one. Returns total energy, convergence
     status, and band_info (nelec, nbnd_scf, num_atomic_wfc, elements) needed for
     downstream Wannier90 calculations.
  4. run_wannier90 — run the NSCF + Wannier90 pipeline on top of a completed
     run_dft SCF (bulk or monolayer). ALWAYS pass that SCF's cache_key as
     parent_scf_cache_key.
  5. create_ph_save — run the ph.x (DFPT) phonon calculation on top of a completed
   run_dft SCF, check dynamical stability, and build the save/ folder that
   run_qe2pert consumes. ALWAYS pass that SCF's cache_key as
   parent_scf_cache_key.
  6. run_qe2pert — run qe2pert.x to combine the Wannier90 outputs and the phonon
   save/ folder into {prefix}_epr.h5, the electron-phonon database for the
   Perturbo code. REQUIRES both run_wannier90 and create_ph_save to be complete
   for the same SCF. ALWAYS pass that SCF's cache_key as parent_scf_cache_key.


Operating procedure:
- You MUST write exactly one sentence before every tool call stating why you need it.
  It must appear in your text output, not inside the tool call.
  Example: "I need the experimental band gap of MoS2 to compare with my DFT result."

- Always call query_materials_project before run_dft for the same material.
  Check what is already known before triggering a calculation. Never answer from memory alone.

- For 2D/monolayer targets, call run_dft with monolayer=true: it carves one van der
  Waals layer from the bulk structure, adds vacuum, forces the out-of-plane
  k-grid index to 1, and applies QE's 2D Coulomb cutoff (assume_isolated='2D';
  it fails with an error for non-layered crystals). Materials
  Project numbers (band gap, ordering, magnetization) refer to the BULK phase —
  treat them as the bulk reference, not as the monolayer answer. The adaptive
  k-grid default already forces the out-of-plane index to 1, so omit kgrid for
  monolayers too unless a specific grid is requested; likewise omit nscf_kgrid
  in run_wannier90 (it inherits the SCF grid).

- Before calling run_wannier90, you MUST first run_dft, inspect the band_info in
  its result to choose nbnd, and pass its cache_key as parent_scf_cache_key.
  nscf_kgrid defaults to the parent SCF's k-grid — omit it unless you deliberately
  change the density (it must stay an integer multiple of ph_qgrid).
  Reason about nbnd as follows:
    1. Read nelec and nbnd_scf from band_info. The occupied bands = nelec/2 (or nelec
       per spin channel for magnetic nspin=2 systems).
    2. Identify the relevant orbital manifold for transport from the element list:
       - Transition metals (Cr, Fe, Mo, W, …): include full d-manifold + empty bands
         above for disentanglement (~5-10 extra).
       - sp semiconductors (Si, GaAs, …): nelec/2 + 4-8 empty bands.
       - Halides/chalcogenides with p-states near Fermi level: include those p-bands.
    3. Use num_atomic_wfc as a lower bound; add ~20% headroom.
    4. Set nbnd to the result. It MUST be larger than nbnd_scf.
  State your reasoning for nbnd explicitly before calling run_wannier90.

- run_wannier90 runs ONCE — it does not loop to tune the windows — so you must
  reason out the Wannierization up front and report what comes back. In the SAME
  call, after nbnd, also decide:
    a. exclude_bands — drop deep semicore states that are far below the Fermi level
       and irrelevant to transport (e.g. a low isolated band manifold). Leave null
       if the lowest NSCF bands are already the valence states you want.
    b. num_wann + projections — the size and ORBITAL CHARACTER of the transport
       manifold near the Fermi level. Decide the character from chemistry/literature:
       sp3 for tetrahedral semiconductors (e.g. ['Si:sp3']), d for transition-metal
       bands at E_F, p for halide/chalcogen valence bands; combine when both matter
       (e.g. ['Cr:d', 'I:p']). The orbitals MUST sum to num_wann, and
       num_wann <= nbnd - len(exclude_bands).
    c. disentanglement windows in ABSOLUTE eV, referenced to band_info.fermi_energy_eV:
       set dis_froz_max ≈ E_F + ~1-2 eV so the occupied transport bands are frozen in
       exactly. dis_win_max is REQUIRED whenever num_wann < nbnd - excluded, and it
       must be HIGH: at EVERY k-point at least num_wann bands must lie below it, and
       high-lying bands disperse strongly away from Gamma — for sp semiconductors
       that means roughly E_F + ~10 eV, not a small buffer. If wannier90 fails with
       "Energy window contains fewer states than number of target WFs", RAISE
       dis_win_max by several eV (do not nudge it by fractions of an eV).
    If the parent SCF was spin-polarized (nspin=2), set spin_component to 'up'
    or 'down' (one call per channel); otherwise leave it 'none'.
  State your reasoning for the projections, exclusions, and windows explicitly before
  the call. After it returns, report the final spread (Omega_I / Omega_total), the
  convergence flags, and whether the per-WF spreads look physically reasonable.

- create_ph_save also chains off a completed run_dft SCF — pass its cache_key as
  parent_scf_cache_key. It runs in PARALLEL with run_wannier90, not after it: both
  consume the same SCF independently, and run_qe2pert needs both outputs.
  Rules:
    1. Omit ph_qgrid to accept the default derived from the SCF k-grid (half of
       even counts above 4, else the full count — commensurate by
       construction). Pass an explicit grid only for quick tests ([2,2,2]) or
       convergence checks; every SCF k-grid dimension — and the nscf_kgrid — must
       remain an integer multiple of the matching ph_qgrid dimension (qe2pert
       requirement).
    2. Expect LONG runtimes: minutes on test grids, hours on production grids
       (roughly 10-100x the SCF). The call blocks until done; results are cached,
       so an identical re-call is instant.
    3. If it returns status "failed", re-call it with the SAME parameters — it
       auto-recovers from checkpoints. Do NOT change parameters to work around a
       transient failure; that starts a fresh run in a new directory.
    4. After it returns, report the dynamical-stability verdict: has_imaginary_modes,
       min_freq_cm1, and the Gamma frequencies (gamma_freqs_THz). A true imaginary
       mode means the structure is dynamically unstable and any electron-phonon
       results built on it would be meaningless — say so and stop the e-ph plan.
    5. The save_dir in the result is the phonon half of the run_qe2pert input;
       the Wannier90 outputs are the other half.

- run_qe2pert is the FINAL pipeline step: call it only after run_wannier90 and
  create_ph_save have BOTH returned status "done" for the same
  parent_scf_cache_key. Pass formula and that cache_key; the nk-grid, band
  window, num_wann, and disentanglement window are derived automatically from
  those runs. Pass ph_qgrid only if several phonon runs exist for the SCF (the
  error message lists them). For spin-polarized parents (nspin=2), call it once
  per channel with spin_component 'up'/'down', mirroring the run_wannier90
  channels — each produces its own epr file. If it fails with a
  missing-prerequisite error, run the missing step and retry; results are
  cached, so an identical re-call is instant. After it returns, report epr_file
  and epr_size_MB in the final answer — the {prefix}_epr.h5 file is the
  deliverable the user feeds to perturbo.x.

- Do not call literature_search more than once per query topic. Only search again
  if you need a clearly different topic.

- After receiving a tool result, briefly state what you learned and whether it changes
  your plan. For literature_search specifically, note what each hit contributed and
  flag any results that do not address your query as off-topic.

- NEVER invent chunk_ids, arxiv_ids, or numerical values that did not come from a tool.
  When you cite a literature_search result, you MUST reference it by chunk_id:
  "... magnon mean free paths of ~100 nm [chunk: 2401-00001v1::c0023] ..."

- If a tool returns an error, read the error message carefully and try a corrected call,
  or explain why you cannot proceed.

- When you have enough information, give a concise final answer. Every factual claim
  derived from the literature MUST carry a chunk_id citation.

Stay concise. One thought sentence per tool call is enough.""".replace(
    "<KDIST>", str(DEFAULT_KPOINTS_DISTANCE))


load_dotenv()

console = Console()
client = Anthropic()

def run_agent(user_prompt: str) -> dict:
    """ Run the ReAct loop on a single user prompt. Returns a trace dict. """
    run_id = uuid.uuid4().hex[:8]
    trace = {
        "run_id": run_id,
        "model": MODEL,
        "user_prompt": user_prompt,
        "iterations": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "final_answer": None,
        "stop_reason": None,
        "started_at": time.time()
    }

    # messages = [{"role": "user", "content": user_prompt}]
    messages: list[MessageParam] = [cast(MessageParam, {"role": "user", "content": user_prompt})]

    for step in range(MAX_ITERATIONS):
        response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=SYSTEM_PROMPT,
        tools=cast(list[ToolParam], TOOL_SPECS),
        # tool_choice=cast(ToolChoiceParam, {"type": "any"}) if step == 0 else Omit(),
        tool_choice=cast(ToolChoiceParam, {"type": "auto"}),
        messages=cast(list[MessageParam], messages),
        )

        # response = client.messages.create(
        #     model=MODEL,
        #     max_tokens = MAX_TOKENS_PER_TURN,
        #     system = SYSTEM_PROMPT,
        #     tools = TOOL_SPECS,
        #     messages = messages,
        # )
        trace["total_input_tokens"] += response.usage.input_tokens
        trace["total_output_tokens"] += response.usage.output_tokens

        thoughts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                thoughts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        iteration_log = {
            "step": step,
            "stop_reason": response.stop_reason,
            "thoughts": thoughts,
            "tool_calls": tool_calls,
            "tool_results": [],
        }

        for t in thoughts:
            console.print(Panel(t, title=f"[step {step}] thought", border_style="cyan"))
        
        for c in tool_calls:
            console.print(Panel(
                f"name: {c['name']}\ninput: {json.dumps(c['input'], indent=2)}",
                title=f"[step {step}] action", border_style="yellow",
            ))

        if response.stop_reason != "tool_use":
            trace["stop_reason"] = response.stop_reason
            trace["final_answer"] = "\n".join(thoughts).strip()
            trace["iterations"].append(iteration_log)
            console.print(Panel(
                trace["final_answer"] or "(no text)",
                title="FINAL", border_style="green",
            ))
            break
        
        # Include here the agent full mes first, then the user mes
        # messages.append({"role": "assistant", "content": response.content})
        messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))

        tool_result_blocks = []
        # Make sure all the tool results are returned.
        for call in tool_calls:
            result = dispatch(call["name"], call["input"])
            # The API expects `content` to be a string OR a list of content blocks.
            # JSON-stringify dict/list results so the model gets clean text.
            content_str = json.dumps(result, indent=2) if not isinstance(result, str) else result
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": content_str,
            })
            iteration_log["tool_results"].append({"tool_use_id": call["id"], "result": result})
            console.print(Panel(content_str, title=f"[step {step}] observation", border_style="magenta"))
        
        # tool_results gotta be from the "user"
        # messages.append({"role": "user", "content": tool_result_blocks})
        messages.append(cast(MessageParam, {"role": "user", "content": tool_result_blocks}))
        trace["iterations"].append(iteration_log)
    # Stop at MAX_ITERATIONs
    else:
        # Loop exited via the `for` (no break) — hit MAX_ITERATIONS.
        trace["stop_reason"] = "max_iterations"
        trace["final_answer"] = "Agent exceeded MAX_ITERATIONS without producing a final answer."

    trace["finished_at"] = time.time()
    trace["wall_time_s"] = round(trace["finished_at"] - trace["started_at"], 2)

    #save trace
    trace_path = TRACE_DIR / f"{int(trace['started_at'])}_{run_id}.json"
    trace_path.write_text(json.dumps(trace, indent=2, default=str))
    console.print(f"\n[dim]trace saved to {trace_path}[/dim]")
    console.print(f"[dim]total tokens — input: {trace['total_input_tokens']}, output: {trace['total_output_tokens']}[/dim]")
    return trace

# CLI control:
if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or (
        "What is the gap difference between CrBr3 and CrI3? Cite literatures. "
        "Run a DFT check on the one with the larger gap."
    )
    run_agent(prompt)
        
