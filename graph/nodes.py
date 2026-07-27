"""Node functions for the discovery graph."""
import json
import re
from langchain_anthropic import ChatAnthropic

from config import (
    PLANNER_MODEL, GENERATOR_MODEL,
    CRITIC_MODEL, REPORTER_MODEL, MAX_DISCOVERY_ITERATIONS,
    RUN_CITATION_AUDIT,
)
from graph.state import DiscoveryState, Candidate, CriticVerdict
from graph.prompts import (
    PLANNER_SYSTEM, GENERATOR_SYSTEM, CRITIC_SYSTEM, REPORTER_SYSTEM,
)
from rag.verify import verify_answer
from tools.literature import literature_search
from tools.materials_project import fetch_summary_and_atoms
from tools.dft import run_dft, preflight_dft   # real QE, defined in Step 5


def _get_text(content) -> str:
    """Normalize ChatAnthropic response content to a plain string.

    `.content` is a str for simple replies, but a list of content blocks
    whenever the message has more than one part — concatenate the text parts.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
    )


def _extract_json(text: str) -> dict:
    """Pull the first valid JSON object out of an LLM response. Tolerates code fences."""
    # Scan for each '{' and try to parse a JSON object starting there, using
    # brace-balancing to find the matching close rather than using regex
    # (which would over-capture trailing prose).
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                return json.loads(text[i:])
            except json.JSONDecodeError:
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except json.JSONDecodeError:
                            break
    raise ValueError(f"no valid JSON in response: {text[:200]}")


# ------- Planner ----------------------------------------------------------
def planner_node(state: DiscoveryState) -> dict:
    # Check the DFT toolchain before spending a single model or Materials Project
    # call: a broken QE path invalidates every candidate the run would produce.
    preflight_dft()

    llm = ChatAnthropic(model=PLANNER_MODEL, max_tokens=1500)
    response = llm.invoke([
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"GOAL: {state['goal']}"},
    ])
    plan = _extract_json(_get_text(response.content))

    # Seed the loop with the Planner's starting candidates.
    starting = [
        Candidate(
            formula=f, rationale="initial Planner suggestion",
            structure_source="planner", cif=None,
            dft_status="pending", dft_result=None, critic_notes=None,
            mp_summary=None, evidence_ids=[],
        )
        for f in plan.get("starting_candidates", [])
    ]
    return {
        "plan": plan,
        "iteration": 0,
        "candidates": starting,
        "max_iterations": plan.get("max_iterations", MAX_DISCOVERY_ITERATIONS),
    }


# ------- Generator --------------------------------------------------------
def generator_node(state: DiscoveryState) -> dict:
    tried = [c["formula"] for c in state["candidates"]]
    feedback = (state.get("latest_verdict") or {}).get("new_directions", [])

    # Pull some literature evidence to ground the proposal. Blend in the Critic's
    # latest steer: querying on the plan's fixed search_strategy alone made every
    # iteration issue an identical query and retrieve an identical set of chunks.
    plan_summary = state["plan"]["search_strategy"]
    query = " ".join([plan_summary, *(str(f) for f in feedback)]) if feedback else plan_summary
    chunks = literature_search(query=query, top_k=5)

    llm = ChatAnthropic(model=GENERATOR_MODEL, max_tokens=1000)
    user_msg = (
        f"GOAL: {state['goal']}\n\n"
        f"PLAN: {json.dumps(state['plan'])}\n\n"
        f"TRIED_FORMULAS: {tried}\n\n"
        f"CRITIC_FEEDBACK: {feedback}\n\n"
        f"LITERATURE_EVIDENCE (top 5 chunks):\n"
        + "\n".join(f"[{c['chunk_id']}] {c['text'][:300]}" for c in chunks)
    )
    response = llm.invoke([
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ])
    parsed = _extract_json(_get_text(response.content))

    new_candidates = [
        Candidate(
            formula=c["formula"], rationale=c["rationale"],
            structure_source="generator", cif=None,
            dft_status="pending", dft_result=None, critic_notes=None,
            mp_summary=None, evidence_ids=[],
        )
        for c in parsed.get("candidates", [])
        if c["formula"] not in tried
    ]
    return {
        "candidates": new_candidates,
        "evidence": chunks,
        "iteration": state["iteration"] + 1,
    }


# ------- Simulator --------------------------------------------------------
def simulator_node(state: DiscoveryState) -> dict:
    """For each pending candidate, look up a structure and run DFT."""
    updates = []
    for c in state["candidates"]:
        if c["dft_status"] != "pending":
            continue
        # One MP round-trip: metadata (for nspin) + the structure we hand to DFT.
        try:
            mp, atoms = fetch_summary_and_atoms(c["formula"])
        except Exception as e:
            updates.append({**c, "dft_status": "failed",
                           "critic_notes": f"MP lookup raised: {type(e).__name__}: {e}"})
            continue
        if "error" in mp:
            updates.append({**c, "dft_status": "failed",
                           "critic_notes": f"no structure in MP: {mp['error']}"})
            continue
        try:
            result = run_dft(formula=c["formula"], atoms=atoms,
                             ecutwfc=45.0,
                             nspin=2 if mp.get("is_magnetic") else 1,
                             )
            # Keep the whole MP record, not just is_magnetic. It carries
            # energy_above_hull and formation_energy (which the plan asks about
            # under crystal_stability) and an independent band gap that lets the
            # Critic cross-check our own DFT instead of trusting it blindly.
            updates.append({**c, "dft_status": result.get("dft_status", "done"),
                           "dft_result": result, "mp_summary": mp})
        except Exception as e:
            updates.append({**c, "dft_status": "failed", "mp_summary": mp,
                           "critic_notes": f"DFT raised: {type(e).__name__}: {e}"})

    # `candidates` uses the _merge_candidates reducer (merge-by-formula, new wins),
    # so returning only the changed subset updates those entries in place.
    return {"candidates": updates}


# ------- Evidence ---------------------------------------------------------
def _evidence_query(formula: str) -> str:
    """Retrieval query for one candidate.

    Deliberately short. Appending the plan's target-property names made ~8 tokens
    of boilerplate identical across every candidate, which dominated both the
    embedding and the cross-encoder: querying "ZnO thin film synthesis optical gap
    carrier mobility ..." returned a barium stannate paper, while a bare
    "ZnO thin film band gap" retrieves ZnO papers. The formula has to carry the
    query, so nothing generic is allowed to crowd it out.

    Known gap: PDF extraction sometimes splits subscripts ("CaZrO 3"), which BM25
    indexes as "cazro"+"3" so a "cazro3" query cannot match it. Such chunks are
    almost always table rows rather than prose, so missing them costs little --
    but a well-covered material written only that way would look uncited.
    """
    return f"{formula} thin film band gap"


def _mentions_formula(text: str, formula: str) -> bool:
    """Does this chunk actually discuss *formula*?

    Tolerates the ways a formula survives PDF extraction -- "CaZrO3",
    "CaZrO$_3$", "CaZrO 3" -- by allowing non-alphanumeric filler between
    characters, while the leading (?<![A-Za-z]) stops a match from starting
    mid-word (without it "TiO2" matches the "tio 2" inside "ratio 2").
    """
    pattern = r"(?<![A-Za-z])" + r"[^A-Za-z0-9]*".join(re.escape(ch) for ch in formula)
    return re.search(pattern, text, re.I) is not None


def evidence_node(state: DiscoveryState) -> dict:
    """Retrieve literature for each candidate that just finished DFT.

    Deliberately separate from the Generator's retrieval: that one *explores*
    (what should we try next?), this one *verifies* (what does the corpus
    actually say about the thing we just computed?). Only the second kind can
    ground a citation, and the pipeline previously had only the first.

    Retrieval alone is not enough: the reranker happily returns a topically
    similar chunk about a *different* material, which is exactly how a CaZrO3
    claim ended up cited to a MoS2 paper. So over-retrieve, then keep only
    chunks that actually name the candidate. A candidate the corpus does not
    cover ends up with NO evidence, which is the honest outcome -- the Critic
    can then refuse to certify it instead of citing something unrelated.
    """
    chunks: list[dict] = []
    updates = []
    for c in state["candidates"]:
        # Skip candidates already looked up: without this, every iteration
        # re-retrieves the whole set and the query never advances.
        if c["dft_status"] not in ("done", "cached") or c.get("evidence_ids"):
            continue
        try:
            hits = literature_search(query=_evidence_query(c["formula"]), top_k=10)
        except Exception as e:
            updates.append({**c, "critic_notes":
                            f"evidence lookup failed: {type(e).__name__}: {e}"})
            continue
        on_topic = [h for h in hits if _mentions_formula(h["text"], c["formula"])][:3]
        note = None if on_topic else (
            f"no chunk in the corpus mentions {c['formula']}; "
            "literature-backed targets cannot be met for it")
        chunks.extend(on_topic)
        updates.append({**c, "evidence_ids": [h["chunk_id"] for h in on_topic],
                        **({"critic_notes": note} if note else {})})
    return {"candidates": updates, "evidence": chunks}


# ------- Critic -----------------------------------------------------------
def critic_node(state: DiscoveryState) -> dict:
    by_id = {e["chunk_id"]: e for e in state["evidence"]}
    summary = []
    for c in state["candidates"]:
        # if c["dft_status"] == "done":
        if c["dft_status"] in ("done", "cached"):
            r = c["dft_result"] or {}
            mp = c.get("mp_summary") or {}
            summary.append({
                "formula": c["formula"],
                "energy_per_atom_eV": r.get("total_energy_eV_per_atom"),
                "scf_converged": r.get("converged"),
                "magnetic_moment": r.get("total_magnetization"),
                "band_gap_eV": r.get("band_gap_eV"),
                "band_gap_direct_eV": r.get("band_gap_direct_eV"),
                "gap_is_direct": r.get("gap_is_direct"),
                "cb_width_eV": r.get("cb_width_eV"),
                "is_metallic": r.get("is_metallic"),
                # Independent cross-checks, so the gap and stability targets are
                # judged against something besides our own single DFT run. These
                # were fetched and discarded before, while the Critic reported
                # them as missing.
                "mp_band_gap_eV": mp.get("band_gap_eV"),
                "energy_above_hull_eV_per_atom": mp.get("energy_above_hull_eV_per_atom"),
                "formation_energy_eV_per_atom": mp.get("formation_energy_eV_per_atom"),
                "space_group": mp.get("space_group"),
                # The chunks this candidate may cite. An empty list means no
                # literature-backed target can be counted as met for it.
                "evidence": [
                    {"chunk_id": cid,
                     "title": by_id[cid].get("title"),
                     "text": (by_id[cid].get("text") or "")[:400]}
                    for cid in (c.get("evidence_ids") or [])[:3] if cid in by_id
                ],
            })

    llm = ChatAnthropic(model=CRITIC_MODEL, max_tokens=2000)
    user_msg = (
        f"GOAL: {state['goal']}\n"
        f"PLAN_TARGETS: {state['plan'].get('target_properties')}\n"
        f"ITERATION: {state['iteration']} of {state['max_iterations']}\n"
        f"RESULTS_SO_FAR: {json.dumps(summary, indent=2)}"
    )
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    response = llm.invoke(messages)
    # RESULTS_SO_FAR grows each iteration; if the reply was cut off before the
    # JSON, retry once with double the budget instead of crashing the graph.
    if response.response_metadata.get("stop_reason") == "max_tokens":
        response = ChatAnthropic(model=CRITIC_MODEL, max_tokens=4000).invoke(messages)
    raw = _extract_json(_get_text(response.content))
    qualified = [str(f) for f in (raw.get("qualified") or [])]
    # The named list is auditable and the bare integer is not, so when the model
    # contradicts itself, trust the list.
    n_meeting = raw.get("n_meeting_targets", 0)
    if not isinstance(n_meeting, int) or n_meeting != len(qualified):
        n_meeting = len(qualified)
    verdict = CriticVerdict(
        decision=raw.get("decision", "continue"),
        reason=raw.get("reason", ""),
        new_directions=raw.get("new_directions", []),
        n_meeting_targets=n_meeting,
        qualified=qualified,
    )

    # Forced stop on max iterations. Preserve the model's target count for the eval.
    if state["iteration"] >= state["max_iterations"]:
        verdict = CriticVerdict(
            decision="done", reason="reached max_iterations",
            new_directions=[], n_meeting_targets=verdict["n_meeting_targets"],
            qualified=verdict["qualified"],
        )
    return {"latest_verdict": verdict, "critic_history": [verdict]}

def _ev_field(e, key):
    if key in e:
        return e[key]
    meta = e.get("metadata", {})
    return meta.get(key, "(unknown)")


# ------- Reporter ---------------------------------------------------------
def reporter_node(state: DiscoveryState) -> dict:
    messages = [
        {"role": "system", "content": REPORTER_SYSTEM},
        {
            "role": "user",
            "content": json.dumps({
                "goal": state["goal"],
                "plan": state["plan"],
                "candidates": state["candidates"],
                "critic_history": state["critic_history"],
                # Rendered verbatim into section 4 rather than re-derived.
                "latest_verdict": state.get("latest_verdict"),
                # Chunk TEXT, not just ids. The Reporter is instructed to cite
                # these; a citation it cannot read is one it cannot check, which
                # is how unsupported markers got attached in the first place.
                "evidence": [
                    {"chunk_id": _ev_field(e, "chunk_id"),
                     "title": _ev_field(e, "title"),
                     "section": _ev_field(e, "section"),
                     "text": (e.get("text") or "")[:800]}
                    for e in state["evidence"]
                ],
            }, indent=2, default=str),
        },
    ]
    response = ChatAnthropic(model=REPORTER_MODEL, max_tokens=4000).invoke(messages)
    # The old 2000-token budget truncated the report mid-section, silently losing
    # the cited-evidence and caveats sections. Same retry the Critic already has.
    if response.response_metadata.get("stop_reason") == "max_tokens":
        response = ChatAnthropic(model=REPORTER_MODEL, max_tokens=8000).invoke(messages)
    return {"final_report": _get_text(response.content)}


# ------- Verifier ---------------------------------------------------------
def verifier_node(state: DiscoveryState) -> dict:
    """Audit every [chunk: ...] marker in the report against the chunk it names.

    rag.verify has done this correctly all along; nothing ever called it, so a
    report whose citations were entirely unsupported scored the same as one whose
    citations held up.
    """
    report = state.get("final_report") or ""
    if not RUN_CITATION_AUDIT or not report:
        return {"citation_audit": None}
    try:
        return {"citation_audit": verify_answer(report)}
    except Exception as e:
        # An audit failure must never sink a finished run, but it must also never
        # look like a clean audit.
        return {"citation_audit": {"error": f"{type(e).__name__}: {e}",
                                   "faithfulness_rate": None}}
