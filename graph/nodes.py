"""Node functions for the discovery graph."""
import json
from langchain_anthropic import ChatAnthropic

from config import (
    PLANNER_MODEL, GENERATOR_MODEL,
    CRITIC_MODEL, REPORTER_MODEL, MAX_DISCOVERY_ITERATIONS,
)
from graph.state import DiscoveryState, Candidate, CriticVerdict
from graph.prompts import (
    PLANNER_SYSTEM, GENERATOR_SYSTEM, CRITIC_SYSTEM, REPORTER_SYSTEM,
)
from tools.literature import literature_search
from tools.materials_project import fetch_summary_and_atoms
from tools.dft import run_dft   # real QE, defined in Step 5


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

    # Pull some literature evidence to ground the proposal.
    plan_summary = state["plan"]["search_strategy"]
    chunks = literature_search(query=plan_summary, top_k=5)

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
            updates.append({**c, "dft_status": result.get("dft_status", "done"),
                           "dft_result": result})
        except Exception as e:
            updates.append({**c, "dft_status": "failed",
                           "critic_notes": f"DFT raised: {type(e).__name__}: {e}"})

    # `candidates` uses the _merge_candidates reducer (merge-by-formula, new wins),
    # so returning only the changed subset updates those entries in place.
    return {"candidates": updates}


# ------- Critic -----------------------------------------------------------
def critic_node(state: DiscoveryState) -> dict:
    summary = []
    for c in state["candidates"]:
        # if c["dft_status"] == "done":
        if c["dft_status"] in ("done", "cached"):
            summary.append({
                "formula": c["formula"],
                "energy_per_atom_eV": c["dft_result"].get("total_energy_eV_per_atom"),
                "scf_converged": c["dft_result"].get("converged"),
                "magnetic_moment": c["dft_result"].get("total_magnetization"),
                "band_gap_eV": c["dft_result"].get("band_gap_eV"),
                "band_gap_direct_eV": c["dft_result"].get("band_gap_direct_eV"),
                "is_metallic": c["dft_result"].get("is_metallic"),
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
    verdict = CriticVerdict(
        decision=raw.get("decision", "continue"),
        reason=raw.get("reason", ""),
        new_directions=raw.get("new_directions", []),
        n_meeting_targets=raw.get("n_meeting_targets", 0),
    )

    # Forced stop on max iterations. Preserve the model's target count for the eval.
    if state["iteration"] >= state["max_iterations"]:
        verdict = CriticVerdict(
            decision="done", reason="reached max_iterations",
            new_directions=[], n_meeting_targets=verdict["n_meeting_targets"],
        )
    return {"latest_verdict": verdict, "critic_history": [verdict]}

def _ev_field(e, key):
    if key in e:
        return e[key]
    meta = e.get("metadata", {})
    return meta.get(key, "(unknown)")


# ------- Reporter ---------------------------------------------------------
def reporter_node(state: DiscoveryState) -> dict:
    llm = ChatAnthropic(model=REPORTER_MODEL, max_tokens=2000)
    response = llm.invoke([
        {"role": "system", "content": REPORTER_SYSTEM},
        {
            "role": "user",
            "content": json.dumps({
                "goal": state["goal"],
                "plan": state["plan"],
                "candidates": state["candidates"],
                "critic_history": state["critic_history"],
                "evidence": [
                    {"chunk_id": _ev_field(e, "chunk_id"),
                     "title": _ev_field(e, "title")}
                    for e in state["evidence"]
                ],
            }, indent=2, default=str),
        },
    ])
    return {"final_report": _get_text(response.content)}