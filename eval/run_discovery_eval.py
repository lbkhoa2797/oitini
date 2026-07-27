"""Evaluate the discovery agent. Multiple metrics, no single 'pass' label."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
# from pathlib import Path
from graph.graph import make_default_graph

TASKS = [json.loads(l) for l in Path("eval/tasks_discovery.jsonl").read_text().splitlines() if l.strip()]

def evaluate_run(state: dict, target_count: int) -> dict:
    candidates = state.get("candidates", [])
    # done = [c for c in candidates if c["dft_status"] == "done"]
    done = [c for c in candidates if c["dft_status"] in ("done", "cached")]
    failed = [c for c in candidates if c["dft_status"] == "failed"]
    verdict = state.get("latest_verdict") or {}
    history = state.get("critic_history", [])
    audit = state.get("citation_audit") or {}

    # Report the claim and the ground truth side by side instead of collapsing
    # them. The previous version scored `n_meeting_targets` alone -- the Critic's
    # own self-report -- so a run that fabricated four qualified candidates
    # scored a perfect four. A claim is only as good as its faithfulness_rate.
    n_grounded = sum(1 for c in candidates if c.get("evidence_ids"))
    return {
        "n_candidates_total": len(candidates),
        "n_completed_dft": len(done),
        "n_failed_dft": len(failed),
        "n_candidates_with_evidence": n_grounded,
        "n_iterations": len(history),
        "final_decision": verdict.get("decision"),
        "candidates_claimed": verdict.get("n_meeting_targets", 0),
        "candidates_expected": target_count,
        "qualified": verdict.get("qualified", []),
        "faithfulness_rate": audit.get("faithfulness_rate"),
        "n_citations_total": audit.get("n_citations_total"),
        "wall_time_s": state.get("_wall_time", 0),
    }

import uuid
graph = make_default_graph()
results = []
for task in TASKS:
    thread_id = f"eval-{task['id']}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    init = {"goal": task["goal"], "max_iterations": 5, "iteration": 0,
            "candidates": [], "evidence": [], "critic_history": [],
            "latest_verdict": None, "plan": None, "final_report": None,
            "citation_audit": None}
    t0 = time.time()
    final_state = graph.invoke(init, config=config)
    final_state["_wall_time"] = round(time.time() - t0, 1)
    metrics = evaluate_run(final_state, task["target_count"])
    results.append({"id": task["id"], **metrics})
    print(f"  {task['id']}: claimed={metrics['candidates_claimed']} "
          f"expected={metrics['candidates_expected']} "
          f"faithfulness={metrics['faithfulness_rate']} "
          f"qualified={metrics['qualified']}")
    print(f"    {metrics}")

Path("eval/results_week3.json").write_text(json.dumps(results, indent=2))
