"""Print the chain of thoughts of each agent for a completed discovery run.

Reads the persisted LangGraph state (the SqliteSaver checkpointer at
GRAPH_DB_PATH) for a given thread_id and replays, in execution order, what
each agent contributed: the Planner's plan, the Generator's per-candidate
rationales, the Simulator's DFT outcomes, the Critic's verdicts, and the
Reporter's final report.

How it works
------------
LangGraph persists the *full* state after every step. We pull the snapshot
history (oldest -> newest) and diff consecutive snapshots: the node that
produced snapshot[i] is whatever snapshot[i-1] had queued in `.next`. The
delta between the two snapshots is that node's contribution.

Caveat
------
Only the *parsed* output each node wrote to state is persisted -- the raw LLM
responses (the literal reasoning text before the JSON) are NOT stored. So the
"thoughts" shown here are the structured reasoning each agent committed to
state (rationales, verdicts, plan), not the model's raw token stream.

Usage
-----
    python -m graph.show_thoughts <thread_id>
    python -m graph.show_thoughts <thread_id> --report   # also print final report
    python -m graph.show_thoughts --list                 # list available runs
"""
import argparse
import json
import sqlite3

from config import GRAPH_DB_PATH
from graph.graph import make_default_graph

try:
    from rich.console import Console
    from rich.panel import Panel
    _console = Console()
except ImportError:  # pragma: no cover
    _console = None

# One accent colour per agent, matching the ReAct trace style in agent.py.
NODE_STYLE = {
    "planner": "magenta",
    "generator": "yellow",
    "simulator": "blue",
    "critic": "red",
    "reporter": "green",
}


def _panel(title: str, body: str, color: str) -> None:
    if _console is not None:
        _console.print(Panel(body or "(empty)", title=title, border_style=color))
    else:
        print(f"\n=== {title} ===\n{body or '(empty)'}")


def list_runs() -> None:
    """List thread_ids present in the checkpoint DB."""
    con = sqlite3.connect(str(GRAPH_DB_PATH))
    try:
        rows = con.execute(
            "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id"
        ).fetchall()
    finally:
        con.close()
    if not rows:
        print("no runs found in", GRAPH_DB_PATH)
        return
    print(f"runs in {GRAPH_DB_PATH}:")
    for thread_id, n in rows:
        print(f"  {thread_id}  ({n} checkpoints)")


def delete_run(thread_id: str) -> None:
    """Delete all checkpoints and writes for a given thread_id."""
    con = sqlite3.connect(str(GRAPH_DB_PATH))
    try:
        n_ckpt = con.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
        if n_ckpt == 0:
            print(f"no run found with thread_id={thread_id!r}. Try --list.")
            return
        n_writes = con.execute(
            "SELECT COUNT(*) FROM writes WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
        con.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        con.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        con.commit()
        print(f"deleted {thread_id}: {n_ckpt} checkpoints, {n_writes} writes")
    finally:
        con.close()


# ------- per-agent formatters --------------------------------------------
def _fmt_planner(prev: dict, cur: dict) -> str:
    plan = cur.get("plan") or {}
    lines = []
    if plan.get("search_strategy"):
        lines.append(f"search strategy: {plan['search_strategy']}")
    if plan.get("target_properties"):
        lines.append(f"target properties: {json.dumps(plan['target_properties'])}")
    if plan.get("starting_candidates"):
        lines.append(f"starting candidates: {', '.join(plan['starting_candidates'])}")
    if plan.get("max_iterations") is not None:
        lines.append(f"max iterations: {plan['max_iterations']}")
    return "\n".join(lines)


def _fmt_generator(prev: dict, cur: dict) -> str:
    prev_formulas = {c["formula"] for c in prev.get("candidates", [])}
    new = [c for c in cur.get("candidates", []) if c["formula"] not in prev_formulas]
    n_evidence = len(cur.get("evidence", [])) - len(prev.get("evidence", []))
    lines = [f"iteration {cur.get('iteration')} | pulled {n_evidence} literature chunks"]
    if not new:
        lines.append("(no new candidates this pass)")
    for c in new:
        lines.append(f"\n• {c['formula']}  [{c['structure_source']}]")
        lines.append(f"  rationale: {c.get('rationale', '')}")
    return "\n".join(lines)


def _fmt_simulator(prev: dict, cur: dict) -> str:
    prev_by = {c["formula"]: c for c in prev.get("candidates", [])}
    lines = []
    for c in cur.get("candidates", []):
        before = prev_by.get(c["formula"], {})
        if c.get("dft_status") == before.get("dft_status"):
            continue  # unchanged this pass
        res = c.get("dft_result") or {}
        lines.append(f"• {c['formula']}: {c['dft_status']}")
        if res:
            lines.append(
                f"    E/atom={res.get('total_energy_eV_per_atom')} eV  "
                f"converged={res.get('converged')}  "
                f"mag={res.get('total_magnetization')}"
            )
        if c.get("critic_notes"):
            lines.append(f"    note: {c['critic_notes']}")
    return "\n".join(lines) or "(no candidates simulated this pass)"


def _fmt_critic(prev: dict, cur: dict) -> str:
    prev_hist = prev.get("critic_history") or []
    cur_hist = cur.get("critic_history") or []
    new = cur_hist[len(prev_hist):]
    lines = []
    for v in new:
        lines.append(f"decision: {v['decision']}  (meeting targets: {v.get('n_meeting_targets')})")
        lines.append(f"reason: {v.get('reason', '')}")
        if v.get("new_directions"):
            lines.append("new directions:")
            lines.extend(f"  - {d}" for d in v["new_directions"])
    return "\n".join(lines)


def _fmt_reporter(prev: dict, cur: dict) -> str:
    return cur.get("final_report") or "(no report produced)"


FORMATTERS = {
    "planner": _fmt_planner,
    "generator": _fmt_generator,
    "simulator": _fmt_simulator,
    "critic": _fmt_critic,
    "reporter": _fmt_reporter,
}


def show_thoughts(thread_id: str, include_report: bool) -> None:
    graph = make_default_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Newest -> oldest; reverse to replay in execution order.
    history = list(graph.get_state_history(config))
    if not history:
        print(f"no checkpoints for thread_id={thread_id!r}. Try --list.")
        return
    history.reverse()

    print(f"=== chain of thoughts for {thread_id} ===")
    print("(reconstructed from persisted state; raw LLM reasoning is not stored)\n")

    step_no = 0
    for prev, cur in zip(history, history[1:]):
        # The node that produced `cur` is whatever `prev` had queued next.
        node = prev.next[0] if prev.next else None
        if node not in FORMATTERS:
            continue  # skip __start__ / input seeding
        if node == "reporter" and not include_report:
            _panel(f"step {step_no} · reporter", "(final report omitted; pass --report)", "green")
            step_no += 1
            continue
        body = FORMATTERS[node](prev.values, cur.values)
        _panel(f"step {step_no} · {node}", body, NODE_STYLE.get(node, "white"))
        step_no += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thread_id", nargs="?", help="Run/thread id to inspect.")
    parser.add_argument("--report", action="store_true",
                        help="Include the full final report from the Reporter.")
    parser.add_argument("--list", action="store_true",
                        help="List available runs and exit.")
    parser.add_argument("--delete", action="store_true",
                        help="Delete the specified run from the checkpoint DB.")
    args = parser.parse_args()

    if args.list or not args.thread_id:
        list_runs()
        return
    if args.delete:
        delete_run(args.thread_id)
        return
    show_thoughts(args.thread_id, include_report=args.report)


if __name__ == "__main__":
    main()
