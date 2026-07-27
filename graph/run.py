"""CLI for the discovery graph."""
import argparse
import uuid
from pathlib import Path

from langsmith import utils as ls_utils

from graph.graph import make_default_graph
from config import MAX_DISCOVERY_ITERATIONS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("goal", type=str, nargs="?", default=None,
                        help="User's discovery goal (optional when resuming with --thread-id).")
    parser.add_argument("--max-iters", type=int, default=MAX_DISCOVERY_ITERATIONS)
    parser.add_argument("--thread-id", type=str, default=None,
                        help="Resume a previous thread, else a new one is created.")
    parser.add_argument("--report-out", type=str, default=None)
    args = parser.parse_args()

    graph = make_default_graph()
    thread_id = args.thread_id or f"run-{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
        # LangSmith annotations — inert when tracing is disabled.
        "run_name": "materials-discovery",
        "tags": ["graph-run"],
        "metadata": {
            "thread_id": thread_id,  # groups traces in LangSmith's Threads view
            "goal": (args.goal or "(resume)")[:300],
            "max_iterations": args.max_iters,
        },
    }
    # Pre-assign the root run id so the trace URL is known after the run.
    run_id = uuid.uuid4() if ls_utils.tracing_is_enabled() is True else None
    if run_id:
        config["run_id"] = run_id

    # A thread that stopped mid-graph has pending node(s) in its checkpoint.
    # Streaming None as the input resumes there with the saved state; any
    # non-None input would instead start a fresh run from START.
    pending = graph.get_state(config).next if args.thread_id else ()
    if pending:
        print(f"=== thread_id: {thread_id} — resuming at: {', '.join(pending)} ===")
        stream_input = None
    else:
        if args.goal is None:
            parser.error(f"goal is required ({thread_id!r} has no pending checkpoint to resume)")
        print(f"=== thread_id: {thread_id} ===")
        stream_input = {
            "goal": args.goal,
            "max_iterations": args.max_iters,
            "iteration": 0,
            "candidates": [],
            "evidence": [],
            "critic_history": [],
            "latest_verdict": None,
            "plan": None,
            "final_report": None,
            "citation_audit": None,
        }

    # Stream intermediate states for visibility.
    for step in graph.stream(stream_input, config=config):
        node_name = next(iter(step.keys()))
        print(f"  ✓ {node_name}")

    if run_id:
        try:
            from datetime import datetime, timezone
            from langsmith import Client
            from langsmith.schemas import RunBase
            stub = RunBase(id=run_id, name="materials-discovery",
                           run_type="chain", start_time=datetime.now(timezone.utc))
            print(f"\nLangSmith trace: {Client().get_run_url(run=stub)}")
        except Exception as e:
            print(f"\n(could not resolve LangSmith trace URL: {e})")

    final = graph.get_state(config).values
    report = final.get("final_report") or "(no report produced)"
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Citation faithfulness is the number that says whether the report above can
    # be believed, so print it next to the report rather than burying it in state.
    audit = final.get("citation_audit")
    if audit:
        if audit.get("error"):
            print(f"\ncitation audit FAILED to run: {audit['error']}")
        else:
            rate = audit.get("faithfulness_rate")
            n_ok, n_all = audit.get("n_supported"), audit.get("n_citations_total")
            print(f"\ncitation faithfulness: {rate} ({n_ok}/{n_all} citations supported)")
            for d in audit.get("details", []):
                if not d.get("supported"):
                    print(f"  UNSUPPORTED [{d['chunk_id']}] {d.get('reason', '')}")

    if args.report_out:
        Path(args.report_out).write_text(report)
        print(f"\nreport written to {args.report_out}")


if __name__ == "__main__":
    main()
