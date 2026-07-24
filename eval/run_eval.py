import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import unicodedata
from pathlib import Path
from agent import run_agent
from rag.verify import verify_answer

TASKS = [json.loads(line) for line in Path("eval/tasks.jsonl").read_text().splitlines() if line.strip()]

def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()

def evaluate(trace, task):
    final = normalize(trace.get("final_answer") or "")
    tools_used = [c["name"] for it in trace["iterations"] for c in it["tool_calls"]]

    tool_match = (task["expects_tool"] is None) or (task["expects_tool"] in tools_used)
    substr_match = normalize(task["expected_substr"]) in final
    completed = trace["stop_reason"] == "end_turn"

    # Flag cases where expects_tool is null but tools were still called
    unexpected_tools = (task["expects_tool"] is None) and (len(tools_used) > 0)

    return {
        "id": task["id"],
        "completed": completed,
        "tools_used": tools_used,
        "expected_tool_used": tool_match,
        "unexpected_tools": unexpected_tools,
        "expected_substr_in_answer": substr_match,
        "n_steps": len(trace["iterations"]),
        "wall_time_s": trace.get("wall_time_s"),
        "pass": completed and tool_match and substr_match and not unexpected_tools,
    }

if __name__ == "__main__":
    results = []
    for task in TASKS:
        print(f"\n=== {task['id']}: {task['prompt']} ===")
        trace = run_agent(task["prompt"])
        result = evaluate(trace, task)

        if trace["final_answer"]:
            audit = verify_answer(trace["final_answer"])
            result["faithfulness_rate"] = audit["faithfulness_rate"]   # attach to result dict
            result["audit"] = audit

        results.append(result) 

    Path("eval/results.json").write_text(json.dumps(results, indent=2))
    n_pass = sum(r["pass"] for r in results)
    print(f"\n{n_pass}/{len(results)} tasks passed")

    for r in results:
        flag = "P" if r["pass"] else "F"
        faith = f"  faith={r['faithfulness_rate']:.2f}" if "faithfulness_rate" in r else ""
        warn = "  !! unexpected tools" if r["unexpected_tools"] else ""
        print(f"  {flag} {r['id']}  steps={r['n_steps']}  tools={r['tools_used']}{faith}{warn}")