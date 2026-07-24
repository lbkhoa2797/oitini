#!/usr/bin/env python3
"""
read_trace.py — Pretty-print ReAct agent trace JSON files.

Usage:
    python read_trace.py trace.json
    python read_trace.py trace.json --no-color
    cat trace.json | python read_trace.py -
"""

import json
import sys
import argparse
import textwrap
from datetime import datetime


# ── ANSI color helpers ────────────────────────────────────────────────────────

class C:
    """ANSI color codes; disabled when --no-color is set."""
    enabled = True

    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # Semantic palette
    HEADER  = "\033[38;5;75m"   # sky blue
    STEP    = "\033[38;5;220m"  # gold
    THOUGHT = "\033[38;5;183m"  # lavender
    TOOL    = "\033[38;5;114m"  # sage green
    RESULT  = "\033[38;5;216m"  # peach
    ANSWER  = "\033[38;5;156m"  # mint
    META    = "\033[38;5;244m"  # grey
    WARN    = "\033[38;5;203m"  # red-orange

    @classmethod
    def wrap(cls, code: str, text: str) -> str:
        if not cls.enabled:
            return text
        return f"{code}{text}{cls.RESET}"


def header(s):   return C.wrap(C.HEADER + C.BOLD, s)
def step(s):     return C.wrap(C.STEP + C.BOLD, s)
def thought(s):  return C.wrap(C.THOUGHT, s)
def tool(s):     return C.wrap(C.TOOL + C.BOLD, s)
def result(s):   return C.wrap(C.RESULT, s)
def answer(s):   return C.wrap(C.ANSWER, s)
def meta(s):     return C.wrap(C.META + C.DIM, s)
def warn(s):     return C.wrap(C.WARN + C.BOLD, s)


# ── Formatting helpers ────────────────────────────────────────────────────────

WIDTH = 88

def rule(char="─", width=WIDTH):
    return meta(char * width)

def section_rule(char="═", width=WIDTH):
    return meta(char * width)

def indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(text, prefix)

def pretty_json(obj, indent_level=4) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

def wrap_text(text: str, width: int = WIDTH - 4, prefix: str = "    ") -> str:
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=width))
    return textwrap.indent("\n".join(wrapped), prefix)


# ── Section printers ──────────────────────────────────────────────────────────

def print_header(trace: dict):
    print()
    print(section_rule("═"))
    print(header(f"  ReAct Agent Trace  ·  run_id: {trace.get('run_id', '?')}"))
    print(section_rule("═"))

    model   = trace.get("model", "unknown")
    prompt  = trace.get("user_prompt", "")
    started = trace.get("started_at")
    elapsed = trace.get("wall_time_s")

    if started:
        ts = datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S")
        print(meta(f"  Started : {ts}"))
    if elapsed is not None:
        print(meta(f"  Elapsed : {elapsed:.2f}s"))
    print(meta(f"  Model   : {model}"))
    print()
    print(header("  USER PROMPT"))
    print(wrap_text(prompt))
    print()


def print_thoughts(thoughts: list[str]):
    if not thoughts:
        return
    print(rule("┄"))
    print(thought("  💭  THOUGHTS"))
    for t in thoughts:
        print(wrap_text(t, prefix="      "))


def print_action(call: dict, idx: int):
    name  = call.get("name", "unknown_tool")
    inp   = call.get("input", {})
    cid   = call.get("id", "")
    print()
    print(tool(f"  ┌─ ACTION [{idx + 1}]  {name}"))
    print(meta(f"  │  id: {cid}"))
    inp_str = pretty_json(inp)
    for line in inp_str.splitlines():
        print(meta("  │  ") + line)
    print(tool("  └─"))


def print_tool_result(res: dict, tool_calls: list[dict]):
    # Match result back to its call by tool_use_id
    tid = res.get("tool_use_id", "")
    call_name = next(
        (c.get("name", "") for c in tool_calls if c.get("id") == tid), ""
    )
    label = f"  ┌─ OBSERVATION  {call_name}" if call_name else "  ┌─ OBSERVATION"
    print()
    print(result(label))
    print(meta(f"  │  tool_use_id: {tid}"))

    raw = res.get("result", "")
    if isinstance(raw, (dict, list)):
        res_str = pretty_json(raw)
    else:
        res_str = str(raw)

    for line in res_str.splitlines():
        print(meta("  │  ") + line)
    print(result("  └─"))


def print_step(iteration: dict, step_num: int):
    stop = iteration.get("stop_reason", "")
    calls   = iteration.get("tool_calls", [])
    results = iteration.get("tool_results", [])
    thoughts_list = iteration.get("thoughts", [])

    print()
    print(rule("─"))
    print(step(f"  STEP {step_num}") + meta(f"  (stop_reason: {stop})"))

    print_thoughts(thoughts_list)

    if calls:
        print()
        print(tool(f"  ⚙️  ACTIONS ({len(calls)})"))
        for i, call in enumerate(calls):
            print_action(call, i)

    if results:
        print()
        print(result(f"  📦  ACTION RESULTS  ({len(results)})"))
        for res in results:
            print_tool_result(res, calls)


def print_final_answer(trace: dict):
    ans = trace.get("final_answer", "")
    stop = trace.get("stop_reason", "")
    if not ans:
        return
    print()
    print(section_rule("═"))
    print(answer(f"  ✅  FINAL ANSWER") + meta(f"  (stop_reason: {stop})"))
    print(section_rule("═"))
    print(wrap_text(ans, prefix="  "))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def render_trace(trace: dict):
    print_header(trace)

    iterations = trace.get("iterations", [])
    if not iterations:
        print(warn("  ⚠  No iterations found in trace."))
    else:
        for it in iterations:
            step_num = it.get("step", "?")
            print_step(it, step_num)

    print_final_answer(trace)
    print(section_rule("═"))
    print(meta(f"  End of trace  ·  {len(iterations)} iteration(s)"))
    print(section_rule("═"))
    print()


def load_trace(path: str) -> dict:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Pretty-print a ReAct agent trace JSON file."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="-",
        help="Path to trace JSON file, or '-' to read from stdin (default: stdin)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    args = parser.parse_args()

    if args.no_color:
        C.enabled = False

    try:
        trace = load_trace(args.file)
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON — {e}", file=sys.stderr)
        sys.exit(1)

    render_trace(trace)


if __name__ == "__main__":
    main()
