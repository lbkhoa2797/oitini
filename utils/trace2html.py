#!/usr/bin/env python3
"""
trace_to_html.py — Convert a ReAct agent trace JSON into a self-contained HTML
file viewable in VS Code's built-in Markdown/HTML preview, or any browser.

Usage:
    python trace_to_html.py trace.json                  # writes trace.html
    python trace_to_html.py trace.json -o report.html   # custom output path
    cat trace.json | python trace_to_html.py -          # stdin → stdout
"""

import json
import sys
import argparse
import html as html_mod
from datetime import datetime
from pathlib import Path


# ── Font presets ──────────────────────────────────────────────────────────────
# Each entry: (google_fonts_query, css_mono_stack, css_sans_stack)

FONT_PRESETS = {
    "jetbrains": (
        "family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800",
        "'JetBrains Mono', monospace",
        "'Syne', sans-serif",
    ),
    "fira": (
        "family=Fira+Code:wght@400;600&family=Fira+Sans:wght@400;700;800",
        "'Fira Code', monospace",
        "'Fira Sans', sans-serif",
    ),
    "ibm": (
        "family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;700",
        "'IBM Plex Mono', monospace",
        "'IBM Plex Sans', sans-serif",
    ),
    "source": (
        "family=Source+Code+Pro:wght@400;600&family=Source+Sans+3:wght@400;700",
        "'Source Code Pro', monospace",
        "'Source Sans 3', sans-serif",
    ),
    "cascadia": (
        "family=Cascadia+Code:wght@400;600&family=Nunito:wght@400;700;800",
        "'Cascadia Code', monospace",
        "'Nunito', sans-serif",
    ),
    "inconsolata": (
        "family=Inconsolata:wght@400;600&family=Outfit:wght@400;700;800",
        "'Inconsolata', monospace",
        "'Outfit', sans-serif",
    ),
    "system": (
        None,  # no Google Fonts load
        "ui-monospace, 'Cascadia Code', 'SF Mono', Menlo, Consolas, monospace",
        "system-ui, sans-serif",
    ),
}

FONT_NAMES = list(FONT_PRESETS.keys())


def font_link(preset: str) -> str:
    """Return the <link> tag(s) for the chosen font preset, or empty string."""
    gf_query, _, _ = FONT_PRESETS[preset]
    if gf_query is None:
        return ""
    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        f'<link href="https://fonts.googleapis.com/css2?{gf_query}&display=swap" rel="stylesheet">'
    )


def font_vars(preset: str) -> str:
    """Return CSS variable declarations for --mono and --sans."""
    _, mono, sans = FONT_PRESETS[preset]
    return f"--mono: {mono};\n    --sans: {sans};"


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Trace · {run_id}</title>
{font_link}
<style>
  :root {{
    {theme_vars}
    --radius:    8px;
    {font_vars}
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.65;
    padding: 32px 24px;
    max-width: 900px;
    margin: 0 auto;
  }}

  /* ── Header ── */
  .trace-header {{
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius);
    background: var(--bg2);
    padding: 20px 24px;
    margin-bottom: 28px;
    animation: fadeSlide .4s ease both;
  }}
  .trace-header h1 {{
    font-family: var(--sans);
    font-size: 18px;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: .03em;
    margin-bottom: 8px;
  }}
  .meta-grid {{
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }}
  .meta-item {{ color: var(--dim); font-size: 11px; }}
  .meta-item span {{ color: var(--text); }}
  .prompt-box {{
    background: var(--bg3);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 10px 14px;
    color: var(--text);
    font-size: 13px;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  .prompt-label {{
    font-family: var(--sans);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .1em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 6px;
  }}

  /* ── Steps ── */
  .step {{
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg2);
    margin-bottom: 18px;
    overflow: hidden;
    animation: fadeSlide .4s ease both;
  }}
  .step-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    background: var(--bg3);
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
  }}
  .step-num {{
    font-family: var(--sans);
    font-weight: 800;
    font-size: 13px;
    color: var(--gold);
    letter-spacing: .05em;
  }}
  .stop-badge {{
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 20px;
    background: var(--badge-bg);
    color: var(--dim);
    letter-spacing: .06em;
  }}
  .chevron {{
    margin-left: auto;
    color: var(--dim);
    transition: transform .2s;
    font-size: 11px;
  }}
  .step.collapsed .chevron {{ transform: rotate(-90deg); }}
  .step.collapsed .step-body {{ display: none; }}
  .step-body {{ padding: 16px; display: flex; flex-direction: column; gap: 16px; }}

  /* ── Thoughts ── */
  .section-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 7px;
  }}
  .thoughts .section-label {{ color: var(--lavender); }}
  .thought-text {{
    background: var(--thought-bg);
    border-left: 3px solid var(--lavender);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 9px 12px;
    color: var(--lavender);
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12.5px;
  }}

  /* ── Tool calls / results ── */
  .tools-grid {{ display: flex; flex-direction: column; gap: 10px; }}

  .tool-card {{
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .tool-card-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    font-size: 11.5px;
    font-weight: 600;
  }}
  .tool-call .tool-card-header  {{ background: var(--sage-bg);  color: var(--sage);  border-bottom: 1px solid var(--sage-border); }}
  .tool-result .tool-card-header {{ background: var(--peach-bg); color: var(--peach); border-bottom: 1px solid var(--peach-border); }}

  .tool-id {{ font-size: 10px; color: var(--dim); margin-left: auto; font-weight: 400; }}
  .tool-body {{
    padding: 10px 12px;
    background: var(--bg3);
  }}
  pre.json {{
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text);
    white-space: pre;
    overflow-x: auto;
    line-height: 1.55;
  }}
  /* Minimal JSON syntax highlight */
  .jk {{ color: var(--jk); }}   /* key    */
  .js {{ color: var(--js); }}   /* string */
  .jn {{ color: var(--jn); }}   /* number */
  .jb {{ color: var(--jb); }}   /* bool/null */

  /* ── Final answer ── */
  .final-answer {{
    border: 1px solid var(--mint-border);
    border-top: 3px solid var(--mint);
    border-radius: var(--radius);
    background: var(--bg2);
    padding: 20px 24px;
    margin-top: 8px;
    animation: fadeSlide .4s ease both;
  }}
  .final-answer .section-label {{ color: var(--mint); font-size: 11px; margin-bottom: 10px; }}
  .final-answer-text {{
    color: var(--mint);
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.75;
    font-size: 13px;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    color: var(--dim);
    font-size: 11px;
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }}

  @keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0);   }}
  }}
</style>
</head>
<body>

{header_html}
{steps_html}
{answer_html}

<div class="footer">
  {n_steps} iteration(s) &nbsp;·&nbsp; {run_id} &nbsp;·&nbsp; generated by trace_to_html.py
</div>

<script>
  document.querySelectorAll('.step-header').forEach(h => {{
    h.addEventListener('click', () => h.closest('.step').classList.toggle('collapsed'));
  }});
</script>
</body>
</html>
"""


# ── JSON syntax highlighter (pure string-replace, no regex) ──────────────────

def highlight_json(obj) -> str:
    """Return syntax-highlighted HTML for a JSON object."""
    raw = json.dumps(obj, indent=2, ensure_ascii=False)
    out = []
    i = 0
    n = len(raw)

    def peek():
        return raw[i] if i < n else ""

    while i < n:
        c = raw[i]

        if c == '"':
            # Collect string
            j = i + 1
            while j < n:
                if raw[j] == '\\':
                    j += 2
                    continue
                if raw[j] == '"':
                    break
                j += 1
            token = raw[i:j+1]
            # Is this a key? Look ahead for ':'
            k = j + 1
            while k < n and raw[k] in ' \t\n':
                k += 1
            if k < n and raw[k] == ':':
                out.append(f'<span class="jk">{html_mod.escape(token)}</span>')
            else:
                out.append(f'<span class="js">{html_mod.escape(token)}</span>')
            i = j + 1

        elif c in '-0123456789':
            j = i
            while j < n and raw[j] in '-0123456789.eE+':
                j += 1
            out.append(f'<span class="jn">{html_mod.escape(raw[i:j])}</span>')
            i = j

        elif raw[i:i+4] in ('true', 'fals', 'null'):
            for kw in ('true', 'false', 'null'):
                if raw[i:i+len(kw)] == kw:
                    out.append(f'<span class="jb">{kw}</span>')
                    i += len(kw)
                    break
            else:
                out.append(html_mod.escape(c))
                i += 1

        else:
            out.append(html_mod.escape(c))
            i += 1

    return ''.join(out)


# ── Section builders ──────────────────────────────────────────────────────────

def build_header(trace: dict) -> str:
    run_id  = trace.get('run_id', '?')
    model   = trace.get('model', '?')
    prompt  = html_mod.escape(trace.get('user_prompt', ''))
    started = trace.get('started_at')
    elapsed = trace.get('wall_time_s')

    ts_html = ''
    if started:
        ts = datetime.fromtimestamp(started).strftime('%Y-%m-%d %H:%M:%S')
        ts_html = f'<div class="meta-item">Started <span>{ts}</span></div>'

    elapsed_html = ''
    if elapsed is not None:
        elapsed_html = f'<div class="meta-item">Elapsed <span>{elapsed:.2f}s</span></div>'

    return f"""
<div class="trace-header">
  <h1>⬡ ReAct Agent Trace &nbsp;·&nbsp; {html_mod.escape(run_id)}</h1>
  <div class="meta-grid">
    <div class="meta-item">Model <span>{html_mod.escape(model)}</span></div>
    {ts_html}
    {elapsed_html}
  </div>
  <div class="prompt-label">User Prompt</div>
  <div class="prompt-box">{prompt}</div>
</div>"""


def build_thoughts(thoughts: list) -> str:
    if not thoughts:
        return ''
    items = ''.join(
        f'<div class="thought-text">{html_mod.escape(t)}</div>' for t in thoughts
    )
    return f"""
<div class="thoughts">
  <div class="section-label">💭 Thoughts</div>
  {items}
</div>"""


def build_tool_calls(calls: list) -> str:
    if not calls:
        return ''
    cards = []
    for c in calls:
        name = html_mod.escape(c.get('name', '?'))
        cid  = html_mod.escape(c.get('id', ''))
        inp  = c.get('input', {})
        cards.append(f"""
<div class="tool-card tool-call">
  <div class="tool-card-header">
    ⚙ &nbsp;{name}
    <span class="tool-id">{cid}</span>
  </div>
  <div class="tool-body"><pre class="json">{highlight_json(inp)}</pre></div>
</div>""")
    return f"""
<div>
  <div class="section-label" style="color:var(--sage)">⚙ Tool Calls ({len(calls)})</div>
  <div class="tools-grid">{''.join(cards)}</div>
</div>"""


def build_tool_results(results: list, calls: list) -> str:
    if not results:
        return ''
    id_to_name = {c.get('id', ''): c.get('name', '') for c in calls}
    cards = []
    for r in results:
        tid   = r.get('tool_use_id', '')
        name  = html_mod.escape(id_to_name.get(tid, ''))
        raw   = r.get('result', '')
        label = f"📦 {name}" if name else "📦 Result"
        cards.append(f"""
<div class="tool-card tool-result">
  <div class="tool-card-header">
    {html_mod.escape(label)}
    <span class="tool-id">{html_mod.escape(tid)}</span>
  </div>
  <div class="tool-body"><pre class="json">{highlight_json(raw)}</pre></div>
</div>""")
    return f"""
<div>
  <div class="section-label" style="color:var(--peach)">📦 Tool Results ({len(results)})</div>
  <div class="tools-grid">{''.join(cards)}</div>
</div>"""


def build_step(it: dict, idx: int) -> str:
    step_num = it.get('step', idx)
    stop     = html_mod.escape(it.get('stop_reason', ''))
    calls    = it.get('tool_calls', [])
    results  = it.get('tool_results', [])
    thoughts = it.get('thoughts', [])

    body_parts = [
        build_thoughts(thoughts),
        build_tool_calls(calls),
        build_tool_results(results, calls),
    ]
    body = '\n'.join(p for p in body_parts if p)

    return f"""
<div class="step">
  <div class="step-header">
    <span class="step-num">STEP {step_num}</span>
    <span class="stop-badge">{stop}</span>
    <span class="chevron">▾</span>
  </div>
  <div class="step-body">{body}</div>
</div>"""


def build_final_answer(trace: dict) -> str:
    ans  = trace.get('final_answer', '')
    stop = html_mod.escape(trace.get('stop_reason', ''))
    if not ans:
        return ''
    return f"""
<div class="final-answer">
  <div class="section-label">✅ Final Answer &nbsp;<span style="font-weight:400;color:var(--dim)">({stop})</span></div>
  <div class="final-answer-text">{html_mod.escape(ans)}</div>
</div>"""


# ── Theme palettes ─────────────────────────────────────────────────────────────────────────────────────

THEMES = {
    "dark": """--bg:        #1a1a24;
    --bg2:       #22222f;
    --bg3:       #2a2a3a;
    --border:    #3a3a52;
    --text:      #e8e3f0;
    --dim:       #7a748a;
    --accent:    #c87941;
    --gold:      #c87941;
    --lavender:  #b39ddb;
    --sage:      #81c995;
    --peach:     #e8956d;
    --mint:      #d4a853;
    --badge-bg:  #2a3347;
    --thought-bg: rgba(192,168,240,.07);
    --sage-bg:   rgba(126,201,154,.10);
    --sage-border: rgba(126,201,154,.2);
    --peach-bg:  rgba(240,160,126,.10);
    --peach-border: rgba(240,160,126,.2);
    --mint-border: rgba(142,240,194,.25);
    --jk: #7eb8f7;
    --js: #a8d8a8;
    --jn: #f0c87e;
    --jb: #f0a07e;""",

    "light": """--bg:        #faf9f7;
    --bg2:       #ffffff;
    --bg3:       #f0ede8;
    --border:    #ccc5bb;
    --text:      #1a1714;
    --dim:       #5a534e;
    --accent:    #b06020;
    --gold:      #8f4d18;
    --lavender:  #4e3080;
    --sage:      #165c2e;
    --peach:     #962e10;
    --mint:      #5e3e08;
    --badge-bg:  #e8e2da;
    --thought-bg: rgba(78,48,128,.06);
    --sage-bg:   rgba(22,92,46,.08);
    --sage-border: rgba(22,92,46,.2);
    --peach-bg:  rgba(150,46,16,.08);
    --peach-border: rgba(150,46,16,.2);
    --mint-border: rgba(94,62,8,.2);
    --jk: #1a4f8a;
    --js: #1a5c2e;
    --jn: #7a4800;
    --jb: #8a2010;""",
}


def theme_vars(theme: str) -> str:
    return THEMES[theme]


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_html(trace: dict, font: str = "system", theme: str = "dark") -> str:
    run_id     = trace.get('run_id', '?')
    iterations = trace.get('iterations', [])

    header_html = build_header(trace)
    steps_html  = '\n'.join(build_step(it, i) for i, it in enumerate(iterations))
    answer_html = build_final_answer(trace)

    return HTML_TEMPLATE.format(
        run_id=html_mod.escape(run_id),
        font_link=font_link(font),
        font_vars=font_vars(font),
        theme_vars=theme_vars(theme),
        header_html=header_html,
        steps_html=steps_html,
        answer_html=answer_html,
        n_steps=len(iterations),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a ReAct agent trace JSON to a self-contained HTML file."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="-",
        help="Path to trace JSON (or '-' for stdin)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HTML path (default: <input>.html, or stdout for stdin input)",
    )
    parser.add_argument(
        "--font",
        default="system",
        choices=FONT_NAMES,
        metavar="FONT",
        help=(
            "Font preset. Choices: " + ", ".join(FONT_NAMES) + "  (default: system)"
        ),
    )
    parser.add_argument(
        "--theme",
        default="light",
        choices=["dark", "light"],
        help="Color theme: dark (Claude dark) or light (Claude light)  (default: dark)",
    )
    args = parser.parse_args()

    # Load
    if args.file == "-":
        trace = json.load(sys.stdin)
        out_path = args.output  # None → stdout
    else:
        with open(args.file, encoding="utf-8") as f:
            trace = json.load(f)
        out_path = args.output or str(Path(args.file).with_suffix(".html"))

    html = render_html(trace, font=args.font, theme=args.theme)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✔  Written to {out_path}")
        print(f"   Open in VS Code: code {out_path}")
        print(f"   Then press Ctrl+Shift+V (or Cmd+Shift+V on Mac) to preview.")
    else:
        sys.stdout.write(html)


if __name__ == "__main__":
    main()