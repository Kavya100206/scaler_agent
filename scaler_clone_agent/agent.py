"""Conversational ReAct-style CLI agent that clones the Scaler Academy site.

Architecture
------------
The agent runs in a while loop. On each iteration the LLM emits exactly one
JSON step:

    { "step": "START|THINK|TOOL|OBSERVE|OUTPUT",
      "content": "...",
      "tool_name": "...",   # only for TOOL
      "tool_args": "..." }  # only for TOOL

* We append the assistant's step to the message history.
* If the step is TOOL, we run the tool and inject an OBSERVE message
  (as a `user` role) carrying the tool's output, then loop again.
* If the step is OUTPUT, we stop the inner loop and hand control back to
  the interactive prompt for follow-up questions.

The agent never completes the task in a single LLM call — each call advances
exactly one step.
"""

import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq

from tools import call_tool


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    print("ERROR: GROQ_API_KEY is not set.")
    print("Copy .env.example to .env and paste your Groq API key.")
    sys.exit(1)

client = Groq(api_key=API_KEY)
# Model is overridable via env var so you can switch to a smaller / less
# rate-limited model (e.g. "llama-3.1-8b-instant") without editing code.
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_ITERS_PER_TURN = 40  # safety cap on loop iterations per user message

# When a write_file payload exceeds this many characters, we keep the FULL
# payload only long enough to dispatch the tool and store a stub in history
# instead. This is the single biggest token-saver across long sessions.
WRITE_FILE_HISTORY_STUB_THRESHOLD = 600

SYSTEM_PROMPT = """You are a step-by-step ReAct agent that builds and edits websites.

PROTOCOL (strict). Every response = ONE JSON object, no fences, no prose:
{"step":"START|THINK|TOOL|OBSERVE|OUTPUT","content":"...","tool_name":"...","tool_args":"..."}
tool_name and tool_args only when step == TOOL.

STEPS
- START: acknowledge a new user task + brief plan.
- THINK: reason about next action; multiple THINKs in a row are fine.
- TOOL: invoke ONE tool. Do NOT emit OBSERVE yourself.
- OBSERVE: system-injected after each TOOL. Never produce it yourself.
- OUTPUT: final answer; only when the task is fully done.
RULE: emit exactly ONE step per response. Do not chain steps.

TOOLS
- execute_command -> tool_args = shell command string.
- write_file      -> tool_args = JSON {"filepath":"...","content":"..."}; escape \\n and \\" in content.
- read_file       -> tool_args = path string.
- fetch_url       -> tool_args = URL string.

BUILD GUIDANCE
- Save the site under scaler_clone/. One self-contained index.html with inline <style>/<script>.
- After fetch_url, do NOT copy raw HTML; synthesize a polished modern recreation.
- For follow-up edits: read_file first, then write_file the updated version.

VISUAL SPEC (non-negotiable; a 30-line Arial stub is rejected)
Fonts/color: load Inter via Google Fonts <link>. Bg #0a0b0f, text #fff, muted #9aa0aa, accent gradient #4f46e5 -> #7c3aed -> #06b6d4 (apply to key headline word + primary CTA).
Header: sticky, dark, backdrop-filter:blur(12px), faint bottom border. Brand "Scaler" as styled text (no <img>) with second half in the gradient. Nav (Courses, Events, Blog) with hover-underline animation. Right-side primary "Login" button (gradient bg, hover lift).
Hero: min-height 80vh, max-width ~1100px container. Headline clamp(2.5rem,5vw,4.5rem) weight 700+, key phrase wrapped in <span> using background-clip:text gradient. Muted subheadline max-width 640px, line-height 1.6. Two CTAs: primary (gradient) + secondary (outlined), with hover transitions. Decorative radial-gradient blob OR dot pattern absolutely positioned behind. 3-4 trust pills below the CTAs.
Mid section (REQUIRED, pick one): 3-col features grid, OR stats strip, OR 4-card courses preview.
Footer: 4-col grid (Company / Programs / Resources / Connect, 4-6 links each), top border, bottom row with copyright + Terms/Privacy.
Responsive: @media (max-width:768px) stacks nav, shrinks hero font, single-col footer.
JS: scroll handler that adds .scrolled class to header when scrollY>30 and reduces padding.
Size target: 400-700 lines total. Under 200 lines = under-delivered; add polish before OUTPUT.
"""

INITIAL_TASK = (
    "Clone the Scaler Academy website (scaler.com). Generate a working HTML file "
    "with CSS and JS that visually resembles the site. It must include a Header, "
    "Hero Section, and Footer. Save it as scaler_clone/index.html"
)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove ``` or ```json fences if the model added them despite instructions."""
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _parse_step(raw: str) -> dict:
    """Parse one LLM response into a step dict, tolerating fenced code blocks."""
    return json.loads(_strip_fences(raw))


# ---------------------------------------------------------------------------
# History compaction
# ---------------------------------------------------------------------------

# OBSERVE bodies older than the most recent one get stubbed if longer than this.
# Mostly targets large fetch_url dumps whose research value has already been
# consumed by the THINKs that followed.
OBSERVE_HISTORY_STUB_THRESHOLD = 400


def _compact_history(messages: list[dict]) -> None:
    """Shrink older OBSERVE payloads in place to stay under tight TPM limits.

    The most recent OBSERVE is kept verbatim — the model has just acted on
    it and may still reference it. Older OBSERVE bodies are replaced with a
    short stub. write_file payloads are already stubbed at insert time, so
    the only large items left in older history are fetch_url responses.
    """
    observe_indices: list[int] = []
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        try:
            obs = json.loads(m.get("content", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obs, dict) and obs.get("step") == "OBSERVE":
            observe_indices.append(i)

    # Preserve the most recent OBSERVE; stub earlier large ones.
    for i in observe_indices[:-1]:
        try:
            obs = json.loads(messages[i]["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        body = obs.get("content", "")
        if isinstance(body, str) and len(body) > OBSERVE_HISTORY_STUB_THRESHOLD:
            obs["content"] = f"<elided earlier observation: {len(body)} chars>"
            messages[i] = {"role": "user", "content": json.dumps(obs)}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _llm_step(messages: list[dict]) -> str:
    """Make a single LLM call and return the raw assistant content."""
    _compact_history(messages)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int = 500) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n} more]"


def _print_step(step: dict, tool_result: str | None = None) -> None:
    kind = step.get("step", "?").upper()
    content = step.get("content", "")
    if kind == "START":
        print(f"\n[START] {content}")
    elif kind == "THINK":
        print(f"\n[THINK] {content}")
    elif kind == "TOOL":
        name = step.get("tool_name", "?")
        args_preview = _truncate(step.get("tool_args", ""), 240)
        print(f"\n[TOOL ] {name}({args_preview})")
        if tool_result is not None:
            print(f"   ->  {_truncate(tool_result, 400)}")
    elif kind == "OUTPUT":
        print(f"\n[OUT  ] {content}")
    else:
        print(f"\n[{kind}] {content}")


# ---------------------------------------------------------------------------
# The ReAct loop — one user turn
# ---------------------------------------------------------------------------

def run_turn(messages: list[dict]) -> None:
    """Drive the agent until it emits OUTPUT (or hits the iteration cap).

    `messages` is mutated in place so follow-up turns inherit history.
    """
    for _ in range(MAX_ITERS_PER_TURN):
        try:
            raw = _llm_step(messages)
        except Exception as e:
            print(f"\n[llm error] {e}")
            return

        try:
            step = _parse_step(raw)
        except json.JSONDecodeError as e:
            # Tell the model its output was malformed and let it self-correct.
            print(f"\n[parse error] {e} :: {raw[:200]}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Reply with "
                    "exactly one JSON object matching the protocol."
                ),
            })
            continue

        kind = step.get("step", "").upper()

        if kind == "TOOL":
            tool_name = step.get("tool_name", "")
            tool_args = step.get("tool_args", "")

            # Persist a compacted version of write_file in history: the file is
            # already on disk, so the model doesn't need its own giant payload
            # echoed back. If it needs the content again it can call read_file.
            # This is the single biggest token-saver across long sessions.
            if (
                tool_name == "write_file"
                and isinstance(tool_args, str)
                and len(tool_args) > WRITE_FILE_HISTORY_STUB_THRESHOLD
            ):
                stubbed = dict(step)
                stubbed["tool_args"] = (
                    "<elided write_file payload — content already on disk; "
                    "use read_file to retrieve current contents>"
                )
                messages.append({"role": "assistant", "content": json.dumps(stubbed)})
            else:
                messages.append({"role": "assistant", "content": json.dumps(step)})

            result = call_tool(tool_name, tool_args)
            _print_step(step, tool_result=result)
            # Inject an OBSERVE message as the next user turn.
            observe = {"step": "OBSERVE", "content": result}
            messages.append({"role": "user", "content": json.dumps(observe)})
            continue

        # Non-TOOL steps: persist verbatim so the model sees its own trace.
        messages.append({"role": "assistant", "content": json.dumps(step)})

        if kind == "OBSERVE":
            # Model should not emit OBSERVE itself. Nudge it back on protocol.
            print(f"\n[warn] model emitted OBSERVE; nudging back to protocol")
            messages.append({
                "role": "user",
                "content": "Do not emit OBSERVE yourself — the system injects it. Continue with THINK, TOOL, or OUTPUT.",
            })
            continue

        _print_step(step)
        if kind == "OUTPUT":
            return
        if kind not in {"START", "THINK"}:
            messages.append({
                "role": "user",
                "content": "Unknown step. Use START, THINK, TOOL, or OUTPUT.",
            })

    print("\n[loop hit MAX_ITERS_PER_TURN; pausing for next user input]")


# ---------------------------------------------------------------------------
# Entry point — initial task, then interactive chat
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print(" Scaler Clone Agent")
    print(" model: " + MODEL)
    print("=" * 64)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_TASK},
    ]

    # Phase 1: hard-coded initial task.
    run_turn(messages)

    # Phase 2: interactive follow-up loop.
    print("\n" + "-" * 64)
    print(" Initial task done. Ask follow-ups (e.g. 'add a courses section',")
    print(" 'change the color scheme to light'). Type 'exit' or 'quit' to leave.")
    print("-" * 64)

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("bye.")
            break
        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
