# Scaler Clone Agent

A conversational, ReAct-style CLI agent (Python) that clones the Scaler
Academy website. The agent reasons step-by-step
(`START -> THINK -> TOOL -> OBSERVE -> OUTPUT`), one step per LLM call, and
writes real files to disk. After the initial clone task finishes it drops
into an interactive chat loop so you can ask for follow-up changes.

## How the loop works

1. The agent sends the system prompt + message history to Groq.
2. The model returns exactly one JSON step.
3. If the step is `TOOL`, the agent runs the tool, appends an `OBSERVE`
   message with the result, and loops again.
4. If the step is `OUTPUT`, the agent prints it and waits for your next
   prompt.

The full task is **never** completed in a single LLM call — each call advances
exactly one step.

## Tech stack

- Python 3.10+
- [`groq`](https://pypi.org/project/groq/) — LLM client (`llama-3.3-70b-versatile`)
- [`requests`](https://pypi.org/project/requests/) — fetching pages
- [`python-dotenv`](https://pypi.org/project/python-dotenv/) — loading `GROQ_API_KEY`

## Project layout

```
scaler_clone_agent/
├── agent.py          # main agent loop, prompt, JSON step parser
├── tools.py          # tool implementations + dispatcher
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The agent writes its output to `scaler_clone/index.html` (created at the
project root on first run).

## Setup

### 1. Create a virtualenv and install dependencies

Windows (PowerShell):

```powershell
cd scaler_clone_agent
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd scaler_clone_agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your Groq API key

Get a free key from <https://console.groq.com>, then:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_key_here` with your real key:

```
GROQ_API_KEY=gsk_...
```

### 3. Run the agent

```bash
python agent.py
```

You will see the agent stream its reasoning steps. When it finishes the
initial clone task it will print:

```
Initial task done. Ask follow-ups (e.g. 'add a courses section', ...)
```

Type natural-language follow-ups at the `you>` prompt. For example:

- `add a courses section with 4 cards`
- `change the color scheme to a lighter palette`
- `add a testimonials section before the footer`
- `make the hero CTA button bigger and add a hover effect`

Type `exit` or `quit` (or hit Ctrl-C) to leave.

## Tools the agent has

| Tool | `tool_args` shape | What it does |
| ---- | ----------------- | ------------ |
| `execute_command` | shell command string | Runs the command via `subprocess`, returns stdout/stderr |
| `write_file` | `{"filepath": "...", "content": "..."}` | Writes a file, creating parent dirs |
| `read_file` | path string | Returns the file contents |
| `fetch_url` | URL string | HTTP GET with a browser User-Agent |

## Output

After the first run, open `scaler_clone/index.html` in your browser to see
the cloned site. The clone includes:

- A dark header with logo text and nav (Courses, Events, Blog, Login)
- A hero section with a bold headline, subheadline, CTA, and dark gradient
- A footer with link columns and copyright

## Troubleshooting

- **`ERROR: GROQ_API_KEY is not set`** — ensure `.env` exists in the same
  directory as `agent.py` and contains `GROQ_API_KEY=...`.
- **`Rate limit reached ... tokens per day (TPD)`** — the free tier caps at
  100k tokens/day for the 70b model. Either wait for the reset window shown
  in the error, switch to a smaller model by setting `GROQ_MODEL` in `.env`
  (e.g. `GROQ_MODEL=llama-3.1-8b-instant`), or upgrade at
  <https://console.groq.com/settings/billing>. Long sessions also benefit from
  the built-in history compaction — large `write_file` payloads are stubbed in
  history once the file is on disk so they don't get re-sent each turn.
- **Generated HTML looks too basic** — the system prompt has explicit visual
  requirements (Google Fonts, accent gradient, sticky blurred header, hero
  with decorative bg, mid-section, 4-column footer, responsive breakpoint,
  scroll JS). If the model still under-delivers, ask it directly in the chat
  loop: "rewrite the page with richer styling — gradient hero text, a stats
  strip, and a 4-column footer".
- **`[parse error]` repeating** — the model occasionally wraps its JSON in
  fences; the loop self-corrects, but if it persists, lower `temperature` in
  `agent.py` or try a different model id.
- **`fetch_url` returns `[error]`** — Scaler may rate-limit or block;
  the agent will fall back to writing a recreation from prior knowledge.
