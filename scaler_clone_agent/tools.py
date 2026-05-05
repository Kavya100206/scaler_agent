"""Tool implementations exposed to the agent.

Each tool returns a string (success output or an error message starting with
'[error]'). The dispatcher `call_tool` parses `tool_args` flexibly: a JSON
object becomes kwargs, a JSON scalar becomes a single positional arg, anything
unparseable is passed through as a plain string. This lets the LLM use
single-string tool_args for simple tools (execute_command, fetch_url, read_file)
and a JSON object for write_file which needs two fields.
"""

import json
import subprocess
from pathlib import Path

import requests


def execute_command(cmd: str) -> str:
    """Run a shell command via subprocess and return combined stdout/stderr."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = result.stdout or ""
        if result.stderr:
            out += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            out += f"\n[exit code]: {result.returncode}"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error]: command timed out after 120s"
    except Exception as e:
        return f"[error]: {e}"


def write_file(filepath: str, content: str) -> str:
    """Write `content` to `filepath`, creating parent directories as needed."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {filepath}"
    except Exception as e:
        return f"[error]: {e}"


def read_file(filepath: str) -> str:
    """Return the text contents of `filepath`."""
    try:
        return Path(filepath).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[error]: file not found: {filepath}"
    except Exception as e:
        return f"[error]: {e}"


def fetch_url(url: str) -> str:
    """HTTP GET `url` with a browser User-Agent. Truncates very large bodies."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        text = resp.text
        # Cap response size so we don't blow the LLM context / token budget.
        # The agent only needs page structure cues, not the whole DOM.
        # Tuned conservatively for small models with tight TPM limits.
        limit = 4000
        if len(text) > limit:
            text = text[:limit] + f"\n...[truncated, original length {len(resp.text)} chars]"
        return text
    except requests.RequestException as e:
        return f"[error]: request failed: {e}"
    except Exception as e:
        return f"[error]: {e}"


TOOLS = {
    "execute_command": execute_command,
    "write_file": write_file,
    "read_file": read_file,
    "fetch_url": fetch_url,
}


def call_tool(tool_name: str, tool_args):
    """Dispatch a tool call. `tool_args` may be a JSON string, dict, or scalar."""
    fn = TOOLS.get(tool_name)
    if fn is None:
        return f"[error]: unknown tool '{tool_name}'. Available: {list(TOOLS)}"

    parsed = tool_args
    if isinstance(tool_args, str):
        stripped = tool_args.strip()
        # Only attempt JSON parse if it looks structured — avoids mangling
        # plain strings like shell commands or URLs.
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = tool_args

    try:
        if isinstance(parsed, dict):
            return fn(**parsed)
        if isinstance(parsed, list):
            return fn(*parsed)
        return fn(parsed)
    except TypeError as e:
        return f"[error]: bad arguments for {tool_name}: {e}"
    except Exception as e:
        return f"[error]: tool {tool_name} raised: {e}"
