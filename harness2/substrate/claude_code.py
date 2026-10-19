# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ClaudeCodeSubstrate: the Claude Code CLI (`claude -p`) as a solver substrate.

Runs on the CLI's own login (a Claude subscription via `claude login`, or a
`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`), never on an API key: every
Anthropic API credential is stripped from the child environment so a run cannot fall
back to metered API billing.

Carries the c1..c8 surface (CLAUDE_CODE_SURFACE), the one install mode (claude_root:
CLAUDE.md, .claude/ and scripts/ copied to the workspace root, where `claude -p` run in
that directory finds them as project settings), the `claude -p` flag contract, and the
stream-json trajectory renderer.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

CLAUDE_CODE_SURFACE = SurfaceSpec(
    name="claude_code",
    always_on=("CLAUDE.md",),
    always_on_max=12_000,
    always_on_cap_reason=(
        "CLAUDE.md is loaded whole into the context of EVERY session, so every token "
        "competes with the agent's own working budget"
    ),
    skill_dir=".claude/skills",
    skill_name_rule="overrides_dir",
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to CLAUDE.md, e.g. "When '
        '<the situation this skill handles>, use the `{name}` skill."'
    ),
    autotrigger_file="CLAUDE.md",
    trigger_line="- When {desc}, use the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=".claude/settings.json",
    hook_events=frozenset(
        {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "PermissionRequest",
            "Notification",
            "UserPromptSubmit",
            "SessionStart",
            "SessionEnd",
            "Stop",
            "SubagentStart",
            "SubagentStop",
            "PreCompact",
        }
    ),
    compile_skill_scripts=True,
    reject_empty_always_on=True,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            "sub_agents/README.md",
            "<!-- c5: delegated Claude Code subagents, each run in its OWN context via the "
            "Task tool. Add one as sub_agents/<name>/AGENT.md with frontmatter "
            "`description:`; the runner installs it as .claude/agents/<name>.md. -->\n",
        ),
        (
            ".claude/README.md",
            "<!-- c8: Claude Code lifecycle hooks -- the only lever that acts INSIDE the tool "
            "loop, so the model cannot skip it. Add ONE file, .claude/settings.json: "
            '{"hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", '
            '"command": "..."}]}]}}. A PostToolUse command that prints '
            '{"decision":"block","reason":"..."} feeds the reason back to the model. -->\n',
        ),
        (
            ".claude/skills/README.md",
            "<!-- c6: reusable workflows. Add one as .claude/skills/<name>/SKILL.md with YAML "
            "frontmatter: `description:` is REQUIRED; `name:` is optional and must match the "
            "directory. Only name+description are advertised to the agent; the body is read "
            "on demand. -->\n",
        ),
    ),
)

# What the workspace root receives from a staged tree.
CLAUDE_ROOT_ENTRIES = ("CLAUDE.md", ".claude", "scripts")

# Every variable that could route `claude` to metered API billing, a cloud provider or a
# remapped model. Stripped by prefix, so a provider switch added in a later CLI release is
# covered too; what remains is the CLI's own login -- the subscription OAuth session in
# the keychain, or CLAUDE_CODE_OAUTH_TOKEN when the caller exported one.
API_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
API_ENV_VARS = (
    "CLOUD_ML_REGION",
    # A parent Claude Code session marks its children; a solver is not one of them.
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
)

# Only the workspace's own settings load: the operator's ~/.claude settings, hooks and
# MCP servers must not leak into a rollout and confound the harness being measured.
ISOLATION_FLAGS = ("--setting-sources", "project,local", "--strict-mcp-config")

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)


def subscription_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """`base_env` minus every API/cloud credential, so `claude` uses its own login."""
    env = {
        k: v
        for k, v in base_env.items()
        if k not in API_ENV_VARS and not k.startswith(API_ENV_PREFIXES)
    }
    if config.CLAUDE_CONFIG_DIR:
        env["CLAUDE_CONFIG_DIR"] = str(config.CLAUDE_CONFIG_DIR)
    return env


def claude_binary() -> str:
    """The `claude` executable: HARNESS2_CLAUDE_BIN, else the first one on PATH."""
    return config.CLAUDE_BIN or shutil.which("claude") or "claude"


def agent_markdown(name: str, body: str) -> str:
    """An AGENT.md body as a Claude Code agent file: frontmatter with `name:` guaranteed.

    Claude Code identifies an agent by its frontmatter `name`, so a missing one is
    filled from the directory name and a body without frontmatter is given one.
    """
    m = _FRONTMATTER.match(body)
    if m is None:
        return f"---\nname: {name}\ndescription: the {name} subagent\n---\n\n{body}"
    head = m.group(1)
    if re.search(r"^name:\s*\S", head, re.M):
        return body
    return f"---\nname: {name}\n{head}\n---\n" + body[m.end() :]


class ClaudeCodeSubstrate:
    """Claude Code CLI behind the one Substrate interface; the launcher is benchmark-owned."""

    name = "cc"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return CLAUDE_CODE_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        """Stage the claude_root entries; sub_agents/<n>/AGENT.md become .claude/agents/<n>.md."""
        if spec.install_mode != "claude_root":
            raise ValueError(
                f"install_mode {spec.install_mode!r} is not the Claude Code mode; this "
                f"substrate has exactly one: 'claude_root'"
            )
        tree_dir = Path(tree_dir)
        if not HT.is_tree(CLAUDE_CODE_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not a Claude Code harness tree (no CLAUDE.md)")
        staged = Path(tempfile.mkdtemp(prefix="adapt-cc-root-"))
        try:
            for entry in CLAUDE_ROOT_ENTRIES:
                src = tree_dir / entry
                if src.is_dir():
                    shutil.copytree(src, staged / entry)
                elif src.is_file():
                    shutil.copy2(src, staged / entry)
            for agent in sorted(tree_dir.glob("sub_agents/*/AGENT.md")):
                name = agent.parent.name
                dest = staged / ".claude" / "agents" / f"{name}.md"
                dest.parent.mkdir(parents=True, exist_ok=True)
                body = agent.read_text(encoding="utf-8", errors="replace")
                dest.write_text(agent_markdown(name, body), encoding="utf-8")
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={
                "install_mode": "claude_root",
                **{k: v for k, v in ctx.items() if not k.startswith("file:")},
            },
        )

    def exec_argv(
        self, prompt: str, *, model: str, effort: str = "", stream: bool = True
    ) -> list[str]:
        """The one `claude -p` argv. Run it with cwd=<workspace> and `subscription_env`.

        The prompt travels as an argument; permissions are bypassed because the rollout is
        unattended (the same trust codex gets with --dangerously-bypass-approvals-and-sandbox).
        """
        argv = [
            claude_binary(),
            "-p",
            prompt,
            "--output-format",
            "stream-json" if stream else "json",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            *ISOLATION_FLAGS,
        ]
        # stream-json in print mode requires --verbose, or the CLI refuses to start.
        if stream:
            argv.append("--verbose")
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        return argv

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "ClaudeCodeSubstrate was constructed without a launcher body; the benchmark "
                "adapter must supply one (it owns workspace layout and read-back locations)"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        """Check the CLI runs and that no API key would be used; starts no model session."""
        binary = claude_binary()
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Claude Code CLI not runnable ({binary}: {type(exc).__name__}); install it "
                f"(https://docs.anthropic.com/en/docs/claude-code) and run `claude login`, or "
                f"set HARNESS2_CLAUDE_BIN"
            ) from exc
        if r.returncode != 0:
            raise RuntimeError(
                f"`{binary} --version` exited {r.returncode}: {r.stderr.strip()[-300:]}"
            )


_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def _result_text(content) -> str:
    """tool_result content is a string or a list of {type: text|image} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "image":
                parts.append("(image)")
        return "\n".join(p for p in parts if p)
    return "" if content is None else json.dumps(content, ensure_ascii=False, default=str)


def _tool_args(name: str, tool_input) -> str:
    """Bash shows its command line; every other tool shows its JSON input."""
    if isinstance(tool_input, dict) and name == "Bash" and "command" in tool_input:
        return str(tool_input["command"])
    return json.dumps(tool_input, ensure_ascii=False, default=str)


def parse_events(path: Path) -> list[dict]:
    """Read a stream-json capture (junk lines tolerated) in stream order."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [{"type": "_read_error", "message": str(e)}]
    events = []
    for ln in raw.replace("\x00", "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def _render_assistant(content: list, lines: list[str], obs_trunc: int, args_trunc: int) -> None:
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and str(block.get("text") or "").strip():
            lines.append(f"[ASSISTANT] {_clip(block['text'], obs_trunc)}")
        elif btype == "thinking" and str(block.get("thinking") or "").strip():
            lines.append(f"[REASONING] {_clip(block['thinking'], obs_trunc)}")
        elif btype == "tool_use":
            name = str(block.get("name") or "tool")
            lines.append(
                f"[ACTION] {name}: {_clip(_tool_args(name, block.get('input')), args_trunc)}"
            )


def _render_user(content, lines: list[str], obs_trunc: int) -> None:
    if isinstance(content, str):
        if content.strip():
            lines.append(f"[USER] {_clip(content, obs_trunc)}")
        return
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            tag = "OBSERVATION ERROR" if block.get("is_error") else "OBSERVATION"
            lines.append(f"[{tag}] {_clip(_result_text(block.get('content')), obs_trunc)}")
        elif block.get("type") == "text" and str(block.get("text") or "").strip():
            lines.append(f"[USER] {_clip(block['text'], obs_trunc)}")


def _render_result(ev: dict, lines: list[str]) -> None:
    usage = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
    lines.append(
        "[HARNESS] session {st} | turns={t} | tokens in={i} cache_read={c} out={o}".format(
            st=ev.get("subtype") or "end",
            t=ev.get("num_turns"),
            i=usage.get("input_tokens"),
            c=usage.get("cache_read_input_tokens"),
            o=usage.get("output_tokens"),
        )
    )
    if ev.get("is_error"):
        lines.append(f"[ERROR] {_clip(ev.get('result') or ev.get('subtype'), 1000)}")


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000) -> str:
    """Render a `claude -p --output-format stream-json` capture into the shared grammar."""
    lines: list[str] = []
    for ev in parse_events(Path(path)):
        et = ev.get("type")
        if et == "_read_error":
            return f"(failed to read trajectory: {ev.get('message')})"
        message = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        if et == "assistant":
            _render_assistant(message.get("content") or [], lines, obs_trunc, args_trunc)
        elif et == "user":
            _render_user(message.get("content"), lines, obs_trunc)
        elif et == "result":
            _render_result(ev, lines)
        elif et == "system" and ev.get("subtype") == "init":
            lines.append(f"[HARNESS] session start | model={ev.get('model')}")
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered


def final_message(path: Path) -> str:
    """The `result` event's text, else the last assistant text block."""
    last = ""
    for ev in parse_events(Path(path)):
        if ev.get("type") == "result" and str(ev.get("result") or "").strip():
            return str(ev["result"])
        message = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        if ev.get("type") == "assistant":
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    if str(block.get("text") or "").strip():
                        last = str(block["text"])
    return last
