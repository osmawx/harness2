# Claude Code executor (no API key)

This fork adds a third executor, `--substrate cc`. It runs the whole loop through
the [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI on its own
login (a Claude Pro/Max/Team subscription), so no API key is used anywhere:

| Role | How it runs on `cc` |
|---|---|
| Executor (solver rollouts) | `claude -p … --output-format stream-json` in a staged task workspace |
| Improver (propose / aggregate / select) | the `claude_cli` channel with subscription auth |
| JobBench judge | `harness2-claude-bridge`, a local OpenAI-compatible endpoint backed by `claude -p` |

Every child `claude` process starts with every `ANTHROPIC_*` and
`CLAUDE_CODE_USE_*` variable removed. A run can't fall back to metered API
billing or a cloud provider. Usage counts against your subscription's limits instead. With
`--k 2`, one task is about 8 solver rollouts, 6 improver sessions and one judge
call per rubric, so start with `--max-tasks 1`.

`cc` is implemented for **JobBench** only. LAB and WorkBuddy run their agents
inside their own harnesses (a patched OpenCode and Harbor containers), which
have no Claude Code launcher.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e '.[render]'
claude login                       # or: export CLAUDE_CODE_OAUTH_TOKEN=$(claude setup-token)
harness2-setup jb --substrate cc   # JobBench data and judge env; skips OpenCode/bun
```

`.env` (or shell):

```bash
HARNESS2_CLAUDE_MODEL=sonnet            # executor; any `claude --model` value
HARNESS2_CLAUDE_IMPROVER_MODEL=opus     # improver; defaults to the executor model
HARNESS2_CLAUDE_EFFORT=high             # executor --effort
HARNESS2_JUDGE_MODEL=sonnet
HARNESS2_JUDGE_BASE_URL=http://127.0.0.1:8765/v1
```

## Run

```bash
harness2-claude-bridge --port 8765 &    # judge endpoint; keep it running while judging
harness2-check --bench jb --substrate cc
harness2 --bench jb --substrate cc --k 1 --mode parallel --max-tasks 1
harness2-judge --bench jb --substrate cc --k 1 --mode parallel
harness2-report --bench jb --substrate cc --k 1 --mode parallel
```

`harness2-check` runs `claude --version` but never starts a model session. The
first real run is also the first check of your login.

A rollout where `claude` exits with an error is retried, and its partial output
is thrown away. Usage limits, auth and overload errors are infrastructure
failures, not the agent's. If the last attempt also fails, the rollout is left
unmarked and the next run re-rolls it. A timeout is the agent's own outcome:
whatever it wrote by then is scored.

## The editable surface

The improver edits the harness as Claude Code project files. The runner copies
them to the root of each task workspace.

| Path | Component | In context |
|---|---|---|
| `CLAUDE.md` | persona, rules, durable facts, tool prose | always on (12,000-char cap) |
| `.claude/skills/<name>/SKILL.md` | reusable workflow | description advertised; body on demand |
| `.claude/skills/<name>/scripts/*` | helper carried with its skill | run via Bash |
| `sub_agents/<name>/AGENT.md` | delegated subagent | installed as `.claude/agents/<name>.md` |
| `scripts/<name>.py` | deterministic helper | run on demand |
| `.claude/settings.json` | lifecycle hooks (`{"hooks": …}`) | inside the tool loop |

## Isolation

Rollouts, improver sessions and judge calls run with
`--setting-sources project,local --strict-mcp-config --no-session-persistence`.
Your own `~/.claude/settings.json`, hooks and MCP servers stay out of the
measured harness. To isolate everything else under `~/.claude`, such as a global
`CLAUDE.md`, skills and plugins, point `HARNESS2_CLAUDE_CONFIG_DIR` at an empty
directory. Then authenticate with `CLAUDE_CODE_OAUTH_TOKEN` (from
`claude setup-token`), because a fresh config dir has no stored login.

Rollouts run with `--dangerously-skip-permissions`, the same trust the Codex
executor gets. The prompt confines the agent to its workspace, but nothing
enforces it. Run on a machine or container where that is acceptable.

## Judge bridge

`harness2-claude-bridge` serves `POST /v1/chat/completions`. Each request runs one
`claude -p` session:

- System messages become `--system-prompt`.
- `image_url` data URLs are saved as files, and only the `Read` tool is enabled so
  the judge can open them. Text-only requests get no tools at all.
- Provider prefixes are stripped from the model name, so `vertex/sonnet` becomes
  `sonnet`.

`--concurrency` (default 8, `HARNESS2_CLAUDE_BRIDGE_CONCURRENCY`) caps the number of
parallel sessions. With a loopback judge URL, JobBench matches its judge workers
to this value and allows 900 s per rubric. A request whose client gave up while
it was queued is dropped instead of being run. The bridge binds to loopback. Binding elsewhere requires
`HARNESS2_CLAUDE_BRIDGE_KEY`, which clients then send as their API key.
