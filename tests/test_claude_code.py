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

"""The Claude Code (cc) substrate: surface, install, argv/env, trajectory, JobBench rollout,
judge bridge and subscription-auth improver -- all against a fake `claude` binary, so no
model is called and nothing is billed."""

import base64
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness2 import config
from harness2.cli import _common
from harness2.cli import claude_bridge as bridge
from harness2.core import prompts, tree
from harness2.core.proposer import Proposer
from harness2.core.spec import ChannelSpec
from harness2.substrate import claude_code as cc

# A stand-in for the `claude` CLI. Solver mode (stream-json) writes a deliverable that
# records which API variables leaked into its environment; json mode (improver / bridge)
# echoes what it was given so the caller's contract can be asserted.
FAKE_CLAUDE = r"""
import json, os, re, sys
from pathlib import Path

argv = sys.argv[1:]
if "--version" in argv:
    print("9.9.9 (Claude Code)")
    sys.exit(0)
leaked = sorted(k for k in os.environ if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")))
fmt = argv[argv.index("--output-format") + 1]
if fmt == "stream-json":
    out = Path.cwd() / "output"
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text(json.dumps({
        "leaked": leaked,
        "claude_md": (Path.cwd() / "CLAUDE.md").read_text(),
        "agents": sorted(p.name for p in (Path.cwd() / ".claude" / "agents").glob("*.md")),
        "argv": argv,
    }))
    events = [
        {"type": "system", "subtype": "init", "model": argv[argv.index("--model") + 1]},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "plan the report"},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls output"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "report.md"},
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "Done.",
         "num_turns": 2, "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]
    if os.environ.get("FAKE_CLAUDE_FAIL"):
        print(json.dumps({"type": "result", "subtype": "error_during_execution",
                          "is_error": True, "result": "usage limit reached"}))
        sys.exit(1)
    print("not json noise")
    for ev in events:
        print(json.dumps(ev))
    sys.exit(0)
stdin = sys.stdin.read()
listed = re.findall(r"^- (/\S+)$", stdin, re.M)
echo = {
    "leaked": leaked,
    "system": argv[argv.index("--system-prompt") + 1] if "--system-prompt" in argv else "",
    "tools": argv[argv.index("--tools") + 1] if "--tools" in argv else None,
    "model": argv[argv.index("--model") + 1] if "--model" in argv else "",
    "isolated": "--strict-mcp-config" in argv and "--setting-sources" in argv,
    "images_exist": [os.path.isfile(p) for p in listed],
    "stdin": stdin[:400],
}
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": json.dumps(echo), "total_cost_usd": 0.0,
                  "usage": {"input_tokens": 7, "output_tokens": 3}}))
"""


def make_fake_claude(directory: Path) -> Path:
    """Write the fake CLI as an executable running under this interpreter."""
    path = directory / "claude"
    path.write_text(f"#!{sys.executable}\n{FAKE_CLAUDE}", encoding="utf-8")
    path.chmod(0o755)
    return path


class ClaudeCodeTestCase(unittest.TestCase):
    """A temp root plus the fake CLI, wired in for this process and its children."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fake = make_fake_claude(self.root)
        for patcher in (
            patch.object(config, "CLAUDE_BIN", str(self.fake)),
            patch.dict(
                os.environ,
                {
                    "HARNESS2_CLAUDE_BIN": str(self.fake),
                    "ANTHROPIC_API_KEY": "sk-must-not-leak",
                    "CLAUDE_CODE_USE_VERTEX": "1",
                    "ANTHROPIC_FOUNDRY_API_KEY": "must-not-leak",
                    "CLAUDE_CODE_USE_SOMETHING_NEW": "1",
                },
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)


class SurfaceTests(unittest.TestCase):
    surface = cc.CLAUDE_CODE_SURFACE

    def test_valid_tree_census(self):
        harness = {
            "CLAUDE.md": "Work carefully.\n- When filling a spreadsheet, use the `xlsx-fill` skill.\n",
            ".claude/skills/xlsx-fill/SKILL.md": "---\ndescription: fill xlsx\n---\nSteps.",
            ".claude/skills/xlsx-fill/scripts/fill.py": "print('ok')\n",
            ".claude/settings.json": json.dumps(
                {
                    "hooks": {
                        "PostToolUse": [
                            {
                                "matcher": "Bash",
                                "hooks": [
                                    {"type": "command", "command": "python3 scripts/check.py"}
                                ],
                            }
                        ]
                    }
                }
            ),
            "sub_agents/reader/AGENT.md": "---\ndescription: reads big files\n---\nRead.",
            "scripts/check.py": "print('checked')\n",
        }
        census = tree.validate(self.surface, harness)
        self.assertEqual(
            (census["skills"], census["sub_agents"], census["scripts"], census["hooks"]),
            (1, 1, 1, 1),
        )

    def test_rejects_untriggered_skill_empty_claude_md_and_unknown_hook(self):
        harness = {
            "CLAUDE.md": "",
            ".claude/skills/orphan/SKILL.md": "---\ndescription: never named\n---\n",
            ".claude/settings.json": json.dumps(
                {"hooks": {"OnWhatever": [{"hooks": [{"type": "command", "command": "x"}]}]}}
            ),
        }
        with self.assertRaises(ValueError) as caught:
            tree.validate(self.surface, harness)
        message = str(caught.exception)
        self.assertIn("CLAUDE.md is empty", message)
        self.assertIn("skill 'orphan' is never named", message)
        self.assertIn("unknown event 'OnWhatever'", message)

    def test_autotrigger_writes_claude_md(self):
        harness = {
            "CLAUDE.md": "Base.",
            ".claude/skills/memo/SKILL.md": "---\ndescription: writing a memo\n---\n",
        }
        self.assertEqual(tree.autotrigger(self.surface, harness), ["memo"])
        self.assertIn("use the `memo` skill", harness["CLAUDE.md"])
        tree.validate(self.surface, harness)

    def test_prompt_surface_names_only_claude_code_paths(self):
        text = prompts._st("claude_code")
        joined = "\n".join((text.surface_md, text.inv, text.dom, text.tts_surface))
        for codex_path in ("AGENTS.md", ".codex", ".agents/skills"):
            self.assertNotIn(codex_path, joined)
        for claude_path in ("CLAUDE.md", ".claude/settings.json", ".claude/skills/"):
            self.assertIn(claude_path, joined)


class InstallAndExecTests(ClaudeCodeTestCase):
    def test_materialise_installs_subagents_as_claude_agents(self):
        source = self.root / "harness"
        tree.write_tree(
            cc.CLAUDE_CODE_SURFACE,
            {
                "CLAUDE.md": "Rules.",
                "sub_agents/reader/AGENT.md": "---\ndescription: reads\n---\nRead it.",
                "sub_agents/bare/AGENT.md": "No frontmatter here.",
                "scripts/tool.py": "print(1)\n",
            },
            source,
        )
        install = cc.ClaudeCodeSubstrate().materialise(
            source, SimpleNamespace(install_mode="claude_root"), {"file:x": "dropped", "k": "v"}
        )
        self.addCleanup(lambda: __import__("shutil").rmtree(install.staging_dir, True))
        staged = install.staging_dir
        reader = (staged / ".claude/agents/reader.md").read_text()
        self.assertTrue(reader.startswith("---\nname: reader\ndescription: reads\n---"))
        bare = (staged / ".claude/agents/bare.md").read_text()
        self.assertIn("name: bare", bare)
        self.assertIn("No frontmatter here.", bare)
        self.assertTrue((staged / "scripts/tool.py").is_file())
        self.assertFalse((staged / "sub_agents").exists())
        self.assertEqual(install.extras, {"install_mode": "claude_root", "k": "v"})
        with self.assertRaises(ValueError):
            cc.ClaudeCodeSubstrate().materialise(
                source, SimpleNamespace(install_mode="codex_root"), {}
            )

    def test_argv_is_isolated_and_env_has_no_api_credentials(self):
        argv = cc.ClaudeCodeSubstrate().exec_argv("do it", model="opus", effort="high")
        self.assertEqual(argv[:3], [str(self.fake), "-p", "do it"])
        for flag in ("--strict-mcp-config", "--verbose", "--no-session-persistence"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "project,local")
        self.assertEqual(argv[argv.index("--model") + 1], "opus")
        env = cc.subscription_env(os.environ)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("CLAUDE_CODE_USE_VERTEX", env)

    def test_preflight_runs_version(self):
        cc.ClaudeCodeSubstrate().preflight()
        with patch.object(config, "CLAUDE_BIN", str(self.root / "missing-claude")):
            with self.assertRaises(RuntimeError):
                cc.ClaudeCodeSubstrate().preflight()


class TrajectoryTests(unittest.TestCase):
    def write(self, events: list) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            handle.write("garbage line\n")
            for ev in events:
                handle.write(json.dumps(ev) + "\n")
        return Path(handle.name)

    def test_renders_shared_grammar(self):
        path = self.write(
            [
                {"type": "system", "subtype": "init", "model": "sonnet"},
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Reading."},
                            {"type": "tool_use", "name": "Bash", "input": {"command": "cat a.txt"}},
                            {"type": "tool_use", "name": "Read", "input": {"file_path": "/w/b.md"}},
                        ]
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "content": [{"type": "text", "text": "hello"}]},
                            {"type": "tool_result", "content": "no such file", "is_error": True},
                        ]
                    },
                },
                {
                    "type": "result",
                    "subtype": "error_max_turns",
                    "is_error": True,
                    "result": "",
                    "num_turns": 9,
                    "usage": {"input_tokens": 3},
                },
            ]
        )
        rendered = cc.render_events(path, obs_trunc=1000, max_chars=10_000)
        self.assertEqual(
            rendered.splitlines(),
            [
                "[HARNESS] session start | model=sonnet",
                "[ASSISTANT] Reading.",
                "[ACTION] Bash: cat a.txt",
                '[ACTION] Read: {"file_path": "/w/b.md"}',
                "[OBSERVATION] hello",
                "[OBSERVATION ERROR] no such file",
                "[HARNESS] session error_max_turns | turns=9 | tokens in=3 cache_read=None out=None",
                "[ERROR] error_max_turns",
            ],
        )
        clipped = cc.render_events(path, obs_trunc=1000, max_chars=20)
        self.assertIn("render truncation", clipped)

    def test_final_message_and_empty_stream(self):
        path = self.write(
            [{"type": "assistant", "message": {"content": [{"type": "text", "text": "A"}]}}]
        )
        self.assertEqual(cc.final_message(path), "A")
        self.assertEqual(
            cc.render_events(self.write([]), obs_trunc=10, max_chars=10),
            "(no parseable events in trajectory)",
        )


class JobBenchRolloutTests(ClaudeCodeTestCase):
    """A real JobBenchAdapter.execute on --substrate cc, through jb_claude_runner."""

    def test_execute_rolls_out_with_subscription_env_and_renders_trajectory(self):
        from harness2.benchmarks import jobbench

        task_id = "analyst/task1"
        store = self.root / "store"
        task_folder = store / task_id / "task_folder"
        task_folder.mkdir(parents=True)
        (task_folder / "TASK_INSTRUCTIONS.txt").write_text("Write a report.")
        adapter = jobbench.JobBenchAdapter("cc", store_root=store, timeout=60)
        self.assertEqual(adapter.spec.install_mode, "claude_root")
        self.assertEqual(adapter.spec.channels["proposer"].kind, "claude_cli")
        self.assertTrue(adapter.spec.allow_subagents)

        harness = self.root / "harness"
        tree.write_tree(
            adapter.substrate.surface(),
            {
                "CLAUDE.md": "Adapted rules.",
                "sub_agents/reader/AGENT.md": "---\ndescription: reads\n---\n",
            },
            harness,
        )
        out = adapter.execute(task_id, harness, "jb-cc-agg1/inv1")

        report = json.loads((out / "report.md").read_text())
        self.assertEqual(report["leaked"], [])
        self.assertEqual(report["claude_md"], "Adapted rules.")
        self.assertEqual(report["agents"], ["reader.md"])
        self.assertIn("--strict-mcp-config", report["argv"])
        rendered = adapter.trajectory(out, obs_trunc=1000)
        self.assertIn("[ACTION] Bash: ls output", rendered)
        self.assertIn("[REASONING] plan the report", rendered)
        self.assertTrue(adapter.artifacts_intact(out))
        # A resumed call adopts the marked rollout instead of re-running it.
        self.assertEqual(adapter.execute(task_id, harness, "jb-cc-agg1/inv1"), out)

    def test_infrastructure_failure_leaves_rollout_unmarked(self):
        from harness2.benchmarks import jobbench
        from harness2.core import runmark

        task_id = "analyst/task2"
        store = self.root / "store"
        (store / task_id / "task_folder").mkdir(parents=True)
        adapter = jobbench.JobBenchAdapter("cc", store_root=store, timeout=60)
        with patch.dict(os.environ, {"FAKE_CLAUDE_FAIL": "1"}):
            with self.assertRaises(RuntimeError):
                adapter.execute(task_id, None, "jb-cc-agg1/base1")
        label = "jb-cc-agg1__base1"
        self.assertFalse(any((store / task_id / "model_output").glob(f"{label}/*")))
        self.assertFalse(runmark.is_complete(store / task_id / "model_traj" / label))

    def test_base_harness_is_a_claude_code_tree(self):
        from harness2.benchmarks import jobbench

        base = jobbench.JobBenchAdapter("cc").base_harness()
        tree.validate(cc.CLAUDE_CODE_SURFACE, tree.read_tree(cc.CLAUDE_CODE_SURFACE, base))


class BridgeTests(ClaudeCodeTestCase):
    def setUp(self):
        super().setUp()
        handler = bridge.make_handler(api_key="", timeout=60, concurrency=2)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def post(self, path: str, body) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_judge_style_request_with_image(self):
        png = base64.b64encode(b"\x89PNG fake").decode()
        status, body = self.post(
            "/v1/chat/completions",
            {
                "model": "vertex/sonnet",
                "messages": [
                    {"role": "system", "content": "Return valid JSON only."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Grade this."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{png}"},
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_completion_tokens": 8192,
            },
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(body["object"], "chat.completion")
        echo = json.loads(body["choices"][0]["message"]["content"])
        self.assertEqual(echo["leaked"], [])
        self.assertEqual(echo["system"], "Return valid JSON only.")
        self.assertEqual(echo["model"], "sonnet")
        self.assertEqual(echo["tools"], "Read")
        self.assertTrue(echo["isolated"])
        self.assertEqual(echo["images_exist"], [True])
        self.assertTrue(echo["stdin"].startswith("Grade this."))
        self.assertEqual(body["usage"]["total_tokens"], 10)

    def test_text_request_disables_tools(self):
        status, body = self.post(
            "/v1/chat/completions",
            {"model": "sonnet", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body["choices"][0]["message"]["content"])["tools"], "")

    def test_malformed_bodies_get_a_reply_not_a_dropped_socket(self):
        for raw, length in ((b"\xff\xfe not utf8", None), (b"{}", "abc")):
            request = urllib.request.Request(
                self.base + "/v1/chat/completions", data=raw, method="POST"
            )
            if length is not None:
                request.add_unredirected_header("Content-Length", length)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=30)
            self.assertEqual(caught.exception.code, 400)

    def test_bad_requests(self):
        self.assertEqual(self.post("/v1/embeddings", {})[0], 404)
        self.assertEqual(
            self.post("/v1/chat/completions", {"model": "openai/gpt", "messages": []})[0], 400
        )
        self.assertEqual(self.post("/v1/chat/completions", {"model": "sonnet"})[0], 400)


class ImproverAndCliTests(ClaudeCodeTestCase):
    def test_subscription_proposer_strips_api_and_isolates(self):
        with patch.object(config, "CLAUDE_AUTH", "subscription"):
            proposer = Proposer(ChannelSpec("claude_cli", "opus"), attempts=1)
            env = proposer.env()
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("CLAUDE_CODE_USE_VERTEX", env)
            reply = proposer.call("Propose an edit.", None, "propose.inv1", system="Be brief.")
        echo = json.loads(reply.text)
        self.assertEqual((echo["leaked"], echo["model"], echo["isolated"]), ([], "opus", True))

    def test_cli_and_config_guards(self):
        with self.assertRaises(ValueError):
            _common.make_adapter("wb", "web", "cc")
        with self.assertRaises(ValueError):
            _common.make_adapter("lab", None, "cc")
        with patch.object(config, "CLAUDE_SOLVER_MODEL", "anthropic/sonnet"):
            with self.assertRaises(config.ConfigError):
                config.validate_models(substrate="cc")
        with patch.object(config, "CLAUDE_AUTH", "vertex"):
            with self.assertRaises(config.ConfigError):
                config.validate_models(substrate="cc")
        with patch.object(config, "JUDGE_MODEL", ""):
            config.validate_models(substrate="cc")
            with self.assertRaises(config.ConfigError):
                config.validate_models(substrate="cc", judge=True)
        self.assertEqual(bridge.cli_model("anthropic/claude-opus-5-5"), "claude-opus-5-5")


if __name__ == "__main__":
    unittest.main()
