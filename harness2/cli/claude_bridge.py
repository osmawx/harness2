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

"""Serve an OpenAI-compatible /v1/chat/completions endpoint backed by `claude -p`.

Lets a benchmark judge that speaks the OpenAI API (JobBench's upstream judge) grade on the
Claude Code login instead of an API key. Each request is one tool-less `claude -p` session
(only Read is enabled, and only when images are attached, so the judge can look at them).
Standard library only; binds to loopback by default.

    harness2-claude-bridge --port 8765
    # then, in .env:
    HARNESS2_JUDGE_BASE_URL=http://127.0.0.1:8765/v1
    HARNESS2_JUDGE_MODEL=sonnet
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import config
from ..core.proposer import SYSTEM_ARG_MAX
from ..substrate.claude_code import ISOLATION_FLAGS, claude_binary, subscription_env

DEFAULT_PORT = 8765
DEFAULT_TIMEOUT_S = 900
MAX_BODY_BYTES = 256 * 1024 * 1024
_DATA_URL = re.compile(r"^data:image/(?P<ext>[a-zA-Z0-9.+-]+);base64,(?P<data>.*)$", re.S)
_PROVIDER_PREFIXES = ("anthropic/", "vertex/", "vertex_ai/", "claude_cli/")


class BridgeError(RuntimeError):
    """A request the bridge cannot serve; carries the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def cli_model(requested: str) -> str:
    """An OpenAI-side model name as a Claude CLI --model value (provider prefix dropped)."""
    model = str(requested or "").strip()
    for prefix in _PROVIDER_PREFIXES:
        if model.startswith(prefix):
            model = model[len(prefix) :]
    if not model or "/" in model:
        raise BridgeError(f"model {requested!r} is not a Claude CLI model id", status=400)
    return model


def _save_image(url: str, image_dir: Path, index: int) -> str:
    """Write a data: URL to `image_dir` and return its path; pass other URLs through."""
    m = _DATA_URL.match(url)
    if m is None:
        return url
    ext = m.group("ext").lower().replace("jpeg", "jpg").split("+", 1)[0]
    try:
        payload = base64.b64decode(m.group("data"), validate=False)
    except (binascii.Error, ValueError) as exc:
        raise BridgeError(f"image {index} is not valid base64: {exc}", status=400) from exc
    path = image_dir / f"image_{index:02d}.{ext}"
    path.write_bytes(payload)
    return str(path)


def _content_text(content, image_dir: Path, images: list[str]) -> str:
    """Flatten OpenAI message content (string or parts) to text; images become file paths."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            parts.append(str(part.get("text") or ""))
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            location = _save_image(str(url), image_dir, len(images) + 1)
            images.append(location)
            parts.append(f"[image attached: {location}]")
    return "\n".join(parts)


def build_prompt(messages: list, image_dir: Path) -> tuple[str, str, list[str]]:
    """(system, prompt, image paths) for one chat request.

    A single user turn is passed through verbatim; a longer conversation is rendered as a
    role-labelled transcript, since `claude -p` takes one prompt.
    """
    if not isinstance(messages, list) or not messages:
        raise BridgeError("messages must be a non-empty list", status=400)
    images: list[str] = []
    system, turns = [], []
    for msg in messages:
        if not isinstance(msg, dict):
            raise BridgeError("every message must be an object", status=400)
        text = _content_text(msg.get("content"), image_dir, images)
        role = str(msg.get("role") or "user")
        if role in ("system", "developer"):
            system.append(text)
        else:
            turns.append((role, text))
    if len(turns) == 1 and turns[0][0] == "user":
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"## {role.upper()}\n{text}" for role, text in turns)
        prompt += "\n\n## ASSISTANT\n(continue as the assistant; reply with your message only)"
    if images:
        files = [p for p in images if not p.startswith(("http://", "https://"))]
        if files:
            prompt += (
                "\n\nThe attached images are saved as files. Open EVERY one of them with the "
                "Read tool before you answer:\n" + "\n".join(f"- {p}" for p in files)
            )
    return "\n\n".join(s for s in system if s.strip()), prompt, images


def run_claude(system: str, prompt: str, model: str, *, cwd: Path, with_read: bool, timeout: int):
    """One `claude -p` session; returns (text, usage dict). Raises BridgeError on failure."""
    argv = [
        claude_binary(),
        "-p",
        "--output-format",
        "json",
        "--model",
        model,
        "--no-session-persistence",
        *ISOLATION_FLAGS,
        "--tools",
        "Read" if with_read else "",
    ]
    stdin = prompt
    if system:
        if len(system) > SYSTEM_ARG_MAX:
            stdin = f"# SYSTEM INSTRUCTIONS\n\n{system}\n\n# END\n\n{prompt}"
        else:
            argv += ["--system-prompt", system]
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            cwd=cwd,
            env=subscription_env(os.environ),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise BridgeError(f"claude -p timed out after {timeout}s", status=504) from exc
    except OSError as exc:
        raise BridgeError(f"claude -p could not start: {exc}") from exc
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    except (json.JSONDecodeError, IndexError):
        payload = {}
    text = str(payload.get("result") or "")
    if proc.returncode != 0 or payload.get("is_error") or not text.strip():
        detail = text or (proc.stderr or "").strip()[-400:] or f"rc={proc.returncode}"
        raise BridgeError(f"claude -p failed: {detail}")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return text, usage


def chat_completion(request: dict, *, timeout: int) -> dict:
    """Serve one OpenAI chat.completions request body; returns the response body."""
    model = cli_model(request.get("model", ""))
    work = Path(tempfile.mkdtemp(prefix="harness2-bridge-"))
    try:
        system, prompt, images = build_prompt(request.get("messages"), work)
        text, usage = run_claude(
            system, prompt, model, cwd=work, with_read=bool(images), timeout=timeout
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    prompt_tokens = int(usage.get("input_tokens") or 0) + int(
        usage.get("cache_read_input_tokens") or 0
    )
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_handler(*, api_key: str, timeout: int, concurrency: int):
    """A request handler class closed over the bridge's settings."""
    gate = threading.BoundedSemaphore(max(1, concurrency))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            # An error may leave an unread body on the socket; never reuse the connection.
            self.close_connection = True
            self._reply(status, {"error": {"message": message, "type": "bridge_error"}})

        def _client_gone(self) -> bool:
            """True when the client closed its socket while the request queued (it timed
            out and will retry), so the queued session would only burn quota."""
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
            except OSError:
                return True

        def do_GET(self) -> None:  # noqa: N802 -- http.server's method naming
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._reply(200, {"object": "list", "data": []})
            else:
                self._error(404, f"no route {self.path}")

        def do_POST(self) -> None:  # noqa: N802 -- http.server's method naming
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._error(404, f"no route {self.path}")
                return
            if api_key and self.headers.get("Authorization", "") != f"Bearer {api_key}":
                self._error(401, "bad bridge key")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                self._error(413 if length > 0 else 400, "missing or oversized request body")
                return
            try:
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise BridgeError("request body must be a JSON object", status=400)
                with gate:
                    if self._client_gone():
                        self.close_connection = True
                        return
                    body = chat_completion(request, timeout=timeout)
            except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError alike
                self._error(400, f"request body is not JSON: {exc}")
                return
            except BridgeError as exc:
                self.log_error("%s", exc)
                self._error(exc.status, str(exc))
                return
            except Exception as exc:  # a reply beats a dropped socket the client retries blindly
                self.log_error("unhandled: %r", exc)
                self._error(500, f"bridge failure: {type(exc).__name__}: {exc}")
                return
            self._reply(200, body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="harness2-claude-bridge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="seconds per request")
    p.add_argument(
        "--concurrency",
        type=int,
        default=config.CLAUDE_BRIDGE_CONCURRENCY,
        help="claude -p sessions run at once; the rest queue",
    )
    args = p.parse_args(argv)
    api_key = os.environ.get("HARNESS2_CLAUDE_BRIDGE_KEY", "")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        p.error("set HARNESS2_CLAUDE_BRIDGE_KEY before binding beyond loopback")
    handler = make_handler(api_key=api_key, timeout=args.timeout, concurrency=args.concurrency)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"claude bridge on http://{args.host}:{args.port}/v1 "
        f"(concurrency {args.concurrency}, {claude_binary()})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
