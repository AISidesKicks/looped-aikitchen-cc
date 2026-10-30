"""Smoke test for the looped-model trial (Dory1.0-4B via loop-llama-server).

Run from the project root:

    python tests/test_dory_smoke.py            # stdlib only, exits 0 on success
    python -m pytest tests/test_dory_smoke.py  # also pytest-compatible

What it checks, in order:
    1. the local GGUF file exists and starts with the GGUF magic bytes
    2. the loop-llama-server container answers on the health endpoint
    3. a tiny chat completion returns a token-bearing message
    4. usage shows us the token accounting we track in mds/datasets.md

The server config lives in docker/loop-llama-server/compose.yaml and can be
overridden with env vars LAB_BASE_URL, LAB_SERVER_HOST, LAB_SERVER_PORT.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "models" / "gguf" / "dory1-4b" / "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"

SERVER_HOST = os.environ.get("LAB_SERVER_HOST", "localhost")
SERVER_PORT = int(os.environ.get("LAB_SERVER_PORT", "8777"))
BASE_URL = os.environ.get("LAB_BASE_URL", f"http://{SERVER_HOST}:{SERVER_PORT}")

GGUF_MAGIC = b"GGUF"
MODEL_TAG = "NVIDIA-Nemotron-Labs-Dory1.0-4B"
TEST_RADIUS = 60.0  # seconds, 12GB GPU should answer a one-liner fast


def http_json(path: str, payload: dict | None = None, timeout: float = TEST_RADIUS) -> tuple[int, dict]:
    """POST/GET one JSON request, return (status, body)."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as err:  # keep server error bodies readable
        body = err.read().decode(errors="replace")[:2000]
        raise AssertionError(f"{path} -> HTTP {err.code}: {body}") from err


def test_gguf_local_file() -> None:
    assert GGUF.is_file(), f"missing model file: {GGUF}"
    assert GGUF.stat().st_size >= 1024, f"model file suspiciously small: {GGUF.stat().st_size}"
    with GGUF.open("rb") as fh:
        magic = fh.read(4)
    assert magic == GGUF_MAGIC, f"not a GGUF file (magic={magic!r})"


def test_server_health() -> None:
    status, body = http_json("/health", timeout=TEST_RADIUS)
    assert status == 200, f"health returned {status}: {body}"


def test_chat_completion_roundtrip() -> None:
    payload = {
        "messages": [{"role": "user", "content": "What is 17 * 12? Answer with the number only."}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    status, body = http_json("/v1/chat/completions", payload)
    assert status == 200
    choice = body.get("choices") or []
    assert choice, f"no choices in response: {body}"
    text = choice[0].get("message", {}).get("content", "")
    assert text.strip(), f"empty completion: {body}"
    # the question has answer 204; accept it hiding anywhere in the reply
    assert "204" in text, f"expected 204 in completion, got: {text}"
    usage = body.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens"):
        assert key in usage, f"usage missing {key}: {usage}"
    print(f"[smoke] completion: {text.strip()!r} usage: {usage}")


def main() -> int:
    checks = [test_gguf_local_file, test_server_health, test_chat_completion_roundtrip]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"[ok] {check.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner
            failures += 1
            print(f"[fail] {check.__name__}: {exc}", file=sys.stderr)
    if failures:
        print(f"[summary] {failures}/{len(checks)} checks failed", file=sys.stderr)
        return 1
    print("[summary] all smoke checks passed for", MODEL_TAG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
