"""Score the loop_evals synthetic dataset against Dory loop-count variants.

For each loop count GGUF (1/2/4) an ephemeral container runs the questions
from datasets/loop_evals/loop_evals.jsonl at temperature 0, and we compare
extracted numeric answers against the ground truth. Results land in
scratch/loop_eval_results.json and this script exits 0 in all cases unless a
variant fails to serve.

Usage:
    python tests/test_loop_evals.py              # all cycle counts, all tasks
    python tests/test_loop_evals.py --cycles 2 4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "gguf" / "dory1-4b"
DATA = ROOT / "datasets" / "loop_evals" / "loop_evals.jsonl"
IMAGE = "loop-llama-cpp:pr30306"
CONTAINER = "loop-probe"
PORT = 8779

NUM_RE = re.compile(r"-?\d[\d,]*")


def variants_for(cycles: list[int]) -> dict[int, Path]:
    out = {}
    for n in cycles:
        if n == 4:
            out[n] = GGUF_DIR / "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
        else:
            out[n] = GGUF_DIR / f"NVIDIA-Nemotron-Labs-Dory1.0-4B-{n}loop.gguf"
        if not out[n].is_file():
            raise SystemExit(f"missing variant GGUF for {n} loops: {out[n]}")
    return out


def parse_answer(text: str) -> int | None:
    matches = NUM_RE.findall((text or "").replace(",", " "))
    return int(matches[-1]) if matches else None


def stop() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, text=True)


def start(model_path: Path, ctx: int) -> None:
    stop()
    subprocess.run([
        "docker", "run", "-d", "--rm", "--name", CONTAINER,
        "--gpus", "all",
        "-p", f"127.0.0.1:{PORT}:8080",
        "-v", f"{ROOT / 'models'}:/models:ro",
        IMAGE,
        "-m", f"/models/gguf/dory1-4b/{model_path.name}",
        "--host", "0.0.0.0", "--port", "8080",
        "--ctx-size", str(ctx),
        "-ngl", "99",
    ], check=True, capture_output=True, text=True)


def wait_health(timeout_s: float = 180.0) -> bool:
    url = f"http://127.0.0.1:{PORT}/health"
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            time.sleep(2.0)
    return False


def ask(question: str, max_tokens: int = 512) -> dict:
    payload = json.dumps({
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def eval_variant(cycles: int, tasks: list[dict], ctx: int, max_tokens: int) -> dict:
    start(variants_for([cycles])[cycles], ctx)
    okwait = wait_health()
    if not okwait:
        return {"cycles": cycles, "error": "server never became healthy"}
    per_type: dict[str, dict] = {}
    for task in tasks:
        body = ask(task["question"], max_tokens)
        msg = body["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        got = parse_answer(text)
        ok = got == task["answer"]
        agg = per_type.setdefault(task["type"], {"correct": 0, "total": 0, "tokens": []})
        agg["correct"] += int(ok)
        agg["total"] += 1
        agg["tokens"].append(body.get("usage", {}).get("completion_tokens"))
    timings = body.get("timings", {})
    stop()
    return {"cycles": cycles, "per_type": per_type,
            "tok_per_s": round(timings.get("predicted_per_second") or 0.0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="only first N tasks (0 = all)")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[eval] {len(tasks)} tasks x cycle counts {args.cycles}")

    subprocess.run(["docker", "compose", "-f",
                    str(ROOT / "docker" / "loop-llama-server" / "compose.yaml"),
                    "stop"], capture_output=True, text=True)
    results = []
    try:
        for n in args.cycles:
            print(f"[eval] {n}-loop variant starting")
            row = eval_variant(n, tasks, args.ctx, args.max_tokens)
            results.append(row)
            for ttype, agg in row.get("per_type", {}).items():
                print(f"    {n} loops | {ttype}: {agg['correct']}/{agg['total']}")
            print(f"    {n} loops | ~{row.get('tok_per_s')} tok/s")
    finally:
        stop()
        subprocess.run(["docker", "compose", "-f",
                        str(ROOT / "docker" / "loop-llama-server" / "compose.yaml"),
                        "up", "-d"], capture_output=True, text=True)

    (ROOT / "scratch").mkdir(exist_ok=True)
    out = ROOT / "scratch" / "loop_eval_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"[done] results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
