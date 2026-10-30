"""Loop-count probe for Dory1.0-4B.

Starts an ephemeral loop-llama-server docker container per loop-count GGUF
variant (the compose service is paused while the GPU is borrowed), asks a
fixed set of multi-step arithmetic questions at temperature 0, and records
correctness, output tokens and throughput per loop count. This is the cheap
home-grown proxy for what NVIDIA's internal loop evals measure.

Prereqs:
    python tests/make_dory_loop_variants.py   # 1 and 2 loop GGUF copies exist
    image loop-llama-cpp:pr30306 built (see docker/)

Usage:
    python tests/test_dory_loop_probe.py            # probes 1, 2, 4 loops
    python tests/test_dory_loop_probe.py --cycles 2 # single variant

Note the compose service runs the full 4-loop GGUF permanently; the probe
starts and removes its own container (name: loop-probe) per variant.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "gguf" / "dory1-4b"
IMAGE = "loop-llama-cpp:pr30306"
CONTAINER = "loop-probe"
PORT = 8779  # compose serves 8777 on the host; probes use 8779

QUESTIONS = [
    ("What is 17 * 12? Answer with the number only.", "204"),
    ("A bakery makes 12 trays of 8 croissants each. 19 croissants are sold. "
     "How many croissants remain? Answer with the number only.", "77"),
    ("A shop had 45 apples. It sold 13, then received 20 more, then sold 7. "
     "How many apples are left? Answer with the number only.", "45"),
]


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


def start(mode_path: Path, ctx: int) -> None:
    stop()
    cmd = [
        "docker", "run", "-d", "--rm", "--name", CONTAINER,
        "--gpus", "all",
        "-p", f"127.0.0.1:{PORT}:8080",
        "-v", f"{ROOT / 'models'}:/models:ro",
        IMAGE,
        "-m", f"/models/gguf/dory1-4b/{mode_path.name}",
        "--host", "0.0.0.0", "--port", "8080",
        "--ctx-size", str(ctx),
        "-ngl", "99",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def stop() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, text=True)


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


def ask(question: str, max_tokens: int = 256) -> dict:
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


def probe(cycles: list[int], ctx: int, quiet: bool = False) -> int:
    if not quiet:
        print("[setup] pausing compose service (loop-llama-server) to free the GPU")
        subprocess.run(["docker", "compose", "-f",
                        str(ROOT / "docker" / "loop-llama-server" / "compose.yaml"),
                        "stop"], capture_output=True, text=True)
    variants = variants_for(cycles)
    results = []
    exit_code = 0
    try:
        for n, path in sorted(variants.items()):
            print(f"[probe] starting {n}-loop variant")
            start(path, ctx)
            if not wait_health():
                print(f"[fail] {n} loops: server never became healthy")
                exit_code = 1
                continue
            row = {"cycles": n, "answers": [], "tp": None}
            for question, expect in QUESTIONS:
                body = ask(question)
                choice = body["choices"][0]["message"]
                text = (choice.get("content") or choice.get("reasoning_content") or "").strip()
                hit = expect in text
                row["answers"].append({"expect": expect, "got": text[:60], "correct": hit,
                                       "completion_tokens": body.get("usage", {}).get("completion_tokens")})
            timings = body.get("timings", {})
            row["tok_per_s"] = round(timings.get("predicted_per_second") or 0.0, 1)
            results.append(row)
            mark = " ".join("ok" if a["correct"] else "FAIL" for a in row["answers"])
            print(f"[probe] {n} loops: answers {mark} | ~{row['tok_per_s']} tok/s")
    finally:
        stop()
        if not quiet:
            print("[setup] restarting compose service")
            subprocess.run(["docker", "compose", "-f",
                            str(ROOT / "docker" / "loop-llama-server" / "compose.yaml"),
                            "up", "-d"], capture_output=True, text=True)
    summary = {r["cycles"]: r for r in results}
    (ROOT / "scratch").mkdir(exist_ok=True)
    out_file = ROOT / "scratch" / "loop_probe_results.json"
    out_file.write_text(json.dumps(summary, indent=1))
    print(f"[done] results in {out_file}")
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ctx", type=int, default=2048)
    args = ap.parse_args()
    return probe(args.cycles, args.ctx)


if __name__ == "__main__":
    sys.exit(main())
