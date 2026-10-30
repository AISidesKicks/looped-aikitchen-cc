"""Deterministic generator for the loop-friendly synthetic eval set.

These are the tasks from mds/datasets.md section "Our simulated loop evals":
multi-step arithmetic, directed-graph reachability (loop-friendly iteration)
and a reverse polish stack program (LoopedLAN punchcard inspired). All
answers are normalized to a number token so scoring stays trivial.

Each line of the JSONL dataset is {"type": ..., "question": ..., "answer":
int, "meta": {...}}. Re-running with the same seed produces byte-identical
data, so results are comparable across model versions and loop counts.

Usage:
    python datasets/loop_evals/generate.py                  # 6 per type, seed 7
    python datasets/loop_evals/generate.py --n 8 --seed 11
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 7


def gen_arithmetic(rng: random.Random, idx: int) -> dict:
    """((a ? b) ? c) with two levels, plus one more step as (+ d) or (- d)."""
    a = rng.randint(6, 40)
    b = rng.randint(3, 25)
    c = rng.randint(2, 18)
    d = rng.randint(4, 30)
    op1, op2 = rng.choice(["+", "*"]), rng.choice(["+", "-"])
    part1 = a + b if op1 == "+" else a * b
    part2 = part1 + c if op2 == "+" else part1 - c
    final_op = "+"
    answer = part2 + d
    question = (
        f"Compute step by step: start with {a}, then {op1} {b}, "
        f"then {op2} {c}, then + {d}. What is the final result? "
        "Answer with the number only."
    )
    return {"type": "arithmetic4", "question": question, "answer": answer,
            "meta": {"idx": idx, "ops": [op1, op2, final_op]}}


def gen_graph(rng: random.Random, idx: int) -> dict:
    """Directed graph, edge list in shuffled order, ask reachability as 1/0."""
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    rng.shuffle(nodes)
    nodes = nodes[: rng.choice([5, 6, 7])]
    n_nodes_et = len(nodes)
    potential = [(u, v) for i, u in enumerate(nodes) for j, v in enumerate(nodes) if i != j]
    edges = rng.sample(potential, min(len(potential), n_nodes_et + rng.randint(2, 5)))
    # walk reachability from a random source
    src, dst = rng.sample(nodes, 2)
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
    seen, stack = {src}, list(adj[src])
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur])
    reachable = int(dst in seen)
    edge_txt = ", ".join(f"{u}->{v}" for u, v in edges)
    question = (
        f"A directed graph has exactly these edges: {edge_txt}. "
        f"Following arrows only, is node {dst} reachable from node {src}? "
        "Answer 1 if reachable, 0 if not reachable. Answer with the number only."
    )
    return {"type": "graph_reach", "question": question, "answer": reachable,
            "meta": {"idx": idx, "src": src, "dst": dst, "n_nodes": n_nodes_et}}


def gen_stack_program(rng: random.Random, idx: int) -> dict:
    """A reverse polish stack machine: deterministic program, one final value."""
    def value() -> int:
        return rng.randint(2, 9)
    PUSH, ADD, MUL = "push", "add", "mul"
    tokens: list[tuple[str, int]] = []
    for _ in range(rng.randint(1, 2)):
        tokens.append((PUSH, value()))
        tokens.append((PUSH, value()))
        tokens.append((rng.choice([ADD, MUL]), 0))
    tokens.append((PUSH, value()))
    tokens.append((rng.choice([ADD, MUL]), 0))
    tokens.append((PUSH, value()))
    op = rng.choice([ADD, MUL])
    tokens.append((op, 0))
    stack: list[int] = []
    for op_name, val in tokens:
        if op_name == PUSH:
            stack.append(val)
        elif op_name == ADD:
            top = stack.pop()
            stack[-1] = stack[-1] + top
        elif op_name == MUL:
            top = stack.pop()
            stack[-1] = stack[-1] * top
    answer = stack[-1]
    prog = ", ".join(f"push {v}" if n == PUSH else n for n, v in tokens)
    question = (
        f"A stack machine runs this program: {prog}. After each op it uses the "
        "top two stack values and pushes the result back. What is the single "
        "value left on the stack? Answer with the number only."
    )
    return {"type": "stack_program", "question": question, "answer": answer,
            "meta": {"idx": idx, "ops": len(tokens)}}


GENERATORS = {
    "arithmetic4": gen_arithmetic,
    "graph_reach": gen_graph,
    "stack_program": gen_stack_program,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="items per type")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(HERE / "loop_evals.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, idx = [], 0
    for tname, fn in GENERATORS.items():
        for _ in range(args.n):
            row = fn(rng, idx)
            idx += 1
            rows.append(row)

    out = Path(args.out)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"[ok] {out}: {len(rows)} tasks {by_type} (seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
