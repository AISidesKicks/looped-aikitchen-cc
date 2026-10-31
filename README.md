# looped-aikitchen-cc

Educational lab for looped (latent-recurrent) transformers: running, poking and understanding models that reuse their own layers to trade compute for accuracy, on real local hardware.

## Why this project is here

Running smart AI is expensive, so before renting millennia of GPU time we want to understand the loop trick hands on: what looping a transformer's shared block really buys, where it pays off, and where it does not. The trigger was NVIDIA's looped Dory1.0-4B (October 2026), released with no tech report: everything in `mds/` reconstructs the research the market already knows from older, smaller looped models, then measures the new kid the same way.

## Repo layout

| path | role |
| --- | --- |
| `docker/loop-llama-server/` | llama.cpp with Dory support (PR 30306 pinned), serving on port 8777 over GPU, all containers `loop-` prefixed |
| `datasets/loop_evals/` | deterministic synthetic loop eval generator: arithmetic, graph reachability, stack programs |
| `tests/` | python tests: GGUF magic checks, server smoke test, loop count variant patcher, loop eval runners |
| `models/gguf/` | local weights, gitignored except `models/README.md` (the model card) |
| `mds/` | deep dives, see below |
| `lab/` and `src/` | hands on experiments and node.js lab code |
| `docs/` | static GitHub Pages overview site, hand written HTML only |
| `INTENT.md` | project intent and up to date links |

## What the lab learned so far

The one sentence version: going from 1 to 2 loops buys the most accuracy while deeper loops mostly just cost speed. Stack programs are the loop sensitive kind; easy arithmetic almost never needs a loop. Combined over two independent 36 task draws, temperature 0, Dory1.0-4B Q5_K_M on an RTX 4070:

| task type | 1 loop | 2 loops | 3 loops | 4 loops |
|-----------|--------|---------|---------|---------|
| arithmetic4 | 20/24 | 21/24 | 22/24 | 23/24 |
| graph_reach | 18/24 | 22/24 | 19/24 | 20/24 |
| stack_program | 5/24 | 11/24 | 11/24 | 11/24 |
| **all** | **43/72** | **54/72** | **52/72** | **54/72** |

| variant | tok/s |
|---------|-------|
| 1 loop | 108.8 - 110.2 |
| 2 loops | 69.1 - 76.1 |
| 3 loops | 52.7 - 57.8 |
| 4 loops | 42.6 - 47.0 |

Results live in [mds/lab-report.md](mds/lab-report.md), dataset design and
our eval inventory in [mds/datasets.md](mds/datasets.md), looped model
lineage research in [mds/looped-research.md](mds/looped-research.md).

## The deep dives (mds/)

| doc | what is inside |
| --- | --- |
| [mds/looped-research.md](mds/looped-research.md) | the looped transformer lineage mapped: Universal Transformers, LoopedLAN, Huginn, Ouro, Hyperloop, Loopie, the KV cache fix family, and what each paper actually measured |
| [mds/datasets.md](mds/datasets.md) | public code, harnesses and dataset inventory from every looped paper, plus our own synthetic loop eval design and the live results |
| [mds/lab-report.md](mds/lab-report.md) | the full hands on story: from `unknown model architecture: 'dory'` to a custom CUDA build, loop count surgery on GGUF metadata, and a replicated 4 point loop curve with honest caveats |

## Quick start

```sh
docker compose -f docker/loop-llama-server/compose.yaml up -d
python tests/test_dory_smoke.py
python tests/test_loop_evals.py --cycles 1 2 4
```

Conventions: conventional commits (lowercase), `loop-` prefix for containers, `.env` stays local and gitignored, `HF_TOKEN` is read-only and never baked into anything.
