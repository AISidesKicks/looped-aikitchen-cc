# looped-aikitchen-cc

Educational lab for looped (latent-recurrent) transformers: running, poking and understanding models that reuse their own layers to trade compute for accuracy, on real local hardware.

## Why this project is here

Running smart AI is expensive, so before renting millennia of GPU time we want to understand the loop trick hands on: what looping a transformer's shared block really buys, where it pays off, and where it does not. The trigger was NVIDIA's looped Dory1.0-4B (October 2026), released with no tech report: everything in `mds/` reconstructs the research the market already knows from older, smaller looped models, then measures the new kid the same way.

## Where things live

- `mds/` - deep dives: `mds/looped-research.md` (looped model lineage, old approaches), `mds/datasets.md` (code, tests and dataset inventory + our eval results), `mds/lab-report.md` (the full hands-on report with curves)
- `datasets/loop_evals/` - deterministic synthetic loop-eval generator (arithmetic, graph reachability, stack programs)
- `lab/` and `src/` - hands-on experiments (node.js code in `src/`)
- `models/gguf/` - local GGUF weights, gitignored; `models/README.md` is the model card
- `tests/` - Python tests: GGUF magic checks, server smoke tests, loop-count variant patcher, loop eval runners
- `docker/loop-llama-server/` - llama.cpp with Dory support (PR 30306 pinned) serving on http://localhost:8777 over GPU
- `docs/` - static GitHub Pages overview site (hand written HTML only)
- `INTENT.md` - project intent and up to date links

## Quick start

```sh
docker compose -f docker/loop-llama-server/compose.yaml up -d
python tests/test_dory_smoke.py
python tests/test_loop_evals.py --cycles 1 2 4
```

Conventions: conventional commits (lowercase), `loop-` prefix for containers, `.env` stays local and gitignored, `HF_TOKEN` is read-only and never baked into anything.
