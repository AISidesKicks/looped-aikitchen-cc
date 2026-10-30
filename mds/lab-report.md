# Lab report: Dory1.0-4B under real hardware, first full loop-accounting pass

Status: first complete report of the lab (2026-10-30). Everything that follows was run on this machine, and every artifact from scratch to score is committed under this repo.

## TL;DR

- We got NVIDIA's new looped model Dory1.0-4B running locally in Docker from llama.cpp PR 30306 (the official release image does not know the architecture yet), at about 44 tok/s on an RTX 4070, with a passing smoke test.
- We made loop count a real experiment by patching one GGUF metadata field (`dory.num_loops`, the only knob the PR reads) and cloned 1/2/3-loop variants of the official 4-loop GGUF.
- A deterministic 36-task synthetic eval (multi-step arithmetic, graph reachability, stack programs) was scored twice on two independent data draws, both economical and honest: the loop-2 shoulder is real. Going from 1 to 2 loops lifts accuracy by about 15 percentage points (43/72 to 54/72 combined) while throughput pays roughly proportional cost (110 tok/s drops to 69). Beyond two loops accuracy stays near the plateau and throughput keeps dropping.
- Stack programs are exactly the loop-sensitive category: 5/24 at one loop versus 11/24 at two loops combined over both draws. Easy arithmetic barely moves at all. Both are the signatures the old looped-model research predicts.

## The journey, milestone by milestone

| milestone | artifact |
|-----------|----------|
| Understand what looped transformers are, find the old models with real research | `mds/looped-research.md` (committed 5d05634) |
| Inventory public code, tests, datasets to simulate NVIDIA's closed loop evals | `mds/datasets.md` (committed 4a8e7d9) |
| Download the model + verify gitignore + model card | `models/gguf/dory1-4b/`, `models/README.md` (c2e301c) |
| Test official llama.cpp release image against dory GGUF | verdict: `unknown model architecture: 'dory'` (v0.8.0, built 2026-10-28) |
| Build our own image from PR 30306 head | `docker/loop-llama-server/Dockerfile`, image `loop-llama-cpp:pr30306` (five fix commits: 98ccf90, 9152455, 50665fb, 49ab7cc, 2ee4593, e81e370) |
| Server smoke test via OpenAI-compatible endpoint | `tests/test_dory_smoke.py`, 3/3 checks (2beaab0) |
| Reverse engineer the loop knob from the GGUF | `tests/make_dory_loop_variants.py` (63c6a21) |
| 18-task warm-up eval across 1/2/4 loops | first loop signature observed (fe57c1b) |
| 36-task main eval on two independent draws, 1/2/3/4 loops | `tests/test_loop_evals.py` (fe57c1b, 2482c7f), results in `scratch/` |

## Fixed setup this report ran on

| item | value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4070, 12 GiB (sm_89), used ~7.4 GiB while serving |
| host | pixi `loop` env, python 3.12, Docker Engine 29.9.2 |
| model | nvidia NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF (Q5_K_M, 3.67 GiB, sha256 starts `96080be4`), arch: looped normalized-GPT (nGPT), Input 8 / Recurrent 20 (re-inject per loop) / Output 8 layers, hybrid sliding-window + full attention, 256K context limit, NVFP4 native pretrain |
| server | llama.cpp 0.8.0-dev (build 1, commit 1c3cdd2 = PR 30306 head, branch still open upstream), CUDA build, `GGML_CUDA_NO_VMM=ON`, container `loop-llama-server` on port 8777 |
| secrets | read-only HF token from `.env` inside process env only, never in URLs or images (verified).

## The loop knob we had to build

The PR reads loop count from GGUF metadata only (`LLM_KV_NUM_LOOPS`, the `dory.num_loops` uint32), no CLI flag. Our patcher (`tests/make_dory_loop_variants.py`) parses the exact on-disk layout (v3 header with no alignment padding; verified by scanning all 44 metadata KVs), records the byte offset of `dory.num_loops` (value 4 in the official file), copies the file, and rewrites just that field. Result: cheap variant GGUFs, same weights. One nuance captured from metadata: `dory.recurrent_kv_cache_mode` is `per_loop` in this GGUF, so Dory's advertised last-loop KV caching memory trick was not leverage in these runs.

## Evaluation method

- dataset: `datasets/loop_evals/generate.py` deterministic with seed, 12 arithmetic4 + 12 graph_reach + 12 stack_program, answers normalized to a single integer; a compiled graph/stack-trace also does quick sanity that the expected answers are exact (see examples in the repo).
- runner: `tests/test_loop_evals.py` starts an ephemeral loop-llama-server docker container per loop-count variant with `--ctx-size 2048 -ngl 99`, waits on /health, asks at temperature 0, max 512 new tokens, extracts the last integer from the answer, and returns the compose service to port 8777 afterward.
- two independent dataset draws scored: seed 7 (main) and seed 11 (replication).

## Results

### Seed 7 (main draw)

| task type | 1 loop | 2 loops | 3 loops | 4 loops |
|-----------|--------|---------|---------|---------|
| arithmetic4 | 10/12 | 11/12 | 11/12 | 11/12 |
| graph_reach | 8/12 | 11/12 | 9/12 | 10/12 |
| stack_program | 2/12 | 6/12 | 7/12 | 5/12 |
| **all** | **20/36** | **28/36** | **27/36** | **26/36** |

### Seed 11 (replication draw)

| task type | 1 loop | 2 loops | 3 loops | 4 loops |
|-----------|--------|---------|---------|---------|
| arithmetic4 | 10/12 | 10/12 | 11/12 | 12/12 |
| graph_reach | 10/12 | 11/12 | 10/12 | 10/12 |
| stack_program | 3/12 | 5/12 | 4/12 | 6/12 |
| **all** | **23/36** | **26/36** | **25/36** | **28/36** |

### Combined across both draws (72 tasks per loop count)

| task type | 1 loop | 2 loops | 3 loops | 4 loops |
|-----------|--------|---------|---------|---------|
| arithmetic4 | 20/24 | 21/24 | 22/24 | 23/24 |
| graph_reach | 18/24 | 22/24 | 19/24 | 20/24 |
| stack_program | 5/24 | 11/24 | 11/24 | 11/24 |
| **all** | **43/72** | **54/72** | **52/72** | **54/72** |

### Throughput cost per loop count (same GPU, temp 0)

| variant | seed 7 tok/s | seed 11 tok/s |
|---------|--------------|---------------|
| 1 loop | 108.8 | 110.2 |
| 2 loops | 69.1 | 76.1 |
| 3 loops | 52.7 | 57.8 |
| 4 loops | 42.6 | 47.0 |

## Interpretation

- The combined table shows the shape the Dory card and the old papers agree on: most of the accuracy gain is in the first doubling (43 to 54 of 72), with 3 and 4 loops roughly flat to slightly uneven afterwards (52/72, 54/72). The per-type wins are all concentrated in depth-sensitive tasks.
- stack_program is the star: from 1 loop (5/24 over both draws) to 2 loops (11/24) is a huge jump for the algorithmic category, then flat. Graph reachability peaks at 2 loops in the combined data (22/24) then drops a bit, matching the "loop amount is task dependent" behavior reported for other looped models. Arithmetic moves slowest, exactly as a task that is solvable in one shallow pass should.
- The GPU cost is linear and real: going 1 to 4 loops costs about 2.4x in tok/s for about 11 percentage points in accuracy in these runs. Within these tiny evals that trade is only worth turning up for structured tasks; measuring on your own workload before pinning a loop count is the only honest pattern (NVIDIA says the same thing).
- Our numbers qualify NVIDIA's own benchmark claims: their loop gains cluster in agentic and coding categories (SWE-bench, Terminal-Bench), which we deliberately did not try, since our environment has no agent harness yet. Our synthetic algorithmic set reproduces the shape but not absolute scores.

## Limitations

- 36 or 72 answers per cell: per-type tables of 12 carry real noise (seed 11 notably showed a non-monotone 3-loop dip in stack_program), so treat single cells as evidence, not law; the combined 72-answer tables are the stable part.
- Temperature 0, greedy only; we did not sweep sampling settings.
- Answer scoring extracts the last integer, so a reasoning answer that gets cut off early counts as wrong, even if the model was nearly right: we accept this local bias, it is the same for all loop counts.
- Loop metadata patching is a test-scale knob, not the serving-path knob NVIDIA benchmarks through vLLM/llama.cpp runtime configs; weights identical, but flags such as last-loop KV caching were not exercised.
- One GPU, one model, one build; no cross-quant or cross-backend comparison (MXFP8 or BF16 through vLLM is a natural next baseline).

## Artifacts and repro instructions

- model + card: `models/gguf/dory1-4b/`, `models/README.md`
- docker: `docker/loop-llama-server/{Dockerfile,compose.yaml,README.md}`
- smoke: `tests/test_dory_smoke.py`
- loop variants: `tests/make_dory_loop_variants.py`
- eval: `datasets/loop_evals/generate.py`, `tests/test_loop_evals.py`
- results: `scratch/loop_eval_results_seed7.json`, `scratch/loop_eval_results_seed11.json` (scratch is gitignored)
- background reading: `mds/looped-research.md`, `mds/datasets.md`

Repro from a clean clone:

```sh
# in the pixi loop shell, HF_TOKEN in .env
python tests/make_dory_loop_variants.py --cycles 1 2 3
docker build -t loop-llama-cpp:pr30306 docker/loop-llama-server
docker compose -f docker/loop-llama-server/compose.yaml up -d
python tests/test_dory_smoke.py
python tests/test_loop_evals.py --cycles 1 2 3 4
```

## Next steps

- Agent-shaped eval (the category where NVIDIA claims loop gains): small tool-use harness in `src/`, terminal-style tasks scored programmatically.
- swap the 4070 for a larger GPU or serve through vLLM with the same variant metadata to see whether the 2-loop shoulder moves.
- checkpoint sweep of Huginn intermediate states (8 public checkpoints) once we finish understanding our own loop data: are the gains of loops really the same shape as extra training?
- GUI: a small chat surface over the same endpoint (build output goes to `gui/`, never `docs/`).
