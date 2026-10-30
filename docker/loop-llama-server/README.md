# loop-llama-server

Runs the local looped model (Dory1.0-4B GGUF) behind an OpenAI-compatible HTTP endpoint on the NVIDIA GPU.

## Why our own image

The official llama.cpp release images (v0.8.0, built 2026-10-28) do not know the `dory` GGUF architecture yet. Dory support is PR [ggml-org/llama.cpp 30306](https://github.com/ggml-org/llama.cpp/pull/30306), still open at the time of writing. So `Dockerfile` builds a pinned image `loop-llama-cpp:pr30306` from the PR head.

Once the PR merges upstream and a release includes it:

1. delete this Dockerfile,
2. set `image:` in `compose.yaml` back to `ghcr.io/ggml-org/llama.cpp:server-cuda`,
3. `docker compose pull && docker compose up -d`.

## Loop count

In this PR the loop count is baked into GGUF metadata as `n_recurrent_loops` (max 4), set at conversion time, no server CLI knob reads it. The downloaded GGUF carries 4.

## Usage

```sh
docker compose -f docker/loop-llama-server/compose.yaml up -d
python tests/test_dory_smoke.py     # from repo root
# endpoint: http://localhost:8777/v1 (OpenAI compatible), /health for checks
```

The model volume is mounted read-only from `../../models`. Stop the service with `docker compose -f docker/loop-llama-server/compose.yaml down` - never touch anything outside the `loop-` prefixed resources.
