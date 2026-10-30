# Local models

All local GGUF weights live under this directory. It is gitignored on purpose: weights stay out of git, only this README and `.gitkeep` placeholders are tracked. Downloads use the read-only `HF_TOKEN` from `.env`, never in URLs.

## dory1-4b / NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf

| item | value |
|------|-------|
| source repo | [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF) |
| downloaded | 2026-10-30 |
| file | `dory1-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` |
| size | 3,673,288,864 bytes (~3.42 GiB) |
| sha256 (first 16 hex) | `96080be49ffd67b9` |
| architecture | looped (latent-recurrent) normalized-GPT, dory arch in llama.cpp |
| loop structure | Input 8 layers, shared Recurrent 20 layers (re-inject input each loop), Output 8 layers |
| attention | interleaved sliding-window + full attention |
| context | up to 262144 tokens |
| quantization | Q5_K_M |
| license | NVIDIA Open Model License (card in GGUF repo: openmdw-1.1 style disposition) |
| serve keys | loop count knob at runtime (1/2/4 tested by NVIDIA, 2 recommended); last-loop KV caching keeps memory flat in loops |
| smoke verified | 2026-10-30 via loop-llama-server (image loop-llama-cpp:pr30306, llama.cpp 0.8.0-dev + PR 30306 head 1c3cdd2), 3/3 checks in tests/test_dory_smoke.py, ~44 tok/s on RTX 4070, 7.4/12 GiB VRAM |

Extra files in `dory1-4b/`: `explainability.md` (official card answers), `dory_arch.png` (official architecture diagram).

Serving: see `docker/` (loop-llama-server) and the smoke test in `tests/`. Background research in `mds/looped-research.md`.
