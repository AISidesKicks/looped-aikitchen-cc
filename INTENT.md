# INTENT

## What this lab is about

NVIDIA released its looped model Dory1.0-4B in October 2026 with no tech report. This lab rebuilds the research the market already knows from older, smaller looped models, runs the new kid locally in Docker, and measures loop counts on real hardware - so readers can decide which pockets deserve this kind of compute before renting GPU time for it.

The core question: what does running a transformer's shared block two, three, four times actually buy, where does it pay off, and where does it just burn tokens per second?

## Plan (item by item)

1. **Lineage research** - map the looped model family (Universal Transformers, LoopedLAN, Huginn, Ouro, Hyperloop, Loopie) with what each actually measured. Done: `mds/looped-research.md`.
2. **Hands-on lab** - download Dory, serve it locally, make loop count a real experiment, evaluate it. Done: docker serving, loop variants, the 36 task synthetic eval, `mds/lab-report.md`.
3. **Documentation** - keep `README.md`, `mds/`, and the static `docs/` site in sync with the measured facts, honestly including negative results.
4. **Next candidates** (in the order we plan to pick them up):
   - agent shaped eval where NVIDIA claims the loop gains (SWE-bench, Terminal-Bench style tool use) - needs a small harness in `src/`   - cross backend check: MXFP8 or BF16 through vLLM with the same loop metadata, to see whether the loop-2 shoulder moves
   - Huginn intermediate checkpoint sweep: are loop gains the same shape as extra training?
   - GUI: small chat surface over the local endpoint (build output to `gui/`, never `docs/`)

## Standing decisions

- Everything dynamic runs in Docker, all resources `loop-` prefixed; nothing dynamic in `docs/`.
- The loop knob is GGUF metadata surgery (`dory.num_loops`), not a serving path flag; results are labeled as test scale.
- `.env` stays local and gitignored; `HF_TOKEN` is read only and never baked into URLs, images or datasets.
- Conventional commits, lowercase, plain hyphens in docs.

## Reference links

- Dory GGUF on the hub: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF
- BF16 upstream weights: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
- llama.cpp dory support (still open): https://github.com/ggml-org/llama.cpp/pull/30306
- Background reading on looped depth sharing: https://sebastianraschka.com/llm-architecture-gallery/looped-depth-sharing/
