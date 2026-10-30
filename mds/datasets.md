# Looped models: code, tests and datasets inventory

Goal: NVIDIA evaluates looped models on internal benchmarks we do not have. This inventory collects the public code, harnesses and datasets from the older looped research so we can simulate equivalent evals ourselves. Everything here is public and Apache/MIT/or permissively licensed unless noted.

Related: `mds/looped-research.md` for the research background, `datasets/` for what we eventually host locally.

## Training and modeling code

### Huginn-0125 (the old 3.5B looped model with a full recipe)

Repo: [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining) (Apache-2.0). This is the jackpot for us:

- full preprocessing and training scripts (they trained 3.5B on 4096 GPUs on Frontier, and publicly say the run is repeatable),
- `recpre/raven_modeling_minimal.py` - HF-compatible modeling file (trust_remote_code),
- `evaluate_raven/` - inference and testing scripts,
- data translation and tokenizer scripts in `scripts/`.

Datasets: the whole 800B-token training mixture is public on Hugging Face. Appendix C of the paper has the full table with sources and licenses. The model card states scripts can repeat the entire preprocessing and training run. Intermediate checkpoints (8 of them) are public too - handy for studying how looping matures over training.

### LoopedLAN (programmable computers, ICML 2023, arXiv 2301.13196)

Official implementation: [jysohn1108/Looped-Transformer](https://github.com/jysohn1108/Looped-Transformer). Emulates a calculator, linear algebra ops and backprop as in-loop programs. Community reimplementations exist (e.g. `NarendraPatwardhan/vm-in-transformer`, related WASM-interpreter-in-transformer experiments).

### RASP-L looped learning (ICLR 2024, arXiv 2311.12424)

Code: [Leiay/looped_transformer](https://github.com/Leiay/looped_transformer). Trains looped models on algorithmic tasks (addition, parity, graph reachability style tasks). The task generators in this repo are a direct template for our synthetic loop evals.

### Length generalization (ICLR 2025, arXiv 2409.15647)

Code: [UW-Madison-Lee-Lab/looped-tf](https://github.com/UW-Madison-Lee-Lab/looped-tf).

### Ouro-1.4B / Ouro-2.6B (ByteDance, arXiv 2510.25741)

Modeling ships with the HF repos: `modeling_ouro.py` + `configuration_ouro.py` on `ByteDance/Ouro-1.4B` (trust_remote_code). Knobs in config: `total_ut_steps` (default 4), `early_exit_threshold` (1.0). The implementation computes all configured passes and then selects the representation, so loop count affordability comes from `total_ut_steps`, not the threshold, in the released code. vLLM serving per project page [ouro-llm.github.io](https://ouro-llm.github.io/).

## Test harnesses (borrow their eval flags, not their GPUs)

- **lm-eval-harness** covers all default Huginn benchmarks. From the recurrent-pretraining README:
  ```
  lm_eval --model hf \
    --model_args pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32 \
    --tasks gsm8k_cot --batch_size=auto --apply_chat_template=True --fewshot_as_multiturn \
    --system_instruction="You are a helpful assistant that can assist users with mathematical reasoning."
  ```
  Note how loop behavior enters eval: `mean_recurrence=32` is a model argument. We can run the same idea against Dory by varying its loop knob instead.
- **bigcode-eval** was used for the code tasks in the paper.

## Curated paper lists (for further deep dives)

- [asimfish/awesome_loop_transformer](https://github.com/asimfish/awesome_loop_transformer)
- [huskydoge/Awesome-Loop-Models](https://github.com/huskydoge/Awesome-Loop-Models)

Both are well maintained and cover the 2026 wave (Hyperloop, Loopie, MELT, FlashLoop, LPT, LoopFormer).

## First results from our own loop eval (2026-10-30)

The dataset from `datasets/loop_evals/generate.py` (18 tasks, seed 7: 6 arithmetic4, 6 graph_reach, 6 stack_program, all answers normalized to a number) scored against the three loop GGUF variants with `tests/test_loop_evals.py` at temperature 0:

| task type | 1 loop | 2 loops | 4 loops |
|-----------|--------|---------|---------|
| arithmetic4 | 6/6 | 6/6 | 6/6 |
| graph_reach | 5/6 | 5/6 | 6/6 |
| stack_program | 1/6 | 3/6 | 3/6 |

Read-through: exactly the expected loop signature from the old research. Tasks solvable in one pass (deterministic arithmetic) never change. Algorithmic tasks with iterated structure (reverse polish stack programs, LoopedLAN punchcard kin) collapse at a single loop and gain a lot by deepening. Graph reachability needs just a bit of extra depth. Meanwhile throughput pays for deeper loops (110 tok/s at 1 loop down to 46 tok/s at 4 on this GPU). Raw artifacts: `scratch/loop_eval_results.json`.

## Our simulated loop evals (the plan)

NVIDIA's loop numbers (SWE-bench Verified, Terminal-Bench, GPQA, IFBench...) are internal-harness results on agents and data we cannot run. We simulate cheaply and honestly:

1. **Multi-step arithmetic bundles** - generated problems whose steps are graded per intermediate result, so loop count (1/2/4) visibly moves accuracy. RASP-L generators are the template.
2. **Algorithmic reach tasks** - graph reachability, string transforms, small ISA-style programs inspired by LoopedLAN punchcards: tasks where 1 loop provably fails and k loops succeed.
3. **Huginn-style fixed prompts** - reuse their lm-eval task list (gsm8k_cot first) against a served Dory via the OpenAI-compatible endpoint, scoring ourselves.
4. **Token efficiency check** - the Dory card promises fewer output tokens at higher loop counts. Measure output token count at loop 1 vs 2 vs 4 on our eval set; it is the cheapest loop-proxy metric we have.
5. **Latent thinking probe (stretch)** - logit-lens style decode of the loop trajectory, like the "Latent Chain-of-Thought?" interpretability work (arXiv 2507.02199). Only after the docker trial works.

Where it lives: generators and fixtures in `datasets/`, runnable tests in `tests/` (Python), results as artifacts in `$SCRATCH` (gitignored).

## Secrets policy reminder

`HF_TOKEN` is read-only and lives in `.env`. Never bake it into datasets, manifests, URLs or container images; load it into the process environment at runtime only.
