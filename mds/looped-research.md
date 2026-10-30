# Looped Transformers: the older research we build on

Status: prep document. Written before any NVIDIA tech report or blog for Dory1.0-4B exists. All config facts below were read from the actual Hugging Face repos (via the read-only HF token), not from hearsay.

The new model we care about is [NVIDIA Dory1.0-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF), a looped (latent-recurrent) LLM released around October 2026. It has no technical report yet, so this file collects the older, smaller looped model research that everything in the lab will lean on.

See also: `mds/datasets.md` for code, test harnesses and datasets we can reuse.

## The core idea, in one formula

A looped transformer reuses the same transformer layers for several passes over the hidden state. If the stack has `L` distinct blocks and we loop `T` times:

```
effective block applications = L * T
```

A normal transformer is `T = 1`. Looping is depth-wise weight tying: parameter count stays small, compute per token scales with loop count. The loop count becomes a runtime knob that trades inference compute for accuracy.

## The lineage of "old approaches"

### Universal Transformers (2018/2019)

[Dehghani et al., arXiv 1807.03819](https://arxiv.org/abs/1807.03819). The grandfather: apply the same transformer block repeatedly, with adaptive halting so each token can stop early. Halting ideas go further back to ACT ([Graves 2016](https://arxiv.org/abs/1603.08983)). Modern looped models reuse the same depth-recurrence idea with different loop/exit mechanisms.

### LoopedLAN: transformers as programmable computers (ICML 2023)

[Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos, arXiv 2301.13196](https://arxiv.org/abs/2301.13196). A constant number of encoder layers, frozen into a loop, can emulate lexicographic operations, condition branches, function calls and even full backprop. The input sequence acts as a punchcard program. Official implementation: `jysohn1108/Looped-Transformer` (see datasets.md).

### RASP-L: looped transformers learn learning algorithms (ICLR 2024)

[Yang, Lee, Nowak, Papailiopoulos, arXiv 2311.12424](https://arxiv.org/abs/2311.12424). "Looped Transformers are Better at Learning Learning Algorithms". Trained looped models beat their parameter-matched baseline on tasks like addition, parity and graph problems, and the loop count decides when they succeed.

### Length generalization follow-up (ICLR 2025)

[arXiv 2409.15647](https://arxiv.org/abs/2409.15647), code `UW-Madison-Lee-Lab/looped-tf`. Looping helps length extrapolation on algorithmic tasks.

## The first real pretrained looped model: Huginn-3.5B (Feb 2025)

Paper: [Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", arXiv 2502.05171](https://arxiv.org/abs/2502.05171). Model: [tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) (Raven architecture). Code: `seal-rg/recurrent-pretraining` (Apache-2.0).

Verified straight from its `config.json` on the Hub:

| field | value |
|------|-------|
| prelude (input) layers | 2 |
| recurrent block layers | 4 |
| coda (output) layers | 2 |
| embedding dim | 5280 |
| attention heads | 55 (head_dim 96, MHA, qk_bias) |
| FFN intermediate size | 17920 |
| context block size | 4096 |
| mean_recurrence | 32 |
| effective expected depth | 2 + 4*32 + 2 = 132 |
| notable init | takase strategy, orthogonal init off |
| state init / injection | like-init, linear injection |

Key facts:

- 3.5B effective parameters (roughly 1.5B prelude+coda, 0.5B embedding, 1.5B recurrent core), trained for 800B tokens on 4096 AMD MI250X GPUs on Frontier. HF tensors show ~4B in fp32 storage.
- During training the number of recurrent iterations is randomly sampled (`poisson-lognormal-filling`), with `mean_backprop_depth` 8 keeping gradients stable without full unrolling cost.
- The model iterates `s_i = R(e, s_prev)` on a continuous latent state: no tokens are generated between iterations. Thinking happens in latent space.
- Test-time compute scales: quality keeps improving up to around 64 iterations, sometimes reaching compute equivalent to a 50B parameter dense model on reasoning benchmarks.
- The paper trained a non-recurrent twin at 180B tokens, and the recurrent model beat it especially on harder (ARC-C style) tasks, even at a single recurrence. The loop body encodes the gains.
- KV cache sharing and speculative decoding fall out naturally from the architecture.

The Name "Huginn" comes from Odin's raven of thought, a good mascot for latent thinking.

## ByteDance Ouro: the looped family with an exit gate (Oct 2025)

Paper: [Zhu et al., "Scaling Latent Reasoning via Looped Language Models", arXiv 2510.25741](https://arxiv.org/abs/2510.25741). Project: [ouro-llm.github.io](https://ouro-llm.github.io/). Models: `ByteDance/Ouro-1.4B`, `Ouro-1.4B-Thinking`, `Ouro-2.6B`, `Ouro-2.6B-Thinking`.

Verified from `ByteDance/Ouro-1.4B` `config.json`:

| field | value |
|------|-------|
| num_hidden_layers | 24 (the whole stack loops) |
| hidden_size | 2048 |
| attention | 16 heads, MHA, head_dim 128 |
| FFN intermediate | 5632, SwiGLU |
| RoPE theta | 1,000,000 |
| context | 65536 positions |
| total_ut_steps | 4 |
| early_exit_threshold | 1.0 |
| vocab | 49152 |
| dtype | bfloat16 |

- The entire 24-layer stack loops through 4 recurrent steps: effective 96 block applications from only 24 distinct layers.
- Their public claim: 1.4B with 4 loops approximately matches 3B-4B traditional models.
- The exit gate scores each loop's representation and a Q-exit rule stops early when cumulative exit probability crosses a threshold. In the released HF implementation all configured passes are computed first and the threshold picks among them, so check `total_ut_steps` if you want real compute savings.
- STARS regularization (Jacobian spectral radius control with random loop sampling) keeps looped recursion from collapsing at depth.
- Their own follow-up [Sparse Layers are Critical to Scaling Looped LMs, arXiv 2605.09165](https://arxiv.org/abs/2605.09165) found dense looped models scale worse than plain transformers, but Looped-MoE (different experts firing per pass) scales better, and loop boundaries are natural early-exit points.

## Hyperloop and SenseNova-Loop-9B-Preview

[Hyperloop Transformers, arXiv 2604.21254](https://arxiv.org/abs/2604.21254): begin-middle-end block layout, and instead of a single residual stream it expands to n parallel residual streams that get mixed back after each loop via hyper-connections (n = 4 in their study). A small per-loop write-back matrix does the mixing.

Model: [sensenova/SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview), a large looped multimodal model using a shared looped block and hyper-connections across dual residual streams. Notable: SenseNova U1 already has llama.cpp support (v0.8.0 changelog), so this family serves well.

## The Loopie series: looped MoE at scale (Jul 2026)

[Loop the Loopies!, arXiv 2607.16051](https://arxiv.org/abs/2607.16051) (Gao et al.), plus the related [LoopMoE, arXiv 2606.04438](https://arxiv.org/abs/2606.04438).

- Loopie-20B-A2B: 20B total parameters, 2B active per token. Loopie's smaller sibling: 6B total, 0.6B active.
- Looped Mixture-of-Experts: the shared layers loop, and different experts fire on each loop. This finally closes the gap where parameter scaling always beat looping, with gold-medal tier math and physics scores from the low-active model.
- Important insight for us: which layers to loop matters. Layer-loop (per-layer repetition) beats model-loop (whole-stack repetition) in their ablations, and the survey in the preprints notes Loopie loops per layer.

Related theory line worth one skim each: [LoopFormer: elastic-depth looped transformers via shortcut modulation, ICLR 2026, arXiv 2602.11451](https://arxiv.org/abs/2602.11451) and [Recurrent Looped Transformer (alphaXiv 2609)](https://www.alphaxiv.org/abs/2609.recurrent-looped-transformer).

## The loop KV-cache problem and its fixes

Looping has a dirty secret: naive implementations store a KV cache per loop, so memory grows linearly with loop count. The fixes form a nice family:

| approach | trains? | trick |
|----------|--------|-------|
| [MELT, arXiv 2605.07721](https://arxiv.org/abs/2605.07721) | yes (gating) | one shared cache per layer across loops |
| [FlashLoop, arXiv 2609.29812](https://arxiv.org/abs/2609.29812) | no (inference only) | lazy updates, sparse attention, cross-loop KV residual quantization in packed 4bit; measured on Ouro-1.4B and Huginn-3.5B, up to 1.64x speedup, ~6x cache squeeze |
| [Looped Latent Attention, arXiv 2607.15456](https://arxiv.org/abs/2607.15456) | trains encoders | compresses KV along the recurrence dimension |
| [LPT shared memory, arXiv 2610.02383](https://arxiv.org/abs/2610.02383) | trains (IBM) | later loops read a shared memory window; better quality with up to 79% less context memory |
| Ouro last-step KV reuse | no | keep only the final loop cache for decoding |
| Dory last-loop KV caching | pretrained in | overwrite one cache per loop, memory flat in loop count |

Side note on IDOL ([arXiv 2407.10937, ECCV 2024](https://arxiv.org/abs/2407.10937)): a dual-modal video-depth diffusion framework with a parameter-sharing U-Net. Not a looped LLM, only relevant as parameter sharing beyond language models. We will not chase it further.

## Dory1.0-4B: what we know today

From the GGUF model card on the Hub (still no paper):

- Architecture: looped normalized-GPT (nGPT backbone), layer stack split into three groups: Input (I) 8 layers, Recurrent (R) 20 layers, Output (O) 8 layers. The shared R block re-applies between I and O, and the original input is re-injected into every loop.
- Loop count is a runtime knob: 1, 2 or 4 loops benchmarked, 2 loops recommended as default. Most of the gain is already at 2 loops.
- Hybrid attention: interleaved sliding-window and full-attention layers.
- Context: up to 256K tokens.
- Looping is token-efficient: more compute per token, but fewer output tokens per task. Reasoning happens in latent space between loops.
- Memory: last-loop KV caching keeps memory flat in loop count, no accuracy loss reported.
- Training: pretrained natively in NVFP4 (NVIDIA 4-bit float), then continued in BF16. Ships in BF16, MXFP8, NVFP4, Q5_K_M. License openmdw-1.1.
- llama.cpp support is very fresh: PR [ggml-org/llama.cpp 30306](https://github.com/ggml-org/llama.cpp/pull/30306). It is not visible in the v0.8.0 release notes snapshot, so we verify at serve time.
- The repo also ships an `explainability.md` with notes on what the loops are doing, and a dory_arch.png diagram. The GGUF is ~3.7GB in Q5_K_M.

## Why this matters for our lab

The old models and papers give us the expected behaviors to test against the new Dory: accuracy should jump mostly at loop 2, output tokens should shrink as loops rise, and cache tricks matter more than parameter count. Our simulated evals in `datasets/` will borrow tasks that the old papers already showed reward loops: multi-step arithmetic, algorithmic tasks, agentic-style multi-step tool problems.
