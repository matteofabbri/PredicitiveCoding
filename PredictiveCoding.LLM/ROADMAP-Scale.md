# PredictiveLlm — Scale roadmap (from professional-at-toy-scale to a real trained model)

> **Mission.** ROADMAP-Pro made every mechanism real, measured and tested — but,
> by explicit design, **at toy scale** (dim 64, ~284k params, byte tokenizer). It
> closed with the model *"professional-grade at toy scale."* This roadmap spends the
> one resource that was deliberately left on the table: **scale**. Take the finished
> architecture to a genuinely useful small language model — a real tokenizer, a real
> corpus, real GPU training on a single **RTX 4070 SUPER** — and measure that its
> output is actually coherent, not just that its machinery is correct.

## Honest ceiling — numbers, not vibes

One RTX 4070 SUPER (12 GB GDDR6X, strong fp16/bf16 tensor throughput) trains a
**small** model. Realistic target band: **30M–125M parameters, GPT-2-small class.**

What that buys: a model that forms grammatical sentences, stays on a topic for a
paragraph, and answers simple prompts. What it does **not** buy: anything that rivals
a modern hosted LLM. Frontier quality needs a GPU cluster and trillions of tokens —
not one consumer card. Any plan that promises otherwise on this hardware is lying.
This roadmap is scoped to what one 4070 can actually deliver, and says so at every phase.

## Design invariants (inherited)

- **In-house only.** No new NuGet dependencies. P/Invoke to *system* GPU libraries
  (CUDA's `cudart`/`cublas`, AMD's `hip`/`rocblas`) is allowed — that is exactly how
  the existing CUDA backend already works, no shipped native DLL.
- **Additive / opt-in.** Defaults keep today's behaviour bit-identical; every scale
  step is config, flipped on once measured.
- **Numbers, not vibes.** Every phase quotes tokens/s, GPU memory, and bits/byte —
  before and after — against the Phase-0 yardstick.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | Done (in ROADMAP-Pro or here). |
| 🟡 | Building block exists, needs extension for scale. |
| ❌ | To build. |

---

## The real problem this roadmap exists to solve

"Enable CUDA and it's fast" is **false as stated**, and it is the crux of everything
below. Today `DeviceRegistry.EnableCuda()` swaps `Gemm.Provider` for the cuBLAS
provider — so an isolated matrix multiply runs on the GPU. But the LLM's autograd
(`Autograd/Autograd.cs`) computes on **host** `DenseTensor`s, and everything that is
not a GEMM — softmax, RMSNorm, RoPE, the attention score/value loops, GELU, the
elementwise add/mul of backprop — is hand-written CPU code that never touches
`Gemm.Provider`. So a single training step would bounce host↔device around each matmul,
and the PCIe transfers dominate: net speedup is small, sometimes negative.

**Real GPU training means the forward and backward must live on the device**, not just
the matmuls. That is genuine engineering (Phase A), not a flag. Naming it honestly is
the difference between a professional plan and a wish.

---

## Phase A — Real GPU compute (the gate) · ❌

*Without this, Phases B–D are impractically slow on one card. Everything starts here.*

| Task | Status | Notes |
|------|:------:|-------|
| **A1 · Device-resident training step** | ❌ | Run the autograd on `DeviceTensor` (already in `PredictiveCoding.Device`) so activations stay on the GPU across a full step; matmuls in cuBLAS, and device paths (kernels or batched-GEMM formulations) for the non-GEMM ops — softmax, RMSNorm, RoPE, GELU, add/mul. Success metric: **tokens/s on the 4070 vs CPU** on a fixed Base-config step, and zero per-op host round-trips in the hot loop. |
| **A2 · Mixed-precision training** | 🟡 | Storage-side mixed precision exists (Phase-5 `WeightStoragePrecision`, `IGemmProvider.MatMulMixed`, int8/Half checkpoints). Extend to the *compute* path: fp16/bf16 activations + matmuls on the Ada tensor cores, fp32 master weights + accumulation. Halves memory, unlocks the tensor cores. Metric: memory and tokens/s vs fp32. |
| **A3 · Fused attention kernel** | 🟡 | Correctness version **done + verified** (`DeviceMath.attn_1q`): the online-softmax attention (matching `MultiHeadAttention.AttendOnline`, basic case) as a device kernel, checked identical to standard CPU attention (maxDiff 3e-8 on the 4070). **NEXT STEP — the optimisation:** the current kernel is one thread per head; rebuild it block-per-head with a **parallel reduction across headDim/n and shared memory** (true flash-attention style) for throughput. Correctness is settled; this is purely speed. |
| **A4 · ROCm / AMD backend** | ❌ | `RocmGemmProvider` (hipBLAS/rocBLAS P/Invoke) + `RocmDeviceRuntime`, registered through `DeviceRegistry` exactly like CUDA — the architecture already invites it (`IGemmProvider`/`IComputeDevice`; `docs/07` §"Extending to ROCm"). **Honest caveat, stated up front:** we have **no AMD GPU to test it on** — the server's Ryzen 9700X iGPU is 2 CUs and not a ROCm-supported compute device, and this dev PC is NVIDIA + Intel. ROCm on Windows is also still immature (its real home is Linux + a supported Radeon). So A4 ships *mirroring the verified CUDA path*, gated behind availability, and marked **UNTESTED-WITHOUT-HARDWARE** until a supported Radeon (or a Linux ROCm box) exists. Worth building for portability — the whole point of the `IGemmProvider` seam — but not pretendable as verified. |

*Order inside A: A1 first (it is the enabler), A2 rides on it (same device path), A3 is the
biggest single kernel, A4 is parallel work that reuses A1's device plumbing.*

## Phase B — Scale the model · ✅ (2026-07-20)

| Task | Status | Notes |
|------|:------:|-------|
| **B1 · Real config presets** | ✅ | `DecoderLmConfig.Small` (dim 384, 8 blocks, 6 heads, ffn 1536, ctx 512): **26.8M** params @ 16k vocab. `DecoderLmConfig.Base` (dim 768, 12 blocks, 12 heads, ffn 3072, ctx 1024): **135.4M** @ 32k — GPT-2-small class. Dense GPT-2-style on purpose (2026 features stay in the toy showcase, re-enable per-run via `with`); `PredictiveLlm.CreateSmall/CreateBase(tokenizer)` take the vocab from the tokenizer. Enabling change: the embedding is now a **row gather** (`Ag.GatherRows`, forward bit-identical to the one-hot matmul, scatter-add backward) — at 16k-32k vocab the old `[seq, vocab]` one-hot would have dominated the whole step. `DenseParameterFormula` gives the closed-form count, test-verified against real instances. Gradient checkpointing turned out **unnecessary for the fit** (see B3: Base @ 1024 fits 12 GB at micro-batch 1 even in fp32); it stays a possible later throughput option, not a requirement. |
| **B2 · Real BPE tokenizer** | ✅ | `BpeTokenizer.TrainFromCorpus(CorpusSource, targetVocab)` — word-split BPE at scale: GPT-2-style piece splitter (hand-rolled, no regex), BPE over **unique pieces weighted by frequency** with incremental pair counts + lazy-invalidation priority queue (deterministic tie-break). **Measured (Release): 16,124 merges on a 12 MB / 200k-unique-piece Zipfian corpus in ~2 s of merge loop; encode 13.8 MB/s** (per-piece concurrent cache); the counting pass is one linear stream — a 300 MB crawl stays in the minutes. Piece-mode files carry a `"BPE2"` magic; existing `"BPE1"` whole-stream tokenizers load and encode bit-identically. |
| **B3 · Memory budget, measured** | ✅ | `MemoryBudget.Measure(config, seqLen)` runs one **real forward+backward** and walks the autograd graph from the loss — activations are counted from the live tape, not estimated. Table below. |

### B3 — the measured table (fp32 CPU tape; fp16 = derived projection until A2)

Measured with `MemoryBudget.Measure` (Release, 2026-07-20). "fp16 proj" assumes the
standard A2 recipe — fp16 working weights/grads/activations over fp32 master
weights + Adam moments — and stays a *derivation* until A2 re-measures it on
device. The activation column is the **current CPU autograd tape**: an honest
upper bound. The A3 fused attention will shrink it further (the tape's
`[seq, seq]` per-head score matrices — ~1.2 GB of the Base @ 1024 figure —
disappear into the online-softmax kernel).

| Preset | seq | params | grads | Adam m+v | activations | **total fp32** | **fp16 proj** |
|--------|----:|-------:|------:|---------:|------------:|---------------:|--------------:|
| Small (26.8M @ 16k) | 256 | 102 MB | 102 MB | 204 MB | 323 MB | 0.71 GB | 0.56 GB |
| Small (26.8M @ 16k) | 512 | 102 MB | 102 MB | 204 MB | 790 MB | 1.17 GB | 0.78 GB |
| Base (135.4M @ 32k) | 512 | 516 MB | 516 MB | 1.03 GB | 2.73 GB | 4.68 GB | 3.35 GB |
| Base (135.4M @ 32k) | 1024 | 516 MB | 516 MB | 1.03 GB | 7.19 GB | **9.04 GB** | **5.53 GB** |

**Fit on 12 GB (≈11 GB usable after runtime overhead):** Small @ 512 runs ~13
physical windows per step in fp32 (~26 in fp16) before gradient accumulation;
Base @ 1024 fits at **micro-batch 1 even in fp32** (9.04 GB), ~2 windows in
fp16 — gradient accumulation supplies any effective batch size on top.

## Phase C — The corpus, built by the crawler · ❌ (web-native, continual-first)

*Reframed after Matteo's point: the live crawler is the project's signature, not a
thing to replace with a downloaded dataset. The distinction that actually matters is
not "corpus vs crawler" — it is **seen once vs revisited enough to consolidate**. So
the crawler takes on a second job and keeps both forever.*

| Task | Status | Notes |
|------|:------:|-------|
| **C1 · Decouple fetch from learn (the corpus builder)** | ❌ | Today the crawler learns each page once and drops it. Split the two: the crawler writes cleaned page text to an **append-only, sharded on-disk corpus**, as fast as it can fetch — ungated by the slow `Learn`. That accumulation *is* the pretraining corpus, and it is **our crawl, not a download**. Reuses today's AngleSharp extraction; shards are read back by the Phase-4 `CorpusSource.FromFolder`. |
| **C2 · Hygiene at web scale** | 🟡 | Raw web text is noisy — the honest cost of web-native data vs curated. `CorpusSource.Deduplicated` (64-bit content hash) + `.WithLengthFilter` are ready; add boilerplate stripping and an optional language filter (`PredictiveCoding.Data` already ships language detection). Narrows — does not close — the per-token quality gap to a curated corpus. |
| **C3 · Bootstrap: pretrain on the accumulation, multi-epoch** | 🟡 | Once enough tokens have piled up (target: hundreds of M+), run real **multi-epoch** `Train(CorpusSource)` over the shard folder — the "learn to read" step a once-through stream can't provide. Content-hashed held-out split (Phase-4 `HeldOutHashed`) keeps bits/byte comparable across runs. This is what a from-scratch model needs *before* continual learning can do anything but thrash. |
| **C4 · Then the live stream, forever (the signature)** | ✅→wire | After bootstrap, live pages feed `ContinualLearner` (Phase-6: replay + surprise-lr + forgetting guardrail) on top of the pretrained base — now genuinely *effective*, because there is a competent model to keep current instead of one to overwrite. Replay already turns the once-through stream into repeated exposure, so the crawler both **builds** the base and **maintains** it. |

*Net: the crawler doesn't shrink to a side stream — it grows a second role (sediment a
corpus) so it can bootstrap the very model it then keeps learning forever. Curated data
(Wikipedia/books/code) stays available as an optional `CorpusSource.Mix` ingredient if
we ever want to lift quality, but it is not required and not the default: the web is the corpus.*

## Phase D — The training run · ❌

| Task | Status | Notes |
|------|:------:|-------|
| **D1 · Pretrain for real** | 🟡 | Everything but the run exists: LR schedule, gradient accumulation, `CheckpointEverySteps` + best-on-val + true resume (Phase-5). Pretrain a preset on the 4070 for real wall-clock (days). **Target: bits/byte materially below the byte toy, and samples that form coherent sentences** — the first time output quality, not mechanism correctness, is the metric. |
| **D2 · Monitoring** | 🟡 | Loss curve, tokens/s, periodic samples, and the Phase-8 `ProEval` harness at intervals. Wire into the existing `TrainingOptions.OnEpochEnd`/checkpoint hooks. |

## Phase E — Eval, serve, quantize · ❌

| Task | Status | Notes |
|------|:------:|-------|
| **E1 · Real eval** | ✅→run | Run Phase-8 `ProEval` (bits/byte, per-domain ppl, echo/induction probes, serve tok/s) on the trained model; store a `RegressionHarness` baseline so later continual learning can't silently rot it. |
| **E2 · Serve + quantize** | ✅→use | `CreateServer()` (O(n) KV-cache path) + sampling/chat (Phase 1/2); ship Int8/Half weights (Phase-5) for a small deployable artifact. |

## Phase F — Deploy the real model · ❌ (conditional)

Only decide once D produces something worth serving. The **AMD server serves it on CPU**
— fine for a Small model at low traffic, slow for Base. The fast path is a GPU host (a
box with an NVIDIA card, or serving straight from the 4070). No point CUDA-enabling the
*current* server's container: it has no NVIDIA GPU (see the GPU note in the project state).

---

## Order rationale

**A is the gate** — real GPU compute, or B–D crawl on one card. **B + C** define *what*
model and *on what* data. **D** is the run that finally makes output quality the metric.
**E** measures it and locks a regression baseline. **F** ships it only if it earns serving.

## Deliberately out of scope

- **Frontier scale.** A GPU cluster + trillions of tokens; not one 4070.
- **Loading external open weights** (Llama/Mistral/…). This is its own architecture, not
  a HuggingFace-compatible one — capability comes from training *this* model, not adopting
  someone else's. A separate importer would be a research project, not a phase.
- **RLHF / full reward modelling.** Per ROADMAP-Pro: the on-policy distillation loop
  already captures the practical value at our scale.
- **Packaging/distribution.** House rule, unchanged: professional = internal quality.
