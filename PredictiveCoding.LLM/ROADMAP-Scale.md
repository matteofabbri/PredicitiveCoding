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

*Status 2026-07-20: solved for the Scale presets — `DeviceDecoderTrainer` (A1) runs
the full step device-resident with one host sync per step, measured ×37-38 vs CPU.*

---

## Phase A — Real GPU compute (the gate) · ✅ CLOSED 2026-07-20 (A4 as mirror, untested without AMD hardware)

*Without this, Phases B–D are impractically slow on one card. Everything starts here.*

| Task | Status | Notes |
|------|:------:|-------|
| **A1 · Device-resident training step** | ✅ | `Device/Training/DeviceDecoderTrainer`: the full DecoderLM step (forward, backward, AdamW) resident on the card for the Scale-preset family (dense attention, RMSNorm+RoPE, dropout 0). Weights + Adam moments + a preallocated **activation arena** live on device and are reused every step — zero allocations and **zero per-op host round-trips** in the hot loop (tokens go up, one scalar loss comes down; one sync per step). Per-head CPU weights pack into per-block GEMMs (Q/K/V columns concatenated, Wo rows stacked — same maths, one GEMM instead of one per head); `ImportFrom`/`ExportTo` are bit-exact round-trips. Gradient accumulation on device (`beta=1` GEMMs + atomics), micro-batch mean folded into the Adam read. Parity is test-enforced against the CPU twin (first loss to 1e-3; 8 AdamW steps within 5e-3 on loss and weights). **Measured (Release, RTX 4070 SUPER): Small @ 512 = 12,849 tok/s vs 350 CPU (×36.7), 0.56 GB device; Base @ 1024 = 2,572 tok/s vs 68 CPU (×38.0), 2.87 GB device** — the arena needs ~3× less memory than the CPU tape (no [seq,seq] scores, no CE probability tape). CPU side of the same commit series: `Ag.RmsNorm`/`Ag.Rope` + `UseRmsNorm`/`UseRope` config flags (presets switched to the Llama-style block, finite-difference tested). |
| **A2 · Mixed-precision training** | ✅ | `DeviceDecoderTrainer(mixedPrecision: true)`: every GEMM runs **bf16 on the Ada tensor cores** (`MatMulDeviceExBf16`, cublasGemmEx R16BF, fp32 accumulation) — fp32 master weights + Adam moments, bf16 working weight copies refreshed after each Adam step, bf16 staging of the GEMM operands (forward stagings kept per block and reused by the backward dW products), gradient accumulators stay fp32. bf16 keeps fp32's exponent range → **no loss scaling anywhere**. Hand-rolled RNE f32↔bf16 kernels (no CUDA headers under NVRTC). Verified: conversion within 2⁻⁸, bf16 GEMM vs fp32, first-loss parity with the CPU model (≤2%), 10-step trajectory tracks the fp32 trainer (≤5%). **Measured: Small 13,544 tok/s (+5.6% vs fp32 device), Base 2,680 tok/s (+5.5%), memory +0.08/+0.40 GB (staging + bf16 weights).** The honest finding: the gain is small because **GEMMs are only ~8% of the step** — ~830 GFLOP ≈ 33 ms of the Base's 403 ms; tensor cores cut them to ~10 ms, and the measured +5.5% matches that arithmetic exactly. The step is dominated by the fused attention kernels (serial per-position loops with in-block reductions). **Next real throughput lever → tile the A3 kernels** (K/V tiles staged in shared memory, per-thread online-softmax partials merged once per tile instead of per position), not precision. |
| **A3 · Fused attention kernel** | ✅ | The optimisation is in: `DeviceTrainKernels.AttentionForward` — **block per (query, head), K/V tiles staged in shared memory** (+1 padded stride: the unpadded version put a warp's serial dots on one smem bank — a measured 28% step regression, fixed), per-position scores as per-thread serial dots from smem, and the online-softmax reduction once per TILE instead of once per position; backward replays weights from the saved logsumexp flash-style (`attn_bwd_dq` per query block, `attn_bwd_dkv` per key block, `attn_dot` correction) — no [seq,seq] matrix ever materialises, forward or backward. Verified against the autograd reference (fwd + all three grads to 1e-4). The original thread-per-head `attn_1q` stays for single-query serving. **Measured limit, stated honestly:** at Base @ 1024 the attention is bound by **global K/V bandwidth** (every query re-reads its whole prefix: ~116 GB/step across the three kernels ≈ ~260 ms of the 399 ms step at ~440 GB/s). The next step-level jump is **cross-query reuse** (flash-2-style query tiles sharing each staged K/V tile, bandwidth ÷Tq) — deliberately deferred to the Phase-D run-up, when every ms of the real training run pays for it. |
| **A4 · ROCm / AMD backend** | ✅* | Shipped exactly as this row always promised — *a mirror of the verified CUDA path, gated behind availability, marked UNTESTED-WITHOUT-HARDWARE*: `Interop/Hip.cs` (amdhip64 + hipBLAS P/Invoke, version-suffixed names in `NativeResolver`; hipBLAS's op enum is 111/112, not cuBLAS's 0/1 — the one real API divergence, handled), `RocmGemmProvider : IGemmProvider` (same row-major swap, per-call CPU fallback; `MatMulMixed` stays on the CPU oracle — the Matrix-Core path is deliberately not mirrored blind), `DeviceRuntime.RocmAvailable/RocmDeviceCount` (DllNotFound-safe probing), `DeviceRegistry.EnableRocm` + `AutoSelect` = CUDA → ROCm → CPU. Tensor *placement* stays CPU for ROCm (the A1 trainer is NVRTC/CUDA-only by nature). The everywhere-runnable half — probing, fallback correctness, registry wiring — is test-covered on every machine; **the hip path itself has never touched silicon** (no project machine has a supported Radeon) and stays flagged in every file header until one exists. |

*Order inside A: A1 first (it is the enabler), A2 rides on it (same device path), A3 is the
biggest single kernel, A4 is parallel work that reuses A1's device plumbing.*

## Phase B — Scale the model · ✅ (2026-07-20)

| Task | Status | Notes |
|------|:------:|-------|
| **B1 · Real config presets** | ✅ | `DecoderLmConfig.Small` (dim 384, 8 blocks, 6 heads, ffn 1536, ctx 512): **26.8M** params @ 16k vocab. `DecoderLmConfig.Base` (dim 768, 12 blocks, 12 heads, ffn 3072, ctx 1024): **135.3M** @ 32k — GPT-2-small class. Dense on purpose (2026 features stay in the toy showcase, re-enable per-run via `with`), and since A1 the presets run the **Llama-style block — RMSNorm + RoPE** (`UseRmsNorm`/`UseRope`, default-off flags; the device path implements exactly this family). `PredictiveLlm.CreateSmall/CreateBase(tokenizer)` take the vocab from the tokenizer. Enabling change: the embedding is now a **row gather** (`Ag.GatherRows`, forward bit-identical to the one-hot matmul, scatter-add backward) — at 16k-32k vocab the old `[seq, vocab]` one-hot would have dominated the whole step. `DenseParameterFormula` gives the closed-form count, test-verified against real instances. Gradient checkpointing turned out **unnecessary for the fit** (see B3); it stays a possible later throughput option, not a requirement. |
| **B2 · Real BPE tokenizer** | ✅ | `BpeTokenizer.TrainFromCorpus(CorpusSource, targetVocab)` — word-split BPE at scale: GPT-2-style piece splitter (hand-rolled, no regex), BPE over **unique pieces weighted by frequency** with incremental pair counts + lazy-invalidation priority queue (deterministic tie-break). **Measured (Release): 16,124 merges on a 12 MB / 200k-unique-piece Zipfian corpus in ~2 s of merge loop; encode 13.8 MB/s** (per-piece concurrent cache); the counting pass is one linear stream — a 300 MB crawl stays in the minutes. Piece-mode files carry a `"BPE2"` magic; existing `"BPE1"` whole-stream tokenizers load and encode bit-identically. |
| **B3 · Memory budget, measured** | ✅ | `MemoryBudget.Measure(config, seqLen)` runs one **real forward+backward** and walks the autograd graph from the loss — activations are counted from the live tape, not estimated. Table below. |

### B3 — the measured table (fp32; re-measured on the RMSNorm+RoPE presets)

Measured with `MemoryBudget.Measure` (Release, 2026-07-20, post-A1a presets).
"fp16 proj" assumes the standard A2 recipe — 16-bit working
weights/grads/activations over fp32 master weights + Adam moments — and stays a
*derivation* until A2 re-measures it on device. The activation column is the
**CPU autograd tape**; the last column is the **A1 device arena actually
allocated by `DeviceDecoderTrainer`** (weights + Adam + activations + workspace)
— ~3× smaller than the CPU tape, because the fused A3 attention materialises no
[seq,seq] scores and the fused cross-entropy keeps no probability tape.

| Preset | seq | params | grads | Adam m+v | CPU tape | **total fp32 (CPU)** | fp16 proj | **A1 device arena** |
|--------|----:|-------:|------:|---------:|---------:|---------------------:|----------:|--------------------:|
| Small (26.8M @ 16k) | 256 | 102 MB | 102 MB | 204 MB | 334 MB | 0.72 GB | 0.56 GB | — |
| Small (26.8M @ 16k) | 512 | 102 MB | 102 MB | 204 MB | 811 MB | 1.19 GB | 0.79 GB | **0.56 GB** |
| Base (135.3M @ 32k) | 512 | 516 MB | 516 MB | 1.03 GB | 2.80 GB | 4.75 GB | 3.38 GB | — |
| Base (135.3M @ 32k) | 1024 | 516 MB | 516 MB | 1.03 GB | 7.32 GB | **9.17 GB** | 5.59 GB | **2.87 GB** |

**Fit on 12 GB (≈11 GB usable after runtime overhead):** on the device path the
Base @ 1024 step takes **2.87 GB** — room for ~4 concurrent windows before
gradient accumulation even in fp32, and A2 (bf16) halves the arena's activation
share again. The CPU-tape columns remain the honest cost of the pure-CPU path.

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
