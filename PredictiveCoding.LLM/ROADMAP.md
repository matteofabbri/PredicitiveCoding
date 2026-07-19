# PredictiveCoding.LLM — 2026 research roadmap (predictive-coding edition)

> **Mission — one mandate:** fold the 2026 LLM research directions — adaptive reasoning
> effort, looped/weight-tied depth, multi-token prediction, long-context cache efficiency,
> hybrid attention+SSM, sparse experts, on-policy distillation, latent planning — into the
> in-house stack, expressing each one **through the predictive-coding lens** (iterative
> settling, prediction error, energy, persistent temporal state) instead of bolting on
> foreign machinery.

Design invariants across every phase:

- **In-house only.** No new dependencies, ever.
- **Additive.** The vanilla paths (`DecoderLM` classic forward, `GptModel` MHA) stay
  bit-identical; every new mechanism is opt-in via config, with `0`/`off` meaning
  "exactly the old behaviour".
- **Toy scale, real mechanisms.** We reproduce the *mechanism* with measurable ablations
  at small scale, not the frontier numbers — same stance as the rest of the repo.
- Each phase lands with unit tests; the full suite stays green (912 today).

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Done** — implemented and covered by tests. |
| 🟡 | **Partial** — building block exists, needs extension. |
| ❌ | **To build** — not present yet. |

*Status as of 2026-07-19: Phases 0–5 ✅ done (912/912 green; Phase 0's optional HTML
report deferred); Phases 6–8 ❌ to build. The network-variant family (incl. the PC-level
MoE trio) now lives in core under `PredicitiveCoding/Networks/` — the Expanded project
was dissolved the same day.*

### Paper anchors → phases

| 2026 trend / paper | Phase |
|--------------------|:-----:|
| "The Spike, the Sparse and the Sink" (massive activations, attention sinks) | 0, 4 |
| Controlling reasoning effort (low/med/high effort modes; adaptive compute) | 1, 2 |
| DeepLoop (depth scaling for looped transformers) | 2 |
| Nemotron 3 Super ablations: multi-token prediction | 3 |
| Agent-driven long-context efficiency; LongStraw's fixed-budget spirit | 4 |
| Nemotron 3 / Mamba-3 / Qwen3.6 / Arcee Trinity (hybrid attention+SSM) | 5 |
| Step 3.5 Flash (frontier behaviour, 11B *active* params — sparse MoE) | 6 |
| SEED (self-evolving on-policy distillation for agents) | 7 |
| Cola DLM (plan in continuous latent space, then decode) | 8 |

**Foundation already in place** (pre-existing, not part of this effort):

- Core PC `Network` (`PredicitiveCoding/Networks/Network.cs`): `Infer(inferenceSteps)`,
  `InferTrace` (per-step energy trajectory), temporal variants with persistent hidden
  state, `LearnGated`, mixed-precision twins.
- Trainable LM: `DecoderLM` on the autograd engine (12 ops; **gradients accumulate per
  use**, so weight-tying across reuses already back-propagates correctly), `Trainer`
  (grad accumulation + clipping), `Adam`, `LrSchedule`, `Checkpoint`, BPE tokenizer,
  `StreamingTextDataset`.
- Inference LM: `GptModel` is already 2025-modern — RoPE/ALiBi, GQA/MQA, MLA compressed
  KV cache, SwiGLU, RMSNorm, QK-Norm, logit soft-caps, flash-style attention,
  `KvCache` with `Clone`, `ContinuousBatcher`, `BeamSearch`.
- Mixed precision: `ToHalf`/`ToFloat` + mixed GEMM behind `Gemm.Provider`
  (see `PredicitiveCoding/ROADMAP-MixedPrecision.md`, Phase 5 — the LLM stack is the
  named "GEMM-heavy consumer" to route).
- Teacher & distillation precedents: `LmStudioSource` (chat/generate/embed on :1234),
  `DetectorDistiller`, `PcDistilledFaceEmbedder`.

---

## Phase 0 — Yardstick: eval harness + activation observatory · ✅ done 2026-07-19

*PC angle: measure the prediction error before changing the predictor.*

| Task | Status | Notes |
|------|:------:|-------|
| Perplexity eval harness | ✅ | `Eval/PerplexityEvaluator.cs` — token-weighted NLL → perplexity, `ignoreIndex` masking, `maxWindows` cap; deterministic `HeldOut`/`TrainingSplit` partitions of a window stream; all-zero-weights model measures exactly vocab-size ppl (test). |
| Tiny in-repo eval text | ✅ | `Eval/EvalCorpus.cs` (~1 KB, committed) — the fixed yardstick every phase quotes against. |
| Activation-stats probe | ✅ | `Eval/ActivationProbe.cs` — per-block mean/max abs + `SpikeRatio`; opt-in `probe` parameter on `DecoderLM.Forward` and `GptModel.Step` (null = untouched pass, bit-identical, tested). |
| Attention-sink meter | ✅ | per-(block, head) share of attention mass on position 0, on the trainable stack (`SoftmaxRowsCausal` probs); the inference-stack meter lands with Phase 4's sink handling. |
| Self-contained HTML report | ❌ | optional, deferred; reuse the `HtmlVisualizer` pattern from the Brain project (one file, no server). |

Every later phase quotes its before/after from this harness. Tests: `LlmEvalTests` (7).

## Phase 1 — Adaptive inference effort (core PC) · ✅ done 2026-07-19

*PC angle: "reasoning effort" is our native dial — effort = settling steps, and the
energy trajectory is the built-in stop signal. The transformer world is re-discovering
what the PC formulation gives for free.*

| Task | Status | Notes |
|------|:------:|-------|
| `Network.InferAdaptive(input, maxSteps, epsilon)` | ✅ | early-exit when a step improves the energy by less than ε **relative to the current energy** (scale-free, so presets transfer across network sizes); returns `(topLatent, stepsUsed, energy)`; free layers start from zero (deterministic in the input alone). ε ≤ 0 ⇒ bit-identical to `InferGetTop(maxSteps, resetState: true)`. |
| Effort presets | ✅ | `Networks/EffortLevel.cs` — `Low/Medium/High` → `(maxSteps, ε)`, one place to tune; `InferAdaptive`/`PredictAdaptive` overloads take the enum. |
| `PredictAdaptive` | ✅ | minimum-energy classification with adaptive settling per candidate; returns `(cls, energy, stepsUsed)` with the total steps actually spent. |
| Tests | ✅ | `AdaptiveInferenceTests` (7): ε=0 ≡ fixed path bit-for-bit (infer + predict), exit fires before the budget, ε-monotonicity, budget-invariance of the exit step, preset ordering, argument validation. |

**Finding.** The naive story "easy inputs exit early, noise spends more" is exactly
backwards on a trained net: the trained input keeps descending productively (deep
attractor, 30 steps) while noise hits its shallow "nothing more to explain" minimum
almost immediately (20 steps). The step counter is an **explanation-progress meter**,
not a familiarity meter — the effort goes where there is structure to explain, which
is the behaviour you actually want from adaptive compute.

## Phase 2 — Looped DecoderLM (weight-tied depth) · ✅ done 2026-07-19

*PC angle: reusing one tied block K times is settling toward a fixed point — the discrete
cousin of PC's iterative inference. Depth becomes iteration, which is exactly how the
core `Network` already treats computation.*

| Task | Status | Notes |
|------|:------:|-------|
| `DecoderLmConfig.NumLoops` | ✅ | default 1 = vanilla (bit-identical, parameter list untouched → old checkpoints still load); forward re-applies the block stack `NumLoops` times with tied weights. |
| Tied-reuse gradient test | ✅ | `TiedLoop_Gradients_MatchTheUnrolledSum`: a 1-block × 2-loop model reproduces, gradient for gradient, a 2-block unrolled twin with shared weights (tied grad = sum of the two uses). |
| Loop-step conditioning | ✅ | per-loop bias `[1, dim]`, zero-initialised, allocated only when `NumLoops > 1`. |
| Inference early-exit | ✅ | `LoopExitTau`: stop once `‖x_k − x_{k−1}‖ / ‖x_{k−1}‖ < τ` — eval only, τ=0 disables; `ActivationProbe.LoopsUsed` reports the passes actually run. |
| Tests | ✅ | `LlmLoopedTests` (6): NumLoops=1 bit-identical, tied-grad ≡ unrolled sum, looped training halves the loss, early-exit fires at eval only / τ=0 runs the budget, per-pass probe entries, config validation. |

The highest predictive-codiness-per-line item on the list.

## Phase 3 — Multi-token prediction · ✅ done 2026-07-19

*PC angle: predicting further into the future at multiple horizons is what the core's
temporal path (`LearnTemporal`) already does; an auxiliary t+2 head is the same idea one
level up.*

| Task | Status | Notes |
|------|:------:|-------|
| Aux prediction head | ✅ | `MtpWeight`/`MtpHorizon` in `DecoderLmConfig` (weight 0 = off, head not even allocated → old checkpoints load); aux head predicts t+MtpHorizon off the shared trunk, loss = CE + λ·CE_h. |
| Masked-target plumbing | ✅ | `MtpTargets` shifts the target stream MtpHorizon−1 further and masks the tail with `ignoreIndex`; all-masked short sequences skip the aux term. |
| Self-speculative decoding | ✅ | `Generation/SelfSpeculative.cs` — the t+2 head drafts, the next forward verifies; output identical to plain greedy **by construction**, acceptance/forwards counters reported. Pays off precisely because DecoderLM has no KV cache: every accepted draft saves a whole re-forward. |
| Tests | ✅ | `LlmMtpTests` (6): λ=0 ≡ vanilla (params + loss), one extra tensor when on, target shift/mask, both heads improve in training, speculative ≡ greedy with fewer forwards on a cyclic corpus, clean fallback without the head. |

## Phase 4 — Long-context cache efficiency · ✅ done 2026-07-19

*PC angle: bounded state — keep only what still helps prediction. The core's temporal
state is O(1) already; the KV cache should move toward it, not away.*

| Task | Status | Notes |
|------|:------:|-------|
| Sliding-window attention | ✅ | `TransformerConfig.WindowSize` (0 = full); `KvCacheLayer.Evict` drops the oldest post-sink entries; every entry carries its **absolute position**, so ALiBi distances and RoPE geometry survive eviction (window-never-filled ≡ full attention, bit for bit — tested). Standard MHA/GQA path; MLA + window is rejected at validation. |
| Attention sinks | ✅ | `NumSinkTokens` retained forever alongside the window — per the Spike/Sink findings and Phase 0's meter. |
| Half-precision KV cache | ✅ | `KvCacheHalf`: `KvCacheLayer` stores `Half[]`, attention computes in float via per-forward scratch (no hot-loop allocs) — the first concrete landing of Mixed-Precision Phase 5's "route the LLM to mixed". |
| Cache accounting | ✅ | `ILayerCache.BytesUsed` (KV and MLA latent) + `KvCache.BytesUsed` — the fixed-budget number LongStraw-style work optimises against; plateaus once the window fills (tested). |
| Tests | ✅ | `LlmKvCacheTests` (6): bit-identical when the window never fills, sink/window positions after eviction, bytes exactly halved + logits within tolerance on Half, plateau under budget, MLA byte accounting, config validation. |

MLA (compressed cache) already existed; this phase completes the efficiency menu.

## Phase 5 — Hybrid temporal-state blocks (attention + SSM) · ✅ done 2026-07-19

*PC angle: a gated linear recurrence `h_t = a_t ⊙ h_{t−1} + b_t ⊙ x_t` is a learned
temporal predictor with persistent state — the LM-side sibling of the core's
`InferTemporal` hidden state. O(1) state instead of a growing transcript is the PC way.*

| Task | Status | Notes |
|------|:------:|-------|
| New autograd ops | ✅ | `Ag.Mul`, `Ag.Sigmoid`, `Ag.RecurrentScan` (h_t = a_t⊙h_{t−1} + b_t; backward = BPTT inside the op); SiLU composes as `Mul(x, Sigmoid(x))`, no dedicated op needed. |
| Numeric grad check | ✅ | central-difference check of the full gated chain σ/⊙/scan → CE against the autograd gradients. |
| `TemporalBlock` (trainable) | ✅ | decay gate + input gate + scan + out-projection, then the standard FFN sublayer; `DecoderLmConfig.BlockPattern` ("TTAT"-style, null = all attention, `"AA…"` ≡ vanilla bit-for-bit — tested). |
| Inference twin | ✅ | `Model/RecurrentLayer.cs`: `RecurrentTemporalLayer : ISelfAttention` with `RecurrentStateCache` — the whole cache is ONE state vector (BytesUsed constant, tested); `ModernPresets.NemotronStyle` ships the 3:1 T:A interleave. |
| Tests | ✅ | `LlmHybridTests` (6): numeric grad check, `"AA"` ≡ vanilla, pattern validation, pure-T model learns to echo the previous token (the state provably carries the past — attention-free), O(1) vs O(n) cache growth, NemotronStyle end-to-end. |

**Honesty note on the original test claim.** "Hybrid beats attention-only on long-range
recall" was the wrong thing to promise: exact recall is attention's home turf — SSM
hybrids win on cache/compute, not recall. What is certified instead: the temporal state
demonstrably carries information (echo task solved with zero attention) and the cache is
O(1) where attention's is O(n). The efficiency-vs-recall trade is Phase 4+5's whole point.

## Phase 6 — Sparse experts (MoE feed-forward) · ❌

*PC angle: routing as precision-weighted gating — send the token to the expert expected
to predict it best; the conceptual sibling of `LearnGated` and of Cortex's area routing.
The PC-level MoE family already exists and lives in core (`Networks/MoE`:
`MixtureOfExperts`, `TemporalMixtureOfExperts`, `HierarchicalMixtureOfExperts`, plus
`Networks/Routed.RoutedNetwork` — moved out of the dissolved Expanded project,
2026-07-19); this phase is their FFN-level sibling inside `DecoderLM`.*

| Task | Status | Notes |
|------|:------:|-------|
| `ExpertsFfn` | ❌ | N expert MLPs + softmax router, top-k selection, outputs weighted by gate probability (differentiable through the taken path). |
| Load-balancing aux loss | ❌ | standard fraction·probability penalty to prevent expert collapse. |
| Config + accounting | ❌ | `NumExperts`/`TopK` (1/dense = vanilla); report **active params per token** — the Step-3.5-Flash headline metric. |
| Tests | ❌ | k=N matches dense reference within tolerance; balance loss spreads load on the toy corpus; accounting correct. |

## Phase 7 — On-policy distillation from the local teacher · ❌

*PC angle: the training signal is the mismatch between the student's own generation and
the teacher's correction — prediction error on the student's own trajectory (on-policy),
the generative sequel to `PcDistilledFaceEmbedder`.*

| Task | Status | Notes |
|------|:------:|-------|
| `LmDistiller` — offline KD | ❌ | prompt set → teacher completions via `LmStudioSource.ChatAsync` + `ChatTemplate` → student fine-tune with masked CE on completion tokens (sequence-level KD; the endpoint is text-level, logit-matching only if it ever exposes logprobs). |
| On-policy round (SEED-style) | ❌ | student generates a draft prefix → teacher continues/corrects from that prefix → student trains on the correction *from its own prefix*; iterate. |
| Dataset cache | ❌ | teacher transcripts cached under `~/.predictivecoding/` (runtime data — stays out of git, house rule). |
| Tests | ❌ | offline round-trip on a canned transcript (no network in CI); live LM Studio behind a manual flag. |

## Phase 8 — Latent planning before decoding · ❌ (ambitious)

*PC angle: the most PC-native idea in the 2026 crop. "Plan in a continuous latent, then
decode" is literally a PC generative model: infer a top-down latent from the evidence
(the prompt) by iterative settling, then predict the rest of the sequence under it.*

| Task | Status | Notes |
|------|:------:|-------|
| Plan-token conditioning | ❌ | a learned `plan` vector prepended as a soft position; trained end-to-end. |
| Prefix-energy settling | ❌ | at inference: freeze weights, run K gradient steps on `plan` alone to minimise the prompt's CE (settle the latent on the evidence), then decode. Reuses the autograd engine as the inner loop. |
| Ablation | ❌ | held-out continuation perplexity with settled plan vs no plan vs random plan. |

Exploratory: success = a measurable gain, not a promise. Park it if the ablation is flat.

---

## Notes

- **Deliberately skipped.** World models — the Brain project (thousand-brains demo,
  SE(3) sensor) already owns that direction. Full-scale agentic RL (LongStraw) — Phase 7
  captures the practical value at our scale; revisit only if a real reward signal shows up.
- **Order rationale.** 0–1 give the yardstick and the cheapest PC-native win; 2–4 are
  self-contained LM upgrades with hard equivalence tests; 5–6 change the architecture and
  lean on new ops; 7 needs the LM to be worth distilling into; 8 is research.
- **Mixed-precision hand-off.** Phase 4's Half KV cache and (later) routing the
  attention/FFN GEMMs through `Gemm.Provider.MatMulMixed` are the LLM-side landings of
  ROADMAP-MixedPrecision Phase 5 — the tensor-core verification on real CUDA hardware
  stays tracked *there*, not here.

## How this maps onto the stack

- **Core** (`PredicitiveCoding`): Phase 1.
- **LLM / Autograd + Training** (`Autograd/DecoderLM.cs`, `Autograd/Autograd.cs`,
  `Training/`): Phases 2, 3, 5 (ops + `TemporalBlock`), 6, 7, 8.
- **LLM / Model + Generation** (`Model/Attention.cs`, `Model/TransformerConfig.cs`,
  `Generation/TextGenerator.cs`): Phases 0 (probes), 3 (speculative), 4, 5
  (`RecurrentLayer`, preset), 6.
- **Data** (`PredictiveCoding.Data/LmStudio/LmStudioSource.cs`): Phase 7.
- **Tests** (`PredicitiveCoding.UnitTests`): every phase.
