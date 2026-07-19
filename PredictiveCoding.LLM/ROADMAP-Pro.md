# PredictiveLlm — Pro roadmap (from toy to professional)

> **Mission — one mandate:** take `PredictiveLlm` — the finished product of the 2026
> roadmap — from *demonstrably correct at toy scale* to **professional and powerful**:
> a real tokenizer, an O(n) serving path, real sampling and chat, training throughput,
> disciplined data/checkpoint handling, robust continual learning, and a long-term
> memory grafted from the cortical stack where the interfaces genuinely meet.

Design invariants (inherited from every previous roadmap):

- **In-house only.** No new dependencies.
- **Additive.** Defaults keep today's behaviour bit-identical; every upgrade is opt-in
  config until proven, then flipped on in `Create2026`.
- **Numbers, not vibes.** Every phase quotes before/after from the Phase-0 yardstick
  (perplexity, tokens/s, bytes).
- **Not for distribution.** Professional = internal quality (API surface, robustness,
  measurements) — no packaging, no docs-site, no publishing work. House rule.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Done** — implemented and covered by tests. |
| 🟡 | **Partial** — building block exists, needs extension. |
| ❌ | **To build** — not present yet. |

*Status as of 2026-07-19: Phases 0–7 ✅ done (real BPE tokenizer travelling with the
model, the O(n) serving path, real sampling + chat, parallel training + adaptive GEMM
routing, streaming data pipeline with mixing/hygiene/hashed split, professional
checkpointing with true resume + quantized weights, robust continual learning with
replay + surprise-lr + forgetting guardrail, episodic memory graft); Phase 8 ❌ to build.*

*Architecture map: **`ARCHITECTURE.html`** — the layered diagram from `PredictiveLlm`
down to FasterKV (product subsystems → shared tensor foundations → the PC branch),
with the honest boundary note: the LLM persists via `Checkpoint` files; FasterKV backs
the core PC network, and the two meet at Phase 7 (episodic memory).*

**Where this starts from** (all ✅, see `ROADMAP.md`): `PredictiveLlm` with hybrid
T/A blocks, looped depth, t+2 head + self-speculative decoding, FLOP-sparse MoE,
latent plan, online `Learn(text)`, distillation, save/load, HTML observatory;
GptModel inference stack with RoPE/GQA/MLA, sliding window + sinks, Half KV cache,
O(1) recurrent layers; mixed-precision core down to int8 storage (ROADMAP-MixedPrecision).

---

## The Brain / Cortex question — answered first

Matteo's question: *should the LLM use the Brain and Cortex we built?*

**Where the answer is YES — one real interface:**

- **Episodic memory (Phase 7).** `Networks/Hopfield.HopfieldMemory` and
  `Networks/Memory.MemoryNetwork` are content-addressable associative stores — exactly
  what an LM needs to remember beyond its context window. The interface is real, not
  metaphorical: the LM's final hidden state IS an embedding; store (embedding → text)
  pairs, recall by similarity at generation time, feed the recall back as context (or
  as a plan-vector prior — Phase 8 machinery). That is retrieval-augmented generation
  built from OUR memory primitives, and it is the hippocampus doing for the LM exactly
  what `Cortex/Memory.HippocampalFormation` does for the cortical stack.

**Where the answer is NO — and why, honestly:**

- **Thousand Brains / SE(3) (the Brain project).** Its power is pose-referenced
  sensorimotor voting: columns agree on *objects in space*. Language has no SE(3)
  structure; wiring the LM into it would be metaphor, not mechanism. Nothing measurable
  would improve.
- **Cortex areas as LM internals (thalamus routing, basal-ganglia gating…).** The LM is
  a dense autograd transformer; the areas are settling PC networks with per-vector
  state. Different computational regimes — grafting areas into the forward pass would
  add cost and story, not perplexity. Where a Cortex *concept* earns its keep we
  already ported the mechanism natively (MoE routing = gating, temporal blocks = the
  core's temporal state, adaptive effort = settling).
- **The INVERSE direction is the interesting one** — `PredictiveLlm` as the language
  area *of the Cortex*: Wernicke/Broca today wrap toy recurrent nets; the real LM could
  stand in. That improves the CORTEX, not the LLM, so it belongs on the Cortex roadmap
  (noted there as an opening, not a phase here).

---

## Phase 0 — A real tokenizer · ✅ done 2026-07-19

*Today: raw bytes (vocab 260) — honest but weak: ~4× longer sequences than needed and
no lexical structure. A trained subword vocabulary is the single cheapest capability
jump.*

| Task | Status | Notes |
|------|:------:|-------|
| BPE training pipeline | ✅ | `BpeTokenizer.Train` already existed (byte-level merges, deterministic — tested: two trainings agree token-for-token); wired into the product via `Create2026(tokenizerCorpus, merges)`. |
| Vocab persistence | ✅ | the tokenizer TRAVELS with the model: `Save` writes `tokenizer.bpe` + `tokenizer.json` (kind, vocab size, SHA-256 fingerprint); `Load` reconstructs it — tampered vocab, wrong vocab size, or passing a tokenizer to a BPE model all fail loudly. Legacy dirs (no manifest) fall back to bytes. |
| `Create2026` on BPE | ✅ | overload `Create2026(corpus, merges = 512, …)` trains the vocab and builds the model on it (the parameterless preset stays byte — it has no corpus to learn from). |
| Tests | ✅ | `LlmProPhase0Tests` (5): lossless round-trip incl. Unicode, determinism, >2× compression, save/load with fingerprint verification, and **bits/byte: the BPE model beats the byte model after identical training** on the yardstick corpus. |

Metric delivered: `PredictiveLlm.EvaluateBitsPerByte` — non-overlapping windows, bytes
counted on the decoded scored targets, so byte and BPE models are measured on the same
footing (per-token perplexity is incomparable across tokenizers).

## Phase 1 — The serving path: train ↔ serve, unified · ✅ done 2026-07-19

*Today's structural flaw: the trainable `DecoderLM` re-runs the WHOLE sequence per
generated token (O(n²·d)); the `GptModel` stack has the O(n·d) KV cache, sliding
window, sinks, Half cache and O(1) recurrent layers — but random weights. They never
talk. Train with one, serve with the other.*

| Task | Status | Notes |
|------|:------:|-------|
| Weight export DecoderLM → GptModel | ✅ | `Serving/GptExporter` — positional walk over `Parameters()` with double guards (per-tensor shape check + must-end-exactly), transposed onto the inference `Linear`s; GELU/positional formulas verified identical across the stacks beforehand. Parity: max logit drift < 1e-3 on a trained full-feature model (T/A + MoE + MTP + plan). |
| Supported subset, declared | ✅ | T/A pattern maps 1:1 (`RecurrentTemporalLayer`); MoE got its inference twin (`MoeFeedForward` — identical unrenormalised top-k gate semantics, FLOP-sparse for free; temporal blocks keep dense FFN on both stacks); the plan is absorbed into the embedding table (it conditioned every position identically); the t+2 head is skipped; **NumLoops > 1 is refused with a clear error** (single-pass stack). |
| `PredictiveLlm.CreateServer()` | ✅ | `Serving/LlmServer` — incremental greedy decoding, one `Step` per token through the KV cache (window/sink/Half per config); the export is a snapshot (re-create after training/Learn). Greedy text provably equal to the reforward path on a trained model. Speculative serving moves to Phase 2 with the sampling rework. |
| Throughput yardstick | ✅ | measured in-test: served path beats the full-reforward path on the wall clock at 120 tokens (structural O(n) vs O(n²), asserted). |

The highest power-per-line phase: every Phase-4 cache feature finally serves the model
we actually train. Tests: `LlmProPhase1Tests` (4).

## Phase 2 — Sampling & chat · ✅ done 2026-07-19

*Greedy only is a demo, not a product.*

| Task | Status | Notes |
|------|:------:|-------|
| Sampling in the product | ✅ | `GenerationOptions.Sampling` exposes the EXISTING `Generation.SamplingOptions`/`Sampler` (temperature, top-k, top-p, repetition/presence/frequency penalties, seed — zero duplication); null = the deterministic greedy/speculative path, bit-identical by construction; product sampling is reproducible by default (seed 42 when unset). Same knobs on the served path: `LlmServer.Generate(prompt, n, GenerationOptions)` — incremental sampling through the KV cache. |
| Speculative + sampling | ✅ | honestly greedy-only, as the task allowed: exact-match draft verification is a greedy notion; acceptance-threshold speculative sampling doesn't pay at toy scale. Sampling decodes token-by-token; the speculative path stays the greedy default. |
| `Chat()` surface | ✅ | `StartChat(system)` → `ChatSession`: role-tagged history via `ChatTemplate`, replies stop at the next role tag, caller stop sequences merge in. The self-feeding rule extends to chat: `learnFromUser` absorbs ONLY the user's words — never the formatted history, never the assistant's own turns. Window overflow throws clearly (history compression is a later phase). |
| Tests | ✅ | `LlmProPhase2Tests` (7): greedy-flag and top-k=1 ≡ default greedy, per-seed reproducibility + cross-seed variety, repetition penalty breaks the learned loop, stop sequences cut greedy AND end sampling early (forwards saved, asserted), server takes the same knobs, chat history/role-tag hygiene, learn-from-user drops the user's surprise. |

## Phase 3 — Training throughput · ✅ done 2026-07-19

*One CPU core per micro-example was the ceiling; the batch now fans across cores.*

| Task | Status | Notes |
|------|:------:|-------|
| Parallel micro-batch | ✅ | `Trainer.StepParallel` — the batch is sharded across N model replicas (independent param/grad buffers, weights synced from the primary each step), gradients merged into the primary in a FIXED worker order. **Bit-stability policy:** 1 worker ≡ serial `Step` bit-for-bit (tested); N > 1 re-associates the float accumulation → reproducible across runs (tested) but not identical to serial — the standard data-parallel trade. `TrainingOptions.Workers` (1 = serial, 0 = auto = `ProcessorCount`, capped at batch size). |
| GEMM routing by default | ✅ | `MatMulRoute.Auto` + `Ag.ProviderCrossoverMacs` (default 2²⁰): inline SIMD below the crossover (no output-buffer alloc), `Gemm.Provider` above — where a CUDA backend repays its overhead. Opt-in per run via `TrainingOptions.AutoGemmRoute` (restored after). Numerically ≡ inline with the CPU provider (tested); a real speed win only with an accelerated provider, honestly. |
| Sequence bucketing | ➖ | not needed: `StreamingTextDataset` emits fixed-length windows (all = `SeqLen`), so there is no padding waste to bucket away. Recorded as N/A rather than built. |
| Yardstick | ✅ | `LlmProPhase3Tests.ParallelTraining_ThroughputYardstick` logs serial vs parallel wall-clock. Measured (24-dim/3-block toy on a 28-core box): 1474 ms → 405 ms, **×3.64**. Below the aspirational ×4 — the toy model is small and replica sync + merge cost is fixed overhead; the ratio grows with model/batch size. Correctness (loss reduction) is the asserted part; the ratio is informational. |

## Phase 4 — Data pipeline · ✅ done 2026-07-19

*Training on `string corpus` doesn't scale past a pasted paragraph — now it feeds from disk.*

| Task | Status | Notes |
|------|:------:|-------|
| File/folder corpora | ✅ | `Data/CorpusSource` — `FromText`/`FromFile`/`FromFolder`(sorted, recursive)/`FromGlob`, a re-enumerable lazy line factory (streams; two passes for the split). `Stats` tallies lines + UTF-8 bytes as pulled. `PredictiveLlm.Train(CorpusSource)` feeds it straight into `StreamingTextDataset`. |
| Mixing & weighting | ✅ | `CorpusSource.Mix((src, weight)…)` — deterministic weighted interleave (no RNG: at each step pull from the source lowest on its emitted/weight ratio, so 80/20 stays ≈80/20 throughout; exhausted sources drop out). |
| Deterministic split | ✅ | `PerplexityEvaluator.HeldOutHashed`/`TrainingSplitHashed` — a window is held out iff FNV-1a(its tokens) ≡ 0 (mod every). Position-independent, so reorder/re-mix/grow keeps the same windows in eval (tested: reversal doesn't move the split); `Train(CorpusSource)` uses it. |
| Hygiene | ✅ | `.Deduplicated()` (64-bit content hash, memory bounded by distinct-line count) + `.WithLengthFilter(min,max)` — composable decorators. |
| Tests | ✅ | `LlmProPhase4Tests` (8): folder order + accounting, recursive glob, dedup, length filter, weighted-mix ratio, weight validation, hashed-split complement + reorder-stability, end-to-end train from a mixed/deduped folder corpus. |

## Phase 5 — Checkpointing, professionally · ✅ done 2026-07-19

| Task | Status | Notes |
|------|:------:|-------|
| Periodic + best-on-val | ✅ | `TrainingOptions.CheckpointEverySteps`/`CheckpointDir`/`KeepBestByPerplexity`: every N steps evaluate held-out ppl and (keeping the best) write a full resumable dir; `TrainingReport.BestCheckpointPath`/`BestPerplexity` record it. |
| True resume | ✅ | `OptimizerCheckpoint` serialises Adam's moments + step counter (`optimizer.bin`) next to the weights; `PredictiveLlm.LoadWithOptimizer` restores both, `TrainingOptions.ResumeOptimizer` continues from them. `Adam` gained `StepCount`/`RestoreStepCount`/`MomentsFor`/`SetMoments`. |
| Quantized checkpoints | ✅ | `Checkpoint.Save(dir, params, WeightStoragePrecision)` consumes the mixed-precision enum: Half/BFloat16 halve the file, Int8 (per-row min+scale affine — anchored to [min,max] so near-constant rows far from zero survive) quarters it; `Load` widens back to fp32. `PredictiveLlm.Save(dir, precision)` exposes it. |
| Tests | ✅ | `LlmProPhase5Tests` (7): resume ≡ uninterrupted 20-step run to 4 decimals, control that a FRESH optimizer diverges (so the equivalence is really carried by the moments), quantized reload within format tolerance (Half/bf16 5e-2, int8 1e-1), Half file < Float32, Train writes best-on-val + `LoadWithOptimizer` restores step count. |

## Phase 6 — Continual learning, robustly · ✅ done 2026-07-19

*`Learn(text)` is the PC symmetry made real — and naked. Professional means it cannot
quietly destroy the model. `Training/ContinualLearner` wraps a model with the antidotes.*

| Task | Status | Notes |
|------|:------:|-------|
| Replay mixing | ✅ | rolling ring of past windows (`ReplayCapacity`); each `Absorb` mixes `ReplayPerStep` of them into the step via a deterministic moving cursor (no RNG). Sampled BEFORE the fresh windows are pushed, so replay is always past text. |
| Surprise-proportional lr | ✅ | `lr = BaseLr · clamp(surprise / EMA(surprise), MinLrScale, MaxLrScale)` — learn hard from the unexpected, barely from the already-predicted; the weight-side twin of the core's precision-weighting. EMA seeds from the first absorb (scale 1); `SurpriseEmaDecay = 0` disables it. |
| Forgetting guardrail | ✅ | a fixed yardstick's ppl is re-measured every `GuardrailEvery` absorbs; relative drift past `DriftThreshold` fires `OnDrift(ContinualDrift)` (warn/rollback hook). Baseline measured at construction. |
| Tests | ✅ | `LlmProPhase6Tests` (4): the ablation — absorbing domain B degrades domain-A ppl markedly less WITH replay than without (and without visibly degrades); surprising text draws a higher (bounded) lr than familiar; the guardrail fires on drift; `Absorb` lowers the text's own loss and counts. |

## Phase 7 — Episodic memory (the cortical graft) · ✅ done 2026-07-19

*The YES answer to the Brain/Cortex question, as a phase — the one sanctioned reuse.*

| Task | Status | Notes |
|------|:------:|-------|
| Embedding hook | ✅ | `PredictiveLlm.Embed(text)` — mean-pool of the final hidden state over the content positions (BOS skipped, as it carries no content and only dilutes the key); `DecoderLM.HiddenStates` exposes the trunk output. |
| Hippocampal store | ✅ | `Training/EpisodicMemory` on the core's `Networks.Memory.MemoryBank` (cosine kNN, FIFO eviction — the hippocampus): `Remember(key, text)` / `Recall(query, k)`; the bank holds the payload slot id, a parallel ring keeps keys+texts as the durable trace. `Save`/`Load` (age-ordered replay reproduces the ring). `PredictiveLlm.Remember(text)` + `Memory` property; persisted with the model (`memory.bin`). |
| Recall into generation | ✅ | `GenerationOptions.UseMemory`/`MemoryTopK`: the nearest remembered texts are prepended to the prompt (most-similar first, kept while they fit the window) before decoding. No-op without a memory. |
| The PC story, tested | ✅ | `LlmProPhase7Tests` (5): the two-session test — store 6 facts, then recall each by its SETUP (all but the answer word); nearest = the full fact ≥ 66% (vs 17% chance); a matched cue outscores an unrelated query; recall-into-generation stays within the window; memory survives save/load; controlled-key FIFO eviction. **Lesson:** recall quality tracks embedding quality — an *undertrained* model's hidden states are near-degenerate and recall collapses; once the model actually fits the facts, byte-level mean-pool recall is clean (measured 6/6). |

**Scope note (per the memory verdict).** This is the ONLY Brain/Cortex graft into the LM: the
hippocampal `MemoryBank` as episodic store. No SE(3)/Thousand-Brains (no pose structure in
language) and no Cortex areas inside the forward (different compute regime); the useful cortical
ideas are already carried natively (MoE = gating, T-blocks = temporal state, effort = settling).

## Phase 8 — Pro evaluation & observability · ❌

| Task | Status | Notes |
|------|:------:|-------|
| Multi-metric eval | ❌ | bits/byte, per-domain ppl, probe tasks (echo/recall/induction — the Phase-5 probes productized), tokens/s train + serve. |
| Report v2 | ❌ | the HTML observatory grows: training curves, memory hit-rate, speculative acceptance, MoE expert load, cache bytes — still one light-theme file, no external refs. |
| Regression harness | ❌ | one command that runs the full yardstick and diffs against the last committed numbers — the "did I make it worse?" button. |

---

## Deliberately out of scope

- **Brain/SE(3) integration** — see the answer above: no shared structure with language.
- **Packaging/distribution** (NuGet, docs sites, licences) — house rule: professional
  means internal quality, not shipping artifacts.
- **RLHF-style preference training** — the on-policy distillation loop (teacher corrects
  the student's own draft) already captures the practical value at our scale; full
  reward modelling is a research program, not a phase.

## Order rationale

0–1 change what every later number means (tokenizer → the metric, serving → the speed
floor), so they go first. 2 makes it usable, 3–5 make it industrial, 6 protects the
product's signature feature (learning while used), 7 is the cortical graft once the
serving path exists to feed it, 8 wraps the whole thing in measurements.
