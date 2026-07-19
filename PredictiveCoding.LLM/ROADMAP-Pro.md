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

*Status as of 2026-07-19: Phases 0–3 ✅ done (real BPE tokenizer travelling with the
model, the O(n) serving path, real sampling + chat, parallel training + adaptive GEMM
routing); Phases 4–8 ❌ to build.*

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

## Phase 4 — Data pipeline · ❌

*Training on `string corpus` doesn't scale past a pasted paragraph.*

| Task | Status | Notes |
|------|:------:|-------|
| File/folder corpora | ❌ | `Train(CorpusSource)`: files, folders, glob — streaming (the dataset already streams), UTF-8, size accounting. |
| Mixing & weighting | ❌ | multiple sources with sampling weights (e.g. 80% domain, 20% yardstick for stability). |
| Deterministic split | ❌ | held-out selection stable across runs (hash-based, not index-based) so ppl numbers are comparable between sessions. |
| Hygiene | ❌ | line dedup + length filters — cheap, measurable on held-out ppl. |

## Phase 5 — Checkpointing, professionally · ❌

| Task | Status | Notes |
|------|:------:|-------|
| Periodic + best-on-val | ❌ | save every N steps and keep the best held-out ppl; `TrainingReport` records the path. |
| True resume | ❌ | Adam moments serialized next to the weights — resume continues, not restarts (today's `Load` restarts the optimizer). |
| Quantized checkpoints | ❌ | weights persisted Half/int8 via the mixed-precision Phase-6/7 machinery (`WeightStoragePrecision`) — the LLM side finally consumes it. |
| Tests | ❌ | kill/resume ≡ uninterrupted run (same final loss within tolerance); quantized reload parity within format tolerance. |

## Phase 6 — Continual learning, robustly · ❌

*`Learn(text)` is the PC symmetry made real — and naked. Professional means it cannot
quietly destroy the model.*

| Task | Status | Notes |
|------|:------:|-------|
| Replay mixing | ❌ | each `Learn` step mixes k replay windows from a rolling buffer of past text — the standard forgetting antidote, cheap at our scale. |
| Surprise-proportional lr | ❌ | the PC move: scale the step by how surprising the text was (bounded) — learn hard from the unexpected, barely from the known. Precision-weighting, weight-side. |
| Forgetting guardrail | ❌ | automatic probe: held-out yardstick ppl re-measured after every N `Learn` calls; drift beyond a threshold → warn/rollback hook. |
| Tests | ❌ | absorb a new domain WITHOUT the yardstick ppl degrading beyond ε (with replay) vs degrading (without) — the ablation that proves the guardrail. |

## Phase 7 — Episodic memory (the cortical graft) · ❌

*The YES answer to the Brain/Cortex question, as a phase.*

| Task | Status | Notes |
|------|:------:|-------|
| Embedding hook | ❌ | `PredictiveLlm.Embed(text)` — mean/last final-hidden state as the retrieval key (already computed by every forward). |
| Hippocampal store | ❌ | `EpisodicMemory` wrapping `Networks/Hopfield.HopfieldMemory` / `Memory.MemoryBank`: `Remember(text)` (embed → store with the text as payload), `Recall(query, k)` (similarity → texts). Persistence with the model. |
| Recall into generation | ❌ | `GenerationOptions.UseMemory`: recall top-k for the prompt, prepend as context (window permitting) or absorb as a plan prior (Phase-8 machinery) — measured, both ways. |
| The PC story, tested | ❌ | ablation on a two-session task: facts stated in session 1, recalled in session 2 — with memory ≥ hit-rate target, without ≈ chance. Chat's `Remember` opt-in per turn. |

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
