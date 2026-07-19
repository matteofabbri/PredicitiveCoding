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

*Status as of 2026-07-19: plan drafted, nothing started.*

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

## Phase 0 — A real tokenizer · ❌

*Today: raw bytes (vocab 260) — honest but weak: ~4× longer sequences than needed and
no lexical structure. A trained subword vocabulary is the single cheapest capability
jump.*

| Task | Status | Notes |
|------|:------:|-------|
| BPE training pipeline | ❌ | `BpeTokenizer` exists — give it a corpus-in/vocab-out trainer (target 4k–16k merges), deterministic. |
| Vocab persistence | ❌ | tokenizer saved WITH the model (`Save`/`Load` round-trips it; `config.json` records the tokenizer kind + vocab hash so a mismatch fails loudly). |
| `Create2026` on BPE | ❌ | preset flips from bytes to a trained vocab once ppl/byte beats the byte baseline on the yardstick. |
| Tests | ❌ | encode/decode round-trip lossless; ppl **per byte** (not per token) beats byte-level on the yardstick corpus; save/load mismatch detection. |

Metric note: with a new tokenizer, perplexity-per-token is incomparable — the yardstick
becomes **bits per byte**, which the eval harness must learn to report.

## Phase 1 — The serving path: train ↔ serve, unified · ❌

*Today's structural flaw: the trainable `DecoderLM` re-runs the WHOLE sequence per
generated token (O(n²·d)); the `GptModel` stack has the O(n·d) KV cache, sliding
window, sinks, Half cache and O(1) recurrent layers — but random weights. They never
talk. Train with one, serve with the other.*

| Task | Status | Notes |
|------|:------:|-------|
| Weight export DecoderLM → GptModel | ❌ | map embedding/blocks/norms/head onto the inference stack; parity test: same tokens, same logits within tolerance. |
| Supported subset, declared | ❌ | pattern (T/A) maps 1:1 (`RecurrentTemporalLayer` exists); MoE FFN needs an inference-side expert FFN in `GptModel` (same weights, no autograd — cheap); loops & plan: export unrolled/absorbed or documented out. |
| `PredictiveLlm.Generate` on the served model | ❌ | incremental decoding with KV cache + window + sinks + Half cache; self-speculative stays (draft with t+2 head, verify incrementally). |
| Throughput yardstick | ❌ | tokens/s toy-bench: full-reforward vs served path (expect the asymptotic win to show by ~200 tokens). |

The highest power-per-line phase: every Phase-4 cache feature finally serves the model
we actually train.

## Phase 2 — Sampling & chat · ❌

*Greedy only is a demo, not a product.*

| Task | Status | Notes |
|------|:------:|-------|
| Sampling in the product | ❌ | temperature, top-k, top-p, repetition penalty in `GenerationOptions` (the `Sampling` helpers exist on the GptModel side — port/share). Greedy stays the deterministic default. |
| Speculative + sampling | ❌ | drafts verified by acceptance-threshold instead of exact match (standard speculative sampling) — or honestly documented greedy-only if the math doesn't pay at our scale. |
| `Chat()` surface | ❌ | history + system turn via `ChatTemplate`, stop sequences, per-turn `LearnFromPrompt` opt-in (learn from the user, never from the assistant turn — the self-feeding rule extended to chat). |
| Tests | ❌ | temperature 0 ≡ greedy bit-for-bit; top-k/p bounds; stop sequences; chat template round-trip. |

## Phase 3 — Training throughput · ❌

*One CPU core per micro-example is the current ceiling.*

| Task | Status | Notes |
|------|:------:|-------|
| Parallel micro-batch | ❌ | `Trainer.Step` accumulates per-example grads — run examples across cores (thread-local grad buffers, one merge; bit-stability policy documented: deterministic reduction order). |
| GEMM routing by default | ❌ | flip `Ag.MatMulRoute` to `Gemm.Provider` where the Phase-4 bench shows a win (big dims), keep inline below the crossover; CUDA provider when present. |
| Sequence bucketing | ❌ | batch windows of equal length to kill padding waste in the batched path. |
| Yardstick | ❌ | tokens/s (train) before/after on the yardstick corpus; target ≥ 4× on an 8-core box. |

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
