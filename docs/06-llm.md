# 6. LLM — `PredictiveCoding.LLM`

A complete decoder-only **Transformer / language-model** stack, built on the core
`DenseTensor`/`TensorUtils` primitives. Unlike the rest of the solution it adds a
separate **reverse-mode autodiff** engine, because Transformers need gradients.

Three roadmaps track its maturity:
[`ROADMAP.md`](../PredictiveCoding.LLM/ROADMAP.md) (predictive-coding edition),
[`ROADMAP-Pro.md`](../PredictiveCoding.LLM/ROADMAP-Pro.md) (from toy to
professional) and [`ROADMAP-Scale.md`](../PredictiveCoding.LLM/ROADMAP-Scale.md)
(from professional-at-toy-scale to a **real trained model** — the web-corpus
bootstrap below is its current phase).

## `PredictiveLlm` — the facade

The front door for everything above the layers: presets (`CreateSmall`, …) build
a tokenizer + model pair; `Save`/`Load` round-trip a whole model folder
(`config.json` + weights + tokenizer manifest); `Generate` produces text;
`CreateServer()` returns the O(n) serving side. Supporting types:
`TrainingOptions`/`TrainingReport`, `GenerationOptions`/`GenerationResult`,
`ChatSession` (multi-turn), `TokenOdometer` (token accounting),
`BitsPerByteResult`, `OnlineLearnReport`.

## Tokenization (`Tokenization/`)
- **`ByteTokenizer`** — lossless byte-level, BOS/EOS/PAD/UNK.
- **`BpeTokenizer`** — *trainable* byte-level BPE: `Train` on text or
  `TrainFromCorpus` on a `CorpusSource` with a target vocabulary (the
  pretraining path trains a 16k vocab directly on the crawled corpus).
- **`WordPieceTokenizer`** — trainable greedy longest-match subword (`##` pieces).

## Model (`Model/`)
- **`TransformerConfig`** — all hyperparameters and feature switches.
- **`Embedding`** (tied LM head), sinusoidal `PositionalEncoding`, `RotaryEmbedding`
  (RoPE), `LayerNorm`, `RmsNorm`, `Linear`.
- **`MultiHeadAttention`** — MHA / **GQA / MQA** (`NumKvHeads`), RoPE or ALiBi,
  optional **QK-Norm** and **attention soft-capping**, flash-style online softmax,
  with a `KvCache`.
- **`LatentAttention`** — **Multi-head Latent Attention (MLA, DeepSeek-V2/V3)**: the
  cache stores a compressed latent + decoupled-RoPE key instead of full K/V
  (`LatentKvCacheLayer`).
- **`RecurrentTemporalLayer`** — the temporal/recurrent block variant used by the
  trainable stack's `TemporalBlock`s.
- Feed-forward: `FeedForward` (GELU) or **`SwiGluFeedForward`** (gated SwiGLU).
- **`GptModel`** — embeds → blocks → final norm → logits, stepped one token at a
  time with a KV cache; optional **final-logit soft-capping** (Gemma 2).
- **`ModernPresets`** — `Llama3Style` (RMSNorm+SwiGLU+RoPE+GQA) and `DeepSeekStyle`
  (+ MLA + QK-Norm + soft-caps).

## Generation (`Generation/`)
- **`Sampler`** — greedy / temperature / top-k / top-p, repetition + frequency +
  presence penalties.
- **`TextGenerator`** — autoregressive loop with prefill.
- **`BeamSearch`** — length-normalised, forks KV caches.
- **`BatchedGenerator`** — N independent sequences in lockstep.
- **`ContinuousBatcher`** — in-process serving scheduler (admit on free slots, evict
  on finish).
- **`SelfSpeculative`** — self-speculative decoding (draft fast, verify once).
- **`ChatTemplate`** — role-tagged prompt formatting.

## Training (`Training/`, `Autograd/`)
The trainable path runs on a small reverse-mode autodiff engine:
- **`Autograd`** — a `Tensor` (data + grad + backward) with ops: MatMul, Transpose,
  Add/AddBias, Scale, Gelu, **Dropout**, causal SoftmaxRows, LayerNorm,
  CrossEntropy; topological `Backward()`.
- **`TinyTransformerLM`** — a trainable single-block Transformer on the engine,
  **gradient-checked** against numerical differences.
- **`DecoderLM`** (`Autograd/`) — the *full* trainable decoder (multi-block,
  multi-head, MoE feed-forward, temporal blocks) used by the translator and the
  pretraining path.
- **`Adam`/AdamW**, global-norm **`GradientClipping`**, warmup+cosine **`LrSchedule`**,
  gradient-accumulating **`Trainer`**, binary **`Checkpoint`** save/resume
  (+ **`OptimizerCheckpoint`** for Adam moments), **`MemoryBudget`** reports.
- **`CrossEntropy`** — forward loss + perplexity for evaluation.
- **`ContinualLearner` + `EpisodicMemory`** — online absorption of new text into a
  *trained* model (rehearsed against episodic replay to limit forgetting): the
  "second life" of the crawler stream.
- **`LmDistiller`** — distil a live teacher (LM Studio's OpenAI-compatible server,
  via `PredictiveCoding.Data`'s `LmStudio` source) into a sparse-MoE student.

## Serving (`Serving/`)
- **`GptExporter`** — *train on `DecoderLM`, serve on `GptModel`*: maps the trained
  weights 1:1 onto the inference stack (per-head Q/K/V/O stacking, temporal-block
  gates → `RecurrentTemporalLayer`, MoE router/experts, plan absorbed into the
  embedding table). The positional parameter walk is shape-guarded and fails
  loudly on any layout drift.
- **`LlmServer`** — the O(n) serving side: the prompt is fed once through the KV
  cache, then each new token costs exactly one forward step.

## Translation (`Translation/`) & Summarization (`Summarization/`)
- **`Translator`** + **`TranslationFormat`** + **`ParallelCorpus`** — seq2seq
  translation as prompting: a tagged prompt format with target-side loss masking
  (`IgnoreIndex`). Driven end-to-end by the
  [translator CLI](16-tools-and-cli.md), including a numerical `gradcheck` of the
  masked loss.
- **`ExtractiveSummarizer`** (+ `StopWords`, `SummaryOptions`) — extractive
  summarisation over model-agnostic text statistics.

## Evaluation (`Eval/`)
- **`PerplexityEvaluator`** (+ `EvalCorpus`) — held-out perplexity / **bits per
  byte**, the pretraining quality metric.
- **`ProEval`** / **`RegressionHarness`** — repeatable evaluation suites with
  pass/fail thresholds against a baseline.
- **`ActivationProbe`** — introspection of layer activations.
- **`HtmlReport`** — self-contained HTML evaluation reports.

## Data (`Data/`)
- **`TextDataset`** — shifted next-token windows.
- **`StreamingTextDataset`** — bounded-memory, line-buffered windows for large corpora.
- **`CorpusSource`** — the corpus abstraction the web pipeline writes into:
  `FromFolder(shards)` plus hygiene decorators
  (`.WithBoilerplateStripped(…)`, `.Deduplicated()`).

## Infrastructure (`Infrastructure/`)
CPU implementations with pluggable extension points (see [doc 7](07-device-gpu.md)):
- **Mixed precision** — `LowPrecisionMatrix` (fp16 via `System.Half`, bf16), store-16/accumulate-32.
- **`Gemm`** — SIMD + multithreaded GEMM with a pluggable `IGemmProvider`.
- **Tensor parallelism** — `ShardedLinear` (column/row-parallel) over a `DeviceMesh`.
- **Quantization** — `Int8Matrix`, packed `Int4Matrix`.

## The web-corpus pretraining pipeline

Demos **12 + 13** in `PredictiveCoding.Main` wire the whole stack together
([doc 12](12-data.md) covers the crawler side):

1. **Crawl & sediment** (demo 12) — `WebCrawler` + `ShardedCorpusWriter`
   accumulate `shard-*.txt` under `~/.predictivecoding/web-corpus`; if a trained
   model exists, every page *also* feeds its `ContinualLearner`.
2. **Hygiene** — `CorpusSource.FromFolder(…).WithBoilerplateStripped().Deduplicated()`.
3. **Bootstrap** (demo 13) — train a 16k BPE on the corpus, build the `Small`
   preset, and hand it to **`DevicePretrainer`**
   ([doc 7](07-device-gpu.md)): gradient-accumulated AdamW, warmup+cosine LR,
   periodic held-out eval + live samples, best-on-val and rolling checkpoints,
   JSONL log, graceful Ctrl-C, and **resume** (weights + device Adam moments) —
   a multi-day run that survives restarts, measured in bits/byte.

## Modern techniques and their papers

| Technique | Paper | Switch |
|-----------|-------|--------|
| Multi-head Latent Attention | DeepSeek-V3 (arXiv:2412.19437) | `AttentionKind.Latent` |
| SwiGLU | Shazeer 2020 | `FeedForwardKind.SwiGlu` |
| RMSNorm | Zhang & Sennrich 2019 | `NormKind.RmsNorm` |
| RoPE | Su et al. 2021 | `PositionalKind.Rotary` |
| QK-Norm | OLMo 2 / Gemma 2 | `UseQkNorm` |
| Logit soft-capping | Gemma 2 | `AttnLogitSoftcap`, `FinalLogitSoftcap` |
| Grouped-query attention | Llama 3 | `NumKvHeads` |

## Examples

```csharp
// A modern decoder, DeepSeek-V3 style.
var model = new GptModel(ModernPresets.DeepSeekStyle(vocab: 260), new Random(0));
var tok   = new ByteTokenizer();
string text = new TextGenerator(model, tok)
    .Generate("hello", new SamplingOptions { MaxNewTokens = 64, TopP = 0.9f });

// Train the autodiff Transformer.
var lm  = new TinyTransformerLM(vocab: 5, dim: 8, ff: 16, new Random(0), dropout: 0.1f);
var opt = new Adam(weightDecay: 0.01f);
float loss = Trainer.Step(lm, batch, opt, learningRate: 3e-4f, clipNorm: 1f);
```
