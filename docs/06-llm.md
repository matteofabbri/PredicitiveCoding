# 6. LLM — `PredictiveCoding.LLM`

A complete decoder-only **Transformer / language-model** stack, built on the core
`DenseTensor`/`TensorUtils` primitives. Unlike the rest of the solution it adds a
separate **reverse-mode autodiff** engine, because Transformers need gradients.
A living feature checklist lives in
[`PredictiveCoding.LLM/LLM_Roadmap.md`](../PredictiveCoding.LLM/LLM_Roadmap.md).

## Tokenization (`Tokenization/`)
- **`ByteTokenizer`** — lossless byte-level, BOS/EOS/PAD/UNK.
- **`BpeTokenizer`** — *trainable* byte-level BPE (learns merges, greedy encode).
- **`WordPieceTokenizer`** — trainable greedy longest-match subword (`##` pieces).

## Model (`Model/`)
- **`TransformerConfig`** — all hyperparameters and feature switches.
- **`Embedding`** (tied LM head), sinusoidal `PositionalEncoding`, `RotaryEmbedding`
  (RoPE), `LayerNorm`, `RmsNorm`, `Linear`.
- **`MultiHeadAttention`** — MHA / **GQA / MQA** (`NumKvHeads`), RoPE or ALiBi,
  optional **QK-Norm** and **attention soft-capping**, flash-style online softmax,
  with a `KvCache`.
- **`LatentAttention`** — **Multi-head Latent Attention (MLA, DeepSeek-V2/V3)**: the
  cache stores a compressed latent + decoupled-RoPE key instead of full K/V.
- **Feed-forward**: `FeedForward` (GELU) or **`SwiGluFeedForward`** (gated SwiGLU).
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
- **`ChatTemplate`** — role-tagged prompt formatting.

## Training (`Training/`, `Autograd/`)
The trainable path runs on a small reverse-mode autodiff engine:
- **`Autograd`** — a `Tensor` (data + grad + backward) with ops: MatMul, Transpose,
  Add/AddBias, Scale, Gelu, **Dropout**, causal SoftmaxRows, LayerNorm,
  CrossEntropy; topological `Backward()`.
- **`TinyTransformerLM`** — a trainable single-block Transformer on the engine,
  **gradient-checked** against numerical differences.
- **`Adam`/AdamW**, global-norm **`GradientClipping`**, warmup+cosine **`LrSchedule`**,
  gradient-accumulating **`Trainer`**, binary **`Checkpoint`** save/resume.
- **`CrossEntropy`** — forward loss + perplexity for evaluation.

## Data (`Data/`)
- **`TextDataset`** — shifted next-token windows.
- **`StreamingTextDataset`** — bounded-memory, line-buffered windows for large corpora.

## Infrastructure (`Infrastructure/`)
CPU implementations with pluggable extension points (see [doc 7](07-device-gpu.md)):
- **Mixed precision** — `LowPrecisionMatrix` (fp16 via `System.Half`, bf16), store-16/accumulate-32.
- **`Gemm`** — SIMD + multithreaded GEMM with a pluggable `IGemmProvider`.
- **Tensor parallelism** — `ShardedLinear` (column/row-parallel) over a `DeviceMesh`.
- **Quantization** — `Int8Matrix`, packed `Int4Matrix`.

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

Still on the roadmap: a Mixture-of-Experts FFN (DeepSeekMoE), Multi-Token
Prediction, a fused flash-attention GPU kernel, and a SentencePiece tokenizer.

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

> **Caveat.** `GptModel` weights are randomly initialised (the engine is
> energy-/inference-based); meaningful text needs the autodiff trainer. The
> existing tests assert shapes, finiteness and gradient-correctness.
