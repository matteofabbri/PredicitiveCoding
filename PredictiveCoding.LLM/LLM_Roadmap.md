# Building an LLM on PredictiveCoding — Component Checklist

This document tracks the functions/components required to build a Large Language Model.

- **Implemented = `true`** → the capability exists in this solution today (class shown in parentheses).
- **Implemented = `false`** → still to be implemented.

> **Status:** the full **inference** stack of a decoder-only GPT-style transformer is
> implemented in `PredictiveCoding.LLM` (tokenizer → embeddings → attention with KV-cache →
> transformer blocks → LM head → sampling → autoregressive generation), plus evaluation and a
> dataset loader. Weights are randomly initialised, so generation is untrained until a trainer
> exists. **Gradient training (backprop + optimizer) is the main remaining gap** — the
> surrounding PredictiveCoding library is backprop-free, so the trainer needs its own autodiff
> or a predictive-coding learning rule.

## Tensor & predictive-coding building blocks (from the core library)

| Implemented | Function / component |
|:-----------:|----------------------|
| true | Dense tensor ops — matmul, outer product, axpy (`TensorUtils`) |
| true | Activation functions — Tanh, ReLU, Sigmoid, Softmax (`ActivationFn`) |
| true | Weight initialization — Xavier / He (`RandomUtils`) |
| true | Mixture-of-Experts sparse routing (`MixtureOfExperts` / `HierarchicalMixtureOfExperts`) |
| true | Associative / episodic memory, key-value retrieval (`MemoryNetwork` / `HopfieldNetwork`) |
| true | Weight checkpoint / persistence (`LayerStore`, FasterKV) |

## Tokenization & data

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | Byte-level tokenizer (`ByteTokenizer`) |
| true  | Detokenizer / text decoder (`ByteTokenizer.Decode`) |
| true  | Special tokens — BOS / EOS / PAD / UNK (`ITokenizer`) |
| true  | Text corpus dataset + shifted next-token windows (`TextDataset`) |
| false | Subword tokenizer — BPE / WordPiece / SentencePiece |
| false | Streaming loader for large corpora (memory-mapped / sharded) |

## Model (transformer)

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | Token embedding table (`Embedding`) |
| true  | Positional encoding — sinusoidal (`PositionalEncoding`) |
| true  | Linear / dense projection (`Linear`) |
| true  | Scaled dot-product attention (`MultiHeadAttention`) |
| true  | Multi-head attention (`MultiHeadAttention`) |
| true  | Causal masking (implicit via KV cache) |
| true  | Layer normalization (`LayerNorm`) |
| true  | Feed-forward block with GELU (`FeedForward`) |
| true  | Transformer block — pre-norm + residual (`TransformerBlock`) |
| true  | Stacked decoder + final norm + LM head (`GptModel`) |
| true  | Weight tying — embedding ↔ LM head (`TransformerConfig.TieEmbeddings`) |
| false | Rotary / ALiBi positional encoding (RoPE) |
| false | RMSNorm |
| false | Grouped-query / multi-query attention |
| false | Flash-attention style fused kernels |

## Generation / decoding

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | KV cache for incremental decoding (`KvCache`) |
| true  | Autoregressive generation loop (`TextGenerator`) |
| true  | Sampling — greedy, temperature, top-k, top-p (`Sampler`) |
| true  | Repetition penalty (`SamplingOptions.RepetitionPenalty`) |
| true  | Context-window enforcement (`MaxSeqLen`) |
| true  | Prompt / chat template formatting (`ChatTemplate`) |
| false | Beam search |
| false | Frequency / presence penalties |
| false | Streaming server / batched decoding |

## Training

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | Cross-entropy next-token loss — forward (`CrossEntropy.TokenLoss`) |
| true  | Perplexity / loss evaluation (`CrossEntropy.Evaluate`) |
| false | Backpropagation / autodiff through the transformer |
| false | Optimizer — Adam / AdamW |
| false | Gradient clipping |
| false | Learning-rate schedule — warmup + cosine decay |
| false | Dropout |
| false | Gradient accumulation / batching |
| false | Checkpoint save & resume of model weights |

## Infrastructure

| Implemented | Function / component |
|:-----------:|----------------------|
| false | Mixed precision — fp16 / bf16 |
| false | Batched / GPU matmul — GEMM (CUDA / Metal) |
| false | Multi-device / tensor parallelism |
| false | Quantization (int8 / int4) for inference |
