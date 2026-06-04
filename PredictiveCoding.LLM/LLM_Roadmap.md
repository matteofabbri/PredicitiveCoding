# Building an LLM on PredictiveCoding — Component Checklist

This document tracks the functions/components required to build a Large Language Model.

- **Implemented = `true`** → the capability exists in this solution today (class shown in parentheses).
- **Implemented = `false`** → still to be implemented.

> **Status:** the full **inference** stack of a decoder-only GPT-style transformer is
> implemented (tokenizer → embeddings → attention → blocks → LM head → sampling →
> generation), including the advanced attention variants (RoPE, RMSNorm, grouped-query,
> flash-style online softmax), BPE training, a streaming dataset, beam search, sampling
> penalties and batched decoding. A **reverse-mode autodiff engine** (`Autograd`) provides
> gradient-checked backpropagation through a transformer (`TinyTransformerLM`) with a plain
> SGD step. The infrastructure primitives (mixed precision, batched SIMD GEMM, tensor
> parallelism, int8/int4 quantization) are implemented on CPU with a pluggable GEMM backend
> for a future GPU provider. The main remaining gap is the **production training stack** —
> a real optimizer (Adam/AdamW), LR schedules, dropout, gradient clipping and checkpointing.

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
| true  | Trainable subword tokenizer — byte-level BPE (`BpeTokenizer`) |
| true  | Streaming loader for large corpora — bounded memory (`StreamingTextDataset`) |
| false | WordPiece / SentencePiece tokenizers |

## Model (transformer)

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | Token embedding table (`Embedding`) |
| true  | Positional encoding — sinusoidal (`PositionalEncoding`) |
| true  | Rotary positional encoding — RoPE (`RotaryEmbedding`, `PositionalKind.Rotary`) |
| true  | Linear / dense projection (`Linear`) |
| true  | Scaled dot-product attention (`MultiHeadAttention`) |
| true  | Multi-head attention (`MultiHeadAttention`) |
| true  | Grouped-query / multi-query attention (`TransformerConfig.NumKvHeads`) |
| true  | Flash-attention online softmax — single pass (`UseFlashAttention`) |
| true  | Causal masking (implicit via KV cache) |
| true  | Layer normalization (`LayerNorm`) |
| true  | RMSNorm (`RmsNorm`, `NormKind.RmsNorm`) |
| true  | Feed-forward block with GELU (`FeedForward`) |
| true  | Transformer block — pre-norm + residual (`TransformerBlock`) |
| true  | Stacked decoder + final norm + LM head (`GptModel`) |
| true  | Weight tying — embedding ↔ LM head (`TransformerConfig.TieEmbeddings`) |
| false | ALiBi positional bias |
| false | Flash-attention fused GPU kernel |

## Generation / decoding

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | KV cache for incremental decoding (`KvCache`) |
| true  | Autoregressive generation loop (`TextGenerator`) |
| true  | Sampling — greedy, temperature, top-k, top-p (`Sampler`) |
| true  | Repetition penalty (`SamplingOptions.RepetitionPenalty`) |
| true  | Frequency / presence penalties (`SamplingOptions`) |
| true  | Beam search (`BeamSearch`) |
| true  | Batched decoding (`BatchedGenerator`) |
| true  | Context-window enforcement (`MaxSeqLen`) |
| true  | Prompt / chat template formatting (`ChatTemplate`) |
| false | Serving layer — continuous batching / streaming server |

## Training

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | Cross-entropy next-token loss — forward (`CrossEntropy.TokenLoss`) |
| true  | Perplexity / loss evaluation (`CrossEntropy.Evaluate`) |
| true  | Reverse-mode autodiff / backprop through the transformer (`Autograd`, `TinyTransformerLM`) |
| true  | SGD parameter update (`Sgd`) |
| false | Optimizer — Adam / AdamW |
| false | Gradient clipping |
| false | Learning-rate schedule — warmup + cosine decay |
| false | Dropout |
| false | Gradient accumulation / mini-batching |
| false | Checkpoint save & resume of trained weights |

## Infrastructure

> Implemented on CPU (managed .NET): SIMD-vectorised + multithreaded, with a
> pluggable GEMM backend and a device-mesh abstraction where a GPU/multi-GPU
> backend would slot in. The algorithms are identical to the accelerated versions.

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | Mixed precision — fp16 / bf16 weight storage, fp32 accumulate (`LowPrecisionMatrix`, `BFloat16`) |
| true  | Batched / SIMD-parallel GEMM with pluggable backend (`Gemm`, `IGemmProvider`) |
| true  | Multi-device / tensor parallelism — column/row-parallel linear + all-reduce (`ShardedLinear`, `DeviceMesh`) |
| true  | Quantization — int8 and packed int4 weight matrices (`Int8Matrix`, `Int4Matrix`) |
| false | GPU / Metal GEMM backend implementing `IGemmProvider` |
| false | Real cross-device collectives (NCCL-style) for multi-GPU |
