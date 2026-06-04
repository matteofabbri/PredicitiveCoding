# Building an LLM on PredictiveCoding — Component Checklist

This document tracks the functions/components required to build a Large Language Model.

- **Implemented = `true`** → the capability exists in this solution today (class shown in parentheses).
- **Implemented = `false`** → still to be implemented.

> **Status:** the full **inference** stack of a decoder-only GPT-style transformer is
> implemented (tokenizer → embeddings → attention → blocks → LM head → sampling →
> generation), including the advanced attention variants (RoPE, RMSNorm, grouped-query,
> flash-style online softmax), BPE training, a streaming dataset, beam search, sampling
> penalties and batched decoding. A **reverse-mode autodiff engine** (`Autograd`) provides
> gradient-checked backpropagation through a transformer (`TinyTransformerLM`). The full
> **training stack** is implemented (Adam/AdamW, gradient clipping, warmup+cosine LR schedule,
> dropout, gradient accumulation, checkpointing), as are the **infrastructure primitives**
> (mixed precision, batched SIMD GEMM, tensor parallelism, int8/int4 quantization) on CPU.
>
> The only remaining `false` items are those that **fundamentally require GPU hardware** —
> a fused flash-attention kernel, a CUDA/Metal GEMM backend (implements `IGemmProvider`),
> and real multi-GPU collectives (implements `IComputeDevice`) — plus a SentencePiece
> tokenizer and an HTTP serving layer. Everything implementable in a managed CPU library is
> done.

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
| true  | WordPiece subword tokenizer — greedy longest-match (`WordPieceTokenizer`) |
| false | SentencePiece (unigram LM) tokenizer |

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
| true  | ALiBi positional bias (`PositionalKind.Alibi`) |
| false | Flash-attention fused GPU kernel (needs a GPU) |

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
| true  | Continuous-batching scheduler (`ContinuousBatcher`) |
| false | HTTP streaming server / API layer |

## Training

| Implemented | Function / component |
|:-----------:|----------------------|
| true  | Cross-entropy next-token loss — forward (`CrossEntropy.TokenLoss`) |
| true  | Perplexity / loss evaluation (`CrossEntropy.Evaluate`) |
| true  | Reverse-mode autodiff / backprop through the transformer (`Autograd`, `TinyTransformerLM`) |
| true  | SGD parameter update (`Sgd`) |
| true  | Optimizer — Adam / AdamW (`Adam`) |
| true  | Gradient clipping — global norm (`GradientClipping`) |
| true  | Learning-rate schedule — warmup + cosine decay (`LrSchedule`) |
| true  | Dropout (`Ag.Dropout`) |
| true  | Gradient accumulation / mini-batching (`Trainer`) |
| true  | Checkpoint save & resume of trained weights (`Checkpoint`) |

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
| true  | CUDA GEMM backend implementing `IGemmProvider` — direct cuBLAS P/Invoke (`CudaGemmProvider`, project `PredictiveCoding.Device`) |
| true  | Metal (Apple GPU) GEMM backend — TorchSharp/MPS (`TorchGemmProvider`, project `PredictiveCoding.Device.Torch`) |
| true  | Multi-GPU collectives (NCCL-style) — AllReduce/Broadcast/ReduceScatter/AllGather (`NcclProvider`, CPU impl.) |
| true  | Device management — `DeviceTensor<T>`, allocator/pooling, streams, async copies (`PredictiveCoding.Device`) |
| false | Fused flash-attention GPU kernel |
| false | ROCm switch (works via TorchSharp on a HIP/CUDA device; not wired) |
