# 1. Overview

## What this is

A layered research solution that takes a single idea — **predictive coding (PC)**
— and builds increasingly ambitious systems on top of it, all in managed .NET 10
with no Python and (by default) no GPU dependency.

Everything rests on one engine: a hierarchical network whose neurons hold a latent
state `μ`, generate top-down predictions, and update both states and weights by
**local** rules that minimise prediction error. There is no global
backpropagation in the core (the LLM project adds a separate autodiff engine for
the Transformer training path).

## The projects and how they stack

```
                       ┌─────────────────────────────┐
                       │  PredicitiveCoding (core)    │  Network, Layer, TensorUtils
                       └──────────────┬──────────────┘
                ┌─────────────────────┼───────────────────────┐
                ▼                     ▼                        ▼
   ┌──────────────────────┐  ┌────────────────┐   ┌────────────────────────┐
   │ PredictiveCoding.     │  │ PredictiveCoding│   │ PredictiveCoding.LLM    │
   │ Expanded              │  │ .Main (demos)  │   │ (Transformer / autodiff)│
   │ (brain architectures) │  └────────────────┘   └───────────┬────────────┘
   └──────────┬───────────┘                                    │
              ▼                                                 ▼
   ┌──────────────────────┐                        ┌────────────────────────┐
   │ PredictiveCoding.     │                        │ PredictiveCoding.Device │ CUDA (cuBLAS)
   │ Cortex                │                        │  └ Device.Torch         │ Metal (TorchSharp)
   │ (visual cortex + 14   │                        └────────────────────────┘
   │  cortical areas)      │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ PredictiveCoding.     │  WinUI 3 demo
   │ CortexApp             │
   └──────────────────────┘

   PredicitiveCoding.UnitTests references core, Expanded, Cortex, LLM, Device(.Torch)
```

### Dependency rules
- **Core depends on nothing** in the solution (only NuGet: FASTER KV, System.Numerics.Tensors).
- **Expanded → core.** Cortex → Expanded → core. CortexApp → Cortex.
- **LLM → core.** Device → LLM + core (to register its CUDA `IGemmProvider`).
  Device.Torch → Device.
- No cycles; accelerators plug in through interfaces (`IGemmProvider`,
  `IComputeDevice`) so models never reference a specific backend.

## What each layer adds

| Layer | Adds |
|-------|------|
| **Core** | `DenseTensor`, the `Layer`/`Network` PC engine, FASTER-KV weight persistence, a `Trainer`. |
| **Expanded** | 12+ architectures: deep modular hierarchies, active inference, temporal cascades, gated working memory, modern Hopfield memory, graph PC, hierarchical MoE, spiking PC, plus generative/multimodal/memory/routed nets and the relocated Attentive/Ensemble/MoE/Residual/Temporal networks. |
| **Cortex** | A predictive-coding **visual cortex** (V1→V2→V4→IT) and 14 models of other brain regions (auditory, somatosensory, dorsal stream, retina, hippocampus, entorhinal grid cells, piriform, PFC, motor, cerebellum, association, thalamus, basal ganglia, default-mode). |
| **CortexApp** | A WinUI 3 desktop app that trains and visualises several of these models. |
| **LLM** | A complete decoder-only Transformer (tokenizers, attention, KV cache, sampling, beam search, batching), a reverse-mode **autodiff** trainer (Adam, dropout, schedules, checkpoints), modern techniques (MLA, SwiGLU, QK-Norm, soft-capping), and infrastructure (mixed precision, GEMM, tensor parallelism, quantization). |
| **Device / Device.Torch** | `DeviceTensor<T>`, allocator, streams, and pluggable GPU backends: **CUDA** via direct cuBLAS P/Invoke, **Metal** via TorchSharp/MPS. |

## Design principles
- **Locality** — core learning is Hebbian; no backprop through the PC graph.
- **Composition** — higher modules wrap lower ones (e.g. `VisualCortex` = shared V1
  column + `HierarchicalNetwork`); the cortical areas mostly compose Expanded nets.
- **Pluggable acceleration** — `IGemmProvider` / `IComputeDevice` let a CPU, CUDA or
  Metal backend slot in without touching model code; everything falls back to CPU.
- **SIMD everywhere** — tensor ops use `System.Numerics.Tensors` / `Vector<float>`.
