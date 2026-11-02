# 1. Overview

## What this is

A layered research solution that takes a single idea — **predictive coding (PC)**
— and builds increasingly ambitious systems on top of it, all in managed .NET 10
with no Python and (by default) no GPU dependency: brain-inspired architectures,
a model of the visual cortex, a Thousand-Brains sensorimotor system, a full
Transformer/LLM stack trained on a self-collected web corpus, an in-house ONNX
runtime, and applied computer-vision stacks (people counting, face recognition).

Everything rests on one engine: a hierarchical network whose neurons hold a
latent state `μ`, generate top-down predictions, and update both states and
weights by **local** rules that minimise prediction error. There is no global
backpropagation in the core (the LLM project adds a separate autodiff engine for
the Transformer training path).

## The projects and how they stack

```
APPS          PredictiveCoding.Main            13-option demo menu (CIFAR, LFW, LLM, …)
              PredictiveCoding.Translate.Cli   build / verify / run a seq2seq translator
              SAMPLES/PredictiveCoding.CortexApp   WinUI 3 visualiser
              tools/ : DetectorDistiller · OnnxPack · MixedPrecisionBench
                                 │
SYSTEMS       PredictiveCoding.Brain           Thousand Brains (columns + voting)
                 └─► PredictiveCoding.Cortex      visual cortex + cortical areas
                        └─► PredictiveCoding.LLM  (language areas on PredictiveLlm)
              PredictiveCoding.Vision          people counting: detect · track · count
                 └─► PredictiveCoding.FaceRecognition   faces + PC distillation
                                 │
FOUNDATIONS   PredicitiveCoding (core)         PC engine + ALL advanced architectures
              PredictiveCoding.LLM             Transformer + autodiff + training
              PredictiveCoding.Data            45+ data sources → tensors / text
              PredictiveCoding.Onnx            in-house ONNX runtime + PCNX format
              PredictiveCoding.Imaging         zero-dependency RGB image layer
              PredictiveCoding.Device          device management + CUDA (cuBLAS)
```

### Dependency rules (as wired in the `.csproj`s)

- **Core depends on nothing** in the solution (only NuGet: FASTER KV,
  System.Numerics.Tensors). The advanced architectures (Hopfield, MoE, Spiking,
  Graph, Hierarchical, Active, …) live **inside the core** under
  `PredicitiveCoding/Networks/*` — the former `PredictiveCoding.Expanded`
  project was merged into it.
- **Dependency-free leaves**: `PredictiveCoding.Onnx` (no packages at all) and
  `PredictiveCoding.Imaging` (pure-managed Stb codecs).
- **LLM → core.** **Cortex → core + LLM** (Wernicke/Broca can run on
  `PredictiveLlm`; no cycle — both rest on the core only).
  **Brain → core + Cortex.**
- **Device → core + LLM** (registers its CUDA `IGemmProvider` into the LLM's
  GEMM hook). **Data → core + Imaging.**
- **FaceRecognition → core + Onnx + Imaging** (the `OnnxRuntime.Gpu` package is
  there for speed, with the in-house executor as an opt-in replacement);
  **Vision → FaceRecognition + Onnx + Imaging**.
- **UnitTests** references core, Cortex, Brain, LLM, Device, Data, Vision, Onnx
  and Imaging. No cycles anywhere; accelerators and heavy runtimes plug in
  through interfaces (`IGemmProvider`, `IFaceEmbedder`, `IHtmlSource`) so models
  never reference a specific backend.

## What each project adds

| Project | Adds |
|---------|------|
| **PredicitiveCoding** (core) | `DenseTensor`, the `Layer`/`Network` PC engine, FASTER-KV weight persistence, a `Trainer`, mixed-precision inference — and every advanced architecture: Hierarchical, Active inference, DeepRecurrent, Gated memory, Hopfield, Graph, Hierarchical MoE, Spiking, Generative, Memory, MultiModal, Routed, Attentive, Ensemble, MoE, Residual, Temporal ([doc 4](04-expanded.md)). |
| **PredictiveCoding.LLM** | Decoder-only Transformer (tokenizers, MLA/GQA attention, KV cache, sampling, beam search, continuous batching), reverse-mode autodiff trainer (Adam, schedules, checkpoints), serving/translation/summarization, continual learning + distillation, and the CUDA pretraining pipeline ([doc 6](06-llm.md)). |
| **PredictiveCoding.Cortex** | A predictive-coding **visual cortex** (V1→V2→V4→IT) and models of the other brain regions — sensory, memory/MTL, limbic, executive, control, language, neuromodulation ([doc 5](05-cortex.md)). |
| **PredictiveCoding.Brain** | The **Thousand Brains** sensorimotor system: SDRs, HTM sequence memory, reference frames & path integration, voting columns, heterarchy ([doc 11](11-brain.md)). |
| **PredictiveCoding.Data** | 45+ data sources (media, tabular, documents, web APIs, knowledge bases) → `DenseTensor` or text; the polite web crawler + sharded corpus writer ([doc 12](12-data.md)). |
| **PredictiveCoding.Onnx** | In-house ONNX runtime: protobuf parser, kernels, graph optimiser, PCNX save format — no ONNX Runtime, no protobuf package ([doc 13](13-onnx.md)). |
| **PredictiveCoding.Vision** | People counting on live feeds/video: YOLOX detection (ONNX), SORT tracking, line/zone counting, re-identification ([doc 10](10-people-counting-roadmap.md)). |
| **PredictiveCoding.FaceRecognition** | The face pipeline — detection, alignment, embedders (LBP, PC, ArcFace), LFW benchmarks — with **PC distillation** of the deep teacher ([doc 15](15-face-recognition.md)). |
| **PredictiveCoding.Imaging** | `RgbImage` + codecs: managed RGB pixel I/O with no native binaries ([doc 14](14-imaging.md)). |
| **PredictiveCoding.Device** | `DeviceTensor<T>`, allocator, streams, cuBLAS P/Invoke GEMM, device-side LLM pretraining ([doc 7](07-device-gpu.md)). |
| **Main · Translate.Cli · CortexApp · tools** | The runnable surfaces — demos, translator CLI, WinUI app, distiller/converter/benchmark tools ([doc 9](09-build-and-run.md), [doc 16](16-tools-and-cli.md)). |

## Design principles

- **Locality** — core learning is Hebbian; no backprop through the PC graph.
  Where gradients are unavoidable (the Transformer), they live in a separate,
  clearly-bounded autodiff engine.
- **Composition** — higher modules wrap lower ones (`VisualCortex` = shared V1
  column + Hierarchical network; `ThousandBrainsSystem` = heterarchy of voting
  columns); the cortical areas and Brain phases mostly compose core networks.
- **Pluggable acceleration & runtimes** — `IGemmProvider` / `IComputeDevice`
  let CPU or CUDA slot in without touching model code; the same face/vision
  models run on ONNX Runtime *or* the in-house executor behind one flag.
- **Teachers, then no teachers** — heavy pretrained models (ArcFace, YOLOX,
  LM Studio) are used as *offline* labelers and distilled into in-house PC /
  linear students, so run-time dependencies shrink instead of growing.
- **SIMD everywhere** — tensor ops use `System.Numerics.Tensors` /
  `Vector<float>`, with optional `Half`-weight mixed-precision paths.
