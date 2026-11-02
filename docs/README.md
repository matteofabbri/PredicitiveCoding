# PredictiveCoding — Documentation

A .NET 10 / C# research codebase built around **predictive coding** (Rao & Ballard
1999; Friston 2005): a backprop-free, biologically-plausible alternative to
standard deep learning where perception and learning are framed as local
free-energy minimisation. On top of the core PC engine the solution grows into
brain-inspired architectures, a model of the visual cortex, a **Thousand Brains
sensorimotor system** (implementing the Hawkins/Numenta papers), a full
Transformer/LLM stack with its own data pipeline, an in-house ONNX runtime, and
applied vision stacks (people counting, face recognition).

## Documentation map

| # | Document | What it covers |
|---|----------|----------------|
| 1 | [Overview](01-overview.md) | The solution, its projects and how they depend on each other |
| 2 | [Predictive coding theory](02-predictive-coding.md) | The maths: free energy, inference, local Hebbian learning |
| 3 | [Core library](03-core.md) | `PredictiveCoding` — tensors, `Layer`, `Network`, `Trainer` |
| 4 | [Advanced PC architectures](04-expanded.md) | The brain-inspired networks (Hopfield, MoE, Spiking, Graph, …) — now in the core's `Networks/` |
| 5 | [Cortex](05-cortex.md) | `PredictiveCoding.Cortex` — visual cortex + cortical areas, WinUI demo |
| 6 | [LLM](06-llm.md) | `PredictiveCoding.LLM` — Transformer, autodiff training, serving/translation, web-corpus pretraining |
| 7 | [Device & GPU](07-device-gpu.md) | `PredictiveCoding.Device` — CPU/CUDA acceleration, device-side training |
| 8 | [Testing](08-testing.md) | The xUnit suite and its conventions |
| 9 | [Build & run](09-build-and-run.md) | Building, testing, running the demos and the GPU paths |
| 10 | [Live people counting](10-people-counting-roadmap.md) | `PredictiveCoding.Vision` — detect (YOLOX) · track (SORT) · count people in a live feed |
| 11 | [Brain](11-brain.md) | `PredictiveCoding.Brain` — the Thousand Brains sensorimotor system (columns, reference frames, voting) |
| 12 | [Data](12-data.md) | `PredictiveCoding.Data` — 45+ data sources, the web crawler, corpus sedimentation, language detection |
| 13 | [Onnx](13-onnx.md) | `PredictiveCoding.Onnx` — the in-house ONNX runtime and PCNX format |
| 14 | [Imaging](14-imaging.md) | `PredictiveCoding.Imaging` — the zero-dependency RGB image layer |
| 15 | [Face recognition](15-face-recognition.md) | `PredictiveCoding.FaceRecognition` — LFW benchmark, embedders, PC distillation |
| 16 | [Tools & CLIs](16-tools-and-cli.md) | `DetectorDistiller`, `OnnxPack`, `MixedPrecisionBench`, the translator CLI |

## At a glance

```
PredicitiveCoding/                 core PC library: engine + all advanced architectures (net10)
PredictiveCoding.Brain/            Thousand Brains sensorimotor system (net10)
PredictiveCoding.Cortex/           visual cortex + cortical areas (net10)
PredictiveCoding.LLM/              Transformer / LLM stack (net10)
PredictiveCoding.Device/           device management + CUDA backend (net10)
PredictiveCoding.Data/             45+ data sources → tensors / text (net10)
PredictiveCoding.Onnx/             in-house ONNX runtime + PCNX format (net10)
PredictiveCoding.FaceRecognition/  face pipeline + PC distillation (net10)
PredictiveCoding.Vision/           people counting: detect · track · count (net10)
PredictiveCoding.Imaging/          zero-dependency RGB image layer (net10)
PredictiveCoding.Main/             demo entry point — 13 options (net10)
PredictiveCoding.Translate.Cli/    seq2seq translator CLI (net10, SAMPLES)
SAMPLES/PredictiveCoding.CortexApp/  WinUI 3 demo app (net10-windows)
TEST/PredicitiveCoding.UnitTests/    xUnit test suite (net10)
tools/                             DetectorDistiller · OnnxPack · MixedPrecisionBench
```

> **Note on the folder name.** The original core folder is spelled `PredicitiveCoding`
> (a typo kept for git history); the C# namespace is the correct `PredictiveCoding`.

## Related docs already in the repo
- [`CODEBASE.md`](../CODEBASE.md) — quick orientation for the core library.
- [`CLAUDE_PROJECT_RULES.md`](../CLAUDE_PROJECT_RULES.md) — important paths and conventions.
- [`PredictiveCoding.Brain/ROADMAP.md`](../PredictiveCoding.Brain/ROADMAP.md) — the Thousand Brains roadmap (closed: all phases P0–P9 done).
- [`PredictiveCoding.LLM/LLM_Roadmap.md`](../PredictiveCoding.LLM/LLM_Roadmap.md) — the LLM feature checklist, plus
  [`ROADMAP-Pro.md`](../PredictiveCoding.LLM/ROADMAP-Pro.md) and
  [`ROADMAP-Scale.md`](../PredictiveCoding.LLM/ROADMAP-Scale.md) (toy → professional → trained-on-the-web).
- [`PredictiveCoding.Cortex/ROADMAP.md`](../PredictiveCoding.Cortex/ROADMAP.md) — the cortical-areas roadmap.
- [`PredictiveCoding.Data/DATA_SOURCES.md`](../PredictiveCoding.Data/DATA_SOURCES.md) — per-source implementation table.
- [`PredicitiveCoding/ROADMAP-MixedPrecision.md`](../PredicitiveCoding/ROADMAP-MixedPrecision.md) — the mixed-precision roadmap.
- [`PredictiveCoding.Device/README.md`](../PredictiveCoding.Device/README.md).
