# PredictiveCoding — Documentation

A .NET 10 / C# research codebase built around **predictive coding** (Rao & Ballard
1999; Friston 2005): a backprop-free, biologically-plausible alternative to
standard deep learning where perception and learning are framed as local
free-energy minimisation. On top of the core PC engine the solution grows into
brain-inspired architectures, a model of the visual cortex (and 14 other cortical
areas), a **Thousand Brains sensorimotor system** (implementing the Hawkins/Numenta
papers), a full Transformer/LLM stack, and a CPU/CUDA/Metal device layer.

## Documentation map

| # | Document | What it covers |
|---|----------|----------------|
| 1 | [Overview](01-overview.md) | The solution, its projects and how they depend on each other |
| 2 | [Predictive coding theory](02-predictive-coding.md) | The maths: free energy, inference, local Hebbian learning |
| 3 | [Core library](03-core.md) | `PredictiveCoding` — tensors, `Layer`, `Network`, `Trainer` |
| 4 | [Expanded architectures](04-expanded.md) | `PredictiveCoding.Expanded` — 12+ brain-inspired networks |
| 5 | [Cortex](05-cortex.md) | `PredictiveCoding.Cortex` — visual cortex + 14 cortical areas, WinUI demo |
| 6 | [LLM](06-llm.md) | `PredictiveCoding.LLM` — Transformer, training, modern (DeepSeek/Gemma) techniques |
| 7 | [Device & GPU](07-device-gpu.md) | `PredictiveCoding.Device(.Torch)` — CPU/CUDA/Metal acceleration |
| 8 | [Testing](08-testing.md) | The xUnit suite (300+ tests) and its conventions |
| 9 | [Build & run](09-build-and-run.md) | Building, testing, running the demos and the GPU paths |

## At a glance

```
PredicitiveCoding/            core PC library (net10)
PredictiveCoding.Expanded/    brain-inspired architectures (net10)
PredictiveCoding.Cortex/      visual cortex + cortical areas (net10)
PredictiveCoding.Brain/       Thousand Brains sensorimotor system (net10)
PredictiveCoding.CortexApp/   WinUI 3 demo app (net10-windows)
PredictiveCoding.LLM/         Transformer / LLM stack (net10)
PredictiveCoding.Device/      device management + CUDA backend (net10)
PredictiveCoding.Device.Torch/ Metal (MPS) backend via TorchSharp (net10)
PredicitiveCoding.UnitTests/  xUnit test suite (net10)
PredictiveCoding.Main/        demo entry point + CIFAR-10 (net10)
```

> **Note on the folder name.** The original core folder is spelled `PredicitiveCoding`
> (a typo kept for git history); the C# namespace is the correct `PredictiveCoding`.

## Related docs already in the repo
- [`CODEBASE.md`](../CODEBASE.md) — quick orientation for the core library.
- [`CLAUDE_PROJECT_RULES.md`](../CLAUDE_PROJECT_RULES.md) — important paths and conventions.
- [`PredictiveCoding.LLM/LLM_Roadmap.md`](../PredictiveCoding.LLM/LLM_Roadmap.md) — the LLM feature checklist.
- [`PredictiveCoding.Brain/ROADMAP.md`](../PredictiveCoding.Brain/ROADMAP.md) — the Thousand Brains roadmap: a phase-by-phase plan to implement the Hawkins/Numenta papers (P1–P9) in code.
- [`PredictiveCoding.Device/README.md`](../PredictiveCoding.Device/README.md) and
  [`PredictiveCoding.Device.Torch/README.md`](../PredictiveCoding.Device.Torch/README.md).
