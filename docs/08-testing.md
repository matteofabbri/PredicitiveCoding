# 8. Testing — `PredicitiveCoding.UnitTests`

An xUnit suite (net10) of **300+ tests** covering the core, Expanded, Cortex, LLM
and Device layers. It is the primary test project (the WinUI `PredictiveCoding.Tests`
harness is not in the solution).

## Conventions
- **Deterministic** — every test uses a fixed `new Random(seed)` so results are reproducible.
- **Isolated FASTER-KV** — components that persist weights get a unique temp directory
  per test via `IDisposable`:
  ```csharp
  private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"x_{Guid.NewGuid():N}");
  public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true); }
  ```
- **What is asserted** — shapes, value ranges and behaviour. Because most models are
  untrained (energy/inference-based, random init), correctness is checked as:
  finite outputs, valid class ranges, *and* "learning reduces error" wherever the
  objective is deterministically monotone (e.g. autoencoding with a linear activation).
- **No global-state races** — tests avoid asserting on shared mutable statics
  (e.g. `Gemm.Provider`) that other parallel tests mutate.

## Coverage by area

| Area | Test files |
|------|-----------|
| Core | `ActivationFnTests`, `TensorUtilsTests`, `DenseTensorTests`, `PCLayerTests`, `NetworkTests`, `NetworkImageTests`, `NetworkLargeDimsTests`, `NetworkVolume3DTests`, `NetworkConcurrencyTests`, `BugAndCrashTests`, `TemporalNetworkTests`, `MixtureOfExpertsTests` |
| Expanded | `HierarchicalNetworkTests`, `ActiveNetworkTests`, `DeepRecurrentNetworkTests`, `GatedStateNetworkTests`, `HopfieldMemoryTests`, `SpikingNetworkTests`, `ExpandedNetworkTests` (Generative/Hopfield/Graph/HierMoE/MultiModal/Memory), `MovedNetworkTests` (Ensemble/Residual/Attentive/TemporalMoE/Routed) |
| Cortex | `RetinotopicMapTests`, `GridCellEncoderTests`, `VisualCortexTests`, `CortexAreaTests`, `CortexWrapperTests` (all 14 areas) |
| LLM | `LlmSmokeTests`, `LlmFeaturesTests`, `LlmTrainingTests`, `LlmInfrastructureTests`, `LlmModernTests`, `ChatTemplateTests` |
| Device | `LlmDeviceTests` (CPU fallback path), `LlmTorchTests` (probes/fallback) |

## Notable test types
- **Numerical gradient check** — `LlmTrainingTests` perturbs autograd parameters and
  compares analytic vs. finite-difference gradients (proves backprop correctness).
- **Equivalence** — flash online-softmax == standard attention; quantized/low-precision
  matvec ≈ fp32; sharded linear == single-device.
- **Round-trips** — tokenizer encode/decode, retinotopic extract/assemble, checkpoint
  save/load, device tensor `To`/`ToHost`.

## Running
```bash
dotnet test PredicitiveCoding.UnitTests/PredicitiveCoding.UnitTests.csproj
# or a subset
dotnet test --filter "FullyQualifiedName~Cortex"
```
The only assemblies without unit tests are the WinUI `CortexApp` (a GUI) and the
real GPU/Metal execution paths (no hardware in CI; they are smoke-tested on the CPU
fallback).
