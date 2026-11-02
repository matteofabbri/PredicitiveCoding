# 8. Testing — `PredicitiveCoding.UnitTests`

An xUnit suite (net10) of **hundreds of tests across ~130 test files**, covering
every library in the solution: core + advanced architectures, Cortex, Brain, LLM,
Device, Data, Onnx, Vision, FaceRecognition and Imaging. It is the primary test
project (the WinUI `PredictiveCoding.Tests` harness is not in the solution).

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
- **Network-free where possible** — the few tests that touch remote sources
  (Data integration) are shaped so a missing connection fails the source, not the
  suite's deterministic core.

## Coverage by area

| Area | Test files |
|------|-----------|
| Core | `ActivationFnTests`, `TensorUtilsTests`, `DenseTensorTests`, `PCLayerTests`, `NetworkTests`, `NetworkImageTests`, `NetworkLargeDimsTests`, `NetworkVolume3DTests`, `NetworkConcurrencyTests`, `BugAndCrashTests` |
| Advanced architectures (in core `Networks/`) | `TemporalNetworkTests`, `MixtureOfExpertsTests`, `HierarchicalNetworkTests`, `ActiveNetworkTests`, `DeepRecurrentNetworkTests`, `GatedStateNetworkTests`, `HopfieldMemoryTests`, `SpikingNetworkTests`, `ExpandedNetworkTests` (Generative/Hopfield/Graph/HierMoE/MultiModal/Memory), `MovedNetworkTests` (Ensemble/Residual/Attentive/TemporalMoE/Routed) |
| Cortex | `RetinotopicMapTests`, `GridCellEncoderTests`, `VisualCortexTests`, `CortexAreaTests`, `CortexWrapperTests` |
| Brain | `BrainSdrTests`, `BrainSequenceTests`, `BrainVotingTests`, `BrainConnectivityTests`, `BrainSystemTests`, `BrainCompositionTests`, `BrainEvalTests`, `BrainSe3PoseTests`, `BrainSe3PathIntegrationTests`, `BrainSe3RecognitionTests`, `BrainImageSensorTests`, `BrainVizTests` |
| LLM | `LlmSmokeTests`, `LlmFeaturesTests`, `LlmTrainingTests`, `LlmInfrastructureTests`, `LlmModernTests`, `LlmScaleBTests`, `ChatTemplateTests` |
| Device | `LlmDeviceTests` (CPU fallback path), `LlmDeviceTrainTests` (the device pretraining engine) |
| Data | `DataIntegrationTests` + language-detection suites (`langdetect` profiles vs native n-gram classifier) |
| Onnx | `OnnxOpsTests`, `OnnxTransformerOpsTests`, `OnnxParserTests`, `OnnxExternalDataTests`, `OnnxPcnxFormatTests`, `OnnxPcnxIntegrationTests`, `OnnxResizeTests`, `OnnxReviewFixTests`, `OnnxDetectorWiringTests` |
| Vision / FaceRecognition / Imaging | people-counting pipeline tests, `PcDistilledFaceEmbedderTests` (checkpoint/probe semantics), imaging codec round-trips |

## Notable test types
- **Numerical gradient check** — `LlmTrainingTests` (and the translator CLI's
  `gradcheck` command) perturbs autograd parameters and compares analytic vs.
  finite-difference gradients, including *zero gradient on masked rows*.
- **Equivalence** — flash online-softmax == standard attention; quantized/low-precision
  matvec ≈ fp32; sharded linear == single-device; optimiser leaves ONNX `Run`
  results **bit-identical**; batched PC forward == per-sample settling.
- **Round-trips** — tokenizer encode/decode, retinotopic extract/assemble, checkpoint
  save/load (+ resume with Adam moments), device tensor `To`/`ToHost`, PCNX
  save/reload, distilled-embedder manifest probe, SDR/object-model persistence.

## Running
```bash
dotnet test PredicitiveCoding.UnitTests/PredicitiveCoding.UnitTests.csproj
# or a subset
dotnet test --filter "FullyQualifiedName~Cortex"
```
The only assemblies without unit tests are the GUI/apps (`CortexApp`, `Main`'s
menu itself, the tools) and the real GPU execution paths (no hardware in CI;
they are smoke-tested on the CPU fallback, and demo 13 refuses to run without a
visible CUDA device).
