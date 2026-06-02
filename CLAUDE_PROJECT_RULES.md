# Claude Project Rules — Assistant Mode

## Environment

- The project is entirely in .NET 10 / C# 12+.
- Read and modify only: `*.cs`, `*.csproj`, `*.slnx`, `*.json`, `*.yaml`, `*.md`.
- Ignore all other file types.

## General behavior

- Act as the technical assistant for this project.
- Maintain a persistent mental model of the code structure using the "Important Paths" section.
- Do not read the entire solution unless explicitly requested.
- Before taking an action, describe in one sentence what you will do.
- If an action may have side effects, ask for confirmation.

## Reading and navigation

- When searching for something, limit the search to the specified folder.
- If no folder is specified, search only in the paths listed in "Important Paths".
- If additional context is needed, ask before reading multiple files.

## File modifications

- Modify only what is requested.
- Do not rewrite entire files unless necessary.
- Preserve existing style and structure.
- When creating new files, ask for confirmation.

## Auto-update rule

- You may only auto-update the "Important Paths" section.
- Ask for confirmation before adding or changing a path.
- Do not modify other sections without explicit instructions.

---

## Important Paths

### Solution & projects

| Path | Description |
|------|-------------|
| `PrediciiveCoding.slnx` | Solution file |
| `PredicitiveCoding/` | Core PC library — layers, networks, training, data, logging |
| `PredictiveCoding.Brain/` | Brain-inspired extensions (generative, memory, multi-modal, routed) |
| `PredictiveCoding.Main/` | Demo entry point and CIFAR-10 runner |
| `PredicitiveCoding.UnitTests/` | Unit test suite (xUnit) |
| `PredictiveCoding.Tests/` | Integration test harness (WinUI) |
| `cifar-10-data/` | CIFAR-10 binary dataset files |

### Core library (`PredicitiveCoding/`)

| Path | Key types |
|------|-----------|
| `PredicitiveCoding/Core/Activation.cs` | `ActivationFn`, `ActivationType` |
| `PredicitiveCoding/Core/Shape.cs` | `DenseTensor<T>`, `ShapeCheck` |
| `PredicitiveCoding/Core/TensorUtils.cs` | SIMD tensor ops — MatVecMul, OuterProduct, Axpy, etc. |
| `PredicitiveCoding/Core/RandomUtils.cs` | Weight initialization |
| `PredicitiveCoding/Model/Layer.cs` | `Layer` — μ, W, b, ε, precision, temporal prior |
| `PredicitiveCoding/Model/LayerStore.cs` | `LayerStore` — FASTER KV persistence |
| `PredicitiveCoding/Model/LayerSerializer.cs` | Serialization for FASTER checkpoints |
| `PredicitiveCoding/Networks/Network.cs` | `Network` — base hierarchical PC network |
| `PredicitiveCoding/Networks/NetworkConfig.cs` | `NetworkConfig` record |
| `PredicitiveCoding/Networks/ResidualNetwork.cs` | `ResidualNetwork` — 2-hop skip connections |
| `PredicitiveCoding/Networks/ResidualNetworkConfig.cs` | `ResidualNetworkConfig` |
| `PredicitiveCoding/Networks/TemporalNetwork.cs` | `TemporalNetwork` — carries μ^{t-1} across steps |
| `PredicitiveCoding/Networks/AttentiveNetwork.cs` | `AttentiveNetwork` — per-unit learnable attention gates |
| `PredicitiveCoding/Networks/MixtureOfExperts.cs` | `MixtureOfExperts` — energy-based gating |
| `PredicitiveCoding/Networks/MixtureOfExpertsConfig.cs` | `MixtureOfExpertsConfig` |
| `PredicitiveCoding/Networks/TemporalMixtureOfExperts.cs` | `TemporalMixtureOfExperts` |
| `PredicitiveCoding/Networks/TemporalMoEConfig.cs` | `TemporalMoEConfig` |
| `PredicitiveCoding/Networks/EnsembleNetwork.cs` | `EnsembleNetwork` — Bayesian model averaging |
| `PredicitiveCoding/Networks/EnsembleConfig.cs` | `EnsembleConfig` |
| `PredicitiveCoding/Networks/RoutedNetwork.cs` | `RoutedNetwork` — learned router + energy gating |
| `PredicitiveCoding/Networks/RoutedNetworkConfig.cs` | `RoutedNetworkConfig` |
| `PredicitiveCoding/Training/Trainer.cs` | `Trainer` — epoch loop, evaluate, full train |
| `PredicitiveCoding/Training/TrainingConfig.cs` | `TrainingConfig` |
| `PredicitiveCoding/Training/Batch.cs` | `Batch` |
| `PredicitiveCoding/Data/IDataset.cs` | `IDataset` interface |
| `PredicitiveCoding/Data/SyntheticDataset.cs` | `SyntheticDataset` |
| `PredicitiveCoding/Logging/MetricLogger.cs` | Console logger |
| `PredicitiveCoding/Logging/CsvLogger.cs` | CSV export |
| `PredicitiveCoding/Logging/EpochMetrics.cs` | `EpochMetrics` record |

### Brain module (`PredictiveCoding.Brain/`)

| Path | Key types |
|------|-----------|
| `PredictiveCoding.Brain/GenerativeNetwork/GenerativeNetwork.cs` | VAE-like unsupervised PC — Encode, Decode, Sample |
| `PredictiveCoding.Brain/GenerativeNetwork/GenerativeNetworkConfig.cs` | `GenerativeNetworkConfig` |
| `PredictiveCoding.Brain/MemoryNetwork/MemoryNetwork.cs` | Hippocampal memory prior — episodic retrieval + top-layer bias |
| `PredictiveCoding.Brain/MemoryNetwork/MemoryBank.cs` | `MemoryBank` — cosine-similarity key-value store |
| `PredictiveCoding.Brain/MemoryNetwork/MemoryNetworkConfig.cs` | `MemoryNetworkConfig` |
| `PredictiveCoding.Brain/MultiModal/MultiModalNetwork.cs` | N-modality encoders + shared associative PCN |
| `PredictiveCoding.Brain/MultiModal/MultiModalNetworkConfig.cs` | `MultiModalNetworkConfig` |
| `PredictiveCoding.Brain/MultiModal/ModalityConfig.cs` | Per-modality config |
| `PredictiveCoding.Brain/RoutedNetwork/RoutedNetwork.cs` | Router PCN — amortized gating |
| `PredictiveCoding.Brain/RoutedNetwork/RoutedNetworkConfig.cs` | `RoutedNetworkConfig` |

### Main application (`PredictiveCoding.Main/`)

| Path | Description |
|------|-------------|
| `PredictiveCoding.Main/Program.cs` | Entry point |
| `PredictiveCoding.Main/Demos/HardClassificationDemo.cs` | Hard classification demo |
| `PredictiveCoding.Main/Demos/MixtureOfExpertsDemo.cs` | MoE demo |
| `PredictiveCoding.Main/Demos/TemporalDemo.cs` | Temporal processing demo |
| `PredictiveCoding.Main/Demos/MillidgeReview2022Demo.cs` | Millidege 2022 review demo |
| `PredictiveCoding.Main/Data/CifarDataset.cs` | CIFAR-10 loader |

### Unit tests (`PredicitiveCoding.UnitTests/`)

| Path | Coverage |
|------|----------|
| `ActivationFnTests.cs` | Activation correctness |
| `TensorUtilsTests.cs` | Tensor ops |
| `DenseTensorTests.cs` | Indexing / reshape |
| `PCLayerTests.cs` | Layer state updates |
| `NetworkTests.cs` | Inference / learning loops |
| `NetworkImageTests.cs` | Image input |
| `NetworkLargeDimsTests.cs` | High-dimensional networks |
| `NetworkVolume3DTests.cs` | 3D tensors |
| `NetworkConcurrencyTests.cs` | Thread safety |
| `MixtureOfExpertsTests.cs` | MoE gating |
| `TemporalNetworkTests.cs` | Temporal continuity |
| `BugAndCrashTests.cs` | Regression / edge cases |

---

## Architecture summary

### Namespaces

| Namespace | Main types |
|-----------|-----------|
| `PredictiveCoding.Core` | `DenseTensor<T>`, `ActivationFn`, `TensorUtils`, `RandomUtils` |
| `PredictiveCoding.Model` | `Layer`, `LayerStore`, `LayerSerializer` |
| `PredictiveCoding.Networks` | `Network`, `ResidualNetwork`, `TemporalNetwork`, `AttentiveNetwork`, `MixtureOfExperts`, `EnsembleNetwork`, `RoutedNetwork` + all `*Config` |
| `PredictiveCoding.Training` | `Trainer`, `TrainingConfig`, `Batch` |
| `PredictiveCoding.Data` | `IDataset`, `SyntheticDataset` |
| `PredictiveCoding.Logging` | `MetricLogger`, `CsvLogger`, `EpochMetrics` |
| `PredictiveCoding.Brain.GenerativeNetwork` | `GenerativeNetwork`, `GenerativeNetworkConfig` |
| `PredictiveCoding.Brain.MemoryNetwork` | `MemoryNetwork`, `MemoryBank`, `MemoryNetworkConfig` |
| `PredictiveCoding.Brain.MultiModal` | `MultiModalNetwork`, `ModalityConfig`, `MultiModalNetworkConfig` |
| `PredictiveCoding.Brain.RoutedNetwork` | `RoutedNetwork`, `RoutedNetworkConfig`, `RoutedPrediction` |

### Core inference loop (`Network`)

1. Top-down predictions: `layers[l].ComputePrediction()` → `pred_l = f(W_l μ_l + b_l)`
2. Bottom-up errors: `layers[l-1].SetPredictionError(layers[l].Prediction)` → `ε_l = Π_l(μ_l − pred)`
3. Top layer: `SetPredictionError(null)` — flat prior
4. State updates: `layers[l].UpdateState(...)` — gradient descent on μ
5. Weight updates (learning only): `layers[l].UpdateWeights(...)` — Hebbian: `ΔW = η(f'⊙ε)⊗μᵀ`

Classification principle: `c* = argmin_c F(x, one-hot(c))`

### Key design principles

- **Free energy minimization** (Bogacz 2017, Friston 2005) — precision-weighted prediction errors
- **No global backpropagation** — all updates are local Hebbian rules
- **Temporal continuity** (Friston 2008) — `grad += Π_t(μ_l − μ_l^{t-1})`
- **Energy-based gating** — `g_i = softmax(−β·F_i)` for MoE routing
- **Lazy FASTER persistence** — in-memory cache, batch flush to disk
- **SIMD via `System.Numerics.Tensors`** — all tensor ops use hardware vectorization

### Key dependencies

| Package | Version | Use |
|---------|---------|-----|
| `Microsoft.FASTER.Core/Client/Server` | 2.6.5 | Persistent KV store for layers |
| `System.Numerics.Tensors` | 10.0.8 | SIMD tensor primitives |

Target framework: **.NET 10** — Language: **C# 12+** with nullable annotations and implicit usings.
