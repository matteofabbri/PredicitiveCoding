# PredictiveCoding — Codebase Reference

Quick orientation document for future sessions. Read this instead of scanning every file.

---

## Project layout

```
PredicitiveCoding/          ← core library (net10)
  Core/                     Tensor primitives, activations, shapes, random utils
  Model/                    Layer, LayerStore, LayerSerializer (FasterKV persistence)
  Networks/                 Network + NetworkConfig
                            (TemporalNetwork, MixtureOfExperts, Ensemble, Attentive, Residual, TemporalMoE → PredictiveCoding.Expanded/Networks)
  Training/                 Trainer (static), TrainingConfig
  Data/                     IDataset, SyntheticDataset, CifarDataset
  Logging/                  MetricLogger (console), CsvLogger

PredicitiveCoding.UnitTests/ ← xUnit tests (net10)
PredictiveCoding.Main/       ← entry point + Demos/ + Data/
PredictiveCoding.Tests/      ← WinUI test harness (not the primary test suite)
```

> **Note**: the folder is spelled `PredicitiveCoding` (typo in original). The C# namespace is `PredictiveCoding`.

---

## Theory in one paragraph

Predictive coding (Rao & Ballard 1999; Friston 2005) frames perception and learning as free-energy minimisation. A hierarchical network holds a latent state `μ_l` at each layer. Each layer generates a top-down **prediction** `pred_l = f(W_l μ_l + b_l)` for the layer below. The **prediction error** `ε_{l-1} = μ_{l-1} − pred_l` drives both hidden-state updates (inference) and weight updates (learning). At convergence the network minimises `F = ½ Σ ||ε_l||²`.

Classification is energy-based: `c* = argmin_c F(x, one-hot(c))` — the class that produces the smallest total prediction error wins.

---

## Key classes

### `Network` (`Networks/Network.cs`)

The core model. Thread-safe via `ReaderWriterLockSlim`.

| Method | What it does |
|---|---|
| `Infer(input, steps)` | Read-lock. Runs inference, does not persist. |
| `Learn(input, label, steps)` | Write-lock. Inference + Hebbian weight update + persist. |
| `Predict(input, steps)` | Write-lock. Min-energy classification over all classes; persists winner state. |
| `PredictWithEnergy` | Same as `Predict` but also returns winning `F`. |
| `InferEnergy` | Read-lock. Returns `F` after clamped inference, no weight update. Used by MoE. |
| `LearnGated(…, gateWeight)` | Write-lock. Learning scaled by `gateWeight`; skips update if weight == 0. Used by MoE. |
| `LearnTemporal` / `InferTemporal` / `PredictTemporal` | Temporal variants; inject `_hiddenPriors` before inference. Used by `TemporalNetwork`. |
| `FlushStore()` | Flush FasterKV writes (only relevant when `LazyPersistence = true`). |

**Layer layout**: `layer[0]` = sensory (clamped to input). `layer[L-1]` = label (clamped during training). `layer[1..L-2]` = hidden (free to update).

**Feedforward Gist Init** (`UseFeedforwardGistInit`): single top-down pass before the iterative loop seeds hidden-state `μ` — warm start, reduces required inference steps (Haider et al. 2024).

All layer state lives in `LayerStore` (FasterKV). `ReadAllLayers()` loads all layers into RAM for a single pass; only modified layers are written back.

---

### `NetworkConfig` (`Networks/NetworkConfig.cs`)

`sealed record` — immutable after construction (except `Path` normalisation in `Validate()`).

| Field | Default | Notes |
|---|---|---|
| `LayerSizes` | `[20,64,32,4]` | Input→output, e.g. `[784,256,10]` |
| `Activation` | `Tanh` | Applied in generative prediction |
| `StateLearningRate` | `0.1` | η for hidden-state updates |
| `WeightLearningRate` | `0.001` | η for weight updates |
| `WeightDecay` | `0` | L2 on weights |
| `UseDiagonalPrecision` | `false` | Per-unit Π_l (Huang & Rao 2011) |
| `LazyPersistence` | `false` | Batch FasterKV writes; call `FlushStore()` each epoch |
| `UseFeedforwardGistInit` | `false` | FFG warm-start (Haider 2024) |
| `ActivityRegularization` | `0` | L1 on μ (Keller & Mrsic-Flogel 2018) |
| `InputShape` / `OutputShape` | `null` | Optional ND shape; auto-flattened. Product must equal `LayerSizes[0/^1]`. |

---

### `Layer` (`Model/Layer.cs`)

Holds a single layer's tensors: `Mu` (μ_l), `GenerativeWeights` (W_l), `Bias` (b_l), `Prediction`, `PredictionError`. Methods: `ComputePrediction()`, `SetPredictionError()`, `UpdateState()`, `UpdateWeights()`, `LocalEnergy()`, `WeightNorm()`. Supports `TemporalPrior` / `TemporalPrecision` for temporal mode.

---

### `LayerStore` (`Model/LayerStore.cs`)

FasterKV-backed store keyed by layer index. `ReadLayer(i)` → `Layer`. `Write(i, layer)` → persists. `Flush()` forces all deferred writes. With `LazyPersistence = true`, writes go to an in-memory write-buffer and are flushed explicitly — important for CIFAR-scale experiments.

---

### `TemporalNetwork` (`PredictiveCoding.Expanded/Networks/TemporalNetwork.cs`)

Thin wrapper around `Network`. Maintains `_hiddenPriors` (μ^{t-1} for each hidden layer) across timesteps. Temporal continuity penalty: `grad_l += Π_t · (μ_l − μ_l^{t-1})` (Friston 2008 DCM).

```csharp
using var tnet = new TemporalNetwork(network, temporalPrecision: 1f);
foreach (var (x, y) in sequence) tnet.Step(x, y, inferenceSteps: 20);
tnet.Reset();  // new sequence
int c = tnet.Predict(x, 20);
```

`Advance()` advances temporal context during evaluation without updating weights.

---

### `MixtureOfExperts` (`PredictiveCoding.Expanded/Networks/MixtureOfExperts.cs`)

Ensemble of `N` independent `Network` instances. Energy-based gating (novel):

1. Collect `F_i` from each expert via `InferEnergy` (no weight update).
2. `g_i = softmax(−β · F_i)` — lowest-energy expert gets highest weight.
3. Each expert updates weights via `LearnGated(…, g_i)`.

Prediction: `i* = argmin_i  min_c  F_i(x, one-hot(c))`.

Config via `MixtureOfExpertsConfig` (wraps a shared `ExpertConfig: NetworkConfig`, plus `ExpertCount`, `GatingTemperature`, `MinGateWeight`).

`LastGateWeights` and `LastExpertEnergies` expose the last step's gating state for diagnostics.

---

### `Trainer` (`Training/Trainer.cs`) + `TrainingConfig`

Static training loop. `TrainingConfig` embeds a `NetworkConfig` and adds: `InferenceSteps`, `Epochs`, `BatchSize`, `Seed`, `CsvOutputPath`, `LogEveryNEpochs`.

The old `PCTrainer` (comments in Italian) is deleted; `Trainer.cs` is the current replacement.

---

### `TrainingConfig` pass-throughs

`TrainingConfig` exposes `LayerSizes`, `Activation`, `StateLearningRate`, `WeightLearningRate`, `WeightDecay`, `UseDiagonalPrecision` as read-only pass-throughs to the inner `Network` field.

---

## Data

| Class | Notes |
|---|---|
| `IDataset` | `GetBatches(batchSize, rng?)` → `IEnumerable<Batch>` |
| `SyntheticDataset` | Randomly generated classification data |
| `CifarDataset` | Loads CIFAR-10 binary batches; normalises to [−1, 1] |

`Batch` (`Training/Batch.cs`): holds `Inputs[]` and `Labels[]` as `DenseTensor<float>`.

---

## Core tensors

`DenseTensor<float>` (`Core/`) — flat float buffer with shape. Key ops: `Zero()`, `Clone()`, `CopyTo()`. `TensorUtils`: `Reshape()`, `Argmax()`, `MatMul()`, `Add()`.

`Shape` (`Core/Shape.cs`): thin wrapper around `int[]` for ND shapes.

---

## Logging

`MetricLogger`: console pretty-print per epoch. `CsvLogger`: appends epoch metrics CSV. `EpochMetrics` record: `Epoch`, `TrainEnergy`, `TrainAccuracy`, `TestEnergy`, `TestAccuracy`, `WeightNorm`, `ElapsedSeconds`.

---

## Demos (`PredictiveCoding.Main/Demos/`)

| File | What it demonstrates |
|---|---|
| `MillidgeReview2022Demo.cs` | Baseline PC classification |
| `HardClassificationDemo.cs` | Harder synthetic task |
| `TemporalDemo.cs` | `TemporalNetwork` on sequences |
| `MixtureOfExpertsDemo.cs` | `MixtureOfExperts` specialisation |

---

## Common patterns

**Minimal training loop**
```csharp
var cfg = new TrainingConfig {
    Network = new NetworkConfig { LayerSizes = [784, 256, 10] },
    Epochs = 50, InferenceSteps = 20
};
cfg.Validate();
using var net = new Network(cfg.Network, new Random(42));
// per sample:
net.Learn(input, label, cfg.InferenceSteps);
// per epoch (if LazyPersistence):
net.FlushStore();
// predict:
int cls = net.Predict(input, cfg.InferenceSteps);
```

**Energy-based gating is the MoE's only routing signal** — no learned router, no supervision. Experts specialise purely through differential free-energy minimisation.

---

## Important invariants

- `layer[0]` is **always** clamped to input during inference/learning/prediction.
- `layer[L-1]` is clamped to label only during `Learn*` and `Predict*` (not free `Infer`).
- `PredictCore` re-runs inference for the winning class before writing state, so persisted state always reflects the actual prediction.
- `_rwLock` is `SupportsRecursion`; read paths use read-lock, all write paths (including `Predict`) use write-lock.
- `TemporalNetwork` requires `LayerCount >= 3` (at least one hidden layer for `_hiddenPriors`).
- `MixtureOfExperts.Learn` runs two inference passes per expert per sample (one energy probe + one gated update) — keep `ExpertCount` small for speed.

---

## References

- Rao & Ballard (1999) — original hierarchical PC model
- Friston (2005) — free energy principle
- Bogacz (2017) — tutorial on PC mathematics
- Huang & Rao (2011) — diagonal precision, L2 weight decay
- Whittington & Bogacz (2017) — PC learning rules
- Haider et al. (2024) — Feedforward Gist initialisation
- Keller & Mrsic-Flogel (2018) — L1 activity regularisation
- Jacobs et al. (1991) — mixture of experts
- Friston (2008) — temporal / dynamic causal modelling
