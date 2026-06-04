# 3. Core library — `PredictiveCoding`

The foundation. Folder `PredicitiveCoding/`, namespaces `PredictiveCoding.*`.

## Layout

```
Core/      DenseTensor<T>, TensorUtils, ActivationFn, RandomUtils, Shape
Model/     Layer, LayerStore, LayerSerializer
Networks/  Network, NetworkConfig, TemporalNetwork*
Training/  Trainer, TrainingConfig, Batch
Data/      IDataset, SyntheticDataset
Logging/   MetricLogger, CsvLogger, EpochMetrics
```
\* `TemporalNetwork` and the advanced networks now live in
`PredictiveCoding.Expanded.Networks` (see [doc 4](04-expanded.md)); core `Networks/`
holds only `Network` and `NetworkConfig`.

## Tensors (`Core/`)

- **`DenseTensor<T>`** — a flat row-major buffer (`Buffer`) with a shape
  (`Dimensions`). `T : struct, INumber<T>`. Helpers: `Zeros`, `Like`, `Clone`,
  `CopyTo`, `Zero`, `Fill`, plus 1-D/2-D/N-D indexers.
- **`TensorUtils`** — SIMD ops via `System.Numerics.Tensors.TensorPrimitives`:
  `MatVecMul` (W·v), `MatVecMulT` (Wᵀ·v), `OuterProduct`, `ElementwiseAdd/Sub/Mul`,
  `InPlaceAxpy`, `SumSquares`, `Reshape`, `Concat`.
- **`ActivationFn`** — `Apply` and `Derivative` for `Linear / Tanh / ReLU / Sigmoid`,
  plus a stable `Softmax`.
- **`RandomUtils`** — Xavier/He weight init, Gaussian sampling, Fisher-Yates shuffle.

## `Layer` (`Model/Layer.cs`)

Holds one layer's tensors: `Mu` (μ), `GenerativeWeights` (W), `Bias` (b),
`Prediction`, `PredictionError`, plus optional `PriorMean`/`Precision` and
`TemporalPrior`/`TemporalPrecision`. Methods implement the PC update primitives:
`ComputePrediction`, `SetPredictionError`, `UpdateState`, `UpdateWeights`,
`LocalEnergy`, `WeightNorm`.

## `LayerStore` (`Model/LayerStore.cs`)

All layer state lives in a **FASTER-KV** store keyed by layer index. `ReadLayer`,
`Write`, `Flush`. With `LazyPersistence = true` writes are buffered in memory and
flushed once per epoch — important for CIFAR-scale runs.

## `Network` (`Networks/Network.cs`)

The PC engine, thread-safe via `ReaderWriterLockSlim`. Key methods:

| Method | Behaviour |
|--------|-----------|
| `Infer(input, steps)` | Settle hidden states (read-lock, transient). |
| `Learn(input, label, steps)` | Inference + Hebbian weight update + persist. |
| `Predict(input, steps)` | Min-energy classification `argmin_c F(x, one-hot(c))`. |
| `PredictWithEnergy`, `InferEnergy`, `ClassEnergies` | Energy probes used by MoE / Ensemble. |
| `LearnGated(…, gateWeight)` | Weight update scaled by a gate (MoE). |
| `Learn/Infer/PredictTemporal` | Temporal variants with injected `μ^{t-1}` priors. |
| `InferGetTop`, `Generate`, `LearnUnsupervised`, `SetTopLayerPrior` | "Brain-support" API used by Expanded/Cortex (encode, decode, VAE-style learning, memory priors). |
| `LearnUnsupervisedTemporal`, `InferGetTopTemporal` | Temporal encode+learn for the deep-recurrent cascade. |

**Layer layout**: `layer[0]` = sensory (clamped to input), `layer[L-1]` = label
(clamped during training/prediction), `layer[1..L-2]` = free hidden.

## `NetworkConfig`

Immutable `record`. Notable fields: `LayerSizes`, `Activation`,
`StateLearningRate`, `WeightLearningRate`, `WeightDecay`, `UseDiagonalPrecision`,
`LazyPersistence`, `UseFeedforwardGistInit`, `ActivityRegularization`,
`InputShape`/`OutputShape` (auto-flattened), and `Path` (FASTER-KV directory).

## `Trainer` (`Training/`)

Static epoch loop over an `IDataset`. `TrainingConfig` embeds a `NetworkConfig`
and adds `InferenceSteps`, `Epochs`, `BatchSize`, `Seed`, `CsvOutputPath`,
`LogEveryNEpochs`.

## Minimal example

```csharp
var cfg = new NetworkConfig { LayerSizes = [784, 256, 10], Path = "./data" };
using var net = new Network(cfg, new Random(42));
net.Learn(image, oneHotLabel, inferenceSteps: 20);   // per sample
int cls = net.Predict(image, 20);                    // argmin-energy class
```
