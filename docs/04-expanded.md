# 4. Advanced PC architectures — `PredictiveCoding.Networks` (in the core)

Brain-inspired networks built by composing the core `Network`. Each is a small
`IDisposable` wrapper with a paired `*Config` record.

> **Where they live.** These architectures were originally a separate
> `PredictiveCoding.Expanded` project; it has been **merged into the core**.
> They now live under `PredicitiveCoding/Networks/` (one folder per family,
> namespaces `PredictiveCoding.Networks.*`), so the whole PC stack ships with a
> single reference to `PredictiveCoding` — exactly what `PredictiveCoding.Brain`,
> `PredictiveCoding.Cortex` and `PredictiveCoding.FaceRecognition` consume.

## The advanced PC architectures

| Module | Idea | Key API |
|--------|------|---------|
| **HierarchicalNetwork** (`Hierarchical/`) | Deep modular hierarchy: a stack of autonomous generative PCNs where each module encodes the **latent** of the one below (greedy layer-wise, like a hierarchical VAE without backprop). | `Encode`, `EncodeAll`, `Decode`, `Reconstruct`, `Learn` |
| **ActiveNetwork** (`Active/`) | Active inference: a world-model PCN predicts the next observation from `[obs ⊕ action]`; a policy picks the action minimising expected free energy toward a goal (Rao 2024; Friston 2017). | `PredictNext`, `SelectAction`, `Observe`, `Step` |
| **DeepRecurrentNetwork** (`DeepRecurrent/`) | Cascade of temporal PC stages; each stage carries its own `μ^{t-1}` and feeds its latent up the stack (deep recurrent PC). | `Step`, `Encode`, `Reset` |
| **GatedStateNetwork** (`Gated/`) | LSTM-style per-unit write gate over a working-memory cell, injected as the top-layer prior (surprise-modulated retention). | `Step`, `Predict`, `Learn`, `Reset` |
| **HopfieldNetwork** (`Hopfield/`) | Generative PCN + a **modern (continuous) Hopfield** attractor over the latent (Ramsauer 2021): pattern completion from partial cues. | `Learn`, `Store`, `Recall`, `RecallLatent` |
| **GraphPredictiveNetwork** (`Graph/`) | Predictive coding over arbitrary graphs: a shared edge-predictor learns `node → neighbour`; inference is message passing by error minimisation. | `Learn`, `Complete`, `GraphEnergy` |
| **HierarchicalMixtureOfExperts** (`MoE/`) | Two-level energy-gated MoE (Jordan & Jacobs 1994): within-group + across-group softmax over free energies; emergent specialisation. | `Learn`, `Predict` |
| **SpikingNetwork** (`Spiking/`) | Self-contained LIF spiking PCN: rate-coded states, errors as firing-rate differences, local Hebbian learning (neuromorphic). | `Infer`, `Learn`, `Predict` |

## The generative / associative modules

| Module | Idea |
|--------|------|
| **GenerativeNetwork** (`Generative/`) | VAE-like PCN reused in two directions: `Encode` (recognition) / `Decode` (generation) / `Sample`; Gaussian top-layer prior as a KL analogue. |
| **MemoryNetwork** + **MemoryBank** (`Memory/`) | Hippocampal episodic prior: cosine-similarity key-value store that biases the top-layer prior before inference. |
| **MultiModalNetwork** (`MultiModal/`) | N modality encoders → shared associative PCN; cross-modal completion. |
| **RoutedNetwork** (`Routed/`) | Learned router PCN blended with per-sample free-energy gating (amortised MoE). |

## The "classic" advanced networks (same folder)

| Type | Idea |
|------|------|
| **TemporalNetwork** (`Temporal/`) | Carries `μ^{t-1}` across timesteps (Friston 2008 temporal continuity). |
| **MixtureOfExperts** (`MoE/`) | Energy-based gating `g_i = softmax(−β·F_i)`; experts specialise unsupervised. |
| **TemporalMixtureOfExperts** (`MoE/`) | Temporal MoE: per-expert hidden-state context + energy gating. |
| **EnsembleNetwork** (`Ensemble/`) | Bayesian model averaging over PCN members (probability or energy averaging). |
| **AttentiveNetwork** (`Attentive/`) | Per-unit learnable attention gates on the prediction errors. |
| **ResidualNetwork** (`Residual/`) | 2-hop skip connections in the PC hierarchy. |

`MixtureOfExperts.ComputeGates` is the shared gating helper reused by
`HierarchicalMixtureOfExperts`, `RoutedNetwork`, and the Cortex
`Thalamus`/`BasalGanglia`.

Also at the root of `Networks/`: **`EffortLevel`** (inference-effort control)
and **`GenerativeSnapshot`** — an exported weights snapshot that lets
`BatchedPcForward` (`PredictiveCoding.Device`) settle a *batch* of inputs as
GEMMs instead of looping per sample (see [doc 7](07-device-gpu.md)).

## Example — modern Hopfield pattern completion

```csharp
using var net = new HopfieldNetwork(new HopfieldNetworkConfig {
    NetworkConfig = new NetworkConfig { LayerSizes = [144, 96, 48] },
    RetrievalBeta = 8f, RetrievalIterations = 5, BasePath = "./hopfield",
}, new Random(0));

foreach (var p in prototypes) { net.Learn(p, 20); net.Store(p, 20); }
var clean = net.Recall(noisyCue, 20);   // snap to nearest attractor, then decode
```

## Example — active inference

```csharp
var cfg = new ActiveNetworkConfig {
    ObservationDim = 2, ActionCount = 4,
    WorldModelConfig = new NetworkConfig { LayerSizes = [6, 16, 2] },  // [obs ⊕ one-hot(action)] → obs
};
using var agent = new ActiveNetwork(cfg, new Random(0));
agent.Observe(state, action, nextState, 12);          // learn a transition
int a = agent.Act(state, goal, 12);                   // minimise expected free energy
```
