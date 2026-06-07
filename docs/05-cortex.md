# 5. Cortex — `PredictiveCoding.Cortex` (+ `CortexApp`)

A predictive-coding model of the **visual cortex** and 18 other brain regions,
built by composing core and Expanded networks plus a few new primitives.

## The visual cortex

`VisualCortex` (+ `VisualCortexConfig`) models the ventral stream:

```
image [C,H,W]
  │  RetinotopicMap : tile into overlapping P×P receptive fields
  ▼
V1   : one weight-shared cortical-column PCN applied at every position  (= convolution)
  │  concatenate over the grid → V1 feature map
  ▼
V2…IT: HierarchicalNetwork (deep modular hierarchy) → abstract object latent
  ▼
readout (optional): PCN classifier on the IT latent
```

- **`RetinotopicMap`** *(new primitive)* — `Extract` an image into patches and
  `Assemble` them back (overlap-averaged). This is the missing receptive-field
  machinery; the rest reuses tested components.
- Learning is all local PC: the shared V1 column autoencodes every patch (learning
  Gabor-like features, Rao & Ballard 1999), each higher area greedily encodes the
  one below, and the readout uses min-energy classification.
- `Encode`, `EncodeAreas`, `Learn`, `LearnSupervised`, `Predict`, `Reconstruct`.

## The cortical areas

Organised by system. Most areas are an `IDisposable` wrapping tested structures;
the pure controllers (`Thalamus`, `BasalGanglia`, and the neuromodulators) hold no
persistent state.

### Sensory (`Sensory/`)
| Area | What it is |
|------|-----------|
| **AuditoryCortex** | Tonotopic A1→belt→parabelt: the visual hierarchy on a 1-channel spectrogram. |
| **SomatosensoryCortex** | Somatotopic S1: same microcircuit on an `[R,H,W]` skin map. |
| **DorsalStream** | "Where/how" path: per-frame `VisualCortex` + `DeepRecurrentNetwork` for motion. |
| **RetinaLgn** *(new)* | Center-surround (Difference-of-Gaussians) whitening front-end. |

### Memory / medial temporal lobe (`Memory/`)
| Area | What it is |
|------|-----------|
| **HippocampalFormation** | Fast `HopfieldNetwork` episodic store + offline **replay** consolidating into a neocortex (McClelland 1995). |
| **EntorhinalCortex** + **GridCellEncoder** *(new)* | Hexagonal grid-cell code for a 2-D position + a predictive spatial association. |
| **PiriformCortex** | Olfactory content-addressable memory (Hopfield). |

### Executive / frontal (`Executive/`)
| Area | What it is |
|------|-----------|
| **PrefrontalCortex** | `GatedStateNetwork` working memory + `HierarchicalMixtureOfExperts` task selection. |
| **MotorCortex** | Active-inference control loop over `ActiveNetwork` (`ReachGoal`). |
| **CerebellarCortex** | Fast supervised forward model `[state ⊕ command] → next state`. |

### Integration / control (`Control/`)
| Area | What it is |
|------|-----------|
| **AssociationCortex** | Multimodal binding via `MultiModalNetwork`. |
| **Thalamus** *(new)* | Precision/gain controller: allocates gain across areas by inverse free energy. |
| **BasalGanglia** *(new)* | Action selection by energy gating (go/no-go). |
| **DefaultModeNetwork** | Offline generative imagination via `GenerativeNetwork.Sample`. |

### Neuromodulation (`Neuromodulation/`)
| Area | What it is |
|------|-----------|
| **DopaminergicSystem** *(new)* | VTA/SNc dopamine: reward-prediction-error (TD) + average-reward baseline; modulates plasticity (`ModulateLearningRate`). |
| **LocusCoeruleus** *(new)* | Noradrenaline: global-surprise monitor; a phasic burst applies a broad precision reset across areas (mirrors `Thalamus.Modulate`). |
| **NucleusBasalis** *(new)* | Acetylcholine: per-area EMA of *expected* uncertainty → per-area learning-rate modulation. |
| **RapheNuclei** *(new)* | Serotonin: time-horizon / patience → effective temporal-discount factor γ. |

These four are pure controllers (no persistence), in the style of `Thalamus` /
`BasalGanglia`. They emit signals (RPE, gain, learning-rate factors); applying them is
the caller's job, since `Network` exposes no public runtime learning-rate setter.

## Roadmap — planned cortical areas

The next areas to build are tracked in
[`PredictiveCoding.Cortex/ROADMAP.md`](../PredictiveCoding.Cortex/ROADMAP.md).
They close whole missing systems (**neuromodulation is done** — see above; limbic
and language next) plus targeted single-area gaps, and — like every area above —
each is just an assembly of already-tested primitives (no new learning algorithm).

| Group (new folder) | Areas | Why |
|--------------------|-------|-----|
| ~~**Neuromodulation** (`Neuromodulation/`)~~ | DopaminergicSystem, LocusCoeruleus, NucleusBasalis, RapheNuclei | ✅ **Done (2026-06-07)** — now documented under *The cortical areas → Neuromodulation* above; 15 tests. |
| **Limbic** (`Limbic/`) | Amygdala (salience/valence), OrbitofrontalCortex (value), CingulateCortex/ACC (conflict & surprise), InsularCortex (interoception), Hypothalamus (homeostatic drives) | Adds the emotion / valuation axis; plugs into `Thalamus`, `BasalGanglia`, `PrefrontalCortex`. |
| **Language** (`Language/`) | WernickeArea (comprehension), BrocaArea (production), AngularGyrus (semantic hub) | Bridges the cortex to `PredictiveCoding.LLM` via `TemporalNetwork` / `GenerativeNetwork`. |
| **Extra Sensory** (`Sensory/`) | OlfactoryBulb, GustatoryCortex, VestibularSystem (+head-direction), FrontalEyeFields/SuperiorColliculus (saccades) | Fills sensory gaps; saccades drive `RetinotopicMap` patch selection. |
| **Extra Memory** (`Memory/`) | PerirhinalCortex (familiarity/novelty), ParahippocampalCortex (scene/context) | Recognition memory + scene gist; complements grid cells. |
| **Extra Motor / Integration** (`Executive/`, `Control/`) | PremotorCortex/SMA (action sequencing), PosteriorParietalCortex (coordinate transforms), Claustrum (global binding) | Sensorimotor transforms + global workspace. |

See the roadmap for the per-area primitive mapping, target paths, references and the
recommended build order.

## `CortexApp` — WinUI 3 demo

An **unpackaged** WinUI 3 desktop app (`net10-windows`) with a `NavigationView`
shell and one page per example, each running on a background thread:

- **Visual Cortex** — train on synthetic shapes, classify + reconstruct a noisy test image.
- **Hopfield Memory** — pattern completion (original / corrupted / recalled).
- **Grid Cells** — render the firing maps of entorhinal grid cells.
- **Spiking Network** — train, report accuracy / free energy.
- **Active Inference** — a 2-D agent learns a world model and reaches a goal.

`ImageRender.ToBitmap` renders a `[C,H,W]` tensor to a `WriteableBitmap`. The app
is unpackaged so the solution builds under AnyCPU; build/run notes are in
[doc 9](09-build-and-run.md).

## Example

```csharp
var cfg = new VisualCortexConfig {
    ImageChannels = 3, ImageHeight = 32, ImageWidth = 32, PatchSize = 8, PatchStride = 4,
    V1Column    = new NetworkConfig { LayerSizes = [192, 128, 32] },      // 3·8·8 → K=32
    HigherAreas = new HierarchicalNetworkConfig { Modules = [
        new NetworkConfig { LayerSizes = [7*7*32, 512, 128] },           // V2
        new NetworkConfig { LayerSizes = [128, 64] }] },                 // V4→IT
    Classifier  = new NetworkConfig { LayerSizes = [64, 10] },
};
using var cortex = new VisualCortex(cfg, new Random(42));
cortex.LearnSupervised(image, label, 20);
int cls = cortex.Predict(image, 20);
```
