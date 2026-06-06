# PredictiveCoding.Brain — Thousand Brains Project roadmap

Goal: implement a **Thousand-Brains sensorimotor system** — many cortical columns
that each learn models of objects in their own reference frames and **vote** to
reach consensus — on top of the existing predictive-coding stack
(`PredictiveCoding` Core, `PredictiveCoding.Expanded`, `PredictiveCoding.Cortex`).

Each task below carries a **status flag** so it is clear what can be reused and
what still has to be built.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Available** — already implemented in the repo; reuse directly. |
| 🟡 | **Partial** — a reusable building block exists; needs adaptation/extension. |
| ❌ | **To build** — not present yet. |

*Status reflects the repository as of 2026-06-06.*

### Source papers

| Key | Reference |
|:---:|-----------|
| **P1** | Hawkins, J. (2021). *A Thousand Brains: A New Theory of Intelligence.* |
| **P2** | Clay, V. et al. (2024). *The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence.* |
| **P3** | Hawkins, J. et al. (2019). *A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex.* Frontiers in Neural Circuits. |
| **P4** | Hawkins, J. & Ahmad, S. (2016). *Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex.* Frontiers in Neural Circuits. |
| **P5** | Harris, K.D. & Shepherd, G.M.G. (2015). *The neocortical circuit: themes and variations.* Nature Neuroscience. |
| **P6** | Haueis, P. (2016). *The life of the cortical column…* History and Philosophy of the Life Sciences. |
| **P7** | Hawkins, J. (n.d.). *A Theory of How Columns in the Neocortex Enable Learning the Structure of the World.* Numenta whitepaper. |
| **P8** | Hawkins, J., Leadholm, N., Clay, V. (2025). *Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain.* |
| **P9** | Leadholm, N., Clay, V., Knudstrup, S., Lee, H., Hawkins, J. (2025). *Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference.* |

---

## Phase 0 — Representations & foundations

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Sparse Distributed Representation (SDR) type: sparse bitset with overlap, union, subsample, add-noise | P1, P4 | ❌ | New `Sdr` core type. |
| SDR similarity & metrics (overlap score, match, stability, capacity) | P4 | ❌ | New. |
| Scalar / category / coordinate → SDR encoders | P3 | 🟡 | `GridCellEncoder` (coordinates); add scalar/category encoders. |
| Dense ↔ SDR bridges to `DenseTensor<float>` | — | 🟡 | `DenseTensor` exists in Core. |
| Deterministic RNG / seeding utilities | — | ✅ | `Core/RandomUtils`. |

## Phase 1 — Cortical microcircuit substrate

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Mini-column abstraction (group of cells sharing feed-forward input, L4) | P5, P6, P7 | ❌ | New. |
| Layered column scaffold mapped to the canonical microcircuit (L4 input · L2/3 object output · L5 motor · L6a location) | P5, P6 | ❌ | Design grounded in P5/P6; new types. |
| Anatomical cortical-area modules to host columns | P5 | ✅ | `Cortex` (`VisualCortex`, `AssociationCortex`, `Sensory/*`, `Thalamus`, …). |
| Inhibition / k-winners-take-all for sparsity | P4 | ❌ | New (used by spatial pooler & TM). |

## Phase 2 — Sequence memory (HTM Temporal Memory)

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Distal dendritic segments + synaptic permanences | P4 | ❌ | New cell/segment model. |
| Temporal Memory: high-order sequence learning, predictive cells, bursting on surprise | P4 | ❌ | New; the core of P4. |
| Spatial Pooler: stable sparse encoding of input → SDR | P4 | ❌ | New. |
| Anomaly / surprise (prediction-error) signal | P4 | 🟡 | PCN free-energy/error already computed in `Networks/Network`. |
| Sequence-model baselines for comparison | P4 | 🟡 | `Expanded` `TemporalNetwork`, `DeepRecurrentNetwork`, `GatedStateNetwork`. |
| Associative recall of stored patterns | P4 | 🟡 | `Expanded/Hopfield` (`HopfieldMemory`, `HopfieldNetwork`). |

## Phase 3 — Reference frames & location (grid cells)

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Grid-cell location module (multi-scale, hexagonal population code) | P3 | ✅ | `Cortex/Memory/GridCellEncoder`. |
| Path integration: movement vector → updated location code | P3 | 🟡 | `EntorhinalCortex` (spatial assoc.); add motor-driven update. |
| Location ↔ feature predictive association | P3, P7 | 🟡 | `Cortex/Memory/EntorhinalCortex` learns location→observation. |
| Object-centric reference frames (anchor location to an object, not the room) | P3, P7 | ❌ | New. |
| Displacement cells (relative pose between features / sub-objects) | P3, P9 | ❌ | New; basis for compositionality. |
| Orientation / pose representation (not just position) | P3 | ❌ | New (extend grid code with orientation). |

## Phase 4 — Single sensorimotor cortical column

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| `CorticalColumn`: feature-at-location → object model | P7, P3 | ❌ | New; integrates Phases 1–3. |
| Sensorimotor inference: predict next sensation given a movement | P2, P7 | ❌ | New. |
| Object/pose belief update + disambiguation over a sensation sequence | P7, P9 | ❌ | New. |
| One/few-shot online object learning | P2, P9 | ❌ | New. |
| Active sensing policy (choose the next movement) | P2 | 🟡 | `Cortex/Executive/MotorCortex` + `Expanded/Active/ActiveNetwork` (active inference). |
| Goal-directed action / perception-action loop | P2 | 🟡 | `MotorCortex` closed-loop controller. |

## Phase 5 — Voting across columns (the "thousand brains")

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Lateral voting connections between columns | P1, P7 | ❌ | New; defining mechanism of TBT. |
| Consensus / union of object hypotheses across columns | P1, P2, P9 | ❌ | New. |
| Cross-modal voting (e.g. vision + touch) into a shared object id | P1, P9 | 🟡 | `Expanded/MultiModal/MultiModalNetwork`. |
| Voting / ensembling baselines | P1 | 🟡 | `Expanded` `EnsembleNetwork`, `MixtureOfExperts`, `RoutedNetwork`. |
| Faster consensus convergence (settling dynamics) | P9 | ❌ | New. |

## Phase 6 — Long-range connections & heterarchy

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Long-range cortico-cortical connections between distant columns/areas | P8 | ❌ | New. |
| Heterarchical (non-strictly-hierarchical) message passing | P8 | ❌ | New; the thesis of P8. |
| Hierarchical composition baseline | P8 | 🟡 | `Expanded/Hierarchical/HierarchicalNetwork`. |
| Thalamo-cortical routing / gating of long-range signals | P8 | 🟡 | `Cortex/Control/Thalamus`, `AssociationCortex`. |

## Phase 7 — Thousand-Brains system & sensorimotor agent

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Multi-column, multi-modal system assembly (Monty-style learning/sensor/motor modules) | P2, P9 | ❌ | New top-level orchestrator. |
| Sensorimotor agent loop: sensor → column(s) → vote → action → environment | P2, P9 | 🟡 | `MotorCortex` loop as a starting point. |
| Environment / embodiment interface (move sensor over an object) | P2, P9 | ❌ | New abstraction + simple environments. |
| Persistence of learned object models | P9 | 🟡 | `Core/Model/LayerStore`, LLM `Checkpoint` (KV-backed). |
| Rapid, robust learning/inference orchestration & policies | P9 | ❌ | New. |

## Phase 8 — Compositionality, behaviours & abstraction

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Objects composed of objects (nested reference frames via displacement) | P3, P9 | ❌ | New; builds on Phase 3 displacement cells. |
| Behaviours / sequences represented as objects in reference frames | P1, P3 | ❌ | New. |
| Abstract / conceptual reference frames (generalise beyond physical space) | P1 | ❌ | New; long-horizon. |

## Phase 9 — Evaluation, demos & tests

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Object-recognition benchmark (novel poses, occlusion, sensor noise; YCB-like) | P2, P9 | ❌ | New; mirror the `Benchmark` project's structure. |
| Sensorimotor environments / datasets | P2, P9 | ❌ | New. |
| Visual demo of columns sensing & voting | P1, P2 | 🟡 | Reuse WinUI patterns (`SAMPLES/PredictiveCoding.CortexApp`, `OmniMind`). |
| Unit tests for SDR, TM, grid frames, column, voting | P4, P7 | 🟡 | Reuse `TEST/PredicitiveCoding.UnitTests` (xUnit) conventions. |

---

## Background references (design inputs, not code)

| Task | Papers | Status | Notes |
|------|:------:|:------:|-------|
| Canonical neocortical microcircuit (themes & variations) informing the L2/3·L4·L5·L6 layer mapping | P5 | ✅ | Reference; partially realised by the `Cortex` modules. |
| Conceptual grounding & history of the cortical-column construct | P6 | ✅ | Reference; informs the column abstraction. |

---

## How this maps onto the existing stack

- **Core** (`PredictiveCoding`): tensors, the predictive-coding network (PCN), and
  prediction-error/free-energy — reused as the per-layer learning substrate and
  the surprise signal.
- **Expanded** (`PredictiveCoding.Expanded`): temporal, hierarchical, multimodal,
  Hopfield, ensemble/MoE and active-inference networks — reused as baselines and
  building blocks for sequence memory, voting and active sensing.
- **Cortex** (`PredictiveCoding.Cortex`): grid cells, entorhinal/hippocampal
  location coding, motor cortex (active inference) and the anatomical areas —
  reused directly for reference frames and the sensorimotor loop.

### Suggested namespace layout for new code

```
PredictiveCoding.Brain
├── Sdr/            (Phase 0)  Sdr, SdrEncoders, SdrMetrics
├── Columns/        (Phase 1, 4)  MiniColumn, CorticalColumn
├── Sequence/       (Phase 2)  SpatialPooler, TemporalMemory
├── ReferenceFrames/(Phase 3)  LocationModule, DisplacementCells
├── Voting/         (Phase 5)  ColumnVoting, Consensus
├── Connectivity/   (Phase 6)  LongRange, Heterarchy
├── System/         (Phase 7)  ThousandBrainsSystem, SensorimotorAgent, Environment
└── Eval/           (Phase 9)  benchmarks, demos
```

> Recommended build order: **Phase 0 → 2 → 3 → 4 → 5**, then 6–9. A single
> column that recognises a small set of objects by movement (end of Phase 4)
> is the first demonstrable milestone; multi-column voting (Phase 5) is the
> first result that is recognisably "Thousand Brains".
