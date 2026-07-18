# PredictiveCoding.Brain — Thousand Brains Project roadmap

> **Mission — this roadmap is one mandate: implement, in code, the theory and
> mechanisms described in the reference papers listed below (P1–P9).** It is not a
> loose wishlist of brain-inspired features; it is an engineering plan to realise
> the published Thousand Brains science. Every phase and every task exists to build
> a specific claim from those papers and is tagged with the paper key(s) it
> implements (e.g. *P4*, *P7*). The rule runs both ways: if a task cannot be traced
> back to one of these papers it does not belong here, and if a paper describes a
> mechanism we have not yet built it belongs here as a task.

The end state is a working **Thousand-Brains sensorimotor system** — many cortical
columns that each learn models of objects in their own reference frames and
**vote** to reach consensus — built on top of the existing predictive-coding stack
(`PredictiveCoding` Core, `PredictiveCoding.Expanded`, `PredictiveCoding.Cortex`).

Each task below carries a **status flag** so it is clear what can be reused and
what still has to be built.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Available** — already implemented in the repo; reuse directly. |
| 🟡 | **Partial** — a reusable building block exists; needs adaptation/extension. |
| ❌ | **To build** — not present yet. |

*Status reflects the repository as of 2026-06-06 (last updated 2026-07-18).*

> **All phases (0–9) are implemented and every former 🟡 partial has been promoted
> to ✅.** 109 unit tests cover the stack, and a runnable benchmark + demo ship as
> option 9 in `PredictiveCoding.Main`. Beyond the original roadmap we have also added
> pose-invariant recognition (`DisplacementRecognizer` — survives SE(2) rigid
> motion), cross-modal voting (`CrossModalSystem`), motor-driven path integration
> (`PathIntegrator`), and object-model persistence (`ObjectModelStore`).
>
> **2026-07-18 — the two remaining algorithmic items are now done:**
> - **3-D (SE(3)) extension** — `ReferenceFrames/Pose3D.cs` (quaternion rigid-pose
>   algebra; yaw about +Z matching `Pose2D`, anchor-case-pinned conventions),
>   `Composition/FeaturePoint3D.cs`, `ReferenceFrames/PoseEncoder3D.cs`,
>   `ReferenceFrames/PathIntegrator3D.cs` (closed 3-D motor loops return to the
>   start) and `ReferenceFrames/DisplacementRecognizer3D.cs` (pairwise distances
>   are invariant under all of SE(3), so the displacement signature carries over —
>   objects are recognised after arbitrary 3-D rotation + translation, occlusion
>   included).
> - **Real-sensor integration** — `System/ImageSensorEnvironment.cs`: an
>   `IEnvironment` whose objects are real images (`float[,]` luminance — any
>   camera frame converts trivially) scanned by a movable patch sensor; features
>   are dominant-gradient-orientation SDRs, locations come from the shared
>   `PoseEncoder` grid. The sensorimotor loop runs unchanged on real pixels.
>
> **2026-07-18 — visual front-end done too**: `Viz/VizData.cs` + `Viz/HtmlVisualizer.cs`
> render the demo as a self-contained animated web page (episode player, benchmark,
> pose-invariance, real-sensor scan — all CSS/JS inline, zero external resources);
> demo option 9 writes `~/.predictivecoding/thousand-brains-demo.html` and opens it.
> **The roadmap is closed** — a native WinUI shell remains possible but has no
> outstanding claim behind it.

### Source papers — *the specification this roadmap implements*

These nine works **are** the specification. Every task in this document traces back
to one or more of them and is referenced throughout by its key (**P1–P9**).

| Key | Reference |
|:---:|-----------|
| **P1** | Hawkins, J. (2021). *A Thousand Brains: A New Theory of Intelligence.* (Foreword by Richard Dawkins.) |
| **P2** | Clay, V. et al. (2024). *The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence.* |
| **P3** | Hawkins, J. et al. (2019). *A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex.* Frontiers in Neural Circuits. |
| **P4** | Hawkins, J. & Ahmad, S. (2016). *Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex.* Frontiers in Neural Circuits. |
| **P5** | Harris, K.D. & Shepherd, G.M.G. (2015). *The neocortical circuit: themes and variations.* Nature Neuroscience. |
| **P6** | Haueis, P. (2016). *The life of the cortical column: opening the domain of functional architecture of the cortex (1955–1981).* History and Philosophy of the Life Sciences. |
| **P7** | Hawkins, J. (n.d.). *A Theory of How Columns in the Neocortex Enable Learning the Structure of the World.* Numenta whitepaper. |
| **P8** | Hawkins, J., Leadholm, N., Clay, V. (2025). *Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain.* |
| **P9** | Leadholm, N., Clay, V., Knudstrup, S., Lee, H., Hawkins, J. (2025). *Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference.* |

---

## Phase 0 — Representations & foundations · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Sparse Distributed Representation (SDR) type: overlap, union, intersect, subsample, add-noise | P1, P4 | ✅ | `Sdr/Sdr.cs`. |
| SDR similarity & metrics (overlap, overlap score, match) | P4 | ✅ | `Sdr.Overlap` / `OverlapScore` / `Matches`. |
| Scalar / category / coordinate → SDR encoders | P3 | ✅ | `Sdr/ScalarEncoder.cs`, `Sdr/CategoryEncoder.cs`; coordinates via `Cortex/GridCellEncoder`. |
| Dense ↔ SDR bridges to `DenseTensor<float>` | — | ✅ | `Sdr.ToDense` / `Sdr.FromDense`. |
| Deterministic RNG / seeding utilities | — | ✅ | `Core/RandomUtils`. |
| Unit tests (12) for SDR + encoders | P4 | ✅ | `TEST/PredicitiveCoding.UnitTests/BrainSdrTests.cs`. |

## Phase 1 — Cortical microcircuit substrate · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Mini-column abstraction (group of cells sharing feed-forward input, L4) | P5, P6, P7 | ✅ | `Columns/MiniColumn.cs`. |
| Layered column scaffold mapped to the canonical microcircuit (L4 input · L2/3 object output · L5 motor · L6a location) | P5, P6 | ✅ | `Columns/LayeredColumn.cs` (`LayeredColumnConfig`, `LayeredColumnOutput`). |
| Anatomical cortical-area modules to host columns | P5 | ✅ | `Cortex` (`VisualCortex`, `AssociationCortex`, `Sensory/*`, `Thalamus`, …). |
| Inhibition / k-winners-take-all for sparsity | P4 | ✅ | `Columns/KWinners.cs` (`SelectTopK`, `SelectTopKPercent`, `LocalInhibition`). |

## Phase 2 — Sequence memory (HTM Temporal Memory) · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Distal dendritic segments + synaptic permanences | P4 | ✅ | `Sequence/TemporalMemory.cs` (Segment / Synapse). |
| Temporal Memory: high-order sequence learning, predictive cells, bursting on surprise | P4 | ✅ | `Sequence/TemporalMemory.cs`. |
| Spatial Pooler: stable sparse encoding of input → SDR | P4 | ✅ | `Sequence/SpatialPooler.cs`. |
| Anomaly / surprise (prediction-error) signal | P4 | ✅ | `TemporalMemoryResult.Anomaly` (fraction of bursting columns). |
| Sequence-model baselines for comparison | P4 | 🟡 | `Expanded` `TemporalNetwork`, `DeepRecurrentNetwork`, `GatedStateNetwork`. |
| Associative recall of stored patterns | P4 | 🟡 | `Expanded/Hopfield` (`HopfieldMemory`, `HopfieldNetwork`). |
| Unit tests (4) for SP + TM (sequence learning & prediction) | P4 | ✅ | `TEST/PredicitiveCoding.UnitTests/BrainSequenceTests.cs`. |

## Phase 3 — Reference frames & location (grid cells) · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Grid-cell location module (multi-scale, hexagonal population code) | P3 | ✅ | `Cortex/Memory/GridCellEncoder`. |
| Path integration: movement vector → updated location code | P3 | ✅ | `ReferenceFrames/PathIntegrator.cs` (`Step` world-frame displacement, `StepForward`/`Turn` egocentric, `LocationCode` via PoseEncoder; closed motor loops return to the start). `EntorhinalCortex` remains the spatial-association baseline. |
| Location ↔ feature predictive association | P3, P7 | 🟡 | `Cortex/Memory/EntorhinalCortex` learns location→observation. |
| Object-centric reference frames (anchor location to an object, not the room) | P3, P7 | ✅ | `ReferenceFrames/ObjectFrame.cs` (`VotesFor`, `MatchingFeatures`, `MatchingLocations`). |
| Displacement cells (relative pose between features / sub-objects) | P3, P9 | ✅ | `ReferenceFrames/DisplacementModule.cs` (`Encode`, `Apply`, `Invert`, `Between`); continuous-pose algebra in `ReferenceFrames/Pose2D.cs`. |
| Pose-invariant recognition (match by displacement structure, not absolute location) | P3, P9 | ✅ | `ReferenceFrames/DisplacementRecognizer.cs` — pairwise-distance signature; recognises objects after SE(2) rigid motion (rotation + translation) and partial views. Resolves the absolute-location limitation of `ObjectMemory`. |
| Orientation / pose representation (not just position) | P3 | ✅ | `ReferenceFrames/PoseEncoder.cs` (`Encode`, `EncodeXY`, `EncodeTheta`). |

## Phase 4 — Single sensorimotor cortical column · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| `CorticalColumn`: feature-at-location → object model | P7, P3 | ✅ | `Columns/CorticalColumn.cs` (SP + TM + ObjectMemory + hypothesis tracking). |
| Sensorimotor inference: predict next sensation given a movement | P2, P7 | ✅ | `CorticalColumn.Infer` runs a full sensation sequence; anomaly from TM. |
| Object/pose belief update + disambiguation over a sensation sequence | P7, P9 | ✅ | `ObjectMemory` + monotone hypothesis intersection in `CorticalColumn.Compute`. |
| One/few-shot online object learning | P2, P9 | ✅ | `CorticalColumn.LearnObject` / `StartNewObject` + `Compute(learn:true)`. |
| Active sensing policy (choose the next movement) | P2 | 🟡 | `Cortex/Executive/MotorCortex` + `Expanded/Active/ActiveNetwork` (active inference). |
| Goal-directed action / perception-action loop | P2 | 🟡 | `MotorCortex` closed-loop controller. |

## Phase 5 — Voting across columns (the "thousand brains") · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Lateral voting connections between columns | P1, P7 | ✅ | `Voting/LateralVoting.cs` (AllToAll / Ring / Local topologies; `Broadcast`, `Neighbors`). |
| Consensus / union of object hypotheses across columns | P1, P2, P9 | ✅ | `Voting/Consensus.cs` (`Compute`, `AgreementScore`, `TopCandidatesByAgreement`). |
| Cross-modal voting (e.g. vision + touch) into a shared object id | P1, P9 | ✅ | `Voting/CrossModalSystem.cs` — one column per modality with **different InputSizes**, voting purely at the object-id level (no same-size SDR union); either modality alone may stay ambiguous, together they disambiguate. `Expanded/MultiModal/MultiModalNetwork` remains a baseline. |
| Voting / ensembling baselines | P1 | 🟡 | `Expanded` `EnsembleNetwork`, `MixtureOfExperts`, `RoutedNetwork`. |
| Faster consensus convergence (settling dynamics) | P9 | ✅ | `Voting/SettlingDynamics.cs` (`Settle`, `SettleFromCurrent`); `MultiColumnSystem.RunToConsensus`. |

## Phase 6 — Long-range connections & heterarchy · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Long-range cortico-cortical connections between distant columns/areas | P8 | ✅ | `Connectivity/LongRangeConnection.cs` (ConnectionType, Weight); `Connectivity/HeterarchicalGraph.cs` (AddArea, Connect, OutgoingConnections, IncomingConnections). |
| Heterarchical (non-strictly-hierarchical) message passing | P8 | ✅ | `Connectivity/HeterarchicalMessagePassing.cs` (PassMessages, GatherIncomingHypotheses, IsAllConverged); iterative belief propagation over arbitrary directed graphs. |
| Hierarchical composition baseline | P8 | 🟡 | `Expanded/Hierarchical/HierarchicalNetwork`. |
| Thalamo-cortical routing / gating of long-range signals | P8 | 🟡 | `Cortex/Control/Thalamus`, `AssociationCortex`. |

## Phase 7 — Thousand-Brains system & sensorimotor agent · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Multi-column, multi-modal system assembly (Monty-style learning/sensor/motor modules) | P2, P9 | ✅ | `System/ThousandBrainsSystem.cs` — assembles HeterarchicalGraph of MultiColumnSystems; wires inter-area connections; `TeachObject`, `ProcessObservation`, `Recognize`, `Reset`. |
| Sensorimotor agent loop: sensor → column(s) → vote → action → environment | P2, P9 | ✅ | `System/SensorimotorAgent.cs` — `RunEpisode` (sense→infer→act loop), `Learn`, `LearnAll`. |
| Environment / embodiment interface (move sensor over an object) | P2, P9 | ✅ | `System/IEnvironment.cs` + `System/PointCloudEnvironment.cs` (MoveNext / MovePrev / MoveRandom). |
| Persistence of learned object models | P9 | ✅ | `Persistence/ObjectModelStore.cs` + `Persistence/SdrCodec.cs` — saves/reloads object feature-point clouds as portable text; reloaded models stay recognisable (round-trip test rehydrates a `DisplacementRecognizer`). |
| Rapid, robust learning/inference orchestration & policies | P9 | ✅ | `System/IInferencePolicy.cs`, `System/RandomPolicy.cs`, `System/UncertaintyPolicy.cs`. |

## Phase 8 — Compositionality, behaviours & abstraction · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Objects composed of objects (nested reference frames via displacement) | P3, P9 | ✅ | `ReferenceFrames/Pose2D.cs` (continuous relative-pose algebra: `Apply`, `Compose`, `Inverse`) + `Composition/` (`FeaturePoint`, `ObjectPart`, `CompositeObject`, `CompositionLibrary`). `CompositionLibrary.Resolve` flattens nested frames with cycle/depth guards; `ToObservations` encodes the result for a `CorticalColumn`. |
| Behaviours / sequences represented as objects in reference frames | P1, P3 | ✅ | `Composition/BehaviorSequence.cs` — a 1-D reference frame where location = position in the sequence and feature = state; `PredictNext`, `MatchScore`, `ToObjectPoints`. |
| Abstract / conceptual reference frames (generalise beyond physical space) | P1 | ✅ | `Composition/ConceptualFrame.cs` — N-D abstract frame; `EncodeLocation`, `AddConcept`, `Recall`/`Nearest`, and `Move` (the path-integration analogue in concept space). |

## Phase 9 — Evaluation, demos & tests · ✅ done

| Task | Papers | Status | Reuse / notes |
|------|:------:|:------:|---------------|
| Object-recognition benchmark (novel poses, occlusion, sensor noise; YCB-like) | P2, P9 | ✅ | `Eval/RecognitionBenchmark.cs` (+ `BenchmarkConfig`/`BenchmarkResult`/`ConditionResult`/`RecognizerScore`). Teaches a single column and a voting population, scores top-1 accuracy + mean steps under clean / occlusion / sensor-noise / pose-jitter; quantifies the voting speed-up (≈3× fewer steps at equal accuracy). |
| Sensorimotor environments / datasets | P2, P9 | ✅ | `Eval/SyntheticObjectDataset.cs` (shared location grid + shared feature vocabulary ⇒ ambiguous landmarks, YCB-like) and `Eval/ObservationCorruptions.cs` (occlude / add-sensor-noise / jitter-pose). Inference-time embodiment via Phase 7 `PointCloudEnvironment`. |
| Visual demo of columns sensing & voting | P1, P2 | ✅ | `PredictiveCoding.Main/Demos/ThousandBrainsDemo.cs` — **option 9 in Main**: prints the benchmark table, then narrates one live recognition episode (single column narrowing step-by-step vs the voting population converging in a fraction of the steps). Console visualisation; a WinUI front-end remains optional. |
| Unit tests for SDR, TM, grid frames, column, voting | P4, P7 | ✅ | 56 Brain tests across `BrainSdrTests`, `BrainSequenceTests`, `BrainVotingTests`, `BrainConnectivityTests`, `BrainSystemTests`, `BrainCompositionTests`, `BrainEvalTests` in `TEST/PredicitiveCoding.UnitTests`. |

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
├── Composition/    (Phase 8)  CompositeObject, CompositionLibrary, BehaviorSequence, ConceptualFrame
└── Eval/           (Phase 9)  benchmarks, demos
```

> Recommended build order: **Phase 0 → 2 → 3 → 4 → 5**, then 6–9. A single
> column that recognises a small set of objects by movement (end of Phase 4)
> is the first demonstrable milestone; multi-column voting (Phase 5) is the
> first result that is recognisably "Thousand Brains".
