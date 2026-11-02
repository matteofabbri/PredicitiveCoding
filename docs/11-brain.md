# 11. Brain — `PredictiveCoding.Brain` (Thousand Brains)

A **sensorimotor intelligence system** implementing the Hawkins / Numenta
**Thousand Brains** theory in code: many cortical columns, each learning models
of objects in its *own* reference frame, that **vote** to reach consensus. It
is the "moving sensor" answer to recognition — objects are identified by how
sensations change as the sensor moves, not by one static snapshot.

The project is specification-driven: nine papers (**P1–P9**, listed at the
bottom) *are* the spec, and every component traces back to one of them. The
[roadmap](../PredictiveCoding.Brain/ROADMAP.md) is **closed** — all phases
(0–9) are implemented, including the SE(3) 3-D extension and real-sensor
integration.

## Layout

```
PredictiveCoding.Brain/
  Sdr/             sparse distributed representations + encoders
  Columns/         the cortical column microcircuit
  Sequence/        HTM spatial pooler + temporal memory
  ReferenceFrames/ grid-cell locations, poses, path integration, displacement
  Voting/          lateral voting, consensus, settling, cross-modal voting
  Connectivity/    long-range connections, heterarchical graph message passing
  System/          the assembled system, agent, environments, policies
  Composition/     objects-of-objects, behaviours, conceptual frames
  Persistence/     saving / reloading learned object models
  Eval/            recognition benchmark + corrupted observation generators
  Viz/             self-contained HTML visualisation of the demo
```

## The stack, bottom-up

| Stage | Types | Idea |
|-------|-------|------|
| **Representations** | `Sdr` (overlap / union / intersect / subsample / noise), `ScalarEncoder`, `CategoryEncoder` | Everything is a sparse binary vector; `ToDense`/`FromDense` bridge to core tensors. |
| **Microcircuit** | `MiniColumn`, `LayeredColumn`, `KWinners` | The canonical L4-input · L2/3-output · L5-motor · L6a-location scaffold with k-winners sparsity. |
| **Sequence memory** | `SpatialPooler`, `TemporalMemory` | HTM-style: distal segments + synaptic permanences, predictive cells, bursting on surprise (`TemporalMemoryResult.Anomaly`). |
| **Reference frames** | `ObjectFrame`, `PoseEncoder`, `PathIntegrator`, `Pose2D`/`Pose3D`, `DisplacementModule`, `DisplacementRecognizer` (+ 3-D variants) | Location is a grid-cell code; movement is path-integrated; objects are recognised by their **displacement structure**, invariant under SE(2)/SE(3) rigid motion and partial views. |
| **The column** | `CorticalColumn` | Feature-at-location → object model: SP + TM + `ObjectMemory` with monotone hypothesis intersection over a sensation sequence; one/few-shot online learning. |
| **Voting** | `LateralVoting` (all-to-all / ring / local), `Consensus`, `SettlingDynamics`, `CrossModalSystem` | The "thousand brains" step: columns share object hypotheses and converge several times faster than any single column. `CrossModalSystem` votes across *different modalities* purely at the object-id level. |
| **Heterarchy** | `LongRangeConnection`, `HeterarchicalGraph`, `HeterarchicalMessagePassing` | Long-range cortico-cortical links and iterative belief propagation over an arbitrary directed graph — brain wiring is not a strict hierarchy (P8). |
| **The system** | `ThousandBrainsSystem`, `SensorimotorAgent`, `IEnvironment`, `IInferencePolicy` (`Random`, `Uncertainty`) | Assembles a heterarchy of multi-column systems (`TeachObject`, `Recognize`, `Reset`) and runs the sense → infer → act loop over an environment. |
| **Composition** | `CompositeObject`, `CompositionLibrary`, `BehaviorSequence`, `ConceptualFrame` | Objects made of objects (nested frames via displacement), behaviours as 1-D reference frames, concepts as N-D abstract frames. |

## Environments (embodiment)

- **`PointCloudEnvironment`** — synthetic objects as feature points; the sensor
  moves with `MoveNext` / `MovePrev` / `MoveRandom`.
- **`ImageSensorEnvironment`** — *real pixels*: objects are `float[,]`
  luminance images (any camera frame converts trivially) scanned by a movable
  patch sensor; features are dominant-gradient-orientation SDRs, locations come
  from the shared `PoseEncoder` grid. The sensorimotor loop runs unchanged on
  real images.
- Motor-driven path integration (`PathIntegrator`, `PathIntegrator3D`) closes
  the loop: egocentric `StepForward`/`Turn` update the location code, and
  closed motor loops return to the start.

## Persistence & evaluation

- **`ObjectModelStore` + `SdrCodec`** save learned object feature-point clouds
  as portable text; reloaded models stay recognisable (round-trip tests
  rehydrate a `DisplacementRecognizer`).
- **`RecognitionBenchmark`** teaches a single column and a voting population,
  then scores top-1 accuracy and mean steps under clean / occluded / noisy /
  pose-jittered conditions (corruptions from `ObservationCorruptions`,
  ambiguous-landmark datasets from `SyntheticObjectDataset`). Headline result:
  **voting converges in ≈3× fewer steps at equal accuracy**.

## Try it

Demo **option 9** in `PredictiveCoding.Main` prints the benchmark table, then
narrates one live recognition episode (a single column narrowing step-by-step
vs the voting population converging in a fraction of the steps), and writes an
animated self-contained HTML report to
`~/.predictivecoding/thousand-brains-demo.html` (`Viz/HtmlVisualizer`).

Tests: the `Brain*Tests` files in the unit-test suite (SDR, sequences, SE(2)/
SE(3) poses and path integration, image sensor, voting, connectivity, systems,
composition, eval, viz).

## Source papers (the specification)

| Key | Reference |
|:---:|-----------|
| P1 | Hawkins (2021). *A Thousand Brains: A New Theory of Intelligence.* |
| P2 | Clay et al. (2024). *The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence.* |
| P3 | Hawkins et al. (2019). *A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex.* |
| P4 | Hawkins & Ahmad (2016). *Why Neurons Have Thousands of Synapses…* |
| P5 | Harris & Shepherd (2015). *The neocortical circuit: themes and variations.* |
| P6 | Haueis (2016). *The life of the cortical column…* |
| P7 | Hawkins (n.d.). *A Theory of How Columns in the Neocortex Enable Learning the Structure of the World.* |
| P8 | Hawkins, Leadholm & Clay (2025). *Hierarchy or Heterarchy?…* |
| P9 | Leadholm et al. (2025). *Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference.* |
