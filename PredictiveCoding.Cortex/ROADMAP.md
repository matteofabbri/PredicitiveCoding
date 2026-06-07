# PredictiveCoding.Cortex — cortical-areas roadmap

> **Mission — extend the anatomical predictive-coding cortex model with the brain
> regions it was still missing.** The project began with 15 areas (the ventral stream
> + 14 regions across Sensory / Memory / Executive / Control). This roadmap added
> **three whole missing systems** — **neuromodulation**, the **limbic / affective**
> system, and **language** — plus a set of targeted single-area gaps. **All groups are
> now implemented** (36 areas total).

Same design contract as every existing area: each region is an `IDisposable` (or a
pure controller) that **composes already-tested primitives** from `PredictiveCoding`
(Core) and `PredictiveCoding.Expanded`; learning stays **local predictive coding** (no
backprop). The deliberate consequence is that **none of these areas needed a new
learning algorithm** — they are assemblies of building blocks that already existed and
are unit-tested.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Available** — implemented in `PredictiveCoding.Cortex`. |
| 🟡 | **Ready to assemble** — a reusable primitive exists; wrapper not yet written. |
| ❌ | **To build** — needs a genuinely new primitive or mechanism. |

*Status reflects the repository as of 2026-06-07.*

---

## Pre-existing areas · ✅

| Folder | Areas |
|--------|-------|
| *(root)* | **VisualCortex** (ventral stream V1→IT), `RetinotopicMap` |
| `Sensory/` | **AuditoryCortex**, **SomatosensoryCortex**, **DorsalStream**, **RetinaLgn** |
| `Memory/` | **HippocampalFormation**, **EntorhinalCortex** (+ `GridCellEncoder`), **PiriformCortex** |
| `Executive/` | **PrefrontalCortex**, **MotorCortex**, **CerebellarCortex** |
| `Control/` | **AssociationCortex**, **Thalamus**, **BasalGanglia**, **DefaultModeNetwork** |

---

## Group A — Neuromodulation · `Neuromodulation/` · ✅ done

Completes the **control layer**. In predictive coding the knobs for precision and
learning-rate *are* the neuromodulators; `Thalamus` (precision) and `BasalGanglia`
(action) already existed, so these are small controllers in the same style.
Implemented 2026-06-07 (4 controllers, 15 tests in `CortexNeuromodulationTests`).

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **DopaminergicSystem** (VTA / SNc) | Reward-prediction-error (TD) + average-reward baseline → modulates plasticity | `RewardPredictionError`, `Step`, `ModulateLearningRate` | ✅ |
| **LocusCoeruleus** (noradrenaline) | *Unexpected* uncertainty: global surprise spike → broad precision reset | `Thalamus.Modulate` pattern (`Network.SetTopLayerPrior` / `ComputeEnergy`) | ✅ |
| **NucleusBasalis** (acetylcholine) | *Expected* uncertainty: per-area EMA → per-area learning-rate | per-area controller (`Update`, `ModulateLearningRates`) | ✅ |
| **RapheNuclei** (serotonin) | Time-horizon / patience → effective discount factor γ | scalar controller (`DiscountFactor`, `TimeHorizon`) | ✅ |

## Group B — Limbic / affective · `Limbic/` · ✅ done

The emotion / valuation axis; plugs into `Thalamus`, `BasalGanglia`, `PrefrontalCortex`.
Implemented 2026-06-07 (5 areas, 6 tests in `CortexLimbicTests`).

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **Amygdala** | Salience / valence; emits a SalienceGain to steer Thalamus | `Network` (PCN) → affect vector | ✅ |
| **OrbitofrontalCortex** | Expected outcome value → feeds `BasalGanglia` | `Network` → scalar value | ✅ |
| **CingulateCortex (ACC)** | Conflict (gate entropy) & surprise over area free-energies | pure controller (`MixtureOfExperts.ComputeGates`) | ✅ |
| **InsularCortex** | Interoception (body-state stream → feeling latent) | `DeepRecurrentNetwork` | ✅ |
| **Hypothalamus** | Homeostatic set-points → drive + goal for `MotorCortex` | pure controller | ✅ |

## Group C — Language · `Language/` · ✅ done

Bridges the cortex toward `PredictiveCoding.LLM`. Implemented 2026-06-07
(3 areas, 3 tests in `CortexLanguageTests`).

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **WernickeArea** | Comprehension: token sequence → semantic latent | `DeepRecurrentNetwork` | ✅ |
| **BrocaArea** | Production: semantic latent → utterance (pairs with `MotorCortex`) | `GenerativeNetwork` | ✅ |
| **AngularGyrus** | Relational semantic hub (concept graph completion) | `GraphPredictiveNetwork` | ✅ |

## Group D — Additional sensory · `Sensory/` · ✅ done

Implemented 2026-06-07 (4 areas, 4 tests in `CortexSensoryExtraTests`).

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **OlfactoryBulb** | Decorrelation / whitening front-end to `PiriformCortex` | pure transform (mirrors `RetinaLgn`) | ✅ |
| **GustatoryCortex** | Taste: content-addressable memory | `HopfieldNetwork` (as `PiriformCortex`) | ✅ |
| **VestibularSystem** | Self-motion integration → head-direction code | `DeepRecurrentNetwork` | ✅ |
| **FrontalEyeFields** | Saccades / overt attention: select next `RetinotopicMap` patch | saliency + IOR over `RetinotopicMap` | ✅ |

## Group E — Additional memory / MTL · `Memory/` · ✅ done

Implemented 2026-06-07 (2 areas, 2 tests in `CortexMemoryExtraTests`).

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **PerirhinalCortex** | Item familiarity / novelty (recognition memory) | `Network` encoder + `MemoryBank` | ✅ |
| **ParahippocampalCortex** | Scene / context gist ("place area") | `VisualCortex` gist + context `Network` | ✅ |

## Group F — Additional motor / integration · `Executive/`, `Control/` · ✅ done

Implemented 2026-06-07 (3 areas, 3 tests in `CortexIntegrationTests`).

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **PosteriorParietalCortex** | Coordinate transforms / intentional maps (dorsal→motor) | `MultiModalNetwork` | ✅ |
| **PremotorCortex / SMA** | Action sequencing / chunking → feeds `MotorCortex` | `TemporalMixtureOfExperts` | ✅ |
| **Claustrum** | Global binding / workspace synchronisation (consensus + coherence) | pure controller | ✅ |

---

## Namespace layout (delivered)

```
PredictiveCoding.Cortex
├── Neuromodulation/  (Group A)  DopaminergicSystem, LocusCoeruleus, NucleusBasalis, RapheNuclei
├── Limbic/           (Group B)  Amygdala, OrbitofrontalCortex, CingulateCortex, InsularCortex, Hypothalamus
├── Language/         (Group C)  WernickeArea, BrocaArea, AngularGyrus
├── Sensory/          (Group D)  OlfactoryBulb, GustatoryCortex, VestibularSystem, FrontalEyeFields (+ existing)
├── Memory/           (Group E)  PerirhinalCortex, ParahippocampalCortex (+ existing)
├── Executive/        (Group F)  PremotorCortex (+ existing)
└── Control/          (Group F)  PosteriorParietalCortex, Claustrum (+ existing)
```

Tests: `CortexNeuromodulationTests` (15), `CortexLimbicTests` (6), `CortexLanguageTests` (3),
`CortexSensoryExtraTests` (4), `CortexMemoryExtraTests` (2), `CortexIntegrationTests` (3)
— **33 new tests**, all green, alongside the pre-existing cortex tests.

## Build order (all complete)

1. ✅ **Group A — Neuromodulation** (2026-06-07).
2. ✅ **Group B — Limbic** (2026-06-07).
3. ✅ **Group C — Language** (2026-06-07).
4. ✅ **Groups D / E / F** — extra sensory, memory, motor/integration (2026-06-07).

> Possible future work (genuinely new mechanism, not just composition): an
> active-inference saccade policy for `FrontalEyeFields` (current version is bottom-up
> saliency); wiring the neuromodulators/limbic controllers into an end-to-end demo.

---

## Reference papers — *design inputs grounding each group*

| Key | Reference | Group |
|:---:|-----------|:-----:|
| **N1** | Friston, K. (2005). *A theory of cortical responses.* Phil. Trans. R. Soc. B. — precision = gain. | A |
| **N2** | Bogacz, R. (2017). *A tutorial on the free-energy framework.* J. Math. Psychol. | A, B |
| **N3** | Schultz, W., Dayan, P., Montague, P.R. (1997). *A neural substrate of prediction and reward.* Science. — dopamine RPE. | A |
| **N4** | Aston-Jones, G. & Cohen, J.D. (2005). *Adaptive gain and the locus coeruleus.* Annu. Rev. Neurosci. | A |
| **N5** | Yu, A.J. & Dayan, P. (2005). *Uncertainty, neuromodulation, and attention.* Neuron. — ACh vs NE. | A |
| **N6** | Doya, K. (2002). *Metalearning and neuromodulation.* Neural Networks. — serotonin & time scale. | A |
| **N7** | Botvinick, M. et al. (2001). *Conflict monitoring and cognitive control.* Psychol. Rev. — ACC. | B |
| **N8** | LeDoux, J. (2000). *Emotion circuits in the brain.* Annu. Rev. Neurosci. — amygdala. | B |
| **N9** | Rushworth, M.F.S. et al. (2011). *Frontal cortex and reward-guided learning.* Neuron. — OFC. | B |
| **N10** | Craig, A.D. (2002). *How do you feel? Interoception.* Nat. Rev. Neurosci. — insula. | B |
| **N11** | Hickok, G. & Poeppel, D. (2007). *The cortical organization of speech processing.* Nat. Rev. Neurosci. | C |
| **N12** | Binder, J.R. & Desai, R.H. (2011). *The neurobiology of semantic memory.* Trends Cogn. Sci. | C |
| **N13** | Itti, L. & Koch, C. (2001). *Computational modelling of visual attention.* Nat. Rev. Neurosci. — saliency. | D |
| **N14** | Taube, J.S. (2007). *The head direction signal.* Annu. Rev. Neurosci. | D |
| **N15** | Epstein, R. & Kanwisher, N. (1998). *A cortical representation of the local visual environment.* Nature. | E |
| **N16** | Brown, M.W. & Aggleton, J.P. (2001). *Recognition memory: perirhinal vs hippocampus.* Nat. Rev. Neurosci. | E |
| **N17** | Andersen, R.A. & Buneo, C.A. (2002). *Intentional maps in posterior parietal cortex.* Annu. Rev. Neurosci. | F |
| **N18** | Crick, F.C. & Koch, C. (2005). *What is the function of the claustrum?* Phil. Trans. R. Soc. B. | F |

> Note: unlike `PredictiveCoding.Brain/ROADMAP.md` (bound to the Thousand-Brains
> papers P1–P9), this roadmap is grounded in general systems-neuroscience and
> predictive-coding references. Keep the two roadmaps separate.
