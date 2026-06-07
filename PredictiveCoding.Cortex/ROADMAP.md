# PredictiveCoding.Cortex — cortical-areas roadmap

> **Mission — extend the anatomical predictive-coding cortex model with the brain
> regions it is still missing.** Today the project models 15 areas (the ventral
> stream + 14 regions across Sensory / Memory / Executive / Control). This roadmap
> captures the *next* areas to build, chosen to close **three whole missing
> systems** — **neuromodulation**, the **limbic / affective** system, and
> **language** — plus a set of targeted single-area gaps.

Same design contract as every existing area: each region is an `IDisposable` that
**composes already-tested primitives** from `PredictiveCoding` (Core) and
`PredictiveCoding.Expanded`; learning stays **local predictive coding** (no
backprop). The deliberate consequence is that **none of these areas needs a new
learning algorithm** — they are assemblies of building blocks that already exist
and are unit-tested. That is what makes the whole list low-risk.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Available** — already implemented in `PredictiveCoding.Cortex`. |
| 🟡 | **Ready to assemble** — a directly reusable primitive exists; only the area wrapper + wiring is new. |
| ❌ | **To build** — needs a genuinely new primitive or non-trivial mechanism. |

*Status reflects the repository as of 2026-06-07.*

---

## Already implemented · ✅

| Folder | Areas |
|--------|-------|
| *(root)* | **VisualCortex** (ventral stream V1→IT), `RetinotopicMap` |
| `Sensory/` | **AuditoryCortex**, **SomatosensoryCortex**, **DorsalStream**, **RetinaLgn** |
| `Memory/` | **HippocampalFormation**, **EntorhinalCortex** (+ `GridCellEncoder`), **PiriformCortex** |
| `Executive/` | **PrefrontalCortex**, **MotorCortex**, **CerebellarCortex** |
| `Control/` | **AssociationCortex**, **Thalamus**, **BasalGanglia**, **DefaultModeNetwork** |
| `Neuromodulation/` | **DopaminergicSystem**, **LocusCoeruleus**, **NucleusBasalis**, **RapheNuclei** *(2026-06-07)* |

---

## Group A — Neuromodulation · `Neuromodulation/` · ✅ done

Completes the **control layer**. In predictive coding the knobs for precision and
learning-rate *are* the neuromodulators; `Thalamus` (precision) and `BasalGanglia`
(action) already exist, so these are small controllers in the same style.
**Implemented 2026-06-07** — four pure controllers + 15 unit tests in
`CortexNeuromodulationTests`.

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **DopaminergicSystem** (VTA / SNc) | Reward-prediction-error (TD) + average-reward baseline → modulates plasticity | `RewardPredictionError`, `Step`, `ModulateLearningRate` | ✅ |
| **LocusCoeruleus** (noradrenaline) | *Unexpected* uncertainty: global surprise spike → broad precision reset across areas | `Thalamus.Modulate` pattern (`Network.SetTopLayerPrior` / `ComputeEnergy`) | ✅ |
| **NucleusBasalis** (acetylcholine) | *Expected* uncertainty: per-area EMA → per-area learning-rate modulation | per-area controller (`Update`, `ModulateLearningRates`) | ✅ |
| **RapheNuclei** (serotonin) | Time-horizon / patience → effective discount factor γ | scalar controller (`DiscountFactor`, `TimeHorizon`) | ✅ |

## Group B — Limbic / affective · `Limbic/` (new)

The emotion / valuation axis is currently absent. All of these plug into the
existing `Thalamus`, `BasalGanglia` and `PrefrontalCortex`.

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **Amygdala** | Salience / valence / threat: multimodal latent → arousal+valence; modulates `Thalamus` (threat → more gain) | `Network` (small PCN) / `MultiModalNetwork` | 🟡 |
| **OrbitofrontalCortex** | Expected outcome value → feeds `BasalGanglia` go/no-go | `Network` / `MixtureOfExperts` | 🟡 |
| **CingulateCortex (ACC)** | Conflict / surprise monitoring over area free-energies → engages `PrefrontalCortex` control | `Thalamus`-style energy reader (`Network.ComputeEnergy`) | 🟡 |
| **InsularCortex** | Interoception (body-state stream) → feeling states | `TemporalNetwork` | 🟡 |
| **Hypothalamus** | Homeostatic set-points / drives → generates goals for `MotorCortex` | reference controller for `ActiveNetwork` | 🟡 |

## Group C — Language · `Language/` (new)

Whole missing system; naturally bridges `PredictiveCoding.Cortex` to
`PredictiveCoding.LLM`.

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **WernickeArea** | Comprehension: phoneme / token sequence → semantic latent | `TemporalNetwork` / `DeepRecurrentNetwork` | 🟡 |
| **BrocaArea** | Production: semantic latent → phoneme / motor sequence (pairs with `MotorCortex`) | `GenerativeNetwork` + `TemporalNetwork` | 🟡 |
| **AngularGyrus** | Cross-modal semantic hub | `GraphPredictiveNetwork` / `MultiModalNetwork` | 🟡 |

## Group D — Additional sensory · `Sensory/`

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **OlfactoryBulb** | Decorrelation / whitening front-end to `PiriformCortex` (mirror of `RetinaLgn`→`VisualCortex`) | `RetinaLgn` DoG/whitening pattern | 🟡 |
| **GustatoryCortex** | Taste: content-addressable memory | `HopfieldNetwork` (as `PiriformCortex`) | 🟡 |
| **VestibularSystem** (+ head-direction) | Motion integration → head-direction code; pairs with grid cells (path integration) | `TemporalNetwork` | 🟡 |
| **FrontalEyeFields + SuperiorColliculus** | Saccades / overt attention: choose the next `RetinotopicMap` patch | `ActiveNetwork` + `AttentiveNetwork` over `RetinotopicMap` | ❌ |

## Group E — Additional memory / MTL · `Memory/`

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **PerirhinalCortex** | Item familiarity / novelty (recognition memory) | `MemoryNetwork` / `MemoryBank` | 🟡 |
| **ParahippocampalCortex** (place area) | Scene / context gist; complements grid cells | `VisualCortex` latent + `EntorhinalCortex` | 🟡 |

## Group F — Additional motor / integration · `Executive/`, `Control/`

| Area | Function | Reuse / primitive | Status |
|------|----------|-------------------|:------:|
| **PosteriorParietalCortex** | Coordinate transforms (retina+eye → body); dorsal→motor bridge | `MultiModalNetwork` | 🟡 |
| **PremotorCortex / SMA** | Action sequencing / chunking → feeds `MotorCortex` | `TemporalMixtureOfExperts` | 🟡 |
| **Claustrum** | Global binding / synchronization ("conductor"; global workspace) | `MultiModalNetwork` | 🟡 |

---

## Suggested namespace layout for new code

```
PredictiveCoding.Cortex
├── Neuromodulation/  (Group A)  DopaminergicSystem, LocusCoeruleus, NucleusBasalis, RapheNuclei
├── Limbic/           (Group B)  Amygdala, OrbitofrontalCortex, CingulateCortex, InsularCortex, Hypothalamus
├── Language/         (Group C)  WernickeArea, BrocaArea, AngularGyrus
├── Sensory/          (Group D)  OlfactoryBulb, GustatoryCortex, VestibularSystem, FrontalEyeFields
├── Memory/           (Group E)  PerirhinalCortex, ParahippocampalCortex
├── Executive/        (Group F)  PremotorCortex
└── Control/          (Group F)  PosteriorParietalCortex, Claustrum
```

Each area ships with xUnit coverage in `TEST/PredicitiveCoding.UnitTests` (mirror the
existing `Brain*`/cortex tests).

## Recommended build order

1. **Group A — Neuromodulation.** ✅ **Done (2026-06-07).** Smallest code, biggest
   systemic effect; reuses the `Thalamus` / `BasalGanglia` controller pattern and makes
   precision + learning-rate *dynamic* across the whole system.
2. **Group B — Limbic.** `Amygdala` + `OrbitofrontalCortex` + `CingulateCortex` open
   the affective/valuation axis and plug straight into A's outputs and the existing
   `Thalamus`/`BasalGanglia`/`PrefrontalCortex`.
3. **Group C — Language.** Connects the cortex to `PredictiveCoding.LLM`.
4. **Groups D / E / F** — opportunistic, as time allows.

---

## Reference papers — *design inputs grounding each group*

| Key | Reference | Group |
|:---:|-----------|:-----:|
| **N1** | Friston, K. (2005). *A theory of cortical responses.* Phil. Trans. R. Soc. B. — precision = gain. | A |
| **N2** | Bogacz, R. (2017). *A tutorial on the free-energy framework for modelling perception and learning.* J. Math. Psychol. | A, B |
| **N3** | Schultz, W., Dayan, P., Montague, P.R. (1997). *A neural substrate of prediction and reward.* Science. — dopamine RPE. | A |
| **N4** | Aston-Jones, G. & Cohen, J.D. (2005). *An integrative theory of locus coeruleus–norepinephrine function: adaptive gain.* Annu. Rev. Neurosci. | A |
| **N5** | Yu, A.J. & Dayan, P. (2005). *Uncertainty, neuromodulation, and attention.* Neuron. — ACh (expected) vs NE (unexpected) uncertainty. | A |
| **N6** | Botvinick, M. et al. (2001). *Conflict monitoring and cognitive control.* Psychol. Rev. — ACC. | B |
| **N7** | LeDoux, J. (2000). *Emotion circuits in the brain.* Annu. Rev. Neurosci. — amygdala. | B |
| **N8** | Rushworth, M.F.S. et al. (2011). *Frontal cortex and reward-guided learning and decision-making.* Neuron. — OFC value. | B |
| **N9** | Craig, A.D. (2002). *How do you feel? Interoception.* Nat. Rev. Neurosci. — insula. | B |
| **N10** | Hickok, G. & Poeppel, D. (2007). *The cortical organization of speech processing.* Nat. Rev. Neurosci. — dual-stream language. | C |
| **N11** | Taube, J.S. (2007). *The head direction signal.* Annu. Rev. Neurosci. | D |
| **N12** | Epstein, R. & Kanwisher, N. (1998). *A cortical representation of the local visual environment.* Nature. — parahippocampal place area. | E |
| **N13** | Brown, M.W. & Aggleton, J.P. (2001). *Recognition memory: perirhinal vs hippocampus.* Nat. Rev. Neurosci. | E |
| **N14** | Andersen, R.A. & Buneo, C.A. (2002). *Intentional maps in posterior parietal cortex.* Annu. Rev. Neurosci. | F |
| **N15** | Crick, F.C. & Koch, C. (2005). *What is the function of the claustrum?* Phil. Trans. R. Soc. B. | F |

> Note: unlike `PredictiveCoding.Brain/ROADMAP.md` (bound to the Thousand-Brains
> papers P1–P9), this roadmap is grounded in general systems-neuroscience and
> predictive-coding references. Keep the two roadmaps separate.
