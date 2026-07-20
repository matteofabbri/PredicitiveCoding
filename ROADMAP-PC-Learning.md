# ROADMAP — Allineare l'apprendimento alla Predictive Coding

**Tesi del progetto:** in una libreria che si chiama *Predictive Coding*, l'apprendimento deve
avvenire **in PC** (rilassamento a variabili latenti + regole locali di aggiornamento), **non**
con backprop né con metodi classici a gradiente. Questo documento elenca i componenti che oggi
imparano in un modo diverso e come portarli alla PC — dai guadagni facili alla frontiera.

**Cos'è un "bersaglio":** un componente lo è solo se (1) **impara in casa** e (2) con metodo
**non-PC** (backprop o classico) e (3) la PC è applicabile. NON sono bersagli: il motore PC e le
reti, l'HTM/Hebbian del Brain (già locale), l'inference di modelli ONNX esterni, le feature a mano,
gli store non-parametrici, i tokenizer, i data-loader (vedi §Già allineati e §Lasciare).

**Ordine di attacco (deciso da Matteo, 2026-07-20):** si parte dall'**LLM (Bersaglio 5)** per
**road-testare su strada la teoria centrale** — modello in **PC pura** rimesso in riga *di tanto in
tanto* dal metodo classico (l'ancora Adam) — poi **5 → 3 → 2 → 1**, e da lì si decide. Il
**Bersaglio 4 esce**: `NGramLanguageClassifier` vive in `PredictiveCoding.Data`, è una **sorgente di
dati** che consumiamo, non un modello su cui operiamo. Nota: dentro il track LLM vale comunque il
principio *small-first* (prima il **cancello** su un singolo FFN, poi si sale).

---

## Bersaglio 1 — `LinearRegressor` (PredictiveCoding.Vision) · difficoltà BASSA
- **Oggi:** regressore lineare multi-output `W·x+b` che distilla un teacher embedder (HOG→embedding),
  addestrato in casa con SGD su MSE.
- **Perché tradisce la tesi:** apprendimento a gradiente su una loss globale, non locale.
- **PC applicabile? Sì, già dimostrato nel repo:** `PcDistilledFaceEmbedder` fa *esattamente* questa
  distillazione in PC pura (layer0 clampato al descrittore, top clampato all'embedding teacher,
  update Hebbiani locali). Conversione quasi a costo zero: rete PC a 1 hop con top-layer clampato al target.
- **Valore/fattibilità:** il migliore. Gemello PC esistente, singolo layer.

## Bersaglio 2 — `LinearModel` / SVM Pegasos (PredictiveCoding.Vision) · difficoltà BASSA
- **Oggi:** testa lineare del detector HOG, SVM addestrata con sub-gradiente Pegasos su hinge loss.
- **Perché tradisce la tesi:** classificatore lineare a sub-gradiente, non PC.
- **PC applicabile? Sì:** la PC classificatoria (energy-based, `Network.Predict = argmin_c free energy`)
  è il rimpiazzo diretto — clampi l'output alla classe e rilassi. Singolo layer, nessun autodiff da smontare.
- **Valore/fattibilità:** alto. Il detector `HogPersonDetector` che lo consuma non va toccato
  (riceve i pesi dall'esterno).

## Bersaglio 3 — `PcaProjector` (+ `EnhancedLbpEmbedder`, `FusedEmbedder`) (PredictiveCoding.FaceRecognition) · difficoltà BASSA-MEDIA
- **Oggi:** PCA troncata con whitening via power iteration; riduzione dimensionale non supervisionata.
  `EnhancedLbpEmbedder` e `FusedEmbedder` sono bersagli solo perché il loro unico apprendimento in casa
  **è** il fit di `PcaProjector` → convertito il core, si convertono anche loro (un solo intervento).
- **Perché tradisce la tesi:** apprendimento classico (autovettori/whitening), non generativo-locale.
- **PC applicabile? Sì, gemello già nel repo:** `PcFaceEmbedder` è un autoencoder PC (`GenerativeNetwork`,
  rilassamento + update locali) che occupa la stessa nicchia "descrittore→latente compresso".
  Attenzione a preservare la semantica di whitening (varianza unitaria) nel prior del top layer.
- **Valore/fattibilità:** alto (un solo core, tre componenti sistemati).

## ~~Bersaglio 4~~ — `NGramLanguageClassifier` → **DECLASSATO (lasciare)**
Decisione di Matteo (2026-07-20): vive in `PredictiveCoding.Data`, è una **sorgente di dati** che
consumiamo (rilevamento lingua per corpus/crawler), **non** un modello su cui operiamo. Resta com'è.
(Il confronto PC "apples-to-apples" invitato da `LanguageSamples.cs` resta un'idea da demo, non un
bersaglio di allineamento.)

## Bersaglio 5 — Famiglia backprop LLM/Device · difficoltà FRONTIERA
Un unico bersaglio, perché tutto converge su `DecoderLM`:
- **`DecoderLM` (+ `Block`, `TemporalBlock`)** — il transformer decoder reale (autograd `Ag` →
  `Backward()` → Adam). È **il** target centrale.
- **`TinyTransformerLM`** — gemello didattico/gradcheck, stesso meccanismo backprop.
- **`DeviceDecoderTrainer`** — la realizzazione GPU dello stesso backprop (`DeviceTrainKernels` è
  solo backprop, nessun kernel Hebbiano/PC on-device).
- **`ContinualLearner`** e **`LmDistiller`** — regimi distinti (replay + LR proporzionale alla sorpresa;
  distillazione sequence-level), ma girano su `Trainer.Step`/Adam.
- (`PredictiveLlm` e il driver `Translate.Cli` NON sono target separati: convertirli = convertire `DecoderLM`.)
- **Perché tradisce la tesi — nel modo più clamoroso:** è il fiore all'occhiello di un repo chiamato
  *PredictiveCoding*, ma impara al 100% con Adam su gradienti globali. `DecoderLM.SettlePlan()` è chiamato
  "predictive-coding reading" ma è `loss.Backward` su un vettore-piano, NON il motore `Layer`/`Network`.
- **`ContinualLearner` è il punto d'ingresso concettualmente più vicino:** la sua pesatura "per sorpresa"
  già rispecchia il precision-weighting della PC.

### Il disegno per l'LLM (dal design dedicato)
- **Idea guida:** tagliare il grafo Autograd ai **confini fra blocchi**. Il **residual stream** (lo stato
  che già passa da blocco a blocco) diventa una catena di **latenti persistenti** `μ_b` ("lavagnette") con
  errore `ε_b` e precisione `Π_b`. Ogni blocco vede solo `μ_{b-1}`, la sua `μ_b`, e l'errore dal blocco
  sopra — apprendimento **locale ai confini**. Dentro il blocco (attention/RMSNorm/RoPE/MoE) si tiene
  l'**autodiff corto**, un blocco alla volta. È Millidge 2020 ("PC su grafi arbitrari") col grafo tagliato
  ai bordi dei blocchi, in direzione discriminativa (Whittington & Bogacz 2017), non top-down.
- **Regole:** stato `μ_b ← μ_b − η_s·(ε_b − VJP_{b+1}(ε_{b+1}))` per `T` step di rilassamento; pesi
  `Δθ_b ∝ VJP_b(ε_b)`. I VJP riusano l'Autograd esistente ma su **un solo blocco**.
- **Unica estensione al motore:** `BackwardFrom(output, seed)` in `Autograd.cs` (semina un grad arbitrario
  su un output non-scalare; riusa Parents/BackwardFn/ordine topologico che ci sono già).
- **Nuovi componenti:** `PcDecoderTrainer` (gemello di `Trainer.cs`, non tocca il backprop) + `PcLatentState`
  (buffer `μ_b`/`ε_b`/`Π_b`).
- **Ancora Adam (idea di Matteo):** PC primaria + burst Adam periodici come "ri-centraggio" sulla loss vera.
  Regole di stabilità: Adam **re-inizializzato** (m/v azzerati) a ogni burst; PC tenuto ≈backprop
  (precisione identità) per coerenza d'obiettivo; mini ri-settling dopo ogni ancora. Metriche: `‖Δθ_ancora‖ /
  ‖Δθ_PC‖` e F-vs-CE per capire se l'ancora serve o è ridondante.
- **Primo passo = CANCELLO:** un singolo FFN denso, PC-allenato, con l'update di peso **gradient-checkato
  contro backprop** (a convergenza, precisione identità → devono coincidere). Poi: attn → 2 blocchi →
  accendi l'ancora → MoE/loop per ultimi.
- **Rischi onesti:** PC su transformer a scala competitiva mai dimostrata; l'ibrido "PC + Adam periodico"
  non pubblicato (esperimento, non ricetta); MoE-come-latente-discreto fuori scope (routing congelato);
  rapporto LR ancora↔PC (`η_s·T·Π`) è il punto dove ci si impantana.
- **Valore/fattibilità:** valore massimo, fattibilità minima. Da affrontare *dopo* i quattro guadagni facili.

---

## Frontiera aperta — il "centralino" MoE sotto Predictive Coding

Il preset PC attuale è **denso** (niente MoE) apposta: il MoE ha una **scelta discreta**
(quale esperto per ogni token) che il PC non sa imparare — il rilassamento muove quantità
*continue*, non "scegline uno tra N". Il centralino fa due gesti: **votare** (liscio,
allenabile) e **scegliere** top-k (a gradino, non differenziabile — il nodo). Due strade,
dalla fattibile alla ricerca:

**MoE-PC.1 — Routing congelato (fattibile, MVP del MoE-PC).** Portare nel PC il trucco che
il backprop usa già: la scelta top-k è una maschera 0/1 **costante** nel passo corrente, e
il VJP (via `BackwardFrom`) scorre solo attraverso gli esperti scelti e i gate sopravvissuti.
Così il PC allena gli **esperti** e i **voti**; la *decisione* resta guidata dai voti lisci,
non "inferita" dal PC. Difficoltà: media. Sblocca il MoE nel `PcDecoderTrainer` (oggi lo
rifiuta) a costo contenuto.

**MoE-PC.2 — Routing come latente (PC "vero", ricerca aperta).** Far **decidere al
rilassamento** lo smistamento: l'assegnazione token→esperto come variabile latente che il
settling porta verso una scelta. Serve rilassare una scelta discreta in modo liscio — es.
assegnazione *soft* (stile Gumbel-softmax/concrete) spinta verso l'one-hot dal rilassamento,
oppure un latente di gate `μ_gate` inferito come i `μ` dei blocchi. Nessun risultato noto:
è la parte davvero di frontiera.

**Interfaccia (bozza).** Un `IExpertRouter` separa i due gesti: `Gate(hidden)` (i voti,
LISCIO → PC-friendly) e `Route(hidden)` → `RoutingPlan` (la scelta top-k, A GRADINO → il
nodo). Un router PC sostituirebbe il `Route` a gradino con un'assegnazione rilassata che il
settling porta a decisione.

```csharp
public interface IExpertRouter
{
    int ExpertCount { get; }
    int TopK { get; }
    Tensor Gate(Tensor hidden);          // voti [seq, ExpertCount] — liscio, PC-friendly
    RoutingPlan Route(Tensor hidden);    // scelta top-k — a gradino, il nodo del PC
}
public readonly record struct RoutingPlan(int[][] Experts, float[][] Weights);
```

Ordine: **MoE-PC.1 prima** (sblocca il MoE nel PC a costo medio), MoE-PC.2 come ricerca a parte.

## Già allineati (locale/bio, non-backprop) — NON toccare
- **Motore PC e reti:** `Layer`, `Network` e ogni variante (`Temporal`, `Generative`, `Hierarchical`,
  `DeepRecurrent`, `Residual`, `Attentive`, `Spiking`, tutte le MoE, `Ensemble`, `Routed` — anche il
  router impara in PC —, `MultiModal`, `Graph`, `Gated`, `Memory`, `Hopfield`). Rilassamento + update
  Hebbiani locali, zero backprop.
- **Aree Cortex** costruite su quel motore (VisualCortex e sensori, Motor/Premotor/Prefrontal,
  Hippocampus, Broca/Wernicke-toy…) e i due embedder facciali PC (`PcFaceEmbedder`, `PcDistilledFaceEmbedder`).
- **`BatchedPcForward` (Device):** unico componente PC genuino della tripletta LLM/Onnx/Device — ma è
  inference/settling, non impara pesi. Già PC.
- **HTM/Hebbian-locale:** `SpatialPooler`, `TemporalMemory`, `CorticalColumn`, `LayeredColumn`. Imparano
  con adattamento locale di permanenze sinaptiche (non backprop). Bio-plausibili: lasciare anche se non
  sono PC-classico.
- **`DopaminergicSystem`:** tracker scalare 1-D di reward (EMA/TD). Segnale neuromodulatorio bio-locale;
  la PC non è applicabile → lasciare.

## Legittimamente non-PC — lasciare
- **Inference ONNX esterna (strumenti/teacher):** `ArcFaceOnnxEmbedder` (teacher della distillazione),
  `RetinaFaceDetector`, `YoloPersonDetector`, `FaceCountDetector`, `OnnxModel`+`Executor`, profili
  pre-addestrati di `LanguageDetector`. Addestrati altrove, servono da input/teacher.
- **Feature a mano:** `FaceDescriptor`/`HogDescriptor`, `ColorHistogramEmbedder`, `DescriptorEmbedder`,
  `RetinaLgn` (DoG), `GridCellEncoder`, `ScalarEncoder`/`PoseEncoder`, `FacePreprocessor`, `FaceAligner`.
  Trasformazioni deterministiche, zero parametri appresi.
- **Store non-parametrici / kNN:** `MemoryBank`, `HopfieldMemory`, `EpisodicMemory`, `FaceDatabase`,
  `PersonGallery`, `ObjectMemory`, `DisplacementRecognizer(3D)`. Memorizzano esemplari, non adattano pesi.
- **Tokenizer a frequenza:** `BpeTokenizer`, `WordPieceTokenizer`. Tabelle di merge/vocabolario per
  conteggio, non modelli differenziabili.
- **Algoritmi classici senza pesi appresi:** `ExtractiveSummarizer` (TF-IDF/PageRank), `SortTracker`, `Track`.
- **Twin di sola inferenza del backprop LLM:** `GptModel`, `Translator`, `DevicePretrainer`/`Trainer`/`Adam`/
  `Autograd`/`DeviceTrainKernels`. Si sistemano convertendo `DecoderLM`, non separatamente.
- **Infrastruttura:** data-loader, codec, `PredictiveCoding.Imaging`, harness di training, suite di test.

## Sintesi
Bersagli veri: **quattro** (l'n-gram declassato a datasource) — tre piccoli learner classici
(`LinearRegressor`, `LinearModel`/SVM, `PcaProjector`+wrapper) e **una** famiglia backprop (l'LLM
`DecoderLM` e satelliti). **Scelta di Matteo (2026-07-20): si parte dall'LLM** per road-testare *su
strada* la teoria PC-pura + ancora classica, poi **3 → 2 → 1**. Dentro l'LLM il primo passo è il
**cancello**: gradient-check dell'update PC su un singolo FFN contro il backprop.
