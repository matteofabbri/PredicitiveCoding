# 16. Tools & CLIs

Small console projects that exist to serve the libraries — distilling,
converting, measuring, translating. Three live under `tools/` (in the
solution's `/tools/` folder), one CLI under the samples.

## `tools/DetectorDistiller` — teachers → in-house students

Trains the **download-free** detectors by distilling the heavy ONNX "teacher"
models the repo otherwise depends on at inference:

```
YOLOX       → labels person boxes  → trains an in-house HOG person detector
RetinaFace  → labels face boxes    → trains an in-house HOG face detector
ArcFace     → labels face embeddings → trains an in-house face embedder
                                       (regress face-HOG → ArcFace 512-D)
```

The teachers supply labels **once, offline**; afterwards the produced text
models drive `HogPersonDetector` / `LinearRegressor` with **no ONNX runtime
and no model download**.

```bash
dotnet run --project tools/DetectorDistiller -- \
    --source <video|imageFolder> [--frames N] [--neg K] [--models <dir>] [--out <dir>]
# → hog-person.txt · hog-face.txt · face-embedder.txt  (default out:
#   ~/.predictivecoding/vision-inhouse; teachers auto-download to
#   ~/.predictivecoding/vision-models)
```

## `tools/OnnxPack` — .onnx → PCNX

Converts a `.onnx` model into the in-house **PCNX** format
([doc 13](13-onnx.md)). With optimisation (the default) the graph passes
through `GraphOptimizer` — constant folding, Identity elision, dead-code
removal — with `Run` results staying bit-identical. The tool reloads what it
wrote to prove the file parses, and prints honest before/after stats.

```bash
dotnet run --project tools/OnnxPack -- model.onnx [output.pcnx] [--no-optimize]
```

## `tools/MixedPrecisionBench` — float32 vs mixed, measured

Phase 4 of the core
[Mixed-Precision roadmap](../PredicitiveCoding/ROADMAP-MixedPrecision.md):
Stopwatch-based (warmup + median, no benchmark framework) tables for

1. `MatVecMul`/`MatVecMulT` — float32 vs `Half` weights with fp32 accumulate
   (storage halved), 512² and 2048²;
2. network-level PC inference — `InferGetTop` vs `InferGetTopMixed` on a
   CIFAR-scale net, plus `BatchedPcForward` (B settlings as GEMMs) on CPU and,
   when the CUDA runtime is discoverable, on the GPU (float and tensor-core
   mixed);
3. GEMM 1024³ — CPU float32 vs CPU mixed vs CUDA mixed.

## `tools/LangDetectSmoke` *(not in the solution)*

A ten-line harness comparing the two offline language detectors of
[`PredictiveCoding.Data`](12-data.md) — `langdetect` (53 profiles) vs the
native n-gram classifier — on fixed multilingual snippets.

## `PredictiveCoding.Translate.Cli` — our own translator

A sample console app that builds, verifies and runs a **seq2seq translator**
on the LLM stack (`Autograd/DecoderLM`, `BpeTokenizer`, `Translation/`):

```
translate gradcheck      numerically verify masked cross-entropy + DecoderLM gradients
translate smoke          overfit a handful of pairs and decode them back (end-to-end)
translate train …        train on a parallel corpus
translate train-tatoeba  train on auto-downloaded Tatoeba sentence pairs
translate translate …    load a trained model and translate stdin / an argument
```

The prompt format (`TranslationFormat.PromptText`, target-side masking with
`IgnoreIndex`) is the same the `Translation/` library uses; `gradcheck` is the
numerical proof that the masked loss contributes zero gradient on masked rows.

## Other executables

- **`PredictiveCoding.Main`** — the 13-option demo menu; see
  [doc 9](09-build-and-run.md).
- **`SAMPLES/PredictiveCoding.CortexApp`** — the WinUI 3 visualiser; see
  [doc 5](05-cortex.md).
