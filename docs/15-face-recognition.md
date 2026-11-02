# 15. Face recognition — `PredictiveCoding.FaceRecognition`

A complete face-recognition stack — detection, alignment, embedding, enrolment,
verification and identification — benchmarked on **LFW**, with **predictive
coding** doing the learning wherever learning is needed, and with the heavy
pretrained "teacher" models distillable away until none are needed at run time.

## The embedding contract

Everything reduces to one interface:

```csharp
public interface IFaceEmbedder
{
    int Dimension { get; }
    string Name { get; }
    void Fit(IReadOnlyList<string> trainingImagePaths);   // no-op for fixed descriptors
    float[] Embed(string imagePath);
    IReadOnlyList<float[]?> EmbedBatch(IReadOnlyList<string> imagePaths); // GPU batch
}
```

Similarity between embeddings is cosine distance (`FaceMath`). `Fit` lets
learnable embedders train on a sample first.

## The embedders

| Embedder | What it is |
|----------|------------|
| **`DescriptorEmbedder`** | The raw LBP `FaceDescriptor` used directly — the fixed, training-free baseline. |
| **`EnhancedLbpEmbedder`** | Improved LBP features (uniform patterns, rotation-invariant coding). |
| **`PcFaceEmbedder`** | A **predictive-coding network** trained as an autoencoder over face descriptors — the pure-PC learner. |
| **`ArcFaceOnnxEmbedder`** | The "how the best do it" teacher: pretrained **ArcFace** (InsightFace `w600k_r50`, 512-D), auto-downloaded on first use, with RetinaFace detection + 5-point alignment warped to the canonical 112×112 template (closed-form similarity transform — the production InsightFace pipeline). |
| **`PcDistilledFaceEmbedder`** | **Distillation**: supervised PC regression clamps layer 0 to the LBP descriptor and the top layer to the *teacher's* embedding; Hebbian learning moves the generative weights. At inference the embedding is the free-settled top state — **no backprop, no ONNX, no teacher**. The trained network persists (`LayerStore` + `distilled.json` manifest); a manifest probe recomputes a fixed reference embedding on load so a corrupt checkpoint fails loudly instead of serving garbage. |
| **`FusedEmbedder`** | Concatenates / averages several embedders for a stronger combined vector. |

### Two ONNX runtimes, one opt-in flag

`ArcFaceOnnxEmbedder` (and the RetinaFace detector) run on
**`Microsoft.ML.OnnxRuntime`** for speed by default. Passing
`useInHouseRuntime: true` switches the same models onto the dependency-free
[`PredictiveCoding.Onnx`](13-onnx.md) executor — CPU-only and ~15–25× slower,
meant for offline work, but it removes the native dependency entirely.

## Pipeline & recognition

```
image → FacePreprocessor → RetinaFaceDetector (ONNX) → FaceAligner
      → descriptor/embedder (table above) → FaceDatabase (enrol · identify · verify)
```

- **`Descriptors/`** — `FacePreprocessor` (normalise/resize), `HogDescriptor`
  (HOG over face crops), `PcaProjector` (dimensionality reduction, fitted on
  the enrolled gallery).
- **`FaceAlignment/`** — `RetinaFaceDetector` (pretrained ONNX, auto-download)
  and `FaceAligner` (landmark-based similarity warp).
- **`Recognition/`** — `FaceDatabase` / `FaceRecord`: enrol identities, then
  identify (best match above threshold) or verify (1:1 cosine).

## Benchmarks & datasets

- **`FaceVerificationBenchmark`** / **`FaceIdentificationBenchmark`** over a
  **`FaceGallery`** score every embedder uniformly (the demo prints a table).
- **`LfwDataset` + `FaceDatasetDownloader`** — Labeled Faces in the Wild is
  fetched on first use; pairs/people splits follow the standard protocol.

## GPU provisioning — `CudaProvisioner`

`Detect()` reports the machine's GPU/driver/CUDA state; `InstallAsync()`
installs the **CUDA runtime + cuDNN DLLs headlessly, without admin**, by
fetching only NVIDIA's official *redistributable* archives (the runtime
libraries — no multi-GB Toolkit, no installer, no UAC) into a folder you put on
`PATH`, so ONNX Runtime's CUDA execution provider can load. This is what lets
the GPU path "just work" on a clean machine.

## Try it

Demo **option 6** in `PredictiveCoding.Main` runs the LFW benchmark across the
embedders (LBP · PC · distilled · ArcFace). The
[`DetectorDistiller` tool](16-tools-and-cli.md) distils ArcFace into an
in-house face embedder offline; tests include `PcDistilledFaceEmbedderTests`
(checkpoint/probe semantics) and the face pipeline suites.
