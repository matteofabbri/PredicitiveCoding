# 10. Live people-counting — roadmap

> **Goal.** Take a **live camera feed** (USB webcam or IP/RTSP camera) and answer
> *"how many people are in view?"* in real time — both the **instantaneous count**
> (how many are visible *right now*) and the **cumulative unique count** (how many
> *distinct* people have appeared / crossed a line over time) — entirely in managed
> .NET, no Python, with an optional CUDA path.

The good news: roughly **half of the pipeline already exists** in the solution.
Camera capture, video decoding, the ONNX-Runtime dependency, a complete worked
ONNX detector (RetinaFace), NMS/IoU maths, face embeddings and a face gallery are
all in the repo and reusable. What is missing is a **person (body) detector that
returns *all* boxes**, a **multi-object tracker**, the **counting logic**, and a
**streaming runtime loop**.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Available** — already implemented in the repo; reuse directly. |
| 🟡 | **Partial** — a reusable building block exists; needs adaptation/extension. |
| ❌ | **To build** — not present yet. |

*Status reflects the repository as of 2026-06-06.*

---

## What already exists (and where)

| Capability | Status | Where |
|------------|:------:|-------|
| Webcam frame capture (OpenCvSharp) | 🟡 | `PredictiveCoding.Data/Video/WebcamCapture.cs` — `CaptureFrames(frameCount, deviceIndex, options)`; grabs N frames into a `VideoClip` then returns (batch, not a live loop). |
| Video-file decode + frame → tensor | ✅ | `PredictiveCoding.Data/Video/OpenCvVideoLoader.cs` (`ProcessFrame`, `MatToBytes`, `AssembleTensor`), `VideoFiles.cs`, `VideoClip`. |
| ONNX Runtime (GPU build) | ✅ | `Microsoft.ML.OnnxRuntime.Gpu` 1.26.0, referenced by `PredictiveCoding.FaceRecognition`. |
| Worked ONNX detector (download → NCHW → run → decode → NMS) | ✅ | `PredictiveCoding.FaceRecognition/FaceAlignment/RetinaFaceDetector.cs` — the exact template a person detector follows. |
| NMS + IoU + Sigmoid + model auto-download | ✅ | private helpers in `RetinaFaceDetector.cs` (lift into a shared utility). |
| Detection result type | 🟡 | `FaceAlignment/FaceDetection.cs` (box + score + landmarks) — mirror it for people. |
| CUDA provisioning / EP selection | ✅ | `PredictiveCoding.FaceRecognition/CudaProvisioner.cs`. |
| Identity embedding + gallery (for *unique* counting / re-ID) | 🟡 | `ArcFaceOnnxEmbedder.cs`, `Recognition/FaceDatabase.cs`, `Recognition/FaceRecord.cs`. |
| Demo wiring pattern (menu entry in Main) | ✅ | Face demo is **option 6** in `PredictiveCoding.Main/Program.cs`; `Demos/FaceRecognitionDemo.cs`. |
| **Person / full-body detector returning all boxes** | ❌ | — |
| **Multi-object tracker (Kalman + association)** | ❌ | — |
| **Counting logic (per-frame / line-crossing / unique)** | ❌ | — |
| **Continuous live streaming loop** | ❌ | — |

---

## Phase 0 — Capture & infrastructure · 🟡 mostly there

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Open a USB webcam and read frames | ✅ | `WebcamCapture` already wraps OpenCV `VideoCapture(deviceIndex)`. |
| Open an **IP / RTSP** camera | 🟡 | OpenCV `VideoCapture` accepts an `rtsp://…` **string**; the wrapper only takes an `int` index — add a `string source` overload. |
| Frame → NCHW float tensor preprocessing | ✅ | Pattern in `RetinaFaceDetector.Detect` (resize → normalise → `DenseTensor`) and `OpenCvVideoLoader`. |
| **Continuous live frame producer** (grab forever, drop stale frames) | ❌ | New: a loop yielding `Mat`/frame with a bounded buffer; today `CaptureFrames` returns after N frames. |
| GPU execution provider + CPU fallback | ✅ | `Microsoft.ML.OnnxRuntime.Gpu` + `CudaProvisioner`; falls back to CPU automatically. |

## Phase 1 — Person detection (per frame) · ❌ core new work

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| `PersonDetector` — ONNX body detector returning **every** person box | ❌ | Follow `RetinaFaceDetector` end-to-end; load a COCO detector, keep class `person` (id 0). |
| Decode model output to candidate boxes | 🟡 | Reuse the decode/anchor logic shape from `RetinaFaceDetector.ParseAndDecode`. |
| **Return a list, not the single best** | 🟡 | `RetinaFaceDetector.Detect` returns one top face — generalise to return all detections above threshold. |
| NMS + IoU | ✅ | Already in `RetinaFaceDetector` (`Iou`, greedy NMS) — extract to `Detection/Nms.cs`. |
| Confidence threshold + min-box-size + ROI filter | ❌ | Trivial post-filters. |
| `Detection` result type (box, score, class) | 🟡 | Mirror `FaceDetection`. |
| **Face detector as a "head count" cross-check** | 🟡 | `RetinaFaceDetector` already detects faces — useful in crowds where bodies overlap but heads are visible. |

## Phase 2 — Per-frame counting · ❌ first demonstrable milestone

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Count = number of detections after NMS/threshold | ❌ | Trivial once Phase 1 lands → **"N people visible now"**. |
| Region-of-interest mask (count only inside a zone) | ❌ | Polygon/rect test against box centre. |
| Temporal smoothing (median over a short window) to stop flicker | ❌ | Reduces frame-to-frame jitter in the displayed count. |

## Phase 3 — Multi-object tracking · ❌ enables stable, non-double-counted IDs

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| `Track` type (id, box, velocity, age, hits, time-since-update) | ❌ | New. |
| Motion model — constant-velocity **Kalman filter** | ❌ | Pure maths; unit-testable (SORT-style). |
| Data association — IoU cost + Hungarian (or greedy) | 🟡 | IoU exists; add assignment. |
| Track lifecycle: tentative → confirmed → lost | ❌ | Avoids counting one-frame false positives. |
| Reference: **SORT** (no extra model) / **ByteTrack** (uses low-score boxes) | — | SORT is detector-only; ByteTrack improves recall in crowds. |

## Phase 4 — Counting semantics · ❌

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| **Current occupancy** = number of confirmed tracks | ❌ | The live "how many are here". |
| **Cumulative unique count** = distinct track IDs ever confirmed | ❌ | The "how many people came through". |
| **Line-crossing / tripwire** (in / out across a virtual line) | ❌ | Sign of (track centre − line) flips between frames. |
| Zone dwell / per-region occupancy | ❌ | Builds on Phase 2 ROI + Phase 3 tracks. |

## Phase 5 — Re-identification (optional, robust unique count) · 🟡

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Merge tracks that are the **same person** returning after occlusion / leaving | ❌ | Needed for an accurate cumulative count. |
| Face-based identity as a proxy (when faces are visible) | 🟡 | Reuse `ArcFaceOnnxEmbedder` + `FaceDatabase` (cosine gallery) directly. |
| Body **Re-ID** appearance embedding (e.g. OSNet ONNX) | ❌ | For when faces are not visible; same ONNX pattern as the face embedder. |
| "Seen people" gallery + dedup | 🟡 | `FaceDatabase` is already a cosine-similarity register/identify store. |

## Phase 6 — Live runtime & performance · ❌

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Streaming pipeline: capture → detect → track → count | ❌ | New orchestrator. |
| Decouple capture thread from inference thread | ❌ | Bounded `Channel<Frame>`; drop frames under load so latency stays low. |
| Detect every N-th frame, track in between | ❌ | Detection is the bottleneck; the tracker bridges skipped frames. |
| FPS / latency / count metrics | 🟡 | Reuse logging conventions (`PredictiveCoding.Logging`). |
| CUDA acceleration | ✅ | Available via `OnnxRuntime.Gpu` + `CudaProvisioner`. |

## Phase 7 — Output, demo & tests · 🟡 patterns exist

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Console live count | ❌ | Simplest sink. |
| Overlay boxes + IDs + count on the frame | 🟡 | OpenCvSharp can draw (`Cv2.Rectangle`/`PutText`) and show/save. |
| WinUI live demo | 🟡 | Reuse `SAMPLES/PredictiveCoding.CortexApp` / `OmniMind` WinUI patterns. |
| Menu entry in `PredictiveCoding.Main` | 🟡 | Mirror the Face demo (**option 6**) wiring. |
| Unit tests: NMS, IoU, Kalman, association, line-crossing | 🟡 | All pure functions; reuse the xUnit conventions in `TEST/PredicitiveCoding.UnitTests`. |

---

## Model & licensing note

Detection model choice matters here, and this repo is deliberately careful about
licences (ImageSharp is pinned to 3.1.x to avoid the 4.x licence gate; Sci-Hub was
rejected on legal grounds). Apply the same care to the detector weights:

| Model | Licence | Note |
|-------|---------|------|
| YOLOv8 / YOLO11 / YOLOv5 (Ultralytics) | **AGPL-3.0** | Strong copyleft — avoid for anything not open-sourced under AGPL. |
| **YOLOX** | **Apache-2.0** | Recommended permissive default; good speed/accuracy, easy ONNX export. |
| **RT-DETR** (original Baidu weights) | **Apache-2.0** | Transformer detector, NMS-free; permissive (the Ultralytics port is AGPL — use the original). |
| torchvision SSD / FCOS / RetinaNet | **BSD-3** | Permissive fallback if a known-good ONNX export is preferred. |

A **nano**-class model (e.g. YOLOX-Nano/Tiny) runs real-time on CPU and is trivial
on GPU. Download-on-first-use exactly like `RetinaFaceDetector` does (HuggingFace,
checksum/size guard, atomic `.part` → rename).

---

## How this maps onto the existing stack

- **`PredictiveCoding.Data`** — live capture and frame decoding (`WebcamCapture`,
  `OpenCvVideoLoader`, `VideoClip`); extend for RTSP + a streaming loop.
- **`PredictiveCoding.FaceRecognition`** — the reusable **vision/ONNX toolbox**:
  the detector template (`RetinaFaceDetector`), NMS/IoU, CUDA provisioning, and the
  ArcFace embedder + `FaceDatabase` for identity-based unique counting.
- **`PredictiveCoding.Core`** — `DenseTensor<float>` and SIMD ops for any custom
  pre/post-processing.

### Suggested layout for the new code

A new project, e.g. **`PredictiveCoding.Vision`** (or `PredictiveCoding.People`),
referencing `Data` + `FaceRecognition` + `Core`:

```
PredictiveCoding.Vision
├── Capture/      live frame source (RTSP + streaming loop; extends WebcamCapture)
├── Detection/    PersonDetector (ONNX), Detection, Nms (extracted from RetinaFace)
├── Tracking/     Track, KalmanFilter, SortTracker / ByteTracker
├── Counting/     PerFrameCounter, LineCounter, ZoneCounter, UniqueCounter
├── ReId/         (optional) PersonReId + identity gallery (reuse ArcFace/FaceDatabase)
├── Runtime/      PeopleCountingPipeline (capture → detect → track → count)
└── Demo/         console + WinUI demo, Main menu entry
```

> **Build order & first milestone.** Phase 0 (extend capture) → Phase 1 (person
> detector returning all boxes) → Phase 2 (per-frame count) gives the first
> demonstrable result — **"N people visible now"** on a live feed. Then Phase 3
> (tracking) → Phase 4 (unique / line counting) answers **"how many distinct
> people"**. Phases 5–7 (Re-ID, streaming performance, demo/tests) are polish.
