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

> **✅ Phases 0–5 are implemented** in the new **`PredictiveCoding.Vision`**
> project, wired as **demo option 8** in `PredictiveCoding.Main`
> (`Demos/PeopleCountingDemo.cs`). A live feed / video / image is decoded by
> `Capture/FrameStream`; each frame is run through a `YoloPersonDetector`
> (OpenCV-Zoo YOLOX, COCO, Apache-2.0, auto-downloaded) or the bundled
> `FaceCountDetector` fallback; a SORT-style `Tracking/SortTracker` turns the boxes
> into stable IDs and reports both **current** ("people now") and cumulative
> **unique** ("distinct people seen") counts. `Counting/LineCounter` and
> `Counting/ZoneCounter` add directional tripwire crossings and per-zone occupancy /
> dwell (Phase 4); `ReId/ReidPeopleCounter` + `PersonGallery` dedup the same person
> returning after an occlusion for a robust unique count (Phase 5). Verified on still
> images (`bus.jpg` → 4, `zidane.jpg` → 2), on the `vtest.avi` pedestrian clip
> (≈7 concurrent / 21 unique over 100 frames). `Runtime/PeopleCountingPipeline` runs
> the whole capture → detect → track → count loop with capture decoupled from
> inference (drop-under-load) and detection every N-th frame (Phase 6). Covered by
> **25 unit tests** (tracker, line/zone, re-ID, pipeline). **Next up: Phase 7** —
> overlay/console output and an optional WinUI demo.

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
| **Person / full-body detector returning all boxes** | ✅ | `Vision/Detection/YoloPersonDetector.cs` (YOLOX, COCO). |
| **Multi-object tracker (Kalman + association)** | ✅ | `Vision/Tracking/SortTracker.cs` + `Track.cs` (alpha-beta + greedy IoU). |
| **Counting logic (per-frame / line-crossing / unique)** | ✅ | `SortTracker` (current/unique) + `Counting/LineCounter` (tripwire) + `Counting/ZoneCounter` (zones). |
| **Re-identification (robust unique count)** | ✅ | `ReId/{ReidPeopleCounter, PersonGallery, ColorHistogramEmbedder}`; face/body ONNX embedders plug into `IAppearanceEmbedder`. |
| **Streaming runtime (decoupled capture↔inference, detect-every-N, FPS)** | ✅ | `Runtime/PeopleCountingPipeline` + `Runtime/LatestFrameBuffer`. |
| **Continuous live streaming loop** | ✅ | `Vision/Capture/FrameStream.cs`. |

---

## Phase 0 — Capture & infrastructure · ✅ done

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Open a USB webcam and read frames | ✅ | `Vision/Capture/FrameStream.cs` opens `VideoCapture(index)`; `WebcamCapture` stays for batch grabs. |
| Open an **IP / RTSP** camera | ✅ | `FrameStream` passes a non-numeric `source` to `VideoCapture(string)`, so `rtsp://…` / `http://…` URLs work. |
| Frame → NCHW float tensor preprocessing | ✅ | `YoloPersonDetector` (letterbox + BGR) and `RetinaFaceDetector` (normalised RGB). |
| **Continuous live frame producer** (grab forever, drop stale frames) | ✅ | `FrameStream.Read` streams `Image<Rgb24>` until cancel/EOF, with a `frameStride` skip. |
| GPU execution provider + CPU fallback | ✅ | `OnnxRuntime.Gpu`; `YoloPersonDetector` probes `CudaProvisioner` and falls back to CPU. |

## Phase 1 — Person detection (per frame) · ✅ done

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| `PersonDetector` — ONNX body detector returning **every** person box | ✅ | `Vision/Detection/YoloPersonDetector.cs` (OpenCV-Zoo YOLOX, COCO, Apache-2.0, auto-download). |
| Decode model output to candidate boxes | ✅ | YOLOX grid/stride/exp decode in `YoloPersonDetector.Decode`. |
| **Return a list, not the single best** | ✅ | `RetinaFaceDetector.DetectAll` added; detectors return `IReadOnlyList<Detection>`. |
| NMS + IoU | ✅ | Extracted to `Vision/Detection/Nms.cs` (`Iou`, `Apply`). |
| Confidence threshold + min-box-size + ROI filter | 🟡 | Score + NMS thresholds done; ROI filter still to add. |
| `Detection` result type (box, score, class) | ✅ | `Vision/Detection/Detection.cs`. |
| **Face detector as a "head count" cross-check** | ✅ | `Vision/Detection/FaceCountDetector.cs` wraps `RetinaFaceDetector.DetectAll`. |

## Phase 2 — Per-frame counting · ✅ done (first milestone)

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Count = number of detections after NMS/threshold | ✅ | `PeopleCountingDemo` prints the live count → **"N people visible now"**. |
| Region-of-interest mask (count only inside a zone) | ❌ | Polygon/rect test against box centre. |
| Temporal smoothing (median over a short window) to stop flicker | ✅ | `Vision/Counting/CountSmoother.cs` (sliding-window median). |

## Phase 3 — Multi-object tracking · ✅ done

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| `Track` type (id, box, velocity, age, hits, time-since-update) | ✅ | `Vision/Tracking/Track.cs`. |
| Motion model — constant-velocity Kalman | ✅ | `Track` uses a fixed-gain **alpha-beta** filter (lightweight Kalman); full-covariance Kalman is an optional upgrade. |
| Data association — IoU cost + Hungarian (or greedy) | ✅ | Greedy, highest-IoU-first, in `SortTracker.Update` (reuses `Nms.Iou`). |
| Track lifecycle: tentative → confirmed → lost | ✅ | `minHits` to confirm, `maxAge` to retire; suppresses one-frame false positives. |
| Reference: **SORT** / **ByteTrack** | — | Built SORT-style; ByteTrack (low-score 2nd pass) is a future upgrade for crowds. |

## Phase 4 — Counting semantics · ✅ done

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| **Current occupancy** = number of confirmed tracks | ✅ | `SortTracker.CurrentCount` (confirmed tracks matched this frame). |
| **Cumulative unique count** = distinct track IDs ever confirmed | ✅ | `SortTracker.UniqueCount`. |
| **Line-crossing / tripwire** (in / out across a virtual line) | ✅ | `Vision/Counting/LineCounter.cs` — directional In/Out from segment-intersection of the track path with the tripwire; off-segment passes ignored. |
| Zone dwell / per-region occupancy | ✅ | `Vision/Counting/ZoneCounter.cs` — point-in-polygon on the track's feet; `Occupancy`, distinct `Entered`, per-track `DwellFrames`. |

## Phase 5 — Re-identification (robust unique count) · ✅ done (appearance baseline)

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Merge tracks that are the **same person** returning after occlusion / leaving | ✅ | `Vision/ReId/ReidPeopleCounter.cs` — embeds each newly-confirmed track and re-identifies it against the gallery, so a re-entry reuses its person id instead of inflating the count. |
| "Seen people" gallery + dedup | ✅ | `Vision/ReId/PersonGallery.cs` — cosine register/identify (mirrors `FaceDatabase`); `UniquePeople` = distinct people. |
| Appearance embedding (dependency-free baseline) | ✅ | `Vision/ReId/ColorHistogramEmbedder.cs` behind the `IAppearanceEmbedder` interface. |
| Face-based identity as a proxy (when faces are visible) | 🟡 | Plug `ArcFaceOnnxEmbedder` into `IAppearanceEmbedder` (needs the model + face crops). |
| Body **Re-ID** appearance embedding (e.g. OSNet ONNX) | 🟡 | Interface ready; OSNet ONNX weights are a drop-in upgrade over the colour baseline. |

## Phase 6 — Live runtime & performance · ✅ done

| Task | Status | Reuse / notes |
|------|:------:|---------------|
| Streaming pipeline: capture → detect → track → count | ✅ | `Vision/Runtime/PeopleCountingPipeline.cs` — `Process` (one frame) and `RunBlocking` (over a frame source). |
| Decouple capture thread from inference thread | ✅ | `Vision/Runtime/LatestFrameBuffer.cs` — single-slot latest-wins hand-off; `RunBlocking` reads on a background thread while the inference loop always takes the freshest frame, dropping (and disposing) the rest so latency stays bounded. |
| Detect every N-th frame, track in between | ✅ | `detectEveryN`: the detector runs on 1/N frames; the tracker advances every frame, carrying boxes between detections. |
| FPS / latency / count metrics | ✅ | `PeopleCountingPipeline` exposes `Fps`, `ProcessedFrames`, `DetectionFrames`, `CurrentCount`, `UniqueCount`. |
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
