# 9. Build & run

## Prerequisites
- **.NET 10 SDK** (the whole solution targets `net10.0`; `CortexApp` targets
  `net10.0-windows10.0.19041.0`).
- Windows for the WinUI demo app (`CortexApp`) and for the OpenCvSharp capture
  paths (the `OpenCvSharp4.runtime.win` package). The libraries themselves are
  cross-platform in intent.
- Optional: **CUDA 12 runtime** (`cudart`/`cublas` on `PATH`) for the CUDA
  backend and the GPU pretraining run. The ONNX Runtime GPU path (faces,
  people counting) can provision its own DLLs — see `CudaProvisioner` in
  [doc 15](15-face-recognition.md).

## Build & test

```bash
# Build the whole solution
dotnet build PrediciiveCoding.slnx

# Run the test suite (hundreds of tests)
dotnet test PredicitiveCoding.UnitTests/PredicitiveCoding.UnitTests.csproj
```

The solution builds under the default **AnyCPU** configuration. `CortexApp` is an
*unpackaged* WinUI app specifically so AnyCPU works (the packaged WinUI harness
`PredictiveCoding.Tests` is deliberately not in the solution).

## Run the console demos (`PredictiveCoding.Main`)

```bash
dotnet run --project PredictiveCoding.Main
```

The menu offers 13 options:

| # | Demo |
|---|------|
| 1 | Standard training run (synthetic blobs) |
| 2 | Millidge, Seth & Buckley (2022) demo |
| 3 | CIFAR-10 hard classification (auto-downloads the dataset) |
| 4 | Temporal PC demo (sequence learning) |
| 5 | Mixture of Experts demo (bimodal specialisation) |
| 6 | Face recognition (LFW benchmark · LBP · PC · ArcFace ONNX) |
| 7 | Language ID (PC network vs naive-Bayes · `PredictiveCoding.Data`) |
| 8 | People counting (live cam / video · YOLOX ONNX · per-frame count) |
| 9 | Thousand Brains (sensorimotor recognition · columns voting) |
| 10 | Full LLM demo (`PredictiveLlm` · train → generate → report) |
| 11 | LM distillation (live LM Studio teacher → sparse-MoE student) |
| 12 | Web corpus builder (crawler → sharded corpus; feeds the model if one exists) |
| 13 | Web pretraining run (bootstrap the Small preset on the GPU · resumable · bits/byte) |

Heavy assets (CIFAR-10, LFW, ONNX models) download on first use into
`~/.predictivecoding/`. Demos 12→13 are a pair: sediment a corpus first, then
pretrain on it (a CUDA GPU is required for 13; it refuses to run on CPU).

## Run the translator CLI (`PredictiveCoding.Translate.Cli`)

```bash
dotnet run --project PredictiveCoding.Translate.Cli -- gradcheck     # verify gradients
dotnet run --project PredictiveCoding.Translate.Cli -- smoke         # overfit 6 pairs
dotnet run --project PredictiveCoding.Translate.Cli -- train-tatoeba # Tatoeba corpus
dotnet run --project PredictiveCoding.Translate.Cli -- translate "hello"
```

Details in [doc 16](16-tools-and-cli.md).

## Run the tools (`tools/`)

```bash
dotnet run --project tools/OnnxPack -- model.onnx            # .onnx → optimised .pcnx
dotnet run --project tools/DetectorDistiller -- --source <video|folder>   # teachers → in-house
dotnet run --project tools/MixedPrecisionBench               # float32 vs mixed tables
```

## Run the WinUI app (`CortexApp`)

It is unpackaged, so it needs the **Windows App Runtime**.

- **Quickest** — launch it; the runtime prompt offers a one-time install.
- **Self-contained** — bundle the runtime so no install is needed (the csproj turns
  this on when a RuntimeIdentifier is given):
  ```powershell
  dotnet run --project PredictiveCoding.CortexApp -p:Platform=x64 -r win-x64
  ```
  The published `…/win-x64/publish/PredictiveCoding.CortexApp.exe` carries the
  runtime alongside it.

The app's `NavigationView` opens pages for the Visual Cortex, Hopfield memory, grid
cells, spiking network, and active-inference demos (see [doc 5](05-cortex.md)).

## Enable GPU acceleration

### CUDA (NVIDIA)
Install the CUDA Toolkit (so `cudart`/`cublas` are on `PATH` / `LD_LIBRARY_PATH`).
Then:
```csharp
PredictiveCoding.Device.DeviceRegistry.EnableCuda();   // LLM matmuls now run on cuBLAS
Console.WriteLine($"cuda devices: {DeviceRuntime.CudaDeviceCount}");
```
No native DLL to build — see [doc 7](07-device-gpu.md). If a fresh shell cannot
find the DLLs, add the CUDA `bin` directory to `PATH` (demo 13 prints a hint).
For the ONNX Runtime GPU path, `CudaProvisioner.InstallAsync` fetches the
runtime + cuDNN DLLs headlessly, without admin.

## Git / commit conventions
See [`CLAUDE_PROJECT_RULES.md`](../CLAUDE_PROJECT_RULES.md). In short: all code
strings and comments are in **English**; commits are made only when requested and
are co-authored.

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `CortexApp` won't start, asks for Windows App Runtime | Click "install", or build self-contained with `-r win-x64`. |
| GPU not used | Confirm `DeviceRuntime.CudaDeviceCount > 0`; ensure the CUDA Toolkit is installed and on the library path. |
| Demo 13 exits with "No CUDA GPU visible" | The pretraining run needs the 4070-class GPU and the CUDA 12 runtime on `PATH`. |
| Face/vision demos download models every run | They cache under `~/.predictivecoding/` (vision-models, faces); keep the folder. |
| FASTER-KV files left on disk | Tests use temp dirs; for app runs, point `NetworkConfig.Path` / `BasePath` at a scratch directory. |
