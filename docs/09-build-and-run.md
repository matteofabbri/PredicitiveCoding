# 9. Build & run

## Prerequisites
- **.NET 10 SDK** (the whole solution targets `net10.0`; `CortexApp` targets
  `net10.0-windows10.0.19041.0`).
- Windows for the WinUI demo app (`CortexApp`). The libraries themselves are
  cross-platform.
- Optional: **CUDA Toolkit ≥ 12** for the CUDA backend; a macOS LibTorch backend
  for Metal.

## Build & test

```bash
# Build the whole solution
dotnet build PrediciiveCoding.slnx

# Run the test suite (300+ tests)
dotnet test PredicitiveCoding.UnitTests/PredicitiveCoding.UnitTests.csproj
```

The solution builds under the default **AnyCPU** configuration. `CortexApp` is an
*unpackaged* WinUI app specifically so AnyCPU works (the packaged WinUI harness
`PredictiveCoding.Tests` is deliberately not in the solution).

## Run the console demos (`PredictiveCoding.Main`)

```bash
dotnet run --project PredictiveCoding.Main
```
Includes `MillidgeReview2022Demo`, `HardClassificationDemo`, `TemporalDemo`,
`MixtureOfExpertsDemo`, and a CIFAR-10 runner (`Data/CifarDataset` auto-downloads
the dataset on first use).

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
No native DLL to build — see [doc 7](07-device-gpu.md).

### Metal (Apple Silicon)
Add a LibTorch backend with MPS, then enable it:
```bash
dotnet add PredictiveCoding.Device.Torch package libtorch-cpu-macos-arm64
```
```csharp
PredictiveCoding.Device.Torch.TorchBackend.EnableMetal();
```

## Git / commit conventions
See [`CLAUDE_PROJECT_RULES.md`](../CLAUDE_PROJECT_RULES.md). In short: all code
strings and comments are in **English**; commits are made only when requested and
are co-authored.

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `CortexApp` won't start, asks for Windows App Runtime | Click "install", or build self-contained with `-r win-x64`. |
| GPU not used | Confirm `DeviceRuntime.CudaDeviceCount > 0`; ensure the CUDA Toolkit is installed and on the library path. |
| `TorchBackend.MetalAvailable` is false | Add the macOS-arm64 LibTorch backend package; you must be on Apple Silicon. |
| FASTER-KV files left on disk | Tests use temp dirs; for app runs, point `NetworkConfig.Path` / `BasePath` at a scratch directory. |
