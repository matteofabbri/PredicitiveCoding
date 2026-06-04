# PredictiveCoding.Device.Torch  (Apple Metal / MPS backend)

A GEMM backend on **TorchSharp** (managed bindings to LibTorch). It exists because
Metal has **no flat C ABI** like cudart/cublas — the only managed way to run GPU
matmul on Apple Silicon is LibTorch's **MPS** backend.

It plugs into the LLM stack via `IGemmProvider`, exactly like the direct-cuBLAS
`CudaGemmProvider` — the two **coexist**: cuBLAS stays the default on NVIDIA, this
covers Metal. Any failure (missing backend, unsupported op) falls back to the CPU.

## Install a LibTorch backend (per platform)
The `TorchSharp` managed package compiles everywhere; to actually run you add **one**
native backend package for your platform:

| Platform | Backend package | Device |
|---|---|---|
| macOS (Apple Silicon) | `libtorch-cpu-macos-arm64` | **Metal / MPS** |
| NVIDIA | `libtorch-cuda-12.x` | CUDA |
| CPU only | `libtorch-cpu` | CPU |

```bash
# On a Mac:
dotnet add PredictiveCoding.Device.Torch package libtorch-cpu-macos-arm64
```

## Usage
```csharp
using PredictiveCoding.Device.Torch;

// Route the whole LLM GEMM stack to the Apple GPU (Metal) when available.
if (TorchBackend.EnableMetal())
    Console.WriteLine("Running matmuls on Metal (MPS).");
// else: the existing provider (CPU, or the direct-cuBLAS CUDA backend) stays.

// Or pick the best torch device (Metal → CUDA):
TorchBackend.EnableBest();
```

`TorchBackend.MetalAvailable` / `CudaAvailable` probe the device safely (they never
throw if LibTorch is missing). The provider is only constructed once a device is
known to be available.

## Notes
- This is the **GEMM** provider; the CUDA `DeviceTensor<T>` memory/stream layer in
  `PredictiveCoding.Device` is cudart-specific and unrelated — Metal acceleration
  flows through the pluggable `IGemmProvider`, leaving CUDA untouched.
- ROCm (AMD) also works through this provider on a HIP device that LibTorch exposes
  as CUDA; it just isn't wired to a dedicated switch yet.
