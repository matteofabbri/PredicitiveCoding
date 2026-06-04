# 7. Device & GPU — `PredictiveCoding.Device` (+ `.Torch`)

Device management for tensors with pluggable CPU / **CUDA** / **Metal** backends,
layered on top of `System.Numerics.Tensors` (which is never modified). Everything
degrades gracefully: with no GPU the whole API runs on the CPU.

## Design

The key idea is that models never reference a backend — they use the interfaces
`IGemmProvider` (compute) and `IComputeDevice` (collectives) defined in the LLM
infrastructure. A backend registers itself and the model code is unchanged.

```
PredictiveCoding.Device          → CUDA backend (direct cuBLAS P/Invoke)
PredictiveCoding.Device.Torch    → Metal backend (TorchSharp / MPS)
```

## `PredictiveCoding.Device`

| Type | Role |
|------|------|
| **`Device`** | `Cpu`, `Cuda(id)`, `CudaMulti`, `Metal`, `Rocm`. |
| **`DeviceTensor<T>`** | Host (`DenseTensor<T>`) + device buffer, `DevicePtr`, `Stream`, `.To(device)` / `.ToHost()`. |
| **`DeviceRuntime`** | Detects CUDA via `cudaGetDeviceCount`; `CudaAvailable`, `Resolve` (CUDA→CPU fallback). |
| **`CudaAllocator`** | Bucketed pooling, pinned host memory, `cudaMallocAsync`/`FreeAsync`. |
| **`CudaStreamManager`** | Per-thread streams, events, synchronisation. |
| **`CudaGemmProvider`** | cuBLAS `sgemm` (column-major trick); implements `IGemmProvider`. |
| **`NcclProvider`** | AllReduce / Broadcast / ReduceScatter / AllGather (CPU implementation; NCCL is not part of cudart/cublas). |
| **`TensorOps`** | `MatMul`, `MatMulAsync`, `CopyTo`, `CopyToAsync`, `Synchronize`. |
| **`DeviceRegistry`** | `EnableCuda()` installs the cuBLAS provider into the LLM stack; `AutoSelect()`. |

### No custom native DLL
GPU work P/Invokes the **standard CUDA Toolkit libraries** (`cudart`, `cublas`)
directly — `Interop/Cuda.cs`, `Interop/Cublas.cs`, with `NativeResolver` mapping
the logical names to their version-suffixed files (`cudart64_12.dll` /
`libcudart.so.12`). Install the CUDA Toolkit for acceleration; nothing to compile.

```csharp
DeviceRegistry.EnableCuda();                       // route LLM matmuls to cuBLAS if a GPU is present
var g = host.ToDevice(Device.Cuda());              // upload (CUDA→CPU fallback automatic)
using var c = TensorOps.MatMul(g, b);              // cuBLAS GEMM
var result = c.ToHostTensor();                     // download
```

## `PredictiveCoding.Device.Torch` (Metal / MPS)

Metal has no flat C ABI like cudart/cublas, so the managed route to GPU matmul on
Apple Silicon is LibTorch's **MPS** backend via **TorchSharp**.

| Type | Role |
|------|------|
| **`TorchGemmProvider`** | Runs the matmul on a torch device (`mps` / `cuda` / `cpu`); implements `IGemmProvider`; falls back to CPU. |
| **`TorchBackend`** | `MetalAvailable` / `CudaAvailable` probes (never throw if LibTorch is absent); `EnableMetal()` / `EnableBest()`. |

The `TorchSharp` package compiles everywhere; to run you add **one** LibTorch
backend per platform — e.g. `libtorch-cpu-macos-arm64` (ships MPS) on a Mac.

```csharp
if (TorchBackend.EnableMetal())          // route the LLM GEMM stack to the Apple GPU
    Console.WriteLine("Running on Metal (MPS).");
```

## Extending to ROCm / new backends
Implement `IGemmProvider` (and `IComputeDevice` for collectives) and register it.
ROCm already works through `TorchGemmProvider` on a HIP device LibTorch exposes as
CUDA; it just isn't wired to a dedicated switch.

## What runs where
| Path | Status |
|------|--------|
| CPU | Always (SIMD + multithreaded). |
| CUDA (cuBLAS) | When the CUDA Toolkit + an NVIDIA GPU are present. |
| Metal (MPS) | On Apple Silicon with the macOS LibTorch backend. |
| Multi-GPU NCCL | CPU implementation today; a `libnccl` binding is a clean extension. |
| Fused flash-attention kernel | Not implemented (needs a GPU kernel). |
