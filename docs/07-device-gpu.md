# 7. Device & GPU — `PredictiveCoding.Device`

Device management for tensors with pluggable CPU / **CUDA** backends, layered on
top of `System.Numerics.Tensors` (which is never modified). Everything degrades
gracefully: with no GPU the whole API runs on the CPU.

> **Note:** the optional `PredictiveCoding.Device.Torch` project (Apple **Metal/MPS**
> via TorchSharp) has been removed. The solution now targets **CPU** and **NVIDIA
> CUDA** only; there is no Apple-GPU backend.

## Design

The key idea is that models never reference a backend — they use the interfaces
`IGemmProvider` (compute) and `IComputeDevice` (collectives) defined in the LLM
infrastructure. A backend registers itself and the model code is unchanged.

```
PredictiveCoding.Device          → CUDA backend (direct cuBLAS P/Invoke)
```

## `PredictiveCoding.Device`

| Type | Role |
|------|------|
| **`Device`** | `Cpu`, `Cuda(id)`, `CudaMulti`. (Any `Metal` / `Rocm` enum value resolves to the CPU now that the Torch backend is gone.) |
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

## Extending to ROCm / new backends
Implement `IGemmProvider` (and `IComputeDevice` for collectives) and register it —
model code is backend-agnostic, so a new accelerator slots in without touching it.

## What runs where
| Path | Status |
|------|--------|
| CPU | Always (SIMD + multithreaded). |
| CUDA (cuBLAS) | When the CUDA Toolkit + an NVIDIA GPU are present. |
| Multi-GPU NCCL | CPU implementation today; a `libnccl` binding is a clean extension. |
| Fused flash-attention kernel | Not implemented (needs a GPU kernel). |
