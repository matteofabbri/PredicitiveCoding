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
| **`BatchedPcForward`** | Batched **predictive-coding** inference: exports a `GenerativeSnapshot` from a core `Network` and settles B inputs as GEMMs (float or mixed) instead of looping per sample. |
| **`RocmGemmProvider`** | ROCm placeholder routing to the CPU path. |

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

## Training on the device (`Training/`, `Kernels/`)

The device layer is not just a GEMM hook — it owns the **pretraining engine**
the web-corpus run (demo 13, [doc 6](06-llm.md)) executes:

- **`DevicePretrainer`** — the resumable run loop over a `CorpusSource`:
  gradient-accumulated AdamW, warmup+cosine LR, periodic held-out eval and
  **quality probes** (bits/byte + a live sample), best-on-val and rolling
  checkpoints, JSONL progress log, graceful Ctrl-C, and **resume** including the
  device-side Adam moments. Configured through `DevicePretrainOptions`
  (batching, LR plan, eval/checkpoint cadence, mixed precision) and reported via
  `DevicePretrainReport`/`DevicePretrainProgress`.
- **`DeviceDecoderTrainer`** — the step engine underneath: block-wise forward /
  backward on the device.
- **`Kernels/`** — `DeviceTrainKernels` / `DeviceMath`: CUDA C kernels carried as
  source and compiled at runtime through the NVRTC bindings in `Interop/`
  (`CudaKernel` plumbing). A full forward+backward+Adam step is **one
  uninterrupted chain of kernel and cuBLAS launches** on a caller-supplied
  stream — the host synchronises once, when it reads the loss. The attention
  trio is flash-style: block per (query, head), K/V tiles staged in shared
  memory, online-softmax reduction per tile, backward replaying softmax weights
  from the saved logsumexp — **no `[seq, seq]` matrix ever materialises**.

## Extending to ROCm / new backends
Implement `IGemmProvider` (and `IComputeDevice` for collectives) and register it —
model code is backend-agnostic, so a new accelerator slots in without touching it.

## What runs where
| Path | Status |
|------|--------|
| CPU | Always (SIMD + multithreaded). |
| CUDA (cuBLAS) | When the CUDA Toolkit + an NVIDIA GPU are present. |
| CUDA training kernels | Same requirement; driven by `DevicePretrainer` (demo 13 needs a real GPU — a 4070-class card is the reference). |
| Multi-GPU NCCL | CPU implementation today; a `libnccl` binding is a clean extension. |
