# PredictiveCoding.Device

Device management for tensors: CPU / CUDA / multi-GPU, layered **on top of**
`System.Numerics.Tensors` (never modifying it).

**No custom native library.** GPU work P/Invokes the standard CUDA Toolkit
libraries directly (`cudart`, `cublas`) — there is nothing to compile or ship.
Install the CUDA Toolkit for acceleration; with no toolkit or GPU the whole API
runs on the CPU automatically.

- `Device` — `Cpu`, `Cuda(id)`, `CudaMulti`, `Metal`, `Rocm`.
- `DeviceTensor<T>` — host (`DenseTensor<T>`) + device buffer, `DevicePtr`, stream.
- `CudaAllocator` — pooling, pinned host memory, `cudaMallocAsync`/`FreeAsync`.
- `CudaStreamManager` — per-thread streams, events, synchronisation.
- `CudaGemmProvider` — cuBLAS GEMM, plugs into the LLM `IGemmProvider`.
- `NcclProvider` — AllReduce / Broadcast / ReduceScatter / AllGather (CPU impl.).
- `TensorOps` — `MatMul`, `MatMulAsync`, `CopyTo`, `CopyToAsync`, `Synchronize`.
- `DeviceRegistry` — install the CUDA backend into the LLM stack.

The native bindings live in `Interop/` (`Cuda.cs`, `Cublas.cs`,
`NativeResolver.cs`); the resolver maps `cudart`/`cublas` to their version-suffixed
files (`cudart64_12.dll` / `libcudart.so.12`, …).

## Requirements for GPU
- NVIDIA driver + **CUDA Toolkit ≥ 12** on `PATH` / `LD_LIBRARY_PATH`
  (provides `cudart` and `cublas`). Verify:
  ```csharp
  Console.WriteLine($"cuda devices: {DeviceRuntime.CudaDeviceCount}");
  ```
- Multi-GPU NCCL is **not** part of cudart/cublas, so `NcclProvider` uses a CPU
  implementation. A real NCCL path is a clean extension (add a `libnccl` binding)
  that does not reintroduce a custom native build.

## Examples

```csharp
using PredictiveCoding.Core;
using PredictiveCoding.Device;
using PredictiveCoding.Device.Nccl;
using PredictiveCoding.Device.Streams;
using PredictiveCoding.Device.Interop;

DeviceRegistry.AutoSelect();                                   // install CUDA backend if a GPU is present

var host = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);   // 1. CPU tensor
DeviceTensor<float> g = host.ToDevice(Device.Cuda());          // 2. → GPU (CUDA→CPU fallback automatic)

var b = new DenseTensor<float>(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2).ToDevice(Device.Cuda());
using var c = TensorOps.MatMul(g, b);                          // 3. cuBLAS GEMM [2,3]·[3,2]

using var nccl = NcclProvider.ForAllDevices();                 // 4. multi-GPU all-reduce
float[][] perRank = Enumerable.Range(0, nccl.RankCount).Select(_ => new[] { 1f, 2f, 3f }).ToArray();
nccl.AllReduce(perRank, ReduceOp.Sum);

CudaStreamHandle stream = CudaStreamManager.Shared.CreateStream(0);
using var d = TensorOps.MatMulAsync(g, b, stream);             // 5. async on a stream
TensorOps.Synchronize(stream);

DenseTensor<float> result = c.ToHostTensor();                  // 6. back to the CPU
```

## Extending to Metal / ROCm
Implement `IGemmProvider` (compute) and `IComputeDevice` (collectives) for the new
backend and register them — the model code is backend-agnostic.
