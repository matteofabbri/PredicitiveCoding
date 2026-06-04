# PredictiveCoding.Device

Device management for tensors: CPU / CUDA / multi-GPU, layered **on top of**
`System.Numerics.Tensors` (never modifying it). Everything degrades gracefully —
with no native library or GPU the whole API runs on the CPU.

- `Device` — `Cpu`, `Cuda(id)`, `CudaMulti`, `Metal`, `Rocm`.
- `DeviceTensor<T>` — host (`DenseTensor<T>`) + device buffer, `DevicePtr`, stream.
- `CudaAllocator` — pooling, pinned host memory, `cudaMallocAsync`/`FreeAsync`.
- `CudaStreamManager` — per-thread streams, events, synchronisation.
- `CudaGemmProvider` — cuBLASLt GEMM, plugs into the LLM `IGemmProvider`.
- `NcclProvider` — AllReduce / Broadcast / ReduceScatter / AllGather.
- `TensorOps` — `MatMul`, `MatMulAsync`, `CopyTo`, `CopyToAsync`, `Synchronize`.
- `DeviceRegistry` — install the CUDA backend into the LLM stack.

Native build & NCCL setup: see [`native/BUILD.md`](native/BUILD.md) and
[`native/NCCL.md`](native/NCCL.md). Runnable samples: [`DeviceExamples.cs`](DeviceExamples.cs).

## Examples

```csharp
using PredictiveCoding.Core;
using PredictiveCoding.Device;
using PredictiveCoding.Device.Nccl;
using PredictiveCoding.Device.Streams;
using PredictiveCoding.Device.Interop;

// 0. Pick and install the best backend (CUDA when available, else CPU).
DeviceRegistry.AutoSelect();

// 1. Create a tensor on the CPU.
var host = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);

// 2. Move it onto the GPU (PyTorch-style; CUDA→CPU fallback is automatic).
DeviceTensor<float> g = host.ToDevice(Device.Cuda());        // or Device.Cuda(deviceId)

// 3. MatMul on the GPU (cuBLASLt).
var b = new DenseTensor<float>(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2).ToDevice(Device.Cuda());
using var c = TensorOps.MatMul(g, b);                        // [2,3]·[3,2] = [2,2]

// 4. Multi-GPU all-reduce (NCCL, or CPU mock with the same API).
using var nccl = NcclProvider.ForAllDevices();
float[][] perRank = /* one buffer per device */ Enumerable.Range(0, nccl.RankCount)
    .Select(_ => new[] { 1f, 2f, 3f }).ToArray();
nccl.AllReduce(perRank, ReduceOp.Sum);

// 5. Asynchronous matmul on a stream.
CudaStreamHandle stream = CudaStreamManager.Shared.CreateStream(deviceId: 0);
using var d = TensorOps.MatMulAsync(g, b, stream);
TensorOps.Synchronize(stream);

// 6. Back to the CPU.
DenseTensor<float> result = c.ToHostTensor();
```

## Extending to Metal / ROCm
Implement `IGemmProvider` (compute) and `IComputeDevice` (collectives) for the
new backend and register them — the model code is backend-agnostic, so nothing
above the device layer changes.
