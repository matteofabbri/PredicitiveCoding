# Configuring NCCL (multi-GPU collectives)

NCCL powers `NcclProvider`'s `AllReduce` / `Broadcast` / `ReduceScatter` /
`AllGather` across multiple CUDA devices. It is optional — with a single GPU (or
no NCCL) `NcclProvider` transparently uses an equivalent CPU implementation, so
the API behaves identically everywhere.

## Install
- Linux: install `libnccl2` + `libnccl-dev` from NVIDIA's apt/yum repos, or the
  tarball from developer.nvidia.com/nccl. Ensure `libnccl.so` is on the linker
  path when building (`-DPCD_USE_NCCL=ON`) and on `LD_LIBRARY_PATH` at runtime.
- Windows: NCCL is Linux-first; for Windows multi-GPU use WSL2 or a Linux host.

## Build with NCCL
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPCD_USE_NCCL=ON
cmake --build build --config Release
```
Without `-DPCD_USE_NCCL` the collective entry points return `NotSupported` and
the managed layer uses its CPU collectives.

## Runtime environment (typical, single node, N GPUs)
```bash
export NCCL_DEBUG=INFO          # log topology / ring construction
export NCCL_P2P_LEVEL=NVL       # prefer NVLink peer-to-peer when available
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Usage
```csharp
// One communicator per device, created via ncclCommInitAll under the hood.
using var nccl = NcclProvider.ForAllDevices();   // ranks = #CUDA devices
nccl.AllReduce(perDeviceBuffers, ReduceOp.Sum);  // sum-reduce + broadcast to all
```

## Multi-node
This bridge wires up single-process, single-node, multi-GPU NCCL
(`ncclCommInitAll`). Multi-node training needs a bootstrap (MPI or
`ncclGetUniqueId` + an out-of-band exchange) and one rank per process — a
straightforward extension of `pcd_nccl_comm_init`.
