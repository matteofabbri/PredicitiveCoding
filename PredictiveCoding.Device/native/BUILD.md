# Building the CUDA bridge (`pcdevice`)

The managed `PredictiveCoding.Device` assembly P/Invokes a native library named
**`pcdevice`** (`pcdevice.dll` on Windows, `libpcdevice.so` on Linux). It is
**optional**: if the library (or a GPU) is missing, the .NET stack runs entirely
on the CPU. Build it only when you want GPU acceleration.

## Prerequisites
- NVIDIA CUDA Toolkit ≥ 12.x (provides `nvcc`, `cudart`, `cublas`, `cublasLt`).
- CMake ≥ 3.24 and a host C++17 compiler (MSVC on Windows, GCC/Clang on Linux).
- (Optional, multi-GPU) NCCL ≥ 2.18.

## Build (CPU/single-GPU GEMM, no NCCL)
```bash
cd PredictiveCoding.Device/native
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Build with multi-GPU NCCL collectives
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPCD_USE_NCCL=ON
cmake --build build --config Release
```

## Deploy
Copy the produced library next to your application binary so the .NET runtime's
default DLL search finds it:

- Windows: `pcdevice.dll` → `…/bin/Release/net10.0/`
- Linux:   `libpcdevice.so` → same output folder (or a directory on
  `LD_LIBRARY_PATH`).

`DeviceRuntime` probes the library once with `NativeLibrary.TryLoad`; a missing
library simply means `DeviceRuntime.CudaAvailable == false`.

## Choosing GPU architectures
Set `-DCMAKE_CUDA_ARCHITECTURES="80;90"` for the SM versions you deploy on
(70 = V100, 80 = A100, 86/89 = Ada, 90 = Hopper). The default builds a fat binary
covering common recent architectures.

## Quick verification
```csharp
Console.WriteLine($"native loaded : {DeviceRuntime.NativeLoaded}");
Console.WriteLine($"cuda devices  : {DeviceRuntime.CudaDeviceCount}");
```
