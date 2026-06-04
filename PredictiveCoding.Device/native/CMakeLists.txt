cmake_minimum_required(VERSION 3.24)
project(pcdevice LANGUAGES CXX CUDA)

# Build: cmake -B build -DCMAKE_BUILD_TYPE=Release [-DPCD_USE_NCCL=ON]
#        cmake --build build --config Release
# Output: pcdevice.dll (Windows) / libpcdevice.so (Linux), to place next to the .NET app.

option(PCD_USE_NCCL "Enable NCCL multi-GPU collectives" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target the GPU architectures you deploy on (e.g. 70=V100, 80=A100, 89=Ada, 90=Hopper).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
endif()

add_library(pcdevice SHARED pcdevice.cu pcdevice.h)
set_target_properties(pcdevice PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

# cuBLAS + cuBLASLt ship with the CUDA Toolkit.
target_link_libraries(pcdevice PRIVATE cublas cublasLt cudart)

if(PCD_USE_NCCL)
  target_compile_definitions(pcdevice PRIVATE PCD_USE_NCCL)
  find_library(NCCL_LIB nccl REQUIRED)
  target_link_libraries(pcdevice PRIVATE ${NCCL_LIB})
endif()
