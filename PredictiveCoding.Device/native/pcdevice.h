// ─────────────────────────────────────────────────────────────────────────────
// pcdevice.h — C ABI for the PredictiveCoding CUDA device bridge.
//
// All functions return 0 on success and a non-zero DeviceStatus on failure
// (1 NotInitialized, 2 NoDevice, 3 OutOfMemory, 4 InvalidValue, 5 CudaError,
//  6 CublasError, 7 NcclError, 8 NotSupported). Matches NativeMethods.cs.
//
// Build into pcdevice.dll / libpcdevice.so (see BUILD.md).
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PCDEVICE_H
#define PCDEVICE_H

#include <stddef.h>

#if defined(_WIN32)
  #define PCD_API __declspec(dllexport)
#else
  #define PCD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ── Device / context ──────────────────────────────────────────────────────────
PCD_API int pcd_init(void);
PCD_API int pcd_shutdown(void);
PCD_API int pcd_cuda_device_count(void);
PCD_API int pcd_set_device(int deviceId);
PCD_API int pcd_device_synchronize(void);

// ── Memory ────────────────────────────────────────────────────────────────────
PCD_API int pcd_malloc_async(void** ptr, size_t bytes, void* stream);
PCD_API int pcd_free_async(void* ptr, void* stream);
PCD_API int pcd_malloc(void** ptr, size_t bytes);
PCD_API int pcd_free(void* ptr);
PCD_API int pcd_host_alloc(void** ptr, size_t bytes);   // pinned (page-locked) host memory
PCD_API int pcd_host_free(void* ptr);

// ── Copies ────────────────────────────────────────────────────────────────────
PCD_API int pcd_memcpy_h2d(void* dst, const void* src, size_t bytes);
PCD_API int pcd_memcpy_d2h(void* dst, const void* src, size_t bytes);
PCD_API int pcd_memcpy_d2d(void* dst, const void* src, size_t bytes);
PCD_API int pcd_memcpy_async(void* dst, const void* src, size_t bytes, int kind, void* stream); // 1=H2D 2=D2H 3=D2D

// ── Streams / events ──────────────────────────────────────────────────────────
PCD_API int pcd_stream_create(void** stream);
PCD_API int pcd_stream_destroy(void* stream);
PCD_API int pcd_stream_synchronize(void* stream);
PCD_API int pcd_event_create(void** evt);
PCD_API int pcd_event_record(void* evt, void* stream);
PCD_API int pcd_event_synchronize(void* evt);
PCD_API int pcd_event_destroy(void* evt);

// ── GEMM  (row-major)  C[m,n] = alpha * A[m,k] * B[k,n] + beta * C ──────────────
PCD_API int pcd_sgemm(int m, int n, int k, const void* a, const void* b, void* c,
                      float alpha, float beta, void* stream);
PCD_API int pcd_sgemm_lt(int m, int n, int k, const void* a, const void* b, void* c,
                         float alpha, float beta, void* stream);
PCD_API int pcd_sgemm_batched(int batch, int m, int n, int k, const void* a, const void* b, void* c,
                              float alpha, float beta, void* stream);

// ── NCCL-style collectives ────────────────────────────────────────────────────
// op: 0=Sum 1=Prod 2=Max 3=Min. comms[] receives one communicator per device.
PCD_API int pcd_nccl_comm_init(int nDevices, const int* devices, void** comms);
PCD_API int pcd_nccl_comm_destroy(void* comm);
PCD_API int pcd_nccl_all_reduce(void* comm, const void* send, void* recv, size_t count, int op, void* stream);
PCD_API int pcd_nccl_broadcast(void* comm, void* buf, size_t count, int root, void* stream);
PCD_API int pcd_nccl_reduce_scatter(void* comm, const void* send, void* recv, size_t recvCount, int op, void* stream);
PCD_API int pcd_nccl_all_gather(void* comm, const void* send, void* recv, size_t sendCount, void* stream);

#ifdef __cplusplus
}
#endif

#endif // PCDEVICE_H
