// ─────────────────────────────────────────────────────────────────────────────
// pcdevice.cu — CUDA implementation of the PredictiveCoding device bridge.
//
// Implements: device/context management, stream-ordered allocation
// (cudaMallocAsync), pinned host memory, sync/async copies, CUDA streams &
// events, GEMM via cuBLAS and cuBLASLt, and NCCL-style collectives.
//
// Compile NCCL support with -DPCD_USE_NCCL (and link -lnccl). Without it the
// collective entry points return DeviceStatus::NotSupported (8) and the managed
// layer falls back to its CPU collective implementation.
// ─────────────────────────────────────────────────────────────────────────────
#include "pcdevice.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <unordered_map>
#include <mutex>
#include <vector>

#ifdef PCD_USE_NCCL
#include <nccl.h>
#endif

namespace {

enum Status { OK = 0, NotInit = 1, NoDevice = 2, OOM = 3, Invalid = 4,
              CudaErr = 5, CublasErr = 6, NcclErr = 7, NotSupported = 8 };

std::mutex g_mutex;
std::unordered_map<int, cublasHandle_t>   g_cublas;     // per device
std::unordered_map<int, cublasLtHandle_t> g_cublasLt;   // per device

inline int currentDevice() { int d = 0; cudaGetDevice(&d); return d; }

inline int fromCuda(cudaError_t e)   { return e == cudaSuccess ? OK : (e == cudaErrorMemoryAllocation ? OOM : CudaErr); }
inline int fromCublas(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? OK : CublasErr; }

cublasHandle_t cublasFor(int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_cublas.find(device);
    if (it != g_cublas.end()) return it->second;
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    g_cublas[device] = h;
    return h;
}

cublasLtHandle_t cublasLtFor(int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_cublasLt.find(device);
    if (it != g_cublasLt.end()) return it->second;
    cublasLtHandle_t h = nullptr;
    cublasLtCreate(&h);
    g_cublasLt[device] = h;
    return h;
}

} // namespace

// ── Device / context ──────────────────────────────────────────────────────────
PCD_API int pcd_init(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess ? OK : NoDevice; }

PCD_API int pcd_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& kv : g_cublas)   cublasDestroy(kv.second);
    for (auto& kv : g_cublasLt) cublasLtDestroy(kv.second);
    g_cublas.clear(); g_cublasLt.clear();
    return OK;
}

PCD_API int pcd_cuda_device_count(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0; }
PCD_API int pcd_set_device(int deviceId) { return fromCuda(cudaSetDevice(deviceId)); }
PCD_API int pcd_device_synchronize(void) { return fromCuda(cudaDeviceSynchronize()); }

// ── Memory ────────────────────────────────────────────────────────────────────
PCD_API int pcd_malloc_async(void** ptr, size_t bytes, void* stream) {
    return fromCuda(cudaMallocAsync(ptr, bytes, (cudaStream_t)stream));
}
PCD_API int pcd_free_async(void* ptr, void* stream) {
    return fromCuda(cudaFreeAsync(ptr, (cudaStream_t)stream));
}
PCD_API int pcd_malloc(void** ptr, size_t bytes) { return fromCuda(cudaMalloc(ptr, bytes)); }
PCD_API int pcd_free(void* ptr) { return fromCuda(cudaFree(ptr)); }
PCD_API int pcd_host_alloc(void** ptr, size_t bytes) { return fromCuda(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault)); }
PCD_API int pcd_host_free(void* ptr) { return fromCuda(cudaFreeHost(ptr)); }

// ── Copies ────────────────────────────────────────────────────────────────────
PCD_API int pcd_memcpy_h2d(void* dst, const void* src, size_t bytes) { return fromCuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)); }
PCD_API int pcd_memcpy_d2h(void* dst, const void* src, size_t bytes) { return fromCuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)); }
PCD_API int pcd_memcpy_d2d(void* dst, const void* src, size_t bytes) { return fromCuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)); }
PCD_API int pcd_memcpy_async(void* dst, const void* src, size_t bytes, int kind, void* stream) {
    cudaMemcpyKind k = kind == 1 ? cudaMemcpyHostToDevice : kind == 2 ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    return fromCuda(cudaMemcpyAsync(dst, src, bytes, k, (cudaStream_t)stream));
}

// ── Streams / events ──────────────────────────────────────────────────────────
PCD_API int pcd_stream_create(void** stream)      { return fromCuda(cudaStreamCreate((cudaStream_t*)stream)); }
PCD_API int pcd_stream_destroy(void* stream)      { return fromCuda(cudaStreamDestroy((cudaStream_t)stream)); }
PCD_API int pcd_stream_synchronize(void* stream)  { return fromCuda(cudaStreamSynchronize((cudaStream_t)stream)); }
PCD_API int pcd_event_create(void** evt)          { return fromCuda(cudaEventCreate((cudaEvent_t*)evt)); }
PCD_API int pcd_event_record(void* evt, void* st) { return fromCuda(cudaEventRecord((cudaEvent_t)evt, (cudaStream_t)st)); }
PCD_API int pcd_event_synchronize(void* evt)      { return fromCuda(cudaEventSynchronize((cudaEvent_t)evt)); }
PCD_API int pcd_event_destroy(void* evt)          { return fromCuda(cudaEventDestroy((cudaEvent_t)evt)); }

// ── GEMM (cuBLAS) ─────────────────────────────────────────────────────────────
// Row-major C[m,n] = A[m,k]·B[k,n]. cuBLAS is column-major, so compute the
// transpose: Cᵀ[n,m] = Bᵀ · Aᵀ, i.e. sgemm(N, N, n, m, k, B, A, C).
PCD_API int pcd_sgemm(int m, int n, int k, const void* a, const void* b, void* c,
                      float alpha, float beta, void* stream) {
    cublasHandle_t h = cublasFor(currentDevice());
    cublasSetStream(h, (cudaStream_t)stream);
    cublasStatus_t s = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                   n, m, k, &alpha,
                                   (const float*)b, n,
                                   (const float*)a, k,
                                   &beta, (float*)c, n);
    return fromCublas(s);
}

// ── GEMM (cuBLASLt) ────────────────────────────────────────────────────────────
// Same row-major result via cuBLASLt with ROW order on all operands.
PCD_API int pcd_sgemm_lt(int m, int n, int k, const void* a, const void* b, void* c,
                         float alpha, float beta, void* stream) {
    cublasLtHandle_t lt = cublasLtFor(currentDevice());

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasOperation_t opN = CUBLAS_OP_N;
    int status = CublasErr;

    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) goto done;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, m, k, k);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, n, n);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, m, n, n);
    cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));

    if (cublasLtMatmul(lt, op, &alpha,
                       a, la, b, lb, &beta,
                       c, lc, c, lc,
                       nullptr, nullptr, 0, (cudaStream_t)stream) == CUBLAS_STATUS_SUCCESS)
        status = OK;

done:
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (op) cublasLtMatmulDescDestroy(op);
    return status;
}

// ── Batched GEMM (strided) ─────────────────────────────────────────────────────
PCD_API int pcd_sgemm_batched(int batch, int m, int n, int k, const void* a, const void* b, void* c,
                              float alpha, float beta, void* stream) {
    cublasHandle_t h = cublasFor(currentDevice());
    cublasSetStream(h, (cudaStream_t)stream);
    cublasStatus_t s = cublasSgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
        (const float*)b, n, (long long)n * k,
        (const float*)a, k, (long long)m * k,
        &beta, (float*)c, n, (long long)m * n, batch);
    return fromCublas(s);
}

// ── NCCL collectives ──────────────────────────────────────────────────────────
#ifdef PCD_USE_NCCL
inline ncclRedOp_t reduceOp(int op) {
    switch (op) { case 1: return ncclProd; case 2: return ncclMax; case 3: return ncclMin; default: return ncclSum; }
}

PCD_API int pcd_nccl_comm_init(int nDevices, const int* devices, void** comms) {
    std::vector<ncclComm_t> c(nDevices);
    if (ncclCommInitAll(c.data(), nDevices, devices) != ncclSuccess) return NcclErr;
    for (int i = 0; i < nDevices; ++i) comms[i] = (void*)c[i];
    return OK;
}
PCD_API int pcd_nccl_comm_destroy(void* comm) { ncclCommDestroy((ncclComm_t)comm); return OK; }

PCD_API int pcd_nccl_all_reduce(void* comm, const void* send, void* recv, size_t count, int op, void* stream) {
    return ncclAllReduce(send, recv, count, ncclFloat, reduceOp(op), (ncclComm_t)comm, (cudaStream_t)stream) == ncclSuccess ? OK : NcclErr;
}
PCD_API int pcd_nccl_broadcast(void* comm, void* buf, size_t count, int root, void* stream) {
    return ncclBroadcast(buf, buf, count, ncclFloat, root, (ncclComm_t)comm, (cudaStream_t)stream) == ncclSuccess ? OK : NcclErr;
}
PCD_API int pcd_nccl_reduce_scatter(void* comm, const void* send, void* recv, size_t recvCount, int op, void* stream) {
    return ncclReduceScatter(send, recv, recvCount, ncclFloat, reduceOp(op), (ncclComm_t)comm, (cudaStream_t)stream) == ncclSuccess ? OK : NcclErr;
}
PCD_API int pcd_nccl_all_gather(void* comm, const void* send, void* recv, size_t sendCount, void* stream) {
    return ncclAllGather(send, recv, sendCount, ncclFloat, (ncclComm_t)comm, (cudaStream_t)stream) == ncclSuccess ? OK : NcclErr;
}
#else
PCD_API int pcd_nccl_comm_init(int, const int*, void**)                 { return NotSupported; }
PCD_API int pcd_nccl_comm_destroy(void*)                                { return NotSupported; }
PCD_API int pcd_nccl_all_reduce(void*, const void*, void*, size_t, int, void*) { return NotSupported; }
PCD_API int pcd_nccl_broadcast(void*, void*, size_t, int, void*)        { return NotSupported; }
PCD_API int pcd_nccl_reduce_scatter(void*, const void*, void*, size_t, int, void*) { return NotSupported; }
PCD_API int pcd_nccl_all_gather(void*, const void*, void*, size_t, void*) { return NotSupported; }
#endif
