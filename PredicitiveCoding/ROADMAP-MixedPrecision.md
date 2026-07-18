# PredictiveCoding — Mixed-Precision roadmap (state-of-the-art 2026)

> **Mission — one mandate:** bring the predictive-coding stack to 2026 state-of-the-art
> numerics — **16-bit (and eventually 8-bit) storage/compute with full-precision
> accumulation** ("mixed precision") — on CPU (SIMD) and GPU (tensor cores), while the
> float32 path stays **always intact** as the reference implementation.

The design invariant across every phase: *storage light (Half/int8), accumulation full
(float32)*, and *master weights in float32* during training. Mixed precision is an
**additive** path — if you never quantize, the network behaves bit-for-bit as before.

### Legend

| Flag | Meaning |
|:----:|---------|
| ✅ | **Done** — implemented and covered by tests. |
| 🟡 | **Partial** — building block exists, needs extension. |
| ❌ | **To build** — not present yet. |

*Status as of 2026-07-19.*

**Foundation already in place** (pre-existing, not part of this effort):
`DenseTensor<T> where T : struct, INumber<T>` (generic math); all dense ops go through
`System.Numerics.Tensors` (SIMD); `PredictiveCoding.Device` already reaches CUDA/cuBLAS
(`sgemm`, float) with automatic CPU fallback.

---

## Phase 0 — Generic-math tensor foundation · ✅ done

| Task | Status | Notes |
|------|:------:|-------|
| Tensor generic over a numeric type | ✅ | `Core/Shape.cs` — `DenseTensor<T> : INumber<T>`. |
| All dense ops hardware-vectorised | ✅ | `Core/TensorUtils.cs` via `TensorPrimitives`. |

## Phase 1 — Mixed-precision core ops · ✅ done

| Task | Status | Notes |
|------|:------:|-------|
| `MatVecMulMixed` (W in Half, v/accum in float) | ✅ | `Core/TensorUtils.cs`. |
| `MatVecMulTMixed` (Wᵀ, same scheme) | ✅ | `Core/TensorUtils.cs`. |
| `ToHalf` / `ToFloat` SIMD conversions | ✅ | `Core/TensorUtils.cs`. |
| Unit tests (correctness within Half tol; memory halved) | ✅ | `TensorUtilsTests` (3). |

## Phase 2 — Layer / Network end-to-end · ✅ done

| Task | Status | Notes |
|------|:------:|-------|
| Transient Half weight copy on the layer | ✅ | `Layer.GenerativeWeightsHalf` (not persisted; serializer is explicit). |
| Quantize / re-sync master→Half | ✅ | `Layer.QuantizeWeightsToHalf()`. |
| Mixed forward branches | ✅ | `Layer.ComputePrediction` / `UpdateState` (float path untouched when copy is null). |
| **A** — mixed inference at network level | ✅ | `Network.InferGetTopMixed`. |
| **B** — mixed training (master-weights scheme) | ✅ | `Network.LearnMixed` (Half forward, float32 master update, copy never persisted). |
| Tests + full suite green (871/871) | ✅ | `NetworkTests` (2 new). |

## Phase 3 — Complete the mixed inference/prediction surface (CPU) · 🟡 in progress

| Task | Status | Notes |
|------|:------:|-------|
| `PredictMixed` / `PredictWithEnergyMixed` | ✅ | minimum-energy classification with Half weights; `PredictCore(mixed:true)`. Test: `PredictMixed_MatchesFloatPrediction`. |
| `InferMixed` (public `Infer` variant) | ❌ | transient mixed inference. |
| `ClassEnergiesMixed` | ❌ | per-class energies under mixed forward (used by `EnsembleNetwork`). |

Reuse the validated *quantize-clones-in-memory* pattern; the float path stays untouched.
Makes mixed precision usable for the real use case (classify/predict with a light model).
Full suite green after this phase: **872/872**.

## Phase 4 — Benchmarks (measure) · ❌

| Task | Status | Notes |
|------|:------:|-------|
| BenchmarkDotNet harness | ❌ | `MatVecMul(T)` float vs mixed; memory + time. |
| Network-level inference bench | ❌ | float vs mixed on real nets; quantify memory saving. |

Puts the number on the memory/bandwidth win (CPU) and, later, the GPU speedup.

## Phase 5 — GPU: mixed forward on tensor cores · 🟡 in progress (the real speed win)

| Task | Status | Notes |
|------|:------:|-------|
| cuBLAS mixed GEMM (`cublasGemmEx`) | ✅ | `Interop/Cublas.cs` binding + `CudaGemmProvider.MatMulMixed`/`MatMulMixedDevice`: A,B in R16F, C accumulated in Compute32F on the tensor cores; CPU fallback. |
| Mixed on the GEMM abstraction | ✅ | `IGemmProvider.MatMulMixed` (default = float) + `CpuGemmProvider` real mixed override (doubles as the CUDA CPU fallback and test oracle). |
| GPU verification | ⏳ | compiles + CPU-fallback + numeric parity green here (874/874); **the tensor-core run must be confirmed on CUDA hardware** — `dotnet test --filter MixedGemmTests` on a GPU box. |
| Route GEMM-heavy consumers to mixed | ❌ | opt-in mixed path for the LLM (`Gemm.Provider`) / batched inference. |
| Batched PC forward on GPU | ❌ | required for the predictive-coding forward to benefit from the GPU. |

**Caveat — why batching matters.** The predictive-coding forward is many small
matrix×vector products (batch = 1); a single small MatVec on the GPU is dominated by
host↔device transfer, so it does **not** benefit. The tensor-core win is on large
matrix×matrix GEMMs (the LLM stack, or a batched PC forward) — that is what this
primitive accelerates. On CPU mixed precision still saves **memory/bandwidth**.

## Phase 6 — Mixed-precision model I/O · ❌

| Task | Status | Notes |
|------|:------:|-------|
| Persist Half weights | ❌ | `LayerStore`/serializer flag, or a dedicated quantized-model format. |
| Load at half size, mixed-ready | ❌ | avoids re-quantizing on every mixed inference call. |

## Phase 7 — Sub-16-bit quantization · ❌ (ambitious)

| Task | Status | Notes |
|------|:------:|-------|
| BFloat16 storage path | ❌ | `System.Numerics.BFloat16` (net10) — same scheme as Half. |
| int8 post-training quantization | ❌ | per-tensor/per-channel scale + zero-point; calibration. |

The far end of 2026 SOTA.

---

## Notes

- **Loss scaling is NOT needed** in this scheme. The classic FP16-training trick guards
  against gradient underflow when gradients live in FP16. Here gradients are computed and
  applied in **float32** (master weights); only the forward uses Half. No underflow path.
- **Persistence safety.** `LayerSerializer` writes explicit fields, so the transient
  `GenerativeWeightsHalf` is invisible to FASTER — Phase 6 is the deliberate opt-in to
  change that.

## How this maps onto the stack

- **Core** (`PredictiveCoding`): `DenseTensor<T>`, `TensorUtils` mixed ops, `Layer`,
  `Network` — Phases 0–4, 6–7.
- **Device** (`PredictiveCoding.Device`): CUDA/cuBLAS backend — Phase 5.
