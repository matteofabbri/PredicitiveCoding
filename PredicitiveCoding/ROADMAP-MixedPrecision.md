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
| `InferMixed` (public `Infer` variant) | ✅ | transient mixed inference (Half clones, masters untouched — tested bit-identical before/after). |
| `ClassEnergiesMixed` | ✅ | per-class energies under mixed forward; tracks the float energies within Half tolerance and agrees on the minimum-energy class (test). Phase 3 complete 2026-07-19. |

Reuse the validated *quantize-clones-in-memory* pattern; the float path stays untouched.
Makes mixed precision usable for the real use case (classify/predict with a light model).
Full suite green after this phase: **872/872**.

## Phase 4 — Benchmarks (measure) · ✅ done 2026-07-19

| Task | Status | Notes |
|------|:------:|-------|
| Benchmark harness | ✅ | `tools/MixedPrecisionBench` — in-house Stopwatch harness (warmup + median of 5; a memory/bandwidth story does not need BenchmarkDotNet-grade rigor, and the repo stays dependency-light). Run with `dotnet run -c Release`. |
| Network-level inference bench | ✅ | included (CIFAR-scale `[3072,512,128,10]` net, 20 settling steps) plus the 1024³ GEMM CPU/CUDA row. |

Measured numbers (Release, Ryzen-class CPU + RTX 4070 SUPER, 2026-07-19):

| what | float32 | mixed | note |
|------|--------:|------:|------|
| `MatVecMul` 512×512 | 29 µs | 89 µs | conversion dominates at micro scale |
| `MatVecMul` 2048×2048 | 369 µs | 1169 µs | idem — CPU mixed is a MEMORY win, not a speed win |
| Network `InferGetTop`, 20 steps | 42.5 ms | **26.4 ms** | at network scale the halved weight traffic wins (cache/bandwidth-bound); weights 6.3 → 3.1 MB |
| GEMM 1024³ CPU | 14 ms | 16 ms | |
| GEMM 1024³ CUDA mixed | — | **1.9 ms** | tensor cores, transfers included |

**Finding.** The honest CPU story is nuanced: per-op the Half up-conversion makes mixed
~3× slower, but on the real PC network (weights re-streamed 40× per inference) the halved
footprint flips it to **1.6× faster** — the bandwidth win the roadmap predicted, showing up
exactly where the working set outgrows the cache. The GPU speedup lives in the GEMMs.

## Phase 5 — GPU: mixed forward on tensor cores · ✅ done 2026-07-19 (the real speed win)

| Task | Status | Notes |
|------|:------:|-------|
| cuBLAS mixed GEMM (`cublasGemmEx`) | ✅ | `Interop/Cublas.cs` binding + `CudaGemmProvider.MatMulMixed`/`MatMulMixedDevice`: A,B in R16F, C accumulated in Compute32F on the tensor cores; CPU fallback. |
| Mixed on the GEMM abstraction | ✅ | `IGemmProvider.MatMulMixed` (default = float) + `CpuGemmProvider` real mixed override (doubles as the CUDA CPU fallback and test oracle). |
| GPU verification | ✅ | **verified 2026-07-19 on the RTX 4070 SUPER**: `cublasGemmEx` mixed GEMM vs CPU float oracle, max rel err 6.4e-3 over 256×512×256 (Half-rounding territory, as designed). Release-build timing (Phase 4 harness): 1024³ GEMM = **1.9 ms on the tensor cores vs 14 ms CPU float / 16 ms CPU mixed**, transfers included (~7×). The CUDA 12 runtime DLLs (`cudart64_12`/`cublas64_12`) need only be discoverable — no Toolkit install required; LM Studio's bundled copies on PATH (`~/.cache/lm-studio/extensions/backends/vendor/win-llama-cuda12-vendor-v2`) work. `DeviceRuntime.NativeLoaded` fixed to probe through the resolver's version-suffixed names. |
| Route GEMM-heavy consumers to mixed | ✅ | `Ag.MatMulRoute` (2026-07-19): the trainable DecoderLM stack — THE GEMM-heavy consumer — routes MatMul forwards through `Gemm.Provider` (`Provider` = float, `ProviderMixed` = 16-bit operands/fp32 accumulation); backwards stay inline fp32 (master-precision gradients, same no-loss-scaling story as the core). Default `Inline` is bit-identical. Tests: `LlmGemmRouteTests` (4 — spy provider, bit-for-bit float route, mixed tolerance, mixed training). The inference `GptModel` stays unrouted by design: token-at-a-time matvec, see caveat below. |
| Batched PC forward on GPU | ✅ | `Network.ExportGenerativeSnapshot()` (rejects diag-precision/priors — would not be reproduced) + `Device/BatchedPcForward.InferGetTopBatch`: B settlings as two GEMMs per layer per step, routed through `Gemm.Provider`. Parity with per-sample `InferGetTop(resetState: true)` tested (incl. FFG warm-start; mixed within Half tolerance). Measured (64×[3072,512,128,10], 20 steps): per-sample loop 574 ms → **batched CPU 158 ms → batched CUDA 75 ms**. CUDA *mixed* is 94 ms here — B=64 GEMMs are too small to repay the Half conversion; mixed pays on big GEMMs, exactly the Phase 4 finding. |

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
