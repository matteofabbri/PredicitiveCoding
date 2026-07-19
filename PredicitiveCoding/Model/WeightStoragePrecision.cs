namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// WeightStoragePrecision  —  how a layer's generative weights are PERSISTED
// (mixed-precision roadmap, Phases 6-7). Runtime compute is unaffected: masters
// are always widened back to float32 on load, and the mixed forward keeps
// using the Half copy. This is a storage choice, made explicitly and one-shot
// via Network.PersistWeightsQuantized — ordinary training writes Float32.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Persisted format of the generative weights.</summary>
public enum WeightStoragePrecision
{
    /// <summary>Full fp32 — the default, bit-exact.</summary>
    Float32,

    /// <summary>IEEE Half (fp16): 2 bytes/weight; the record ships mixed-ready (Phase 6).</summary>
    Half,

    /// <summary>bfloat16: 2 bytes/weight, fp32 range with an 8-bit mantissa (Phase 7).</summary>
    BFloat16,

    /// <summary>int8 affine per-row (per-channel scale + zero-point): ~1 byte/weight (Phase 7).</summary>
    Int8,
}
