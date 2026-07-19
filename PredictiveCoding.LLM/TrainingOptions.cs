namespace PredictiveCoding.LLM;

// ─────────────────────────────────────────────────────────────────────────────
// PredictiveLlm  —  THE FINISHED PRODUCT.
//
// Everything the 2026 roadmap built, composed behind one class:
//
//   var llm = PredictiveLlm.Create2026();          // hybrid SSM + looped depth
//   llm.Train(corpus);                             //   + multi-token head
//   var text = llm.Generate("the brain is");       //   + sparse MoE + latent plan
//   var ppl  = llm.Evaluate(heldOutText);          // perplexity yardstick
//   llm.DistillFrom(teacher, prompts);             // knowledge distillation
//   llm.Save(dir);   var again = PredictiveLlm.Load(dir);
//   llm.Report("report.html");                     // activation observatory
//
// Feature map (all opt-in via DecoderLmConfig, all on in Create2026):
//   BlockPattern  hybrid attention + temporal-state blocks   (Phase 5)
//   NumLoops      weight-tied looped depth + early exit      (Phase 2)
//   MtpWeight     t+2 head → self-speculative decoding       (Phase 3)
//   NumExperts    FLOP-sparse mixture of experts             (Phase 6)
//   UsePlan       latent plan, settled on the prompt         (Phase 8)
// Training uses the LR-scheduled, gradient-accumulating Trainer; evaluation is
// the Phase 0 perplexity yardstick; Report is the Phase 0 observatory.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Knobs for <see cref="PredictiveLlm.Train"/>. The defaults suit toy corpora.</summary>
public sealed record TrainingOptions
{
    /// <summary>Window length in tokens (clamped to the model's MaxSeq).</summary>
    public int SeqLen { get; init; } = 32;
    /// <summary>Window stride in tokens.</summary>
    public int Stride { get; init; } = 16;
    public int Epochs { get; init; } = 6;
    /// <summary>Windows per gradient-accumulating optimizer step.</summary>
    public int BatchWindows { get; init; } = 8;
    public float PeakLr { get; init; } = 0.01f;
    public float MinLr { get; init; } = 0.001f;
    public float ClipNorm { get; init; } = 1f;
    /// <summary>Every N-th window is held out for perplexity (never trained on).</summary>
    public int HeldOutEvery { get; init; } = 5;
    /// <summary>Cap on training windows (null = use the whole corpus).</summary>
    public int? MaxWindows { get; init; } = null;
    /// <summary>Called after each epoch with (epoch number 1-based, mean loss of the epoch) — progress reporting.</summary>
    public Action<int, float>? OnEpochEnd { get; init; } = null;

    /// <summary>
    /// Parallel micro-batch workers (roadmap-Pro Phase 3). 1 = serial, bit-identical
    /// to the historical path; 0 = auto (<see cref="Environment.ProcessorCount"/>);
    /// N &gt; 1 shards each batch across N model replicas with a deterministic
    /// gradient merge — reproducible, but not bit-identical to serial (re-associated
    /// float sums). Capped at the batch size.
    /// </summary>
    public int Workers { get; init; } = 1;

    /// <summary>
    /// Flip <see cref="Autograd.Ag.MatMulRoute"/> to <c>Auto</c> for the duration of
    /// training (restored afterwards): big matmuls route to <c>Gemm.Provider</c>
    /// (a CUDA backend when present), small ones stay on the inline SIMD loop.
    /// Off by default — the default provider is the CPU GEMM, so this only pays
    /// with an accelerated provider installed.
    /// </summary>
    public bool AutoGemmRoute { get; init; } = false;

    /// <summary>
    /// Checkpoint every N optimizer steps (roadmap-Pro Phase 5). 0 = off. Requires
    /// <see cref="CheckpointDir"/>. Each checkpoint is a full resumable model dir
    /// (config + tokenizer + weights + optimizer moments).
    /// </summary>
    public int CheckpointEverySteps { get; init; } = 0;

    /// <summary>Where periodic checkpoints are written (null = no checkpointing).</summary>
    public string? CheckpointDir { get; init; } = null;

    /// <summary>
    /// Keep only the checkpoint with the best held-out perplexity so far (true, the
    /// default) rather than overwriting on every interval regardless of quality.
    /// </summary>
    public bool KeepBestByPerplexity { get; init; } = true;

    /// <summary>On-disk weight precision for periodic checkpoints (Float32 = exact).</summary>
    public PredictiveCoding.Model.WeightStoragePrecision CheckpointPrecision { get; init; }
        = PredictiveCoding.Model.WeightStoragePrecision.Float32;

    /// <summary>
    /// Continue from these Adam moments instead of a fresh optimizer (true resume).
    /// Load them with <see cref="PredictiveLlm.LoadWithOptimizer"/>. Null = start clean.
    /// </summary>
    public Training.Adam? ResumeOptimizer { get; init; } = null;
}
