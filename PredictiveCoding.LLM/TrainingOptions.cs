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
}
