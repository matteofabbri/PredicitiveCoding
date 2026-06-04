namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// LrSchedule  —  learning-rate schedules.
//
// Linear warmup followed by cosine decay to a floor — the standard LLM schedule.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record LrSchedule
{
    public float PeakLr      { get; init; } = 3e-4f;
    public float MinLr       { get; init; } = 0f;
    public int   WarmupSteps { get; init; } = 100;
    public int   TotalSteps  { get; init; } = 1000;

    /// <summary>Learning rate at training <paramref name="step"/> (0-based).</summary>
    public float At(int step)
    {
        if (step < WarmupSteps)
            return PeakLr * (step + 1) / Math.Max(1, WarmupSteps);
        if (step >= TotalSteps)
            return MinLr;

        float progress = (step - WarmupSteps) / (float)Math.Max(1, TotalSteps - WarmupSteps);
        return MinLr + 0.5f * (PeakLr - MinLr) * (1f + MathF.Cos(MathF.PI * progress));
    }
}
