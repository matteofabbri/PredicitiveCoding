namespace PredictiveCoding.LLM;

/// <summary>What a training run did, in numbers.</summary>
public sealed record TrainingReport(
    double PerplexityBefore, double PerplexityAfter,
    float FirstLoss, float LastLoss,
    int OptimizerSteps, int TrainWindows, int EvalWindows)
{
    /// <summary>Best held-out perplexity seen at a checkpoint interval (NaN if none taken).</summary>
    public double BestPerplexity { get; init; } = double.NaN;

    /// <summary>Directory of the best checkpoint written during the run (null if none).</summary>
    public string? BestCheckpointPath { get; init; } = null;

    public override string ToString() =>
        $"ppl {PerplexityBefore:F1} → {PerplexityAfter:F1} | loss {FirstLoss:F3} → {LastLoss:F3} " +
        $"| {OptimizerSteps} steps over {TrainWindows} windows ({EvalWindows} held out)" +
        (BestCheckpointPath is null ? "" : $" | best ppl {BestPerplexity:F1} → {BestCheckpointPath}");
}
