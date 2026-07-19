namespace PredictiveCoding.LLM;

/// <summary>What a training run did, in numbers.</summary>
public sealed record TrainingReport(
    double PerplexityBefore, double PerplexityAfter,
    float FirstLoss, float LastLoss,
    int OptimizerSteps, int TrainWindows, int EvalWindows)
{
    public override string ToString() =>
        $"ppl {PerplexityBefore:F1} → {PerplexityAfter:F1} | loss {FirstLoss:F3} → {LastLoss:F3} " +
        $"| {OptimizerSteps} steps over {TrainWindows} windows ({EvalWindows} held out)";
}
