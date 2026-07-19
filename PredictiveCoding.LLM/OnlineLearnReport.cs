namespace PredictiveCoding.LLM;

/// <summary>What one online-learning step did: how surprising the text was, before and after absorbing it.</summary>
public sealed record OnlineLearnReport(float SurpriseBefore, float SurpriseAfter, int Steps)
{
    public override string ToString() => $"surprise {SurpriseBefore:F3} → {SurpriseAfter:F3} ({Steps} step{(Steps == 1 ? "" : "s")})";
}
