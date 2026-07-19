namespace PredictiveCoding.LLM;

/// <summary>A generation plus its effort counters.</summary>
public sealed record GenerationResult(string Text, int Forwards, int Drafted, int Accepted)
{
    public float AcceptanceRate => Drafted > 0 ? (float)Accepted / Drafted : 0f;
}
