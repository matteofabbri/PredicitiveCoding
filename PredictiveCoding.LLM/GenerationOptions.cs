namespace PredictiveCoding.LLM;

/// <summary>Knobs for <see cref="PredictiveLlm.GenerateDetailed"/>.</summary>
public sealed record GenerationOptions
{
    /// <summary>Settle the latent plan on the prompt before decoding (0 = off; needs UsePlan).</summary>
    public int SettlePlanSteps { get; init; } = 0;
    public float SettlePlanLr { get; init; } = 0.01f;
    /// <summary>Stop at the tokenizer's EOS token.</summary>
    public bool StopAtEos { get; init; } = true;
    /// <summary>
    /// Learn from the prompt before generating (one <see cref="PredictiveLlm.Learn"/>
    /// step): the model absorbs the text it is being used on — training while using.
    /// </summary>
    public bool LearnFromPrompt { get; init; } = false;
}
