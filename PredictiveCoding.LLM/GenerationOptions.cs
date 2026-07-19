using PredictiveCoding.LLM.Generation;

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

    /// <summary>
    /// Stochastic decoding (Pro roadmap Phase 2): temperature, top-k, top-p,
    /// repetition/presence/frequency penalties, seed. Null (the default) keeps
    /// the deterministic greedy path — self-speculative and bit-identical to
    /// before. When set, decoding samples token by token (no speculation:
    /// drafts can only be verified exactly, which is a greedy notion).
    /// <c>MaxNewTokens</c> inside is ignored — the method parameter rules;
    /// a null <c>Seed</c> defaults to 42, so product sampling is reproducible
    /// unless you ask otherwise.
    /// </summary>
    public SamplingOptions? Sampling { get; init; } = null;

    /// <summary>
    /// Generation stops when any of these strings appears in the decoded text
    /// (the text is cut just before it). The sampling path stops early and
    /// saves the remaining forwards; the greedy path trims after decoding.
    /// </summary>
    public string[]? StopSequences { get; init; } = null;
}
