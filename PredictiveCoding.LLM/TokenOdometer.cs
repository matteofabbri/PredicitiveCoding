namespace PredictiveCoding.LLM;

// ─────────────────────────────────────────────────────────────────────────────
// TokenOdometer  —  the model's lifetime token counter.
//
// LOGICAL tokens, by activity: how much text the model has trained on, scored,
// absorbed online, read as prompts and produced. This is the "how much has it
// seen/said" number, persisted with the model (odometer.json), so it keeps
// counting across sessions like a real odometer.
//
// COMPUTE tokens live elsewhere: DecoderLM.ForwardTokens (every token position
// fed to a trainable forward — a full-reforward generation pays O(n²) there)
// and GptModel.ForwardTokens (one per incremental Step on the served stack).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TokenOdometer
{
    /// <summary>Tokens consumed by <see cref="PredictiveLlm.Train"/> (windows × epochs).</summary>
    public long Trained { get; internal set; }

    /// <summary>Tokens scored by evaluation (Evaluate, bits/byte, Learn's before/after probes).</summary>
    public long Evaluated { get; internal set; }

    /// <summary>Tokens absorbed by online <see cref="PredictiveLlm.Learn"/> (windows × steps).</summary>
    public long Absorbed { get; internal set; }

    /// <summary>Prompt tokens read by generation (including chat turns).</summary>
    public long Prompted { get; internal set; }

    /// <summary>Tokens the model emitted (counted as emitted, before any stop-sequence trim).</summary>
    public long Generated { get; internal set; }

    /// <summary>Everything, summed.</summary>
    public long Total => Trained + Evaluated + Absorbed + Prompted + Generated;

    public void Reset()
    {
        Trained = Evaluated = Absorbed = Prompted = Generated = 0;
    }

    public override string ToString() =>
        $"tokens: {Total:N0} total | trained {Trained:N0} · evaluated {Evaluated:N0} · " +
        $"absorbed {Absorbed:N0} · prompted {Prompted:N0} · generated {Generated:N0}";

    // Snapshot record for JSON persistence (System.Text.Json-friendly).
    internal sealed record Snapshot(long Trained, long Evaluated, long Absorbed, long Prompted, long Generated);

    internal Snapshot ToSnapshot() => new(Trained, Evaluated, Absorbed, Prompted, Generated);

    internal void LoadSnapshot(Snapshot s)
    {
        Trained   = s.Trained;
        Evaluated = s.Evaluated;
        Absorbed  = s.Absorbed;
        Prompted  = s.Prompted;
        Generated = s.Generated;
    }
}
