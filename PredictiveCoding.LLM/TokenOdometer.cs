using System.Numerics;
using System.Text.Json.Serialization;
using PredictiveCoding.LLM.Infrastructure;

namespace PredictiveCoding.LLM;

// ─────────────────────────────────────────────────────────────────────────────
// TokenOdometer  —  the model's lifetime token counter.
//
// LOGICAL tokens, by activity: how much text the model has trained on, scored,
// absorbed online, read as prompts and produced. This is the "how much has it
// seen/said" number, persisted with the model (odometer.json), so it keeps
// counting across sessions like a real odometer.
//
// Counters are BigInteger: a lifelong learner's odometer must never roll over,
// no matter how long it lives or how fast it reads. JSON round-trips through
// BigIntegerJsonConverter (plain JSON numbers, unlimited digits), and files
// written back when these were long parse unchanged.
//
// COMPUTE tokens live elsewhere: DecoderLM.ForwardTokens (every token position
// fed to a trainable forward — a full-reforward generation pays O(n²) there)
// and GptModel.ForwardTokens (one per incremental Step on the served stack).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TokenOdometer
{
    /// <summary>Tokens consumed by <see cref="PredictiveLlm.Train"/> (windows × epochs).</summary>
    public BigInteger Trained { get; internal set; }

    /// <summary>Tokens scored by evaluation (Evaluate, bits/byte, Learn's before/after probes).</summary>
    public BigInteger Evaluated { get; internal set; }

    /// <summary>Tokens absorbed by online <see cref="PredictiveLlm.Learn"/> (windows × steps).</summary>
    public BigInteger Absorbed { get; internal set; }

    /// <summary>Prompt tokens read by generation (including chat turns).</summary>
    public BigInteger Prompted { get; internal set; }

    /// <summary>Tokens the model emitted (counted as emitted, before any stop-sequence trim).</summary>
    public BigInteger Generated { get; internal set; }

    /// <summary>Everything, summed.</summary>
    public BigInteger Total => Trained + Evaluated + Absorbed + Prompted + Generated;

    public void Reset()
    {
        Trained = Evaluated = Absorbed = Prompted = Generated = BigInteger.Zero;
    }

    /// <summary>
    /// Folds in tokens produced by generation that ran on a SERVED COPY of the
    /// model rather than on this instance — the reader/writer split a host uses
    /// to serve requests without blocking on the learner. The copy is transient
    /// (discarded at the next snapshot), so its own odometer would evaporate;
    /// the host counts the tokens and hands them over here, where they belong,
    /// before the checkpoint is written.
    /// </summary>
    /// <remarks>
    /// Call it from ONE thread at a time (the host's writer path): BigInteger
    /// is immutable but <c>+=</c> is a read-modify-write, so concurrent callers
    /// would drop increments. The host is expected to accumulate concurrently
    /// with its own atomics and drain into this in a single call.
    /// </remarks>
    public void AddServed(BigInteger prompted, BigInteger generated)
    {
        if (prompted.Sign < 0 || generated.Sign < 0)
            throw new ArgumentOutOfRangeException(nameof(prompted), "An odometer only goes forward.");
        Prompted  += prompted;
        Generated += generated;
    }

    public override string ToString() =>
        $"tokens: {Total:N0} total | trained {Trained:N0} · evaluated {Evaluated:N0} · " +
        $"absorbed {Absorbed:N0} · prompted {Prompted:N0} · generated {Generated:N0}";

    // Snapshot record for JSON persistence. The converter keeps the on-disk
    // shape identical to the long era (plain numbers), just without the ceiling.
    internal sealed record Snapshot(
        [property: JsonConverter(typeof(BigIntegerJsonConverter))] BigInteger Trained,
        [property: JsonConverter(typeof(BigIntegerJsonConverter))] BigInteger Evaluated,
        [property: JsonConverter(typeof(BigIntegerJsonConverter))] BigInteger Absorbed,
        [property: JsonConverter(typeof(BigIntegerJsonConverter))] BigInteger Prompted,
        [property: JsonConverter(typeof(BigIntegerJsonConverter))] BigInteger Generated);

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
