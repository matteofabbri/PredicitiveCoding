using System.Text.Json.Serialization;

namespace PredictiveCoding.OmniMind.Web.Models;

/// <summary>
/// A persisted OmniMind run: the topic, which live sources answered, the fused
/// corpus stats, the from-scratch learning curve, the model it grew, and the text
/// it generated. Stored as a JSON document in the Sperone <c>runs</c> collection.
/// Plain get/set POCO so System.Text.Json round-trips it cleanly.
/// </summary>
public sealed class RunRecord
{
    /// <summary>Stable id (also the Sperone ObjectId key).</summary>
    public string Id { get; set; } = Guid.NewGuid().ToString("N");

    public string Topic { get; set; } = "";

    /// <summary>Creation time as Unix milliseconds (indexed for range/sort).</summary>
    public long CreatedAtUnixMs { get; set; }

    public long DurationMs { get; set; }

    public int CorpusDocs { get; set; }
    public int CorpusChars { get; set; }
    public int LiveSources { get; set; }
    public int TotalSources { get; set; }

    public List<SourceOutcome> Sources { get; set; } = [];
    public List<string> TopTerms { get; set; } = [];

    /// <summary>Per-step loss (downsampled for storage), enough to redraw the curve.</summary>
    public List<float> LossHistory { get; set; } = [];

    public float FinalLoss { get; set; }
    public float FinalPerplexity { get; set; }
    public int Vocab { get; set; }
    public long Params { get; set; }

    public string GeneratedText { get; set; } = "";

    [JsonIgnore]
    public DateTimeOffset CreatedAt => DateTimeOffset.FromUnixTimeMilliseconds(CreatedAtUnixMs);
}

/// <summary>The outcome of probing one external source, as persisted in a <see cref="RunRecord"/>.</summary>
public sealed class SourceOutcome
{
    public string Name { get; set; } = "";
    public string Category { get; set; } = "";

    /// <summary>The <c>ProbeState</c> name: Ok, Empty, Skipped, Failed.</summary>
    public string State { get; set; } = "";

    public int ItemCount { get; set; }
    public string Detail { get; set; } = "";
    public long ElapsedMs { get; set; }
}
