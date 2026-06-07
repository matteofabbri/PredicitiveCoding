using PredictiveCoding.OmniMind.Engine;

namespace PredictiveCoding.OmniMind.Web.Models;

/// <summary>The phase a single <see cref="IgniteUpdate"/> reports.</summary>
public enum IgniteStage
{
    Started,        // the full source list is known; build the dashboard
    SourceRunning,  // a source started probing
    SourceDone,     // a source returned (Outcome populated)
    Fused,          // corpus assembled (stats + top terms)
    Train,          // a training step (Progress populated)
    Log,            // a human-readable training log line (Text)
    Token,          // streamed generated text so far (Text)
    Completed,      // the run finished and was saved (Run)
    Failed,         // the run could not complete (Text = reason)
}

/// <summary>
/// One streamed update from <c>MindService.RunAsync</c> to the UI. A single record
/// with a <see cref="Stage"/> tag and only the fields relevant to that stage set.
/// </summary>
public sealed record IgniteUpdate(IgniteStage Stage)
{
    // Started
    public IReadOnlyList<SourceInfo>? Sources { get; init; }

    // SourceRunning / SourceDone
    public int SourceIndex { get; init; }
    public SourceOutcome? Outcome { get; init; }
    public int DoneSources { get; init; }
    public int LiveSources { get; init; }

    // Fused
    public int CorpusDocs { get; init; }
    public int CorpusChars { get; init; }
    public IReadOnlyList<string>? TopTerms { get; init; }

    // Train
    public MindTrainer.Progress? Progress { get; init; }

    // Log / Token / Failed
    public string? Text { get; init; }

    // Completed
    public RunRecord? Run { get; init; }
}

/// <summary>A source's identity, sent up-front so the dashboard can render every tile.</summary>
public sealed record SourceInfo(string Name, string Category);

/// <summary>Lifecycle of one source tile in the live dashboard.</summary>
public enum LiveSourceStatus { Queued, Running, Done }

/// <summary>Mutable view-model for one provider tile (the web counterpart of the old ProbeTile).</summary>
public sealed class LiveSource
{
    public required string Name { get; init; }
    public required string Category { get; init; }
    public LiveSourceStatus Status { get; set; } = LiveSourceStatus.Queued;
    public SourceOutcome? Outcome { get; set; }
}
