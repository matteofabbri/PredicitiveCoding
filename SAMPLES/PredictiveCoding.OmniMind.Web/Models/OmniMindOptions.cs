using PredictiveCoding.OmniMind.Engine;

namespace PredictiveCoding.OmniMind.Web.Models;

/// <summary>
/// Strongly-typed configuration for the OmniMind web app, bound from the
/// <c>"OmniMind"</c> section of appsettings.json.
/// </summary>
public sealed class OmniMindOptions
{
    public const string SectionName = "OmniMind";

    /// <summary>
    /// Root directory of the Sperone run-history database. Defaults to
    /// <c>%USERPROFILE%/.predictivecoding/omnimind</c> (the project's data convention).
    /// </summary>
    public string DataDirectory { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "omnimind");

    /// <summary>Per-source fan-out timeout, in seconds.</summary>
    public int SourceTimeoutSeconds { get; set; } = 25;

    /// <summary>How many from-scratch trainings may run at once (keeps the server responsive).</summary>
    public int MaxConcurrentTrainings { get; set; } = 1;

    public TrainingOptions Training { get; set; } = new();
    public GenerationOptions Generation { get; set; } = new();

    /// <summary>Live, from-scratch learning hyper-parameters (mirror <see cref="MindOptions"/>).</summary>
    public sealed class TrainingOptions
    {
        public int Merges { get; set; } = 700;
        public int Dim { get; set; } = 80;
        public int Blocks { get; set; } = 2;
        public int Heads { get; set; } = 4;
        public int FfDim { get; set; } = 192;
        public int SeqLen { get; set; } = 48;
        public int Steps { get; set; } = 160;
        public int Batch { get; set; } = 8;
        public float PeakLr { get; set; } = 3e-3f;
        public int MaxCorpusChars { get; set; } = 200_000;

        public MindOptions ToMindOptions() =>
            new(Merges, Dim, Blocks, Heads, FfDim, SeqLen, Steps, Batch, PeakLr, MaxCorpusChars);
    }

    /// <summary>Sampling parameters for the freshly-trained model.</summary>
    public sealed class GenerationOptions
    {
        public int MaxNewTokens { get; set; } = 220;
        public float Temperature { get; set; } = 0.9f;
        public int TopK { get; set; } = 40;
    }
}
