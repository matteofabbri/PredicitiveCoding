namespace PredictiveCoding.Brain.Viz;

// ─────────────────────────────────────────────────────────────────────────────
// VizData  —  the data contract between the Thousand Brains demo and the web
// visualization page.
//
// The demo fills this while it runs (benchmark rows, the live episode's
// narrowing hypothesis sets, the pose-invariance contrast, the real-sensor
// image scan); HtmlVisualizer serializes it to JSON and embeds it in a fully
// self-contained HTML page. Every section is optional — the page renders only
// the slots that are populated.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Everything the web visualization can show. Unpopulated sections are skipped.</summary>
public sealed class VizData
{
    public string Title { get; set; } = "Thousand Brains — sensorimotor recognition";
    public List<VizBenchmarkRow> Benchmark { get; set; } = [];
    public VizEpisode? Episode { get; set; }
    public VizPoseInvariance? PoseInvariance { get; set; }
    public VizImageScan? ImageScan { get; set; }
}

/// <summary>One benchmark condition: single-column vs voting accuracy and mean steps.</summary>
public sealed record VizBenchmarkRow(
    string Condition, double SingleAccuracy, double SingleSteps,
    double VotingAccuracy, double VotingSteps);

/// <summary>A live recognition episode: the learned objects and, per step/round,
/// the surviving object-id hypotheses of the single column and the voting population.</summary>
public sealed class VizEpisode
{
    public int TargetId { get; set; }
    public int ObjectCount { get; set; }
    public int ColumnsVoting { get; set; }
    public List<VizObject> Objects { get; set; } = [];

    /// <summary>Single-column hypothesis ids after each sensorimotor step.</summary>
    public List<List<int>> SingleSteps { get; set; } = [];

    /// <summary>Voting-population candidate ids after each round.</summary>
    public List<List<int>> VotingRounds { get; set; } = [];
}

/// <summary>An object's 2-D landmark cloud (FeatureId indexes the shared vocabulary).</summary>
public sealed record VizObject(int Id, List<VizPoint> Points);

/// <summary>One landmark: position + which vocabulary feature sits there.</summary>
public sealed record VizPoint(float X, float Y, int FeatureId);

/// <summary>The pose-invariance contrast: every object rigidly moved, then recognised
/// (or not) by the absolute column vs the displacement recogniser.</summary>
public sealed class VizPoseInvariance
{
    public float Dx { get; set; }
    public float Dy { get; set; }
    public float ThetaDeg { get; set; }
    public List<VizPoseRow> Rows { get; set; } = [];
}

/// <summary>Per-object outcome of the pose-invariance contrast.</summary>
public sealed record VizPoseRow(int Id, bool ColumnCorrect, bool DisplacementCorrect);

/// <summary>A real-sensor scan: the image, the fixation grid and the descriptor
/// (dominant orientation category, share, luminance) at every fixation visited.</summary>
public sealed class VizImageScan
{
    public int Width { get; set; }
    public int Height { get; set; }

    /// <summary>Row-major luminance in [0, 1], length Width × Height.</summary>
    public float[] Pixels { get; set; } = [];

    public int GridCols { get; set; }
    public int GridRows { get; set; }
    public int PatchSize { get; set; }
    public int Stride { get; set; }
    public int OrientationBins { get; set; }

    /// <summary>Fixations in the order the sensor visited them.</summary>
    public List<VizFixation> Fixations { get; set; } = [];
}

/// <summary>One fixation: grid position + the raw patch descriptor
/// (Category == OrientationBins means a flat patch).</summary>
public sealed record VizFixation(int Col, int Row, int Category, float Share, float Luminance);
