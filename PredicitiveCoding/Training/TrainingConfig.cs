using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Training;

// ─────────────────────────────────────────────────────────────────────────────
// TrainingConfig  —  Full configuration for a Predictive Coding experiment.
//
// Network architecture and learning hyperparameters live in NetworkConfig;
// this record owns only the training-loop settings.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record TrainingConfig
{
    // ── Network ───────────────────────────────────────────────────────────────
    public NetworkConfig Network { get; init; } = new();

    // ── Convenience pass-throughs (read-only, not init-able) ─────────────────
    public int[]          LayerSizes           => Network.LayerSizes;
    public ActivationType Activation           => Network.Activation;
    public float          StateLearningRate    => Network.StateLearningRate;
    public float          WeightLearningRate   => Network.WeightLearningRate;
    public float          WeightDecay          => Network.WeightDecay;
    public bool           UseDiagonalPrecision => Network.UseDiagonalPrecision;

    // ── Inference ─────────────────────────────────────────────────────────────
    public int InferenceSteps { get; init; } = 20;

    // ── Training loop ─────────────────────────────────────────────────────────
    public int Epochs    { get; init; } = 50;
    public int BatchSize { get; init; } = 32;

    // ── Reproducibility ───────────────────────────────────────────────────────
    public int Seed { get; init; } = 42;

    // ── Logging ───────────────────────────────────────────────────────────────
    public string? CsvOutputPath   { get; init; }
    public int     LogEveryNEpochs { get; init; } = 1;

    // ── Validation ────────────────────────────────────────────────────────────
    public void Validate()
    {
        Network.Validate();
        if (InferenceSteps <= 0)
            throw new InvalidOperationException("InferenceSteps must be > 0.");
        if (Epochs <= 0)
            throw new InvalidOperationException("Epochs must be > 0.");
        if (BatchSize <= 0)
            throw new InvalidOperationException("BatchSize must be > 0.");
    }

    public void Print()
    {
        Console.WriteLine("─── TrainingConfig ───────────────────────────");
        Console.WriteLine($"  LayerSizes     : [{string.Join(", ", LayerSizes)}]");
        Console.WriteLine($"  Activation     : {Activation}");
        Console.WriteLine($"  η_state        : {StateLearningRate}");
        Console.WriteLine($"  η_weight       : {WeightLearningRate}");
        Console.WriteLine($"  WeightDecay    : {WeightDecay}");
        Console.WriteLine($"  DiagPrecision  : {UseDiagonalPrecision}");
        Console.WriteLine($"  InferenceSteps : {InferenceSteps}");
        Console.WriteLine($"  Epochs         : {Epochs}");
        Console.WriteLine($"  BatchSize      : {BatchSize}");
        Console.WriteLine($"  Seed           : {Seed}");
        if (CsvOutputPath != null)
            Console.WriteLine($"  CsvOutput      : {CsvOutputPath}");
        Console.WriteLine("──────────────────────────────────────────────");
    }
}
