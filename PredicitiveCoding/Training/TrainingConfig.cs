using PredictiveCoding.Core;

namespace PredictiveCoding.Training;

// ─────────────────────────────────────────────────────────────────────────────
// TrainingConfig
//
// Configurazione completa per un esperimento di Predictive Coding.
// Raccoglie tutti gli iperparametri in un'unica struttura immutabile
// per facilitare la riproducibilità dei risultati.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record TrainingConfig
{
    // ── Architettura ─────────────────────────────────────────────────────────
    /// <summary>
    /// Dimensioni dei layer: [inputSize, h1, h2, ..., numClasses].
    /// Esempio: [20, 64, 32, 4] → input=20, 2 hidden, 4 classi.
    /// </summary>
    public int[] LayerSizes { get; init; } = [20, 64, 32, 4];

    /// <summary>Non-linearità del modello generativo f(W μ + b).</summary>
    public ActivationType Activation { get; init; } = ActivationType.Tanh;

    // ── Iperparametri di inference (dinamica degli stati) ─────────────────────
    /// <summary>
    /// Learning rate per l'aggiornamento degli stati latenti μ_l.
    /// Controlla la velocità di convergenza dell'inference.
    /// Tipicamente η_state >> η_weight.
    /// </summary>
    public float StateLearningRate { get; init; } = 0.1f;

    /// <summary>Numero di step di inference per ogni esempio.</summary>
    public int InferenceSteps { get; init; } = 20;

    // ── Iperparametri di learning (aggiornamento pesi) ────────────────────────
    /// <summary>
    /// Learning rate per l'aggiornamento dei pesi generativi W_l.
    /// Regola la plasticità sinaptica Hebbiana.
    /// </summary>
    public float WeightLearningRate { get; init; } = 0.001f;

    // ── Training loop ─────────────────────────────────────────────────────────
    public int Epochs    { get; init; } = 50;
    public int BatchSize { get; init; } = 32;

    // ── Riproducibilità ───────────────────────────────────────────────────────
    public int Seed { get; init; } = 42;

    // ── Logging ───────────────────────────────────────────────────────────────
    public string? CsvOutputPath { get; init; }  // null = nessun CSV
    public int     LogEveryNEpochs { get; init; } = 1;

    // ── Validazione ───────────────────────────────────────────────────────────
    public void Validate()
    {
        if (LayerSizes.Length < 2)
            throw new InvalidOperationException("LayerSizes deve avere almeno 2 elementi.");
        if (StateLearningRate <= 0)
            throw new InvalidOperationException("StateLearningRate deve essere > 0.");
        if (WeightLearningRate <= 0)
            throw new InvalidOperationException("WeightLearningRate deve essere > 0.");
        if (InferenceSteps <= 0)
            throw new InvalidOperationException("InferenceSteps deve essere > 0.");
        if (Epochs <= 0)
            throw new InvalidOperationException("Epochs deve essere > 0.");
        if (BatchSize <= 0)
            throw new InvalidOperationException("BatchSize deve essere > 0.");
    }

    public void Print()
    {
        Console.WriteLine("─── TrainingConfig ───────────────────────────");
        Console.WriteLine($"  LayerSizes     : [{string.Join(", ", LayerSizes)}]");
        Console.WriteLine($"  Activation     : {Activation}");
        Console.WriteLine($"  η_state        : {StateLearningRate}");
        Console.WriteLine($"  η_weight       : {WeightLearningRate}");
        Console.WriteLine($"  InferenceSteps : {InferenceSteps}");
        Console.WriteLine($"  Epochs         : {Epochs}");
        Console.WriteLine($"  BatchSize      : {BatchSize}");
        Console.WriteLine($"  Seed           : {Seed}");
        if (CsvOutputPath != null)
            Console.WriteLine($"  CsvOutput      : {CsvOutputPath}");
        Console.WriteLine("──────────────────────────────────────────────");
    }
}
