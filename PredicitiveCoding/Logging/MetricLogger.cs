namespace PredictiveCoding.Logging;

// ─────────────────────────────────────────────────────────────────────────────
// MetricLogger
//
// Registra e visualizza le metriche di training/evaluation per epoca.
// Thread-safe per uso in loop sequenziali.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Snapshot di tutte le metriche per un'epoca.</summary>
public record EpochMetrics(
    int   Epoch,
    float TrainEnergy,
    float TrainAccuracy,
    float TestEnergy,
    float TestAccuracy,
    float WeightNorm,
    double ElapsedSeconds);

public sealed class MetricLogger
{
    private readonly List<EpochMetrics> _history = [];
    private readonly bool               _verbose;

    public IReadOnlyList<EpochMetrics> History => _history;

    public MetricLogger(bool verbose = true)
    {
        _verbose = verbose;
        if (verbose) PrintHeader();
    }

    // ── Registrazione ─────────────────────────────────────────────────────────

    public void LogEpoch(EpochMetrics m)
    {
        _history.Add(m);
        if (_verbose) PrintRow(m);
    }

    // ── Accumulo in-epoch ─────────────────────────────────────────────────────

    private float _energyAcc;
    private int   _correct;
    private int   _total;
    private float _wNormAcc;
    private int   _wNormCnt;

    public void ResetEpochAccumulators()
    {
        _energyAcc = 0f;
        _correct   = 0;
        _total     = 0;
        _wNormAcc  = 0f;
        _wNormCnt  = 0;
    }

    public void AddSample(float energy, bool correct, float weightNorm)
    {
        _energyAcc += energy;
        if (correct) _correct++;
        _total++;
        _wNormAcc += weightNorm;
        _wNormCnt++;
    }

    public (float avgEnergy, float accuracy, float avgWeightNorm) GetEpochStats()
    {
        float avgEnergy    = _total > 0 ? _energyAcc / _total : 0f;
        float accuracy     = _total > 0 ? (float)_correct / _total : 0f;
        float avgWNorm     = _wNormCnt > 0 ? _wNormAcc / _wNormCnt : 0f;
        return (avgEnergy, accuracy, avgWNorm);
    }

    // ── Riepilogo finale ──────────────────────────────────────────────────────

    public void PrintSummary()
    {
        if (_history.Count == 0) return;

        Console.WriteLine();
        Console.WriteLine("═══════════════════════ RIEPILOGO TRAINING ══════════════════════");
        Console.WriteLine($"  Epoche totali     : {_history.Count}");
        Console.WriteLine($"  Energia iniziale  : {_history[0].TrainEnergy:F4}");
        Console.WriteLine($"  Energia finale    : {_history[^1].TrainEnergy:F4}");
        string trainAccStr = float.IsNaN(_history[^1].TrainAccuracy)
            ? "n/a (non calcolata)"
            : $"{_history[^1].TrainAccuracy * 100:F1}%";
        Console.WriteLine($"  Accuracy train    : {trainAccStr}");
        Console.WriteLine($"  Accuracy test     : {_history[^1].TestAccuracy * 100:F1}%");
        Console.WriteLine($"  Norma pesi finale : {_history[^1].WeightNorm:F4}");

        // Best epoch per accuracy test
        var best = _history.Where(m => !float.IsNaN(m.TestAccuracy)).MaxBy(m => m.TestAccuracy);
        if (best != null)
            Console.WriteLine($"  Best test acc     : {best.TestAccuracy * 100:F1}% (epoch {best.Epoch})");
        Console.WriteLine("══════════════════════════════════════════════════════════════════");
    }

    // ── Stampa tabella ────────────────────────────────────────────────────────

    private static void PrintHeader()
    {
        Console.WriteLine();
        Console.WriteLine(
            $"{"Epoch",6} | {"TrainE",9} | {"TrainAcc",9} | {"TestE",9} | {"TestAcc",9} | {"WNorm",8} | {"t(s)",6}");
        Console.WriteLine(new string('─', 72));
    }

    private static void PrintRow(EpochMetrics m)
    {
        string trainAcc = float.IsNaN(m.TrainAccuracy) ? "    n/a  " : $"{m.TrainAccuracy * 100,8:F2}%";
        Console.WriteLine(
            $"{m.Epoch,6} | " +
            $"{m.TrainEnergy,9:F4} | " +
            $"{trainAcc} | " +
            $"{m.TestEnergy,9:F4} | " +
            $"{m.TestAccuracy * 100,8:F2}% | " +
            $"{m.WeightNorm,8:F4} | " +
            $"{m.ElapsedSeconds,6:F1}");
    }
}
