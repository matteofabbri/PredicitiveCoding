namespace PredictiveCoding.Logging;

// ─────────────────────────────────────────────────────────────────────────────
// CsvLogger
//
// Salva le metriche di training su file CSV per analisi offline
// (es. con Python/pandas, Excel, o strumenti di visualizzazione).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CsvLogger : IDisposable
{
    private readonly StreamWriter _writer;
    private bool _disposed;

    private const string Header =
        "epoch,train_energy,train_accuracy,test_energy,test_accuracy,weight_norm,elapsed_s";

    public string FilePath { get; }

    public CsvLogger(string filePath)
    {
        FilePath = filePath;

        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        _writer = new StreamWriter(filePath, append: false) { AutoFlush = true };
        _writer.WriteLine(Header);
    }

    public void WriteEpoch(EpochMetrics m)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _writer.WriteLine(string.Create(
            System.Globalization.CultureInfo.InvariantCulture,
            $"{m.Epoch},{m.TrainEnergy:F6},{m.TrainAccuracy:F6}," +
            $"{m.TestEnergy:F6},{m.TestAccuracy:F6},{m.WeightNorm:F6},{m.ElapsedSeconds:F3}"));
    }

    /// <summary>Scrive tutte le metriche da un MetricLogger già completato.</summary>
    public void WriteAll(MetricLogger logger)
    {
        foreach (var m in logger.History)
            WriteEpoch(m);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _writer.Dispose();
            _disposed = true;
        }
    }
}
