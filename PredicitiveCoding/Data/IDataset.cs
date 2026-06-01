using PredictiveCoding.Training;

namespace PredictiveCoding.Data;

// ─────────────────────────────────────────────────────────────────────────────
// IDataset
//
// Interfaccia comune per tutti i dataset usati nel framework PC.
// Astrae la sorgente dati, permettendo di sostituire SyntheticDataset
// con MnistDataset o altri dataset senza modificare il trainer.
// ─────────────────────────────────────────────────────────────────────────────

public interface IDataset
{
    /// <summary>Dimensione del vettore di input (es. 784 per MNIST 28×28).</summary>
    int InputSize { get; }

    /// <summary>Numero di classi (dimensione del layer di output).</summary>
    int LabelCount { get; }

    /// <summary>Numero totale di esempi nel dataset.</summary>
    int TotalSamples { get; }

    /// <summary>Nome descrittivo del dataset (per logging).</summary>
    string Name { get; }

    /// <summary>
    /// Itera i mini-batch del dataset.
    /// Il dataset può scegliere di shufflare gli esempi.
    /// </summary>
    IEnumerable<Batch> GetBatches(int batchSize, Random? rng = null);
}
