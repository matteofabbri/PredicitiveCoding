using PredictiveCoding.Core;

namespace PredictiveCoding.Training;

// ─────────────────────────────────────────────────────────────────────────────
// Batch
//
// Contenitore per un mini-batch di esempi di training/evaluation.
// Nel framework PC, ogni esempio viene elaborato individualmente (online);
// il batch aggrega gli esempi per il logging e le statistiche per epoca.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Batch
{
    /// <summary>Input sensoriali: ogni elemento è un vettore [InputSize].</summary>
    public DenseTensor<float>[] Inputs { get; }

    /// <summary>
    /// Etichette one-hot o continue: ogni elemento è un vettore [LabelCount].
    /// Per classificazione: one-hot encoding.
    /// </summary>
    public DenseTensor<float>[] Labels { get; }

    /// <summary>Numero di esempi nel batch.</summary>
    public int Size => Inputs.Length;

    public Batch(DenseTensor<float>[] inputs, DenseTensor<float>[] labels)
    {
        if (inputs.Length != labels.Length)
            throw new ArgumentException(
                $"Inputs ({inputs.Length}) e Labels ({labels.Length}) devono avere la stessa lunghezza.");

        Inputs = inputs;
        Labels = labels;
    }

    /// <summary>Crea un batch da liste esistenti.</summary>
    public static Batch FromLists(List<DenseTensor<float>> inputs, List<DenseTensor<float>> labels) =>
        new([.. inputs], [.. labels]);
}
