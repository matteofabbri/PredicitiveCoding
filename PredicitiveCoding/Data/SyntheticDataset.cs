using PredictiveCoding.Core;
using PredictiveCoding.Training;

namespace PredictiveCoding.Data;

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticDataset
//
// Dataset sintetico di classificazione con blob gaussiani.
//
// Struttura del problema:
//   • C classi, ognuna con un centro (prototipo) casuale in R^D
//   • Ogni esempio = centro_c + N(0, noise²)
//   • Etichette: one-hot encoding
//
// IMPORTANTE per la validità dell'esperimento:
//   Train e Test condividono gli stessi prototipi di classe — solo il rumore
//   di campionamento differisce. Questo rispecchia una vera generalizzazione.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class SyntheticDataset : IDataset
{
    // ── IDataset ──────────────────────────────────────────────────────────────
    public int    InputSize    { get; }
    public int    LabelCount   { get; }
    public int    TotalSamples => _inputs.Length;
    public string Name         { get; }

    // ── Dati interni ──────────────────────────────────────────────────────────
    private readonly DenseTensor<float>[] _inputs;
    private readonly DenseTensor<float>[] _labels;

    // Costruttore interno: accetta array già generati
    private SyntheticDataset(
        DenseTensor<float>[] inputs,
        DenseTensor<float>[] labels,
        int inputDim, int numClasses, string name)
    {
        InputSize  = inputDim;
        LabelCount = numClasses;
        Name       = name;
        _inputs    = inputs;
        _labels    = labels;
    }

    // ── IDataset.GetBatches ───────────────────────────────────────────────────

    public IEnumerable<Batch> GetBatches(int batchSize, Random? rng = null)
    {
        int[] order = Enumerable.Range(0, TotalSamples).ToArray();
        if (rng != null) RandomUtils.Shuffle(order, rng);

        int i = 0;
        while (i < TotalSamples)
        {
            int end  = Math.Min(i + batchSize, TotalSamples);
            int size = end - i;

            var inputs = new DenseTensor<float>[size];
            var labels = new DenseTensor<float>[size];

            for (int b = 0; b < size; b++)
            {
                inputs[b] = _inputs[order[i + b]];
                labels[b] = _labels[order[i + b]];
            }

            yield return new Batch(inputs, labels);
            i += size;
        }
    }

    // ── Factory: Split ────────────────────────────────────────────────────────

    /// <summary>
    /// Genera train e test con i MEDESIMI prototipi di classe.
    ///
    /// Procedura:
    ///   1. Genera C centri in R^D con seed fisso
    ///   2. Campiona totalPerClass esempi per classe con rumore gaussiano
    ///   3. Divide in train (1-testFraction) e test (testFraction)
    ///
    /// Condividere i prototipi garantisce che train e test misurino la vera
    /// generalizzazione del modello (non due problemi diversi).
    /// </summary>
    public static (SyntheticDataset train, SyntheticDataset test) Split(
        int   numClasses      = 4,
        int   inputDim        = 20,
        int   totalPerClass   = 200,
        float testFraction    = 0.2f,
        float noise           = 0.4f,
        int   seed            = 42,
        bool  verbose         = true)
    {
        var rng = new Random(seed);

        // 1. Genera prototipi condivisi
        var centers = new DenseTensor<float>[numClasses];
        for (int c = 0; c < numClasses; c++)
        {
            centers[c] = new DenseTensor<float>(inputDim);
            for (int d = 0; d < inputDim; d++)
                centers[c].Buffer[d] = RandomUtils.SampleNormal(0f, 2f, rng);
        }

        // 2. Campiona tutti gli esempi
        int testPerClass  = Math.Max(1, (int)(totalPerClass * testFraction));
        int trainPerClass = totalPerClass - testPerClass;

        var trainInputs = new List<DenseTensor<float>>();
        var trainLabels = new List<DenseTensor<float>>();
        var testInputs  = new List<DenseTensor<float>>();
        var testLabels  = new List<DenseTensor<float>>();

        for (int c = 0; c < numClasses; c++)
        {
            var oneHot = MakeOneHot(c, numClasses);

            for (int s = 0; s < totalPerClass; s++)
            {
                var x = new DenseTensor<float>(inputDim);
                for (int d = 0; d < inputDim; d++)
                    x.Buffer[d] = centers[c].Buffer[d] +
                                  RandomUtils.SampleNormal(0f, noise, rng);

                if (s < trainPerClass)
                {
                    trainInputs.Add(x);
                    trainLabels.Add(oneHot);
                }
                else
                {
                    testInputs.Add(x);
                    testLabels.Add(oneHot);
                }
            }
        }

        var train = new SyntheticDataset(
            [.. trainInputs], [.. trainLabels], inputDim, numClasses,
            $"Train ({numClasses}cl × {trainPerClass}={trainInputs.Count})");

        var test = new SyntheticDataset(
            [.. testInputs], [.. testLabels], inputDim, numClasses,
            $"Test ({numClasses}cl × {testPerClass}={testInputs.Count})");

        if (verbose)
        {
            Console.WriteLine($"[SyntheticDataset] {numClasses} classi, dim={inputDim}, noise={noise:F2}");
            Console.WriteLine($"  Train: {train.TotalSamples}  |  Test: {test.TotalSamples}");
        }

        return (train, test);
    }

    // ── Helper ────────────────────────────────────────────────────────────────

    private static DenseTensor<float> MakeOneHot(int c, int n)
    {
        var v = DenseTensor<float>.Zeros(n);
        v.Buffer[c] = 1f;
        return v;
    }
}
