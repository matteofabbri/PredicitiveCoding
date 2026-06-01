using System.Diagnostics;
using PredictiveCoding.Core;
using PredictiveCoding.Data;
using PredictiveCoding.Logging;
using PredictiveCoding.Model;

namespace PredictiveCoding.Training;

// ─────────────────────────────────────────────────────────────────────────────
// PCTrainer
//
// Training loop completo per il framework di Predictive Coding.
//
// Per ogni epoca:
//   1. TrainEpoch:
//      Per ogni batch → per ogni esempio:
//        a. Inference (più step): fissa input e label, aggiorna stati nascosti
//        b. Learning: aggiorna pesi W_l con la regola Hebbiana pesata dall'errore
//        c. Accumula energia residua (proxy di qualità dell'apprendimento)
//
//   2. Evaluate:
//      Per ogni esempio di test:
//        a. Classificazione a energia minima:
//           c* = argmin_c F(x, one-hot(c))   [approccio PC principiato]
//        b. Calcola accuracy e energia media
//
// Nota sulla classificazione a energia:
//   Non si usa argmax(μ_{L-1}) — troppo dipendente dall'inizializzazione.
//   Si eseguono invece numClasses inferenze, ognuna con una label candidata,
//   e si sceglie il candidato che produce l'energia minima.
//   Questo riflette il principio fondamentale: "quale spiegazione minimizza
//   l'errore di predizione?" (Friston 2005, Free Energy Principle).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class PCTrainer
{
    // ── Training epoch ────────────────────────────────────────────────────────

    /// <summary>
    /// Esegue un'epoca di training su tutti i batch.
    /// Restituisce energia media e norma pesi media.
    /// (L'accuracy non è calcolata qui per efficienza: la fa Evaluate.)
    /// </summary>
    public static (float avgEnergy, float weightNorm) TrainEpoch(
        Network      net,
        IDataset       trainSet,
        TrainingConfig cfg,
        Random         rng)
    {
        float energySum  = 0f;
        float wNormSum   = 0f;
        int   total      = 0;

        foreach (var batch in trainSet.GetBatches(cfg.BatchSize, rng))
        {
            for (int b = 0; b < batch.Size; b++)
            {
                var input = batch.Inputs[b];
                var label = batch.Labels[b];

                // ── Learning: inference + aggiornamento pesi ──────────────────
                //
                // Durante il training:
                //   • Layer 0 (sensoriale) clamped a x
                //   • Layer L-1 (classe) clamped a y = one-hot(c)
                //   • Layer 1..L-2 (nascosti) liberi di minimizzare F
                //
                // Dopo convergenza dell'inference:
                //   ΔW_l = η_W · (f'(pre_l) ⊙ ε_{l-1}) ⊗ μ_l^T
                //
                net.Learn(input, label, cfg.InferenceSteps);

                // Energia dopo l'inference di training (decresce nel tempo)
                energySum += net.ComputeEnergy();
                wNormSum  += net.AverageWeightNorm();
                total++;
            }
        }

        return (
            avgEnergy  : total > 0 ? energySum / total : 0f,
            weightNorm : total > 0 ? wNormSum  / total : 0f
        );
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// <summary>
    /// Valuta la rete sul test set tramite classificazione a energia minima.
    ///
    /// Per ogni campione x:
    ///   c* = argmin_{c=0..C-1}  F(x, one-hot(c))
    ///
    /// Equivale a chiedere: "quale classe produce la predizione più coerente
    /// con l'input, secondo il modello generativo appreso?"
    /// </summary>
    public static (float avgEnergy, float accuracy) Evaluate(
        Network      net,
        IDataset       testSet,
        TrainingConfig cfg)
    {
        float energySum = 0f;
        int   correct   = 0;
        int   total     = 0;

        foreach (var batch in testSet.GetBatches(cfg.BatchSize, rng: null))
        {
            for (int b = 0; b < batch.Size; b++)
            {
                var input = batch.Inputs[b];
                var label = batch.Labels[b];

                // Classificazione a energia minima (approccio PC principiato)
                int predicted   = net.Predict(input, cfg.InferenceSteps);
                int groundTruth = TensorUtils.Argmax(label);

                if (predicted == groundTruth) correct++;
                energySum += net.ComputeEnergy();
                total++;
            }
        }

        return (
            avgEnergy : total > 0 ? energySum / total : 0f,
            accuracy  : total > 0 ? (float)correct / total : 0f
        );
    }

    // ── Loop di training completo ─────────────────────────────────────────────

    public static void Train(
        Network      net,
        IDataset       trainSet,
        IDataset       testSet,
        TrainingConfig cfg,
        MetricLogger   logger,
        CsvLogger?     csvLogger = null)
    {
        cfg.Validate();
        net.PrintArchitecture();

        var rng       = new Random(cfg.Seed);
        var stopwatch = Stopwatch.StartNew();

        Console.WriteLine($"\n  Training: {trainSet.Name}");
        Console.WriteLine($"  Test:     {testSet.Name}");
        Console.WriteLine($"  (train accuracy = energia; test accuracy = energy-based prediction)");

        for (int epoch = 1; epoch <= cfg.Epochs; epoch++)
        {
            stopwatch.Restart();

            var (trainE, wNorm)      = TrainEpoch(net, trainSet, cfg, rng);
            var (testE,  testAcc)    = Evaluate(net, testSet, cfg);

            double elapsed = stopwatch.Elapsed.TotalSeconds;

            if (epoch == 1 || epoch % cfg.LogEveryNEpochs == 0 || epoch == cfg.Epochs)
            {
                var m = new EpochMetrics(
                    Epoch:          epoch,
                    TrainEnergy:    trainE,
                    TrainAccuracy:  float.NaN,   // non calcolata per efficienza
                    TestEnergy:     testE,
                    TestAccuracy:   testAcc,
                    WeightNorm:     wNorm,
                    ElapsedSeconds: elapsed);

                logger.LogEpoch(m);
                csvLogger?.WriteEpoch(m);
            }
        }

        logger.PrintSummary();
    }
}
