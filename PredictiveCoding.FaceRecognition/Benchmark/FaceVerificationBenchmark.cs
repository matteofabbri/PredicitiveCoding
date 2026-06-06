using PredictiveCoding.FaceRecognition.Datasets;

namespace PredictiveCoding.FaceRecognition.Benchmark;

public record VerificationResult(
    double Accuracy,       // Mean accuracy across 10 folds (best per-fold threshold)
    double AccuracyStd,    // Standard deviation across folds
    double AucRoc,         // Overall AUC-ROC
    double BestThreshold,  // Mean tuned cosine threshold across folds
    double Tpr,            // True positive rate at mean threshold
    double Fpr,            // False positive rate at mean threshold
    double MeanSameScore,  // Mean cosine similarity for same-person pairs
    double MeanDiffScore,  // Mean cosine similarity for different-person pairs
    string EmbedderName,
    int    PairsEvaluated,
    int    MissingImages
);

public static class FaceVerificationBenchmark
{
    /// <summary>
    /// Runs the official LFW 10-fold restricted verification protocol.
    /// The cosine threshold is tuned on 9 folds and evaluated on the held-out fold.
    /// </summary>
    public static VerificationResult RunLfw(
        LfwDataset      dataset,
        IFaceEmbedder   embedder,
        Action<string>? onProgress = null)
    {
        // Fit (no-op for fixed descriptors / pretrained ArcFace)
        onProgress?.Invoke($"Fitting {embedder.Name} …");
        var allPaths = dataset.AllImages().Select(p => p.Path).ToList();
        embedder.Fit(allPaths);

        // Pre-embed every unique image referenced by any pair
        var allPairs    = dataset.AllPairs().ToList();
        var uniquePaths = allPairs
            .SelectMany(pair => new[] { pair.Path1, pair.Path2 })
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        onProgress?.Invoke($"Embedding {uniquePaths.Count} unique images ({allPairs.Count} pairs) …");
        var embedded = embedder.EmbedBatch(uniquePaths);

        var cache = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < uniquePaths.Count; i++)
            if (embedded[i] is { } e)
                cache[uniquePaths[i]] = e;

        // Score all pairs
        double[] scores  = new double[allPairs.Count];
        bool[]   labels  = new bool[allPairs.Count];
        int missingImages = 0;
        for (int i = 0; i < allPairs.Count; i++)
        {
            labels[i] = allPairs[i].Same;
            if (cache.TryGetValue(allPairs[i].Path1, out var e1) &&
                cache.TryGetValue(allPairs[i].Path2, out var e2))
                scores[i] = FaceMath.Cosine(e1, e2);
            else
                missingImages++;
        }

        // Compute fold start offsets so we can index into the flat arrays
        int numFolds = dataset.FoldCount;
        int[] foldStart = new int[numFolds + 1];
        for (int f = 0; f < numFolds; f++)
            foldStart[f + 1] = foldStart[f] + dataset.GetFold(f).Count;

        // 10-fold cross-validation: tune threshold on 9 folds, evaluate on 1
        double[] foldAcc       = new double[numFolds];
        double[] foldThreshold = new double[numFolds];

        for (int fold = 0; fold < numFolds; fold++)
        {
            // Collect training scores (all other folds)
            var trainScores = new List<double>(allPairs.Count - dataset.GetFold(fold).Count);
            var trainLabels = new List<bool>(trainScores.Capacity);
            for (int f2 = 0; f2 < numFolds; f2++)
            {
                if (f2 == fold) continue;
                for (int i = foldStart[f2]; i < foldStart[f2 + 1]; i++)
                {
                    trainScores.Add(scores[i]);
                    trainLabels.Add(labels[i]);
                }
            }

            double threshold = BestThreshold(trainScores, trainLabels);
            foldThreshold[fold] = threshold;

            int correct = 0;
            int foldLen = foldStart[fold + 1] - foldStart[fold];
            for (int i = foldStart[fold]; i < foldStart[fold + 1]; i++)
                if ((scores[i] >= threshold) == labels[i])
                    correct++;
            foldAcc[fold] = (double)correct / foldLen;
        }

        double meanAcc       = foldAcc.Average();
        double stdAcc        = StdDev(foldAcc);
        double meanThreshold = foldThreshold.Average();
        double auc           = ComputeAuc(scores, labels);

        int sameCount = labels.Count(l => l);
        int diffCount = labels.Length - sameCount;
        double meanSame = sameCount > 0
            ? Enumerable.Range(0, scores.Length).Where(i => labels[i]).Average(i => scores[i])
            : 0;
        double meanDiff = diffCount > 0
            ? Enumerable.Range(0, scores.Length).Where(i => !labels[i]).Average(i => scores[i])
            : 0;

        var (tpr, fpr) = TprFpr(scores, labels, meanThreshold);

        return new VerificationResult(
            meanAcc, stdAcc, auc, meanThreshold, tpr, fpr,
            meanSame, meanDiff, embedder.Name, allPairs.Count, missingImages);
    }

    // ── Metrics ───────────────────────────────────────────────────────────────────

    private static double BestThreshold(IList<double> scores, IList<bool> labels)
    {
        double best = 0, bestAcc = -1;
        var candidates = scores.Distinct().OrderBy(x => x).ToArray();
        int step = Math.Max(1, candidates.Length / 200);
        foreach (double t in candidates.Where((_, i) => i % step == 0))
        {
            int correct = 0;
            for (int i = 0; i < scores.Count; i++)
                if ((scores[i] >= t) == labels[i])
                    correct++;
            double acc = (double)correct / scores.Count;
            if (acc > bestAcc) { bestAcc = acc; best = t; }
        }
        return best;
    }

    private static double ComputeAuc(double[] scores, bool[] labels)
    {
        int pos = labels.Count(l => l);
        int neg = labels.Length - pos;
        if (pos == 0 || neg == 0) return 0.5;

        var sorted = scores.Zip(labels).OrderByDescending(p => p.First).ToArray();
        double auc = 0;
        int tp = 0, fp = 0, prevTp = 0, prevFp = 0;
        foreach (var (_, lbl) in sorted)
        {
            if (lbl) tp++; else fp++;
            if (fp != prevFp)
            {
                auc += (double)(fp - prevFp) / neg * (tp + prevTp) / 2.0 / pos;
                prevTp = tp; prevFp = fp;
            }
        }
        return auc;
    }

    private static (double Tpr, double Fpr) TprFpr(double[] scores, bool[] labels, double t)
    {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            bool pred = scores[i] >= t;
            if (pred && labels[i])  tp++;
            else if (pred)          fp++;
            else if (labels[i])     fn++;
            else                    tn++;
        }
        double tpr = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
        double fpr = fp + tn > 0 ? (double)fp / (fp + tn) : 0;
        return (tpr, fpr);
    }

    private static double StdDev(double[] v)
    {
        double mean = v.Average();
        return Math.Sqrt(v.Average(x => (x - mean) * (x - mean)));
    }
}
