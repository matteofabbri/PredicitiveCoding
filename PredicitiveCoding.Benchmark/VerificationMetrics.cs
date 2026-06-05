namespace PredictiveCoding.Benchmark;

/// <summary>
/// 1:1 verification scoring (LFW / RFW / IJB-C style): pair similarity scores + same/different labels →
/// fold-cross-validated accuracy, ROC AUC and TAR@FAR.
/// </summary>
public static class VerificationMetrics
{
    /// <summary>A scored pair. <paramref name="Score"/> is a similarity (higher = more alike).</summary>
    public readonly record struct Pair(double Score, bool Same, int Fold);

    public sealed record Result(double Accuracy, double AccuracyStd, double Auc, double TarAtFar1, double TarAtFar01, int Pairs)
    {
        public override string ToString()
            => $"acc {Accuracy:P2} ± {AccuracyStd:P2} · AUC {Auc:F3} · TAR@FAR=1% {TarAtFar1:P1} · TAR@FAR=0.1% {TarAtFar01:P1} · {Pairs} pairs";
    }

    public static Result Evaluate(IReadOnlyList<Pair> pairs)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        if (pairs.Count == 0)
            return new Result(0, 0, 0, 0, 0, 0);

        int folds = pairs.Max(p => p.Fold) + 1;
        var accuracies = new List<double>();
        for (int f = 0; f < folds; f++)
        {
            var train = pairs.Where(p => p.Fold != f).ToList();
            var test = pairs.Where(p => p.Fold == f).ToList();
            if (test.Count == 0)
                continue;
            double threshold = train.Count > 0 ? BestThreshold(train) : BestThreshold(test);
            accuracies.Add(Accuracy(test, threshold));
        }

        double mean = accuracies.Count > 0 ? accuracies.Average() : 0;
        double std = accuracies.Count > 0 ? Math.Sqrt(accuracies.Select(a => (a - mean) * (a - mean)).Average()) : 0;

        return new Result(mean, std, Auc(pairs), TarAtFar(pairs, 0.01), TarAtFar(pairs, 0.001), pairs.Count);
    }

    private static double BestThreshold(IReadOnlyList<Pair> pairs)
    {
        var thresholds = pairs.Select(p => p.Score).Distinct().OrderBy(x => x).ToArray();
        double best = thresholds.Length > 0 ? thresholds[0] : 0;
        double bestAccuracy = -1;
        foreach (double t in thresholds)
        {
            double a = Accuracy(pairs, t);
            if (a > bestAccuracy)
            {
                bestAccuracy = a;
                best = t;
            }
        }
        return best;
    }

    private static double Accuracy(IReadOnlyList<Pair> pairs, double threshold)
    {
        int correct = 0;
        foreach (Pair p in pairs)
            if ((p.Score >= threshold) == p.Same)
                correct++;
        return (double)correct / pairs.Count;
    }

    // Rank-based AUC = P(score(same) > score(diff)).
    private static double Auc(IReadOnlyList<Pair> pairs)
    {
        double[] positives = pairs.Where(p => p.Same).Select(p => p.Score).OrderBy(x => x).ToArray();
        double[] negatives = pairs.Where(p => !p.Same).Select(p => p.Score).ToArray();
        if (positives.Length == 0 || negatives.Length == 0)
            return 0;

        double sum = 0;
        foreach (double n in negatives)
        {
            int greater = positives.Length - UpperBound(positives, n);
            int greaterOrEqual = positives.Length - LowerBound(positives, n);
            int equal = greaterOrEqual - greater;
            sum += greater + (0.5 * equal);
        }
        return sum / ((double)positives.Length * negatives.Length);
    }

    private static double TarAtFar(IReadOnlyList<Pair> pairs, double far)
    {
        double[] impostors = pairs.Where(p => !p.Same).Select(p => p.Score).OrderByDescending(x => x).ToArray();
        double[] genuine = pairs.Where(p => p.Same).Select(p => p.Score).ToArray();
        if (impostors.Length == 0 || genuine.Length == 0)
            return 0;

        int allowed = Math.Clamp((int)Math.Ceiling(far * impostors.Length), 1, impostors.Length);
        double threshold = impostors[allowed - 1]; // the allowed-th highest impostor score
        int accepted = genuine.Count(s => s >= threshold);
        return (double)accepted / genuine.Length;
    }

    private static int LowerBound(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (sorted[mid] < value) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    private static int UpperBound(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (sorted[mid] <= value) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
}
