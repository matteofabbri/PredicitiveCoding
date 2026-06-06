namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// RecognitionBenchmark  —  the Phase 9 object-recognition benchmark.
//
// Teaches a set of synthetic objects (SyntheticObjectDataset) to two recognisers —
// a single CorticalColumn and a multi-column voting system (MultiColumnSystem) —
// then scores both under four conditions: clean, occlusion, sensor noise and pose
// jitter (ObservationCorruptions). For each it reports top-1 accuracy and the mean
// number of sensorimotor steps to a unique recognition.
//
// The benchmark exists to quantify the central Thousand-Brains claim (P9): a
// population of columns that vote recognises objects more robustly and in fewer
// steps than a single column — each VoteStep lets several columns pool evidence
// from different parts of the object at once.
//
// Reference: Leadholm et al. (2025) P9; Clay et al. (2024) P2.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Score for one recogniser under one condition.</summary>
public sealed record RecognizerScore(double Top1Accuracy, double MeanSteps, int Probes);

/// <summary>Single-column vs multi-column scores for one test condition.</summary>
public sealed record ConditionResult(string Condition, RecognizerScore SingleColumn, RecognizerScore MultiColumn);

/// <summary>Full benchmark outcome across all conditions.</summary>
public sealed record BenchmarkResult(
    int Objects, int LocationsPerObject, int Columns, IReadOnlyList<ConditionResult> Conditions);

/// <summary>Benchmark parameters (all optional; defaults give a quick, meaningful run).</summary>
public sealed record BenchmarkConfig
{
    public int ObjectCount { get; init; } = 10;

    /// <summary>Landmarks per object = size of the shared location grid.</summary>
    public int LocationsPerObject { get; init; } = 12;

    /// <summary>Size of the shared feature vocabulary (smaller ⇒ more ambiguous landmarks).</summary>
    public int FeatureVocabularySize { get; init; } = 5;

    public int FeatureSize { get; init; } = 256;
    public int FeatureActiveBits { get; init; } = 24;
    public float Extent { get; init; } = 5f;

    /// <summary>Columns in the voting system compared against a single column.</summary>
    public int Columns { get; init; } = 4;

    /// <summary>Independent corrupted probes generated per object per condition.</summary>
    public int ProbesPerObject { get; init; } = 3;

    public float OcclusionKeep { get; init; } = 0.5f;
    public float SensorNoise { get; init; } = 0.25f;
    public float PoseJitter { get; init; } = 0.3f;

    /// <summary>Maximum voting rounds before the multi-column system gives up.</summary>
    public int MaxVoteRounds { get; init; } = 12;

    public int Seed { get; init; } = 42;
}

public static class RecognitionBenchmark
{
    public static BenchmarkResult Run(BenchmarkConfig? config = null, Action<string>? onProgress = null)
    {
        var cfg = config ?? new BenchmarkConfig();
        if (cfg.Columns < 2) throw new ArgumentException("Columns must be ≥ 2 for voting.", nameof(config));

        var dataset = SyntheticObjectDataset.Generate(
            cfg.ObjectCount, cfg.LocationsPerObject, cfg.FeatureVocabularySize,
            cfg.FeatureSize, cfg.FeatureActiveBits, cfg.Extent, cfg.Seed);

        float bound = cfg.Extent + cfg.PoseJitter + 1f;
        var pose = new PoseEncoder(-bound, bound, -bound, bound,
            positionBits: 100, positionActiveBits: 10, orientationBuckets: 8, orientationActiveBits: 5);

        // ── Teach both recognisers the same objects (ids 0 … N−1, in order) ──────
        onProgress?.Invoke($"Teaching {dataset.ObjectCount} objects to 1 column and to {cfg.Columns} voting columns …");

        var single = new CorticalColumn(ColumnConfig(cfg, seed: cfg.Seed));
        // Unanimous agreement ⇒ the consensus is the strict intersection of every
        // column's candidates, so columns pool complementary evidence each round.
        var multi = new MultiColumnSystem(
            Enumerable.Range(0, cfg.Columns).Select(c => ColumnConfig(cfg, seed: cfg.Seed + 1 + c)).ToList(),
            VotingTopology.AllToAll,
            minAgreementFraction: 1.0f);

        foreach (var obj in dataset.Objects)
        {
            var obs = Encode(obj.Points, pose);
            single.LearnObject(obs);
            multi.LearnObject(Enumerable.Repeat(obs, cfg.Columns).ToList());
        }

        // ── Score every condition ────────────────────────────────────────────────
        var corruptionRng = new Random(cfg.Seed + 1000);
        var conditions = new (string Name, Func<IReadOnlyList<FeaturePoint>, IReadOnlyList<FeaturePoint>> Corrupt)[]
        {
            ("Clean",     pts => pts),
            ("Occlusion", pts => ObservationCorruptions.Occlude(pts, cfg.OcclusionKeep, corruptionRng)),
            ("Noise",     pts => ObservationCorruptions.AddSensorNoise(pts, cfg.SensorNoise, corruptionRng)),
            ("PoseJitter",pts => ObservationCorruptions.JitterPose(pts, cfg.PoseJitter, corruptionRng)),
        };

        var results = new List<ConditionResult>(conditions.Length);
        foreach (var (name, corrupt) in conditions)
        {
            int singleCorrect = 0, multiCorrect = 0, probes = 0;
            long singleSteps = 0, multiSteps = 0;

            foreach (var obj in dataset.Objects)
            {
                for (int r = 0; r < cfg.ProbesPerObject; r++)
                {
                    var probe = Encode(corrupt(obj.Points), pose);
                    if (probe.Count == 0) continue;

                    var (sOk, sStep) = EvalSingle(single, probe, obj.Id);
                    var (mOk, mStep) = EvalMulti(multi, probe, cfg.Columns, cfg.MaxVoteRounds, obj.Id);

                    if (sOk) singleCorrect++;
                    if (mOk) multiCorrect++;
                    singleSteps += sStep;
                    multiSteps += mStep;
                    probes++;
                }
            }

            results.Add(new ConditionResult(name,
                new RecognizerScore((double)singleCorrect / probes, (double)singleSteps / probes, probes),
                new RecognizerScore((double)multiCorrect / probes, (double)multiSteps / probes, probes)));
            onProgress?.Invoke(
                $"  {name,-10}  single {(double)singleCorrect / probes,6:P0}   voting {(double)multiCorrect / probes,6:P0}");
        }

        return new BenchmarkResult(dataset.ObjectCount, cfg.LocationsPerObject, cfg.Columns, results);
    }

    // ── Recognisers ──────────────────────────────────────────────────────────

    private static (bool Correct, int Steps) EvalSingle(
        CorticalColumn column, IReadOnlyList<(Sdr Feature, Sdr Location)> probe, int trueId)
    {
        var outputs = column.Infer(probe);
        for (int i = 0; i < outputs.Count; i++)
            if (outputs[i].Converged)
                return (outputs[i].ObjectHypotheses[0] == trueId, i + 1);

        var last = outputs[^1];
        int predicted = last.ObjectHypotheses.Count > 0 ? last.ObjectHypotheses[0] : -1;
        return (predicted == trueId, probe.Count);
    }

    private static (bool Correct, int Steps) EvalMulti(
        MultiColumnSystem system, IReadOnlyList<(Sdr Feature, Sdr Location)> probe,
        int columns, int maxRounds, int trueId)
    {
        system.Reset();
        ConsensusResult? last = null;
        int p = probe.Count;

        for (int round = 0; round < maxRounds; round++)
        {
            // Each column senses a *different* landmark this round, so one voting
            // round pools `columns` complementary observations of the object.
            var inputs = new (Sdr, Sdr)[columns];
            for (int c = 0; c < columns; c++) inputs[c] = probe[(round * columns + c) % p];

            last = system.VoteStep(inputs);
            if (last.Candidates.Count == 1)
                return (last.Candidates[0] == trueId, round + 1);
        }

        int predicted = last is { Candidates.Count: > 0 } ? last.Candidates[0] : -1;
        return (predicted == trueId, maxRounds);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static IReadOnlyList<(Sdr Feature, Sdr Location)> Encode(
        IReadOnlyList<FeaturePoint> points, PoseEncoder pose)
    {
        var obs = new List<(Sdr, Sdr)>(points.Count);
        foreach (var pt in points)
            obs.Add((pt.Feature, pose.EncodeXY(pt.X, pt.Y)));
        return obs;
    }

    private static CorticalColumnConfig ColumnConfig(BenchmarkConfig cfg, int seed) => new()
    {
        InputSize = cfg.FeatureSize,
        Columns = 256,
        CellsPerColumn = 4,
        Sparsity = 0.04f,
        MinOverlap = 4,
        Seed = seed,
    };
}
