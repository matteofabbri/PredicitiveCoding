using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for Thousand-Brains Phase 9: synthetic dataset, observation
/// corruptions, and the object-recognition benchmark (single column vs voting).</summary>
public class BrainEvalTests
{
    // ── SyntheticObjectDataset ────────────────────────────────────────────────

    [Fact]
    public void Dataset_SameSeed_IsDeterministic()
    {
        var a = SyntheticObjectDataset.Generate(objectCount: 5, locationsPerObject: 8, seed: 123);
        var b = SyntheticObjectDataset.Generate(objectCount: 5, locationsPerObject: 8, seed: 123);

        Assert.Equal(a.ObjectCount, b.ObjectCount);
        for (int o = 0; o < a.ObjectCount; o++)
            Assert.Equal(a.Objects[o].Points, b.Objects[o].Points);
    }

    [Fact]
    public void Dataset_SharesGridAndVocabulary_AcrossObjects()
    {
        var ds = SyntheticObjectDataset.Generate(
            objectCount: 6, locationsPerObject: 9, featureVocabularySize: 4, seed: 1);

        Assert.Equal(9, ds.Grid.Count);
        Assert.Equal(4, ds.Vocabulary.Count);

        // Every object uses the shared grid (same locations, possibly different features).
        foreach (var obj in ds.Objects)
        {
            Assert.Equal(ds.Grid.Count, obj.Points.Count);
            for (int g = 0; g < ds.Grid.Count; g++)
            {
                Assert.Equal(ds.Grid[g].X, obj.Points[g].X);
                Assert.Equal(ds.Grid[g].Y, obj.Points[g].Y);
                Assert.Contains(obj.Points[g].Feature, ds.Vocabulary);
            }
        }

        // Objects are not all identical (different feature assignments).
        var first = ds.Objects[0].Points.Select(p => p.Feature).ToList();
        Assert.Contains(ds.Objects.Skip(1),
            o => !o.Points.Select(p => p.Feature).SequenceEqual(first));
    }

    // ── ObservationCorruptions ────────────────────────────────────────────────

    [Fact]
    public void Occlude_KeepsRequestedFraction_AsSubset()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 1, locationsPerObject: 12, seed: 2);
        var pts = ds.Objects[0].Points;
        var rng = new Random(2);

        var kept = ObservationCorruptions.Occlude(pts, 0.5f, rng);

        Assert.Equal(6, kept.Count);                 // round(0.5 × 12)
        Assert.All(kept, p => Assert.Contains(p, pts));
    }

    [Fact]
    public void AddSensorNoise_PreservesCardinality_ReducesOverlap()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 1, locationsPerObject: 6, seed: 3);
        var pts = ds.Objects[0].Points;
        var rng = new Random(3);

        var noisy = ObservationCorruptions.AddSensorNoise(pts, 0.25f, rng);

        Assert.Equal(pts.Count, noisy.Count);
        for (int i = 0; i < pts.Count; i++)
        {
            Assert.Equal(pts[i].Feature.Cardinality, noisy[i].Feature.Cardinality);   // cardinality preserved
            Assert.True(noisy[i].Feature.Overlap(pts[i].Feature) < pts[i].Feature.Cardinality,
                "Noise must move at least one active bit.");
        }
    }

    [Fact]
    public void JitterPose_MovesLocations_KeepsFeaturesAndCount()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 1, locationsPerObject: 8, seed: 4);
        var pts = ds.Objects[0].Points;
        var rng = new Random(4);

        var jittered = ObservationCorruptions.JitterPose(pts, 0.3f, rng);

        Assert.Equal(pts.Count, jittered.Count);
        Assert.All(jittered.Zip(pts), pair => Assert.Equal(pair.Second.Feature, pair.First.Feature));
        Assert.Contains(jittered.Zip(pts),
            pair => pair.First.X != pair.Second.X || pair.First.Y != pair.Second.Y);
    }

    // ── RecognitionBenchmark ──────────────────────────────────────────────────

    [Fact]
    public void Benchmark_CleanRecognition_IsAccurate_AndVotingConvergesFaster()
    {
        var result = RecognitionBenchmark.Run(new BenchmarkConfig { Seed = 42 });

        Assert.Equal(4, result.Conditions.Count);
        Assert.All(result.Conditions, c =>
        {
            Assert.True(c.SingleColumn.Probes > 0);
            Assert.True(c.MultiColumn.Probes > 0);
        });

        var clean = result.Conditions.Single(c => c.Condition == "Clean");

        // Clean recognition is near-perfect for both recognisers.
        Assert.True(clean.SingleColumn.Top1Accuracy >= 0.9,
            $"single clean accuracy was {clean.SingleColumn.Top1Accuracy:P0}");
        Assert.True(clean.MultiColumn.Top1Accuracy >= 0.9,
            $"voting clean accuracy was {clean.MultiColumn.Top1Accuracy:P0}");

        // The central Thousand-Brains result: voting reaches a unique identity in
        // fewer sensorimotor steps than a single column (each round pools columns).
        Assert.True(clean.MultiColumn.MeanSteps < clean.SingleColumn.MeanSteps,
            $"voting steps {clean.MultiColumn.MeanSteps:F2} not < single steps {clean.SingleColumn.MeanSteps:F2}");
    }

    [Fact]
    public void Benchmark_Occlusion_DegradesNoMoreThanClean()
    {
        var result = RecognitionBenchmark.Run(new BenchmarkConfig { Seed = 42 });
        var clean = result.Conditions.Single(c => c.Condition == "Clean");
        var occ = result.Conditions.Single(c => c.Condition == "Occlusion");

        // A partial view cannot help; accuracy is at most the clean accuracy.
        Assert.True(occ.SingleColumn.Top1Accuracy <= clean.SingleColumn.Top1Accuracy);
    }
}
