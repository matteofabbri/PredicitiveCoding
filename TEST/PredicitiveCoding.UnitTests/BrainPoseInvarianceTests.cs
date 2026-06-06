using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for pose-invariant recognition via displacement structure
/// (DisplacementRecognizer) — recognition that survives rigid motion of the object.</summary>
public class BrainPoseInvarianceTests
{
    private static DisplacementRecognizer LearnAll(SyntheticObjectDataset ds, int minOverlap = 12)
    {
        var rec = new DisplacementRecognizer(minOverlap, distanceTolerance: 0.6f);
        foreach (var o in ds.Objects) rec.Learn(o.Id, o.Points);
        return rec;
    }

    [Fact]
    public void Recognize_UnderRotationAndTranslation_IdentifiesEveryObject()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 8, locationsPerObject: 12, seed: 5);
        var rec = LearnAll(ds);

        // A large rigid motion that would break absolute-location matching entirely.
        var pose = new Pose2D(6f, -4f, 75f);
        foreach (var o in ds.Objects)
        {
            var moved = o.Points.Select(p => p.Transformed(pose)).ToList();
            Assert.Equal(o.Id, rec.Recognize(moved));
        }
    }

    [Fact]
    public void Recognize_UnderPureTranslation_IdentifiesEveryObject()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 6, locationsPerObject: 10, seed: 6);
        var rec = LearnAll(ds);

        var shift = Pose2D.FromTranslation(20f, -15f);
        foreach (var o in ds.Objects)
        {
            var moved = o.Points.Select(p => p.Transformed(shift)).ToList();
            Assert.Equal(o.Id, rec.Recognize(moved));
        }
    }

    [Fact]
    public void Recognize_TransformedAndOccluded_StillIdentifies()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 6, locationsPerObject: 14, seed: 7);
        var rec = LearnAll(ds);
        var rng = new Random(7);
        var pose = new Pose2D(-3f, 8f, 120f);

        int correct = 0;
        foreach (var o in ds.Objects)
        {
            var moved = o.Points.Select(p => p.Transformed(pose)).ToList();
            var partial = ObservationCorruptions.Occlude(moved, 0.6f, rng);   // 60% of a rotated view
            if (rec.Recognize(partial) == o.Id) correct++;
        }
        // Robust to a rigid move plus a partial view.
        Assert.True(correct >= 5, $"recognised {correct}/6 under transform + occlusion");
    }

    [Fact]
    public void Rank_PutsTrueObjectFirst_WithPositiveConfidence()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 8, locationsPerObject: 12, seed: 8);
        var rec = LearnAll(ds);
        var pose = new Pose2D(5f, 5f, 200f);

        var moved = ds.Objects[3].Points.Select(p => p.Transformed(pose)).ToList();
        var ranked = rec.Rank(moved);

        Assert.Equal(ds.ObjectCount, ranked.Count);
        Assert.Equal(3, ranked[0].ObjectId);
        Assert.True(ranked[0].Confidence > 0);
        Assert.True(ranked[0].MatchingPairs >= ranked[^1].MatchingPairs);
    }

    [Fact]
    public void Recognize_NothingLearned_ReturnsNull()
    {
        var rec = new DisplacementRecognizer();
        var ds = SyntheticObjectDataset.Generate(objectCount: 1, locationsPerObject: 5, seed: 9);
        Assert.Null(rec.Recognize(ds.Objects[0].Points));
        Assert.Equal(0, rec.ObjectCount);
    }
}
