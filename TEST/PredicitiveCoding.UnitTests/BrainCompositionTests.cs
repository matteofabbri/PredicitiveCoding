using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for Thousand-Brains Phase 8: compositionality (nested reference
/// frames via displacement), behaviours as objects, and abstract conceptual frames.</summary>
public class BrainCompositionTests
{
    private const float Eps = 1e-4f;

    // ── Pose2D ────────────────────────────────────────────────────────────────

    [Fact]
    public void Pose2D_Identity_LeavesPointUnchanged()
    {
        var (x, y) = Pose2D.Identity.Apply(3f, -4f);
        Assert.Equal(3f, x, Eps);
        Assert.Equal(-4f, y, Eps);
    }

    [Fact]
    public void Pose2D_Rotate90_MapsXAxisToYAxis()
    {
        var (x, y) = new Pose2D(0f, 0f, 90f).Apply(1f, 0f);
        Assert.Equal(0f, x, Eps);
        Assert.Equal(1f, y, Eps);
    }

    [Fact]
    public void Pose2D_ComposeTranslations_Adds()
    {
        var composed = Pose2D.FromTranslation(2f, 3f).Compose(Pose2D.FromTranslation(1f, 1f));
        Assert.Equal(3f, composed.X, Eps);
        Assert.Equal(4f, composed.Y, Eps);
        Assert.Equal(0f, composed.ThetaDeg, Eps);
    }

    [Fact]
    public void Pose2D_ComposeWithInverse_YieldsIdentity()
    {
        var p = new Pose2D(2f, 3f, 30f);
        var (x, y) = p.Compose(p.Inverse()).Apply(5f, 7f);
        Assert.Equal(5f, x, Eps);
        Assert.Equal(7f, y, Eps);
    }

    // ── CompositionLibrary: nested reference frames ──────────────────────────

    [Fact]
    public void Resolve_Primitive_ReturnsPointsUnchanged()
    {
        var rng = new Random(1);
        var f = Sdr.Random(20, 4, rng);
        var lib = new CompositionLibrary();
        lib.AddPrimitive(0, [new FeaturePoint(1f, 2f, f)]);

        var pts = lib.Resolve(0);

        Assert.Single(pts);
        Assert.Equal(1f, pts[0].X, Eps);
        Assert.Equal(2f, pts[0].Y, Eps);
        Assert.Equal(f, pts[0].Feature);
    }

    [Fact]
    public void Resolve_Composite_AppliesPartTranslation()
    {
        var rng = new Random(2);
        var f = Sdr.Random(20, 4, rng);
        var lib = new CompositionLibrary();
        lib.AddPrimitive(0, [new FeaturePoint(0f, 0f, f), new FeaturePoint(1f, 2f, f)]);
        lib.CreateComposite(10).AddPart(0, Pose2D.FromTranslation(5f, 5f));

        var pts = lib.Resolve(10);

        Assert.Equal(2, pts.Count);
        Assert.Equal(5f, pts[0].X, Eps);
        Assert.Equal(5f, pts[0].Y, Eps);
        Assert.Equal(6f, pts[1].X, Eps);
        Assert.Equal(7f, pts[1].Y, Eps);
    }

    [Fact]
    public void Resolve_Composite_AppliesPartRotation()
    {
        var rng = new Random(3);
        var lib = new CompositionLibrary();
        lib.AddPrimitive(0, [new FeaturePoint(1f, 0f, Sdr.Random(20, 4, rng))]);
        lib.CreateComposite(10).AddPart(0, new Pose2D(0f, 0f, 90f));

        var pts = lib.Resolve(10);

        Assert.Equal(0f, pts[0].X, Eps);
        Assert.Equal(1f, pts[0].Y, Eps);
    }

    [Fact]
    public void Resolve_NestedComposite_ComposesTransformsDownTheTree()
    {
        // primitive point at (1,0); inner composite shifts it by (10,0);
        // outer composite places the inner one at (0,5) rotated +90°.
        var rng = new Random(4);
        var lib = new CompositionLibrary();
        lib.AddPrimitive(0, [new FeaturePoint(1f, 0f, Sdr.Random(20, 4, rng))]);
        lib.CreateComposite(10).AddPart(0, new Pose2D(10f, 0f, 0f));
        lib.CreateComposite(20).AddPart(10, new Pose2D(0f, 5f, 90f));

        var pts = lib.Resolve(20);

        // (1,0) → +(10,0) = (11,0) → rotate 90° about origin = (0,11) → +(0,5) = (0,16).
        Assert.Single(pts);
        Assert.Equal(0f, pts[0].X, Eps);
        Assert.Equal(16f, pts[0].Y, Eps);
    }

    [Fact]
    public void Resolve_CyclicComposition_Throws()
    {
        var lib = new CompositionLibrary();
        lib.CreateComposite(10).AddPart(11, Pose2D.Identity);
        lib.CreateComposite(11).AddPart(10, Pose2D.Identity);

        Assert.Throws<InvalidOperationException>(() => lib.Resolve(10));
    }

    [Fact]
    public void Resolve_UnknownChildId_Throws()
    {
        var lib = new CompositionLibrary();
        lib.CreateComposite(10).AddPart(99, Pose2D.Identity);

        Assert.Throws<KeyNotFoundException>(() => lib.Resolve(10));
    }

    [Fact]
    public void AddPrimitive_DuplicateId_Throws()
    {
        var rng = new Random(5);
        var lib = new CompositionLibrary();
        lib.AddPrimitive(0, [new FeaturePoint(0f, 0f, Sdr.Random(20, 4, rng))]);

        Assert.Throws<ArgumentException>(
            () => lib.AddPrimitive(0, [new FeaturePoint(1f, 1f, Sdr.Random(20, 4, rng))]));
    }

    [Fact]
    public void ToObservations_FeedCorticalColumn_RecognisesCompositeObject()
    {
        // Build three composite objects from disjoint primitives, teach them all to
        // one column, then infer one object's own observations: the consensus must
        // contain that object's id and have narrowed below the full object count.
        var rng = new Random(7);
        const int inputSize = 100;

        var lib = new CompositionLibrary();
        for (int p = 0; p < 6; p++)
            lib.AddPrimitive(p,
            [
                new FeaturePoint(0f, 0f, Sdr.Random(inputSize, 20, rng)),
                new FeaturePoint(1f, 1f, Sdr.Random(inputSize, 20, rng)),
            ]);

        // Composite c → primitives (2c, 2c+1); the second part displaced by (5,5).
        for (int c = 0; c < 3; c++)
            lib.CreateComposite(10 + c)
               .AddPart(2 * c, Pose2D.Identity)
               .AddPart(2 * c + 1, Pose2D.FromTranslation(5f, 5f));

        var pose = new PoseEncoder(-10f, 10f, -10f, 10f,
            positionBits: 60, positionActiveBits: 6, orientationBuckets: 8, orientationActiveBits: 5);

        var column = new CorticalColumn(new CorticalColumnConfig
        {
            InputSize      = inputSize,
            Columns        = 64,
            CellsPerColumn = 4,
            Sparsity       = 0.05f,
            MinOverlap     = 2,
            Seed           = 7,
        });

        int targetId = -1;
        var targetObs = lib.ToObservations(10, pose);
        for (int c = 0; c < 3; c++)
        {
            int id = column.LearnObject(lib.ToObservations(10 + c, pose));
            if (c == 0) targetId = id;
        }
        Assert.Equal(3, column.ObjectCount);

        // Each composite resolves to four (feature, location) points.
        Assert.Equal(4, targetObs.Count);
        Assert.All(targetObs, o =>
        {
            Assert.Equal(inputSize, o.Feature.Size);
            Assert.Equal(pose.EncodeXY(0f, 0f).Size, o.Location.Size);
        });

        var last = column.Infer(targetObs)[^1];
        Assert.Contains(targetId, last.ObjectHypotheses);
        Assert.True(last.ObjectHypotheses.Count < 3,
            $"Expected fewer than 3 hypotheses, got {last.ObjectHypotheses.Count}.");
    }

    // ── BehaviorSequence: behaviours as objects in a 1-D frame ───────────────

    [Fact]
    public void BehaviorSequence_AppendAndPredictNext_Works()
    {
        var rng = new Random(8);
        var s0 = Sdr.Random(40, 8, rng);
        var s1 = Sdr.Random(40, 8, rng);
        var s2 = Sdr.Random(40, 8, rng);
        var seq = new BehaviorSequence(0).Append(s0).Append(s1).Append(s2);

        Assert.Equal(3, seq.Length);
        Assert.Equal(s1, seq.At(1));
        Assert.Equal(s1, seq.PredictNext(0));
        Assert.Equal(s2, seq.PredictNext(1));
        Assert.Null(seq.PredictNext(2));   // past the end
    }

    [Fact]
    public void BehaviorSequence_ToObjectPoints_PlacesEachStepAtItsOwnLocation()
    {
        var rng = new Random(9);
        var s0 = Sdr.Random(40, 8, rng);
        var s1 = Sdr.Random(40, 8, rng);
        var s2 = Sdr.Random(40, 8, rng);
        var seq = new BehaviorSequence(0).Append(s0).Append(s1).Append(s2);
        var posEnc = new ScalarEncoder(0, 10, 60, 6);

        var pts = seq.ToObjectPoints(posEnc);

        Assert.Equal(3, pts.Count);
        Assert.Equal(s0, pts[0].Feature);
        Assert.Equal(60, pts[0].Location.Size);
        // Steps far apart in the sequence map to disjoint location SDRs.
        Assert.Equal(0, pts[0].Location.Overlap(pts[2].Location));
    }

    [Fact]
    public void BehaviorSequence_MatchScore_ExactPrefixIsOne_MismatchLower()
    {
        var rng = new Random(10);
        var s0 = Sdr.Random(40, 8, rng);
        var s1 = Sdr.Random(40, 8, rng);
        var s2 = Sdr.Random(40, 8, rng);
        var other = Sdr.Random(40, 8, rng);   // disjoint from s1 with high probability
        var seq = new BehaviorSequence(0).Append(s0).Append(s1).Append(s2);

        Assert.Equal(1.0, seq.MatchScore([s0, s1], minOverlap: 4), precision: 6);
        Assert.Equal(0.5, seq.MatchScore([s0, other], minOverlap: 4), precision: 6);
    }

    // ── ConceptualFrame: abstract reference frames ───────────────────────────

    [Fact]
    public void ConceptualFrame_EncodeLocation_NearbyOverlapsMoreThanDistant()
    {
        var frame = new ConceptualFrame([(0, 10), (0, 10)], bitsPerAxis: 60, activePerAxis: 6);

        Assert.Equal(120, frame.LocationSize);

        var a = frame.EncodeLocation([1.0, 1.0]);
        var near = frame.EncodeLocation([1.2, 0.9]);
        var far = frame.EncodeLocation([9.0, 9.0]);

        Assert.True(a.Overlap(near) > a.Overlap(far));
    }

    [Fact]
    public void ConceptualFrame_Recall_ReturnsNearestConcept()
    {
        var rng = new Random(11);
        var conceptA = Sdr.Random(50, 10, rng);
        var conceptB = Sdr.Random(50, 10, rng);
        var frame = new ConceptualFrame([(0, 10), (0, 10)]);
        frame.AddConcept([1.0, 1.0], conceptA);
        frame.AddConcept([9.0, 9.0], conceptB);

        Assert.Equal(2, frame.ConceptCount);
        Assert.Equal(conceptA, frame.Recall([1.2, 0.9]));
        Assert.Equal(conceptB, frame.Recall([8.0, 8.5]));
    }

    [Fact]
    public void ConceptualFrame_Move_AppliesDisplacementInConceptSpace()
    {
        var frame = new ConceptualFrame([(0, 10), (0, 10)]);

        var dest = frame.Move([1.0, 1.0], [2.0, 3.0]);

        Assert.Equal(3.0, dest[0], precision: 6);
        Assert.Equal(4.0, dest[1], precision: 6);
    }

    [Fact]
    public void ConceptualFrame_Nearest_EmptyFrame_ReturnsNull()
    {
        var frame = new ConceptualFrame([(0, 10)]);
        Assert.Null(frame.Nearest([5.0]));
    }
}
