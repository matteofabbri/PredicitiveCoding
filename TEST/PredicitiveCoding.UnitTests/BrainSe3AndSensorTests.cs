using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for the SE(3) rigid-body pose (Pose3D) and FeaturePoint3D — the 3-D
/// twins of Pose2D / FeaturePoint. The five anchor cases A1–A5 pin the rotation
/// convention: yaw about +Z (matching Pose2D θ), positive pitch lifts +X toward +Z,
/// roll about the forward +X axis, intrinsic yaw→pitch→roll order.</summary>
public class BrainSe3PoseTests
{
    private const float Tol = 1e-4f;

    /// <summary>Compares two angles in degrees modulo 360 (handles wrap, e.g. -25° ≡ 335°).</summary>
    private static void AssertAngle(float expectedDeg, float actualDeg, float tolDeg = 1e-3f)
    {
        float delta = ((expectedDeg - actualDeg) % 360f + 540f) % 360f - 180f;
        Assert.True(Math.Abs(delta) <= tolDeg,
            $"angle {actualDeg}° != {expectedDeg}° (Δ = {delta}°)");
    }

    private static void AssertApplied(Pose3D pose, float px, float py, float pz,
        float ex, float ey, float ez, float tol = Tol)
    {
        var (x, y, z) = pose.Apply(px, py, pz);
        Assert.Equal(ex, x, tol);
        Assert.Equal(ey, y, tol);
        Assert.Equal(ez, z, tol);
    }

    [Fact]
    public void A1_Yaw90_RotatesXTowardY_AboutZ()
        => AssertApplied(Pose3D.FromYawPitchRoll(90f, 0f, 0f), 1f, 0f, 0f, 0f, 1f, 0f);

    [Fact]
    public void A2_Pitch90_LiftsForwardAxisTowardZ()
        => AssertApplied(Pose3D.FromYawPitchRoll(0f, 90f, 0f), 1f, 0f, 0f, 0f, 0f, 1f);

    [Fact]
    public void A3_Roll90_RotatesYTowardZ_AboutForwardAxis()
        => AssertApplied(Pose3D.FromYawPitchRoll(0f, 0f, 90f), 0f, 1f, 0f, 0f, 0f, 1f);

    [Fact]
    public void A4_FromYawPitchRoll_IsIntrinsicYawThenPitchThenRoll()
    {
        var full = Pose3D.FromYawPitchRoll(40f, 30f, 20f);
        var composed = Pose3D.FromYawPitchRoll(40f, 0f, 0f)
            .Compose(Pose3D.FromYawPitchRoll(0f, 30f, 0f))
            .Compose(Pose3D.FromYawPitchRoll(0f, 0f, 20f));

        // Pure rotations: no translation on either side.
        Assert.Equal(0f, full.X, Tol);
        Assert.Equal(0f, composed.X, Tol);
        Assert.Equal(0f, composed.Y, Tol);
        Assert.Equal(0f, composed.Z, Tol);

        // Same action on the basis vectors and on a generic probe point.
        foreach (var (px, py, pz) in new[]
        {
            (1f, 0f, 0f), (0f, 1f, 0f), (0f, 0f, 1f), (0.3f, -0.7f, 0.5f),
        })
        {
            var (fx, fy, fz) = full.Apply(px, py, pz);
            var (cx, cy, cz) = composed.Apply(px, py, pz);
            Assert.Equal(fx, cx, Tol);
            Assert.Equal(fy, cy, Tol);
            Assert.Equal(fz, cz, Tol);
        }
    }

    [Theory]
    [InlineData(0f, 0f, 0f)]
    [InlineData(30f, 40f, -25f)]
    [InlineData(-60f, 45f, 100f)]
    [InlineData(120f, -80f, 10f)]
    [InlineData(200f, 10f, -170f)]
    public void A5_ToYawPitchRoll_RoundTripsForPitchInsideGimbalRange(float yaw, float pitch, float roll)
    {
        var (y, p, r) = Pose3D.FromYawPitchRoll(yaw, pitch, roll).ToYawPitchRoll();
        AssertAngle(yaw, y);
        AssertAngle(pitch, p);
        AssertAngle(roll, r);
    }

    [Fact]
    public void Apply_PureTranslation_ShiftsPoint()
        => AssertApplied(Pose3D.FromTranslation(2f, -3f, 4f), 1f, 1f, 1f, 3f, -2f, 5f);

    [Fact]
    public void ComposeInverse_IsIdentity()
    {
        var pose = Pose3D.FromTranslation(5f, -3f, 2f)
            .Compose(Pose3D.FromYawPitchRoll(37f, 20f, 65f));
        var roundTrip = pose.Compose(pose.Inverse());

        // Translation vanishes…
        Assert.Equal(0f, roundTrip.X, Tol);
        Assert.Equal(0f, roundTrip.Y, Tol);
        Assert.Equal(0f, roundTrip.Z, Tol);

        // …and the residual rotation leaves a probe vector untouched.
        AssertApplied(roundTrip, 0.4f, -1.2f, 0.8f, 0.4f, -1.2f, 0.8f);
    }

    [Fact]
    public void YawOnlyPose3D_MatchesPose2D_OnPlanarPoints()
    {
        var pose2 = new Pose2D(2f, 3f, 50f);
        var pose3 = new Pose3D(2f, 3f, 0f, Pose3D.FromYawPitchRoll(50f, 0f, 0f).Rotation);

        foreach (var (px, py) in new[] { (1.5f, -0.5f), (-2f, 0.75f), (0f, 0f) })
        {
            var (ex, ey) = pose2.Apply(px, py);
            var (ax, ay, az) = pose3.Apply(px, py, 0f);
            Assert.Equal(ex, ax, Tol);
            Assert.Equal(ey, ay, Tol);
            Assert.Equal(0f, az, Tol);   // the degenerate z = 0 case stays planar
        }
    }

    [Fact]
    public void ComposedChain_AppliesLikeNestedApply()
    {
        var parent = new Pose3D(1f, 2f, 3f, Pose3D.FromYawPitchRoll(30f, 15f, -10f).Rotation);
        var child = new Pose3D(-2f, 0.5f, 1f, Pose3D.FromYawPitchRoll(-45f, 25f, 60f).Rotation);

        var (cx, cy, cz) = child.Apply(0.7f, -1.1f, 0.4f);
        var (ex, ey, ez) = parent.Apply(cx, cy, cz);
        var (ax, ay, az) = parent.Compose(child).Apply(0.7f, -1.1f, 0.4f);

        Assert.Equal(ex, ax, Tol);
        Assert.Equal(ey, ay, Tol);
        Assert.Equal(ez, az, Tol);
    }

    [Fact]
    public void FeaturePoint3D_Transformed_MovesPoint_KeepsFeatureReference()
    {
        var feature = Sdr.Random(64, 8, new Random(3));
        var point = new FeaturePoint3D(1f, 2f, 3f, feature);

        var shifted = point.Transformed(Pose3D.FromTranslation(1f, 1f, 1f));
        Assert.Equal(2f, shifted.X, Tol);
        Assert.Equal(3f, shifted.Y, Tol);
        Assert.Equal(4f, shifted.Z, Tol);
        Assert.Same(feature, shifted.Feature);

        var rotated = new FeaturePoint3D(1f, 0f, 0f, feature)
            .Transformed(Pose3D.FromYawPitchRoll(90f, 0f, 0f));
        Assert.Equal(0f, rotated.X, Tol);
        Assert.Equal(1f, rotated.Y, Tol);
        Assert.Equal(0f, rotated.Z, Tol);
        Assert.Same(feature, rotated.Feature);
    }
}

/// <summary>Tests for the (x, y, z, orientation) → SDR encoder (PoseEncoder3D) and 3-D
/// grid-cell path integration (PathIntegrator3D) — the SE(3) twins of PoseEncoder /
/// PathIntegrator (Hawkins et al. 2019, P3).</summary>
public class BrainSe3PathIntegrationTests
{
    private static PoseEncoder3D Encoder() => new(-10f, 10f, -10f, 10f, -10f, 10f);

    [Fact]
    public void PoseEncoder3D_SamePoseTwice_YieldsIdenticalSdr()
    {
        var enc = Encoder();

        var first = enc.Encode(2f, 3f, 4f, 45f, 15f, -30f);
        var second = enc.Encode(2f, 3f, 4f, 45f, 15f, -30f);

        Assert.Equal(first, second);
        Assert.Equal(enc.Size, first.Size);
    }

    [Fact]
    public void PoseEncoder3D_NearbyPositions_ShareMostActiveBits()
    {
        var enc = new PoseEncoder3D(0f, 10f, 0f, 10f, 0f, 10f);
        var a = enc.EncodeXYZ(1f, 1f, 1f);
        var b = enc.EncodeXYZ(1.2f, 1f, 1f);

        Assert.True(a.Overlap(b) > a.Cardinality / 2,
            $"near overlap {a.Overlap(b)} of {a.Cardinality} active bits");
    }

    [Fact]
    public void PoseEncoder3D_FarPositions_ShareFewActiveBits()
    {
        var enc = new PoseEncoder3D(0f, 10f, 0f, 10f, 0f, 10f);
        var a = enc.EncodeXYZ(1f, 1f, 1f);
        var near = enc.EncodeXYZ(1.2f, 1f, 1f);
        var far = enc.EncodeXYZ(9f, 9f, 9f);

        Assert.True(a.Overlap(far) < a.Cardinality / 4,
            $"far overlap {a.Overlap(far)} of {a.Cardinality} active bits");
        Assert.True(a.Overlap(near) > a.Overlap(far));
    }

    [Fact]
    public void PoseEncoder3D_OrientationChangeAlone_ChangesTheSdr()
    {
        var enc = Encoder();
        var level = enc.Encode(5f, 5f, 5f, 0f, 0f, 0f);
        var yawed = enc.Encode(5f, 5f, 5f, 90f, 0f, 0f);

        Assert.NotEqual(level, yawed);
    }

    [Fact]
    public void PathIntegrator3D_StepForward_FollowsCurrentHeading()
    {
        var pi = new PathIntegrator3D(Encoder());

        pi.StepForward(2f);            // identity orientation → forward is +X
        Assert.Equal(2f, pi.X, 1e-4f);
        Assert.Equal(0f, pi.Y, 1e-4f);
        Assert.Equal(0f, pi.Z, 1e-4f);

        pi.Rotate(0f, 90f, 0f);        // pitch up: forward becomes +Z (anchor A2)
        pi.StepForward(2f);
        Assert.Equal(2f, pi.X, 1e-4f);
        Assert.Equal(0f, pi.Y, 1e-4f);
        Assert.Equal(2f, pi.Z, 1e-4f);
    }

    [Fact]
    public void PathIntegrator3D_PlanarSquareLoop_ReturnsToStart()
    {
        var pi = new PathIntegrator3D(Encoder());

        // Walk a 2×2 square in the x–y plane: forward, yaw 90°, ×4.
        for (int side = 0; side < 4; side++)
        {
            pi.StepForward(2f);
            pi.Rotate(90f, 0f, 0f);
        }

        Assert.True(Math.Abs(pi.X) < 1e-3f, $"X = {pi.X}");
        Assert.True(Math.Abs(pi.Y) < 1e-3f, $"Y = {pi.Y}");
        Assert.True(Math.Abs(pi.Z) < 1e-3f, $"Z = {pi.Z}");
    }

    [Fact]
    public void PathIntegrator3D_VerticalSquareLoop_ReturnsToStart()
    {
        var pi = new PathIntegrator3D(Encoder());

        // The same square stood upright in the x–z plane: forward, pitch 90°, ×4.
        for (int side = 0; side < 4; side++)
        {
            pi.StepForward(2f);
            pi.Rotate(0f, 90f, 0f);
        }

        Assert.True(Math.Abs(pi.X) < 1e-3f, $"X = {pi.X}");
        Assert.True(Math.Abs(pi.Y) < 1e-3f, $"Y = {pi.Y}");
        Assert.True(Math.Abs(pi.Z) < 1e-3f, $"Z = {pi.Z}");
    }

    [Fact]
    public void PathIntegrator3D_ClimbTraverseDescend_ClosesTheLoop()
    {
        var pi = new PathIntegrator3D(Encoder());

        pi.Rotate(0f, 90f, 0f);  pi.StepForward(2f);   // climb +Z to (0, 0, 2)
        pi.Rotate(0f, -90f, 0f); pi.StepForward(3f);   // level again, +X to (3, 0, 2)
        pi.Rotate(0f, -90f, 0f); pi.StepForward(2f);   // dive −Z to (3, 0, 0)
        pi.Rotate(0f, 90f, 0f);                        // level, facing +X
        pi.Rotate(90f, 0f, 0f);  pi.Rotate(90f, 0f, 0f);
        pi.StepForward(3f);                            // yaw 180° → back to the origin

        Assert.True(Math.Abs(pi.X) < 1e-3f, $"X = {pi.X}");
        Assert.True(Math.Abs(pi.Y) < 1e-3f, $"Y = {pi.Y}");
        Assert.True(Math.Abs(pi.Z) < 1e-3f, $"Z = {pi.Z}");
    }

    [Fact]
    public void PathIntegrator3D_StepAndCounterStep_ReturnsToOrigin()
    {
        var pi = new PathIntegrator3D(Encoder());

        pi.Step(1f, 2f, 3f);
        pi.Step(-1f, -2f, -3f);

        Assert.Equal(0f, pi.X, 1e-4f);
        Assert.Equal(0f, pi.Y, 1e-4f);
        Assert.Equal(0f, pi.Z, 1e-4f);
    }

    [Fact]
    public void PathIntegrator3D_LocationCode_IsIdenticalAtIdenticalPoses()
    {
        var enc = Encoder();
        var a = new PathIntegrator3D(enc);
        var b = new PathIntegrator3D(enc);

        Assert.Equal(a.LocationCode, b.LocationCode);
        Assert.Equal(enc.Size, a.LocationCode.Size);

        // The same motor sequence keeps the codes in lock-step.
        foreach (var pi in new[] { a, b })
        {
            pi.StepForward(1.5f);
            pi.Rotate(30f, 10f, 5f);
            pi.Step(0.5f, -0.25f, 1f);
        }
        Assert.Equal(a.LocationCode, b.LocationCode);
    }
}

/// <summary>Tests for pose-invariant 3-D recognition via displacement structure
/// (DisplacementRecognizer3D) — the SE(3) twin of DisplacementRecognizer: pairwise
/// landmark distances are invariant to the full group of 3-D rigid motions.</summary>
public class BrainSe3RecognitionTests
{
    private const int CubeId = 0;
    private const int CrossId = 1;

    /// <summary>Six random features shared by both objects, so recognition must rely on
    /// geometry (the displacement structure), not on which features appear.</summary>
    private static Sdr[] Vocabulary()
    {
        var rng = new Random(11);
        return [.. Enumerable.Range(0, 6).Select(_ => Sdr.Random(256, 20, rng))];
    }

    /// <summary>The 8 corners of an axis-aligned cube of side 4, centred on the origin.</summary>
    private static List<FeaturePoint3D> Cube(Sdr[] vocab)
    {
        var corners = new List<FeaturePoint3D>(8);
        for (int i = 0; i < 8; i++)
            corners.Add(new FeaturePoint3D(
                (i & 1) != 0 ? 2f : -2f,
                (i & 2) != 0 ? 2f : -2f,
                (i & 4) != 0 ? 2f : -2f,
                vocab[i % 6]));
        return corners;
    }

    /// <summary>A flat cross in the z = 0 plane (different pairwise-distance signature).</summary>
    private static List<FeaturePoint3D> Cross(Sdr[] vocab) =>
    [
        new(0f, 0f, 0f, vocab[0]),
        new(4f, 0f, 0f, vocab[1]),
        new(-4f, 0f, 0f, vocab[2]),
        new(0f, 4f, 0f, vocab[3]),
        new(0f, -4f, 0f, vocab[4]),
    ];

    private static DisplacementRecognizer3D LearnBoth(Sdr[] vocab)
    {
        var rec = new DisplacementRecognizer3D();
        rec.Learn(CubeId, Cube(vocab));
        rec.Learn(CrossId, Cross(vocab));
        return rec;
    }

    /// <summary>A large SE(3) motion that would break absolute-location matching entirely.</summary>
    private static Pose3D RigidMotion()
        => Pose3D.FromYawPitchRoll(37f, 20f, 65f).Compose(Pose3D.FromTranslation(5f, -3f, 2f));

    [Fact]
    public void Recognize_CubeUnderRigidMotion_ReturnsCube()
    {
        var vocab = Vocabulary();
        var rec = LearnBoth(vocab);

        var moved = Cube(vocab).Select(p => p.Transformed(RigidMotion())).ToList();
        Assert.Equal(CubeId, rec.Recognize(moved));
    }

    [Fact]
    public void Recognize_OccludedCubeView_StillReturnsCube()
    {
        var vocab = Vocabulary();
        var rec = LearnBoth(vocab);

        // Only 5 of the 8 corners are visible after the rigid motion.
        var partial = Cube(vocab).Take(5).Select(p => p.Transformed(RigidMotion())).ToList();
        Assert.Equal(CubeId, rec.Recognize(partial));
    }

    [Fact]
    public void Rank_CubeObservation_ScoresObjectsWithoutTie()
    {
        var vocab = Vocabulary();
        var rec = LearnBoth(vocab);
        var moved = Cube(vocab).Select(p => p.Transformed(RigidMotion())).ToList();

        var ranked = rec.Rank(moved);
        Assert.Equal(2, ranked.Count);

        RecognitionScore best = ranked[0];      // the shared RecognitionScore record
        Assert.Equal(CubeId, best.ObjectId);
        Assert.True(best.Confidence > 0);

        var cubeScore = ranked.First(s => s.ObjectId == CubeId);
        var crossScore = ranked.First(s => s.ObjectId == CrossId);
        Assert.True(cubeScore.MatchingPairs > crossScore.MatchingPairs,
            $"cube {cubeScore.MatchingPairs} vs cross {crossScore.MatchingPairs} matching pairs");
    }

    [Fact]
    public void Recognize_CrossUnderRigidMotion_ReturnsCross()
    {
        var vocab = Vocabulary();
        var rec = LearnBoth(vocab);

        var pose = Pose3D.FromYawPitchRoll(-50f, 30f, 10f).Compose(Pose3D.FromTranslation(-2f, 7f, -4f));
        var moved = Cross(vocab).Select(p => p.Transformed(pose)).ToList();
        Assert.Equal(CrossId, rec.Recognize(moved));
    }
}

/// <summary>Tests for the real-sensor environment (ImageSensorEnvironment): a patch
/// sensor scanning a grayscale image on a clamped grid, emitting orientation-histogram
/// feature SDRs and grid-position location SDRs (Leadholm et al. 2025, P9 §3).</summary>
public class BrainImageSensorTests
{
    private const int ImageSide = 64;
    private const int BarId = 0;
    private const int StripeId = 1;

    // Full-image patterns (rather than a single bar on a black background) so that every
    // 16×16 patch contains oriented structure — an all-black patch would fall back to the
    // same uniform-share SDR in both images and mask the orientation difference.

    /// <summary>Bright horizontal bars, 4 px thick: every patch is dominated by
    /// vertical gradients (axis-aligned orientation).</summary>
    private static float[,] AxisAlignedBars()
    {
        var img = new float[ImageSide, ImageSide];
        for (int r = 0; r < ImageSide; r++)
            for (int c = 0; c < ImageSide; c++)
                img[r, c] = (r / 4) % 2 == 0 ? 1f : 0f;
        return img;
    }

    /// <summary>Bright 45° diagonal stripes, 4 px thick: every patch is dominated by
    /// diagonal gradients.</summary>
    private static float[,] DiagonalStripes()
    {
        var img = new float[ImageSide, ImageSide];
        for (int r = 0; r < ImageSide; r++)
            for (int c = 0; c < ImageSide; c++)
                img[r, c] = ((r + c) / 4) % 2 == 0 ? 1f : 0f;
        return img;
    }

    private static ImageSensorEnvironment BuildEnvironment()
    {
        var env = new ImageSensorEnvironment();   // defaults: patch 16, stride 8, seed 42
        env.AddObject(BarId, AxisAlignedBars());
        env.AddObject(StripeId, DiagonalStripes());
        return env;
    }

    /// <summary>Drives the sensor to the top-left cell (the clamped borders make the
    /// overshoot safe), then boustrophedon-walks the whole grid with MoveRight/MoveDown,
    /// collecting the feature SDR at every cell in a deterministic order.</summary>
    private static List<Sdr> WalkFeatures(ImageSensorEnvironment env, int objectId)
    {
        env.PresentObject(objectId);
        for (int i = 0; i <= env.GridWidth; i++) env.MoveTo(ImageSensorEnvironment.MoveLeft);
        for (int i = 0; i <= env.GridHeight; i++) env.MoveTo(ImageSensorEnvironment.MoveUp);

        var features = new List<Sdr>(env.GridWidth * env.GridHeight);
        for (int row = 0; row < env.GridHeight; row++)
        {
            for (int col = 0; col < env.GridWidth; col++)
            {
                features.Add(env.Observe().Feature);
                if (col < env.GridWidth - 1) env.MoveTo(ImageSensorEnvironment.MoveRight);
            }
            if (row < env.GridHeight - 1)
            {
                env.MoveTo(ImageSensorEnvironment.MoveDown);
                for (int back = 0; back < env.GridWidth - 1; back++)
                    env.MoveTo(ImageSensorEnvironment.MoveLeft);
            }
        }
        return features;
    }

    [Fact]
    public void Grid_HasExpectedShape_AndEnvironmentContract()
    {
        var env = BuildEnvironment();

        // 64 px, patch 16, stride 8 → (64 − 16) / 8 + 1 = 7 positions per axis.
        Assert.Equal(7, env.GridWidth);
        Assert.Equal(7, env.GridHeight);
        Assert.Equal(2, env.ObjectCount);
        Assert.True(env.AvailableActions >= 5);
        Assert.Equal(-1, env.CurrentObjectId);   // nothing presented yet

        env.PresentObject(BarId);
        Assert.Equal(BarId, env.CurrentObjectId);
    }

    [Fact]
    public void Observe_TwiceAtSamePosition_YieldsIdenticalSdrs()
    {
        var env = BuildEnvironment();
        env.PresentObject(BarId);

        var first = env.Observe();
        var second = env.Observe();

        Assert.Equal(first.Feature, second.Feature);
        Assert.Equal(first.Location, second.Location);
        Assert.Equal(env.FeatureSize, first.Feature.Size);
        Assert.Equal(env.LocationSize, first.Location.Size);
    }

    [Fact]
    public void MoveRight_ChangesTheLocationSdr()
    {
        var env = BuildEnvironment();
        env.PresentObject(BarId);

        var before = env.Observe().Location;
        int colBefore = env.SensorPosition.Col;
        env.MoveTo(ImageSensorEnvironment.MoveRight);

        Assert.NotEqual(before, env.Observe().Location);
        Assert.Equal(colBefore + 1, env.SensorPosition.Col);
    }

    [Fact]
    public void MoveLeft_AtLeftBorder_ClampsWithoutThrowing()
    {
        var env = BuildEnvironment();
        env.PresentObject(BarId);
        int rowBefore = env.SensorPosition.Row;

        for (int i = 0; i < env.GridWidth + 2; i++)
            env.MoveTo(ImageSensorEnvironment.MoveLeft);

        Assert.Equal(0, env.SensorPosition.Col);
        Assert.Equal(rowBefore, env.SensorPosition.Row);
        Assert.Equal(env.FeatureSize, env.Observe().Feature.Size);   // still observable
    }

    [Fact]
    public void PresentObject_RecentresTheSensor()
    {
        var env = BuildEnvironment();
        env.PresentObject(BarId);
        env.MoveTo(ImageSensorEnvironment.MoveRight);
        env.MoveTo(ImageSensorEnvironment.MoveDown);

        env.PresentObject(BarId);

        Assert.Equal(env.GridHeight / 2, env.SensorPosition.Row);
        Assert.Equal(env.GridWidth / 2, env.SensorPosition.Col);
    }

    [Fact]
    public void MoveRandom_WithSameSeed_IsDeterministic()
    {
        var env1 = BuildEnvironment();
        var env2 = BuildEnvironment();
        env1.PresentObject(BarId);
        env2.PresentObject(BarId);

        for (int step = 0; step < 8; step++)
        {
            env1.MoveTo(ImageSensorEnvironment.MoveRandom);
            env2.MoveTo(ImageSensorEnvironment.MoveRandom);

            Assert.Equal(env1.SensorPosition.Row, env2.SensorPosition.Row);
            Assert.Equal(env1.SensorPosition.Col, env2.SensorPosition.Col);
            Assert.Equal(env1.Observe().Feature, env2.Observe().Feature);
            Assert.Equal(env1.Observe().Location, env2.Observe().Location);
        }
    }

    [Fact]
    public void CentrePatch_BarVersusStripe_HaveLowFeatureOverlap()
    {
        var env = BuildEnvironment();

        env.PresentObject(BarId);
        var bar = env.Observe().Feature;
        env.PresentObject(StripeId);
        var stripe = env.Observe().Feature;
        env.PresentObject(BarId);
        var barAgain = env.Observe().Feature;

        Assert.Equal(bar, barAgain);              // same image, same position → identical
        Assert.True(bar.Cardinality > 0);
        Assert.True(bar.Overlap(stripe) * 2 < bar.Cardinality,
            $"cross-image overlap {bar.Overlap(stripe)} of {bar.Cardinality} active bits");
    }

    [Fact]
    public void AddObject_MismatchedDimensions_Throws()
    {
        var env = BuildEnvironment();
        var ex = Record.Exception(() => env.AddObject(2, new float[32, 48]));
        Assert.NotNull(ex);
    }

    [Fact]
    public void FullBlackImage_ObservesWithoutThrowing()
    {
        var env = new ImageSensorEnvironment();
        env.AddObject(0, new float[ImageSide, ImageSide]);   // all zeros
        env.PresentObject(0);

        var obs = env.Observe();                             // uniform-share fallback
        Assert.Equal(env.FeatureSize, obs.Feature.Size);
        Assert.Equal(env.LocationSize, obs.Location.Size);
    }

    [Fact]
    public void WholeGridFeatures_DifferAcrossImages_MatchWithinAnImage()
    {
        var env = BuildEnvironment();

        var barPass1 = WalkFeatures(env, BarId);
        var barPass2 = WalkFeatures(env, BarId);
        var stripe = WalkFeatures(env, StripeId);

        Assert.Equal(env.GridWidth * env.GridHeight, barPass1.Count);
        Assert.Equal(barPass1.Count, barPass2.Count);
        Assert.Equal(barPass1.Count, stripe.Count);

        // Cheap end-to-end recognisability proof: re-observing the same image reproduces
        // the same features, while the other image's features barely overlap.
        double within = barPass1.Zip(barPass2, (a, b) => (double)a.Overlap(b)).Average();
        double across = barPass1.Zip(stripe, (a, b) => (double)a.Overlap(b)).Average();

        Assert.True(within > 0);
        Assert.True(across * 2 < within,
            $"mean cross-image overlap {across:F2} vs mean within-image overlap {within:F2}");
    }
}
