using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Tracking;
using Xunit;

namespace PredictiveCoding.Vision.Tests;

/// <summary>
/// Tests for the people-counting tracker (<see cref="SortTracker"/>): confirmation,
/// unique vs current counts, ID stability under motion, occlusion bridging and
/// false-positive suppression. Detections are scripted so the assertions are
/// deterministic (no model / camera needed).
/// </summary>
public class PeopleTrackingTests
{
    // A person box centred at (cx, cy).
    private static Detection Box(float cx, float cy, float w = 50, float h = 100) =>
        new(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9f, "person");

    [Fact]
    public void SteadyPerson_IsConfirmedAndCountedOnce()
    {
        var t = new SortTracker(iouThreshold: 0.3f, maxAge: 30, minHits: 3);
        for (int i = 0; i < 5; i++) t.Update([Box(100, 100)]);

        Assert.Equal(1, t.UniqueCount);
        Assert.Equal(1, t.CurrentCount);
    }

    [Fact]
    public void TwoSeparatePeople_AreCountedAsTwo()
    {
        var t = new SortTracker(minHits: 3);
        for (int i = 0; i < 5; i++) t.Update([Box(100, 100), Box(400, 300)]);

        Assert.Equal(2, t.UniqueCount);
        Assert.Equal(2, t.CurrentCount);
    }

    [Fact]
    public void MovingPerson_KeepsOneIdentityAndIsNotDoubleCounted()
    {
        var t = new SortTracker(minHits: 3);
        int? firstId = null;
        for (int i = 0; i < 10; i++)
        {
            IReadOnlyList<Track> active = t.Update([Box(100 + i * 10, 200)]); // drifting right
            if (active.Count == 1) firstId ??= active[0].Id;
        }

        Assert.Equal(1, t.UniqueCount);
        Assert.Single(t.ActiveTracks);
        Assert.Equal(firstId, t.ActiveTracks[0].Id);   // same ID throughout
    }

    [Fact]
    public void ShortOcclusion_DoesNotCreateNewIdentity()
    {
        var t = new SortTracker(maxAge: 10, minHits: 3);
        for (int i = 0; i < 5; i++) t.Update([Box(200, 200)]);  // present → confirmed
        Assert.Equal(1, t.UniqueCount);

        t.Update([]);                                            // missed two frames…
        t.Update([]);                                            // …but within maxAge
        for (int i = 0; i < 3; i++) t.Update([Box(200, 200)]);  // reappears at the same spot

        Assert.Equal(1, t.UniqueCount);   // still the same person
        Assert.Equal(1, t.CurrentCount);
    }

    [Fact]
    public void LongAbsence_RetiresTrack_AndReentryCountsAsNew()
    {
        var t = new SortTracker(maxAge: 2, minHits: 1);
        for (int i = 0; i < 3; i++) t.Update([Box(200, 200)]); // person A
        Assert.Equal(1, t.UniqueCount);

        for (int i = 0; i < 4; i++) t.Update([]);              // gone longer than maxAge → dropped
        Assert.Equal(0, t.CurrentCount);

        for (int i = 0; i < 3; i++) t.Update([Box(200, 200)]); // someone shows up again
        Assert.Equal(2, t.UniqueCount);                        // memoryless tracker → new identity
    }

    [Fact]
    public void BriefDetectionBelowMinHits_IsNotCounted()
    {
        var t = new SortTracker(minHits: 3);
        t.Update([Box(100, 100)]);
        t.Update([Box(100, 100)]);   // only two hits < minHits

        Assert.Equal(0, t.UniqueCount);
        Assert.Equal(0, t.CurrentCount);
    }

    [Fact]
    public void Tracker_FollowsConstantVelocityTarget()
    {
        var t = new SortTracker(minHits: 1, iouThreshold: 0.2f);
        IReadOnlyList<Track> active = [];
        for (int i = 0; i < 12; i++) active = t.Update([Box(100 + i * 10, 150)]);

        Assert.Single(active);
        float expected = 100 + 11 * 10;
        Assert.True(Math.Abs(active[0].CenterX - expected) < 12f,
            $"centerX={active[0].CenterX:F1}, expected≈{expected}");
    }

    [Fact]
    public void Reset_ClearsTracksAndCounters()
    {
        var t = new SortTracker(minHits: 1);
        for (int i = 0; i < 3; i++) t.Update([Box(100, 100)]);
        Assert.Equal(1, t.UniqueCount);

        t.Reset();
        Assert.Equal(0, t.UniqueCount);
        Assert.Equal(0, t.CurrentCount);
        Assert.Empty(t.ActiveTracks);
    }
}
