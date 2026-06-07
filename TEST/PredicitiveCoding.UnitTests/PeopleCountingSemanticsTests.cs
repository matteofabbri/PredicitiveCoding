using PredictiveCoding.Vision.Counting;
using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Tracking;
using Xunit;

namespace PredictiveCoding.Vision.Tests;

/// <summary>Tests for Phase 4 counting semantics: line-crossing (LineCounter) and
/// zone occupancy / dwell (ZoneCounter). Positions are scripted for determinism.</summary>
public class PeopleCountingSemanticsTests
{
    // ── LineCounter ───────────────────────────────────────────────────────────

    [Fact]
    public void LineCounter_CountsDirectionalCrossings_AndNets()
    {
        var lc = new LineCounter(0, 100, 200, 100);   // horizontal tripwire at y=100

        lc.Update([(1, 100f, 50f)]);    // above the line — establishes position
        Assert.Equal(0, lc.In);

        lc.Update([(1, 100f, 150f)]);   // crossed downward → "In"
        Assert.Equal(1, lc.In);
        Assert.Equal(0, lc.Out);
        Assert.Equal(1, lc.Net);

        lc.Update([(1, 100f, 40f)]);    // crossed back upward → "Out"
        Assert.Equal(1, lc.Out);
        Assert.Equal(0, lc.Net);
    }

    [Fact]
    public void LineCounter_IgnoresPassesBeyondTheSegmentEnds()
    {
        var lc = new LineCounter(0, 100, 200, 100);   // segment only spans x∈[0,200]

        lc.Update([(7, 300f, 50f)]);    // moving vertically at x=300, off the segment
        lc.Update([(7, 300f, 150f)]);

        Assert.Equal(0, lc.In);
        Assert.Equal(0, lc.Out);
    }

    [Fact]
    public void LineCounter_NoCrossing_LeavesCountsUntouched()
    {
        var lc = new LineCounter(0, 100, 200, 100);
        lc.Update([(1, 50f, 20f)]);
        lc.Update([(1, 60f, 30f)]);     // stays above the line
        Assert.Equal(0, lc.In + lc.Out);
    }

    [Fact]
    public void LineCounter_RejectsDegenerateLine()
        => Assert.Throws<ArgumentException>(() => new LineCounter(10, 10, 10, 10));

    [Fact]
    public void LineCounter_WorksFromTrackerOutput()
    {
        // A person descending through a wide horizontal tripwire, driven by the
        // real tracker, must register exactly one inbound crossing.
        var tracker = new SortTracker(minHits: 2, iouThreshold: 0.2f);
        var lc = new LineCounter(0, 150, 640, 150);

        for (int i = 0; i < 25; i++)
        {
            float cy = 40 + i * 10;                       // 40 → 280, crossing y=150
            var det = new Detection(300, cy - 50, 350, cy + 50, 0.9f, "person");
            lc.Update(tracker.Update([det]));
        }

        Assert.True(lc.In >= 1, $"expected at least one inbound crossing, got In={lc.In}");
        Assert.Equal(0, lc.Out);
    }

    // ── ZoneCounter ───────────────────────────────────────────────────────────

    private static readonly (float, float)[] Square =
        [(100, 100), (300, 100), (300, 300), (100, 300)];

    [Fact]
    public void ZoneCounter_TracksOccupancyEnteredAndDwell()
    {
        var zc = new ZoneCounter(Square);

        zc.Update([(1, 200f, 200f)]);                 // inside
        Assert.Equal(1, zc.Occupancy);
        Assert.Equal(1, zc.Entered);
        Assert.Equal(1, zc.DwellFrames(1));

        zc.Update([(1, 200f, 200f)]);                 // still inside → dwell grows
        Assert.Equal(2, zc.DwellFrames(1));
        Assert.Equal(1, zc.Entered);                  // not double-counted

        zc.Update([(1, 50f, 50f)]);                   // left the zone
        Assert.Equal(0, zc.Occupancy);
        Assert.Equal(1, zc.Entered);
        Assert.Equal(2, zc.DwellFrames(1));           // dwell frozen

        zc.Update([(1, 200f, 200f), (2, 250f, 250f)]); // two inside now
        Assert.Equal(2, zc.Occupancy);
        Assert.Equal(2, zc.Entered);
    }

    [Fact]
    public void ZoneCounter_IsInside_MatchesPolygon()
    {
        var zc = new ZoneCounter(Square);
        Assert.True(zc.IsInside(200, 200));
        Assert.False(zc.IsInside(0, 0));
        Assert.False(zc.IsInside(400, 200));
        Assert.Equal(0, zc.DwellFrames(99));
    }

    [Fact]
    public void ZoneCounter_RejectsDegeneratePolygon()
        => Assert.Throws<ArgumentException>(() => new ZoneCounter([(0, 0), (1, 1)]));
}
