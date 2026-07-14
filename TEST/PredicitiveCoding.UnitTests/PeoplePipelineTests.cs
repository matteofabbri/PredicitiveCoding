using PredictiveCoding.Imaging;
using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Runtime;
using PredictiveCoding.Vision.Tracking;
using Xunit;

namespace PredictiveCoding.Vision.Tests;

/// <summary>Tests for Phase 6 live runtime: the latest-frame hand-off (drop-under-load)
/// and the capture→detect→track→count pipeline (detect-every-N, metrics).</summary>
public class PeoplePipelineTests
{
    private static Detection Person() => new(100, 100, 150, 200, 0.9f, "person");

    private sealed class FakeDetector(params Detection[] dets) : IPersonDetector
    {
        private readonly IReadOnlyList<Detection> _dets = dets;
        public int Calls { get; private set; }
        public string Name => "fake";
        public IReadOnlyList<Detection> Detect(RgbImage frame) { Calls++; return _dets; }
        public void Dispose() { }
    }

    // ── LatestFrameBuffer ─────────────────────────────────────────────────────

    [Fact]
    public void LatestFrameBuffer_KeepsNewest_AndCountsDrops()
    {
        var dropped = new List<string>();
        var buffer = new LatestFrameBuffer<string>(onDrop: dropped.Add);

        Assert.False(buffer.TryTake(out _));         // empty

        buffer.Publish("a");
        buffer.Publish("b");                          // drops "a"
        buffer.Publish("c");                          // drops "b"

        Assert.Equal(3, buffer.Published);
        Assert.Equal(2, buffer.Drops);
        Assert.Equal(["a", "b"], dropped);

        Assert.True(buffer.TryTake(out string? v));
        Assert.Equal("c", v);                         // only the newest survives
        Assert.False(buffer.TryTake(out _));          // consumed
    }

    // ── Pipeline.Process ──────────────────────────────────────────────────────

    [Fact]
    public void Pipeline_DetectsEveryNthFrame_ButTracksEveryFrame()
    {
        var detector = new FakeDetector(Person());
        using var pipe = new PeopleCountingPipeline(detector, new SortTracker(minHits: 1), detectEveryN: 3);
        using var frame = new RgbImage(64, 48);

        for (int i = 0; i < 9; i++) pipe.Process(frame);

        Assert.Equal(3, detector.Calls);          // detection ran on frames 0, 3, 6
        Assert.Equal(9, pipe.ProcessedFrames);    // tracking ran on all 9
        Assert.Equal(3, pipe.DetectionFrames);
        Assert.Equal(1, pipe.UniqueCount);        // one person, carried between detections
        Assert.Equal(1, pipe.CurrentCount);
    }

    [Fact]
    public void Pipeline_Process_FlagsDetectionFrames()
    {
        var detector = new FakeDetector(Person());
        using var pipe = new PeopleCountingPipeline(detector, new SortTracker(minHits: 1), detectEveryN: 2);
        using var frame = new RgbImage(64, 48);

        var r0 = pipe.Process(frame);
        var r1 = pipe.Process(frame);
        var r2 = pipe.Process(frame);

        Assert.True(r0.Detected);
        Assert.False(r1.Detected);
        Assert.True(r2.Detected);
        Assert.Equal(0, r0.FrameIndex);
        Assert.Equal(2, r2.FrameIndex);
    }

    [Fact]
    public void Pipeline_RejectsNonPositiveDetectInterval()
        => Assert.Throws<ArgumentOutOfRangeException>(
            () => new PeopleCountingPipeline(new FakeDetector(), new SortTracker(), 0));

    // ── Pipeline.RunBlocking (decoupled capture → inference) ──────────────────

    [Fact]
    public void Pipeline_RunBlocking_ConsumesSource_AndReportsCounts()
    {
        var detector = new FakeDetector(Person());
        using var pipe = new PeopleCountingPipeline(detector, new SortTracker(minHits: 1), detectEveryN: 1);

        var frames = Enumerable.Range(0, 8).Select(_ => new RgbImage(32, 24)).ToList();
        int results = 0;
        pipe.RunBlocking(frames, _ => results++);

        Assert.True(pipe.ProcessedFrames >= 1 && pipe.ProcessedFrames <= 8);
        Assert.Equal(pipe.ProcessedFrames, results);   // one callback per processed frame
        Assert.Equal(1, pipe.UniqueCount);             // the same person throughout
    }
}
