using System.Diagnostics;
using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Tracking;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.Vision.Runtime;

/// <summary>Outcome of processing one frame.</summary>
public readonly record struct PipelineResult(
    long FrameIndex, bool Detected, IReadOnlyList<Track> Tracks, int CurrentCount, int UniqueCount);

/// <summary>
/// The live people-counting runtime: capture → detect → track → count, wired so the
/// expensive detector runs only every N-th frame while the cheap tracker advances on
/// every frame (it carries the boxes between detections). <see cref="RunBlocking"/>
/// decouples capture from inference through a <see cref="LatestFrameBuffer{T}"/>: a
/// background reader publishes frames and the inference loop always takes the freshest
/// one, dropping anything it could not keep up with — so end-to-end latency stays low
/// under load instead of growing an unbounded backlog. FPS and counts are exposed live.
/// </summary>
public sealed class PeopleCountingPipeline : IDisposable
{
    private readonly IPersonDetector _detector;
    private readonly SortTracker _tracker;
    private readonly int _detectEveryN;
    private readonly Stopwatch _clock = new();
    private IReadOnlyList<Detection> _lastDetections = [];

    /// <summary>Frames processed (detected or tracked).</summary>
    public long ProcessedFrames { get; private set; }

    /// <summary>Frames on which the detector actually ran.</summary>
    public long DetectionFrames { get; private set; }

    /// <summary>People visible right now.</summary>
    public int CurrentCount => _tracker.CurrentCount;

    /// <summary>Distinct people seen over time.</summary>
    public int UniqueCount => _tracker.UniqueCount;

    /// <summary>Wall-clock seconds since the first processed frame.</summary>
    public double ElapsedSeconds => _clock.Elapsed.TotalSeconds;

    /// <summary>Throughput in processed frames per second.</summary>
    public double Fps => ElapsedSeconds > 1e-6 ? ProcessedFrames / ElapsedSeconds : 0;

    /// <param name="detector">Person detector (the bottleneck; runs every N-th frame).</param>
    /// <param name="tracker">Tracker advanced on every frame.</param>
    /// <param name="detectEveryN">Run detection on 1 of every N frames (≥1).</param>
    public PeopleCountingPipeline(IPersonDetector detector, SortTracker tracker, int detectEveryN = 1)
    {
        ArgumentNullException.ThrowIfNull(detector);
        ArgumentNullException.ThrowIfNull(tracker);
        if (detectEveryN < 1) throw new ArgumentOutOfRangeException(nameof(detectEveryN));
        _detector = detector;
        _tracker = tracker;
        _detectEveryN = detectEveryN;
    }

    /// <summary>Processes a single frame: detects on every N-th call, always tracks,
    /// and returns the per-frame result. The frame is not disposed.</summary>
    public PipelineResult Process(Image<Rgb24> frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (!_clock.IsRunning) _clock.Start();

        bool detected = ProcessedFrames % _detectEveryN == 0;
        if (detected)
        {
            _lastDetections = _detector.Detect(frame);
            DetectionFrames++;
        }

        IReadOnlyList<Track> tracks = _tracker.Update(_lastDetections);
        long index = ProcessedFrames++;
        return new PipelineResult(index, detected, tracks, _tracker.CurrentCount, _tracker.UniqueCount);
    }

    /// <summary>Runs the pipeline over a frame source until it ends or
    /// <paramref name="cancel"/> fires. Capture (reading <paramref name="frames"/>) runs
    /// on a background thread decoupled from inference; frames the inference loop cannot
    /// keep up with are dropped. Every consumed frame is disposed.</summary>
    public void RunBlocking(
        IEnumerable<Image<Rgb24>> frames,
        Action<PipelineResult>? onResult = null,
        CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(frames);
        using var buffer = new LatestFrameBuffer<Image<Rgb24>>(onDrop: static f => f.Dispose());
        bool producerDone = false;

        var producer = Task.Run(() =>
        {
            try
            {
                foreach (Image<Rgb24> f in frames)
                {
                    if (cancel.IsCancellationRequested) { f.Dispose(); break; }
                    buffer.Publish(f);
                }
            }
            finally { Volatile.Write(ref producerDone, true); }
        }, CancellationToken.None);

        while (!cancel.IsCancellationRequested)
        {
            if (buffer.TryTake(out Image<Rgb24>? frame))
            {
                PipelineResult result = Process(frame);
                frame.Dispose();
                onResult?.Invoke(result);
            }
            else if (Volatile.Read(ref producerDone))
            {
                break;
            }
            else
            {
                Thread.Sleep(1);
            }
        }

        try { producer.Wait(CancellationToken.None); }
        catch (AggregateException) { /* surfaced below */ }

        buffer.Clear();   // dispose any frame left after a cancellation
        if (producer.IsFaulted)
            throw producer.Exception!.InnerException ?? producer.Exception!;
    }

    /// <summary>Disposes the underlying detector.</summary>
    public void Dispose() => _detector.Dispose();
}
