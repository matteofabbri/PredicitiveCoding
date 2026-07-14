using PredictiveCoding.Imaging;
using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Tracking;

namespace PredictiveCoding.Vision.ReId;

/// <summary>
/// Re-identification-corrected people counter. Wraps a <see cref="SortTracker"/> with
/// an appearance gallery so that a person who leaves and returns — getting a fresh
/// track id from the memoryless tracker — is recognised as the <b>same</b> person and
/// not counted twice. Each newly confirmed track is embedded once
/// (<see cref="IAppearanceEmbedder"/>) and matched against the <see cref="PersonGallery"/>;
/// <see cref="UniquePeople"/> is the number of distinct people ever seen, which is more
/// robust than the tracker's own occlusion-sensitive unique count.
/// </summary>
public sealed class ReidPeopleCounter
{
    private readonly SortTracker _tracker;
    private readonly IAppearanceEmbedder _embedder;
    private readonly PersonGallery _gallery;
    private readonly Dictionary<int, int> _trackToGlobal = [];

    public ReidPeopleCounter(SortTracker tracker, IAppearanceEmbedder embedder, PersonGallery gallery)
    {
        ArgumentNullException.ThrowIfNull(tracker);
        ArgumentNullException.ThrowIfNull(embedder);
        ArgumentNullException.ThrowIfNull(gallery);
        _tracker = tracker;
        _embedder = embedder;
        _gallery = gallery;
    }

    /// <summary>People visible right now (confirmed tracks matched this frame).</summary>
    public int CurrentCount => _tracker.CurrentCount;

    /// <summary>Distinct people seen over time, deduplicated by appearance.</summary>
    public int UniquePeople => _gallery.Count;

    /// <summary>Advances by one frame: tracks the detections, then embeds and
    /// re-identifies each newly confirmed track. Returns the confirmed visible tracks.</summary>
    public IReadOnlyList<Track> Update(RgbImage frame, IReadOnlyList<Detection> detections)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(detections);

        IReadOnlyList<Track> active = _tracker.Update(detections);
        foreach (Track t in active)
        {
            if (_trackToGlobal.ContainsKey(t.Id)) continue;
            float[] embedding = _embedder.Embed(frame, t.ToDetection());
            _trackToGlobal[t.Id] = _gallery.IdentifyOrRegister(embedding);
        }
        return active;
    }

    /// <summary>The deduplicated global person id for a track, or null if unseen.</summary>
    public int? GlobalId(int trackId)
        => _trackToGlobal.TryGetValue(trackId, out int g) ? g : null;

    /// <summary>Clears the tracker, the gallery and the track→person mapping.</summary>
    public void Reset()
    {
        _tracker.Reset();
        _gallery.Reset();
        _trackToGlobal.Clear();
    }
}
