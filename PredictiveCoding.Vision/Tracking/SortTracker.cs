using PredictiveCoding.Vision.Detectors;

namespace PredictiveCoding.Vision.Tracking;

/// <summary>
/// Online multi-object tracker in the spirit of <b>SORT</b> (Bewley et al., 2016):
/// predict every track forward, greedily associate detections to predictions by
/// IoU, correct matched tracks, spawn tracks for unmatched detections and retire
/// tracks that go unseen. Turns per-frame detections into stable identities so the
/// same person is not counted twice.
/// <para>
/// Exposes both the <see cref="CurrentCount"/> (people visible right now) and the
/// cumulative <see cref="UniqueCount"/> (distinct people seen over time).
/// </para>
/// </summary>
public sealed class SortTracker
{
    private readonly float _iouThreshold;
    private readonly int   _maxAge;
    private readonly int   _minHits;
    private readonly List<Track> _tracks = [];
    private int _nextId = 1;

    /// <summary>Distinct people confirmed since the tracker was created/reset.</summary>
    public int UniqueCount { get; private set; }

    /// <param name="iouThreshold">Minimum IoU for a detection to match a track.</param>
    /// <param name="maxAge">Frames a track survives unmatched before being dropped (bridges occlusion).</param>
    /// <param name="minHits">Matched frames before a track is confirmed (suppresses one-frame false positives).</param>
    public SortTracker(float iouThreshold = 0.3f, int maxAge = 30, int minHits = 3)
    {
        _iouThreshold = iouThreshold;
        _maxAge       = Math.Max(1, maxAge);
        _minHits      = Math.Max(1, minHits);
    }

    /// <summary>Confirmed tracks matched in the most recent <see cref="Update"/>.</summary>
    public IReadOnlyList<Track> ActiveTracks =>
        _tracks.Where(t => t.Confirmed && t.TimeSinceUpdate == 0).ToList();

    /// <summary>Number of people visible right now (confirmed tracks matched this frame).</summary>
    public int CurrentCount => _tracks.Count(t => t.Confirmed && t.TimeSinceUpdate == 0);

    /// <summary>
    /// Advances the tracker by one frame with the given detections and returns the
    /// confirmed, currently-visible tracks.
    /// </summary>
    public IReadOnlyList<Track> Update(IReadOnlyList<Detection> detections)
    {
        // 1. Predict every existing track forward one frame.
        foreach (Track t in _tracks) t.Predict();

        // 2. Greedy IoU association: strongest overlaps first, each det/track once.
        var pairs = new List<(float Iou, int Det, int Trk)>();
        for (int d = 0; d < detections.Count; d++)
        {
            for (int k = 0; k < _tracks.Count; k++)
            {
                float iou = Nms.Iou(detections[d], _tracks[k].ToDetection());
                if (iou >= _iouThreshold) pairs.Add((iou, d, k));
            }
        }
        pairs.Sort((a, b) => b.Iou.CompareTo(a.Iou));

        var matchedDet = new bool[detections.Count];
        var matchedTrk = new bool[_tracks.Count];
        foreach ((float _, int d, int k) in pairs)
        {
            if (matchedDet[d] || matchedTrk[k]) continue;
            matchedDet[d] = matchedTrk[k] = true;
            _tracks[k].Update(detections[d]);
        }

        // 3. Spawn a new (tentative) track for each unmatched detection.
        for (int d = 0; d < detections.Count; d++)
        {
            if (!matchedDet[d]) _tracks.Add(new Track(_nextId++, detections[d]));
        }

        // 4. Confirm tracks that reached the hit threshold; tally each once.
        foreach (Track t in _tracks)
        {
            if (!t.Confirmed && t.Hits >= _minHits) t.Confirmed = true;
            if (t.Confirmed && !t.CountedUnique)
            {
                t.CountedUnique = true;
                UniqueCount++;
            }
        }

        // 5. Retire tracks unseen for too long.
        _tracks.RemoveAll(t => t.TimeSinceUpdate > _maxAge);

        return ActiveTracks;
    }

    /// <summary>Clears all tracks and counters.</summary>
    public void Reset()
    {
        _tracks.Clear();
        _nextId = 1;
        UniqueCount = 0;
    }
}
