namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// PointCloudEnvironment  —  simple IEnvironment backed by explicit point clouds.
//
// Each object is a list of (Feature, Location) SDR pairs.  The sensor visits
// one point at a time; motor commands move it sequentially or at random.
//
// Motor command mapping (see MoveNext / MovePrev / MoveRandom constants):
//   0  MoveNext   — advance one point (wraps around at the end)
//   1  MovePrev   — go back one point  (wraps around at the start)
//   2  MoveRandom — jump to a uniformly random point
//
// Reference: Leadholm et al. (2025) P9 §3 — embodied sensorimotor loop.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Environment where objects are represented as (Feature, Location) point clouds.</summary>
public sealed class PointCloudEnvironment : IEnvironment
{
    private readonly Dictionary<int, List<(Sdr Feature, Sdr Location)>> _objects = [];
    private int _currentObjectId = -1;
    private int _currentPointIndex;
    private readonly Random _rng;

    // ── Motor command constants ──────────────────────────────────────────────

    /// <summary>Motor command: advance the sensor to the next point (wraps).</summary>
    public const int MoveNext = 0;

    /// <summary>Motor command: retreat the sensor to the previous point (wraps).</summary>
    public const int MovePrev = 1;

    /// <summary>Motor command: jump the sensor to a uniformly random point.</summary>
    public const int MoveRandom = 2;

    // ── IEnvironment ─────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public int ObjectCount => _objects.Count;

    /// <inheritdoc/>
    public int CurrentObjectId => _currentObjectId;

    /// <inheritdoc/>
    public int AvailableActions => 3;

    // ── Additional properties ────────────────────────────────────────────────

    /// <summary>Ids of all registered objects, in insertion order.</summary>
    public IReadOnlyList<int> ObjectIds => [.. _objects.Keys];

    /// <summary>Current index of the sensor within the current object's point list.</summary>
    public int CurrentPointIndex => _currentPointIndex;

    // ── Constructor ──────────────────────────────────────────────────────────

    /// <param name="seed">RNG seed for <see cref="MoveRandom"/>. Default 42.</param>
    public PointCloudEnvironment(int seed = 42) => _rng = new Random(seed);

    // ── Object registration ──────────────────────────────────────────────────

    /// <summary>Registers <paramref name="points"/> as the point cloud for
    /// <paramref name="objectId"/>. Overwrites any previously registered cloud
    /// for the same id.</summary>
    public void AddObject(int objectId, IReadOnlyList<(Sdr Feature, Sdr Location)> points)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (points.Count == 0)
            throw new ArgumentException("A point cloud must contain at least one point.", nameof(points));
        _objects[objectId] = [.. points];
    }

    /// <summary>Number of points in the registered point cloud for <paramref name="objectId"/>.</summary>
    public int PointCount(int objectId)
        => _objects.TryGetValue(objectId, out var pts) ? pts.Count : 0;

    // ── IEnvironment ─────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public void PresentObject(int objectId)
    {
        if (!_objects.ContainsKey(objectId))
            throw new KeyNotFoundException(
                $"Object {objectId} is not registered in the environment.");
        _currentObjectId = objectId;
        _currentPointIndex = 0;
    }

    /// <inheritdoc/>
    public (Sdr Feature, Sdr Location) Observe()
    {
        if (_currentObjectId < 0)
            throw new InvalidOperationException(
                "Call PresentObject before Observe.");
        var pts = _objects[_currentObjectId];
        return pts[_currentPointIndex];
    }

    /// <inheritdoc/>
    public void MoveTo(int motorCommand)
    {
        if (_currentObjectId < 0)
            throw new InvalidOperationException(
                "Call PresentObject before MoveTo.");
        if (motorCommand < 0 || motorCommand >= AvailableActions)
            throw new ArgumentOutOfRangeException(nameof(motorCommand),
                $"Motor command must be in [0, {AvailableActions}).");

        var pts = _objects[_currentObjectId];
        _currentPointIndex = motorCommand switch
        {
            MoveNext   => (_currentPointIndex + 1) % pts.Count,
            MovePrev   => (_currentPointIndex - 1 + pts.Count) % pts.Count,
            _          => _rng.Next(pts.Count),   // MoveRandom
        };
    }

    /// <inheritdoc/>
    public void Reset() => _currentPointIndex = 0;
}
