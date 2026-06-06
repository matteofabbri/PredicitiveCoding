namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// CompositionLibrary  —  registry of primitive and composite objects, plus the
// recursive resolver that flattens a nested object into concrete FeaturePoints.
//
// This is the heart of Phase 8 "objects composed of objects": a primitive is a
// fixed point cloud in its own frame; a composite places child objects at relative
// poses (Pose2D). Resolve walks the tree, composing each part's pose with the
// accumulated transform, so a deeply nested object collapses to a single point
// cloud in the root frame — the same (location, feature) form a CorticalColumn
// already learns from. ToObservations encodes that cloud through a PoseEncoder so
// a composite can be taught to / recognised by the Phase 4-7 machinery unchanged.
//
// Cycles and unbounded recursion are rejected: each Resolve guards the active
// path and a maximum nesting depth.
//
// Reference: Hawkins (2021) ch. 6; Leadholm et al. (2025) Section 3
// (compositional / nested reference frames via displacement).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CompositionLibrary
{
    private readonly Dictionary<int, IReadOnlyList<FeaturePoint>> _primitives = [];
    private readonly Dictionary<int, CompositeObject> _composites = [];
    private readonly int _maxDepth;

    /// <param name="maxDepth">Maximum nesting depth allowed during resolution
    /// (guards against pathological or accidental deep trees).</param>
    public CompositionLibrary(int maxDepth = 32)
    {
        if (maxDepth < 1) throw new ArgumentOutOfRangeException(nameof(maxDepth));
        _maxDepth = maxDepth;
    }

    /// <summary>Number of registered primitive objects.</summary>
    public int PrimitiveCount => _primitives.Count;

    /// <summary>Number of registered composite objects.</summary>
    public int CompositeCount => _composites.Count;

    /// <summary>Registers a primitive object: a fixed cloud of feature points in its
    /// own reference frame. Ids must be unique across primitives and composites.</summary>
    public void AddPrimitive(int id, IReadOnlyList<FeaturePoint> points)
    {
        ArgumentNullException.ThrowIfNull(points);
        EnsureUnused(id);
        _primitives[id] = [.. points];
    }

    /// <summary>Registers a composite object built from already-known child ids.</summary>
    public void AddComposite(CompositeObject composite)
    {
        ArgumentNullException.ThrowIfNull(composite);
        EnsureUnused(composite.ObjectId);
        _composites[composite.ObjectId] = composite;
    }

    /// <summary>Creates, registers, and returns an empty composite with the given id,
    /// ready for <see cref="CompositeObject.AddPart(int, Pose2D)"/> calls.</summary>
    public CompositeObject CreateComposite(int id)
    {
        EnsureUnused(id);
        var composite = new CompositeObject(id);
        _composites[id] = composite;
        return composite;
    }

    /// <summary>True if <paramref name="id"/> is a registered primitive.</summary>
    public bool IsPrimitive(int id) => _primitives.ContainsKey(id);

    /// <summary>True if <paramref name="id"/> is a registered composite.</summary>
    public bool IsComposite(int id) => _composites.ContainsKey(id);

    /// <summary>True if <paramref name="id"/> is known (primitive or composite).</summary>
    public bool Contains(int id) => IsPrimitive(id) || IsComposite(id);

    /// <summary>Flattens <paramref name="id"/> into the concrete feature points it
    /// produces in its own (root) reference frame, expanding every nested child and
    /// composing displacement poses down the tree.</summary>
    /// <exception cref="KeyNotFoundException">An id in the tree is not registered.</exception>
    /// <exception cref="InvalidOperationException">The composition contains a cycle.</exception>
    public IReadOnlyList<FeaturePoint> Resolve(int id)
    {
        var acc = new List<FeaturePoint>();
        ResolveInto(id, Pose2D.Identity, acc, [], 0);
        return acc;
    }

    /// <summary>Encodes the resolved feature points of <paramref name="id"/> into
    /// (Feature, Location) observations ready for <c>CorticalColumn.LearnObject</c> /
    /// <c>Infer</c> — the bridge from Phase 8 composites to the Phase 4-7 columns.
    /// Location is the point's position encoded by <paramref name="pose"/>.</summary>
    public IReadOnlyList<(Sdr Feature, Sdr Location)> ToObservations(int id, PoseEncoder pose)
    {
        ArgumentNullException.ThrowIfNull(pose);
        var points = Resolve(id);
        var obs = new List<(Sdr, Sdr)>(points.Count);
        foreach (var p in points)
            obs.Add((p.Feature, pose.EncodeXY(p.X, p.Y)));
        return obs;
    }

    private void ResolveInto(int id, Pose2D toRoot, List<FeaturePoint> acc, HashSet<int> path, int depth)
    {
        if (depth > _maxDepth)
            throw new InvalidOperationException(
                $"Composition nesting exceeded max depth {_maxDepth} at object {id}.");
        if (!path.Add(id))
            throw new InvalidOperationException($"Cyclic composition detected at object {id}.");

        if (_primitives.TryGetValue(id, out var points))
        {
            foreach (var p in points) acc.Add(p.Transformed(toRoot));
        }
        else if (_composites.TryGetValue(id, out var composite))
        {
            foreach (var part in composite.Parts)
                ResolveInto(part.ChildId, toRoot.Compose(part.PoseInParent), acc, path, depth + 1);
        }
        else
        {
            throw new KeyNotFoundException($"Object id {id} is not a registered primitive or composite.");
        }

        path.Remove(id);
    }

    private void EnsureUnused(int id)
    {
        if (Contains(id))
            throw new ArgumentException($"Object id {id} is already registered.", nameof(id));
    }
}
