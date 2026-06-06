namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// CompositeObject  —  an object defined as an arrangement of child objects.
//
// Holds the list of ObjectParts that make up the object; each part names a child
// (primitive or itself composite) and the pose at which it sits in this object's
// frame. The CompositionLibrary expands a CompositeObject into concrete
// FeaturePoints by composing each part's pose down the tree.
//
// Reference: Hawkins (2021) ch. 6 (compositionality / nested reference frames).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CompositeObject
{
    private readonly List<ObjectPart> _parts = [];

    /// <summary>Identifier of this composite object.</summary>
    public int ObjectId { get; }

    /// <summary>The child parts that constitute this object (read-only view).</summary>
    public IReadOnlyList<ObjectPart> Parts => _parts;

    public CompositeObject(int objectId) => ObjectId = objectId;

    /// <summary>Adds a child object at the given relative pose. Returns this instance
    /// so calls can be chained.</summary>
    public CompositeObject AddPart(int childId, Pose2D poseInParent)
    {
        if (childId == ObjectId)
            throw new ArgumentException("A composite object cannot contain itself.", nameof(childId));
        _parts.Add(new ObjectPart(childId, poseInParent));
        return this;
    }

    /// <summary>Adds a child object at the given translation and optional rotation.</summary>
    public CompositeObject AddPart(int childId, float offsetX, float offsetY, float rotationDeg = 0f)
        => AddPart(childId, new Pose2D(offsetX, offsetY, rotationDeg));
}
