namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ObjectPart  —  a child object placed inside a parent object's reference frame.
//
// Pairs a child object id with the relative pose (Pose2D) at which the child's
// own frame sits within the parent. This is the "displacement between sub-objects"
// that turns a flat object into a nested, compositional one: a mug = body + handle
// placed at a displacement; a face = two eyes + a nose + a mouth, each at its pose.
//
// Reference: Hawkins (2021) ch. 6 (objects composed of objects);
// Leadholm et al. (2025) Section 3 (compositional reference frames).
// ─────────────────────────────────────────────────────────────────────────────

public readonly record struct ObjectPart(int ChildId, Pose2D PoseInParent);
