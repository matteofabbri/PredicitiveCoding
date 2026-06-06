namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// Intersection-over-union and greedy non-maximum suppression — the same
/// approach used by <c>RetinaFaceDetector</c>, lifted into a shared helper so
/// every detector decodes its raw boxes the same way.
/// </summary>
public static class Nms
{
    /// <summary>Intersection-over-union of two axis-aligned boxes, in [0, 1].</summary>
    public static float Iou(in Detection a, in Detection b)
    {
        float ix1 = MathF.Max(a.X1, b.X1), iy1 = MathF.Max(a.Y1, b.Y1);
        float ix2 = MathF.Min(a.X2, b.X2), iy2 = MathF.Min(a.Y2, b.Y2);
        float inter = MathF.Max(0f, ix2 - ix1) * MathF.Max(0f, iy2 - iy1);
        float union = a.Area + b.Area - inter;
        return union > 0f ? inter / union : 0f;
    }

    /// <summary>
    /// Greedy NMS: sorts <paramref name="boxes"/> by score descending, then keeps
    /// each box unless it overlaps an already-kept box by more than
    /// <paramref name="iouThreshold"/>. The result stays in score order.
    /// </summary>
    public static List<Detection> Apply(List<Detection> boxes, float iouThreshold)
    {
        boxes.Sort((a, b) => b.Score.CompareTo(a.Score));
        var kept = new List<Detection>();
        foreach (Detection d in boxes)
        {
            bool suppressed = false;
            foreach (Detection k in kept)
            {
                if (Iou(d, k) > iouThreshold) { suppressed = true; break; }
            }
            if (!suppressed) kept.Add(d);
        }
        return kept;
    }
}
