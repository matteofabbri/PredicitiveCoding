namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// One object detected in a frame. All coordinates are in <b>original-image</b>
/// pixel space (the same space the frame was supplied in).
/// </summary>
public readonly record struct Detection(
    float X1, float Y1, float X2, float Y2, float Score, string Label)
{
    /// <summary>Box width in pixels.</summary>
    public float Width => X2 - X1;
    /// <summary>Box height in pixels.</summary>
    public float Height => Y2 - Y1;
    /// <summary>Box area in pixels².</summary>
    public float Area => Width * Height;
    /// <summary>Horizontal centre of the box.</summary>
    public float CenterX => (X1 + X2) * 0.5f;
    /// <summary>Vertical centre of the box.</summary>
    public float CenterY => (Y1 + Y2) * 0.5f;
}
