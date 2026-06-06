namespace PredictiveCoding.FaceRecognition.FaceAlignment;

/// <summary>
/// Result of a single face detection from <see cref="RetinaFaceDetector"/>.
/// All coordinates are in <b>original image</b> space (pixels).
/// </summary>
public sealed class FaceDetection
{
    /// <summary>Left edge of the detected bounding box.</summary>
    public float X1 { get; }
    /// <summary>Top edge of the detected bounding box.</summary>
    public float Y1 { get; }
    /// <summary>Right edge of the detected bounding box.</summary>
    public float X2 { get; }
    /// <summary>Bottom edge of the detected bounding box.</summary>
    public float Y2 { get; }

    /// <summary>Detector confidence score in [0, 1].</summary>
    public float Score { get; }

    /// <summary>
    /// 5-point facial landmarks in original image coordinates:
    /// <c>[x0,y0, x1,y1, x2,y2, x3,y3, x4,y4]</c>
    /// — left eye, right eye, nose tip, left mouth corner, right mouth corner.
    /// </summary>
    public float[] Landmarks { get; }

    public float Width  => X2 - X1;
    public float Height => Y2 - Y1;
    public float Area   => Width * Height;

    public FaceDetection(float x1, float y1, float x2, float y2, float score, float[] landmarks)
    {
        X1 = x1; Y1 = y1; X2 = x2; Y2 = y2;
        Score = score;
        Landmarks = landmarks;
    }
}
