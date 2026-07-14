using PredictiveCoding.FaceRecognition.FaceAlignment;
using PredictiveCoding.Imaging;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// Counts people via their <b>faces</b>, reusing the InsightFace
/// <see cref="RetinaFaceDetector"/> already bundled with the solution. Needs no
/// extra model download beyond the ~16 MB face detector, but only sees faces that
/// roughly point toward the camera. Handy as a zero-config fallback and as a
/// "head count" cross-check for <see cref="YoloPersonDetector"/> in crowds where
/// bodies overlap but heads stay visible.
/// </summary>
public sealed class FaceCountDetector : IPersonDetector
{
    private readonly RetinaFaceDetector _detector;

    /// <inheritdoc />
    public string Name => "RetinaFace (face / head count)";

    public FaceCountDetector(string modelsDir) => _detector = new RetinaFaceDetector(modelsDir);

    /// <inheritdoc />
    public IReadOnlyList<Detection> Detect(RgbImage frame)
    {
        IReadOnlyList<FaceDetection> faces = _detector.DetectAll(frame);
        var result = new List<Detection>(faces.Count);
        foreach (FaceDetection f in faces)
            result.Add(new Detection(f.X1, f.Y1, f.X2, f.Y2, f.Score, "face"));
        return result;
    }

    public void Dispose() => _detector.Dispose();
}
