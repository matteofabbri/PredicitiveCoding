using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// Finds people — or a proxy such as visible faces — in a single frame.
/// Implementations are interchangeable so the counting pipeline does not care
/// which underlying model produced the boxes.
/// </summary>
public interface IPersonDetector : IDisposable
{
    /// <summary>Human-readable name, shown in the demo UI.</summary>
    string Name { get; }

    /// <summary>
    /// Returns every person detected in <paramref name="frame"/> (highest score
    /// first). The frame is read only and is <b>not</b> disposed by the detector.
    /// </summary>
    IReadOnlyList<Detection> Detect(Image<Rgb24> frame);
}
