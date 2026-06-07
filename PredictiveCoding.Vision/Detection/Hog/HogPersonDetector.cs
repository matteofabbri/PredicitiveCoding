using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// In-house person/object detector: HOG features + a linear model, applied by sliding
/// the detection window over an image pyramid and keeping high-scoring windows after
/// non-maximum suppression. Implements <see cref="IPersonDetector"/> so it is a drop-in
/// alternative to the ONNX <c>YoloPersonDetector</c> — with <b>no model download and no
/// ONNX runtime</b>. Its weights come from <see cref="HogTrainer"/> (which can be fed by
/// the ONNX teachers offline). Not optimised for real-time HD video; it is the
/// transparent, dependency-free baseline.
/// </summary>
public sealed class HogPersonDetector : IPersonDetector
{
    private readonly HogDescriptor _hog;
    private readonly LinearModel _model;
    private readonly float _threshold;
    private readonly float _pyramidScale;
    private readonly int _stride;
    private readonly float _nmsIou;

    public string Name => $"hog-{_model.Label}";

    /// <param name="hog">The descriptor (window size and HOG parameters).</param>
    /// <param name="model">Trained linear model; its dimension must match the descriptor.</param>
    /// <param name="scoreThreshold">Minimum score to emit a detection.</param>
    /// <param name="pyramidScale">Down-scale factor between pyramid levels (&gt;1).</param>
    /// <param name="strideCells">Window stride in HOG cells.</param>
    /// <param name="nmsIou">IoU threshold for non-maximum suppression.</param>
    public HogPersonDetector(HogDescriptor hog, LinearModel model,
        float scoreThreshold = 0f, float pyramidScale = 1.2f, int strideCells = 1, float nmsIou = 0.4f)
    {
        ArgumentNullException.ThrowIfNull(hog);
        ArgumentNullException.ThrowIfNull(model);
        if (model.Dimension != hog.Dimension)
            throw new ArgumentException("Model dimension does not match the HOG descriptor.");
        if (pyramidScale <= 1f) throw new ArgumentOutOfRangeException(nameof(pyramidScale));
        if (strideCells < 1) throw new ArgumentOutOfRangeException(nameof(strideCells));

        _hog = hog;
        _model = model;
        _threshold = scoreThreshold;
        _pyramidScale = pyramidScale;
        _stride = strideCells * hog.CellSize;
        _nmsIou = nmsIou;
    }

    public IReadOnlyList<Detection> Detect(Image<Rgb24> frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var raw = new List<Detection>();

        for (float scale = 1f; ; scale *= _pyramidScale)
        {
            int sw = (int)(frame.Width / scale);
            int sh = (int)(frame.Height / scale);
            if (sw < _hog.WindowWidth || sh < _hog.WindowHeight) break;

            Image<Rgb24> work = scale == 1f ? frame : frame.Clone(c => c.Resize(sw, sh));
            try
            {
                for (int y = 0; y + _hog.WindowHeight <= sh; y += _stride)
                    for (int x = 0; x + _hog.WindowWidth <= sw; x += _stride)
                    {
                        float score = _model.Score(_hog.FromWindow(work, x, y));
                        if (score < _threshold) continue;

                        float ox = x * scale, oy = y * scale;
                        raw.Add(new Detection(
                            ox, oy, ox + _hog.WindowWidth * scale, oy + _hog.WindowHeight * scale,
                            score, _model.Label));
                    }
            }
            finally
            {
                if (!ReferenceEquals(work, frame)) work.Dispose();
            }
        }

        return Nms.Apply(raw, _nmsIou);
    }

    public void Dispose() { }
}
