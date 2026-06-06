using PredictiveCoding.Vision.Detectors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.Vision;

/// <summary>
/// Minimal, dependency-free box overlay for saved frames — draws a rectangle
/// outline around each detection directly on the pixels (no font/drawing package).
/// </summary>
public static class Annotate
{
    private static readonly Rgb24 BoxColor = new(0, 255, 0); // green

    /// <summary>Draws a rectangle outline around every detection, modifying <paramref name="image"/> in place.</summary>
    public static void DrawBoxes(Image<Rgb24> image, IReadOnlyList<Detection> detections, int thickness = 2)
    {
        foreach (Detection d in detections)
        {
            int x1 = Math.Clamp((int)d.X1, 0, image.Width  - 1);
            int y1 = Math.Clamp((int)d.Y1, 0, image.Height - 1);
            int x2 = Math.Clamp((int)d.X2, 0, image.Width  - 1);
            int y2 = Math.Clamp((int)d.Y2, 0, image.Height - 1);
            for (int k = 0; k < thickness; k++)
            {
                HLine(image, x1, x2, y1 + k);
                HLine(image, x1, x2, y2 - k);
                VLine(image, y1, y2, x1 + k);
                VLine(image, y1, y2, x2 - k);
            }
        }
    }

    private static void HLine(Image<Rgb24> img, int xa, int xb, int y)
    {
        if (y < 0 || y >= img.Height) return;
        int lo = Math.Max(0, xa), hi = Math.Min(img.Width - 1, xb);
        for (int x = lo; x <= hi; x++) img[x, y] = BoxColor;
    }

    private static void VLine(Image<Rgb24> img, int ya, int yb, int x)
    {
        if (x < 0 || x >= img.Width) return;
        int lo = Math.Max(0, ya), hi = Math.Min(img.Height - 1, yb);
        for (int y = lo; y <= hi; y++) img[x, y] = BoxColor;
    }
}
