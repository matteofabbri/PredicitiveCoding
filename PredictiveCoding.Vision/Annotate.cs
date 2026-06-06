using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Tracking;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.Vision;

/// <summary>
/// Minimal, dependency-free box overlay for saved frames — draws rectangle
/// outlines directly on the pixels (no font/drawing package needed).
/// </summary>
public static class Annotate
{
    private static readonly Rgb24 DetColor = new(0, 255, 0); // green

    // Per-track palette: same colour ⇒ same tracked person across frames.
    private static readonly Rgb24[] Palette =
    [
        new(255,  64,  64), new( 64, 255,  64), new( 64, 160, 255), new(255, 220,  32),
        new(255,  64, 255), new( 32, 230, 230), new(255, 150,  40), new(170, 110, 255),
    ];

    /// <summary>Draws a green outline around every raw detection (in place).</summary>
    public static void DrawBoxes(Image<Rgb24> image, IReadOnlyList<Detection> detections, int thickness = 2)
    {
        foreach (Detection d in detections)
            DrawRect(image, (int)d.X1, (int)d.Y1, (int)d.X2, (int)d.Y2, thickness, DetColor);
    }

    /// <summary>Draws each tracked person's box in a stable per-id colour (in place).</summary>
    public static void DrawTracks(Image<Rgb24> image, IReadOnlyList<Track> tracks, int thickness = 2)
    {
        foreach (Track t in tracks)
        {
            Rgb24 c = Palette[(t.Id % Palette.Length + Palette.Length) % Palette.Length];
            DrawRect(image, (int)t.X1, (int)t.Y1, (int)t.X2, (int)t.Y2, thickness, c);
        }
    }

    private static void DrawRect(Image<Rgb24> img, int x1, int y1, int x2, int y2, int t, Rgb24 c)
    {
        x1 = Math.Clamp(x1, 0, img.Width  - 1);
        y1 = Math.Clamp(y1, 0, img.Height - 1);
        x2 = Math.Clamp(x2, 0, img.Width  - 1);
        y2 = Math.Clamp(y2, 0, img.Height - 1);
        for (int k = 0; k < t; k++)
        {
            HLine(img, x1, x2, y1 + k, c);
            HLine(img, x1, x2, y2 - k, c);
            VLine(img, y1, y2, x1 + k, c);
            VLine(img, y1, y2, x2 - k, c);
        }
    }

    private static void HLine(Image<Rgb24> img, int xa, int xb, int y, Rgb24 c)
    {
        if (y < 0 || y >= img.Height) return;
        int lo = Math.Max(0, xa), hi = Math.Min(img.Width - 1, xb);
        for (int x = lo; x <= hi; x++) img[x, y] = c;
    }

    private static void VLine(Image<Rgb24> img, int ya, int yb, int x, Rgb24 c)
    {
        if (x < 0 || x >= img.Width) return;
        int lo = Math.Max(0, ya), hi = Math.Min(img.Height - 1, yb);
        for (int y = lo; y <= hi; y++) img[x, y] = c;
    }
}
