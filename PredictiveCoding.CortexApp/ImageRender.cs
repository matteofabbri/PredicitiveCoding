using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Xaml.Media.Imaging;
using PredictiveCoding.Core;

namespace PredictiveCoding.CortexApp;

/// <summary>Shared helpers to render a [C,H,W] tensor to a WriteableBitmap.</summary>
public static class ImageRender
{
    /// <summary>
    /// Renders a [C,H,W] tensor min-max normalised to [0,1].  Grayscale (C==1) is
    /// broadcast to RGB; C&gt;=3 uses the first three channels.
    /// </summary>
    public static WriteableBitmap ToBitmap(DenseTensor<float> img)
    {
        int c = img.Dimensions[0];
        int h = img.Dimensions[1];
        int w = img.Dimensions[2];

        float min = float.MaxValue, max = float.MinValue;
        foreach (float v in img.Buffer)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float range = max - min;
        float scale = range > 1e-6f ? 1f / range : 0f;

        var wb    = new WriteableBitmap(w, h);
        var bgra  = new byte[w * h * 4];
        int plane = h * w;

        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int pix = y * w + x;
            float r = Norm(img.Buffer[pix], min, scale);
            float g = c > 1 ? Norm(img.Buffer[plane + pix], min, scale) : r;
            float b = c > 2 ? Norm(img.Buffer[2 * plane + pix], min, scale) : r;

            int o = pix * 4;
            bgra[o + 0] = (byte)(b * 255f);
            bgra[o + 1] = (byte)(g * 255f);
            bgra[o + 2] = (byte)(r * 255f);
            bgra[o + 3] = 255;
        }

        using (var stream = wb.PixelBuffer.AsStream())
            stream.Write(bgra, 0, bgra.Length);
        wb.Invalidate();
        return wb;
    }

    private static float Norm(float v, float min, float scale)
    {
        float n = (v - min) * scale;
        return n < 0f ? 0f : n > 1f ? 1f : n;
    }
}
