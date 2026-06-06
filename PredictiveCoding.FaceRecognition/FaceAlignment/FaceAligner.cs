using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.FaceRecognition.FaceAlignment;

/// <summary>
/// Aligns a face image to the canonical ArcFace 112 × 112 template using a 5-point
/// <b>similarity transform</b> (rotation + isotropic scale + translation, 4 DOF).
/// <para>
/// The transform is solved in closed form via the least-squares similarity method
/// (Umeyama 1991, simplified for 2D points). The warp is applied by inverse mapping
/// with bilinear interpolation directly on the source image at its original resolution,
/// maximising quality.
/// </para>
/// </summary>
public static class FaceAligner
{
    // ArcFace v2 canonical template for 112 × 112 (InsightFace convention).
    // Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
    private static readonly float[] TemplateX = [38.2946f, 73.5318f, 56.0252f, 41.5493f, 70.7299f];
    private static readonly float[] TemplateY = [51.6963f, 51.5014f, 71.7366f, 92.3655f, 92.2041f];

    // ── Public API ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Warps <paramref name="image"/> to a geometrically aligned <paramref name="outSize"/> × <paramref name="outSize"/>
    /// face image using the detected 5-point <paramref name="landmarks"/>.
    /// The caller owns (and must dispose) the returned image.
    /// </summary>
    /// <param name="image">Source image at its original (any) resolution.</param>
    /// <param name="landmarks">
    /// 10 floats in original image coordinates:
    /// x0,y0, x1,y1, x2,y2, x3,y3, x4,y4
    /// (left_eye, right_eye, nose, left_mouth, right_mouth).
    /// </param>
    /// <param name="outSize">Output side length in pixels (112 for ArcFace).</param>
    public static Image<Rgb24> AlignFace(Image<Rgb24> image, float[] landmarks, int outSize = 112)
    {
        // Source landmarks (detected, original image coords)
        float[] srcX = [landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8]];
        float[] srcY = [landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]];

        // Destination landmarks: reference template scaled to outSize
        float tScale = outSize / 112f;
        float[] dstX = Array.ConvertAll(TemplateX, v => v * tScale);
        float[] dstY = Array.ConvertAll(TemplateY, v => v * tScale);

        // Forward similarity transform: dst = [a, -b; b, a] * src + [tx; ty]
        (float a, float b, float tx, float ty) = SimilarityTransform(srcX, srcY, dstX, dstY);

        // Inverse transform: src = [a, b; -b, a] / det * (dst − [tx; ty])
        float det = a * a + b * b;
        float ia  =  a / (det + 1e-10f);
        float ib  =  b / (det + 1e-10f);
        float itx = -(ia * tx + ib * ty);   // ia·tx + ib·ty
        float ity =  ib * tx - ia * ty;      // −(−ib·tx + ia·ty)

        // Pre-read source pixels for fast random access
        int srcW = image.Width;
        int srcH = image.Height;
        var src = new Rgb24[srcH * srcW];
        image.ProcessPixelRows(acc =>
        {
            for (int r = 0; r < srcH; r++)
                acc.GetRowSpan(r).CopyTo(src.AsSpan(r * srcW, srcW));
        });

        // Inverse warp: for each output pixel, compute source coordinates and sample
        var aligned = new Image<Rgb24>(outSize, outSize);
        aligned.ProcessPixelRows(acc =>
        {
            for (int yd = 0; yd < outSize; yd++)
            {
                // Pre-compute row constants (as xd varies linearly)
                float xs0 = ib * yd + itx;   // xs when xd=0
                float ys0 = ia * yd + ity;    // ys when xd=0

                Span<Rgb24> row = acc.GetRowSpan(yd);
                for (int xd = 0; xd < outSize; xd++)
                {
                    float xs = ia * xd + xs0;
                    float ys = -ib * xd + ys0;
                    row[xd] = BilinearSample(src, srcW, srcH, xs, ys);
                }
            }
        });

        return aligned;
    }

    // ── Similarity transform ──────────────────────────────────────────────────────

    /// <summary>
    /// Closed-form least-squares 2-D similarity transform mapping <c>(srcX,srcY) → (dstX,dstY)</c>.
    /// Returns <c>(a, b, tx, ty)</c> for:
    ///   x' = a·x − b·y + tx
    ///   y' = b·x + a·y + ty
    /// (where s = √(a²+b²) is the scale and θ = atan2(b,a) the rotation angle).
    /// </summary>
    private static (float a, float b, float tx, float ty) SimilarityTransform(
        float[] srcX, float[] srcY, float[] dstX, float[] dstY)
    {
        int n = srcX.Length;

        // Compute means
        float muSx = 0, muSy = 0, muDx = 0, muDy = 0;
        for (int i = 0; i < n; i++)
        {
            muSx += srcX[i]; muSy += srcY[i];
            muDx += dstX[i]; muDy += dstY[i];
        }
        muSx /= n; muSy /= n; muDx /= n; muDy /= n;

        // Normal equations for (a, b) from centered pairs:
        //   a = Σ(sx·dx + sy·dy) / Σ(sx² + sy²)
        //   b = Σ(sx·dy − sy·dx) / Σ(sx² + sy²)
        float numA = 0, numB = 0, denom = 0;
        for (int i = 0; i < n; i++)
        {
            float sx = srcX[i] - muSx, sy = srcY[i] - muSy;
            float dx = dstX[i] - muDx, dy = dstY[i] - muDy;
            numA  += sx * dx + sy * dy;
            numB  += sx * dy - sy * dx;
            denom += sx * sx + sy * sy;
        }

        float a = denom > 1e-10f ? numA / denom : 1f;
        float b = denom > 1e-10f ? numB / denom : 0f;

        // Translation that maps centroid-of-source → centroid-of-dest
        float tx = muDx - (a * muSx - b * muSy);
        float ty = muDy - (b * muSx + a * muSy);

        return (a, b, tx, ty);
    }

    // ── Bilinear sampling ─────────────────────────────────────────────────────────

    private static Rgb24 BilinearSample(Rgb24[] pixels, int width, int height, float x, float y)
    {
        // Out of bounds → black (padding, same as ArcFace training convention)
        if (x < 0 || y < 0 || x > width - 1 || y > height - 1)
            return default;

        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = Math.Min(x0 + 1, width  - 1);
        int y1 = Math.Min(y0 + 1, height - 1);
        float fx = x - x0, fy = y - y0;

        Rgb24 p00 = pixels[y0 * width + x0];
        Rgb24 p10 = pixels[y0 * width + x1];
        Rgb24 p01 = pixels[y1 * width + x0];
        Rgb24 p11 = pixels[y1 * width + x1];

        float w00 = (1f - fx) * (1f - fy);
        float w10 = fx         * (1f - fy);
        float w01 = (1f - fx) * fy;
        float w11 = fx         * fy;

        return new Rgb24(
            (byte)MathF.Round(p00.R * w00 + p10.R * w10 + p01.R * w01 + p11.R * w11),
            (byte)MathF.Round(p00.G * w00 + p10.G * w10 + p01.G * w01 + p11.G * w11),
            (byte)MathF.Round(p00.B * w00 + p10.B * w10 + p01.B * w01 + p11.B * w11));
    }
}
