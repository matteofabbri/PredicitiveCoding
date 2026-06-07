using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.ReId;
using PredictiveCoding.Vision.Tracking;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace PredictiveCoding.Vision.Tests;

/// <summary>Tests for Phase 5 re-identification: the colour-histogram appearance
/// embedder, the cosine PersonGallery, and the re-ID-corrected people counter that
/// dedups the same person returning after an occlusion.</summary>
public class PeopleReidTests
{
    private static readonly Rgb24 Red = new(255, 0, 0);
    private static readonly Rgb24 Blue = new(0, 0, 255);

    private static Detection Box(float x1, float y1, float x2, float y2)
        => new(x1, y1, x2, y2, 0.9f, "person");

    // A frame (black background) with each person's box filled with a solid colour.
    private static Image<Rgb24> Frame(int w, int h, params (Rgb24 Color, Detection Box)[] people)
    {
        var img = new Image<Rgb24>(w, h);
        foreach (var (color, b) in people)
            for (int y = (int)b.Y1; y < (int)b.Y2; y++)
                for (int x = (int)b.X1; x < (int)b.X2; x++)
                    if ((uint)x < (uint)w && (uint)y < (uint)h) img[x, y] = color;
        return img;
    }

    // ── ColorHistogramEmbedder ────────────────────────────────────────────────

    [Fact]
    public void ColorHistogramEmbedder_SameColourMatches_DifferentColourDoesNot()
    {
        var emb = new ColorHistogramEmbedder(binsPerChannel: 8);
        Assert.Equal(24, emb.Dimension);

        var redA = Box(100, 100, 150, 200);
        var redB = Box(400, 250, 450, 350);
        var blue = Box(200, 300, 250, 400);
        using var frame = Frame(640, 480, (Red, redA), (Red, redB), (Blue, blue));

        float[] eRedA = emb.Embed(frame, redA);
        float[] eRedB = emb.Embed(frame, redB);
        float[] eBlue = emb.Embed(frame, blue);

        Assert.True(Cosine(eRedA, eRedB) > 0.95, "two red crops should match");
        Assert.True(Cosine(eRedA, eBlue) < 0.6, "red vs blue should not match");
    }

    // ── PersonGallery ─────────────────────────────────────────────────────────

    [Fact]
    public void PersonGallery_RegistersNew_AndReusesMatchingId()
    {
        var g = new PersonGallery(matchThreshold: 0.6);
        Assert.Null(g.Identify([1f, 0f, 0f]));      // empty gallery

        int id1 = g.IdentifyOrRegister([1f, 0f, 0f]);
        int id1Again = g.IdentifyOrRegister([0.9f, 0.1f, 0f]);   // ~same direction
        int id2 = g.IdentifyOrRegister([0f, 1f, 0f]);           // orthogonal → new

        Assert.Equal(1, id1);
        Assert.Equal(1, id1Again);
        Assert.Equal(2, id2);
        Assert.Equal(2, g.Count);
    }

    // ── ReidPeopleCounter ─────────────────────────────────────────────────────

    [Fact]
    public void ReidPeopleCounter_SamePersonReturning_IsNotDoubleCounted()
    {
        var counter = new ReidPeopleCounter(
            new SortTracker(iouThreshold: 0.2f, maxAge: 3, minHits: 3),
            new ColorHistogramEmbedder(),
            new PersonGallery(0.6));

        var redA = Box(100, 100, 150, 200);
        Feed(counter, 4, () => Frame(640, 480, (Red, redA)), [redA]);
        Assert.Equal(1, counter.UniquePeople);
        Assert.True(counter.CurrentCount >= 1);

        // Person leaves for longer than maxAge → its track retires.
        Feed(counter, 5, () => new Image<Rgb24>(640, 480), []);
        Assert.Equal(0, counter.CurrentCount);

        // Same red person reappears elsewhere → fresh track id, but re-identified.
        var redB = Box(400, 250, 450, 350);
        Feed(counter, 4, () => Frame(640, 480, (Red, redB)), [redB]);
        Assert.Equal(1, counter.UniquePeople);   // still one distinct person

        // A genuinely different (blue) person → distinct.
        Feed(counter, 5, () => new Image<Rgb24>(640, 480), []);
        var blueC = Box(200, 300, 250, 400);
        Feed(counter, 4, () => Frame(640, 480, (Blue, blueC)), [blueC]);
        Assert.Equal(2, counter.UniquePeople);
    }

    [Fact]
    public void ReidPeopleCounter_Reset_ClearsEverything()
    {
        var counter = new ReidPeopleCounter(
            new SortTracker(minHits: 1), new ColorHistogramEmbedder(), new PersonGallery(0.6));
        var red = Box(100, 100, 150, 200);
        Feed(counter, 3, () => Frame(640, 480, (Red, red)), [red]);
        Assert.True(counter.UniquePeople >= 1);

        counter.Reset();
        Assert.Equal(0, counter.UniquePeople);
        Assert.Equal(0, counter.CurrentCount);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void Feed(ReidPeopleCounter counter, int frames, Func<Image<Rgb24>> makeFrame, Detection[] dets)
    {
        for (int i = 0; i < frames; i++)
        {
            using Image<Rgb24> f = makeFrame();
            counter.Update(f, dets);
        }
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        double d = Math.Sqrt(na) * Math.Sqrt(nb);
        return d > 1e-12 ? dot / d : 0;
    }
}
