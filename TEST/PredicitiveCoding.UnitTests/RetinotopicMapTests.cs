using PredictiveCoding.Core;
using PredictiveCoding.Cortex;

namespace PredicitiveCoding.UnitTests;

public class RetinotopicMapTests
{
    private static DenseTensor<float> Ramp(int c, int h, int w)
    {
        var t = new DenseTensor<float>(c, h, w);
        for (int i = 0; i < t.Length; i++) t.Buffer[i] = i;
        return t;
    }

    [Fact]
    public void Geometry_IsComputedCorrectly()
    {
        var map = new RetinotopicMap(channels: 3, height: 32, width: 32, patchSize: 8, stride: 4);
        Assert.Equal(7, map.GridHeight);          // (32-8)/4 + 1
        Assert.Equal(7, map.GridWidth);
        Assert.Equal(49, map.PatchCount);
        Assert.Equal(3 * 8 * 8, map.PatchDim);
    }

    [Fact]
    public void Extract_ReturnsOnePatchPerColumn_WithPatchDim()
    {
        var map     = new RetinotopicMap(1, 4, 4, patchSize: 2, stride: 2);
        var patches = map.Extract(Ramp(1, 4, 4));

        Assert.Equal(map.PatchCount, patches.Length);
        Assert.All(patches, p => Assert.Equal(map.PatchDim, p.Length));
    }

    [Fact]
    public void ExtractThenAssemble_NonOverlapping_RoundTripsExactly()
    {
        // stride == patchSize and dims divisible ⇒ full cover, no overlap ⇒ identity.
        var map   = new RetinotopicMap(1, 4, 4, patchSize: 2, stride: 2);
        var image = Ramp(1, 4, 4);

        var recon = map.Assemble(map.Extract(image));

        Assert.Equal(image.Length, recon.Length);
        for (int i = 0; i < image.Length; i++)
            Assert.Equal(image.Buffer[i], recon.Buffer[i], precision: 4);
    }

    [Fact]
    public void Extract_WrongImageSize_Throws()
    {
        var map = new RetinotopicMap(1, 4, 4, 2, 2);
        Assert.Throws<ArgumentException>(() => map.Extract(new DenseTensor<float>(10)));
    }
}
