using PredictiveCoding.Onnx;

namespace PredictiveCoding.Onnx.Tests;

/// <summary>Nearest-neighbour Resize — the YOLOX FPN upsample configuration
/// (asymmetric + floor) plus the half_pixel variant and the guard rails.</summary>
public class OnnxResizeTests
{
    [Fact]
    public void Upsample2x_AsymmetricFloor_RepeatsPixels()
    {
        // 1×1×2×2 → 1×1×4×4, scales [1,1,2,2]: srcY = floor(oy/2), srcX = floor(ox/2)
        // → every source pixel becomes a 2×2 block.
        var x = new Tensor([1, 1, 2, 2], [1f, 2f, 3f, 4f]);
        var scales = new Tensor([4], [1f, 1f, 2f, 2f]);

        var y = Ops.Resize(x, scales, sizes: null, "nearest", "asymmetric", "floor");

        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Shape);
        float[] expected =
        {
            1, 1, 2, 2,
            1, 1, 2, 2,
            3, 3, 4, 4,
            3, 3, 4, 4,
        };
        Assert.Equal(expected, y.Data);
    }

    [Fact]
    public void Downsample_HalfPixel_PicksNearestCentre()
    {
        // 1×1×4×1 → 1×1×2×1, scale 0.5 on H: x_orig = (i+0.5)/0.5 − 0.5 = 2i + 0.5,
        // round_prefer_floor → ceil(orig − 0.5) = 2i → rows 0 and 2.
        var x = new Tensor([1, 1, 4, 1], [10f, 20f, 30f, 40f]);
        var scales = new Tensor([4], [1f, 1f, 0.5f, 1f]);

        var y = Ops.Resize(x, scales, sizes: null, "nearest", "half_pixel", "round_prefer_floor");

        Assert.Equal(new[] { 1, 1, 2, 1 }, y.Shape);
        Assert.Equal(new[] { 10f, 30f }, y.Data);
    }

    [Fact]
    public void SizesInput_WinsOverScales_AndKeepsNC()
    {
        var x = new Tensor([1, 1, 2, 2], [1f, 2f, 3f, 4f]);
        var sizes = new Tensor([4], new long[] { 1, 1, 4, 4 });

        var y = Ops.Resize(x, scales: null, sizes, "nearest", "asymmetric", "floor");

        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Shape);
    }

    [Fact]
    public void UnsupportedMode_Throws()
    {
        var x = new Tensor([1, 1, 2, 2], [1f, 2f, 3f, 4f]);
        var scales = new Tensor([4], [1f, 1f, 2f, 2f]);

        Assert.Throws<NotSupportedException>(
            () => Ops.Resize(x, scales, null, "cubic", "asymmetric", "floor"));
        Assert.Throws<NotSupportedException>(          // changing C is not allowed
            () => Ops.Resize(x, new Tensor([4], [1f, 2f, 2f, 2f]), null, "nearest", "asymmetric", "floor"));
    }
}
