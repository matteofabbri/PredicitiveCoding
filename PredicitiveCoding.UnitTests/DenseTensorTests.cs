using PredictiveCoding.Core;

namespace PredicitiveCoding.UnitTests;

public class DenseTensorTests
{
    [Fact]
    public void Constructor_Zeros_AllZero()
    {
        var t = new DenseTensor<float>(3, 4);
        Assert.Equal(12, t.Length);
        Assert.All(t.Buffer, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Constructor_InvalidDimension_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenseTensor<float>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenseTensor<float>(-1));
    }

    [Fact]
    public void Constructor_BufferLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DenseTensor<float>(new float[5], 3));
    }

    [Fact]
    public void Clone_IsDeepCopy()
    {
        var original = new DenseTensor<float>(new float[] { 1f, 2f, 3f }, 3);
        var clone = original.Clone();

        clone[0] = 99f;

        Assert.Equal(1f, original[0]);
    }

    [Fact]
    public void Fill_SetsAllElements()
    {
        var t = new DenseTensor<float>(4);
        t.Fill(7f);
        Assert.All(t.Buffer, v => Assert.Equal(7f, v));
    }

    [Fact]
    public void Zero_ClearsAllElements()
    {
        var t = new DenseTensor<float>(new float[] { 1f, 2f, 3f }, 3);
        t.Zero();
        Assert.All(t.Buffer, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Indexer2D_RowMajorAccess()
    {
        // [[1,2],[3,4]]
        var t = new DenseTensor<float>(new float[] { 1, 2, 3, 4 }, 2, 2);
        Assert.Equal(1f, t[0, 0]);
        Assert.Equal(2f, t[0, 1]);
        Assert.Equal(3f, t[1, 0]);
        Assert.Equal(4f, t[1, 1]);
    }

    [Fact]
    public void CopyTo_CopiesData()
    {
        var src = new DenseTensor<float>(new float[] { 5f, 6f }, 2);
        var dst = new DenseTensor<float>(2);

        src.CopyTo(dst);

        Assert.Equal(5f, dst[0]);
        Assert.Equal(6f, dst[1]);
    }

    [Fact]
    public void Like_CreatesSameShapeZeroTensor()
    {
        var t = new DenseTensor<float>(new float[] { 1f, 2f, 3f }, 3);
        var like = DenseTensor<float>.Like(t);
        Assert.Equal(t.Length, like.Length);
        Assert.All(like.Buffer, v => Assert.Equal(0f, v));
    }
}
