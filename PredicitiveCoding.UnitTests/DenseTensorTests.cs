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
        var original = new DenseTensor<float>([1f, 2f, 3f], 3);
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
        var t = new DenseTensor<float>([1f, 2f, 3f], 3);
        t.Zero();
        Assert.All(t.Buffer, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Indexer2D_RowMajorAccess()
    {
        // [[1,2],[3,4]]
        var t = new DenseTensor<float>([1, 2, 3, 4], 2, 2);
        Assert.Equal(1f, t[0, 0]);
        Assert.Equal(2f, t[0, 1]);
        Assert.Equal(3f, t[1, 0]);
        Assert.Equal(4f, t[1, 1]);
    }

    [Fact]
    public void Indexer3D_RowMajorAccess()
    {
        // Shape [2, 2, 2]: values 0..7
        var data = new float[] { 0, 1, 2, 3, 4, 5, 6, 7 };
        var t = new DenseTensor<float>(data, 2, 2, 2);

        Assert.Equal(0f, t[0, 0, 0]);
        Assert.Equal(1f, t[0, 0, 1]);
        Assert.Equal(2f, t[0, 1, 0]);
        Assert.Equal(3f, t[0, 1, 1]);
        Assert.Equal(4f, t[1, 0, 0]);
        Assert.Equal(7f, t[1, 1, 1]);
    }

    [Fact]
    public void Indexer3D_SetValue()
    {
        var t = new DenseTensor<float>(2, 3, 4);
        t[1, 2, 3] = 42f;
        Assert.Equal(42f, t[1, 2, 3]);
        Assert.Equal(42f, t.Buffer[1 * (3 * 4) + 2 * 4 + 3]);
    }

    [Fact]
    public void IndexerND_FourDimensions_RowMajorAccess()
    {
        // Shape [2, 3, 4, 5]: flat index of [1,2,3,4] = 1*60 + 2*20 + 3*5 + 4 = 119
        var t = new DenseTensor<float>(2, 3, 4, 5);
        t.Buffer[119] = 99f;
        Assert.Equal(99f, t[[1, 2, 3, 4]]);
    }

    [Fact]
    public void IndexerND_SetValue_FiveDimensions()
    {
        // Shape [2, 2, 2, 2, 2] = 32 elements; [1,1,1,1,1] = flat 31
        var t = new DenseTensor<float>(2, 2, 2, 2, 2);
        t[[1, 1, 1, 1, 1]] = 7f;
        Assert.Equal(7f, t.Buffer[31]);
    }

    [Fact]
    public void IndexerND_WrongRank_Throws()
    {
        var t = new DenseTensor<float>(2, 3, 4);
        Assert.Throws<ArgumentException>(() => t[[0, 1]]);
        Assert.Throws<ArgumentException>(() => t[[0, 1, 2, 3]]);
    }

    [Fact]
    public void CopyTo_CopiesData()
    {
        var src = new DenseTensor<float>([5f, 6f], 2);
        var dst = new DenseTensor<float>(2);

        src.CopyTo(dst);

        Assert.Equal(5f, dst[0]);
        Assert.Equal(6f, dst[1]);
    }

    [Fact]
    public void Like_CreatesSameShapeZeroTensor()
    {
        var t = new DenseTensor<float>([1f, 2f, 3f], 3);
        var like = DenseTensor<float>.Like(t);
        Assert.Equal(t.Length, like.Length);
        Assert.All(like.Buffer, v => Assert.Equal(0f, v));
    }
}
