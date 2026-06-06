using PredictiveCoding.Core;
using PredictiveCoding.Data.Datasets;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the MNIST IDX parsers: build tiny IDX blobs in memory and
/// parse them. No download is performed.
/// </summary>
public sealed class MnistDatasetTests
{
    private static void WriteBigEndian(byte[] buffer, int offset, int value)
    {
        buffer[offset] = (byte)(value >> 24);
        buffer[offset + 1] = (byte)(value >> 16);
        buffer[offset + 2] = (byte)(value >> 8);
        buffer[offset + 3] = (byte)value;
    }

    private static byte[] BuildImages(int count, int rows, int cols, byte[] pixels)
    {
        var buffer = new byte[16 + pixels.Length];
        WriteBigEndian(buffer, 0, 0x00000803);
        WriteBigEndian(buffer, 4, count);
        WriteBigEndian(buffer, 8, rows);
        WriteBigEndian(buffer, 12, cols);
        pixels.CopyTo(buffer, 16);
        return buffer;
    }

    private static byte[] BuildLabels(int count, byte[] labels)
    {
        var buffer = new byte[8 + labels.Length];
        WriteBigEndian(buffer, 0, 0x00000801);
        WriteBigEndian(buffer, 4, count);
        labels.CopyTo(buffer, 8);
        return buffer;
    }

    [Fact]
    public void ParseImages_ReadsShapeAndNormalises()
    {
        byte[] pixels = [0, 255, 128, 64, 32, 16, 200, 0]; // 2 images, 2x2
        byte[] idx = BuildImages(count: 2, rows: 2, cols: 2, pixels);

        DenseTensor<float> images = MnistDataset.ParseImages(idx);

        Assert.Equal(new[] { 2, 2, 2 }, images.Dimensions);
        Assert.Equal(0f, images.Buffer[0]);
        Assert.Equal(1f, images.Buffer[1]); // 255 / 255
    }

    [Fact]
    public void ParseImages_WithoutNormalise_KeepsByteValues()
    {
        byte[] idx = BuildImages(1, 1, 2, [10, 250]);
        DenseTensor<float> images = MnistDataset.ParseImages(idx, normalize: false);
        Assert.Equal(10f, images.Buffer[0]);
        Assert.Equal(250f, images.Buffer[1]);
    }

    [Fact]
    public void ParseLabels_ReadsLabels()
    {
        int[] labels = MnistDataset.ParseLabels(BuildLabels(3, [3, 7, 1]));
        Assert.Equal([3, 7, 1], labels);
    }

    [Fact]
    public void ParseImages_BadMagic_Throws()
    {
        var idx = BuildImages(1, 1, 1, [0]);
        idx[3] = 0x09; // corrupt magic
        Assert.Throws<FormatException>(() => MnistDataset.ParseImages(idx));
    }

    [Fact]
    public void ParseImages_Truncated_Throws()
    {
        byte[] idx = BuildImages(2, 2, 2, [0, 1, 2, 3, 4, 5, 6, 7]);
        Array.Resize(ref idx, idx.Length - 2); // drop trailing bytes
        Assert.Throws<FormatException>(() => MnistDataset.ParseImages(idx));
    }

    [Fact]
    public void MnistData_ValidatesImageLabelCount()
    {
        DenseTensor<float> images = MnistDataset.ParseImages(BuildImages(2, 2, 2, new byte[8]));
        Assert.Throws<ArgumentException>(() => new MnistData(images, [0])); // 2 images, 1 label
    }
}
