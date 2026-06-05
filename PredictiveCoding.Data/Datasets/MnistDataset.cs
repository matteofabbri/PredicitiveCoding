using System.IO.Compression;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Datasets;

/// <summary>
/// Loads MNIST / Fashion-MNIST. Downloads the gzipped IDX files and parses them;
/// the IDX parsers are public and run offline.
/// </summary>
public sealed class MnistDataset : IDisposable
{
    private const int ImageMagic = 0x00000803;
    private const int LabelMagic = 0x00000801;

    private const string MnistBaseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    private const string FashionBaseUrl = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public MnistDataset(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Downloads and parses a split (train or test) of the chosen variant.</summary>
    public async Task<MnistData> LoadAsync(
        bool test = false, MnistVariant variant = MnistVariant.Mnist,
        bool normalize = true, CancellationToken cancellationToken = default)
    {
        string baseUrl = variant == MnistVariant.FashionMnist ? FashionBaseUrl : MnistBaseUrl;
        string imageFile = test ? "t10k-images-idx3-ubyte.gz" : "train-images-idx3-ubyte.gz";
        string labelFile = test ? "t10k-labels-idx1-ubyte.gz" : "train-labels-idx1-ubyte.gz";

        byte[] imageGz = await _http.GetByteArrayAsync(baseUrl + imageFile, cancellationToken).ConfigureAwait(false);
        byte[] labelGz = await _http.GetByteArrayAsync(baseUrl + labelFile, cancellationToken).ConfigureAwait(false);

        DenseTensor<float> images = ParseImages(Gunzip(imageGz), normalize);
        int[] labels = ParseLabels(Gunzip(labelGz));
        return new MnistData(images, labels);
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    // ── IDX parsing (offline) ─────────────────────────────────────────────────

    /// <summary>Parses an IDX3 image file (magic 0x00000803) into a <c>[count, rows, cols]</c> tensor.</summary>
    public static DenseTensor<float> ParseImages(byte[] idx, bool normalize = true)
    {
        ArgumentNullException.ThrowIfNull(idx);
        if (idx.Length < 16)
            throw new FormatException("IDX image data is too short for a header.");
        if (ReadBigEndianInt32(idx, 0) != ImageMagic)
            throw new FormatException("Unexpected IDX image magic number.");

        int count = ReadBigEndianInt32(idx, 4);
        int rows = ReadBigEndianInt32(idx, 8);
        int cols = ReadBigEndianInt32(idx, 12);
        if (count <= 0 || rows <= 0 || cols <= 0)
            throw new FormatException("IDX image header has non-positive dimensions.");

        long expected = 16L + (long)count * rows * cols;
        if (idx.Length < expected)
            throw new FormatException("IDX image data is truncated.");

        var data = new float[count * rows * cols];
        float scale = normalize ? 1f / 255f : 1f;
        for (int i = 0; i < data.Length; i++)
            data[i] = idx[16 + i] * scale;

        return new DenseTensor<float>(data, count, rows, cols);
    }

    /// <summary>Parses an IDX1 label file (magic 0x00000801) into an array of labels.</summary>
    public static int[] ParseLabels(byte[] idx)
    {
        ArgumentNullException.ThrowIfNull(idx);
        if (idx.Length < 8)
            throw new FormatException("IDX label data is too short for a header.");
        if (ReadBigEndianInt32(idx, 0) != LabelMagic)
            throw new FormatException("Unexpected IDX label magic number.");

        int count = ReadBigEndianInt32(idx, 4);
        if (count < 0 || idx.Length < 8L + count)
            throw new FormatException("IDX label data is truncated.");

        var labels = new int[count];
        for (int i = 0; i < count; i++)
            labels[i] = idx[8 + i];
        return labels;
    }

    private static int ReadBigEndianInt32(byte[] buffer, int offset)
        => (buffer[offset] << 24) | (buffer[offset + 1] << 16) | (buffer[offset + 2] << 8) | buffer[offset + 3];

    private static byte[] Gunzip(byte[] compressed)
    {
        using var input = new MemoryStream(compressed);
        using var gzip = new GZipStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        gzip.CopyTo(output);
        return output.ToArray();
    }
}
