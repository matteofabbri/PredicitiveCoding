using PredictiveCoding.Core;
using PredictiveCoding.Training;
using System.Formats.Tar;
using System.IO.Compression;

namespace PredictiveCoding.Data;

// ─────────────────────────────────────────────────────────────────────────────
// CifarDataset
//
// Loader for the CIFAR-10 image classification benchmark.
//
// Format (binary version):
//   Each file contains N records of exactly 3073 bytes:
//     byte  0     : class label  (0–9)
//     bytes 1–1024: R channel, row-major 32×32
//     bytes 1025–2048: G channel
//     bytes 2049–3072: B channel
//
// Normalization: pixel values are scaled to [0, 1].
//
// Input tensor shape: [3, 32, 32]  (channels-first, channels × H × W)
// Label tensor: one-hot, length 10
//
// Reference:
//   Krizhevsky, A. (2009). "Learning Multiple Layers of Features from
//   Tiny Images." Technical Report, University of Toronto.
//   https://www.cs.toronto.edu/~kriz/cifar.html
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CifarDataset : IDataset
{
    // ── IDataset ──────────────────────────────────────────────────────────────
    public int    InputSize    => 3 * 32 * 32;
    public int    LabelCount   => 10;
    public int    TotalSamples => _inputs.Length;
    public string Name         { get; }

    // ── Shape exposed for NetworkConfig ──────────────────────────────────────
    public static readonly int[] InputShape  = [3, 32, 32];
    public static readonly int[] OutputShape = null!;   // flat one-hot

    public static readonly string[] ClassNames =
    [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ];

    // ── Internal storage ──────────────────────────────────────────────────────
    private readonly DenseTensor<float>[] _inputs;
    private readonly DenseTensor<float>[] _labels;

    private CifarDataset(DenseTensor<float>[] inputs, DenseTensor<float>[] labels, string name)
    {
        Name    = name;
        _inputs = inputs;
        _labels = labels;
    }

    // ── IDataset.GetBatches ───────────────────────────────────────────────────

    public IEnumerable<Batch> GetBatches(int batchSize, Random? rng = null)
    {
        int[] order = Enumerable.Range(0, TotalSamples).ToArray();
        if (rng != null) RandomUtils.Shuffle(order, rng);

        int i = 0;
        while (i < TotalSamples)
        {
            int end  = Math.Min(i + batchSize, TotalSamples);
            int size = end - i;

            var batchInputs = new DenseTensor<float>[size];
            var batchLabels = new DenseTensor<float>[size];

            for (int b = 0; b < size; b++)
            {
                batchInputs[b] = _inputs[order[i + b]];
                batchLabels[b] = _labels[order[i + b]];
            }

            yield return new Batch(batchInputs, batchLabels);
            i += size;
        }
    }

    // ── Noise corruption (for robustness experiments) ─────────────────────────

    /// <summary>Returns a new dataset with the first <paramref name="n"/> samples.</summary>
    public CifarDataset Subset(int n)
    {
        n = Math.Min(n, TotalSamples);
        return new CifarDataset(_inputs[..n], _labels[..n], $"{Name} [:{n}]");
    }

    /// <summary>
    /// Returns a new dataset where every image has additive Gaussian pixel
    /// noise N(0, sigma²) clipped to [0, 1].
    /// </summary>
    public CifarDataset WithGaussianNoise(float sigma, Random rng)
    {
        var noisy = new DenseTensor<float>[_inputs.Length];
        for (int i = 0; i < _inputs.Length; i++)
        {
            var img = _inputs[i].Clone();
            for (int p = 0; p < img.Length; p++)
                img.Buffer[p] = Math.Clamp(
                    img.Buffer[p] + RandomUtils.SampleNormal(0f, sigma, rng), 0f, 1f);
            noisy[i] = img;
        }
        return new CifarDataset(noisy, _labels, $"{Name} +noise(σ={sigma:F2})");
    }

    // ── Factory: Load ─────────────────────────────────────────────────────────

    /// <summary>
    /// Loads CIFAR-10 from <paramref name="cacheDir"/>.
    /// Downloads and extracts automatically if not present.
    /// </summary>
    /// <param name="cacheDir">Directory for storing the downloaded data.</param>
    /// <param name="maxTrainSamples">Cap on training samples (≤ 50 000).</param>
    /// <param name="maxTestSamples">Cap on test samples (≤ 10 000).</param>
    public static (CifarDataset train, CifarDataset test) Load(
        string cacheDir        = "./cifar-10-data",
        int    maxTrainSamples = 50_000,
        int    maxTestSamples  = 10_000)
    {
        string root    = Path.GetFullPath(cacheDir);
        string dataDir = Path.Combine(root, "cifar-10-batches-bin");

        if (!Directory.Exists(dataDir) ||
            !File.Exists(Path.Combine(dataDir, "data_batch_1.bin")))
        {
            Download(root);
        }

        // Training: batches 1–5 (10 000 images each, 50 000 total)
        var trainIn = new List<DenseTensor<float>>();
        var trainLb = new List<DenseTensor<float>>();

        for (int b = 1; b <= 5 && trainIn.Count < maxTrainSamples; b++)
        {
            string path = Path.Combine(dataDir, $"data_batch_{b}.bin");
            var (inputs, labels) = ParseBatch(path, maxTrainSamples - trainIn.Count);
            trainIn.AddRange(inputs);
            trainLb.AddRange(labels);
        }

        // Test
        var testIn = new List<DenseTensor<float>>();
        var testLb = new List<DenseTensor<float>>();
        {
            string path = Path.Combine(dataDir, "test_batch.bin");
            var (inputs, labels) = ParseBatch(path, maxTestSamples);
            testIn.AddRange(inputs);
            testLb.AddRange(labels);
        }

        var train = new CifarDataset([.. trainIn], [.. trainLb],
            $"CIFAR-10 Train ({trainIn.Count})");
        var test  = new CifarDataset([.. testIn],  [.. testLb],
            $"CIFAR-10 Test ({testIn.Count})");

        return (train, test);
    }

    // ── Binary parsing ────────────────────────────────────────────────────────

    // Each record: 1 byte label + 3072 bytes pixels (R×1024, G×1024, B×1024)
    private const int RecordBytes = 1 + 3 * 32 * 32;

    private static (List<DenseTensor<float>>, List<DenseTensor<float>>) ParseBatch(
        string path, int maxCount)
    {
        var inputs = new List<DenseTensor<float>>(maxCount);
        var labels = new List<DenseTensor<float>>(maxCount);
        var buf    = new byte[RecordBytes];

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1 << 16);

        while (inputs.Count < maxCount)
        {
            int read = fs.ReadAtLeast(buf, RecordBytes, throwOnEndOfStream: false);
            if (read < RecordBytes) break;

            int labelIdx = buf[0];

            var img = new DenseTensor<float>(3, 32, 32);
            for (int i = 0; i < 3 * 32 * 32; i++)
                img.Buffer[i] = buf[1 + i] / 255f;

            var label = DenseTensor<float>.Zeros(10);
            label.Buffer[labelIdx] = 1f;

            inputs.Add(img);
            labels.Add(label);
        }

        return (inputs, labels);
    }

    // ── Download + extraction ─────────────────────────────────────────────────

    private const string ArchiveUrl =
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

    private static void Download(string targetDir)
    {
        Directory.CreateDirectory(targetDir);
        string archivePath = Path.Combine(targetDir, "cifar-10-binary.tar.gz");

        if (!File.Exists(archivePath))
        {
            Console.WriteLine($"  Downloading CIFAR-10 (~162 MB) from:");
            Console.WriteLine($"    {ArchiveUrl}");

            using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(15) };
            using var response = client
                .GetAsync(ArchiveUrl, HttpCompletionOption.ResponseHeadersRead)
                .GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();

            long? totalBytes = response.Content.Headers.ContentLength;
            using var httpStream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
            using var fileStream = new FileStream(archivePath, FileMode.Create, FileAccess.Write,
                FileShare.None, bufferSize: 1 << 16);

            var buf        = new byte[1 << 16];
            long received  = 0;
            int  lastPct   = -1;
            int  n;

            while ((n = httpStream.Read(buf, 0, buf.Length)) > 0)
            {
                fileStream.Write(buf, 0, n);
                received += n;

                if (totalBytes.HasValue)
                {
                    int pct = (int)(received * 100L / totalBytes.Value);
                    if (pct != lastPct)
                    {
                        Console.Write($"\r  {pct,3}%  {received / 1_048_576} / " +
                                      $"{totalBytes.Value / 1_048_576} MB   ");
                        lastPct = pct;
                    }
                }
            }

            Console.WriteLine("\r  Download complete.                     ");
        }

        Console.Write("  Extracting... ");
        using var gz  = new GZipStream(
            new FileStream(archivePath, FileMode.Open, FileAccess.Read, FileShare.Read),
            CompressionMode.Decompress);
        TarFile.ExtractToDirectory(gz, targetDir, overwriteFiles: true);
        Console.WriteLine("done.");
    }
}
