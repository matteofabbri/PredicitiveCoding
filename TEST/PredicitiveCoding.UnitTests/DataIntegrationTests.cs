using ClosedXML.Excel;
using Microsoft.Data.Sqlite;
using NAudio.Wave;
using Parquet.Serialization;
using PredictiveCoding.Core;
using PredictiveCoding.Data.Audio;
using PredictiveCoding.Data.Images;
using PredictiveCoding.Data.Sql;
using PredictiveCoding.Data.Tabular;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// End-to-end integration tests for PredictiveCoding.Data. Unlike the per-loader
/// unit tests, these drive several Data components together and assert that data
/// keeps its meaning as it flows across formats through the shared models
/// (<see cref="TabularData"/>, <see cref="ImageData"/>, <see cref="AudioClip"/>).
///
/// All fixtures are produced on disk at test time (CSV, SQLite, Excel, Parquet,
/// PNG, WAV), so the tests are deterministic, offline and cross-platform.
/// </summary>
public sealed class DataIntegrationTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_integration_{Guid.NewGuid():N}");

    public DataIntegrationTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        SqliteConnection.ClearAllPools();
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private string Temp(string name) => System.IO.Path.Combine(_tmp, name);

    /// <summary>Asserts two tables carry the same columns and (approximately) the same cells.</summary>
    private static void AssertSameTable(TabularData expected, TabularData actual, int precision = 3)
    {
        Assert.Equal(expected.RowCount, actual.RowCount);
        Assert.Equal(expected.ColumnCount, actual.ColumnCount);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int r = 0; r < expected.RowCount; r++)
            for (int c = 0; c < expected.ColumnCount; c++)
                Assert.Equal(expected.Values[r, c], actual.Values[r, c], precision);
    }

    /// <summary>Writes a table to a fresh SQLite table and returns the connection string.</summary>
    private string WriteToSqlite(TabularData table, string tableName = "pipeline")
    {
        string path = Temp("pipeline.db");
        string connectionString = $"Data Source={path};Pooling=False";

        string columnDefs = string.Join(", ", table.Columns.Select(c => $"\"{c}\" REAL"));
        string columnList = string.Join(", ", table.Columns.Select(c => $"\"{c}\""));

        using var connection = new SqliteConnection(connectionString);
        connection.Open();
        using (SqliteCommand create = connection.CreateCommand())
        {
            create.CommandText = $"CREATE TABLE {tableName}({columnDefs});";
            create.ExecuteNonQuery();
        }

        for (int r = 0; r < table.RowCount; r++)
        {
            using SqliteCommand insert = connection.CreateCommand();
            var placeholders = new string[table.ColumnCount];
            for (int c = 0; c < table.ColumnCount; c++)
            {
                placeholders[c] = $"@p{c}";
                insert.Parameters.AddWithValue($"@p{c}", (double)table.Values[r, c]);
            }
            insert.CommandText = $"INSERT INTO {tableName}({columnList}) VALUES ({string.Join(", ", placeholders)});";
            insert.ExecuteNonQuery();
        }

        return connectionString;
    }

    /// <summary>Writes a table to an .xlsx workbook (header + data rows) and returns the path.</summary>
    private string WriteToExcel(TabularData table)
    {
        string path = Temp("pipeline.xlsx");
        using var workbook = new XLWorkbook();
        IXLWorksheet worksheet = workbook.AddWorksheet("Sheet1");
        for (int c = 0; c < table.ColumnCount; c++)
            worksheet.Cell(1, c + 1).Value = table.Columns[c];
        for (int r = 0; r < table.RowCount; r++)
            for (int c = 0; c < table.ColumnCount; c++)
                worksheet.Cell(r + 2, c + 1).Value = table.Values[r, c];
        workbook.SaveAs(path);
        return path;
    }

    private string WriteWav(float[] interleaved, int sampleRate, int channels)
    {
        string path = Temp($"{Guid.NewGuid():N}.wav");
        var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);
        using var writer = new WaveFileWriter(path, format);
        writer.WriteSamples(interleaved, 0, interleaved.Length);
        return path;
    }

    private static float[] Sine(int frames, int sampleRate, float freq, float amplitude = 0.8f)
    {
        var data = new float[frames];
        for (int i = 0; i < frames; i++)
            data[i] = amplitude * MathF.Sin(2f * MathF.PI * freq * i / sampleRate);
        return data;
    }

    // ── Tabular: one table across CSV → SQLite → Excel ──────────────────────────

    [Fact]
    public void Tabular_CsvToSqliteToExcel_PreservesTableAcrossFormats()
    {
        // Source of truth: a CSV file on disk, loaded through the CSV loader.
        // Integer cells keep this independent of the host's decimal-separator
        // locale (Excel.Load formats cells per culture before parsing).
        string csvPath = Temp("source.csv");
        File.WriteAllText(csvPath, "x,y,z\n1,2,3\n4,5,6\n-7,8,9");
        TabularData fromCsv = Csv.Load(csvPath);

        // Hop 1: CSV table → SQLite → numeric matrix.
        string connectionString = WriteToSqlite(fromCsv);
        TabularData fromSqlite = Sqlite.QueryMatrix(connectionString, "SELECT x, y, z FROM pipeline");
        AssertSameTable(fromCsv, fromSqlite);

        // Hop 2: SQLite table → Excel workbook → numeric matrix.
        string excelPath = WriteToExcel(fromSqlite);
        TabularData fromExcel = Excel.Load(excelPath);
        AssertSameTable(fromCsv, fromExcel);

        // Spot-check a couple of cells survived the whole round trip end to end.
        Assert.Equal(-7f, fromExcel.Values[2, 0], 3);
        Assert.Equal(9f, fromExcel.Values[2, 2], 3);
    }

    [Fact]
    public void Tabular_DecimalValues_SurviveCsvThenSqlite()
    {
        // The CSV → SQLite path is culture-safe and must preserve fractional values.
        string csvPath = Temp("decimals.csv");
        File.WriteAllText(csvPath, "p,q\n0.5,-2.25\n3.125,4.75");
        TabularData fromCsv = Csv.Load(csvPath);

        string connectionString = WriteToSqlite(fromCsv);
        TabularData roundTrip = Sqlite.QueryMatrix(connectionString, "SELECT p, q FROM pipeline");

        AssertSameTable(fromCsv, roundTrip);
        Assert.Equal(0.5f, roundTrip.Values[0, 0], 3);
        Assert.Equal(3.125f, roundTrip.Values[1, 0], 3);
    }

    [Fact]
    public void Tabular_CsvColumnSubset_FlowsThroughSqlite()
    {
        // The CSV loader drops a column; the rest must reach SQLite unchanged.
        string csvPath = Temp("subset.csv");
        File.WriteAllText(csvPath, "a,b,c\n10,20,30\n40,50,60");
        TabularData subset = Csv.Load(csvPath, new CsvLoadOptions { Columns = [0, 2] });

        Assert.Equal(new[] { "a", "c" }, subset.Columns);

        string connectionString = WriteToSqlite(subset);
        TabularData roundTrip = Sqlite.QueryMatrix(connectionString, "SELECT a, c FROM pipeline ORDER BY a");

        AssertSameTable(subset, roundTrip);
        Assert.Equal(30f, roundTrip.Values[0, 1], 3); // dropped 'b', kept 'c'
    }

    // ── Tabular: CSV → Parquet → back ───────────────────────────────────────────

    [Fact]
    public async Task Tabular_ParquetRoundTrip_PreservesValues()
    {
        var rows = new List<ParquetRow>
        {
            new() { X = 1.5f, Y = -2.5f },
            new() { X = 3f,   Y = 4f },
            new() { X = 0f,   Y = 10f },
        };

        string path = Temp("table.parquet");
        await using (var stream = File.Create(path))
            await ParquetSerializer.SerializeAsync(rows, stream);

        TabularData table = ParquetLoader.Load(path);

        Assert.Equal(3, table.RowCount);
        Assert.Equal(new[] { "X", "Y" }, table.Columns);
        Assert.Equal(1.5f, table.Values[0, 0], 3);
        Assert.Equal(-2.5f, table.Values[0, 1], 3);
        Assert.Equal(10f, table.Values[2, 1], 3);
    }

    // ── Images: PNG encode → decode in several modes ────────────────────────────

    [Fact]
    public void Image_PngRoundTrip_PreservesPerPixelRgb()
    {
        // A 3×2 image with distinct, lossless-friendly colours per pixel.
        var colors = new[,]
        {
            { new Rgb24(255, 0, 0),   new Rgb24(0, 255, 0),   new Rgb24(0, 0, 255) },
            { new Rgb24(10, 20, 30),  new Rgb24(40, 50, 60),  new Rgb24(200, 100, 50) },
        };
        const int width = 3, height = 2;

        string path = Temp("pixels.png");
        using (var image = new Image<Rgb24>(width, height))
        {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    image[x, y] = colors[y, x];
            image.SaveAsPng(path);
        }

        // Decode without normalisation so we can compare exact byte values (PNG is lossless).
        ImageData decoded = ImageFiles.Load(path, new ImageLoadOptions { Normalize = false });

        Assert.Equal(height, decoded.Height);
        Assert.Equal(width, decoded.Width);
        Assert.Equal(3, decoded.Channels);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Rgb24 expected = colors[y, x];
                Assert.Equal(expected.R, decoded.Pixels[y, x, 0], 0);
                Assert.Equal(expected.G, decoded.Pixels[y, x, 1], 0);
                Assert.Equal(expected.B, decoded.Pixels[y, x, 2], 0);
            }
        }
    }

    [Fact]
    public void Image_DecodeModesCompose_NormalizeGrayscaleResize()
    {
        string path = Temp("solid.png");
        using (var image = new Image<Rgb24>(16, 12, new Rgb24(128, 128, 128)))
            image.SaveAsPng(path);

        // Normalised RGB at native size.
        ImageData rgb = ImageFiles.Load(path);
        Assert.Equal(new[] { 12, 16, 3 }, rgb.Pixels.Dimensions);
        Assert.All(rgb.Pixels.Buffer, v => Assert.Equal(128f / 255f, v, 3));

        // Grayscale collapses channels and drops to rank 2.
        ImageData gray = ImageFiles.Load(path, new ImageLoadOptions { Grayscale = true });
        Assert.Equal(1, gray.Channels);
        Assert.Equal(new[] { 12, 16 }, gray.Pixels.Dimensions);

        // Grayscale + resize compose into the requested shape.
        ImageData small = ImageFiles.Load(path, new ImageLoadOptions
        {
            Grayscale = true,
            TargetWidth = 4,
            TargetHeight = 3,
        });
        Assert.Equal(new[] { 3, 4 }, small.Pixels.Dimensions);
        Assert.All(small.Pixels.Buffer, v => Assert.InRange(v, 0f, 1.0001f));
    }

    // ── Audio: WAV → loader → spectral features ─────────────────────────────────

    [Fact]
    public void Audio_WavToMagnitudeSpectrogram_ProducesFrames()
    {
        const int rate = 16000;
        string path = WriteWav(Sine(rate, rate, 440f), rate, channels: 1);

        AudioClip clip = AudioFiles.Load(path);
        Assert.Equal(rate, clip.SampleRate);
        Assert.Equal(1, clip.Channels);

        const int frameSize = 1024, hopSize = 256;
        DenseTensor<float> spectrogram = Spectrogram.Magnitude(clip.Samples, frameSize, hopSize);

        Assert.Equal(2, spectrogram.Rank);
        Assert.True(spectrogram.Dimensions[0] > 0);          // at least one frame
        Assert.Equal(frameSize / 2 + 1, spectrogram.Dimensions[1]); // STFT bins
        Assert.All(spectrogram.Buffer, v => Assert.True(float.IsFinite(v) && v >= 0f));
    }

    [Fact]
    public void Audio_StereoWavMixedDown_ThenMfcc_ProducesFeatures()
    {
        const int rate = 16000;
        const int frames = rate; // 1 second
        var interleaved = new float[frames * 2];
        float[] left = Sine(frames, rate, 220f);
        float[] right = Sine(frames, rate, 330f);
        for (int f = 0; f < frames; f++)
        {
            interleaved[f * 2] = left[f];
            interleaved[f * 2 + 1] = right[f];
        }
        string path = WriteWav(interleaved, rate, channels: 2);

        // Loader mixes stereo down to mono by default → a 1-D signal MFCC can consume.
        AudioClip clip = AudioFiles.Load(path);
        Assert.Equal(1, clip.Channels);

        const int featureCount = 13;
        DenseTensor<float> mfcc = Spectrogram.Mfcc(clip.Samples, clip.SampleRate, featureCount);

        Assert.Equal(2, mfcc.Rank);
        Assert.True(mfcc.Dimensions[0] > 0);
        Assert.Equal(featureCount, mfcc.Dimensions[1]);
        Assert.All(mfcc.Buffer, v => Assert.True(float.IsFinite(v)));
    }

    /// <summary>Row shape used for the Parquet serialization round trip.</summary>
    public sealed class ParquetRow
    {
        public float X { get; set; }
        public float Y { get; set; }
    }
}
