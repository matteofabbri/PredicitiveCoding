using PredictiveCoding.FaceRecognition;

namespace PredictiveCoding.Benchmark;

/// <summary>One face dataset wired to its on-disk layout, official protocol and metric.</summary>
public interface IFaceBenchmark
{
    string Name { get; }
    FaceTaskKind Kind { get; }
    AcquisitionInfo Acquisition { get; }

    /// <summary>Sub-directory under the data root where this dataset is expected.</summary>
    string RelativePath { get; }

    /// <summary>Whether the dataset (and its protocol files) are present locally.</summary>
    bool IsAvailable(string dataRoot);

    /// <summary>Runs the benchmark and returns a one-line metric summary.</summary>
    string Run(string dataRoot, IFaceEmbedder embedder, BenchmarkBudget budget);

    /// <summary>Downloads the dataset if it is publicly fetchable; returns whether it is now available.</summary>
    Task<bool> TryDownloadAsync(string dataRoot, HttpClient http) => Task.FromResult(false);
}
