using PredictiveCoding.FaceRecognition;

namespace PredictiveCoding.Benchmark;

/// <summary>A 1:1 verification benchmark driven by a list of labelled image pairs (LFW, RFW, …).</summary>
public sealed class PairsVerificationBenchmark : IFaceBenchmark
{
    private readonly Func<string, IReadOnlyList<(string A, string B, bool Same, int Fold)>> _pairs;
    private readonly Func<string, bool>? _available;
    private readonly Func<string, HttpClient, Task<bool>>? _download;

    public PairsVerificationBenchmark(
        string name, string relativePath, AcquisitionInfo acquisition,
        Func<string, IReadOnlyList<(string A, string B, bool Same, int Fold)>> pairs,
        Func<string, bool>? available = null,
        Func<string, HttpClient, Task<bool>>? download = null)
    {
        Name = name;
        RelativePath = relativePath;
        Acquisition = acquisition;
        _pairs = pairs;
        _available = available;
        _download = download;
    }

    public string Name { get; }
    public FaceTaskKind Kind => FaceTaskKind.Verification;
    public AcquisitionInfo Acquisition { get; }
    public string RelativePath { get; }

    public bool IsAvailable(string dataRoot)
        => _available?.Invoke(dataRoot) ?? Directory.Exists(Path.Combine(dataRoot, RelativePath));

    public Task<bool> TryDownloadAsync(string dataRoot, HttpClient http)
        => _download?.Invoke(dataRoot, http) ?? Task.FromResult(false);

    public string Run(string dataRoot, IFaceEmbedder embedder, BenchmarkBudget budget)
    {
        IReadOnlyList<(string A, string B, bool Same, int Fold)> pairs = _pairs(dataRoot);
        if (pairs.Count > budget.MaxPairs)
            pairs = [.. pairs.Take(budget.MaxPairs)];
        if (pairs.Count == 0)
            return "no pairs found (check the protocol file)";

        List<string> paths = pairs.SelectMany(p => new[] { p.A, p.B }).Distinct().Where(File.Exists).ToList();
        if (paths.Count == 0)
            return "no images found on disk for the pairs";

        embedder.Fit(paths);

        var cache = new Dictionary<string, float[]>(paths.Count);
        IReadOnlyList<float[]?> vectors = embedder.EmbedBatch(paths);
        for (int i = 0; i < paths.Count; i++)
            if (vectors[i] is { } v)
                cache[paths[i]] = v;

        var scored = new List<VerificationMetrics.Pair>(pairs.Count);
        foreach ((string a, string b, bool same, int fold) in pairs)
            if (cache.TryGetValue(a, out float[]? va) && cache.TryGetValue(b, out float[]? vb))
                scored.Add(new VerificationMetrics.Pair(FaceMath.Cosine(va, vb), same, fold));

        return $"[{embedder.Name}] {VerificationMetrics.Evaluate(scored)}";
    }
}

/// <summary>A 1:N identification benchmark over a folder-per-identity dataset (or any id→image enumerator).</summary>
public sealed class IdentificationBenchmark : IFaceBenchmark
{
    private readonly Func<string, IEnumerable<(string Id, string Path)>> _enumerate;
    private readonly Func<string, bool>? _available;
    private readonly Func<string, HttpClient, Task<bool>>? _download;

    public IdentificationBenchmark(
        string name, string relativePath, AcquisitionInfo acquisition,
        Func<string, IEnumerable<(string Id, string Path)>> enumerate,
        Func<string, bool>? available = null,
        Func<string, HttpClient, Task<bool>>? download = null)
    {
        Name = name;
        RelativePath = relativePath;
        Acquisition = acquisition;
        _enumerate = enumerate;
        _available = available;
        _download = download;
    }

    public string Name { get; }
    public FaceTaskKind Kind => FaceTaskKind.Identification;
    public AcquisitionInfo Acquisition { get; }
    public string RelativePath { get; }

    public bool IsAvailable(string dataRoot)
        => _available?.Invoke(dataRoot) ?? Directory.Exists(Path.Combine(dataRoot, RelativePath));

    public Task<bool> TryDownloadAsync(string dataRoot, HttpClient http)
        => _download?.Invoke(dataRoot, http) ?? Task.FromResult(false);

    public string Run(string dataRoot, IFaceEmbedder embedder, BenchmarkBudget budget)
    {
        var groups = _enumerate(dataRoot)
            .GroupBy(x => x.Id)
            .Where(g => g.Count() >= 2)
            .Take(budget.MaxIdentities)
            .Select(g => (Id: g.Key, Paths: g.Select(x => x.Path).Where(File.Exists).ToList()))
            .Where(g => g.Paths.Count >= 2)
            .ToList();
        if (groups.Count == 0)
            return "no identities with >= 2 images found";

        var selected = new List<(string Id, List<string> Paths)>();
        int total = 0;
        foreach ((string id, List<string> imgs) in groups)
        {
            List<string> paths = imgs.Take(1 + budget.MaxProbesPerIdentity).ToList();
            if (total + paths.Count > budget.MaxImages)
                break;
            total += paths.Count;
            selected.Add((id, paths));
        }

        var allPaths = selected.SelectMany(s => s.Paths).Distinct().ToList();
        embedder.Fit(allPaths);
        IReadOnlyList<float[]?> vectors = embedder.EmbedBatch(allPaths);
        var emb = new Dictionary<string, float[]>(allPaths.Count);
        for (int i = 0; i < allPaths.Count; i++)
            if (vectors[i] is { } v)
                emb[allPaths[i]] = v;

        var gallery = new List<(string, float[])>();
        var probes = new List<(string, float[])>();
        foreach ((string id, List<string> paths) in selected)
        {
            if (!emb.TryGetValue(paths[0], out float[]? g0))
                continue;
            gallery.Add((id, g0));
            for (int i = 1; i < paths.Count; i++)
                if (emb.TryGetValue(paths[i], out float[]? e))
                    probes.Add((id, e));
        }

        return $"[{embedder.Name}] {IdentificationMetrics.Evaluate(gallery, probes, embedder.Dimension)}";
    }
}
