using System.Text.Json;
using System.Text.Json.Serialization;

namespace PredictiveCoding.FaceRecognition.Recognition;

/// <summary>
/// Production face gallery — register identities once, then do real-time 1:N identification
/// and 1:1 verification.
/// <para>
/// Internally keeps a flat list of L2-normalised embedding vectors. Identification uses cosine
/// dot-product (equivalent to cosine similarity when both vectors are L2-normalised).
/// For up to ~10 k registered faces, linear scan completes in under a millisecond; beyond that,
/// a future upgrade to an HNSW index will be a drop-in swap.
/// </para>
/// Save / Load uses compact JSON — float arrays ≈ 2.5 MB per 1 000 ArcFace embeddings.
/// </summary>
public sealed class FaceDatabase
{
    private readonly List<FaceRecord> _records = [];
    private readonly IFaceEmbedder   _embedder;
    private readonly object           _lock = new();

    /// <summary>Cosine similarity threshold used by <see cref="VerifyPaths"/>. Default: 0.35 for ArcFace.</summary>
    public double VerifyThreshold { get; set; } = 0.35;

    /// <summary>Number of registered face records.</summary>
    public int Count => _records.Count;

    /// <summary>Name of the embedder used by this database.</summary>
    public string EmbedderName => _embedder.Name;

    public FaceDatabase(IFaceEmbedder embedder, double verifyThreshold = 0.35)
    {
        _embedder      = embedder;
        VerifyThreshold = verifyThreshold;
    }

    // ── Registration ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Embeds <paramref name="imagePath"/> and registers it under <paramref name="name"/>.
    /// Returns the new <see cref="FaceRecord"/>.
    /// </summary>
    public FaceRecord Register(string name, string imagePath, string? id = null, float quality = 1.0f)
    {
        float[] emb = _embedder.Embed(imagePath);
        var rec = new FaceRecord
        {
            Id        = id ?? Guid.NewGuid().ToString("N")[..12],
            Name      = name,
            Embedding = emb,
            ImagePath = imagePath,
            Quality   = quality,
            AddedAt   = DateTime.UtcNow,
        };
        lock (_lock) _records.Add(rec);
        return rec;
    }

    /// <summary>
    /// Registers several images for the same <paramref name="name"/> and stores their
    /// <b>centroid embedding</b> (mean, then L2-renormalised). A single record is created.
    /// </summary>
    public FaceRecord RegisterAverage(string name, IReadOnlyList<string> imagePaths, string? id = null)
    {
        if (imagePaths.Count == 0)
            throw new ArgumentException("At least one image is required.", nameof(imagePaths));

        var embeddings = _embedder.EmbedBatch(imagePaths);
        var valid = embeddings.Where(e => e is not null).Select(e => e!).ToList();
        if (valid.Count == 0)
            throw new InvalidOperationException("None of the provided images could be embedded.");

        int d = valid[0].Length;
        var mean = new float[d];
        foreach (var v in valid)
            for (int i = 0; i < d; i++)
                mean[i] += v[i];

        // L2 renormalise centroid
        float norm = MathF.Sqrt(mean.Sum(x => x * x));
        if (norm > 1e-8f)
            for (int i = 0; i < d; i++) mean[i] /= norm;

        var rec = new FaceRecord
        {
            Id        = id ?? Guid.NewGuid().ToString("N")[..12],
            Name      = name,
            Embedding = mean,
            ImagePath = imagePaths[0],   // store first as representative
            Quality   = (float)valid.Count / imagePaths.Count,
            AddedAt   = DateTime.UtcNow,
        };
        lock (_lock) _records.Add(rec);
        return rec;
    }

    // ── Identification (1:N) ──────────────────────────────────────────────────────

    /// <summary>
    /// Identifies the face in <paramref name="imagePath"/> against all registered records.
    /// Returns the best match and its cosine similarity score, or
    /// <c>(null, 0)</c> if the database is empty.
    /// </summary>
    public (FaceRecord? Match, double Score) Identify(string imagePath)
    {
        float[] query = _embedder.Embed(imagePath);
        return IdentifyEmbedding(query);
    }

    /// <summary>
    /// Identifies a pre-computed <paramref name="embedding"/> against all registered records.
    /// </summary>
    public (FaceRecord? Match, double Score) IdentifyEmbedding(float[] embedding)
    {
        List<FaceRecord> snapshot;
        lock (_lock) snapshot = [.. _records];

        if (snapshot.Count == 0) return (null, 0);

        FaceRecord? best = null;
        double bestScore = double.MinValue;
        foreach (FaceRecord rec in snapshot)
        {
            double score = Cosine(embedding, rec.Embedding);
            if (score > bestScore) { bestScore = score; best = rec; }
        }
        return (best, bestScore);
    }

    /// <summary>
    /// Returns the top-<paramref name="k"/> matches, sorted by score descending.
    /// </summary>
    public IReadOnlyList<(FaceRecord Record, double Score)> IdentifyTopK(string imagePath, int k = 5)
    {
        float[] query = _embedder.Embed(imagePath);
        List<FaceRecord> snapshot;
        lock (_lock) snapshot = [.. _records];

        return snapshot
            .Select(r => (Record: r, Score: Cosine(query, r.Embedding)))
            .OrderByDescending(x => x.Score)
            .Take(k)
            .ToList();
    }

    // ── Verification (1:1) ────────────────────────────────────────────────────────

    /// <summary>
    /// Verifies whether the two images show the same person using
    /// cosine similarity ≥ <see cref="VerifyThreshold"/>.
    /// </summary>
    public (bool Same, double Score) VerifyPaths(string path1, string path2)
    {
        var embeddings = _embedder.EmbedBatch([path1, path2]);
        if (embeddings[0] is null || embeddings[1] is null)
            return (false, 0);
        return VerifyEmbeddings(embeddings[0]!, embeddings[1]!);
    }

    /// <summary>Verifies pre-computed embeddings against the configured threshold.</summary>
    public (bool Same, double Score) VerifyEmbeddings(float[] e1, float[] e2)
    {
        double score = Cosine(e1, e2);
        return (score >= VerifyThreshold, score);
    }

    // ── CRUD ──────────────────────────────────────────────────────────────────────

    /// <summary>Returns a snapshot of all registered records.</summary>
    public IReadOnlyList<FaceRecord> GetAll()
    {
        lock (_lock) return [.. _records];
    }

    /// <summary>Returns all records whose <see cref="FaceRecord.Name"/> matches exactly.</summary>
    public IReadOnlyList<FaceRecord> GetByName(string name)
    {
        lock (_lock)
            return _records.Where(r => string.Equals(r.Name, name, StringComparison.OrdinalIgnoreCase)).ToList();
    }

    /// <summary>Removes the record with the given <paramref name="id"/>. Returns <see langword="true"/> if found.</summary>
    public bool Delete(string id)
    {
        lock (_lock)
        {
            int idx = _records.FindIndex(r => r.Id == id);
            if (idx < 0) return false;
            _records.RemoveAt(idx);
            return true;
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────────────

    private static readonly JsonSerializerOptions SerOpts = new()
    {
        WriteIndented              = false,
        DefaultIgnoreCondition     = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy       = JsonNamingPolicy.CamelCase,
    };

    /// <summary>Saves the database to a JSON file at <paramref name="filePath"/>.</summary>
    public void Save(string filePath)
    {
        List<FaceRecord> snapshot;
        lock (_lock) snapshot = [.. _records];

        var dto = new DatabaseDto
        {
            EmbedderName   = EmbedderName,
            VerifyThreshold = VerifyThreshold,
            Records        = snapshot,
        };
        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        File.WriteAllText(filePath, JsonSerializer.Serialize(dto, SerOpts));
    }

    /// <summary>
    /// Loads a previously saved database from <paramref name="filePath"/>.
    /// <paramref name="embedder"/> must produce vectors of the same dimension as the saved ones.
    /// </summary>
    public static FaceDatabase Load(string filePath, IFaceEmbedder embedder)
    {
        string json = File.ReadAllText(filePath);
        var dto = JsonSerializer.Deserialize<DatabaseDto>(json, SerOpts)
                  ?? throw new InvalidDataException("Empty or malformed database file.");

        if (!string.Equals(dto.EmbedderName, embedder.Name, StringComparison.OrdinalIgnoreCase))
            Console.WriteLine(
                $"  [FaceDatabase] WARNING: saved with '{dto.EmbedderName}', " +
                $"now loading with '{embedder.Name}'");

        var db = new FaceDatabase(embedder, dto.VerifyThreshold);
        lock (db._lock) db._records.AddRange(dto.Records ?? []);
        return db;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) dot += (double)a[i] * b[i];
        return dot; // valid because both vectors are L2-normalised
    }

    // ── JSON DTO ──────────────────────────────────────────────────────────────────

    private sealed class DatabaseDto
    {
        public string         EmbedderName    { get; set; } = string.Empty;
        public double         VerifyThreshold { get; set; } = 0.35;
        public List<FaceRecord>? Records      { get; set; }
    }
}
