namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticObjectDataset  —  a YCB-like dataset of synthetic sensorimotor objects.
//
// The recognition problem is only interesting when individual landmarks are
// *ambiguous*: objects share a small vocabulary of features arranged over a common
// grid of locations, so a single "feature F at location L" observation is
// consistent with several objects and the identity emerges only by accumulating a
// sequence of observations (and, faster, by columns voting). This mirrors how real
// objects share parts — many mugs have a cylindrical wall and a handle; what
// differs is the arrangement.
//
// Each object assigns a feature (drawn from the shared vocabulary) to every
// location on the shared grid; its FeaturePoints (Phase 8) are those (location,
// feature) landmarks. Generation is fully deterministic given a seed.
//
// Reference: Clay et al. (2024) P2; Leadholm et al. (2025) P9.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>One synthetic object: an id and its feature points in its own frame.</summary>
public sealed record SyntheticObject(int Id, IReadOnlyList<FeaturePoint> Points);

public sealed class SyntheticObjectDataset
{
    /// <summary>The generated objects, with ids 0 … <see cref="ObjectCount"/>−1.</summary>
    public IReadOnlyList<SyntheticObject> Objects { get; }

    /// <summary>The shared feature vocabulary every object draws its landmarks from.</summary>
    public IReadOnlyList<Sdr> Vocabulary { get; }

    /// <summary>The shared grid of locations (continuous coordinates) all objects use.</summary>
    public IReadOnlyList<(float X, float Y)> Grid { get; }

    /// <summary>Size of every feature SDR in the dataset.</summary>
    public int FeatureSize { get; }

    /// <summary>Half-width of the square region the location grid is drawn from.</summary>
    public float Extent { get; }

    /// <summary>Number of objects.</summary>
    public int ObjectCount => Objects.Count;

    private SyntheticObjectDataset(
        IReadOnlyList<SyntheticObject> objects, IReadOnlyList<Sdr> vocabulary,
        IReadOnlyList<(float, float)> grid, int featureSize, float extent)
    {
        Objects = objects;
        Vocabulary = vocabulary;
        Grid = grid;
        FeatureSize = featureSize;
        Extent = extent;
    }

    /// <summary>Generates <paramref name="objectCount"/> objects over a shared grid of
    /// <paramref name="locationsPerObject"/> locations and a shared vocabulary of
    /// <paramref name="featureVocabularySize"/> features.</summary>
    /// <param name="featureSize">Total bits per feature SDR.</param>
    /// <param name="featureActiveBits">Active bits per feature SDR.</param>
    /// <param name="extent">Grid locations are drawn from <c>[−extent, +extent]²</c>.</param>
    /// <param name="seed">Deterministic generation seed.</param>
    public static SyntheticObjectDataset Generate(
        int objectCount = 10,
        int locationsPerObject = 12,
        int featureVocabularySize = 5,
        int featureSize = 256,
        int featureActiveBits = 24,
        float extent = 5f,
        int seed = 42)
    {
        if (objectCount < 1) throw new ArgumentOutOfRangeException(nameof(objectCount));
        if (locationsPerObject < 1) throw new ArgumentOutOfRangeException(nameof(locationsPerObject));
        if (featureVocabularySize < 1) throw new ArgumentOutOfRangeException(nameof(featureVocabularySize));
        if (featureActiveBits < 1 || featureActiveBits >= featureSize)
            throw new ArgumentOutOfRangeException(nameof(featureActiveBits));
        if (extent <= 0f) throw new ArgumentOutOfRangeException(nameof(extent));

        var rng = new Random(seed);

        // Shared feature vocabulary and location grid (common to every object).
        var vocabulary = new Sdr[featureVocabularySize];
        for (int v = 0; v < featureVocabularySize; v++)
            vocabulary[v] = Sdr.Random(featureSize, featureActiveBits, rng);

        var grid = new (float X, float Y)[locationsPerObject];
        for (int g = 0; g < locationsPerObject; g++)
            grid[g] = ((float)(rng.NextDouble() * 2 - 1) * extent,
                       (float)(rng.NextDouble() * 2 - 1) * extent);

        // Each object assigns a vocabulary feature to every grid location.
        var objects = new List<SyntheticObject>(objectCount);
        for (int o = 0; o < objectCount; o++)
        {
            var points = new List<FeaturePoint>(locationsPerObject);
            for (int g = 0; g < locationsPerObject; g++)
            {
                var (x, y) = grid[g];
                points.Add(new FeaturePoint(x, y, vocabulary[rng.Next(featureVocabularySize)]));
            }
            objects.Add(new SyntheticObject(o, points));
        }

        return new SyntheticObjectDataset(objects, vocabulary, grid, featureSize, extent);
    }
}
