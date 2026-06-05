using PredictiveCoding.FaceRecognition;

namespace PredictiveCoding.Benchmark;

/// <summary>
/// FairFace has no identities; it is a fairness/attribute dataset. This benchmark does gender classification
/// (nearest-centroid in embedding space) and reports overall accuracy plus per-race accuracy — the bias view
/// the dataset is designed for.
/// </summary>
public sealed class FairFaceBenchmark : IFaceBenchmark
{
    public string Name => "FairFace";
    public FaceTaskKind Kind => FaceTaskKind.Attribute;
    public AcquisitionInfo Acquisition => new(
        "https://github.com/joojs/fairface",
        "Label CSVs on GitHub; images via the Google Drive links in the repo. Place under <root>/fairface/.",
        Gated: false);
    public string RelativePath => "fairface";

    private static string CsvPath(string dataRoot) => Path.Combine(dataRoot, "fairface", "fairface_label_val.csv");

    public bool IsAvailable(string dataRoot) => File.Exists(CsvPath(dataRoot));

    public string Run(string dataRoot, IFaceEmbedder embedder, BenchmarkBudget budget)
    {
        string root = Path.Combine(dataRoot, "fairface");
        string[] lines = File.ReadAllLines(CsvPath(dataRoot));
        if (lines.Length < 2)
            return "empty label CSV";

        string[] header = lines[0].Split(',');
        int fileCol = Array.IndexOf(header, "file");
        int genderCol = Array.IndexOf(header, "gender");
        int raceCol = Array.IndexOf(header, "race");
        if (fileCol < 0 || genderCol < 0)
            return "unexpected CSV header (expected file,age,gender,race,…)";

        var rows = new List<(string Path, string Gender, string Race)>();
        foreach (string line in lines.Skip(1))
        {
            string[] c = line.Split(',');
            if (c.Length <= Math.Max(fileCol, genderCol))
                continue;
            string path = Path.Combine(root, c[fileCol]);
            if (File.Exists(path))
                rows.Add((path, c[genderCol], raceCol >= 0 && raceCol < c.Length ? c[raceCol] : "all"));
            if (rows.Count >= budget.MaxImages)
                break;
        }
        if (rows.Count < 10)
            return "too few images found on disk";

        var paths = rows.Select(r => r.Path).ToList();
        embedder.Fit(paths);
        IReadOnlyList<float[]?> vectors = embedder.EmbedBatch(paths);

        var samples = new List<(float[] Vec, string Gender, string Race)>();
        for (int i = 0; i < rows.Count; i++)
            if (vectors[i] is { } v)
                samples.Add((v, rows[i].Gender, rows[i].Race));

        // 70/30 split; gender centroids from the train half.
        int split = (int)(samples.Count * 0.7);
        var train = samples.Take(split).ToList();
        var test = samples.Skip(split).ToList();

        var centroids = train
            .GroupBy(s => s.Gender)
            .ToDictionary(g => g.Key, g => Centroid(g.Select(s => s.Vec), embedder.Dimension));
        if (centroids.Count < 2)
            return "need at least two gender classes in the train split";

        int correct = 0;
        var perRace = new Dictionary<string, (int Ok, int Total)>();
        foreach ((float[] vec, string gender, string race) in test)
        {
            string predicted = centroids.MaxBy(c => FaceMath.Cosine(vec, c.Value)).Key;
            bool ok = predicted == gender;
            if (ok)
                correct++;
            (int Ok, int Total) g = perRace.GetValueOrDefault(race);
            perRace[race] = (g.Ok + (ok ? 1 : 0), g.Total + 1);
        }

        double acc = (double)correct / test.Count;
        string perRaceText = string.Join(", ", perRace.OrderBy(kv => kv.Key)
            .Select(kv => $"{kv.Key} {(double)kv.Value.Ok / kv.Value.Total:P0}"));
        return $"[{embedder.Name}] gender acc {acc:P2} on {test.Count} faces · per-race: {perRaceText}";
    }

    private static float[] Centroid(IEnumerable<float[]> vectors, int dim)
    {
        var sum = new double[dim];
        int n = 0;
        foreach (float[] v in vectors)
        {
            for (int i = 0; i < dim && i < v.Length; i++)
                sum[i] += v[i];
            n++;
        }
        var c = new float[dim];
        if (n > 0)
            for (int i = 0; i < dim; i++)
                c[i] = (float)(sum[i] / n);
        return c;
    }
}
