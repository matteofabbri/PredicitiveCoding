using PredictiveCoding.FaceRecognition;

namespace PredictiveCoding.Benchmark;

/// <summary>
/// IJB-C 1:1 verification. Faces are grouped into <b>templates</b> (a template = several media of one subject);
/// each template is represented by the mean of its media embeddings, and the official match list is scored
/// (TAR@FAR). Expects the standard release layout under <c>&lt;root&gt;/ijbc/</c>:
/// <list type="bullet">
///   <item><c>protocols/ijbc_11_G1_G2_matches.csv</c> — pairs of template ids to compare</item>
///   <item><c>protocols/*g1*.csv</c>, <c>*g2*.csv</c>, <c>*probe*.csv</c> — TEMPLATE_ID, SUBJECT_ID, FILENAME</item>
///   <item><c>images/</c> — the media referenced by FILENAME</item>
/// </list>
/// </summary>
public sealed class IjbcBenchmark : IFaceBenchmark
{
    public string Name => "IJB-C";
    public FaceTaskKind Kind => FaceTaskKind.Verification;
    public AcquisitionInfo Acquisition => new(
        "https://www.nist.gov/programs-projects/face-challenges (IARPA Janus Benchmark-C)",
        "Requires a signed request to NIST. Place the release under <root>/ijbc/ (protocols/ + images/).",
        Gated: true);
    public string RelativePath => "ijbc";

    private static string ProtocolDir(string dataRoot) => Path.Combine(dataRoot, "ijbc", "protocols");

    public bool IsAvailable(string dataRoot)
        => Directory.Exists(ProtocolDir(dataRoot)) &&
           Directory.EnumerateFiles(ProtocolDir(dataRoot), "*matches*.csv").Any();

    public string Run(string dataRoot, IFaceEmbedder embedder, BenchmarkBudget budget)
    {
        string ijbc = Path.Combine(dataRoot, "ijbc");
        string protocols = ProtocolDir(dataRoot);
        string matchesCsv = Directory.EnumerateFiles(protocols, "*matches*.csv").FirstOrDefault() ?? "";
        if (!File.Exists(matchesCsv))
            return "matches protocol CSV not found under protocols/";

        // Build TEMPLATE_ID -> (subject, files) from every template-definition CSV.
        var templates = new Dictionary<long, (string Subject, List<string> Files)>();
        foreach (string csv in Directory.EnumerateFiles(protocols, "*.csv"))
        {
            if (csv == matchesCsv)
                continue;
            ReadTemplateCsv(csv, ijbc, templates);
        }
        if (templates.Count == 0)
            return "no template definitions parsed (expected TEMPLATE_ID, SUBJECT_ID, FILENAME columns)";

        // Pool each template = mean of its (capped) media embeddings.
        var pooled = new Dictionary<long, (string Subject, float[] Vec)>();
        var fitPaths = templates.Values.SelectMany(t => t.Files.Take(3)).Distinct().Take(budget.MaxImages).ToList();
        embedder.Fit(fitPaths);

        foreach ((long id, (string subject, List<string> files)) in templates)
        {
            var sum = new double[embedder.Dimension];
            int n = 0;
            foreach (string file in files.Take(6))
            {
                float[]? e = FaceImages.TryEmbed(embedder, file);
                if (e is null)
                    continue;
                for (int i = 0; i < embedder.Dimension && i < e.Length; i++)
                    sum[i] += e[i];
                n++;
            }
            if (n == 0)
                continue;
            var v = new float[embedder.Dimension];
            for (int i = 0; i < v.Length; i++)
                v[i] = (float)(sum[i] / n);
            pooled[id] = (subject, v);
        }

        // Score the match list; label = same subject.
        var scored = new List<VerificationMetrics.Pair>();
        foreach (string line in File.ReadLines(matchesCsv))
        {
            string[] c = line.Split(',', ' ', '\t');
            if (c.Length < 2 || !long.TryParse(c[0], out long t1) || !long.TryParse(c[1], out long t2))
                continue;
            if (!pooled.TryGetValue(t1, out var a) || !pooled.TryGetValue(t2, out var b))
                continue;
            scored.Add(new VerificationMetrics.Pair(FaceMath.Cosine(a.Vec, b.Vec), a.Subject == b.Subject, 0));
            if (scored.Count >= budget.MaxPairs)
                break;
        }
        if (scored.Count == 0)
            return "no scorable template matches (image files missing?)";

        return $"[{embedder.Name}] {VerificationMetrics.Evaluate(scored)}";
    }

    private static void ReadTemplateCsv(string csv, string ijbcRoot, Dictionary<long, (string Subject, List<string> Files)> templates)
    {
        using var reader = new StreamReader(csv);
        string? header = reader.ReadLine();
        if (header is null)
            return;
        string[] cols = header.Split(',');
        int t = Array.FindIndex(cols, h => h.Equals("TEMPLATE_ID", StringComparison.OrdinalIgnoreCase));
        int s = Array.FindIndex(cols, h => h.Equals("SUBJECT_ID", StringComparison.OrdinalIgnoreCase));
        int f = Array.FindIndex(cols, h => h.Equals("FILENAME", StringComparison.OrdinalIgnoreCase) || h.Equals("FILE", StringComparison.OrdinalIgnoreCase));
        if (t < 0 || f < 0)
            return;

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            string[] c = line.Split(',');
            if (c.Length <= Math.Max(t, f) || !long.TryParse(c[t], out long id))
                continue;
            string subject = s >= 0 && s < c.Length ? c[s] : "?";
            string file = Path.Combine(ijbcRoot, "images", c[f]);
            if (!templates.TryGetValue(id, out var entry))
                templates[id] = entry = (subject, []);
            entry.Files.Add(file);
        }
    }
}
