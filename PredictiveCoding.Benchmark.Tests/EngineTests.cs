using PredictiveCoding.FaceRecognition;

namespace PredictiveCoding.Benchmark.Tests;

public class EngineTests
{
    [Fact]
    public void Cosine_is_one_for_parallel_and_zero_for_orthogonal()
    {
        Assert.True(FaceMath.Cosine([1, 2, 3], [2, 4, 6]) > 0.999);
        Assert.True(Math.Abs(FaceMath.Cosine([1, 0], [0, 1])) < 1e-9);
    }

    [Fact]
    public void Verification_metrics_perfect_separation()
    {
        var pairs = new List<VerificationMetrics.Pair>();
        for (int fold = 0; fold < 5; fold++)
        {
            for (int i = 0; i < 50; i++)
            {
                pairs.Add(new VerificationMetrics.Pair(0.90 + (0.001 * i), true, fold));
                pairs.Add(new VerificationMetrics.Pair(0.10 + (0.001 * i), false, fold));
            }
        }

        VerificationMetrics.Result r = VerificationMetrics.Evaluate(pairs);
        Assert.True(r.Accuracy > 0.99, $"acc {r.Accuracy}");
        Assert.True(r.Auc > 0.99, $"auc {r.Auc}");
        Assert.True(r.TarAtFar1 > 0.99);
    }

    [Fact]
    public void Verification_metrics_random_is_near_chance()
    {
        var rng = new Random(1);
        var pairs = new List<VerificationMetrics.Pair>();
        for (int fold = 0; fold < 5; fold++)
            for (int i = 0; i < 100; i++)
                pairs.Add(new VerificationMetrics.Pair(rng.NextDouble(), i % 2 == 0, fold));

        VerificationMetrics.Result r = VerificationMetrics.Evaluate(pairs);
        Assert.InRange(r.Accuracy, 0.40, 0.62);
        Assert.InRange(r.Auc, 0.40, 0.60);
    }

    [Fact]
    public void Identification_finds_the_right_gallery_identity()
    {
        const int dim = 16;
        var rng = new Random(2);
        var gallery = new List<(string, float[])>();
        var anchors = new List<float[]>();
        for (int id = 0; id < 20; id++)
        {
            float[] v = RandomUnit(rng, dim);
            anchors.Add(v);
            gallery.Add((id.ToString(), v));
        }

        var probes = new List<(string, float[])>();
        for (int id = 0; id < 20; id++)
        {
            float[] v = (float[])anchors[id].Clone();
            for (int d = 0; d < dim; d++)
                v[d] += (float)(rng.NextDouble() - 0.5) * 0.05f; // small jitter
            probes.Add((id.ToString(), v));
        }

        IdentificationMetrics.Result r = IdentificationMetrics.Evaluate(gallery, probes, dim);
        Assert.True(r.Rank1 >= 0.95, $"rank-1 {r.Rank1}");
        Assert.Equal(20, r.Identities);
    }

    [Fact]
    public void EnumerateFolders_groups_by_identity_folder()
    {
        string root = Path.Combine(Path.GetTempPath(), "facebench-enum-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(root, "alice"));
            Directory.CreateDirectory(Path.Combine(root, "bob"));
            File.WriteAllText(Path.Combine(root, "alice", "a1.jpg"), "");
            File.WriteAllText(Path.Combine(root, "alice", "a2.png"), "");
            File.WriteAllText(Path.Combine(root, "alice", "notes.txt"), ""); // ignored
            File.WriteAllText(Path.Combine(root, "bob", "b1.jpeg"), "");

            var byId = FaceImages.EnumerateFolders(root).GroupBy(x => x.Id).ToDictionary(g => g.Key, g => g.Count());
            Assert.Equal(2, byId["alice"]);
            Assert.Equal(1, byId["bob"]);
        }
        finally
        {
            try { Directory.Delete(root, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Registry_lists_all_nine_datasets()
    {
        IReadOnlyList<IFaceBenchmark> all = DatasetRegistry.All();
        Assert.Equal(9, all.Count);

        string[] names = all.Select(b => b.Name).ToArray();
        foreach (string expected in new[] { "LFW", "CASIA-WebFace", "VGGFace2", "CelebA", "FairFace", "RFW", "Glint360K", "DigiFace-1M", "IJB-C" })
            Assert.Contains(expected, names);

        Assert.False(all.Single(b => b.Name == "LFW").Acquisition.Gated);
        Assert.True(all.Single(b => b.Name == "CASIA-WebFace").Acquisition.Gated);
        Assert.Equal(FaceTaskKind.Attribute, all.Single(b => b.Name == "FairFace").Kind);
    }

    private static float[] RandomUnit(Random rng, int dim)
    {
        var v = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++)
        {
            v[i] = (float)(rng.NextDouble() * 2 - 1);
            norm += (double)v[i] * v[i];
        }
        norm = Math.Sqrt(norm);
        for (int i = 0; i < dim; i++)
            v[i] = (float)(v[i] / norm);
        return v;
    }
}
