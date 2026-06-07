using PredictiveCoding.FaceRecognition;
using PredictiveCoding.FaceRecognition.Datasets;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the face-recognition dataset plumbing: folder-per-identity
/// enumeration (incl. PGM), the direct-download catalogue, and the archive-root resolver.
/// No network and no real images are used.</summary>
public sealed class FaceDatasetsTests
{
    [Fact]
    public void FaceImages_IsImage_AcceptsPgmAndCommonFormats()
    {
        Assert.True(FaceImages.IsImage("a.jpg"));
        Assert.True(FaceImages.IsImage("a.PNG"));
        Assert.True(FaceImages.IsImage("orl.pgm"));   // AT&T / ORL grayscale
        Assert.False(FaceImages.IsImage("notes.txt"));
    }

    [Fact]
    public void FaceImages_EnumerateFolders_YieldsIdentityImagePairs()
    {
        string root = NewTempDir();
        try
        {
            Touch(root, "Alice", "a1.jpg");
            Touch(root, "Alice", "a2.pgm");
            Touch(root, "Alice", "notes.txt");   // not an image
            Touch(root, "Bob", "b1.png");

            var items = FaceImages.EnumerateFolders(root).ToList();

            Assert.Equal(3, items.Count);
            Assert.Contains(items, t => t.Id == "Alice" && t.Path.EndsWith("a2.pgm"));
            Assert.Contains(items, t => t.Id == "Bob" && t.Path.EndsWith("b1.png"));
            Assert.DoesNotContain(items, t => t.Path.EndsWith("notes.txt"));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void DirectDatasets_CatalogueIncludesLfwGeorgiaTechAndOrl()
    {
        var keys = FaceDatasetDownloader.DirectDatasets.Select(d => d.RepoId).ToList();
        Assert.Contains("lfw", keys);
        Assert.Contains("gt_db", keys);
        Assert.Contains("att_faces", keys);
        Assert.All(FaceDatasetDownloader.DirectDatasets,
            d => Assert.False(string.IsNullOrWhiteSpace(d.Description)));
    }

    [Fact]
    public void ResolveImagesRoot_DescendsThroughASingleWrapperFolder()
    {
        string root = NewTempDir();
        try
        {
            // wrapper case: extract/gt_db/s01  → root should be extract/gt_db
            string wrapper = Path.Combine(root, "extract", "gt_db");
            Directory.CreateDirectory(Path.Combine(wrapper, "s01"));
            Assert.Equal(wrapper, FaceDatasetDownloader.ResolveImagesRoot(Path.Combine(root, "extract")));

            // flat case: extract2/s1, extract2/s2 → root should be extract2 itself
            string flat = Path.Combine(root, "extract2");
            Directory.CreateDirectory(Path.Combine(flat, "s1"));
            Directory.CreateDirectory(Path.Combine(flat, "s2"));
            Assert.Equal(flat, FaceDatasetDownloader.ResolveImagesRoot(flat));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "facetest_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Touch(string root, string identity, string file)
    {
        string dir = Path.Combine(root, identity);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, file), "x");
    }
}
