namespace PredictiveCoding.FaceRecognition;

public static class FaceImages
{
    private static readonly string[] Extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm"];

    public static bool IsImage(string file) => Extensions.Contains(Path.GetExtension(file).ToLowerInvariant());

    /// <summary>Enumerates (identity = folder name, image path) for a folder-per-identity dataset.</summary>
    public static IEnumerable<(string Id, string Path)> EnumerateFolders(string root)
    {
        if (!Directory.Exists(root))
            yield break;
        foreach (string dir in Directory.EnumerateDirectories(root))
        {
            string id = System.IO.Path.GetFileName(dir);
            foreach (string file in Directory.EnumerateFiles(dir))
                if (IsImage(file))
                    yield return (id, file);
        }
    }

    public static float[]? TryEmbed(IFaceEmbedder embedder, string path)
    {
        try { return embedder.Embed(path); }
        catch { return null; }
    }
}
