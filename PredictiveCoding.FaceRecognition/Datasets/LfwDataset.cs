namespace PredictiveCoding.FaceRecognition.Datasets;

/// <summary>
/// Parses the standard LFW 10-fold cross-validation verification protocol.
/// Each fold contains 300 matched (same-person) and 300 mismatched (different-person) pairs —
/// 6000 pairs in total across 10 folds, as defined by pairs.txt.
/// </summary>
public sealed class LfwDataset
{
    // Each element of _folds is a list of (path1, path2, isSamePerson) triples for that fold.
    private readonly (string Path1, string Path2, bool Same)[][] _folds;

    public string ImagesRoot { get; }
    public int    FoldCount  => _folds.Length;
    public int    TotalPairs => _folds.Sum(f => f.Length);

    public LfwDataset(string imagesRoot, string pairsFilePath)
    {
        ImagesRoot = imagesRoot;
        _folds     = ParsePairs(pairsFilePath, imagesRoot);
    }

    public IReadOnlyList<(string Path1, string Path2, bool Same)> GetFold(int i) => _folds[i];

    public IEnumerable<(string Path1, string Path2, bool Same)> AllPairs()
        => _folds.SelectMany(f => f);

    public IEnumerable<(string Id, string Path)> AllImages()
        => FaceImages.EnumerateFolders(ImagesRoot);

    // ── Parser ────────────────────────────────────────────────────────────────────
    // pairs.txt layout:
    //   Line 0  : "<numFolds>\t<pairsPerFold>"
    //   Lines 1 … numFolds*pairsPerFold          : matched pairs  "<name>\t<i1>\t<i2>"
    //   Lines numFolds*pairsPerFold+1 … end      : mismatched     "<n1>\t<i1>\t<n2>\t<i2>"
    // Fold assignment is positional: pair k belongs to fold k/pairsPerFold.

    private static (string, string, bool)[][] ParsePairs(string pairsFile, string root)
    {
        string[] lines = File.ReadAllLines(pairsFile);
        string[] hdr   = lines[0].Split('\t');
        int numFolds     = int.Parse(hdr[0].Trim());
        int pairsPerFold = int.Parse(hdr[1].Trim());
        int total        = numFolds * pairsPerFold;

        var folds = Enumerable.Range(0, numFolds)
            .Select(_ => new List<(string, string, bool)>(pairsPerFold * 2))
            .ToArray();

        int samePairsStart = 1;
        int diffPairsStart = 1 + total;

        for (int i = 0; i < total && samePairsStart + i < lines.Length; i++)
        {
            string line = lines[samePairsStart + i];
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] p = line.Split('\t');
            if (p.Length < 3) continue;
            folds[i / pairsPerFold].Add(
                (ImgPath(root, p[0], p[1]), ImgPath(root, p[0], p[2]), true));
        }

        for (int i = 0; i < total && diffPairsStart + i < lines.Length; i++)
        {
            string line = lines[diffPairsStart + i];
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] p = line.Split('\t');
            if (p.Length < 4) continue;
            folds[i / pairsPerFold].Add(
                (ImgPath(root, p[0], p[1]), ImgPath(root, p[2], p[3]), false));
        }

        return folds.Select(f => f.ToArray()).ToArray();
    }

    private static string ImgPath(string root, string name, string idxStr)
    {
        int idx = int.Parse(idxStr.Trim());
        return Path.Combine(root, name, $"{name}_{idx:D4}.jpg");
    }
}
