using System.Formats.Tar;
using System.IO.Compression;
using PredictiveCoding.FaceRecognition;

namespace PredictiveCoding.Benchmark;

/// <summary>The nine face datasets, each wired to its layout, official protocol, metric and acquisition info.</summary>
public static class DatasetRegistry
{
    // figshare mirror used by scikit-learn (the umass host no longer resolves).
    private const string LfwTgzUrl = "https://ndownloader.figshare.com/files/5976018";
    private const string LfwPairsUrl = "https://ndownloader.figshare.com/files/5976006";

    public static IReadOnlyList<IFaceBenchmark> All() =>
    [
        new PairsVerificationBenchmark(
            "LFW", "lfw",
            new AcquisitionInfo("http://vis-www.cs.umass.edu/lfw/ (figshare mirror)", "Public; auto-downloaded by this runner.", Gated: false),
            LfwPairs,
            available: r => File.Exists(Path.Combine(r, "lfw", "pairs.txt")) && HasSubdirs(Path.Combine(r, "lfw")),
            download: DownloadLfwAsync),

        new IdentificationBenchmark(
            "CASIA-WebFace", "casia-webface",
            new AcquisitionInfo("https://paperswithcode.com/dataset/casia-webface", "Gated (license). Place identity folders under <root>/casia-webface/.", Gated: true),
            r => FaceImages.EnumerateFolders(Path.Combine(r, "casia-webface"))),

        new IdentificationBenchmark(
            "VGGFace2", "vggface2",
            new AcquisitionInfo("https://github.com/ox-vgg/vgg_face2", "Gated. Place <root>/vggface2/train/<id>/<img>.jpg.", Gated: true),
            r => FaceImages.EnumerateFolders(Path.Combine(r, "vggface2", "train")),
            available: r => Directory.Exists(Path.Combine(r, "vggface2", "train"))),

        new IdentificationBenchmark(
            "CelebA", "celeba",
            new AcquisitionInfo("https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html", "Public-ish (Google Drive). Need img_align_celeba/ + identity_CelebA.txt under <root>/celeba/.", Gated: false),
            CelebAImages,
            available: r => File.Exists(Path.Combine(r, "celeba", "identity_CelebA.txt"))),

        new FairFaceBenchmark(),

        new PairsVerificationBenchmark(
            "RFW", "rfw",
            new AcquisitionInfo("http://www.whdeng.cn/RFW/index.html", "Gated (application to BUPT). Place <root>/rfw/test/{data,txts}/<Race>/.", Gated: true),
            RfwPairs,
            available: r => Directory.Exists(Path.Combine(r, "rfw", "test", "data"))),

        new IdentificationBenchmark(
            "Glint360K", "glint360k",
            new AcquisitionInfo("https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_", "Gated; distributed as MXNet .rec. Extract to <root>/glint360k/<id>/<img> to benchmark.", Gated: true),
            r => FaceImages.EnumerateFolders(Path.Combine(r, "glint360k"))),

        new IdentificationBenchmark(
            "DigiFace-1M", "digiface",
            new AcquisitionInfo("https://github.com/microsoft/DigiFace1M", "Public (synthetic, Azure blobs). Place subject folders under <root>/digiface/.", Gated: false),
            r => FaceImages.EnumerateFolders(Path.Combine(r, "digiface"))),

        new IjbcBenchmark(),
    ];

    // ── LFW ────────────────────────────────────────────────────────────────

    private static IReadOnlyList<(string A, string B, bool Same, int Fold)> LfwPairs(string dataRoot)
    {
        string lfwDir = Path.Combine(dataRoot, "lfw");
        string pairsFile = Path.Combine(lfwDir, "pairs.txt");
        var result = new List<(string, string, bool, int)>();
        if (!File.Exists(pairsFile))
            return result;

        string[] lines = File.ReadAllLines(pairsFile);
        string[] head = lines[0].Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
        int folds = int.Parse(head[0]);
        int per = int.Parse(head[1]);
        int li = 1;

        string Img(string name, int idx) => Path.Combine(lfwDir, name, $"{name}_{idx:D4}.jpg");

        for (int f = 0; f < folds && li < lines.Length; f++)
        {
            for (int i = 0; i < per && li < lines.Length; i++)
            {
                string[] t = lines[li++].Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
                if (t.Length >= 3)
                    result.Add((Img(t[0], int.Parse(t[1])), Img(t[0], int.Parse(t[2])), true, f));
            }
            for (int i = 0; i < per && li < lines.Length; i++)
            {
                string[] t = lines[li++].Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
                if (t.Length >= 4)
                    result.Add((Img(t[0], int.Parse(t[1])), Img(t[2], int.Parse(t[3])), false, f));
            }
        }
        return result;
    }

    private static async Task<bool> DownloadLfwAsync(string dataRoot, HttpClient http)
    {
        string lfwDir = Path.Combine(dataRoot, "lfw");
        Directory.CreateDirectory(lfwDir);

        string pairs = Path.Combine(lfwDir, "pairs.txt");
        if (!File.Exists(pairs))
            await File.WriteAllBytesAsync(pairs, await http.GetByteArrayAsync(LfwPairsUrl));

        if (!HasSubdirs(lfwDir))
        {
            byte[] tgz = await http.GetByteArrayAsync(LfwTgzUrl);
            using var gz = new GZipStream(new MemoryStream(tgz), CompressionMode.Decompress);
            await TarFile.ExtractToDirectoryAsync(gz, dataRoot, overwriteFiles: true); // creates <root>/lfw/<Name>/...
        }
        return File.Exists(pairs) && HasSubdirs(lfwDir);
    }

    // ── RFW (per-race verification; LFW-style pairs files) ─────────────────

    private static readonly string[] RfwRaces = ["African", "Asian", "Caucasian", "Indian"];

    private static IReadOnlyList<(string A, string B, bool Same, int Fold)> RfwPairs(string dataRoot)
    {
        var result = new List<(string, string, bool, int)>();
        for (int race = 0; race < RfwRaces.Length; race++)
        {
            string r = RfwRaces[race];
            string pairsFile = Path.Combine(dataRoot, "rfw", "test", "txts", r, $"{r}_pairs.txt");
            string dataDir = Path.Combine(dataRoot, "rfw", "test", "data", r);
            if (!File.Exists(pairsFile))
                continue;

            string Img(string id, int idx) => Path.Combine(dataDir, id, $"{id}_{idx:D4}.jpg");
            foreach (string line in File.ReadLines(pairsFile))
            {
                string[] t = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
                if (t.Length == 3)
                    result.Add((Img(t[0], int.Parse(t[1])), Img(t[0], int.Parse(t[2])), true, race));
                else if (t.Length == 4)
                    result.Add((Img(t[0], int.Parse(t[1])), Img(t[2], int.Parse(t[3])), false, race));
            }
        }
        return result;
    }

    // ── CelebA (identity file, not folders) ────────────────────────────────

    private static IEnumerable<(string Id, string Path)> CelebAImages(string dataRoot)
    {
        string root = Path.Combine(dataRoot, "celeba");
        string idFile = Path.Combine(root, "identity_CelebA.txt");
        string imgDir = Path.Combine(root, "img_align_celeba");
        if (!File.Exists(idFile))
            yield break;
        foreach (string line in File.ReadLines(idFile))
        {
            string[] t = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
            if (t.Length < 2)
                continue;
            string path = Path.Combine(imgDir, t[0]);
            if (File.Exists(path))
                yield return (t[1], path);
        }
    }

    private static bool HasSubdirs(string dir)
        => Directory.Exists(dir) && Directory.EnumerateDirectories(dir).Any();
}
