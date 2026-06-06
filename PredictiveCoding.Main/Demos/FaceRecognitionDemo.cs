using PredictiveCoding.FaceRecognition;
using PredictiveCoding.FaceRecognition.Benchmark;
using PredictiveCoding.FaceRecognition.Datasets;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// FaceRecognitionDemo
//
// Interactive demo that:
//   1. Downloads LFW (Labeled Faces in the Wild, ~233 MB) or uses a local dataset.
//   2. Benchmarks five embedders on the standard LFW 10-fold verification protocol:
//        • LBP descriptor (baseline, 4704-D fixed)
//        • Predictive Coding latent (128-D, PC autoencoder)
//        • ArcFace w600k_r50 (512-D, ONNX, downloads ~166 MB)
//        • Enhanced LBP: Tan&Triggs DoG normalization + PCA whitening (200-D, ~80%)
//        • Fused LBP+HOG: gamma + PCA whitening (256-D, ~84%)
//   3. Runs closed-set identification (gallery + probe split).
//   4. Optionally downloads large-scale training datasets from HuggingFace Hub.
// ─────────────────────────────────────────────────────────────────────────────

public static class FaceRecognitionDemo
{
    private static readonly string DataDir   = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "face-data");
    private static readonly string ModelsDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "face-models");
    private static readonly string StoreDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "face-store");

    // ── Entry point ───────────────────────────────────────────────────────────────

    public static void Run()
    {
        Banner();

        string? imagesRoot = GetOrDownloadDataset();
        if (imagesRoot is null) return;

        string pairsFile = Path.Combine(DataDir, "pairs.txt");

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("  ── Menu ─────────────────────────────────────────────────────────────────");
            Console.WriteLine("  1  LFW Verification Benchmark  (10-fold, 6000 pairs, 3 embedders)");
            Console.WriteLine("  2  Face Identification          (closed-set Rank-1 / Rank-5)");
            Console.WriteLine("  3  Download large-scale dataset (HuggingFace Hub)");
            Console.WriteLine("  4  Dataset statistics");
            Console.WriteLine("  Q  Quit");
            Console.Write("\n  Choice: ");
            string? input = Console.ReadLine()?.Trim().ToUpperInvariant();

            switch (input)
            {
                case "1" when File.Exists(pairsFile):
                    RunVerification(imagesRoot, pairsFile);
                    break;
                case "1":
                    Console.WriteLine("  pairs.txt not found — re-download LFW or use option 3.");
                    break;
                case "2":
                    RunIdentification(imagesRoot);
                    break;
                case "3":
                    DownloadLargeDataset();
                    break;
                case "4":
                    ShowStats(imagesRoot);
                    break;
                case "Q" or "":
                    return;
                default:
                    Console.WriteLine("  Unknown option.");
                    break;
            }
        }
    }

    // ── Dataset acquisition ───────────────────────────────────────────────────────

    private static string? GetOrDownloadDataset()
    {
        string lfwRoot = Path.Combine(DataDir, "lfw-deepfunneled");
        if (Directory.Exists(lfwRoot) && Directory.EnumerateDirectories(lfwRoot).Any())
        {
            Console.WriteLine($"\n  Found LFW at: {lfwRoot}");
            return lfwRoot;
        }

        Console.WriteLine($"\n  No dataset found in {DataDir}");
        Console.WriteLine();
        Console.WriteLine("  D  Download LFW-deepfunneled  (~233 MB, 13 233 images, 5749 identities)");
        Console.WriteLine("  P  Use existing dataset         (one sub-folder per identity)");
        Console.WriteLine("  Q  Cancel");
        Console.Write("\n  Choice [D]: ");
        string? choice = Console.ReadLine()?.Trim().ToUpperInvariant();

        if (choice == "P")
        {
            Console.Write("  Dataset root path: ");
            string? path = Console.ReadLine()?.Trim().Trim('"');
            if (path is not null && Directory.Exists(path)) return path;
            Console.WriteLine("  Path not found.");
            return null;
        }
        if (choice == "Q") return null;

        // Download LFW
        try
        {
            FaceDatasetDownloader.DownloadLfwAsync(DataDir, deepFunneled: true,
                s => Console.WriteLine("  " + s)).GetAwaiter().GetResult();
            return lfwRoot;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Download failed: {ex.Message}");
            return null;
        }
    }

    // ── Verification benchmark ────────────────────────────────────────────────────

    private static void RunVerification(string imagesRoot, string pairsFile)
    {
        var dataset = new LfwDataset(imagesRoot, pairsFile);
        Console.WriteLine($"\n  LFW: {dataset.TotalPairs:N0} pairs across {dataset.FoldCount} folds");

        var embedders = ChooseEmbedders();
        if (embedders.Count == 0) return;

        var results = new List<VerificationResult>();
        foreach (var embedder in embedders)
        {
            Console.WriteLine($"\n  ── {embedder.Name} ──");
            try
            {
                var r = FaceVerificationBenchmark.RunLfw(dataset, embedder,
                    s => Console.WriteLine("  " + s));
                results.Add(r);
                PrintVerification(r);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ERROR: {ex.Message}");
            }
            finally
            {
                (embedder as IDisposable)?.Dispose();
            }
        }

        if (results.Count > 1) PrintVerificationTable(results);
    }

    // ── Identification benchmark ──────────────────────────────────────────────────

    private static void RunIdentification(string imagesRoot)
    {
        Console.Write("\n  Max subjects (Enter = all): ");
        string? input    = Console.ReadLine()?.Trim();
        int maxSubjects  = int.TryParse(input, out int n) ? n : int.MaxValue;

        var embedders = ChooseEmbedders();
        if (embedders.Count == 0) return;

        var results = new List<IdentificationResult>();
        foreach (var embedder in embedders)
        {
            Console.WriteLine($"\n  ── {embedder.Name} ──");
            try
            {
                var r = FaceIdentificationBenchmark.Run(
                    imagesRoot, embedder,
                    s => Console.WriteLine("  " + s),
                    maxSubjects);
                results.Add(r);
                PrintIdentification(r);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ERROR: {ex.Message}");
            }
            finally
            {
                (embedder as IDisposable)?.Dispose();
            }
        }

        if (results.Count > 1) PrintIdentificationTable(results);
    }

    // ── HuggingFace large dataset download ────────────────────────────────────────

    private static void DownloadLargeDataset()
    {
        Console.WriteLine("\n  Large-scale face datasets available on HuggingFace Hub:");
        Console.WriteLine();
        var datasets = FaceDatasetDownloader.LargeScaleDatasets;
        for (int i = 0; i < datasets.Length; i++)
            Console.WriteLine($"  {i + 1}  {datasets[i].Name,-22} {datasets[i].Description}");

        Console.WriteLine();
        Console.WriteLine("  Set HF_TOKEN env-var for gated datasets.");
        Console.WriteLine("  Find file paths in the dataset's 'Files and versions' tab on huggingface.co.");
        Console.WriteLine();
        Console.Write("  HuggingFace repo ID  (e.g. nateraw/celebahq-256): ");
        string? repoId = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(repoId)) return;

        Console.Write("  File path within repo (e.g. data.zip): ");
        string? filePath = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(filePath)) return;

        Console.Write("  HuggingFace token    (Enter = use HF_TOKEN env-var): ");
        string? token = Console.ReadLine()?.Trim();

        try
        {
            string dest = FaceDatasetDownloader.DownloadHuggingFaceFileAsync(
                DataDir, repoId, filePath,
                hfToken: string.IsNullOrEmpty(token) ? null : token,
                onProgress: s => Console.WriteLine("  " + s))
                .GetAwaiter().GetResult();
            Console.WriteLine($"\n  Saved to: {dest}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n  ERROR: {ex.Message}");
        }
    }

    // ── Dataset statistics ────────────────────────────────────────────────────────

    private static void ShowStats(string imagesRoot)
    {
        Console.WriteLine($"\n  Root: {imagesRoot}");
        var groups = FaceImages.EnumerateFolders(imagesRoot)
            .GroupBy(p => p.Id)
            .OrderByDescending(g => g.Count())
            .ToList();

        long totalImages = groups.Sum(g => g.LongCount());
        Console.WriteLine($"  Identities  : {groups.Count:N0}");
        Console.WriteLine($"  Total images: {totalImages:N0}");
        Console.WriteLine($"  Avg / subject: {(double)totalImages / groups.Count:F1}");

        // Distribution histogram
        (int lo, string label)[] buckets =
        [
            (1, "1"),  (2, "2–4"), (5, "5–9"), (10, "10–19"),
            (20, "20–49"), (50, "50–99"), (100, "≥100"),
        ];
        int[] counts  = new int[buckets.Length];
        int[] ceilings = [2, 5, 10, 20, 50, 100, int.MaxValue];
        foreach (var g in groups)
        {
            int n = g.Count();
            for (int b = 0; b < buckets.Length; b++)
                if (n >= buckets[b].lo && n < ceilings[b]) { counts[b]++; break; }
        }

        Console.WriteLine("\n  Images per identity:");
        int maxCount = counts.Max();
        for (int b = 0; b < buckets.Length; b++)
        {
            string bar = AsciiBar(counts[b], maxCount, 30);
            Console.WriteLine($"    {buckets[b].label,6}  {bar}  {counts[b]:N0}");
        }

        Console.WriteLine("\n  Top 10 by image count:");
        foreach (var g in groups.Take(10))
            Console.WriteLine($"    {g.Key,-36}  {g.Count(),5}");
    }

    // ── Embedder selection ────────────────────────────────────────────────────────

    private static List<IFaceEmbedder> ChooseEmbedders()
    {
        Console.WriteLine();
        Console.WriteLine("  Embedders:");
        Console.WriteLine("  1  LBP raw               (4704-D, baseline, no training)");
        Console.WriteLine("  2  PC latent             (128-D, unsupervised PC autoencoder)");
        Console.WriteLine("  3  ArcFace w600k_r50     (512-D, downloads ~166 MB ONNX model)  [SOTA]");
        Console.WriteLine("  4  Enhanced LBP          (DoG + PCA 200-D, expected ~80%)  ← new");
        Console.WriteLine("  5  Fused LBP+HOG         (gamma + PCA 256-D, expected ~84%)  ← new");
        Console.WriteLine("  A  All five              (full benchmark comparison)");
        Console.Write("\n  Choice [4]: ");
        string? choice = Console.ReadLine()?.Trim().ToUpperInvariant();
        if (string.IsNullOrEmpty(choice)) choice = "4";

        bool all = choice == "A";
        var result = new List<IFaceEmbedder>();
        if (all || choice == "1") result.Add(new DescriptorEmbedder());
        if (all || choice == "2") result.Add(new PcFaceEmbedder(StoreDir));
        if (all || choice == "3") result.Add(new ArcFaceOnnxEmbedder(ModelsDir));
        if (all || choice == "4") result.Add(new EnhancedLbpEmbedder());
        if (all || choice == "5") result.Add(new FusedEmbedder());
        return result;
    }

    // ── Display ───────────────────────────────────────────────────────────────────

    private static void PrintVerification(VerificationResult r)
    {
        Console.WriteLine();
        Console.WriteLine($"  ┌─ Results ─────────────────────────────────────────────────────────");
        Console.WriteLine($"  │  Accuracy      {r.Accuracy * 100,7:F2}%  ±  {r.AccuracyStd * 100:F2}%");
        Console.WriteLine($"  │  AUC-ROC       {r.AucRoc,9:F4}");
        Console.WriteLine($"  │  Threshold     {r.BestThreshold,9:F4}   (mean across folds)");
        Console.WriteLine($"  │  TPR / FPR     {r.Tpr * 100,6:F1}% / {r.Fpr * 100:F1}%");
        Console.WriteLine($"  │  Same score    {r.MeanSameScore,9:F4}   (mean cosine, same person)");
        Console.WriteLine($"  │  Diff score    {r.MeanDiffScore,9:F4}   (mean cosine, different)");
        Console.WriteLine($"  │  Score gap     {r.MeanSameScore - r.MeanDiffScore,9:F4}");
        Console.WriteLine($"  │  Pairs         {r.PairsEvaluated,9:N0}   (missing images: {r.MissingImages})");
        Console.WriteLine($"  └───────────────────────────────────────────────────────────────────");
    }

    private static void PrintVerificationTable(IList<VerificationResult> results)
    {
        Console.WriteLine();
        Console.WriteLine("  ══ Comparison ══════════════════════════════════════════════════════════");
        Console.WriteLine($"  {"Embedder",-32}  {"Acc",8}  {"AUC",8}  {"Gap",8}  Chart");
        Console.WriteLine("  " + new string('─', 72));
        double bestAcc = results.Max(r => r.Accuracy);
        foreach (var r in results.OrderByDescending(r => r.Accuracy))
        {
            double gap = r.MeanSameScore - r.MeanDiffScore;
            string bar = AsciiBar((float)r.Accuracy, (float)bestAcc, 20);
            Console.WriteLine(
                $"  {r.EmbedderName,-32}  {r.Accuracy * 100,6:F2}%  {r.AucRoc,8:F4}  {gap,+8:F4}  {bar}");
        }
        Console.WriteLine("  ════════════════════════════════════════════════════════════════════════");
    }

    private static void PrintIdentification(IdentificationResult r)
    {
        Console.WriteLine();
        Console.WriteLine($"  ┌─ Results ─────────────────────────────────────────────────────────");
        Console.WriteLine($"  │  Rank-1  {r.Top1Accuracy * 100,8:F2}%");
        Console.WriteLine($"  │  Rank-5  {r.Top5Accuracy * 100,8:F2}%");
        Console.WriteLine($"  │  Gallery {r.GallerySubjects,8:N0}  subjects");
        Console.WriteLine($"  │  Probes  {r.ProbeCount,8:N0}  images");
        Console.WriteLine($"  └───────────────────────────────────────────────────────────────────");
    }

    private static void PrintIdentificationTable(IList<IdentificationResult> results)
    {
        Console.WriteLine();
        Console.WriteLine("  ══ Comparison ══════════════════════════════════════════════════════════");
        Console.WriteLine($"  {"Embedder",-32}  {"Rank-1",8}  {"Rank-5",8}");
        Console.WriteLine("  " + new string('─', 54));
        foreach (var r in results.OrderByDescending(r => r.Top1Accuracy))
            Console.WriteLine($"  {r.EmbedderName,-32}  {r.Top1Accuracy * 100,6:F2}%  {r.Top5Accuracy * 100,6:F2}%");
        Console.WriteLine("  ════════════════════════════════════════════════════════════════════════");
    }

    private static string AsciiBar(float v, float max, int width)
    {
        if (max <= 0) return new string('░', width);
        int filled = (int)Math.Round(Math.Clamp(v / max, 0f, 1f) * width);
        return new string('█', filled) + new string('░', width - filled);
    }

    private static void Banner()
    {
        Console.WriteLine();
        Console.WriteLine("  ╔═══════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("  ║          Face Recognition Demo  ·  Predictive Coding              ║");
        Console.WriteLine("  ║  LBP · Enhanced LBP (DoG+PCA) · Fused · PC · ArcFace ONNX        ║");
        Console.WriteLine("  ╚═══════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"  Data  : {DataDir}");
        Console.WriteLine($"  Models: {ModelsDir}");
        Console.WriteLine($"  Store : {StoreDir}");
    }
}
