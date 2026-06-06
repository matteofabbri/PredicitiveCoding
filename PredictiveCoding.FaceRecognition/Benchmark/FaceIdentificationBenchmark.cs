namespace PredictiveCoding.FaceRecognition.Benchmark;

public record IdentificationResult(
    double Top1Accuracy,
    double Top5Accuracy,
    int    GallerySubjects,
    int    ProbeCount,
    string EmbedderName
);

public static class FaceIdentificationBenchmark
{
    /// <summary>
    /// Closed-set identification benchmark: uses the first image of each identity as
    /// the gallery template; all remaining images are probes. Reports Rank-1 and Rank-5 accuracy.
    /// Only subjects with at least 2 images are included.
    /// </summary>
    public static IdentificationResult Run(
        string          imagesRoot,
        IFaceEmbedder   embedder,
        Action<string>? onProgress  = null,
        int             maxSubjects = int.MaxValue)
    {
        var bySubject = FaceImages.EnumerateFolders(imagesRoot)
            .GroupBy(p => p.Id)
            .Where(g => g.Count() >= 2)
            .Take(maxSubjects)
            .ToList();

        var galleryPaths  = bySubject.Select(g => g.First().Path).ToList();
        var galleryIds    = bySubject.Select(g => g.Key).ToList();
        var probePairs    = bySubject
            .SelectMany(g => g.Skip(1).Select(p => (Id: g.Key, Path: p.Path)))
            .ToList();

        onProgress?.Invoke($"Gallery: {galleryPaths.Count} subjects  |  Probes: {probePairs.Count}");

        // Fit on combined set so unsupervised embedders (PC) see all faces
        onProgress?.Invoke($"Fitting {embedder.Name} …");
        var allPaths = galleryPaths.Concat(probePairs.Select(p => p.Path)).ToList();
        embedder.Fit(allPaths);

        // Embed gallery
        onProgress?.Invoke("Embedding gallery …");
        var galleryEmbeds = embedder.EmbedBatch(galleryPaths);
        var gallery = new FaceGallery();
        for (int i = 0; i < galleryPaths.Count; i++)
            if (galleryEmbeds[i] is { } e)
                gallery.Add(galleryIds[i], e);

        // Embed probes
        onProgress?.Invoke("Embedding probes …");
        var probePaths  = probePairs.Select(p => p.Path).ToList();
        var probeEmbeds = embedder.EmbedBatch(probePaths);

        int top1 = 0, top5 = 0, total = 0;
        for (int i = 0; i < probePairs.Count; i++)
        {
            if (probeEmbeds[i] is null) continue;
            string trueId   = probePairs[i].Id;
            var    top5hits  = gallery.FindTopK(probeEmbeds[i]!, 5);
            total++;
            if (top5hits.Count > 0 && top5hits[0].Id == trueId) top1++;
            if (top5hits.Any(h => h.Id == trueId))              top5++;
        }

        return new IdentificationResult(
            total > 0 ? (double)top1 / total : 0,
            total > 0 ? (double)top5 / total : 0,
            gallery.Count, total, embedder.Name);
    }
}
