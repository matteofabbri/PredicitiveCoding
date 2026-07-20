using PredictiveCoding.Data.Web;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Training;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// WebCorpusDemo  —  ROADMAP-Scale C1/C4: the crawler that SEDIMENTS a corpus.
//
// Crawls politely from a seed, extracts block-level text (chrome stripped,
// short boilerplate blocks dropped) and appends every page to the append-only
// shard folder — the accumulation that later feeds the C3 bootstrap
// (DevicePretrainer over CorpusSource.FromFolder + hygiene decorators).
//
// C4, the second life of the same stream: if a trained PredictiveLlm sits in
// the model folder, every crawled page ALSO feeds its ContinualLearner —
// the crawler both builds the corpus and keeps the model current, forever.
// ─────────────────────────────────────────────────────────────────────────────

public static class WebCorpusDemo
{
    public static void Run()
    {
        string root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                                   ".predictivecoding");
        string shardDir = Path.Combine(root, "web-corpus");
        string modelDir = Path.Combine(root, "web-model");

        Console.WriteLine("\n── Web corpus builder (ROADMAP-Scale C1/C4) ─────────────────");
        Console.Write($"  Seed URL [https://it.wikipedia.org/wiki/Cervello]: ");
        string? seed = Console.ReadLine()?.Trim();
        if (string.IsNullOrWhiteSpace(seed)) seed = "https://it.wikipedia.org/wiki/Cervello";

        Console.Write("  Max pages [50]: ");
        if (!int.TryParse(Console.ReadLine()?.Trim(), out int maxPages) || maxPages < 1) maxPages = 50;

        // C4: a saved model in the folder turns the crawl into a live stream too.
        ContinualLearner? learner = null;
        if (Directory.Exists(modelDir) && Directory.GetFiles(modelDir).Length > 0)
        {
            try
            {
                var llm = PredictiveLlm.Load(modelDir);
                learner = new ContinualLearner(llm);
                Console.WriteLine($"  Model found in {modelDir} → pages will ALSO feed its ContinualLearner (C4).");
            }
            catch (Exception e)
            {
                Console.WriteLine($"  (model folder present but unloadable: {e.Message} — sedimenting only)");
            }
        }
        else
        {
            Console.WriteLine("  No trained model yet → sedimenting only (bootstrap it later with DevicePretrainer).");
        }

        using var writer = new ShardedCorpusWriter(shardDir);
        using var crawler = new WebCrawler(new CrawlOptions { MaxPages = maxPages });

        long startBytes = writer.BytesWritten;
        var report = crawler.CrawlAsync([seed], page =>
        {
            writer.Append(page.Blocks);
            learner?.Absorb(string.Join('\n', page.Blocks));
            Console.WriteLine($"  + {page.Blocks.Count,3} blocks  {page.Url}");
        }).GetAwaiter().GetResult();
        writer.Flush();

        Console.WriteLine($"\n  fetched {report.PagesFetched}, kept {report.PagesKept} " +
                          $"(thin {report.PagesSkippedThin}, failed {report.PagesFailed}), " +
                          $"blocks {report.BlocksEmitted}");
        Console.WriteLine($"  sedimented {(writer.BytesWritten - startBytes) / 1024.0:F0} KB → {shardDir}");
        Console.WriteLine("  Read it back with CorpusSource.FromFolder(dir).WithBoilerplateStripped().Deduplicated();");
        Console.WriteLine("  bootstrap on the 4070 with DevicePretrainer.Pretrain(llm, corpus, …).");
    }
}
