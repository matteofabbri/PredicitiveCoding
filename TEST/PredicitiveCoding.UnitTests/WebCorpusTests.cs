using System.Net;
using System.Text;
using PredictiveCoding.Data.Web;
using PredictiveCoding.LLM.Data;

namespace PredicitiveCoding.UnitTests;

// ROADMAP-Scale C1/C2: the crawler that sediments a corpus, the sharded
// append-only writer, and the read-side hygiene. The crawl runs fully offline
// against a fake HttpClient serving a small synthetic site.
public class WebCorpusTests : IDisposable
{
    private readonly string _tempRoot = Path.Combine(Path.GetTempPath(), $"pcweb_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true);
    }

    // ── A tiny fake web ──────────────────────────────────────────────────────

    private const string Menu = "<nav><ul><li><a href='/'>Home</a></li><li><a href='/about'>About</a></li></ul></nav>";

    private static readonly Dictionary<string, string> Site = new()
    {
        ["http://test.local/"] =
            $"<html><head><title>Home</title></head><body>{Menu}" +
            "<p>The brain is a prediction machine that constantly minimizes its own surprise about the world.</p>" +
            "<p>Predictive coding explains perception as inference over the hidden causes of sensory input.</p>" +
            "<a href='/a'>a</a> <a href='/b'>b</a> <a href='http://other.local/x'>ext</a>" +
            "<footer><p>© test.local — all rights reserved</p></footer></body></html>",
        ["http://test.local/a"] =
            $"<html><body>{Menu}" +
            "<p>Every cortical area predicts its own input and only the prediction errors climb upward.</p>" +
            "<p>The hierarchy settles when the model of the world explains the evidence well enough to act.</p>" +
            "<a href='/'>home</a> <a href='/b'>b</a></body></html>",
        ["http://test.local/b"] =
            $"<html><body>{Menu}" +
            "<p>Questa pagina è scritta interamente in italiano per esercitare il filtro della lingua del crawler.</p>" +
            "<p>Il cervello è una macchina predittiva che riduce continuamente la propria sorpresa sul mondo.</p>" +
            "</body></html>",
    };

    private sealed class FakeHandler : HttpMessageHandler
    {
        public List<string> Requested { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            string url = request.RequestUri!.GetLeftPart(UriPartial.Query);
            Requested.Add(url);
            if (!Site.TryGetValue(url, out string? html))
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
            var resp = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(html, Encoding.UTF8, "text/html"),
            };
            return Task.FromResult(resp);
        }
    }

    private static WebCrawler NewCrawler(CrawlOptions opts, out FakeHandler handler)
    {
        handler = new FakeHandler();
        return new WebCrawler(opts, new HttpClient(handler));
    }

    // ── C1: the crawler ──────────────────────────────────────────────────────

    [Fact]
    public async Task Crawl_FollowsSameHostLinks_OncePerPage_AndStripsBoilerplate()
    {
        using var crawler = NewCrawler(new CrawlOptions
        {
            MaxPages = 10, PerHostDelay = TimeSpan.Zero, MinPageChars = 50,
        }, out var handler);

        var pages = new List<CrawledPage>();
        var report = await crawler.CrawlAsync(["http://test.local/"], pages.Add);

        // The three real pages plus the menu's dead /about link (404 → failed);
        // the external host is never touched, nothing is fetched twice.
        Assert.Equal(4, report.PagesFetched);
        Assert.Equal(3, report.PagesKept);
        Assert.Equal(1, report.PagesFailed);
        Assert.DoesNotContain(handler.Requested, u => u.Contains("other.local"));
        Assert.Equal(4, handler.Requested.Distinct().Count());            // visited-once

        // Menu items ("Home", "About") are short blocks → boilerplate-filtered.
        Assert.All(pages, p => Assert.DoesNotContain(p.Blocks, b => b == "Home" || b == "About"));
        Assert.Contains(pages, p => p.Blocks.Any(b => b.Contains("prediction machine")));
        Assert.True(report.BlocksEmitted >= 6);
    }

    [Fact]
    public async Task Crawl_LanguageFilter_KeepsOnlyConfiguredLanguages()
    {
        using var crawler = NewCrawler(new CrawlOptions
        {
            MaxPages = 10, PerHostDelay = TimeSpan.Zero, MinPageChars = 50,
            Languages = ["English"],
            // The embedded detector is too weak on short text (tracked
            // separately); the crawler's filter MECHANISM is what this covers.
            LanguageClassifier = text => text.Contains(" è ") ? "Italian" : "English",
        }, out _);

        var pages = new List<CrawledPage>();
        var report = await crawler.CrawlAsync(["http://test.local/"], pages.Add);

        Assert.Equal(4, report.PagesFetched);                             // incl. the dead /about link
        Assert.Equal(1, report.PagesSkippedLanguage);                     // the Italian page
        Assert.All(pages, p => Assert.Equal("English", p.Language));
        Assert.DoesNotContain(pages, p => p.Blocks.Any(b => b.Contains("macchina predittiva")));
    }

    [Fact]
    public async Task Crawl_MaxPages_StopsTheFrontier()
    {
        using var crawler = NewCrawler(new CrawlOptions
        {
            MaxPages = 1, PerHostDelay = TimeSpan.Zero, MinPageChars = 50,
        }, out var handler);

        var report = await crawler.CrawlAsync(["http://test.local/"], _ => { });
        Assert.Equal(1, report.PagesFetched);
        Assert.Single(handler.Requested);
    }

    // ── C1: the sharded writer ───────────────────────────────────────────────

    [Fact]
    public void Writer_RotatesShards_AndFromFolderReadsEverythingBack()
    {
        string dir = Path.Combine(_tempRoot, "shards");
        var lines = new List<string>();
        using (var w = new ShardedCorpusWriter(dir, maxShardBytes: 2048))
        {
            for (int d = 0; d < 40; d++)
            {
                string line = $"document {d}: the quick brown fox jumps over the lazy dog number {d}.";
                lines.Add(line);
                w.Append(line);
            }
            Assert.Equal(40, w.DocumentsWritten);
        }

        Assert.True(Directory.GetFiles(dir, "shard-*.txt").Length > 1, "no rotation happened");

        var back = CorpusSource.FromFolder(dir).Lines().ToList();
        Assert.Equal(lines, back);                                        // order + content preserved
    }

    [Fact]
    public void Writer_Reopen_AppendsAfterTheLastShard()
    {
        string dir = Path.Combine(_tempRoot, "reopen");
        using (var w = new ShardedCorpusWriter(dir, maxShardBytes: 4096))
            w.Append("first run line with enough words to be a real block of text.");

        using (var w = new ShardedCorpusWriter(dir, maxShardBytes: 4096))
            w.Append("second run line appended after reopening the same folder.");

        var back = CorpusSource.FromFolder(dir).Lines().ToList();
        Assert.Equal(2, back.Count);
        Assert.Contains("first run", back[0]);
        Assert.Contains("second run", back[1]);
    }

    [Fact]
    public void Writer_MultiBlockDocument_BecomesOneLinePerBlock()
    {
        string dir = Path.Combine(_tempRoot, "blocks");
        using (var w = new ShardedCorpusWriter(dir))
            w.Append(["block one has plenty of words in it", "block two also has plenty of words", "  ", ""]);

        var back = CorpusSource.FromFolder(dir).Lines().ToList();
        Assert.Equal(2, back.Count);
    }

    // ── C2: read-side hygiene ────────────────────────────────────────────────

    [Fact]
    public void BoilerplateStripped_DropsShortLines_AndDedupDropsRepeatedChrome()
    {
        string[] raw =
        [
            "Home",                                                       // menu
            "About us",                                                   // menu
            "The brain is a prediction machine that minimizes surprise about the world.",
            "© example.com all rights reserved worldwide every single year",   // footer, long enough to survive words filter
            "Every cortical area predicts its own input and only errors climb upward.",
            "© example.com all rights reserved worldwide every single year",   // repeated footer → dedup
            "Home",
        ];
        var source = CorpusSource.FromText(string.Join('\n', raw))
                                 .WithBoilerplateStripped(minWords: 5)
                                 .Deduplicated();

        var kept = source.Lines().ToList();
        Assert.Equal(3, kept.Count);
        Assert.Contains(kept, l => l.Contains("prediction machine"));
        Assert.Single(kept, l => l.Contains("all rights reserved"));
        Assert.DoesNotContain(kept, l => l == "Home");
        Assert.Contains("boiler", source.Description);
    }

    [Fact]
    public void Where_ComposesArbitraryPredicates_IntoTheDescription()
    {
        var source = CorpusSource.FromText("keep this line\ndrop me now\nkeep that too")
                                 .Where(l => l.StartsWith("keep"), "starts-with-keep");
        Assert.Equal(2, source.Lines().Count());
        Assert.EndsWith("starts-with-keep", source.Description);
    }
}
