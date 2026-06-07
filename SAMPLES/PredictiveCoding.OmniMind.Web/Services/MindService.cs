using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.Extensions.Options;
using PredictiveCoding.OmniMind.Engine;
using PredictiveCoding.OmniMind.Web.Models;

namespace PredictiveCoding.OmniMind.Web.Services;

/// <summary>
/// Orchestrates one OmniMind run: fan out across every live provider, fuse what
/// they return into a corpus, train the project's own transformer on it from
/// scratch, then sample a continuation — streaming every step to the UI as
/// <see cref="IgniteUpdate"/>s, and saving the finished <see cref="RunRecord"/> to
/// the <see cref="HistoryStore"/>. Singleton: a semaphore caps concurrent (CPU-bound)
/// trainings so the server stays responsive.
/// </summary>
public sealed class MindService
{
    private readonly IHttpClientFactory _httpFactory;
    private readonly HistoryStore _history;
    private readonly ILogger<MindService> _log;
    private readonly OmniMindOptions _opt;
    private readonly SemaphoreSlim _trainGate;

    public MindService(
        IHttpClientFactory httpFactory,
        HistoryStore history,
        IOptions<OmniMindOptions> options,
        ILogger<MindService> log)
    {
        _httpFactory = httpFactory;
        _history = history;
        _log = log;
        _opt = options.Value;
        _trainGate = new SemaphoreSlim(Math.Max(1, _opt.MaxConcurrentTrainings));
    }

    public async Task<RunRecord?> RunAsync(string topic, IProgress<IgniteUpdate> progress, CancellationToken ct)
    {
        topic = (topic ?? "").Trim();
        if (topic.Length == 0) return null;

        var sw = Stopwatch.StartNew();
        HttpClient http = _httpFactory.CreateClient("omnimind");

        List<KnowledgeProbe> probes = KnowledgeSources.All(http);
        var sourceInfos = probes.Select(p => new SourceInfo(p.Name, p.Category)).ToList();
        progress.Report(new IgniteUpdate(IgniteStage.Started) { Sources = sourceInfos });

        // ── 1. Fan out: every provider in parallel ───────────────────────────────
        var corpus = new ConcurrentQueue<string>();
        var outcomes = new SourceOutcome?[probes.Count];
        int done = 0, live = 0;
        var timeout = TimeSpan.FromSeconds(_opt.SourceTimeoutSeconds);

        var tasks = new List<Task>(probes.Count);
        for (int i = 0; i < probes.Count; i++)
        {
            int idx = i;
            KnowledgeProbe probe = probes[idx];
            tasks.Add(Task.Run(async () =>
            {
                progress.Report(new IgniteUpdate(IgniteStage.SourceRunning) { SourceIndex = idx });

                var psw = Stopwatch.StartNew();
                ProbeResult res = await KnowledgeFanOut.ProbeAsync(probe, topic, timeout, ct);
                psw.Stop();

                if (res.State == ProbeState.Ok)
                    foreach (string chunk in res.Text)
                        if (!string.IsNullOrWhiteSpace(chunk)) corpus.Enqueue(chunk);

                var outcome = new SourceOutcome
                {
                    Name = probe.Name,
                    Category = probe.Category,
                    State = res.State.ToString(),
                    ItemCount = res.ItemCount,
                    Detail = res.Detail,
                    ElapsedMs = psw.ElapsedMilliseconds,
                };
                outcomes[idx] = outcome;

                int doneCount = Interlocked.Increment(ref done);
                if (res.State == ProbeState.Ok) Interlocked.Increment(ref live);
                progress.Report(new IgniteUpdate(IgniteStage.SourceDone)
                {
                    SourceIndex = idx,
                    Outcome = outcome,
                    DoneSources = doneCount,
                    LiveSources = Volatile.Read(ref live),
                });
            }, ct));
        }

        try { await Task.WhenAll(tasks); }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex) { _log.LogWarning(ex, "OmniMind fan-out: one or more sources errored"); }

        ct.ThrowIfCancellationRequested();

        // ── 2. Fuse ──────────────────────────────────────────────────────────────
        string[] chunks = corpus.ToArray();
        string fused = string.Join("\n", chunks);
        int liveCount = Volatile.Read(ref live);
        IReadOnlyList<string> topTerms = Insights.TopTerms(fused, topic, 14);

        progress.Report(new IgniteUpdate(IgniteStage.Fused)
        {
            CorpusDocs = chunks.Length,
            CorpusChars = fused.Length,
            LiveSources = liveCount,
            TopTerms = topTerms,
        });

        if (fused.Length < 400)
        {
            progress.Report(new IgniteUpdate(IgniteStage.Failed)
            {
                Text = "Not enough live data to train — is network access available?",
            });
            return null;
        }

        // ── 3. Learn + 4. Speak (CPU-bound, gated) ───────────────────────────────
        var lossHistory = new List<float>();
        MindTrainer.Progress? lastProgress = null;
        MindTrainer.Trained? trained = null;
        string generated = "";

        await _trainGate.WaitAsync(ct);
        try
        {
            await Task.Run(() =>
            {
                trained = MindTrainer.Learn(
                    fused, _opt.Training.ToMindOptions(),
                    p =>
                    {
                        lossHistory.Add(p.Loss);
                        lastProgress = p;
                        progress.Report(new IgniteUpdate(IgniteStage.Train) { Progress = p });
                    },
                    msg => progress.Report(new IgniteUpdate(IgniteStage.Log) { Text = msg }),
                    ct);

                generated = MindTrainer.Speak(
                    trained, topic + " ",
                    _opt.Generation.MaxNewTokens, _opt.Generation.Temperature, _opt.Generation.TopK,
                    text => progress.Report(new IgniteUpdate(IgniteStage.Token) { Text = text }),
                    ct);
            }, ct);
        }
        finally
        {
            _trainGate.Release();
        }

        ct.ThrowIfCancellationRequested();
        sw.Stop();

        // ── 5. Persist + announce ────────────────────────────────────────────────
        var run = new RunRecord
        {
            Topic = topic,
            CreatedAtUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            DurationMs = sw.ElapsedMilliseconds,
            CorpusDocs = chunks.Length,
            CorpusChars = fused.Length,
            LiveSources = liveCount,
            TotalSources = probes.Count,
            Sources = outcomes.Where(o => o is not null).Select(o => o!).ToList(),
            TopTerms = topTerms.ToList(),
            LossHistory = Downsample(lossHistory, 400),
            FinalLoss = lastProgress?.Loss ?? 0f,
            FinalPerplexity = lastProgress?.Perplexity ?? 0f,
            Vocab = trained?.Vocab ?? 0,
            Params = trained?.Params ?? 0,
            GeneratedText = generated,
        };

        _history.Save(run);
        progress.Report(new IgniteUpdate(IgniteStage.Completed) { Run = run });
        return run;
    }

    private static List<float> Downsample(List<float> xs, int max)
    {
        if (xs.Count <= max) return [.. xs];
        var res = new List<float>(max);
        double step = (double)xs.Count / max;
        for (int i = 0; i < max; i++) res.Add(xs[(int)(i * step)]);
        return res;
    }
}
