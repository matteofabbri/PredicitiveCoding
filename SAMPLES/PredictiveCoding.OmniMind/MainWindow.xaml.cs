using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Net.Http;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using PredictiveCoding.OmniMind.Engine;
using Windows.Foundation;
using Windows.Graphics;
using Windows.UI;

namespace PredictiveCoding.OmniMind;

/// <summary>
/// Shell window: drives the whole experience — fan out across every external
/// provider, fuse what they return into one corpus, then train and sample the
/// project's own transformer on it, live.
/// </summary>
public sealed partial class MainWindow : Window
{
    private readonly DispatcherQueue _ui;
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(30) };
    private readonly ObservableCollection<ProbeTile> _tiles = new();
    private readonly Polyline _lossLine = new()
    {
        Stroke = new SolidColorBrush(Color.FromArgb(255, 0x4F, 0x9D, 0xFF)),
        StrokeThickness = 2,
    };

    private CancellationTokenSource? _cts;
    private float _lossMax;

    public MainWindow()
    {
        InitializeComponent();
        _ui = DispatcherQueue.GetForCurrentThread();
        TilesView.ItemsSource = _tiles;
        LossCanvas.Children.Add(_lossLine);
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("PredictiveCoding.OmniMind/1.0 (+demo)");

        Title = "OmniMind — Predictive Coding";
        try { AppWindow?.Resize(new SizeInt32(1340, 880)); } catch { /* best effort */ }
    }

    private void TopicBox_KeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == Windows.System.VirtualKey.Enter && IgniteButton.IsEnabled)
            Ignite_Click(IgniteButton, new RoutedEventArgs());
    }

    private async void Ignite_Click(object sender, RoutedEventArgs e)
    {
        string topic = TopicBox.Text?.Trim() ?? "";
        if (topic.Length == 0) { StatusText.Text = "Enter a topic first."; return; }

        _cts?.Cancel();
        _cts = new CancellationTokenSource();
        CancellationToken ct = _cts.Token;

        IgniteButton.IsEnabled = false;
        Ring.IsActive = true;
        GeneratedText.Text = "";
        ConnectionsText.Text = "—";
        CorpusText.Text = "—";
        LossText.Text = "fanning out across every source…";
        _lossLine.Points.Clear();
        _lossMax = 0f;
        _tiles.Clear();

        var probes = KnowledgeSources.All(_http);
        foreach (var p in probes) _tiles.Add(new ProbeTile(p.Name, p.Category));

        var corpus = new ConcurrentQueue<string>();
        int done = 0, live = 0;

        // ── 1. Fan out: every provider in parallel ───────────────────────────────
        try
        {
            var tasks = new List<Task>(probes.Count);
            for (int i = 0; i < probes.Count; i++)
            {
                int idx = i;
                KnowledgeProbe probe = probes[idx];
                tasks.Add(Task.Run(async () =>
                {
                    UI(() => _tiles[idx].SetRunning());
                    var sw = Stopwatch.StartNew();
                    ProbeResult res = await KnowledgeFanOut.ProbeAsync(probe, topic, TimeSpan.FromSeconds(25), ct);
                    sw.Stop();

                    if (res.State == ProbeState.Ok)
                        foreach (string chunk in res.Text)
                            if (!string.IsNullOrWhiteSpace(chunk)) corpus.Enqueue(chunk);

                    UI(() =>
                    {
                        _tiles[idx].SetResult(res, sw.ElapsedMilliseconds);
                        done++;
                        if (res.State == ProbeState.Ok) live++;
                        SourcesText.Text = $"Sources — {done}/{probes.Count} probed · {live} live";
                        StatusText.Text = $"Fanning out… {done}/{probes.Count}";
                    });
                }, ct));
            }
            await Task.WhenAll(tasks);
        }
        catch (Exception ex)
        {
            StatusText.Text = "Fan-out error: " + ex.Message;
        }

        // ── 2. Fuse ──────────────────────────────────────────────────────────────
        string[] chunks = corpus.ToArray();
        string fused = string.Join("\n", chunks);
        CorpusText.Text = $"{chunks.Length:N0} documents · {fused.Length:N0} characters fused from {live} live sources.";
        ConnectionsText.Text = chunks.Length == 0 ? "—" : string.Join("     ", Insights.TopTerms(fused, topic, 14));

        if (fused.Length < 400)
        {
            StatusText.Text = "Not enough live data to train — is network access available?";
            Ring.IsActive = false;
            IgniteButton.IsEnabled = true;
            return;
        }

        // ── 3. Learn + 4. Speak (background thread, streamed to the UI) ───────────
        StatusText.Text = "Learning…";
        try
        {
            await Task.Run(() =>
            {
                MindTrainer.Trained trained = MindTrainer.Learn(
                    fused, new MindOptions(),
                    prog => UI(() => UpdateLoss(prog)),
                    log => UI(() => StatusText.Text = log),
                    ct);

                UI(() => StatusText.Text = "Generating in its own words…");
                MindTrainer.Speak(trained, topic + " ", maxNew: 220, temperature: 0.9f, topK: 40,
                    text => UI(() => GeneratedText.Text = text), ct);
            }, ct);

            StatusText.Text = "Done — the mind read every source and learned the topic from scratch.";
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Cancelled.";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Learning error: " + ex.Message;
        }
        finally
        {
            Ring.IsActive = false;
            IgniteButton.IsEnabled = true;
        }
    }

    private void UpdateLoss(MindTrainer.Progress p)
    {
        if (_lossMax <= 0f) _lossMax = MathF.Max(p.Loss, 0.1f);

        double w = LossCanvas.ActualWidth > 1 ? LossCanvas.ActualWidth : 360;
        double h = LossCanvas.ActualHeight > 1 ? LossCanvas.ActualHeight : 130;
        double x = (double)p.Step / Math.Max(1, p.TotalSteps) * w;
        double norm = Math.Clamp(p.Loss / _lossMax, 0d, 1d);
        double y = h - 4 - norm * (h - 8);
        _lossLine.Points.Add(new Point(x, y));

        LossText.Text =
            $"step {p.Step}/{p.TotalSteps}  ·  loss {p.Loss:F3}  ·  perplexity {p.Perplexity:F1}  ·  " +
            $"{p.StepsPerSec:F1} steps/s  ·  vocab {p.Vocab}  ·  {p.Params:N0} params";
    }

    private void UI(Action action) => _ui.TryEnqueue(() => action());
}
