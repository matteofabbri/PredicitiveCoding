using System.Diagnostics;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Eval;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// FullLlmDemo  —  THE FINISHED PRODUCT, live: PredictiveLlm driven end to end.
//
//   create   every 2026 roadmap mechanism on at once — hybrid temporal/attention
//            blocks, weight-tied loops with early exit, the t+2 head, FLOP-sparse
//            mixture of experts, the latent plan;
//   train    windowed corpus, held-out split, LR schedule — perplexity before
//            and after, epoch by epoch in front of you;
//   generate self-speculative decoding (identical to greedy, fewer forwards),
//            plus one generation with the plan settled on the prompt;
//   learn    online, WHILE USING — the model absorbs brand-new text on the
//            spot (the PC symmetry: states settle, weights absorb);
//   save     config + weights round-trip, verified on the spot;
//   report   the activation observatory as one light-theme HTML file.
// ─────────────────────────────────────────────────────────────────────────────

public static class FullLlmDemo
{
    private const int Epochs = 6;

    public static void Run()
    {
        Console.WriteLine("\n═══ Full LLM (PredictiveLlm) · train → generate → report ═══\n");

        // ── 1 · Create ───────────────────────────────────────────────────────
        var llm = PredictiveLlm.Create2026(dim: 48, blocks: 4, heads: 4, ffDim: 96, maxSeq: 96);
        Console.WriteLine("  The product class, every roadmap feature enabled at once:");
        Console.WriteLine($"    blocks {llm.Config.BlockPattern}  (T = temporal state · A = attention + MoE)");
        Console.WriteLine($"    loops {llm.Config.NumLoops}/pass, early-exit τ {llm.Config.LoopExitTau}  ·  t+2 head λ {llm.Config.MtpWeight}");
        Console.WriteLine($"    {llm.Config.NumExperts} experts top-{llm.Config.MoeTopK} (FLOP-sparse)  ·  latent plan on");
        Console.WriteLine($"    {llm.ParameterCount:N0} parameters, {llm.ActiveParameterCountPerToken:N0} active per token\n");

        // ── 2 · Train ────────────────────────────────────────────────────────
        string corpus = string.Join('\n', Enumerable.Repeat(EvalCorpus.TinyLines, 12).SelectMany(l => l));
        Console.WriteLine($"  Training on the yardstick corpus ({Epochs} epochs, ~1 min on CPU):");
        var sw = Stopwatch.StartNew();
        var report = llm.Train(corpus, new TrainingOptions
        {
            SeqLen = 32, Stride = 16, Epochs = Epochs, BatchWindows = 8, MaxWindows = 72,
            OnEpochEnd = (epoch, loss) => Console.WriteLine($"    epoch {epoch}/{Epochs} · mean loss {loss:F3}"),
        });
        sw.Stop();
        Console.WriteLine($"\n  {report}");
        Console.WriteLine($"  ({sw.Elapsed.TotalSeconds:F0}s)\n");

        // ── 3 · Generate ─────────────────────────────────────────────────────
        Console.WriteLine("  Greedy generation, self-speculative (t+2 head drafts, main head verifies):");
        foreach (string prompt in (string[])["The brain is", "Every layer", "Learning is"])
        {
            var g = llm.GenerateDetailed(prompt, maxNewTokens: 48, new GenerationOptions { StopAtEos = false });
            Console.WriteLine($"    » {prompt}▸{OneLine(g.Text)}");
            Console.WriteLine($"      {g.Forwards} forwards for 48 tokens · drafts accepted {g.Accepted}/{g.Drafted} ({g.AcceptanceRate:P0})");
        }

        var planned = llm.GenerateDetailed("The brain is", maxNewTokens: 48,
            new GenerationOptions { StopAtEos = false, SettlePlanSteps = 5 });
        Console.WriteLine($"    » (latent plan settled on the prompt first)");
        Console.WriteLine($"      The brain is▸{OneLine(planned.Text)}\n");

        // ── 4 · Learn while using (the PC symmetry) ──────────────────────────
        Console.WriteLine("  Training while using: settling moved the STATES — now the WEIGHTS");
        Console.WriteLine("  absorb brand-new text as the model works (observed text only,");
        Console.WriteLine("  never its own output):");
        string novel = string.Join('\n',
            "Prediction is compression: a model that predicts well has understood.",
            "Surprise is the bill the world sends for a bad model.");
        for (int pass = 1; pass <= 3; pass++)
        {
            var r = llm.Learn(novel, steps: 2);
            Console.WriteLine($"    pass {pass} · {r}");
        }
        var continued = llm.GenerateDetailed("Prediction is", maxNewTokens: 40,
            new GenerationOptions { StopAtEos = false });
        Console.WriteLine($"    » Prediction is▸{OneLine(continued.Text)}");
        Console.WriteLine("    (GenerationOptions.LearnFromPrompt = true does this on every call)\n");

        // ── 5 · Save / reload ────────────────────────────────────────────────
        string home     = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string modelDir = Path.Combine(home, ".predictivecoding", "predictive-llm-demo");
        llm.Save(modelDir);
        var reloaded = PredictiveLlm.Load(modelDir);
        bool identical = reloaded.Generate("The brain is", 24) == llm.Generate("The brain is", 24);
        Console.WriteLine($"  Saved + reloaded from {modelDir}");
        Console.WriteLine($"    reloaded model generates identically: {(identical ? "✓" : "✗")}\n");

        // ── 6 · Report ───────────────────────────────────────────────────────
        string reportPath = llm.Report(Path.Combine(home, ".predictivecoding", "predictive-llm-report.html"));
        Console.WriteLine($"  Observatory report (spikes · sink shares · loop passes · perplexity):");
        Console.WriteLine($"    {reportPath}");
        // Hand the file to the shell asynchronously; harmless if no browser is around.
        try { Process.Start(new ProcessStartInfo(reportPath) { UseShellExecute = true }); } catch { }

        Console.WriteLine("\n  Done — one class, the whole roadmap.");
    }

    private static string OneLine(string s)
    {
        string t = s.Replace('\n', ' ').Replace('\r', ' ');
        return t.Length > 64 ? t[..64] + "…" : t;
    }
}
