using System.Text.Json;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredictiveCoding.LLM;

// ─────────────────────────────────────────────────────────────────────────────
// PredictiveLlm  —  THE FINISHED PRODUCT.
//
// Everything the 2026 roadmap built, composed behind one class:
//
//   var llm = PredictiveLlm.Create2026();          // hybrid SSM + looped depth
//   llm.Train(corpus);                             //   + multi-token head
//   var text = llm.Generate("the brain is");       //   + sparse MoE + latent plan
//   var ppl  = llm.Evaluate(heldOutText);          // perplexity yardstick
//   llm.DistillFrom(teacher, prompts);             // knowledge distillation
//   llm.Save(dir);   var again = PredictiveLlm.Load(dir);
//   llm.Report("report.html");                     // activation observatory
//
// Feature map (all opt-in via DecoderLmConfig, all on in Create2026):
//   BlockPattern  hybrid attention + temporal-state blocks   (Phase 5)
//   NumLoops      weight-tied looped depth + early exit      (Phase 2)
//   MtpWeight     t+2 head → self-speculative decoding       (Phase 3)
//   NumExperts    FLOP-sparse mixture of experts             (Phase 6)
//   UsePlan       latent plan, settled on the prompt         (Phase 8)
// Training uses the LR-scheduled, gradient-accumulating Trainer; evaluation is
// the Phase 0 perplexity yardstick; Report is the Phase 0 observatory.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Knobs for <see cref="PredictiveLlm.Train"/>. The defaults suit toy corpora.</summary>
public sealed record TrainingOptions
{
    /// <summary>Window length in tokens (clamped to the model's MaxSeq).</summary>
    public int SeqLen { get; init; } = 32;
    /// <summary>Window stride in tokens.</summary>
    public int Stride { get; init; } = 16;
    public int Epochs { get; init; } = 6;
    /// <summary>Windows per gradient-accumulating optimizer step.</summary>
    public int BatchWindows { get; init; } = 8;
    public float PeakLr { get; init; } = 0.01f;
    public float MinLr { get; init; } = 0.001f;
    public float ClipNorm { get; init; } = 1f;
    /// <summary>Every N-th window is held out for perplexity (never trained on).</summary>
    public int HeldOutEvery { get; init; } = 5;
    /// <summary>Cap on training windows (null = use the whole corpus).</summary>
    public int? MaxWindows { get; init; } = null;
    /// <summary>Called after each epoch with (epoch number 1-based, mean loss of the epoch) — progress reporting.</summary>
    public Action<int, float>? OnEpochEnd { get; init; } = null;
}

/// <summary>What a training run did, in numbers.</summary>
public sealed record TrainingReport(
    double PerplexityBefore, double PerplexityAfter,
    float FirstLoss, float LastLoss,
    int OptimizerSteps, int TrainWindows, int EvalWindows)
{
    public override string ToString() =>
        $"ppl {PerplexityBefore:F1} → {PerplexityAfter:F1} | loss {FirstLoss:F3} → {LastLoss:F3} " +
        $"| {OptimizerSteps} steps over {TrainWindows} windows ({EvalWindows} held out)";
}

/// <summary>Knobs for <see cref="PredictiveLlm.GenerateDetailed"/>.</summary>
public sealed record GenerationOptions
{
    /// <summary>Settle the latent plan on the prompt before decoding (0 = off; needs UsePlan).</summary>
    public int SettlePlanSteps { get; init; } = 0;
    public float SettlePlanLr { get; init; } = 0.01f;
    /// <summary>Stop at the tokenizer's EOS token.</summary>
    public bool StopAtEos { get; init; } = true;
    /// <summary>
    /// Learn from the prompt before generating (one <see cref="PredictiveLlm.Learn"/>
    /// step): the model absorbs the text it is being used on — training while using.
    /// </summary>
    public bool LearnFromPrompt { get; init; } = false;
}

/// <summary>What one online-learning step did: how surprising the text was, before and after absorbing it.</summary>
public sealed record OnlineLearnReport(float SurpriseBefore, float SurpriseAfter, int Steps)
{
    public override string ToString() => $"surprise {SurpriseBefore:F3} → {SurpriseAfter:F3} ({Steps} step{(Steps == 1 ? "" : "s")})";
}

/// <summary>A generation plus its effort counters.</summary>
public sealed record GenerationResult(string Text, int Forwards, int Drafted, int Accepted)
{
    public float AcceptanceRate => Drafted > 0 ? (float)Accepted / Drafted : 0f;
}

public sealed class PredictiveLlm
{
    /// <summary>The underlying trainable model — every roadmap feature lives on its config.</summary>
    public DecoderLM Model { get; }

    public ITokenizer Tokenizer { get; }

    public DecoderLmConfig Config => Model.Config;

    public long ParameterCount => Model.ParameterCount;

    /// <summary>The Step-3.5-Flash metric: parameters a token actually runs through.</summary>
    public long ActiveParameterCountPerToken => Model.ActiveParameterCountPerToken;

    // Long-lived optimizer for online learning: Learn() keeps absorbing across
    // calls with its momentum intact — the model is a lifelong learner, not a
    // train-once artifact. (Not persisted by Save: moments restart on Load.)
    private Adam? _onlineOptimizer;

    public PredictiveLlm(DecoderLmConfig config, ITokenizer? tokenizer = null, int seed = 42)
    {
        Tokenizer = tokenizer ?? new ByteTokenizer();
        if (config.Vocab != Tokenizer.VocabSize)
            throw new ArgumentException($"Config vocab ({config.Vocab}) must match the tokenizer ({Tokenizer.VocabSize}).");
        Model = new DecoderLM(config, new Random(seed));
    }

    /// <summary>
    /// The 2026 preset: EVERY roadmap mechanism on at once — hybrid
    /// temporal/attention pattern, weight-tied loops with early exit, the t+2
    /// head (self-speculative decoding), FLOP-sparse experts in the attention
    /// blocks, and the latent plan. Byte tokenizer, toy-friendly sizes.
    /// </summary>
    public static PredictiveLlm Create2026(
        int dim = 64, int blocks = 4, int heads = 4, int ffDim = 128, int maxSeq = 128, int seed = 42)
    {
        var tok = new ByteTokenizer();
        var pattern = new char[blocks];
        for (int b = 0; b < blocks; b++) pattern[b] = b % 2 == 0 ? 'T' : 'A';   // T A T A …

        var cfg = new DecoderLmConfig(
            tok.VocabSize, dim, blocks, heads, ffDim, maxSeq,
            NumLoops: 2, LoopExitTau: 0.05f,
            MtpWeight: 0.3f,
            BlockPattern: new string(pattern),
            NumExperts: 4, MoeTopK: 1,
            UsePlan: true);
        return new PredictiveLlm(cfg, tok, seed);
    }

    // ── Train ────────────────────────────────────────────────────────────────

    /// <summary>
    /// End-to-end training on a plain-text corpus: windowed dataset, held-out
    /// split, LR-scheduled gradient-accumulating steps, perplexity before and
    /// after on the held-out windows.
    /// </summary>
    public TrainingReport Train(string corpus, TrainingOptions? options = null)
    {
        var opts   = options ?? new TrainingOptions();
        int seqLen = Math.Min(opts.SeqLen, Config.MaxSeq);
        var data   = new StreamingTextDataset(corpus.Split('\n'), Tokenizer, seqLen, opts.Stride);

        var evalSet  = PerplexityEvaluator.HeldOut(data.Windows(), opts.HeldOutEvery).ToList();
        var trainSet = PerplexityEvaluator.TrainingSplit(data.Windows(), opts.HeldOutEvery).ToList();
        if (opts.MaxWindows is { } cap && trainSet.Count > cap) trainSet = trainSet.GetRange(0, cap);
        if (trainSet.Count == 0)
            throw new InvalidOperationException(
                $"Corpus too small: no training window of {seqLen}+1 tokens. Shorten SeqLen or feed more text.");
        if (evalSet.Count == 0) evalSet = [WholeTextWindow(corpus)];

        double before = PerplexityEvaluator.Evaluate(Model, evalSet).Perplexity;

        var batches    = trainSet.Chunk(opts.BatchWindows).ToArray();
        int totalSteps = opts.Epochs * batches.Length;
        var schedule   = new LrSchedule
        {
            PeakLr = opts.PeakLr, MinLr = opts.MinLr,
            WarmupSteps = Math.Max(1, totalSteps / 10), TotalSteps = totalSteps,
        };
        var optimizer = new Adam();
        var ps        = Model.Parameters();

        float first = 0f, last = 0f;
        int   step  = 0;
        for (int epoch = 0; epoch < opts.Epochs; epoch++)
        {
            double epochSum = 0d;
            foreach (var batch in batches)
            {
                last = Trainer.Step(ps, batch, Loss, optimizer, schedule.At(step), opts.ClipNorm);
                if (step == 0) first = last;
                epochSum += last;
                step++;
            }
            opts.OnEpochEnd?.Invoke(epoch + 1, (float)(epochSum / batches.Length));
        }

        double after = PerplexityEvaluator.Evaluate(Model, evalSet).Perplexity;
        return new TrainingReport(before, after, first, last, step, trainSet.Count, evalSet.Count);

        Tensor Loss(int[] t, int[] y, bool training) => Model.Loss(t, y, ignoreIndex: -1, training);
    }

    // ── Generate ─────────────────────────────────────────────────────────────

    /// <summary>Greedy generation (self-speculative when the t+2 head is on). Returns the new text only.</summary>
    public string Generate(string prompt, int maxNewTokens = 64, GenerationOptions? options = null)
        => GenerateDetailed(prompt, maxNewTokens, options).Text;

    /// <summary>
    /// Greedy generation with the effort counters. When the model has the
    /// multi-token head, decoding is self-speculative — identical output to
    /// plain greedy, fewer forwards. With <see cref="GenerationOptions.SettlePlanSteps"/>
    /// the latent plan is first settled on the prompt (and restored after, so
    /// the model stays stateless between calls).
    /// </summary>
    public GenerationResult GenerateDetailed(string prompt, int maxNewTokens = 64, GenerationOptions? options = null)
    {
        var opts   = options ?? new GenerationOptions();
        var ctxIds = Tokenizer.Encode(prompt, addBos: true);
        if (ctxIds.Count >= Config.MaxSeq)
            throw new ArgumentException(
                $"Prompt is {ctxIds.Count} tokens; the window holds {Config.MaxSeq}. Shorten the prompt.");

        // Training while using: absorb the observed prompt into the weights
        // first, then settle the states, then predict.
        if (opts.LearnFromPrompt && ctxIds.Count >= 2)
            Learn(prompt);

        float[]? savedPlan = null;
        if (opts.SettlePlanSteps > 0 && Config.UsePlan && ctxIds.Count >= 2)
        {
            savedPlan = Model.GetPlan();
            Model.SettlePlan([.. ctxIds], opts.SettlePlanSteps, opts.SettlePlanLr);
        }

        try
        {
            int eos = opts.StopAtEos ? Tokenizer.Eos : -1;
            var run = SelfSpeculative.Generate(Model, ctxIds, maxNewTokens, eos);
            string text = Tokenizer.Decode(run.Tokens[ctxIds.Count..]);
            return new GenerationResult(text, run.Forwards, run.Drafted, run.Accepted);
        }
        finally
        {
            if (savedPlan is not null) Model.SetPlan(savedPlan);
        }
    }

    // ── Evaluate ─────────────────────────────────────────────────────────────

    /// <summary>Perplexity of the model on <paramref name="text"/> (the Phase 0 yardstick).</summary>
    public PerplexityResult Evaluate(string text, int? maxWindows = null)
        => PerplexityEvaluator.Evaluate(Model, TextWindows(text), maxWindows: maxWindows);

    // ── Learn while using (the predictive-coding symmetry) ───────────────────

    /// <summary>
    /// Online learning: the model ABSORBS <paramref name="text"/> right now —
    /// a few gradient steps on what was just observed, with a persistent
    /// optimizer so successive calls keep building on each other.
    ///
    /// This is the other half of the predictive-coding loop the product was
    /// missing: settling (adaptive effort, loops, the latent plan) descends the
    /// prediction error in the STATES; Learn descends the same surprise in the
    /// WEIGHTS — the core's <c>Network.Learn</c> does exactly these two moves.
    /// By design it only ever learns from OBSERVED text, never from the model's
    /// own output (self-feeding collapses). Returns how surprising the text was
    /// before and after.
    /// </summary>
    public OnlineLearnReport Learn(string text, int steps = 1, float learningRate = 0.001f)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), steps, "steps must be >= 1.");
        var windows = TextWindows(text);
        _onlineOptimizer ??= new Adam();
        var ps = Model.Parameters();

        float before = MeanLoss(windows);
        for (int s = 0; s < steps; s++)
            Trainer.Step(ps, windows, LossFn, _onlineOptimizer, learningRate, clipNorm: 1f);
        float after = MeanLoss(windows);
        return new OnlineLearnReport(before, after, steps);

        Tensor LossFn(int[] t, int[] y, bool training) => Model.Loss(t, y, ignoreIndex: -1, training);

        float MeanLoss(List<(int[] input, int[] target)> ws)
        {
            double sum = 0d;
            foreach (var (t, y) in ws) sum += Model.Loss(t, y, ignoreIndex: -1).Data[0];
            return (float)(sum / ws.Count);
        }
    }

    // ── Distill ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Offline knowledge distillation: the teacher (any string → string —
    /// e.g. <c>p =&gt; lmStudio.ChatAsync(p).Result.Text</c>) answers each
    /// prompt once (cached on disk), the model trains on the completions.
    /// Returns the last step's mean loss.
    /// </summary>
    public float DistillFrom(
        Func<string, string> teacher, IReadOnlyList<string> prompts,
        int steps = 20, float learningRate = 0.01f, string? cacheDir = null)
    {
        var data = LmDistiller.BuildDataset(prompts, teacher, cacheDir);
        var opt  = new Adam();
        float loss = 0f;
        for (int s = 0; s < steps; s++)
            loss = LmDistiller.Step(Model, Tokenizer, data, opt, learningRate, clipNorm: 1f);
        return loss;
    }

    /// <summary>
    /// One SEED-style on-policy round: the model drafts, the teacher corrects
    /// the model's own draft, the model trains on the correction.
    /// </summary>
    public float DistillOnPolicy(
        Func<string, string, string> teacherCorrect, IReadOnlyList<string> prompts,
        float learningRate = 0.01f, int draftTokens = 16)
        => LmDistiller.OnPolicyRound(Model, Tokenizer, prompts, teacherCorrect, new Adam(), learningRate, draftTokens, clipNorm: 1f);

    // ── Save / Load ──────────────────────────────────────────────────────────

    /// <summary>Saves config + weights under <paramref name="directory"/> (created if missing).</summary>
    public void Save(string directory)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"),
            JsonSerializer.Serialize(Config, new JsonSerializerOptions { WriteIndented = true }));
        Checkpoint.Save(Path.Combine(directory, "weights"), Model.Parameters());
    }

    /// <summary>
    /// Reloads a saved model. Pass the same non-default tokenizer used at save
    /// time (the byte tokenizer is stateless and needs nothing).
    /// </summary>
    public static PredictiveLlm Load(string directory, ITokenizer? tokenizer = null, int seed = 42)
    {
        var cfg = JsonSerializer.Deserialize<DecoderLmConfig>(
                      File.ReadAllText(Path.Combine(directory, "config.json")))
                  ?? throw new InvalidDataException($"Unreadable config in {directory}.");
        var llm = new PredictiveLlm(cfg, tokenizer, seed);
        Checkpoint.Load(Path.Combine(directory, "weights"), llm.Model.Parameters());
        return llm;
    }

    // ── Report ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs the activation observatory (block stats, sink shares, loop passes,
    /// MoE row counts) plus perplexity on <paramref name="evalText"/> (default:
    /// the built-in yardstick corpus) and writes the self-contained HTML report.
    /// </summary>
    public string Report(string path, string? evalText = null)
    {
        string text = evalText ?? EvalCorpus.Tiny;

        var probe = new ActivationProbe();
        var probeIds = Tokenizer.Encode(text.Split('\n')[0], addBos: true);
        int take = Math.Min(probeIds.Count, Config.MaxSeq);
        Model.Forward([.. probeIds.Take(take)], training: false, probe);

        var ppl = Evaluate(text);
        return HtmlReport.Save(path, "PredictiveLlm yardstick", ppl, probe);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    // Windows a text for evaluation/online learning; short texts fall back to
    // one whole-text window.
    private List<(int[] input, int[] target)> TextWindows(string text)
    {
        int seqLen = Math.Min(32, Config.MaxSeq);
        var data   = new StreamingTextDataset(text.Split('\n'), Tokenizer, seqLen, Math.Max(1, seqLen / 2));
        var windows = data.Windows().ToList();
        if (windows.Count == 0) windows = [WholeTextWindow(text)];
        return windows;
    }

    // Fallback eval window for texts too short to fill a streaming window.
    private (int[] input, int[] target) WholeTextWindow(string text)
    {
        var ids = Tokenizer.Encode(text, addBos: true, addEos: true);
        int n   = Math.Min(ids.Count, Config.MaxSeq + 1);
        if (n < 2)
            throw new InvalidOperationException("Text too short to evaluate (needs at least 2 tokens).");
        var input  = new int[n - 1];
        var target = new int[n - 1];
        for (int i = 0; i < n - 1; i++) { input[i] = ids[i]; target[i] = ids[i + 1]; }
        return (input, target);
    }
}
