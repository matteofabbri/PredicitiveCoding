using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using PredictiveCoding.Core;
using PredictiveCoding.Model;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Serving;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredictiveCoding.LLM;

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
    // train-once artifact. Persisted by Save (online-optimizer.bin) and restored
    // by Load, so the momentum survives a save/reload too — a reloaded learner
    // continues, it does not restart.
    private Adam? _onlineOptimizer;

    // Seed of the primary model — reused (offset per worker) to build the
    // structural replicas of the parallel trainer, then overwritten by the
    // synced weights anyway.
    private readonly int _seed;

    public PredictiveLlm(DecoderLmConfig config, ITokenizer? tokenizer = null, int seed = 42)
    {
        Tokenizer = tokenizer ?? new ByteTokenizer();
        if (config.Vocab != Tokenizer.VocabSize)
            throw new ArgumentException($"Config vocab ({config.Vocab}) must match the tokenizer ({Tokenizer.VocabSize}).");
        _seed = seed;
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
        return new PredictiveLlm(Config2026(tok.VocabSize, dim, blocks, heads, ffDim, maxSeq), tok, seed);
    }

    /// <summary>
    /// The 2026 preset on a REAL tokenizer (Pro roadmap Phase 0): trains a
    /// byte-level BPE vocabulary on <paramref name="tokenizerCorpus"/> first,
    /// then builds the model on it. Sequences shrink (~2-4× fewer tokens per
    /// text), so every window sees more context; compare models across
    /// tokenizers with <see cref="EvaluateBitsPerByte"/>, never with
    /// per-token perplexity.
    /// </summary>
    public static PredictiveLlm Create2026(
        string tokenizerCorpus, int merges = 512,
        int dim = 64, int blocks = 4, int heads = 4, int ffDim = 128, int maxSeq = 128, int seed = 42)
    {
        var tok = new BpeTokenizer();
        tok.Train(tokenizerCorpus, merges);
        return new PredictiveLlm(Config2026(tok.VocabSize, dim, blocks, heads, ffDim, maxSeq), tok, seed);
    }

    private static DecoderLmConfig Config2026(int vocab, int dim, int blocks, int heads, int ffDim, int maxSeq)
    {
        var pattern = new char[blocks];
        for (int b = 0; b < blocks; b++) pattern[b] = b % 2 == 0 ? 'T' : 'A';   // T A T A …

        return new DecoderLmConfig(
            vocab, dim, blocks, heads, ffDim, maxSeq,
            NumLoops: 2, LoopExitTau: 0.05f,
            MtpWeight: 0.3f,
            BlockPattern: new string(pattern),
            NumExperts: 4, MoeTopK: 1,
            UsePlan: true);
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
        return TrainCore(data, opts, seqLen, hashSplit: false, () => WholeTextWindow(corpus));
    }

    /// <summary>
    /// End-to-end training from a <see cref="CorpusSource"/> (roadmap-Pro Phase 4):
    /// files/folders/globs, weighted mixing and hygiene, streamed. The held-out
    /// split is CONTENT-HASHED, so the same windows stay in eval across runs even
    /// when the corpus is reordered, re-mixed or grown — perplexity stays
    /// comparable between sessions.
    /// </summary>
    public TrainingReport Train(CorpusSource source, TrainingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        var opts   = options ?? new TrainingOptions();
        int seqLen = Math.Min(opts.SeqLen, Config.MaxSeq);
        var data   = new StreamingTextDataset(source.Lines(), Tokenizer, seqLen, opts.Stride);
        return TrainCore(data, opts, seqLen, hashSplit: true, fallback: null);
    }

    // Shared training core. hashSplit selects the content-hashed held-out split
    // (Phase 4) over the positional one; fallback supplies an eval window when the
    // corpus is too small to spare one (null ⇒ borrow the last training window).
    private TrainingReport TrainCore(
        StreamingTextDataset data, TrainingOptions opts, int seqLen, bool hashSplit,
        Func<(int[] input, int[] target)>? fallback)
    {
        var evalSet = hashSplit
            ? PerplexityEvaluator.HeldOutHashed(data.Windows(), opts.HeldOutEvery).ToList()
            : PerplexityEvaluator.HeldOut(data.Windows(), opts.HeldOutEvery).ToList();
        var trainSet = hashSplit
            ? PerplexityEvaluator.TrainingSplitHashed(data.Windows(), opts.HeldOutEvery).ToList()
            : PerplexityEvaluator.TrainingSplit(data.Windows(), opts.HeldOutEvery).ToList();

        if (opts.MaxWindows is { } cap && trainSet.Count > cap) trainSet = trainSet.GetRange(0, cap);
        if (trainSet.Count == 0)
            throw new InvalidOperationException(
                $"Corpus too small: no training window of {seqLen}+1 tokens. Shorten SeqLen or feed more text.");
        if (evalSet.Count == 0) evalSet = [fallback?.Invoke() ?? trainSet[^1]];

        var beforeEval = PerplexityEvaluator.Evaluate(Model, evalSet);
        double before  = beforeEval.Perplexity;
        Odometer.Evaluated += beforeEval.Tokens;

        var batches    = trainSet.Chunk(opts.BatchWindows).ToArray();
        int totalSteps = opts.Epochs * batches.Length;
        var schedule   = new LrSchedule
        {
            PeakLr = opts.PeakLr, MinLr = opts.MinLr,
            WarmupSteps = Math.Max(1, totalSteps / 10), TotalSteps = totalSteps,
        };
        // Resume from a loaded optimizer (true resume) or start fresh.
        var optimizer = opts.ResumeOptimizer ?? new Adam();
        var ps        = Model.Parameters();

        if (opts.CheckpointEverySteps > 0 && string.IsNullOrEmpty(opts.CheckpointDir))
            throw new ArgumentException("CheckpointEverySteps > 0 requires CheckpointDir.", nameof(opts));
        double bestPpl  = double.PositiveInfinity;
        string? bestDir = null;

        // Parallel micro-batch (Pro Phase 3): 0 = auto, else the requested count,
        // capped at the batch size (no idle workers). 1 ⇒ the serial path.
        int workers = opts.Workers == 0 ? Environment.ProcessorCount : opts.Workers;
        workers = Math.Max(1, Math.Min(workers, opts.BatchWindows));

        // Structural replicas share the primary's shapes; StepParallel syncs their
        // weights in each step, so their initial values are irrelevant.
        List<IReadOnlyList<Tensor>>? replicaPs = null;
        List<Func<int[], int[], bool, Tensor>>? replicaLoss = null;
        if (workers > 1)
        {
            var models  = new List<DecoderLM> { Model };
            for (int w = 1; w < workers; w++) models.Add(new DecoderLM(Config, new Random(_seed + w)));
            replicaPs   = [.. models.Select(m => m.Parameters())];
            replicaLoss = [.. models.Select(m => (Func<int[], int[], bool, Tensor>)
                                ((t, y, tr) => m.Loss(t, y, ignoreIndex: -1, tr)))];
        }

        var keepRoute = Ag.MatMulRoute;
        if (opts.AutoGemmRoute) Ag.MatMulRoute = MatMulRoute.Auto;

        float first = 0f, last = 0f;
        int   step  = 0;
        try
        {
            for (int epoch = 0; epoch < opts.Epochs; epoch++)
            {
                double epochSum = 0d;
                foreach (var batch in batches)
                {
                    last = workers > 1
                        ? Trainer.StepParallel(replicaPs!, replicaLoss!, batch, optimizer, schedule.At(step), opts.ClipNorm)
                        : Trainer.Step(ps, batch, Loss, optimizer, schedule.At(step), opts.ClipNorm);
                    if (step == 0) first = last;
                    epochSum += last;
                    step++;

                    // Periodic checkpoint (Phase 5): evaluate held-out ppl and,
                    // when keeping the best, write only when it improves.
                    if (opts.CheckpointEverySteps > 0 && step % opts.CheckpointEverySteps == 0)
                    {
                        double ppl = PerplexityEvaluator.Evaluate(Model, evalSet).Perplexity;
                        if (!opts.KeepBestByPerplexity || ppl < bestPpl)
                        {
                            bestPpl = ppl;
                            bestDir = opts.CheckpointDir!;
                            SaveResumable(bestDir, optimizer, opts.CheckpointPrecision);
                        }
                    }
                }
                opts.OnEpochEnd?.Invoke(epoch + 1, (float)(epochSum / batches.Length));
            }
        }
        finally { Ag.MatMulRoute = keepRoute; }

        var afterEval = PerplexityEvaluator.Evaluate(Model, evalSet);
        double after  = afterEval.Perplexity;
        Odometer.Evaluated += afterEval.Tokens;
        long epochTokens = 0;
        foreach (var (input, _) in trainSet) epochTokens += input.Length;
        Odometer.Trained += epochTokens * opts.Epochs;

        return new TrainingReport(before, after, first, last, step, trainSet.Count, evalSet.Count)
        {
            BestPerplexity     = bestDir is null ? double.NaN : bestPpl,
            BestCheckpointPath = bestDir,
        };

        Tensor Loss(int[] t, int[] y, bool training) => Model.Loss(t, y, ignoreIndex: -1, training);
    }

    // ── Episodic memory (roadmap-Pro Phase 7) ────────────────────────────────

    /// <summary>
    /// The optional hippocampal store. Attach one (dim = <see cref="Config"/>.Dim)
    /// to enable <see cref="Remember"/> and <see cref="GenerationOptions.UseMemory"/>.
    /// Persisted with the model when present.
    /// </summary>
    public EpisodicMemory? Memory { get; set; }

    /// <summary>
    /// The model's lifetime token counter, by activity (trained / evaluated /
    /// absorbed / prompted / generated) — persisted with the model, so it keeps
    /// counting across sessions. Compute-side counters live on
    /// <see cref="DecoderLM.ForwardTokens"/> (and <c>GptModel.ForwardTokens</c>
    /// for the served stack). Train's periodic checkpoint probes are not
    /// itemized — only the before/after pair counts as Evaluated.
    /// </summary>
    public TokenOdometer Odometer { get; } = new();

    /// <summary>
    /// Mean-pooled final hidden state of <paramref name="text"/> [Dim] — the
    /// retrieval key for episodic memory. Every forward already computes this
    /// representation; here it is pooled over the sequence into one vector.
    /// </summary>
    public float[] Embed(string text)
    {
        var ids = Tokenizer.Encode(text, addBos: true);
        if (ids.Count == 0) ids = [Tokenizer.Bos];
        var h   = Model.HiddenStates([.. ids]);          // [seq, Dim]
        int dim = h.Cols, seq = h.Rows;

        // Pool over the CONTENT positions, skipping BOS at position 0 (it carries
        // no content and is common to every text, so it only dilutes the key).
        int from  = seq > 1 ? 1 : 0;
        int count = seq - from;
        var pooled = new float[dim];
        for (int r = from; r < seq; r++)
            for (int c = 0; c < dim; c++) pooled[c] += h.Data[r * dim + c];
        float inv = count > 0 ? 1f / count : 1f;
        for (int c = 0; c < dim; c++) pooled[c] *= inv;
        return pooled;
    }

    /// <summary>
    /// Stores <paramref name="text"/> in episodic memory under its embedding key
    /// (creates the store on first use). The hippocampal half of the loop:
    /// facts observed now become retrievable later, across sessions.
    /// </summary>
    public void Remember(string text)
    {
        Memory ??= new EpisodicMemory(Config.Dim);
        Memory.Remember(Embed(text), text);
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
        var opts = options ?? new GenerationOptions();

        // Recall into generation (Phase 7): prepend the nearest remembered texts
        // as context, newest-first, while they fit the window alongside the prompt.
        if (opts.UseMemory && Memory is { Count: > 0 })
            prompt = PrependRecalled(prompt, opts.MemoryTopK);

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
            if (opts.Sampling is { } so)
                return SampleLoop(ctxIds, maxNewTokens, opts, so);

            int eos = opts.StopAtEos ? Tokenizer.Eos : -1;
            var run = SelfSpeculative.Generate(Model, ctxIds, maxNewTokens, eos);
            string text = Tokenizer.Decode(run.Tokens[ctxIds.Count..]);
            CutAtStop(ref text, opts.StopSequences);
            Odometer.Prompted  += ctxIds.Count;
            Odometer.Generated += run.Tokens.Length - ctxIds.Count;
            return new GenerationResult(text, run.Forwards, run.Drafted, run.Accepted);
        }
        finally
        {
            if (savedPlan is not null) Model.SetPlan(savedPlan);
        }
    }

    // Stochastic decoding (Pro roadmap Phase 2): token-by-token sampling via the
    // Generation.Sampler — penalties see the whole visible sequence (prompt +
    // generated), stop sequences end the loop early and save the forwards.
    private GenerationResult SampleLoop(IReadOnlyList<int> ctxIds, int maxNewTokens, GenerationOptions opts, SamplingOptions so)
    {
        var sampler   = new Sampler(so.Seed ?? 42);        // reproducible by default in the product
        var seq       = new List<int>(ctxIds);
        var generated = new List<int>();
        int vocab     = Config.Vocab;
        int forwards  = 0;
        string text   = "";

        while (generated.Count < maxNewTokens && seq.Count < Config.MaxSeq)
        {
            var logits = Model.Forward([.. seq]);
            forwards++;

            var row = new DenseTensor<float>(vocab);
            Array.Copy(logits.Data, (seq.Count - 1) * vocab, row.Buffer, 0, vocab);
            int next = sampler.Sample(row, seq, so);

            if (opts.StopAtEos && next == Tokenizer.Eos) break;
            generated.Add(next);
            seq.Add(next);

            text = Tokenizer.Decode(generated);
            if (CutAtStop(ref text, opts.StopSequences)) break;
        }
        Odometer.Prompted  += ctxIds.Count;
        Odometer.Generated += generated.Count;
        return new GenerationResult(text, forwards, Drafted: 0, Accepted: 0);
    }

    // Cuts the text just before the first stop sequence; true when a cut happened.
    private static bool CutAtStop(ref string text, string[]? stops)
    {
        if (stops is null) return false;
        int cut = -1;
        foreach (string s in stops)
        {
            if (string.IsNullOrEmpty(s)) continue;
            int i = text.IndexOf(s, StringComparison.Ordinal);
            if (i >= 0 && (cut < 0 || i < cut)) cut = i;
        }
        if (cut < 0) return false;
        text = text[..cut];
        return true;
    }

    // ── Serve ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Exports the trained weights onto the O(n) inference stack and returns a
    /// server that decodes incrementally through the KV cache — one forward per
    /// token instead of a full re-forward (ROADMAP-Pro Phase 1). Requires
    /// NumLoops = 1 (the served stack is single-pass); the export is a snapshot:
    /// later training or Learn calls need a new server.
    /// </summary>
    public LlmServer CreateServer() => new(GptExporter.Export(Model), Tokenizer);

    /// <summary>
    /// Opens a chat session on this model (Pro roadmap Phase 2): role-tagged
    /// history via <see cref="Generation.ChatTemplate"/>, replies stopped at the
    /// next role tag. Learning during chat follows the self-feeding rule: only
    /// the USER's words can be absorbed, never the assistant's own output.
    /// </summary>
    public ChatSession StartChat(string? system = null) => new(this, system);

    // ── Evaluate ─────────────────────────────────────────────────────────────

    /// <summary>Perplexity of the model on <paramref name="text"/> (the Phase 0 yardstick).</summary>
    public PerplexityResult Evaluate(string text, int? maxWindows = null)
    {
        var result = PerplexityEvaluator.Evaluate(Model, TextWindows(text), maxWindows: maxWindows);
        Odometer.Evaluated += result.Tokens;
        return result;
    }

    /// <summary>
    /// Bits per byte of <paramref name="text"/> — the tokenizer-independent
    /// yardstick (Pro roadmap Phase 0): per-token perplexity cannot compare a
    /// byte model against a BPE model, bits spent per byte of text can.
    /// Non-overlapping windows; bytes counted on the decoded scored targets.
    /// </summary>
    public BitsPerByteResult EvaluateBitsPerByte(string text)
    {
        int seqLen  = Math.Min(32, Config.MaxSeq);
        var data    = new StreamingTextDataset(text.Split('\n'), Tokenizer, seqLen, stride: seqLen);
        var windows = data.Windows().ToList();
        if (windows.Count == 0) windows = [WholeTextWindow(text)];

        var ppl = PerplexityEvaluator.Evaluate(Model, windows);
        Odometer.Evaluated += ppl.Tokens;
        long bytes = 0;
        foreach (var (_, target) in windows)
            bytes += Encoding.UTF8.GetByteCount(Tokenizer.Decode(target));

        double totalBits = ppl.NllPerToken * ppl.Tokens / Math.Log(2);
        return new BitsPerByteResult(totalBits / Math.Max(1, bytes), bytes, ppl.Tokens);
    }

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

        long windowTokens = 0;
        foreach (var (input, _) in windows) windowTokens += input.Length;
        Odometer.Absorbed  += windowTokens * steps;
        Odometer.Evaluated += windowTokens * 2;           // the before/after surprise probes

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

    /// <summary>
    /// Saves config + weights + TOKENIZER under <paramref name="directory"/>
    /// (created if missing). The tokenizer travels with the model (Pro roadmap
    /// Phase 0): a BPE vocab is written next to the weights and fingerprinted,
    /// so <see cref="Load"/> reconstructs exactly the vocabulary the weights
    /// were trained on — or fails loudly.
    /// </summary>
    public void Save(string directory) => Save(directory, WeightStoragePrecision.Float32);

    /// <summary>
    /// <see cref="Save(string)"/> with an explicit on-disk weight precision
    /// (roadmap-Pro Phase 5): Half/BFloat16 halve the weight file, Int8 quarters
    /// it; <see cref="Load"/> widens back to fp32 within the format's tolerance.
    /// </summary>
    public void Save(string directory, WeightStoragePrecision precision)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"),
            JsonSerializer.Serialize(Config, new JsonSerializerOptions { WriteIndented = true }));
        Checkpoint.Save(Path.Combine(directory, "weights"), Model.Parameters(), precision);

        string kind = Tokenizer switch
        {
            BpeTokenizer  => "bpe",
            ByteTokenizer => "byte",
            _             => Tokenizer.GetType().Name,   // custom: caller re-supplies it at Load
        };
        string hash = "";
        if (Tokenizer is BpeTokenizer bpe)
        {
            string vocabPath = Path.Combine(directory, "tokenizer.bpe");
            bpe.Save(vocabPath);
            hash = Fingerprint(File.ReadAllBytes(vocabPath));
        }
        File.WriteAllText(Path.Combine(directory, "tokenizer.json"),
            JsonSerializer.Serialize(new TokenizerManifest(kind, Tokenizer.VocabSize, hash)));

        // Episodic memory travels with the model when present (Phase 7).
        string memPath = Path.Combine(directory, "memory.bin");
        if (Memory is not null) Memory.Save(memPath);
        else if (File.Exists(memPath)) File.Delete(memPath);

        // The online-learning optimizer's momentum travels too: without it a
        // reloaded lifelong learner would restart its Adam moments and take the
        // next Learn step less stably than an uninterrupted one. (The training
        // optimizer of Train is separate — resumed via TrainingOptions.ResumeOptimizer.)
        string onlineOptPath = Path.Combine(directory, "online-optimizer.bin");
        if (_onlineOptimizer is not null)
            OptimizerCheckpoint.Save(onlineOptPath, Model.Parameters(), _onlineOptimizer);
        else if (File.Exists(onlineOptPath)) File.Delete(onlineOptPath);

        // The odometer travels too: a real odometer keeps counting across sessions.
        File.WriteAllText(Path.Combine(directory, "odometer.json"),
            JsonSerializer.Serialize(Odometer.ToSnapshot()));
    }

    /// <summary>
    /// Reloads a saved model, tokenizer included: "byte" needs nothing, "bpe"
    /// is reloaded from the persisted vocab (fingerprint-verified), a custom
    /// tokenizer must be re-supplied via <paramref name="tokenizer"/> and match
    /// the recorded vocab size. Any mismatch fails loudly.
    /// </summary>
    public static PredictiveLlm Load(string directory, ITokenizer? tokenizer = null, int seed = 42)
    {
        var cfg = JsonSerializer.Deserialize<DecoderLmConfig>(
                      File.ReadAllText(Path.Combine(directory, "config.json")))
                  ?? throw new InvalidDataException($"Unreadable config in {directory}.");

        string manifestPath = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(manifestPath))
        {
            var manifest = JsonSerializer.Deserialize<TokenizerManifest>(File.ReadAllText(manifestPath))
                           ?? throw new InvalidDataException($"Unreadable tokenizer manifest in {directory}.");
            switch (manifest.Kind)
            {
                case "byte":
                    tokenizer ??= new ByteTokenizer();
                    break;
                case "bpe":
                {
                    if (tokenizer is not null)
                        throw new ArgumentException(
                            "This model carries its own BPE tokenizer — do not pass one to Load.", nameof(tokenizer));
                    string vocabPath = Path.Combine(directory, "tokenizer.bpe");
                    string actual    = Fingerprint(File.ReadAllBytes(vocabPath));
                    if (actual != manifest.Hash)
                        throw new InvalidDataException(
                            $"Tokenizer vocab in {directory} does not match the model's fingerprint " +
                            $"(expected {manifest.Hash}, found {actual}) — the file was modified or corrupted.");
                    tokenizer = BpeTokenizer.Load(vocabPath);
                    break;
                }
                default:
                    if (tokenizer is null)
                        throw new ArgumentException(
                            $"This model was saved with a custom tokenizer ({manifest.Kind}) — pass the same one to Load.",
                            nameof(tokenizer));
                    break;
            }
            if (tokenizer.VocabSize != manifest.VocabSize)
                throw new InvalidDataException(
                    $"Tokenizer vocab size {tokenizer.VocabSize} does not match the saved model's {manifest.VocabSize}.");
        }

        var llm = new PredictiveLlm(cfg, tokenizer, seed);
        Checkpoint.Load(Path.Combine(directory, "weights"), llm.Model.Parameters());

        string memPath = Path.Combine(directory, "memory.bin");
        if (File.Exists(memPath)) llm.Memory = EpisodicMemory.Load(memPath, cfg.Dim);

        // Restore the online-learning optimizer so lifelong Learn continues its
        // momentum instead of restarting it (keyed onto the reloaded parameters).
        string onlineOptPath = Path.Combine(directory, "online-optimizer.bin");
        if (File.Exists(onlineOptPath))
        {
            var opt = new Adam();
            OptimizerCheckpoint.Load(onlineOptPath, llm.Model.Parameters(), opt);
            llm._onlineOptimizer = opt;
        }

        string odoPath = Path.Combine(directory, "odometer.json");
        if (File.Exists(odoPath))
        {
            var snap = JsonSerializer.Deserialize<TokenOdometer.Snapshot>(File.ReadAllText(odoPath));
            if (snap is not null) llm.Odometer.LoadSnapshot(snap);
        }
        return llm;
    }

    // Full resumable checkpoint: model (weights at the requested precision) +
    // Adam moments in optimizer.bin next to it (roadmap-Pro Phase 5).
    private void SaveResumable(string directory, Adam optimizer, WeightStoragePrecision precision)
    {
        Save(directory, precision);
        OptimizerCheckpoint.Save(Path.Combine(directory, "optimizer.bin"), Model.Parameters(), optimizer);
    }

    /// <summary>
    /// Loads a model AND its Adam optimizer state (roadmap-Pro Phase 5 true resume):
    /// weights via <see cref="Load"/>, moments + step counter from <c>optimizer.bin</c>
    /// when present (else a fresh optimizer). Feed the returned optimizer back via
    /// <see cref="TrainingOptions.ResumeOptimizer"/> to continue, not restart.
    /// </summary>
    public static (PredictiveLlm llm, Adam optimizer) LoadWithOptimizer(
        string directory, ITokenizer? tokenizer = null, int seed = 42)
    {
        var llm = Load(directory, tokenizer, seed);
        var opt = new Adam();
        string optPath = Path.Combine(directory, "optimizer.bin");
        if (File.Exists(optPath))
            OptimizerCheckpoint.Load(optPath, llm.Model.Parameters(), opt);
        return (llm, opt);
    }

    private static string Fingerprint(byte[] bytes)
        => Convert.ToHexStringLower(SHA256.HashData(bytes))[..16];

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

    // Prepends the nearest remembered texts (most-similar first) to the prompt,
    // keeping as many as fit the window alongside it (Phase 7 recall-into-generation).
    private string PrependRecalled(string prompt, int topK)
    {
        var hits = Memory!.Recall(Embed(prompt), topK);
        if (hits.Count == 0) return prompt;

        var kept = new List<string>();
        foreach (var (text, _) in hits)
        {
            string candidate = string.Join("\n", kept.Append(text)) + "\n" + prompt;
            if (Tokenizer.Encode(candidate, addBos: true).Count < Config.MaxSeq) kept.Add(text);
            else break;
        }
        return kept.Count == 0 ? prompt : string.Join("\n", kept) + "\n" + prompt;
    }
}
