using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;
using PredictiveCoding.LLM.Translation;

// ─────────────────────────────────────────────────────────────────────────────
// PredictiveCoding.Translate.Cli  —  build / verify / run "our own" translator.
//
//   gradcheck   numerically verify the new autograd code (masked CE + DecoderLM)
//   smoke       overfit a handful of pairs and decode them back (end-to-end check)
//   train       train on a parallel corpus (see TrainCommand)
//   translate   load a trained model and translate stdin / an argument
// ─────────────────────────────────────────────────────────────────────────────

return (args.Length > 0 ? args[0].ToLowerInvariant() : "help") switch
{
    "gradcheck" => Commands.GradCheck(),
    "smoke"     => Commands.Smoke(),
    "train"     => Commands.Train(args[1..]),
    "translate" => Commands.Translate(args[1..]),
    _           => Commands.Help(),
};

internal static class Commands
{
    public static int Help()
    {
        Console.WriteLine("usage: translate <gradcheck|smoke|train|translate> [options]");
        return 0;
    }

    // ── gradcheck ────────────────────────────────────────────────────────────────

    public static int GradCheck()
    {
        var rng = new Random(1234);
        bool ok = true;

        // (1) Masked cross-entropy: rows whose target is IgnoreIndex must contribute
        //     no gradient; check the analytic logits gradient against finite differences.
        {
            int m = 6, vocab = 10;
            var logits = AgRandn(m, vocab, rng, 1.0f);
            var targets = new int[m];
            for (int i = 0; i < m; i++) targets[i] = rng.Next(vocab);
            targets[1] = TranslationFormat.IgnoreIndex;          // masked rows
            targets[4] = TranslationFormat.IgnoreIndex;

            var (maxAbs, maxRel, pass) = GradCheck([logits],
                () => Ag.CrossEntropy(logits, targets, TranslationFormat.IgnoreIndex), rng, samplesPerParam: 60);
            ok &= pass;
            Console.WriteLine($"[gradcheck] masked cross-entropy : maxAbs={maxAbs:E2} maxRel={maxRel:E2} -> {(pass ? "PASS" : "FAIL")}");

            // Also assert masked rows truly received zero gradient.
            var L = Ag.CrossEntropy(logits, targets, TranslationFormat.IgnoreIndex); L.Backward();
            double maskedGrad = 0;
            foreach (int i in new[] { 1, 4 })
                for (int j = 0; j < vocab; j++) maskedGrad += Math.Abs(logits.Grad[i * vocab + j]);
            bool zero = maskedGrad == 0d;
            ok &= zero;
            Console.WriteLine($"[gradcheck] masked rows zero-grad : sum|g|={maskedGrad:E2} -> {(zero ? "PASS" : "FAIL")}");
        }

        // (2) Full DecoderLM (2 blocks × 2 heads): check the masked translation loss
        //     gradient for a sample of every parameter tensor.
        {
            var cfg = new DecoderLmConfig(Vocab: 12, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 16);
            var model = new DecoderLM(cfg, new Random(7));
            int seq = 6;
            var tokens  = new int[seq];
            var targets = new int[seq];
            for (int i = 0; i < seq; i++) { tokens[i] = rng.Next(cfg.Vocab); targets[i] = rng.Next(cfg.Vocab); }
            targets[0] = TranslationFormat.IgnoreIndex;
            targets[1] = TranslationFormat.IgnoreIndex;          // simulate a masked prompt prefix

            var (maxAbs, maxRel, pass) = GradCheck(model.Parameters(),
                () => model.Loss(tokens, targets, TranslationFormat.IgnoreIndex, training: false), rng, samplesPerParam: 6);
            ok &= pass;
            Console.WriteLine($"[gradcheck] DecoderLM (2x2)       : maxAbs={maxAbs:E2} maxRel={maxRel:E2} -> {(pass ? "PASS" : "FAIL")}");
        }

        Console.WriteLine(ok ? "\ngradcheck: ALL PASS" : "\ngradcheck: FAILURES");
        return ok ? 0 : 1;
    }

    // ── smoke ────────────────────────────────────────────────────────────────────

    public static int Smoke()
    {
        // A tiny fixed bilingual set: overfitting it proves the whole path works —
        // formatting + target-side masking + DecoderLM + Adam + greedy decode.
        (string src, string tgt)[] pairs =
        [
            ("hello", "ciao"), ("dog", "cane"), ("cat", "gatto"),
            ("house", "casa"), ("water", "acqua"), ("sun", "sole"),
        ];
        const string tag = "ita";

        // Train a byte-level BPE on all the text we will see.
        var tok = new BpeTokenizer();
        var corpus = string.Join("\n", pairs.SelectMany(p =>
            new[] { TranslationFormat.PromptText(p.src, tag), p.tgt }));
        tok.Train(corpus, numMerges: 80);
        Console.WriteLine($"[smoke] tokenizer vocab = {tok.VocabSize}");

        var cfg = new DecoderLmConfig(Vocab: tok.VocabSize, Dim: 48, NumBlocks: 2, NumHeads: 3, FfDim: 96, MaxSeq: 48, Dropout: 0f);
        var model = new DecoderLM(cfg, new Random(0));
        var examples = pairs.Select(p => TranslationFormat.BuildExample(tok, p.src, p.tgt, tag)).ToList();

        var opt = new Adam(weightDecay: 0f);
        var ps  = model.Parameters();
        Console.WriteLine($"[smoke] parameters = {ps.Sum(t => (long)t.Data.Length):N0}");

        const int steps = 400;
        for (int step = 1; step <= steps; step++)
        {
            float lr = 3e-3f * Math.Min(1f, step / 30f);        // short warmup
            float loss = Trainer.Step(ps, examples,
                (t, y, train) => model.Loss(t, y, TranslationFormat.IgnoreIndex, train),
                opt, lr, clipNorm: 1f);
            if (step % 50 == 0 || step == 1)
                Console.WriteLine($"[smoke] step {step,4}  loss {loss:F4}");
        }

        var translator = new Translator(model, tok);
        int correct = 0;
        Console.WriteLine("\n[smoke] decoding:");
        foreach (var (src, tgt) in pairs)
        {
            string outp = translator.Translate(src, tag, maxNewTokens: 16);
            bool hit = outp == tgt;
            correct += hit ? 1 : 0;
            Console.WriteLine($"   >>{tag}<< {src,-6} -> \"{outp}\"   (expected \"{tgt}\")  {(hit ? "OK" : "x")}");
        }

        bool pass = correct >= pairs.Length - 1;                 // allow one miss
        Console.WriteLine($"\nsmoke: {correct}/{pairs.Length} reproduced -> {(pass ? "PASS" : "FAIL")}");
        return pass ? 0 : 1;
    }

    // ── train ────────────────────────────────────────────────────────────────────

    public static int Train(string[] args)
    {
        var a = ArgMap(args);
        string? corpus = Get(a, "corpus");
        if (corpus is null) { Console.Error.WriteLine("error: --corpus <path.tsv> is required"); return 2; }
        string outDir = Get(a, "out") ?? "translator-out";

        int   dim    = IntArg(a, "dim", 256);
        int   blocks = IntArg(a, "blocks", 4);
        int   heads  = IntArg(a, "heads", 4);
        int   ff     = IntArg(a, "ff", dim * 4);
        int   maxSeq = IntArg(a, "maxseq", 96);
        int   merges = IntArg(a, "merges", 8000);
        int   epochs = IntArg(a, "epochs", 1);
        int   batch  = IntArg(a, "batch", 16);
        float peakLr = FloatArg(a, "lr", 3e-4f);
        int   limit  = IntArg(a, "limit", 0);
        int   sampleEvery = IntArg(a, "sample-every", 200);

        IEnumerable<SentencePair> stream = ParallelCorpus.ReadTsv(corpus);
        if (limit > 0) stream = stream.Take(limit);
        var pairs = stream.ToList();
        if (pairs.Count == 0) { Console.Error.WriteLine("error: no pairs read from corpus"); return 2; }
        Console.WriteLine($"[train] pairs = {pairs.Count:N0}");

        var tok = ParallelCorpus.TrainTokenizer(pairs, merges);
        Console.WriteLine($"[train] tokenizer vocab = {tok.VocabSize}");

        var examples = ParallelCorpus.BuildExamples(pairs, tok, maxSeq);
        Console.WriteLine($"[train] examples (<= maxSeq {maxSeq}) = {examples.Count:N0}");
        if (examples.Count == 0) { Console.Error.WriteLine("error: every example exceeds --maxseq"); return 2; }

        var cfg   = new DecoderLmConfig(tok.VocabSize, dim, blocks, heads, ff, maxSeq, Dropout: 0.1f);
        var model = new DecoderLM(cfg, new Random(0));
        var ps    = model.Parameters();
        Console.WriteLine($"[train] model dim={dim} blocks={blocks} heads={heads} ff={ff}  params={ps.Sum(t => (long)t.Data.Length):N0}");

        int stepsPerEpoch = (examples.Count + batch - 1) / batch;
        int totalSteps    = Math.Max(1, stepsPerEpoch * epochs);
        var sched = new LrSchedule { PeakLr = peakLr, MinLr = peakLr * 0.1f, WarmupSteps = Math.Min(200, totalSteps / 10 + 1), TotalSteps = totalSteps };
        var opt   = new Adam(weightDecay: 0.01f);
        var rng   = new Random(42);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        int step = 0;
        for (int ep = 0; ep < epochs; ep++)
        {
            Shuffle(examples, rng);
            for (int b = 0; b < examples.Count; b += batch)
            {
                var slice = examples.GetRange(b, Math.Min(batch, examples.Count - b));
                float lr   = sched.At(step);
                float loss = Trainer.Step(ps, slice,
                    (t, y, train) => model.Loss(t, y, TranslationFormat.IgnoreIndex, train), opt, lr, clipNorm: 1f);
                step++;
                if (step % 20 == 0 || step == 1)
                    Console.WriteLine($"[train] ep {ep} step {step}/{totalSteps} lr {lr:E2} loss {loss:F4}  ({step / Math.Max(1e-3, sw.Elapsed.TotalSeconds):F1} steps/s)");
                if (sampleEvery > 0 && step % sampleEvery == 0) SampleSome(model, tok, pairs, rng);
            }
            new Translator(model, tok).Save(outDir);
            Console.WriteLine($"[train] checkpoint saved to '{outDir}' (epoch {ep} done)");
        }

        new Translator(model, tok).Save(outDir);
        Console.WriteLine($"[train] finished in {sw.Elapsed.TotalSeconds:F1}s — translator saved to '{outDir}'");
        return 0;
    }

    // ── translate ────────────────────────────────────────────────────────────────

    public static int Translate(string[] args)
    {
        var a   = ArgMap(args);
        string dir = Get(a, "model") ?? "translator-out";
        string tag = Get(a, "to") ?? "ita";
        var tr = Translator.Load(dir);

        string? text = Get(a, "text");
        if (text is not null) { Console.WriteLine(tr.Translate(text, tag)); return 0; }

        for (string? line; (line = Console.ReadLine()) is not null; )
            Console.WriteLine(tr.Translate(line, tag));
        return 0;
    }

    private static void SampleSome(DecoderLM model, BpeTokenizer tok, List<SentencePair> pairs, Random rng, int n = 4)
    {
        var tr = new Translator(model, tok);
        Console.WriteLine("   [sample]");
        for (int i = 0; i < n; i++)
        {
            var p = pairs[rng.Next(pairs.Count)];
            string outp = tr.Translate(p.Source, p.TargetTag, maxNewTokens: 32);
            Console.WriteLine($"     >>{p.TargetTag}<< {p.Source} -> \"{outp}\"  (ref \"{p.Target}\")");
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────────

    private static Dictionary<string, string> ArgMap(string[] args)
    {
        var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--")) continue;
            string key = args[i][2..];
            d[key] = (i + 1 < args.Length && !args[i + 1].StartsWith("--")) ? args[++i] : "true";
        }
        return d;
    }

    private static string? Get(Dictionary<string, string> a, string k) => a.TryGetValue(k, out var v) ? v : null;

    private static int IntArg(Dictionary<string, string> a, string k, int def)
        => a.TryGetValue(k, out var v) && int.TryParse(v, out var r) ? r : def;

    private static float FloatArg(Dictionary<string, string> a, string k, float def)
        => a.TryGetValue(k, out var v) &&
           float.TryParse(v, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var r) ? r : def;

    private static void Shuffle<T>(List<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--) { int j = rng.Next(i + 1); (list[i], list[j]) = (list[j], list[i]); }
    }

    private static Tensor AgRandn(int rows, int cols, Random rng, float scale)
    {
        var t = new Tensor(rows, cols);
        for (int i = 0; i < t.Data.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            t.Data[i] = scale * (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return t;
    }

    /// <summary>
    /// Central-difference gradient check: for a random sample of elements of each
    /// parameter, compares the analytic gradient (from one Backward) to (L+ − L−)/2ε.
    /// Returns the worst absolute and relative error seen.
    /// </summary>
    private static (double maxAbs, double maxRel, bool pass) GradCheck(
        Tensor[] parameters, Func<Tensor> lossFn, Random rng, int samplesPerParam,
        float eps = 1e-3f, float atol = 1e-3f, float rtol = 5e-2f)
    {
        var L = lossFn(); L.Backward();
        var analytic = parameters.Select(p => (float[])p.Grad.Clone()).ToArray();

        double maxAbs = 0, maxRel = 0;
        bool pass = true;
        for (int pi = 0; pi < parameters.Length; pi++)
        {
            var p = parameters[pi];
            int samples = Math.Min(samplesPerParam, p.Data.Length);
            for (int s = 0; s < samples; s++)
            {
                int idx = samplesPerParam >= p.Data.Length ? s : rng.Next(p.Data.Length);
                float orig = p.Data[idx];
                p.Data[idx] = orig + eps; float lp = lossFn().Data[0];
                p.Data[idx] = orig - eps; float lm = lossFn().Data[0];
                p.Data[idx] = orig;

                float num = (lp - lm) / (2f * eps);
                float ana = analytic[pi][idx];
                double abs = Math.Abs(num - ana);

                // numpy.allclose criterion: |num − ana| ≤ atol + rtol·|ana|. This avoids the
                // spurious blow-up of relative error where the true gradient is ~0.
                if (abs > atol + rtol * Math.Abs(ana)) pass = false;

                maxAbs = Math.Max(maxAbs, abs);
                if (Math.Abs(ana) > 1e-2) maxRel = Math.Max(maxRel, abs / Math.Abs(ana));   // rel only where grad is meaningful
            }
        }
        return (maxAbs, maxRel, pass);
    }
}
