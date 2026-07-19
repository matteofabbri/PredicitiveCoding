using PredictiveCoding.Data.LmStudio;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// LmDistillationDemo  —  LIVE distillation from the local LM Studio server
// (the "one-line delegate" hookup promised by LLM roadmap Phase 7), wired as
// the capstone of the three post-roadmap openings:
//
//   · the student is a sparse-MoE DecoderLM whose experts run FLOP-sparse
//     (only the routed rows compute — ActivationProbe.MoeRowsComputed);
//   · its MatMul forwards are routed through Gemm.Provider's MIXED path
//     (16-bit operands, fp32 accumulation — Ag.MatMulRoute);
//   · the teacher is whatever chat model LM Studio is serving on :1234,
//     for offline KD first, then SEED-style on-policy correction rounds.
//
// Teacher completions are cached one-file-per-prompt under
// ~/.predictivecoding/lm-distill, so re-runs never re-ask the teacher.
// ─────────────────────────────────────────────────────────────────────────────

public static class LmDistillationDemo
{
    private const string TeacherSystem =
        "Answer in one short sentence. No preamble, no explanation.";

    private const string CorrectorSystem =
        "You are the teacher of a tiny student language model. " +
        "Reply with only the correct short answer to the question - nothing else.";

    private static readonly string[] Prompts =
    [
        "What color is the sky on a clear day?",
        "How many legs does a spider have?",
        "What is 2 + 2?",
        "What is the capital of France?",
    ];

    public static void Run()
    {
        Console.WriteLine("\n═══ Live LM distillation (LM Studio teacher → MoE student) ═══\n");

        using var lm = new LmStudioSource();
        string teacherId;
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            var models = lm.ListModelsAsync(cts.Token).GetAwaiter().GetResult();
            if (models.Count == 0)
            {
                Console.WriteLine("  LM Studio answered but no model is loaded — load a chat model and rerun.");
                return;
            }
            teacherId = models[0].Id;
        }
        catch (Exception ex) when (ex is HttpRequestException or OperationCanceledException)
        {
            Console.WriteLine($"  LM Studio is not reachable on {LmStudioSource.DefaultBaseUrl}.");
            Console.WriteLine("  Start LM Studio, load a chat model, enable the local server, then rerun.");
            return;
        }
        Console.WriteLine($"  Teacher online: {teacherId}");

        // The one-line delegates the roadmap promised: string → string, nothing else.
        // The model id is passed explicitly so LM Studio can JIT-load it; a
        // reasoning model may put everything in reasoning_content, so fall back
        // to its last line; byte-level students get a short completion.
        string Teach(string prompt)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(300));
            return ShortAnswer(lm.GenerateAsync(prompt, TeacherSystem, teacherId, temperature: 0,
                                                maxTokens: 512, cancellationToken: cts.Token)
                                 .GetAwaiter().GetResult());
        }

        string Correct(string prompt, string draft)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(300));
            return ShortAnswer(lm.ChatAsync(
            [
                LmStudioMessage.System(CorrectorSystem),
                LmStudioMessage.User($"Question: {prompt}\nThe student answered: \"{draft}\""),
            ], teacherId, temperature: 0, maxTokens: 512, cancellationToken: cts.Token)
            .GetAwaiter().GetResult());
        }

        // ── Offline KD dataset (disk-cached: delete a file to re-ask) ────────
        string cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".predictivecoding", "lm-distill");
        Console.WriteLine($"  Completion cache: {cacheDir}\n");

        IReadOnlyList<LmDistiller.DistillSample> data;
        try
        {
            data = LmDistiller.BuildDataset(Prompts, Teach);
        }
        catch (Exception ex) when (ex is HttpRequestException or OperationCanceledException)
        {
            Console.WriteLine($"  The teacher did not answer: {ex.Message}");
            Console.WriteLine("  Load the model in LM Studio (or enable just-in-time loading) and rerun.");
            return;
        }
        for (int i = 0; i < Prompts.Length; i++)
            Console.WriteLine($"  Q: {Prompts[i]}\n  T: {data[i].Completion}\n");

        // ── The student: byte-level sparse-MoE DecoderLM, mixed-routed GEMMs ─
        var tok = new ByteTokenizer();
        var cfg = new DecoderLmConfig(tok.VocabSize, 32, NumBlocks: 2, NumHeads: 2, FfDim: 64,
                                      MaxSeq: 192, NumExperts: 4, MoeTopK: 1, MoeBalanceWeight: 0.01f);
        var student = new DecoderLM(cfg, new Random(42));

        Console.WriteLine($"  Student: {student.ParameterCount:N0} params, " +
                          $"{student.ActiveParameterCountPerToken:N0} active/token (4 experts, top-1)");

        var probe = new ActivationProbe();
        student.Forward([.. tok.Encode(ChatTemplate.Format(Prompts[0]), addBos: true)], probe: probe);
        Console.WriteLine($"  FLOP-sparse MoE: {probe.MoeRowsComputed} expert-row executions " +
                          $"(dense would be {probe.MoeRowsComputed * cfg.NumExperts / cfg.MoeTopK})");

        var keepRoute  = Ag.MatMulRoute;
        Ag.MatMulRoute = MatMulRoute.ProviderMixed;
        Console.WriteLine("  GEMM route: ProviderMixed (16-bit operands, fp32 accumulation)\n");
        try
        {
            // ── Offline KD: learn the teacher's cached answers ───────────────
            var opt = new Adam();
            Console.WriteLine("  Offline KD (masked CE on completion tokens):");
            for (int step = 1; step <= 150; step++)
            {
                float loss = LmDistiller.Step(student, tok, data, opt, learningRate: 0.005f, clipNorm: 1f);
                if (step % 25 == 0) Console.WriteLine($"    step {step,3}   loss {loss:F3}");
            }

            // ── On-policy rounds: the teacher corrects the student's drafts ──
            Console.WriteLine("\n  On-policy rounds (teacher corrects the student's own draft):");
            try
            {
                for (int round = 1; round <= 3; round++)
                {
                    float loss = LmDistiller.OnPolicyRound(student, tok, Prompts, Correct, opt,
                                                           learningRate: 0.002f, draftTokens: 24, clipNorm: 1f);
                    Console.WriteLine($"    round {round}   loss {loss:F3}");
                }
            }
            catch (Exception ex) when (ex is HttpRequestException or OperationCanceledException)
            {
                Console.WriteLine($"    teacher lost mid-round ({ex.Message}) — keeping the offline-KD student");
            }

            // ── What the student says now ────────────────────────────────────
            Console.WriteLine("\n  Student answers after distillation (greedy, byte-level):");
            foreach (string prompt in Prompts)
            {
                var ctx = tok.Encode(ChatTemplate.Format(prompt), addBos: true);
                var seq = SelfSpeculative.Greedy(student, ctx, maxNewTokens: 48, tok.Eos);
                Console.WriteLine($"    Q: {prompt}\n    S: {Printable(tok.Decode(seq[ctx.Count..]))}\n");
            }
        }
        finally
        {
            Ag.MatMulRoute = keepRoute;
        }

        Console.WriteLine("  Done. Re-running skips the teacher for cached prompts.");
    }

    /// <summary>
    /// One short teacher line for the byte-level student: the answer, or the
    /// last reasoning line when a thinking model left the content empty,
    /// truncated to fit the student's tiny window.
    /// </summary>
    private static string ShortAnswer(LmStudioCompletion completion)
    {
        string answer = completion.Content.Trim();
        if (answer.Length == 0 && !string.IsNullOrWhiteSpace(completion.Reasoning))
            answer = completion.Reasoning.Trim().Split('\n')[^1].Trim();
        if (answer.Length == 0) answer = "(no answer)";
        return answer.Length <= 100 ? answer : answer[..100];
    }

    /// <summary>A raw byte-level sample can contain control bytes; keep the console clean.</summary>
    private static string Printable(string s)
    {
        var chars = s.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
            if (char.IsControl(chars[i]) && chars[i] != '\n') chars[i] = '·';
        return new string(chars).Replace("\n", " ⏎ ");
    }
}
