using System.Security.Cryptography;
using System.Text;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// LmDistiller  —  sequence-level knowledge distillation into a DecoderLM
// (roadmap Phase 7).
//
// The teacher is any string → string function. In the app, wire the local
// LM Studio server (PredictiveCoding.Data):
//
//     using var lm = new LmStudioSource();
//     Func<string, string> teacher = p => lm.ChatAsync(p).GetAwaiter().GetResult().Text;
//
// Offline KD: prompts → teacher completions (cached on disk) → the student
// trains with masked CE on the completion tokens only.
//
// On-policy round (SEED-style): the student first generates its own draft, the
// teacher continues/corrects FROM that draft, and the student trains on the
// correction from its own prefix — the training signal is the mismatch between
// the student's trajectory and the teacher's correction of it, which is
// prediction error on-policy: the predictive-coding way to distil.
// ─────────────────────────────────────────────────────────────────────────────

public static class LmDistiller
{
    /// <summary>One distillation pair: what was asked, what the teacher answered.</summary>
    public sealed record DistillSample(string Context, string Completion);

    /// <summary>
    /// Builds (or replays) the offline distillation set: one teacher completion
    /// per prompt, cached as one file per prompt under
    /// <paramref name="cacheDir"/> (default ~/.predictivecoding/lm-distill —
    /// runtime data, stays out of git). Delete a file to re-ask the teacher.
    /// </summary>
    public static IReadOnlyList<DistillSample> BuildDataset(
        IEnumerable<string>  prompts,
        Func<string, string> teacher,
        string?              cacheDir = null)
    {
        cacheDir ??= Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".predictivecoding", "lm-distill");
        Directory.CreateDirectory(cacheDir);

        var samples = new List<DistillSample>();
        foreach (string prompt in prompts)
        {
            string file = Path.Combine(cacheDir, CacheKey(prompt) + ".txt");
            string completion;
            if (File.Exists(file))
            {
                completion = File.ReadAllText(file);
            }
            else
            {
                completion = teacher(prompt);
                File.WriteAllText(file, completion);
            }
            samples.Add(new DistillSample(ChatTemplate.Format(prompt), completion));
        }
        return samples;
    }

    /// <summary>
    /// Turns (context, completion) into a masked training example: every row
    /// whose target still lies inside the context gets <paramref name="ignoreIndex"/>
    /// (the context only conditions), the completion tokens + EOS are the
    /// trained targets. Truncated to the student's window.
    /// </summary>
    public static (int[] tokens, int[] targets) BuildExample(
        ITokenizer tok, string context, string completion, int maxSeq, int ignoreIndex = -1)
    {
        var ctxIds  = tok.Encode(context, addBos: true);
        var compIds = tok.Encode(completion, addEos: true);

        var full = new List<int>(ctxIds.Count + compIds.Count);
        full.AddRange(ctxIds);
        full.AddRange(compIds);

        int n = Math.Min(full.Count, maxSeq + 1);          // n-1 input rows, each with a next-token target
        var tokens  = new int[n - 1];
        var targets = new int[n - 1];
        for (int i = 0; i < n - 1; i++)
        {
            tokens[i]  = full[i];
            targets[i] = i + 1 < ctxIds.Count ? ignoreIndex : full[i + 1];
        }
        return (tokens, targets);
    }

    /// <summary>
    /// One gradient-accumulating distillation step over a batch of samples
    /// (masked CE on completion tokens only). Returns the mean loss, or 0 when
    /// every example truncated to nothing scorable.
    /// </summary>
    public static float Step(
        DecoderLM                    student,
        ITokenizer                   tok,
        IReadOnlyList<DistillSample> batch,
        Adam                         optimizer,
        float                        learningRate,
        float?                       clipNorm    = null,
        int                          ignoreIndex = -1)
    {
        var examples = new List<(int[] tokens, int[] targets)>();
        foreach (var s in batch)
        {
            var ex = BuildExample(tok, s.Context, s.Completion, student.Config.MaxSeq, ignoreIndex);
            foreach (int t in ex.targets)
                if (t != ignoreIndex) { examples.Add(ex); break; }
        }
        if (examples.Count == 0) return 0f;

        return Trainer.Step(
            student.Parameters(), examples,
            (t, y, tr) => student.Loss(t, y, ignoreIndex, tr),
            optimizer, learningRate, clipNorm);
    }

    /// <summary>
    /// One SEED-style on-policy round: for each prompt the student greedily
    /// drafts <paramref name="draftTokens"/> tokens, the teacher receives the
    /// prompt AND the student's draft and answers with the correction, and the
    /// student trains on that correction from its own prefix. Returns the mean
    /// loss of the round's training step.
    /// </summary>
    public static float OnPolicyRound(
        DecoderLM                    student,
        ITokenizer                   tok,
        IReadOnlyList<string>        prompts,
        Func<string, string, string> teacherCorrect,     // (prompt, studentDraft) → correction
        Adam                         optimizer,
        float                        learningRate,
        int                          draftTokens = 16,
        float?                       clipNorm    = null,
        int                          ignoreIndex = -1)
    {
        var samples = new List<DistillSample>();
        foreach (string prompt in prompts)
        {
            string context = ChatTemplate.Format(prompt);
            var ctxIds     = tok.Encode(context, addBos: true);
            if (ctxIds.Count >= student.Config.MaxSeq) continue;   // no room to draft

            var withDraft = SelfSpeculative.Greedy(student, ctxIds, draftTokens, tok.Eos);
            string draft  = tok.Decode(withDraft[ctxIds.Count..]);

            string correction = teacherCorrect(prompt, draft);
            samples.Add(new DistillSample(context + draft, correction));
        }
        return Step(student, tok, samples, optimizer, learningRate, clipNorm, ignoreIndex);
    }

    private static string CacheKey(string prompt)
    {
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(prompt));
        var sb   = new StringBuilder(16);
        for (int i = 0; i < 8; i++) sb.Append(hash[i].ToString("x2"));
        return sb.ToString();
    }
}
