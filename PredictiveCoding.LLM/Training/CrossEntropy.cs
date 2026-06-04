using PredictiveCoding.Core;
using PredictiveCoding.LLM.Model;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// CrossEntropy  —  language-modelling loss and perplexity (evaluation only).
//
// Computes the forward next-token cross-entropy.  The gradient / optimizer half
// of training is a separate (future) component: the surrounding PredictiveCoding
// library is backprop-free, so a transformer trainer needs its own autodiff or a
// predictive-coding learning rule — see LLM_Roadmap.md.
// ─────────────────────────────────────────────────────────────────────────────

public static class CrossEntropy
{
    /// <summary>Negative log-likelihood of <paramref name="target"/> under <paramref name="logits"/>.</summary>
    public static float TokenLoss(DenseTensor<float> logits, int target)
    {
        var v   = logits.Buffer;
        float max = float.NegativeInfinity;
        for (int i = 0; i < v.Length; i++) if (v[i] > max) max = v[i];

        float sumExp = 0f;
        for (int i = 0; i < v.Length; i++) sumExp += MathF.Exp(v[i] - max);
        float logZ = max + MathF.Log(sumExp);
        return logZ - v[target];                     // −log softmax(logits)[target]
    }

    /// <summary>
    /// Teacher-forced evaluation over a token sequence: returns the mean per-token
    /// cross-entropy and the corresponding perplexity exp(mean loss).
    /// </summary>
    public static (float loss, float perplexity) Evaluate(GptModel model, IReadOnlyList<int> tokens)
    {
        if (tokens.Count < 2) return (0f, 1f);

        var   cache = model.CreateCache();
        double sum  = 0d;
        int    count = 0;
        int    pos   = 0;

        for (int i = 0; i < tokens.Count - 1 && pos < model.MaxSeqLen; i++)
        {
            var logits = model.Step(tokens[i], pos++, cache);
            sum += TokenLoss(logits, tokens[i + 1]);
            count++;
        }

        float loss = count > 0 ? (float)(sum / count) : 0f;
        return (loss, MathF.Exp(loss));
    }
}
