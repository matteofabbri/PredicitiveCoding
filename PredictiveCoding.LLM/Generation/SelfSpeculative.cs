using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Generation;

// ─────────────────────────────────────────────────────────────────────────────
// SelfSpeculative  —  greedy decoding with the model's own t+2 head as drafter
// (roadmap Phase 3).
//
// DecoderLM has no KV cache: every decoding step re-runs the whole sequence, so
// each forward saved is a whole forward saved. The multi-token head drafts the
// token two positions ahead; the next forward verifies the draft against the
// main head. Acceptance commits two tokens per forward, rejection still commits
// one (the verifying forward already computed the true token) — so the output
// is IDENTICAL to plain greedy decoding by construction, only cheaper.
// ─────────────────────────────────────────────────────────────────────────────

public static class SelfSpeculative
{
    /// <summary>Outcome of a speculative run, with the effort counters.</summary>
    /// <param name="Tokens">Prompt + generated tokens (identical to plain greedy).</param>
    /// <param name="Forwards">Full forward passes actually spent.</param>
    /// <param name="Drafted">Drafts proposed by the t+2 head.</param>
    /// <param name="Accepted">Drafts the main head confirmed.</param>
    public sealed record Result(int[] Tokens, int Forwards, int Drafted, int Accepted)
    {
        public float AcceptanceRate => Drafted > 0 ? (float)Accepted / Drafted : 0f;
    }

    /// <summary>
    /// Greedy generation using the model's own multi-token head as the drafter.
    /// Requires a model built with <c>MtpWeight &gt; 0</c> (falls back to plain
    /// greedy — zero drafts — when the head is absent).
    /// </summary>
    public static Result Generate(DecoderLM model, IReadOnlyList<int> prompt, int maxNewTokens, int eosToken = -1)
    {
        var seq       = new List<int>(prompt);
        int maxSeq    = model.Config.MaxSeq;
        int generated = 0, forwards = 0, drafted = 0, accepted = 0;
        int pending   = -1;                    // draft for the next position, -1 = none

        while (generated < maxNewTokens && seq.Count < maxSeq)
        {
            if (pending >= 0 && seq.Count + 1 <= maxSeq)
            {
                // Verify pass: run the sequence WITH the draft appended.
                seq.Add(pending);
                var (main, mtp) = model.ForwardMtp([.. seq]);
                forwards++;
                int vocab = main.Cols;
                int real  = Argmax(main.Data, (seq.Count - 2) * vocab, vocab);   // true token where the draft sits

                if (real == pending)
                {
                    accepted++;                                                  // draft survives, position committed
                    generated++;
                    if (seq[^1] == eosToken || generated >= maxNewTokens || seq.Count >= maxSeq) break;

                    // Bonus from the same forward: the row over the accepted
                    // draft is valid, so its main head commits a second token…
                    seq.Add(Argmax(main.Data, (seq.Count - 1) * vocab, vocab));
                    generated++;
                    if (seq[^1] == eosToken) break;

                    // …and its t+2 head drafts the position after that.
                    pending = Argmax(mtp!.Data, (seq.Count - 2) * vocab, vocab);
                    drafted++;
                }
                else
                {
                    seq[^1] = real;                                              // draft rejected: the same forward already knows the true token
                    generated++;
                    if (real == eosToken) break;

                    // The row BEFORE the draft saw only committed tokens, so its
                    // t+2 head validly drafts the position after `real`.
                    pending = Argmax(mtp!.Data, (seq.Count - 2) * vocab, vocab);
                    drafted++;
                }
            }
            else
            {
                // Plain step (no pending draft, or no room to speculate).
                pending = -1;
                var (main, mtp) = model.ForwardMtp([.. seq]);
                forwards++;
                int vocab = main.Cols;
                seq.Add(Argmax(main.Data, (seq.Count - 1) * vocab, vocab));
                generated++;
                if (seq[^1] == eosToken) break;

                if (mtp is not null)
                {
                    pending = Argmax(mtp.Data, (seq.Count - 2) * vocab, vocab);
                    drafted++;
                }
            }
        }

        return new Result([.. seq], forwards, drafted, accepted);
    }

    /// <summary>Plain greedy decoding (one forward per token) — the reference the speculative path must match.</summary>
    public static int[] Greedy(DecoderLM model, IReadOnlyList<int> prompt, int maxNewTokens, int eosToken = -1)
    {
        var seq = new List<int>(prompt);
        for (int i = 0; i < maxNewTokens && seq.Count < model.Config.MaxSeq; i++)
        {
            var logits = model.Forward([.. seq]);
            seq.Add(Argmax(logits.Data, (seq.Count - 1) * logits.Cols, logits.Cols));
            if (seq[^1] == eosToken) break;
        }
        return [.. seq];
    }

    private static int Argmax(float[] data, int offset, int length)
    {
        int best = 0;
        for (int j = 1; j < length; j++)
            if (data[offset + j] > data[offset + best]) best = j;
        return best;
    }
}
