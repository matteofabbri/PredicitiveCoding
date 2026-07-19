namespace PredictiveCoding.LLM.Eval;

// ─────────────────────────────────────────────────────────────────────────────
// ActivationProbe  —  opt-in activation observatory (roadmap Phase 0).
//
// Pass one to DecoderLM.Forward or GptModel.Step; null (the default) keeps the
// forward pass untouched. Records:
//   • per-block hidden-state statistics — the max/mean ratio is the
//     massive-activation "spike" signature ("The Spike, the Sparse and the
//     Sink", 2026);
//   • per-(block, head) attention-sink share — the average attention mass that
//     rows place on position 0 (trainable stack; the inference-stack meter
//     lands with Phase 4's sink handling).
//
// A probe only observes: it never touches data, gradients or control flow.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ActivationProbe
{
    /// <summary>Hidden-state statistics after one block.</summary>
    public readonly record struct BlockStats(int Block, float MeanAbs, float MaxAbs)
    {
        /// <summary>Max/mean ratio — values ≫ 10 are the massive-activation signature.</summary>
        public float SpikeRatio => MeanAbs > 0f ? MaxAbs / MeanAbs : 0f;
    }

    /// <summary>Average attention mass one head places on position 0 (the "sink").</summary>
    public readonly record struct SinkStats(int Block, int Head, float SinkShare);

    public List<BlockStats> Blocks { get; } = [];
    public List<SinkStats>  Sinks  { get; } = [];

    /// <summary>
    /// Loop passes actually executed by the last looped DecoderLM forward
    /// (≤ NumLoops when the early exit fires; 1 for a non-looped model).
    /// On a looped forward, Blocks/Sinks receive one entry set per pass.
    /// </summary>
    public int LoopsUsed { get; set; }

    /// <summary>Clears all recordings so the probe can observe another forward pass.</summary>
    public void Reset()
    {
        Blocks.Clear();
        Sinks.Clear();
        LoopsUsed = 0;
    }

    /// <summary>Records mean/max absolute activation of one block's output.</summary>
    public void RecordBlock(int block, ReadOnlySpan<float> activations)
    {
        float sum = 0f, max = 0f;
        foreach (float a in activations)
        {
            float x = MathF.Abs(a);
            sum += x;
            if (x > max) max = x;
        }
        float mean = activations.Length > 0 ? sum / activations.Length : 0f;
        Blocks.Add(new BlockStats(block, mean, max));
    }

    /// <summary>
    /// Records the sink share of one head from a row-major causal attention
    /// matrix [rows, rows]: the mean over rows of the mass on column 0.
    /// Row 0 always contributes 1 (it can only attend to position 0), so the
    /// share is at least 1/rows by construction.
    /// </summary>
    public void RecordCausalAttention(int block, int head, ReadOnlySpan<float> probs, int rows)
    {
        float share = 0f;
        for (int r = 0; r < rows; r++) share += probs[r * rows];
        Sinks.Add(new SinkStats(block, head, rows > 0 ? share / rows : 0f));
    }
}
