using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// MemoryBudget  —  ROADMAP-Scale B3: the training-step memory table, measured,
// not guessed.
//
// Parameters, gradients and Adam moments follow exactly from the tensor shapes
// (grads are allocated with their tensors; Adam keeps float[] m + v per
// parameter). Activations are the honest part: they are MEASURED by running one
// real forward+backward at the requested sequence length and walking the
// autograd graph from the loss — every tensor the tape keeps alive, parameters
// excluded, Data + Grad. Deterministic: same config + seqLen ⇒ same bytes.
//
// Everything is fp32, the compute reality today. ProjectFp16 derives the A2
// mixed-precision budget from the measured numbers under the standard recipe
// (fp16 working weights/grads/activations, fp32 master weights and Adam
// moments); it is a DERIVATION, and stays labelled that way until Phase A2
// re-measures it on the device.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>One training step's memory, in bytes, for a single model replica.</summary>
public sealed record MemoryBudgetReport(
    DecoderLmConfig Config,
    int SeqLen,
    long ParameterBytes,
    long GradientBytes,
    long AdamStateBytes,
    long ActivationBytes)
{
    /// <summary>Whole fp32 training step: params + grads + Adam moments + live tape.</summary>
    public long TotalBytes => ParameterBytes + GradientBytes + AdamStateBytes + ActivationBytes;

    /// <summary>
    /// The A2 mixed-precision projection (derived, not yet measured): fp16
    /// working weights + fp16 gradients + halved activations, on top of fp32
    /// master weights and fp32 Adam moments.
    /// </summary>
    public long ProjectFp16Bytes =>
        ParameterBytes / 2            // fp16 working copy of the weights
        + ParameterBytes              // fp32 master weights
        + GradientBytes / 2           // fp16 gradients
        + AdamStateBytes              // Adam m+v stay fp32
        + ActivationBytes / 2;        // fp16 activations

    public string Describe() =>
        $"seq {SeqLen}, params {Gb(ParameterBytes)} + grads {Gb(GradientBytes)} " +
        $"+ adam {Gb(AdamStateBytes)} + activations {Gb(ActivationBytes)} " +
        $"= {Gb(TotalBytes)} fp32  (fp16 projection {Gb(ProjectFp16Bytes)})";

    private static string Gb(long bytes) => $"{bytes / 1024.0 / 1024.0 / 1024.0:F2} GB";
}

public static class MemoryBudget
{
    /// <summary>
    /// Measures one training step of <paramref name="cfg"/> at
    /// <paramref name="seqLen"/>: instantiates the model, runs a real
    /// forward+backward, and walks the autograd graph from the loss. Heavy for
    /// big configs (it IS the real step) — that is the point.
    /// </summary>
    public static MemoryBudgetReport Measure(DecoderLmConfig cfg, int seqLen, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        if (seqLen < 2 || seqLen > cfg.MaxSeq)
            throw new ArgumentOutOfRangeException(nameof(seqLen), $"seqLen ({seqLen}) must be in [2, MaxSeq {cfg.MaxSeq}].");

        var model = new DecoderLM(cfg, new Random(seed));
        var parameters = model.Parameters();

        long paramBytes = 0;
        var paramSet = new HashSet<Tensor>();
        foreach (var p in parameters)
        {
            paramBytes += p.Data.Length * 4L;
            paramSet.Add(p);
        }

        // One real step: deterministic tokens, next-token targets.
        var rng     = new Random(seed + 1);
        var tokens  = new int[seqLen];
        var targets = new int[seqLen];
        for (int i = 0; i < seqLen; i++) tokens[i]  = rng.Next(cfg.Vocab);
        for (int i = 0; i < seqLen; i++) targets[i] = i + 1 < seqLen ? tokens[i + 1] : rng.Next(cfg.Vocab);

        var loss = model.Loss(tokens, targets, ignoreIndex: -1, training: true);
        loss.Backward();

        // Walk the tape: every tensor the backward kept alive, parameters excluded.
        long activationBytes = 0;
        var seen  = new HashSet<Tensor>();
        var stack = new Stack<Tensor>();
        stack.Push(loss);
        while (stack.Count > 0)
        {
            var t = stack.Pop();
            if (!seen.Add(t)) continue;
            if (!paramSet.Contains(t))
                activationBytes += (t.Data.Length + t.Grad.Length) * 4L;
            foreach (var p in t.Parents) stack.Push(p);
        }

        return new MemoryBudgetReport(cfg, seqLen,
            ParameterBytes:  paramBytes,
            GradientBytes:   paramBytes,          // Grad rides along with every parameter tensor
            AdamStateBytes:  2 * paramBytes,      // float[] m + v per parameter (Optimizers.cs)
            ActivationBytes: activationBytes);
    }
}
