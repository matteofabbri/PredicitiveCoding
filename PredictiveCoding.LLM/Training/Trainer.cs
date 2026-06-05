using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// Trainer  —  one optimizer step over a mini-batch with gradient accumulation.
//
// Accumulates gradients across the micro-examples of a batch (so an arbitrarily
// large effective batch fits in memory), averages them, optionally clips the
// global norm, then applies the optimizer at the scheduled learning rate.
// ─────────────────────────────────────────────────────────────────────────────

public static class Trainer
{
    /// <summary>
    /// Runs forward/backward over every (tokens, targets) example in
    /// <paramref name="batch"/>, accumulating gradients, then performs a single
    /// optimizer update. Returns the mean loss over the batch.
    /// </summary>
    public static float Step(
        TinyTransformerLM                              model,
        IReadOnlyList<(int[] tokens, int[] targets)>   batch,
        Adam                                           optimizer,
        float                                          learningRate,
        float?                                         clipNorm = null,
        bool                                           training = true)
    {
        var ps    = model.Parameters();
        var accum = new float[ps.Length][];
        for (int k = 0; k < ps.Length; k++) accum[k] = new float[ps[k].Data.Length];

        double total = 0d;
        foreach (var (tokens, targets) in batch)
        {
            var loss = model.Loss(tokens, targets, training);
            loss.Backward();                              // zeroes + fills this micro-example's grads
            total += loss.Data[0];
            for (int k = 0; k < ps.Length; k++)
                for (int i = 0; i < ps[k].Grad.Length; i++) accum[k][i] += ps[k].Grad[i];
        }

        int b = Math.Max(1, batch.Count);
        for (int k = 0; k < ps.Length; k++)
            for (int i = 0; i < ps[k].Grad.Length; i++) ps[k].Grad[i] = accum[k][i] / b;

        if (clipNorm is { } c) GradientClipping.ClipGlobalNorm(ps, c);
        optimizer.Step(ps, learningRate);
        return (float)(total / b);
    }

    /// <summary>
    /// Model-agnostic gradient-accumulating step. Works for any autograd model: pass
    /// its <paramref name="parameters"/> and a <paramref name="lossFn"/> that maps
    /// (tokens, targets, training) to a scalar loss — e.g. a DecoderLM's masked
    /// translation loss. Accumulates grads over the batch, averages, optionally clips
    /// the global norm, then applies the optimizer. Returns the mean loss.
    /// </summary>
    public static float Step(
        IReadOnlyList<Tensor>                         parameters,
        IReadOnlyList<(int[] tokens, int[] targets)>  batch,
        Func<int[], int[], bool, Tensor>              lossFn,
        Adam                                          optimizer,
        float                                         learningRate,
        float?                                        clipNorm = null,
        bool                                          training = true)
    {
        var accum = new float[parameters.Count][];
        for (int k = 0; k < parameters.Count; k++) accum[k] = new float[parameters[k].Data.Length];

        double total = 0d;
        foreach (var (tokens, targets) in batch)
        {
            var loss = lossFn(tokens, targets, training);
            loss.Backward();                              // zeroes + fills this micro-example's grads
            total += loss.Data[0];
            for (int k = 0; k < parameters.Count; k++)
                for (int i = 0; i < parameters[k].Grad.Length; i++) accum[k][i] += parameters[k].Grad[i];
        }

        int b = Math.Max(1, batch.Count);
        for (int k = 0; k < parameters.Count; k++)
            for (int i = 0; i < parameters[k].Grad.Length; i++) parameters[k].Grad[i] = accum[k][i] / b;

        if (clipNorm is { } c) GradientClipping.ClipGlobalNorm(parameters, c);
        optimizer.Step(parameters, learningRate);
        return (float)(total / b);
    }
}
