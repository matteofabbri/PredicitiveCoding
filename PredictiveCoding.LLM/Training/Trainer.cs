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

    /// <summary>
    /// Parallel micro-batch step (roadmap-Pro Phase 3). The batch is sharded
    /// across <paramref name="replicas"/> worker models that share the primary's
    /// SHAPES but keep independent parameter/gradient buffers, so their forward
    /// and backward passes never collide. Replica <c>[0]</c> is the primary; the
    /// others are synced from it (weights copied in) before every step, so each
    /// worker differentiates the current weights. Worker gradients are then
    /// merged into the primary in a FIXED worker order and applied there.
    ///
    /// <para><b>Bit-stability policy.</b> The reduction order is deterministic —
    /// same shards, same within-shard order, same merge order every run. With a
    /// single replica it is bit-identical to <see cref="Step(IReadOnlyList{Tensor},
    /// IReadOnlyList{ValueTuple{int[],int[]}}, Func{int[],int[],bool,Tensor}, Adam,
    /// float, float?, bool)"/>. With more, the per-shard partial sums re-associate
    /// the floating-point accumulation, so results are reproducible but not
    /// bit-identical to the serial path — the standard data-parallel trade.</para>
    ///
    /// <paramref name="lossFns"/>[w] must run its forward/backward on
    /// <paramref name="replicas"/>[w]. All replicas must expose parameters in the
    /// same order and shape as the primary.
    /// </summary>
    public static float StepParallel(
        IReadOnlyList<IReadOnlyList<Tensor>>          replicas,
        IReadOnlyList<Func<int[], int[], bool, Tensor>> lossFns,
        IReadOnlyList<(int[] tokens, int[] targets)>  batch,
        Adam                                          optimizer,
        float                                         learningRate,
        float?                                        clipNorm = null,
        bool                                          training = true)
    {
        int workers = replicas.Count;
        if (workers == 0) throw new ArgumentException("Need at least one replica.", nameof(replicas));
        if (lossFns.Count != workers)
            throw new ArgumentException("One loss function per replica is required.", nameof(lossFns));

        var primary = replicas[0];

        // Sync current weights primary → the extra replicas (worker 0 is primary).
        for (int w = 1; w < workers; w++)
        {
            var rp = replicas[w];
            for (int k = 0; k < primary.Count; k++)
                Array.Copy(primary[k].Data, rp[k].Data, primary[k].Data.Length);
        }

        // Contiguous, balanced shards — deterministic assignment.
        var bounds = ShardBounds(batch.Count, workers);

        var localGrads = new float[workers][][];
        var localLoss  = new double[workers];

        void RunWorker(int w)
        {
            var ps  = replicas[w];
            var acc = new float[ps.Count][];
            for (int k = 0; k < ps.Count; k++) acc[k] = new float[ps[k].Data.Length];

            double sum = 0d;
            for (int e = bounds[w]; e < bounds[w + 1]; e++)
            {
                var (tokens, targets) = batch[e];
                var loss = lossFns[w](tokens, targets, training);
                loss.Backward();
                sum += loss.Data[0];
                for (int k = 0; k < ps.Count; k++)
                {
                    var g = ps[k].Grad; var a = acc[k];
                    for (int i = 0; i < g.Length; i++) a[i] += g[i];
                }
            }
            localGrads[w] = acc;
            localLoss[w]  = sum;
        }

        if (workers == 1) RunWorker(0);
        else Parallel.For(0, workers, RunWorker);

        // Merge into the primary grads in FIXED worker order (0..W-1).
        int b = Math.Max(1, batch.Count);
        double total = 0d;
        for (int k = 0; k < primary.Count; k++)
        {
            var pg = primary[k].Grad;
            Array.Clear(pg);
            for (int w = 0; w < workers; w++)
            {
                var a = localGrads[w][k];
                for (int i = 0; i < pg.Length; i++) pg[i] += a[i];
            }
            for (int i = 0; i < pg.Length; i++) pg[i] /= b;
        }
        for (int w = 0; w < workers; w++) total += localLoss[w];

        if (clipNorm is { } c) GradientClipping.ClipGlobalNorm(primary, c);
        optimizer.Step(primary, learningRate);
        return (float)(total / b);
    }

    // Balanced contiguous shard boundaries: worker w owns [bounds[w], bounds[w+1]).
    private static int[] ShardBounds(int count, int workers)
    {
        var bounds = new int[workers + 1];
        int baseSize = count / workers, extra = count % workers, pos = 0;
        for (int w = 0; w < workers; w++)
        {
            bounds[w] = pos;
            pos += baseSize + (w < extra ? 1 : 0);
        }
        bounds[workers] = count;
        return bounds;
    }
}
