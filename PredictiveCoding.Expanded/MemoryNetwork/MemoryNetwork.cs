using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.MemoryNetwork;

// ─────────────────────────────────────────────────────────────────────────────
// MemoryNetwork  —  PC network augmented with differentiable episodic memory
//
// The memory module is inspired by the hippocampal–neocortical interaction in
// predictive processing theories (Friston & Buzsáki 2016): a fast-binding
// episodic store (hippocampus ≈ MemoryBank) biases the cortical prior before
// each inference pass, then the cortex updates the store with its refined
// latent representation.
//
// Learning loop (per sample):
//   1. Quick encode: z = InferGetTop(x, steps)
//   2. Memory query: retrieve top-k memories near z by cosine similarity
//   3. Compute attention-weighted memory prior: μ_mem = Σ_i w_i · v_i
//   4. Set top-layer prior to μ_mem with MemoryPrecision
//   5. Supervised PC learning with the active prior
//   6. Write z to episodic memory
//   7. Reset prior to flat (removes memory bias for next sample)
//
// Prediction loop (per sample):
//   Steps 1-4 above, then Predict with memory prior active.
//
// References:
//   Friston & Buzsáki (2016) "The Functional Anatomy of Time" Trends Cogn Sci.
//   Graves et al. (2014) NTM — conceptual analogue without backprop.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class MemoryNetwork : IDisposable
{
    private readonly Network             _network;
    private readonly MemoryBank          _memory;
    private readonly MemoryNetworkConfig _config;

    public int MemoryCount              => _memory.Count;
    public MemoryNetworkConfig Configuration => _config;

    public MemoryNetwork(MemoryNetworkConfig config, Random rng)
    {
        _config = config;
        var netCfg = config.NetworkConfig with { Path = config.BasePath };
        _network   = new Network(netCfg, rng);

        int latentDim = netCfg.LayerSizes[^1];
        _memory = new MemoryBank(config.MemoryCapacity, latentDim, latentDim);
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Memory-augmented classification:
    ///   1. Encode input → rough z.
    ///   2. Retrieve and attend over memories.
    ///   3. Set memory prior.
    ///   4. Classify with memory-biased inference.
    ///   5. Reset prior.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        var prior = BuildMemoryPrior(input, inferenceSteps);
        SetPrior(prior);
        try
        {
            return _network.Predict(input, inferenceSteps);
        }
        finally
        {
            ResetPrior();
        }
    }

    /// <summary>
    /// Memory-augmented classification with energy diagnostics.
    /// </summary>
    public (int cls, float energy) PredictWithEnergy(DenseTensor<float> input, int inferenceSteps)
    {
        var prior = BuildMemoryPrior(input, inferenceSteps);
        SetPrior(prior);
        try
        {
            return _network.PredictWithEnergy(input, inferenceSteps);
        }
        finally
        {
            ResetPrior();
        }
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Memory-augmented learning step:
    ///   1. Encode input → z (before memory).
    ///   2. Retrieve memory prior from z.
    ///   3. Run supervised PC learning with the memory prior active.
    ///   4. Write z to episodic memory for future retrieval.
    ///   5. Reset prior.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        var z     = _network.InferGetTop(input, inferenceSteps);
        var prior = AttendMemory(z.Buffer.ToArray());
        SetPrior(prior);
        try
        {
            _network.Learn(input, label, inferenceSteps);
        }
        finally
        {
            ResetPrior();
        }

        _memory.Write(z.Buffer.ToArray(), z.Buffer.ToArray());
    }

    // ── Memory attention ─────────────────────────────────────────────────────

    /// <summary>
    /// Returns the attention-weighted memory prior for <paramref name="query"/>,
    /// or null when the memory bank is empty.
    ///
    /// Weights:  w_i = softmax(τ · cos(query, key_i))
    /// Prior:    μ_mem = Σ_i w_i · value_i
    /// </summary>
    public DenseTensor<float>? AttendMemory(float[] query)
    {
        if (_memory.Count == 0) return null;

        var entries = _memory.QueryTopK(query, _config.RetrievalTopK);
        if (entries.Length == 0) return null;

        float temp   = _config.AttentionTemperature;
        float maxSim = float.NegativeInfinity;
        foreach (var e in entries) if (e.Similarity > maxSim) maxSim = e.Similarity;

        float   sum     = 0f;
        float[] weights = new float[entries.Length];
        for (int i = 0; i < entries.Length; i++)
        {
            weights[i] = MathF.Exp(temp * (entries[i].Similarity - maxSim));
            sum += weights[i];
        }
        for (int i = 0; i < entries.Length; i++) weights[i] /= sum;

        int dim   = entries[0].Value.Length;
        var prior = new DenseTensor<float>(dim);
        for (int i = 0; i < entries.Length; i++)
            for (int d = 0; d < dim; d++)
                prior.Buffer[d] += weights[i] * entries[i].Value[d];

        return prior;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public float ComputeEnergy() => _network.ComputeEnergy();

    public void FlushStore() => _network.FlushStore();

    // ── Private helpers ───────────────────────────────────────────────────────

    private DenseTensor<float>? BuildMemoryPrior(DenseTensor<float> input, int inferenceSteps)
    {
        if (_memory.Count == 0) return null;
        var z = _network.InferGetTop(input, inferenceSteps);
        return AttendMemory(z.Buffer.ToArray());
    }

    private void SetPrior(DenseTensor<float>? prior)
    {
        if (prior is not null)
            _network.SetTopLayerPrior(prior, _config.MemoryPrecision);
    }

    private void ResetPrior() => _network.SetTopLayerPrior(null, 1f);

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => _network.Dispose();
}
