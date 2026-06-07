using PredictiveCoding.Core;
using PredictiveCoding.Expanded.MemoryNetwork;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// PerirhinalCortex  —  item familiarity / novelty (recognition memory)
//
// The perirhinal cortex supports recognition memory: it signals whether an item has
// been seen before (familiarity) independently of recalling where or when.  Here an
// encoder PCN maps an item to a latent, and a key-value MemoryBank stores the latents
// of remembered items; familiarity is the top-1 cosine similarity of a probe to the
// store, and an item is novel when that similarity falls below a threshold.
//
// References: Brown & Aggleton (2001) recognition memory; Eichenbaum et al. (2007).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record PerirhinalCortexConfig
{
    /// <summary>Item encoder PCN.  LayerSizes[^1] = item latent dim.</summary>
    public NetworkConfig Encoder { get; init; } = new();

    /// <summary>Maximum number of remembered items (FIFO eviction).</summary>
    public int Capacity { get; init; } = 1024;

    /// <summary>Familiarity (top-1 cosine) below which an item is judged novel.</summary>
    public float NoveltyThreshold { get; init; } = 0.8f;

    public string BasePath { get; init; } = "./cortex/perirhinal";

    public void Validate()
    {
        if (Capacity < 1)
            throw new InvalidOperationException("Capacity must be >= 1.");
        if (NoveltyThreshold is < -1f or > 1f)
            throw new InvalidOperationException("NoveltyThreshold must be in [-1, 1] (a cosine similarity).");
    }
}

public sealed class PerirhinalCortex : IDisposable
{
    private readonly Network                _encoder;
    private readonly MemoryBank             _items;
    private readonly PerirhinalCortexConfig _config;

    public int ItemDim     => _encoder.Configuration.LayerSizes[0];
    public int LatentDim   => _encoder.Configuration.LayerSizes[^1];
    public int MemoryCount => _items.Count;

    public PerirhinalCortex(PerirhinalCortexConfig config, Random rng)
    {
        config.Validate();
        _config  = config;
        _encoder = new Network(config.Encoder with { Path = config.BasePath }, rng);
        int latent = _encoder.Configuration.LayerSizes[^1];
        _items = new MemoryBank(config.Capacity, latent, latent);
    }

    /// <summary>Encodes an item to its latent representation.</summary>
    public DenseTensor<float> Encode(DenseTensor<float> item, int steps) => _encoder.InferGetTop(item, steps);

    /// <summary>Refines the item encoder on one exemplar (unsupervised).</summary>
    public void Learn(DenseTensor<float> item, int steps) => _encoder.LearnUnsupervised(item, steps);

    /// <summary>
    /// Familiarity ∈ [−1, 1]: top-1 cosine similarity of the item's latent to any
    /// remembered item.  Returns 0 when nothing has been remembered yet.
    /// </summary>
    public float Familiarity(DenseTensor<float> item, int steps)
    {
        if (_items.Count == 0) return 0f;
        var z   = _encoder.InferGetTop(item, steps).Buffer.ToArray();
        var top = _items.QueryTopK(z, 1);
        return top.Length == 0 ? 0f : top[0].Similarity;
    }

    /// <summary>True when the item is novel (familiarity below NoveltyThreshold).</summary>
    public bool IsNovel(DenseTensor<float> item, int steps) => Familiarity(item, steps) < _config.NoveltyThreshold;

    /// <summary>Remembers an item — stores its latent for future familiarity judgements.</summary>
    public void Remember(DenseTensor<float> item, int steps)
    {
        var z = _encoder.InferGetTop(item, steps).Buffer.ToArray();
        _items.Write(z, z);
    }

    public void FlushAll() => _encoder.FlushStore();
    public void Dispose()  => _encoder.Dispose();
}
