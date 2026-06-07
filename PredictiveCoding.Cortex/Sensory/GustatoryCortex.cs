using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Hopfield;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// GustatoryCortex  —  taste associative memory
//
// Primary gustatory cortex (in the insula) represents tastes as distributed
// activity patterns and recognises a taste by completing it from a partial cue —
// content-addressable memory, the gustatory counterpart of the olfactory
// PiriformCortex.  Implemented as a modern-Hopfield store over taste vectors.
//
// References: Yarmolinsky, Zuker & Ryba (2009); Hopfield (1982); Ramsauer et al. (2021).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record GustatoryCortexConfig
{
    /// <summary>Generative + attractor store over taste vectors.</summary>
    public HopfieldNetworkConfig Memory { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/gustatory";

    public void Validate() => Memory.Validate();
}

public sealed class GustatoryCortex : IDisposable
{
    private readonly HopfieldNetwork _memory;

    public int TasteDim    => _memory.InputDim;
    public int MemoryCount => _memory.MemoryCount;

    public GustatoryCortex(GustatoryCortexConfig config, Random rng)
    {
        config.Validate();
        _memory = new HopfieldNetwork(config.Memory with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Learns the generative taste model on one exemplar.</summary>
    public void Learn(DenseTensor<float> taste, int steps) => _memory.Learn(taste, steps);

    /// <summary>Memorises a taste as an attractor (call after some Learn passes).</summary>
    public void Memorize(DenseTensor<float> taste, int steps) => _memory.Store(taste, steps);

    /// <summary>Completes / denoises a partial taste cue to the nearest stored taste.</summary>
    public DenseTensor<float> Recognize(DenseTensor<float> partialTaste, int steps)
        => _memory.Recall(partialTaste, steps);

    public void ClearMemory() => _memory.ClearMemory();
    public void FlushAll()    => _memory.FlushStore();
    public void Dispose()     => _memory.Dispose();
}
