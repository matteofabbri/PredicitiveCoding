using PredictiveCoding.Core;
using PredictiveCoding.Networks.Hopfield;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// PiriformCortex  —  olfactory associative memory
//
// Primary olfactory (piriform) cortex behaves like a content-addressable
// associative memory: an odour is a distributed pattern of receptor activations,
// and the cortex completes / denoises it from a partial sniff.  Implemented as a
// thin specialisation of the modern-Hopfield HopfieldNetwork over odour vectors.
//
// References: Haberly (2001); Hopfield (1982); Ramsauer et al. (2021).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record PiriformCortexConfig
{
    /// <summary>Generative + attractor store over odour vectors.</summary>
    public HopfieldNetworkConfig Memory { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/piriform";

    public void Validate() => Memory.Validate();
}

public sealed class PiriformCortex : IDisposable
{
    private readonly HopfieldNetwork _memory;

    public int OdorDim     => _memory.InputDim;
    public int MemoryCount => _memory.MemoryCount;

    public PiriformCortex(PiriformCortexConfig config, Random rng)
    {
        config.Validate();
        _memory = new HopfieldNetwork(config.Memory with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Learns the generative odour model on one exemplar.</summary>
    public void Learn(DenseTensor<float> odor, int steps) => _memory.Learn(odor, steps);

    /// <summary>Memorises an odour as an attractor (call after some Learn passes).</summary>
    public void Memorize(DenseTensor<float> odor, int steps) => _memory.Store(odor, steps);

    /// <summary>Completes / denoises a partial sniff to the nearest stored odour.</summary>
    public DenseTensor<float> Recognize(DenseTensor<float> partialOdor, int steps)
        => _memory.Recall(partialOdor, steps);

    public void ClearMemory() => _memory.ClearMemory();
    public void FlushAll()    => _memory.FlushStore();
    public void Dispose()     => _memory.Dispose();
}
