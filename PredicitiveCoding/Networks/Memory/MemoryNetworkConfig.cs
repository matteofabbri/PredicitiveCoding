using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Memory;

// ─────────────────────────────────────────────────────────────────────────────
// MemoryNetworkConfig  —  configuration for PCN + episodic memory
//
// The memory network augments a standard PCN with an external key-value store.
// Before each inference/learning pass:
//   1. A rough latent z is computed via free inference.
//   2. z is used to query the memory for the top-k most similar stored entries.
//   3. An attention-weighted average of the retrieved values is set as a prior
//      on the network's top layer.
//   4. A second inference pass runs with the memory prior active.
//
// After each learning pass, the new latent z is written to the memory bank,
// creating an auto-associative episodic store that grows over training.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record MemoryNetworkConfig
{
    public NetworkConfig NetworkConfig { get; init; } = new();

    /// <summary>Maximum number of episodic memories stored (FIFO eviction).</summary>
    public int MemoryCapacity { get; init; } = 1024;

    /// <summary>Number of memories retrieved per query pass.</summary>
    public int RetrievalTopK { get; init; } = 4;

    /// <summary>
    /// Precision Π_mem of the memory-derived top-layer prior.
    /// Higher values give more weight to retrieved memories relative to
    /// the bottom-up prediction error signal.
    /// </summary>
    public float MemoryPrecision { get; init; } = 0.5f;

    /// <summary>
    /// Temperature for the attention softmax over retrieved memory similarities.
    /// Higher temperature makes the weighting sharper (nearest-neighbour-like).
    /// </summary>
    public float AttentionTemperature { get; init; } = 10f;

    public string BasePath { get; init; } = "./brain/memory";
}
