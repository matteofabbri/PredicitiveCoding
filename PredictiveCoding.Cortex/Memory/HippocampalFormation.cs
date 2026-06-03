using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Hopfield;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// HippocampalFormation  —  fast episodic memory + systems consolidation
//
// Complementary-learning-systems model (McClelland et al. 1995):
//   • a fast hippocampal store (HopfieldNetwork) memorises individual episodes
//     as attractors and completes them from partial cues;
//   • a slow neocortex (a PCN) learns the statistical structure of those episodes
//     through offline REPLAY — re-presenting stored episodes interleaved, the way
//     hippocampal replay trains cortex during rest/sleep.
//
// References: McClelland, McNaughton & O'Reilly (1995); Friston & Buzsáki (2016).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record HippocampalFormationConfig
{
    /// <summary>Fast hippocampal attractor store.</summary>
    public HopfieldNetworkConfig Hippocampus { get; init; } = new();

    /// <summary>Slow neocortex consolidated by replay.
    /// LayerSizes[0] must equal the hippocampal input dim.</summary>
    public NetworkConfig Neocortex { get; init; } = new();

    /// <summary>Maximum episodes retained for replay (FIFO eviction).</summary>
    public int MaxEpisodes { get; init; } = 256;

    public string BasePath { get; init; } = "./cortex/hippocampus";

    public void Validate()
    {
        Hippocampus.Validate();
        int hipIn = Hippocampus.NetworkConfig.LayerSizes[0];
        if (Neocortex.LayerSizes[0] != hipIn)
            throw new InvalidOperationException(
                $"Neocortex.LayerSizes[0] ({Neocortex.LayerSizes[0]}) must equal the hippocampal input dim ({hipIn}).");
        if (MaxEpisodes < 1)
            throw new InvalidOperationException("MaxEpisodes must be >= 1.");
    }
}

public sealed class HippocampalFormation : IDisposable
{
    private readonly HopfieldNetwork          _hippocampus;
    private readonly Network                  _neocortex;
    private readonly HippocampalFormationConfig _config;
    private readonly List<DenseTensor<float>> _episodes = [];

    public int EpisodeCount => _episodes.Count;
    public int MemoryCount  => _hippocampus.MemoryCount;

    public HippocampalFormation(HippocampalFormationConfig config, Random rng)
    {
        config.Validate();
        _config      = config;
        _hippocampus = new HopfieldNetwork(config.Hippocampus with { BasePath = Path.Combine(config.BasePath, "hpc") }, new Random(rng.Next()));
        _neocortex   = new Network(config.Neocortex with { Path = Path.Combine(config.BasePath, "neo") }, new Random(rng.Next()));
    }

    /// <summary>
    /// Encodes and memorises one episode: trains the hippocampal generative model,
    /// stores its latent as an attractor, and keeps the episode for later replay.
    /// </summary>
    public void Experience(DenseTensor<float> input, int steps)
    {
        _hippocampus.Learn(input, steps);
        _hippocampus.Store(input, steps);

        _episodes.Add(input.Clone());
        if (_episodes.Count > _config.MaxEpisodes) _episodes.RemoveAt(0);
    }

    /// <summary>Pattern completion of a partial / noisy cue via the hippocampal attractor.</summary>
    public DenseTensor<float> Recall(DenseTensor<float> partialCue, int steps)
        => _hippocampus.Recall(partialCue, steps);

    /// <summary>
    /// Offline consolidation: replays stored episodes (interleaved, shuffled) into
    /// the neocortex for <paramref name="rounds"/> passes, transferring episodic
    /// knowledge into slow cortical weights.
    /// </summary>
    public void Replay(int rounds, int steps, Random rng)
    {
        if (_episodes.Count == 0) return;
        var order = Enumerable.Range(0, _episodes.Count).ToArray();
        for (int r = 0; r < rounds; r++)
        {
            RandomUtils.Shuffle(order, rng);
            foreach (int i in order)
                _neocortex.LearnUnsupervised(_episodes[i], steps);
        }
    }

    /// <summary>Neocortical (consolidated) encoding of an input.</summary>
    public DenseTensor<float> NeocorticalEncode(DenseTensor<float> input, int steps)
        => _neocortex.InferGetTop(input, steps);

    public void FlushAll()
    {
        _hippocampus.FlushStore();
        _neocortex.FlushStore();
    }

    public void Dispose()
    {
        _hippocampus.Dispose();
        _neocortex.Dispose();
    }
}
