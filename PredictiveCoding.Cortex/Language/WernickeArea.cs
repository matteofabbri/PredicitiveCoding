using PredictiveCoding.Core;
using PredictiveCoding.Networks.DeepRecurrent;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// WernickeArea  —  language comprehension (sequence → meaning)
//
// Wernicke's area maps an incoming stream of speech/word units onto meaning.  The
// "comprehension" stream of the dual-stream model integrates the token sequence
// over time into a semantic representation.  Implemented over a DeepRecurrentNetwork
// that consumes a sequence of token vectors and emits a semantic latent.
//
// References: Hickok & Poeppel (2007) dual-stream speech processing; Friston (2008).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record WernickeAreaConfig
{
    /// <summary>Temporal sequence model.  Stages[0].LayerSizes[0] = token-vector dim;
    /// Stages[^1].LayerSizes[^1] = semantic latent dim.</summary>
    public DeepRecurrentNetworkConfig Sequence { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/wernicke";

    public void Validate() => Sequence.Validate();
}

public sealed class WernickeArea : IDisposable
{
    private readonly DeepRecurrentNetwork _seq;

    public int TokenDim    => _seq.InputDim;
    public int SemanticDim => _seq.TopLatentDim;

    public WernickeArea(WernickeAreaConfig config, Random rng)
    {
        config.Validate();
        _seq = new DeepRecurrentNetwork(config.Sequence with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Learns one utterance (token sequence); returns the final semantic latent.</summary>
    public DenseTensor<float> LearnUtterance(IReadOnlyList<DenseTensor<float>> tokens, int steps)
    {
        if (tokens.Count == 0) throw new ArgumentException("At least one token is required.", nameof(tokens));
        _seq.Reset();
        DenseTensor<float> top = null!;
        foreach (var tok in tokens) top = _seq.Step(tok, steps);
        return top;
    }

    /// <summary>Comprehends a token sequence into a semantic latent (inference, no learning).</summary>
    public DenseTensor<float> Comprehend(IReadOnlyList<DenseTensor<float>> tokens, int steps)
    {
        if (tokens.Count == 0) throw new ArgumentException("At least one token is required.", nameof(tokens));
        _seq.Reset();
        DenseTensor<float> top = null!;
        foreach (var tok in tokens) top = _seq.Encode(tok, steps);
        return top;
    }

    public void FlushAll() => _seq.FlushAll();
    public void Dispose()  => _seq.Dispose();
}
