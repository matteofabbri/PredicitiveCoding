using PredictiveCoding.Core;
using PredictiveCoding.Networks.Generative;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// BrocaArea  —  language production (meaning → sequence)
//
// Broca's area drives speech production: it turns an intended meaning into an
// articulatory/word sequence.  The "production" stream of the dual-stream model is
// a generative model that decodes a semantic latent into an utterance; pairs with
// MotorCortex for articulation.  Implemented over a GenerativeNetwork whose latent
// IS the semantic code.
//
// References: Hickok & Poeppel (2007); Friston (2009) active inference of action.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record BrocaAreaConfig
{
    /// <summary>Generative model of utterances.  LatentDim = semantic (meaning) dim.</summary>
    public GenerativeNetworkConfig Generative { get; init; } = new();

    /// <summary>Std-dev of the latent walk used by <see cref="BrocaArea.ProduceSequence"/>.</summary>
    public float DriftScale { get; init; } = 0.3f;

    public string BasePath { get; init; } = "./cortex/broca";

    public void Validate()
    {
        if (DriftScale < 0f) throw new InvalidOperationException("DriftScale must be >= 0.");
    }
}

public sealed class BrocaArea : IDisposable
{
    private readonly GenerativeNetwork _gen;
    private readonly BrocaAreaConfig   _config;

    public int SemanticDim  => _gen.LatentDim;
    public int UtteranceDim => _gen.InputDim;

    public BrocaArea(BrocaAreaConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        _gen    = new GenerativeNetwork(config.Generative with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Learns the generative model of utterances (awake speech experience).</summary>
    public void LearnUtterance(DenseTensor<float> utterance, int steps) => _gen.Learn(utterance, steps);

    /// <summary>Produces an utterance from a semantic latent (top-down generation).</summary>
    public DenseTensor<float> Produce(DenseTensor<float> semantic, int steps)
    {
        if (semantic.Length != _gen.LatentDim)
            throw new ArgumentException($"semantic has {semantic.Length} dims; expected SemanticDim ({_gen.LatentDim}).");
        return _gen.Decode(semantic, steps);
    }

    /// <summary>
    /// Produces a smoothly-varying sequence of utterances by drifting the semantic
    /// latent by DriftScale each step (a spoken phrase / prosodic contour).
    /// </summary>
    public List<DenseTensor<float>> ProduceSequence(DenseTensor<float> semantic, int length, int steps, Random rng)
    {
        if (length < 1) throw new ArgumentOutOfRangeException(nameof(length));
        if (semantic.Length != _gen.LatentDim)
            throw new ArgumentException($"semantic has {semantic.Length} dims; expected SemanticDim ({_gen.LatentDim}).");

        var seq = new List<DenseTensor<float>>(length);
        var z   = semantic.Clone();
        for (int t = 0; t < length; t++)
        {
            seq.Add(_gen.Decode(z, steps));
            if (_config.DriftScale > 0f)
                for (int d = 0; d < z.Length; d++)
                    z.Buffer[d] += RandomUtils.SampleNormal(0f, _config.DriftScale, rng);
        }
        return seq;
    }

    public void FlushAll() => _gen.FlushStore();
    public void Dispose()  => _gen.Dispose();
}
