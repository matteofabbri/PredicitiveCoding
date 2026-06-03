using PredictiveCoding.Core;
using PredictiveCoding.Expanded.GenerativeNetwork;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// DefaultModeNetwork  —  offline generative imagination / simulation
//
// The default mode network is active during rest and internally-generated
// cognition (memory, prospection, mind-wandering): the brain runs its generative
// model "offline", sampling experiences rather than perceiving them.  Implemented
// over the GenerativeNetwork: it learns a generative model of experience and can
// then sample novel samples, or roll a latent random-walk to imagine a coherent
// sequence (simulated trajectory).
//
// References: Raichle et al. (2001) DMN; Hassabis & Maguire (2007) scene construction.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record DefaultModeNetworkConfig
{
    public GenerativeNetworkConfig Generative { get; init; } = new();

    /// <summary>Std-dev of the latent random walk used by <see cref="DefaultModeNetwork.ImagineSequence"/>.</summary>
    public float DriftScale { get; init; } = 0.3f;

    public string BasePath { get; init; } = "./cortex/dmn";

    public void Validate()
    {
        if (DriftScale < 0f) throw new InvalidOperationException("DriftScale must be >= 0.");
    }
}

public sealed class DefaultModeNetwork : IDisposable
{
    private readonly GenerativeNetwork        _gen;
    private readonly DefaultModeNetworkConfig _config;

    public int LatentDim => _gen.LatentDim;
    public int InputDim  => _gen.InputDim;

    public DefaultModeNetwork(DefaultModeNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        _gen    = new GenerativeNetwork(config.Generative with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Learns the generative model of experience (awake encoding).</summary>
    public void LearnFromExperience(DenseTensor<float> input, int steps) => _gen.Learn(input, steps);

    /// <summary>Imagines a single novel sample by decoding z ~ N(0, I).</summary>
    public DenseTensor<float> Imagine(int steps, Random rng) => _gen.Sample(steps, rng);

    /// <summary>
    /// Imagines a coherent sequence by decoding a latent random walk: z starts at
    /// N(0, I) and drifts by DriftScale each step, so consecutive imagined samples
    /// vary smoothly — a simulated trajectory / prospective episode.
    /// </summary>
    public List<DenseTensor<float>> ImagineSequence(int length, int steps, Random rng)
    {
        if (length < 1) throw new ArgumentOutOfRangeException(nameof(length));

        var seq = new List<DenseTensor<float>>(length);
        var z   = RandomUtils.SampleNormal([LatentDim], 0f, 1f, rng);
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
