using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// Amygdala  —  affective salience / valence appraisal
//
// The amygdala assigns emotional significance to stimuli: it learns which inputs
// predict reward or threat and emits a valence/arousal signal that biases the rest
// of the brain (attention, memory, autonomic response).  Implemented as a PCN that
// maps a stimulus latent to an affect vector (e.g. [valence, arousal]); the arousal
// /salience it reports can steer Thalamus precision (threat → more gain).
//
// References: LeDoux (2000) emotion circuits; Pessoa (2010) salience.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record AmygdalaConfig
{
    /// <summary>Appraisal PCN: stimulus → affect vector.  LayerSizes[^1] = AffectDim.</summary>
    public NetworkConfig Appraisal { get; init; } = new();

    /// <summary>Number of affect dimensions (convention: 2 = [valence, arousal]).</summary>
    public int AffectDim { get; init; } = 2;

    public string BasePath { get; init; } = "./cortex/amygdala";

    public void Validate()
    {
        if (AffectDim < 1)
            throw new InvalidOperationException("AffectDim must be >= 1.");
        if (Appraisal.LayerSizes[^1] != AffectDim)
            throw new InvalidOperationException(
                $"Appraisal.LayerSizes[^1] ({Appraisal.LayerSizes[^1]}) must equal AffectDim ({AffectDim}).");
    }
}

public sealed class Amygdala : IDisposable
{
    private readonly Network        _appraisal;
    private readonly AmygdalaConfig _config;

    public int StimulusDim => _appraisal.Configuration.LayerSizes[0];
    public int AffectDim   => _config.AffectDim;

    public Amygdala(AmygdalaConfig config, Random rng)
    {
        config.Validate();
        _config    = config;
        _appraisal = new Network(config.Appraisal with { Path = config.BasePath }, rng);
    }

    /// <summary>Appraises a stimulus into an affect vector (e.g. [valence, arousal]).</summary>
    public DenseTensor<float> Appraise(DenseTensor<float> stimulus, int steps)
        => _appraisal.InferGetTop(stimulus, steps);

    /// <summary>Learns the stimulus → affect association.</summary>
    public void Learn(DenseTensor<float> stimulus, DenseTensor<float> affect, int steps)
    {
        if (affect.Length != _config.AffectDim)
            throw new ArgumentException($"affect has {affect.Length} elements; expected AffectDim ({_config.AffectDim}).");
        _appraisal.Learn(stimulus, affect, steps);
    }

    /// <summary>
    /// Salience = max |affect component|.  Returns a precision multiplier
    /// baseGain·(1 + salience) for steering Thalamus gain — a salient/threatening
    /// stimulus raises the gain.
    /// </summary>
    public float SalienceGain(DenseTensor<float> stimulus, int steps, float baseGain = 1f)
    {
        var affect   = Appraise(stimulus, steps);
        float salience = 0f;
        for (int i = 0; i < affect.Length; i++) salience = MathF.Max(salience, MathF.Abs(affect.Buffer[i]));
        return baseGain * (1f + salience);
    }

    public void FlushAll() => _appraisal.FlushStore();
    public void Dispose()  => _appraisal.Dispose();
}
