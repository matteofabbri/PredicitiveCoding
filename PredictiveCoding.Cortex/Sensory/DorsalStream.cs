using PredictiveCoding.Core;
using PredictiveCoding.Expanded.DeepRecurrent;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// DorsalStream  —  the "where / how" visual pathway (V1 → MT/V5 → parietal)
//
// Complements the ventral VisualCortex ("what").  Each frame is encoded
// spatially by a VisualCortex (no readout); the sequence of object latents is
// integrated over time by a DeepRecurrentNetwork (cascade of temporal PC stages,
// our motion-modelling structure), and a readout classifies motion (e.g.
// direction).  Captures spatio-temporal / motion structure rather than identity.
//
// References: Goodale & Milner (1992) two-stream hypothesis; Friston (2008).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record DorsalStreamConfig
{
    /// <summary>Per-frame spatial encoder (use Classifier = null).</summary>
    public VisualCortexConfig Spatial { get; init; } = new();

    /// <summary>Temporal motion integrator over the per-frame latents.
    /// Stages[0].LayerSizes[0] must equal Spatial's IT latent dim.</summary>
    public DeepRecurrentNetworkConfig Temporal { get; init; } = new();

    /// <summary>Motion readout. LayerSizes[0] must equal the temporal top latent dim.</summary>
    public NetworkConfig MotionReadout { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/dorsal";

    public void Validate()
    {
        Spatial.Validate();
        Temporal.Validate();
        if (Temporal.Stages[0].LayerSizes[0] != Spatial.ItLatentDim)
            throw new InvalidOperationException(
                $"Temporal.Stages[0].LayerSizes[0] ({Temporal.Stages[0].LayerSizes[0]}) must equal " +
                $"the spatial IT latent dim ({Spatial.ItLatentDim}).");
        if (MotionReadout.LayerSizes[0] != Temporal.Stages[^1].LayerSizes[^1])
            throw new InvalidOperationException(
                $"MotionReadout.LayerSizes[0] ({MotionReadout.LayerSizes[0]}) must equal " +
                $"the temporal top latent dim ({Temporal.Stages[^1].LayerSizes[^1]}).");
    }
}

public sealed class DorsalStream : IDisposable
{
    private readonly VisualCortex          _spatial;
    private readonly DeepRecurrentNetwork  _temporal;
    private readonly Network               _readout;

    public DorsalStream(DorsalStreamConfig config, Random rng)
    {
        config.Validate();
        _spatial  = new VisualCortex(config.Spatial with { Classifier = null, BasePath = Path.Combine(config.BasePath, "spatial") }, new Random(rng.Next()));
        _temporal = new DeepRecurrentNetwork(config.Temporal with { BasePath = Path.Combine(config.BasePath, "temporal") }, new Random(rng.Next()));
        _readout  = new Network(config.MotionReadout with { Path = Path.Combine(config.BasePath, "readout") }, new Random(rng.Next()));
    }

    /// <summary>
    /// Learns one motion sequence: spatially encode each frame, integrate the
    /// latents over time, then train the readout on the final motion state.
    /// </summary>
    public void LearnSequence(IReadOnlyList<DenseTensor<float>> frames, DenseTensor<float> motionLabel, int steps)
    {
        _temporal.Reset();
        DenseTensor<float> top = null!;
        foreach (var frame in frames)
        {
            _spatial.Learn(frame, steps);                 // refine the spatial encoder
            var it = _spatial.Encode(frame, steps);
            top    = _temporal.Step(it, steps);           // integrate motion (learning)
        }
        _readout.Learn(top, motionLabel, steps);
    }

    /// <summary>Classifies the motion of a sequence.</summary>
    public int PredictMotion(IReadOnlyList<DenseTensor<float>> frames, int steps)
    {
        _temporal.Reset();
        DenseTensor<float> top = null!;
        foreach (var frame in frames)
        {
            var it = _spatial.Encode(frame, steps);
            top    = _temporal.Encode(it, steps);         // integrate motion (inference)
        }
        return _readout.Predict(top, steps);
    }

    public void FlushAll()
    {
        _spatial.FlushAll();
        _temporal.FlushAll();
        _readout.FlushStore();
    }

    public void Dispose()
    {
        _spatial.Dispose();
        _temporal.Dispose();
        _readout.Dispose();
    }
}
