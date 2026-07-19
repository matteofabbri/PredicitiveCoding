using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Gated;

// ─────────────────────────────────────────────────────────────────────────────
// GatedStateNetwork  —  PCN with an LSTM-style working-memory gate
//
// Wraps a standard PC network with a working-memory cell m.  Each step encodes
// the input to a latent z, then a per-unit write gate decides how much of z
// overwrites m versus how much of the old memory is retained:
//
//   d_i  = |z_i − m_i|                         per-unit surprise
//   g_i  = σ( slope · (d_i − bias) )           write gate
//   m_i ← (1 − g_i) · m_i + g_i · z_i          gated update
//
// m is then installed as the top-layer prior (precision Π_m), so the next
// inference is pulled toward the retained context.  Stable units are held while
// surprising ones are refreshed — giving the network a controllable working
// memory for multi-step reasoning without any trained gate weights.
//
// References:
//   Hochreiter & Schmidhuber (1997) LSTM; O'Reilly & Frank (2006) gated WM.
//   Friston (2005) — precision-weighted top-down priors.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class GatedStateNetwork : IDisposable
{
    private readonly Network                  _network;
    private readonly GatedStateNetworkConfig  _config;
    private DenseTensor<float>?               _memory;   // working-memory cell m

    /// <summary>Dimensionality of the working-memory / latent vector.</summary>
    public int MemoryDim => _network.Configuration.LayerSizes[^1];

    /// <summary>True once at least one step has populated the memory since Reset.</summary>
    public bool HasMemory => _memory != null;

    /// <summary>A clone of the current working-memory state, or null before the first step.</summary>
    public DenseTensor<float>? MemoryState => _memory?.Clone();

    public GatedStateNetworkConfig Configuration => _config;

    public GatedStateNetwork(GatedStateNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        var netCfg = config.NetworkConfig with { Path = config.BasePath };
        _network = new Network(netCfg, rng);
    }

    // ── Memory update ──────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes <paramref name="input"/>, applies the gated update to the
    /// working memory, installs it as the top-layer prior, and returns a clone of
    /// the new memory.  Does not modify weights.
    /// </summary>
    public DenseTensor<float> Step(DenseTensor<float> input, int inferenceSteps)
    {
        var z = _network.InferGetTop(input, inferenceSteps);
        UpdateMemory(z);
        ApplyMemoryPrior();
        return _memory!.Clone();
    }

    // ── Prediction ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Classifies <paramref name="input"/> with the current working memory active
    /// as the top-layer prior.  Call <see cref="Step"/> first to populate memory;
    /// with no memory yet this behaves like a plain network prediction.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        ApplyMemoryPrior();
        return _network.Predict(input, inferenceSteps);
    }

    /// <summary>Prediction with free-energy diagnostics under the active memory prior.</summary>
    public (int cls, float energy) PredictWithEnergy(DenseTensor<float> input, int inferenceSteps)
    {
        ApplyMemoryPrior();
        return _network.PredictWithEnergy(input, inferenceSteps);
    }

    // ── Learning ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Supervised learning step with gated working memory: refresh the memory from
    /// the current input, install it as the top-layer prior, then run supervised
    /// PC learning.  The memory bias persists into the weight update so the
    /// network learns to use its retained context.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        var z = _network.InferGetTop(input, inferenceSteps);
        UpdateMemory(z);
        ApplyMemoryPrior();
        _network.Learn(input, label, inferenceSteps);
    }

    // ── Sequence control ───────────────────────────────────────────────────────

    /// <summary>
    /// Clears the working memory and restores a flat top-layer prior.  Call at the
    /// start of each new independent sequence.
    /// </summary>
    public void Reset()
    {
        _memory = null;
        _network.SetTopLayerPrior(null, 1f);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public float ComputeEnergy() => _network.ComputeEnergy();

    public void FlushStore() => _network.FlushStore();

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>Applies the per-unit gated update m ← (1−g)·m + g·z.</summary>
    private void UpdateMemory(DenseTensor<float> z)
    {
        if (_memory == null)
        {
            // First observation: nothing to retain — adopt the encoding wholesale.
            _memory = z.Clone();
            return;
        }

        float slope = _config.GateSlope;
        float bias  = _config.GateBias;
        for (int i = 0; i < _memory.Length; i++)
        {
            float d = MathF.Abs(z.Buffer[i] - _memory.Buffer[i]);
            float g = 1f / (1f + MathF.Exp(-slope * (d - bias)));
            _memory.Buffer[i] = (1f - g) * _memory.Buffer[i] + g * z.Buffer[i];
        }
    }

    private void ApplyMemoryPrior()
    {
        if (_memory != null)
            _network.SetTopLayerPrior(_memory, _config.WorkingMemoryPrecision);
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => _network.Dispose();
}
