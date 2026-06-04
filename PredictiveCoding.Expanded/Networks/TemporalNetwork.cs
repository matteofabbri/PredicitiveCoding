using PredictiveCoding.Networks;
using PredictiveCoding.Core;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// TemporalNetwork  —  Predictive Coding network with temporal continuity
//
// Wraps a standard Network and carries a hidden-state buffer across timesteps.
// Before each inference the previous hidden states μ^{t-1} are injected as
// temporal priors, adding a quadratic penalty toward continuity:
//
//   grad_l += Π_t · (μ_l − μ_l^{t-1})
//
// This biases inference toward representations that vary smoothly in time,
// implementing the dynamic causal modelling idea of Friston (2008):
//   "Hierarchical models in the brain", PLoS Comput. Biol.
//
// Usage:
//   using var tnet = new TemporalNetwork(network, temporalPrecision: 1f);
//
//   foreach (var (input, label) in sequence)
//   {
//       tnet.Step(input, label, inferenceSteps: 20);   // learn
//   }
//   tnet.Reset();                                       // new sequence
//
//   int predicted = tnet.Predict(input, inferenceSteps: 20);
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TemporalNetwork : IDisposable
{
    /// <summary>The underlying stateless PC network (weights + store).</summary>
    public Network BaseNetwork { get; }

    /// <summary>
    /// Π_t : temporal precision — strength of the continuity gradient.
    /// Higher values enforce stronger temporal smoothness.
    /// Typical range: 0.5–5.0.
    /// </summary>
    public float TemporalPrecision { get; }

    /// <summary>True once at least one timestep has been processed.</summary>
    public bool HasPrior => _hiddenPriors != null;

    private DenseTensor<float>[]? _hiddenPriors;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <param name="net">Underlying network (TemporalNetwork takes ownership).</param>
    /// <param name="temporalPrecision">Π_t ≥ 0.  0 = pure feed-forward (no memory).</param>
    public TemporalNetwork(Network net, float temporalPrecision = 1f)
    {
        if (temporalPrecision < 0f)
            throw new ArgumentOutOfRangeException(nameof(temporalPrecision), "Must be >= 0.");
        if (net.LayerCount < 3)
            throw new ArgumentException(
                "TemporalNetwork requires at least 3 layers (input, ≥1 hidden, output).",
                nameof(net));

        BaseNetwork       = net;
        TemporalPrecision = temporalPrecision;
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Processes one timestep: injects the previous hidden state as a temporal
    /// prior, runs inference with both ends clamped, updates weights, then
    /// stores the resulting hidden states for the next call.
    /// </summary>
    public void Step(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        _hiddenPriors = BaseNetwork.LearnTemporal(
            input, label, inferenceSteps, _hiddenPriors, TemporalPrecision);
    }

    // ── Prediction ────────────────────────────────────────────────────────────

    /// <summary>
    /// Classifies <paramref name="input"/> conditioned on the current hidden
    /// state buffer (temporal context from the last <see cref="Step"/> call).
    /// Does not advance the hidden state — call <see cref="Advance"/> if you
    /// want the prediction to update the temporal context without training.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps) =>
        BaseNetwork.PredictTemporal(input, inferenceSteps, _hiddenPriors, TemporalPrecision);

    /// <summary>
    /// Runs inference and advances the hidden-state buffer without updating
    /// weights.  Use during evaluation to keep the temporal context rolling.
    /// </summary>
    public int Advance(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        _hiddenPriors = BaseNetwork.InferTemporal(
            input, inferenceSteps, _hiddenPriors, TemporalPrecision,
            clampTop: true, label: label);
        return BaseNetwork.PredictTemporal(input, inferenceSteps, _hiddenPriors, TemporalPrecision);
    }

    // ── Sequence control ─────────────────────────────────────────────────────

    /// <summary>
    /// Clears the hidden-state buffer.  Call at the start of each new
    /// independent sequence to prevent cross-sequence contamination.
    /// </summary>
    public void Reset() => _hiddenPriors = null;

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => BaseNetwork.Dispose();
}
