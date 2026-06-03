using PredictiveCoding.Networks;
using PredictiveCoding.Core;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// TemporalMixtureOfExperts  —  MoE ensemble of temporal PC networks
//
// Combines energy-based gating (MixtureOfExperts) with per-expert temporal
// context (TemporalNetwork).  Each expert i maintains its own hidden-state
// buffer μ^{t-1}_i and temporal precision Π_t[i], allowing different experts
// to specialise on different temporal regimes of the input sequence:
//
//   • Slow experts  (high Π_t): enforce strong continuity; excel on gradual
//     trends and long-range structure.
//   • Fast experts  (low  Π_t): react quickly; excel on abrupt regime shifts.
//
// On every Step the gating distribution is derived from each expert's
// temporal free energy (conditioned on its own prior):
//
//   g_i = softmax(−β · F_i^temporal(x_t, y_t | μ_i^{t-1}))
//
// Only the expert with the best temporal prediction of the current input
// receives a large weight update — temporal specialisation emerges without
// explicit supervision.
//
// On Predict the minimum-temporal-energy winner-take-all rule is used:
//
//   i* = argmin_i  min_c  F_i^temporal(x_t, one-hot(c) | μ_i^{t-1})
//
// References:
//   Jacobs et al. (1991) "Adaptive mixtures of local experts", Neural Comput.
//   Friston (2008) "Hierarchical models in the brain", PLoS Comput. Biol.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TemporalMixtureOfExperts : IDisposable
{
    private readonly Network[]                _experts;
    private readonly float[]                  _temporalPrecisions;

    // Per-expert hidden-state priors: _hiddenPriors[i] is null before the first
    // Step of a new sequence, then holds μ^{t-1} for expert i.
    private readonly DenseTensor<float>[]?[]  _hiddenPriors;

    private readonly TemporalMoEConfig        _config;

    // ── Public state ─────────────────────────────────────────────────────────

    public int ExpertCount => _experts.Length;

    public TemporalMoEConfig Configuration => _config;

    /// <summary>
    /// Gate weights g_i computed during the most recent <see cref="Step"/> call.
    /// Length = <see cref="ExpertCount"/>.
    /// </summary>
    public float[] LastGateWeights { get; private set; }

    /// <summary>
    /// Temporal free energies F_i^temporal recorded during the most recent
    /// <see cref="Step"/> call (before weight updates).
    /// Length = <see cref="ExpertCount"/>.
    /// </summary>
    public float[] LastExpertEnergies { get; private set; }

    // ── Construction ─────────────────────────────────────────────────────────

    public TemporalMixtureOfExperts(TemporalMoEConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _experts            = new Network[config.ExpertCount];
        _temporalPrecisions = new float[config.ExpertCount];
        _hiddenPriors       = new DenseTensor<float>[]?[config.ExpertCount];

        for (int i = 0; i < config.ExpertCount; i++)
        {
            var expertCfg = config.ExpertConfig with
            {
                Path = Path.Combine(config.BasePath, $"expert_{i}"),
            };
            _experts[i]            = new Network(expertCfg, new Random(rng.Next()));
            _temporalPrecisions[i] = config.PrecisionFor(i);
        }

        LastGateWeights    = new float[config.ExpertCount];
        LastExpertEnergies = new float[config.ExpertCount];
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Processes one timestep across all experts:
    /// <list type="number">
    ///   <item>
    ///     Probe each expert's temporal free energy conditioned on its current
    ///     hidden-state prior — no weight updates.
    ///   </item>
    ///   <item>
    ///     Compute gate weights  g_i = softmax(−β · F_i^temporal).
    ///   </item>
    ///   <item>
    ///     Run a gated temporal weight update for each expert, then advance its
    ///     hidden-state prior to μ_i^t.
    ///   </item>
    /// </list>
    /// Experts with low temporal free energy (best fit to the current regime)
    /// receive the highest gate weight and update their weights most strongly.
    /// </summary>
    public void Step(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        // Phase 1: collect temporal energies (no weight update)
        for (int i = 0; i < _experts.Length; i++)
            LastExpertEnergies[i] = _experts[i].InferEnergyTemporal(
                input, label, inferenceSteps, _hiddenPriors[i], _temporalPrecisions[i]);

        // Compute gate weights from temporal energies
        MixtureOfExperts.ComputeGates(
            LastExpertEnergies, LastGateWeights,
            _config.GatingTemperature, _config.MinGateWeight);

        // Phase 2: gated temporal weight update + advance each expert's prior
        for (int i = 0; i < _experts.Length; i++)
        {
            var (newPriors, _) = _experts[i].LearnTemporalGated(
                input, label, inferenceSteps,
                _hiddenPriors[i], _temporalPrecisions[i], LastGateWeights[i]);

            _hiddenPriors[i] = newPriors;
        }
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Classifies <paramref name="input"/> via temporal minimum-energy
    /// winner-take-all:
    ///   i* = argmin_i  min_c  F_i^temporal(x, one-hot(c) | μ_i^{t-1})
    /// Does not advance any expert's hidden-state prior — call
    /// <see cref="Advance"/> for context-rolling evaluation without training.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        float bestEnergy = float.MaxValue;
        int   bestPred   = 0;

        for (int i = 0; i < _experts.Length; i++)
        {
            var (cls, energy) = _experts[i].PredictWithEnergyTemporal(
                input, inferenceSteps, _hiddenPriors[i], _temporalPrecisions[i]);

            if (energy < bestEnergy) { bestEnergy = energy; bestPred = cls; }
        }

        return bestPred;
    }

    /// <summary>
    /// Classifies <paramref name="input"/> and returns per-expert predictions,
    /// energies, and the winning expert index for diagnostics.
    /// Does not advance hidden-state priors.
    /// </summary>
    public TemporalMoEPrediction PredictDetailed(DenseTensor<float> input, int inferenceSteps)
    {
        var expertCls    = new int[_experts.Length];
        var expertEnergy = new float[_experts.Length];
        float bestEnergy = float.MaxValue;
        int   winnerIdx  = 0;

        for (int i = 0; i < _experts.Length; i++)
        {
            (expertCls[i], expertEnergy[i]) = _experts[i].PredictWithEnergyTemporal(
                input, inferenceSteps, _hiddenPriors[i], _temporalPrecisions[i]);

            if (expertEnergy[i] < bestEnergy) { bestEnergy = expertEnergy[i]; winnerIdx = i; }
        }

        return new TemporalMoEPrediction(
            expertCls[winnerIdx], winnerIdx, expertCls, expertEnergy);
    }

    // ── Sequence control ─────────────────────────────────────────────────────

    /// <summary>
    /// Advances every expert's hidden-state prior from the current input without
    /// updating weights, then returns the winning prediction.
    /// Use during evaluation to keep all temporal contexts rolling.
    /// </summary>
    public int Advance(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        for (int i = 0; i < _experts.Length; i++)
            _hiddenPriors[i] = _experts[i].InferTemporal(
                input, inferenceSteps, _hiddenPriors[i], _temporalPrecisions[i],
                clampTop: true, label: label);

        return Predict(input, inferenceSteps);
    }

    /// <summary>
    /// Clears every expert's hidden-state prior.  Call at the start of each new
    /// independent sequence to prevent cross-sequence contamination.
    /// </summary>
    public void Reset()
    {
        for (int i = 0; i < _experts.Length; i++)
            _hiddenPriors[i] = null;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>Average weight norm across all experts.</summary>
    public float AverageWeightNorm()
    {
        float sum = 0f;
        foreach (var e in _experts) sum += e.AverageWeightNorm();
        return sum / _experts.Length;
    }

    /// <summary>Per-expert average weight norms (length = ExpertCount).</summary>
    public float[] ExpertWeightNorms()
    {
        var norms = new float[_experts.Length];
        for (int i = 0; i < _experts.Length; i++)
            norms[i] = _experts[i].AverageWeightNorm();
        return norms;
    }

    /// <summary>
    /// Returns true for expert <paramref name="i"/> if it has processed at
    /// least one timestep and holds a non-null temporal prior.
    /// </summary>
    public bool ExpertHasPrior(int i) => _hiddenPriors[i] != null;

    /// <summary>
    /// Flushes all expert stores.  Required when LazyPersistence is enabled.
    /// </summary>
    public void FlushAll()
    {
        foreach (var e in _experts) e.FlushStore();
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var e in _experts) e.Dispose();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalMoEPrediction  —  detailed per-expert result from PredictDetailed
// ─────────────────────────────────────────────────────────────────────────────

public sealed record TemporalMoEPrediction(
    int     PredictedClass,
    int     WinningExpert,
    int[]   ExpertClasses,
    float[] ExpertEnergies);
