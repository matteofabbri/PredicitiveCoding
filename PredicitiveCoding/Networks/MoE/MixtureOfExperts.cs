using PredictiveCoding.Networks;
using PredictiveCoding.Core;

namespace PredictiveCoding.Networks.MoE;

// ─────────────────────────────────────────────────────────────────────────────
// MixtureOfExperts  —  PC network ensemble with energy-based gating
//
// Each expert is an independent Network.  On every learning step the gate
// weights are computed from the experts' free energies:
//
//   g_i = softmax(−β · F_i(x, y))
//
// The expert with the lowest free energy (best prediction of the current
// input/label pair) receives the highest gate weight and therefore performs
// the largest weight update.  Experts automatically specialize over training
// without any explicit routing supervision.
//
// On prediction the expert with the minimum winning energy is chosen:
//
//   i* = argmin_i  min_c  F_i(x, one-hot(c))
//
// Reference:
//   Jacobs et al. (1991) "Adaptive mixtures of local experts", Neural Comput.
//   (PC-native energy gating is novel to this library)
// ─────────────────────────────────────────────────────────────────────────────

public sealed class MixtureOfExperts : IDisposable
{
    private readonly Network[]              _experts;
    private readonly MixtureOfExpertsConfig _config;

    // ── Public state ─────────────────────────────────────────────────────────

    public int ExpertCount => _experts.Length;

    public MixtureOfExpertsConfig Configuration => _config;

    /// <summary>
    /// Gate weights g_i computed on the most recent <see cref="Learn"/> call.
    /// Length = <see cref="ExpertCount"/>.
    /// </summary>
    public float[] LastGateWeights { get; private set; }

    /// <summary>
    /// Free energies F_i recorded during the most recent <see cref="Learn"/> call
    /// (before weight updates).  Length = <see cref="ExpertCount"/>.
    /// </summary>
    public float[] LastExpertEnergies { get; private set; }

    // ── Construction ─────────────────────────────────────────────────────────

    public MixtureOfExperts(MixtureOfExpertsConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _experts = new Network[config.ExpertCount];
        for (int i = 0; i < config.ExpertCount; i++)
        {
            var expertCfg = config.ExpertConfig with
            {
                Path = Path.Combine(config.BasePath, $"expert_{i}"),
            };
            _experts[i] = new Network(expertCfg, new Random(rng.Next()));
        }

        LastGateWeights    = new float[config.ExpertCount];
        LastExpertEnergies = new float[config.ExpertCount];
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// One learning step across all experts:
    /// <list type="number">
    ///   <item>Run inference on every expert to collect free energies F_i.</item>
    ///   <item>Compute gate weights  g_i = softmax(−β · F_i).</item>
    ///   <item>Update each expert's weights scaled by g_i.</item>
    /// </list>
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        // Phase 1: collect energies (no weight update yet)
        for (int i = 0; i < _experts.Length; i++)
            LastExpertEnergies[i] = _experts[i].InferEnergy(input, label, inferenceSteps);

        // Compute gate weights
        ComputeGates(LastExpertEnergies, LastGateWeights,
                     _config.GatingTemperature, _config.MinGateWeight);

        // Phase 2: gated weight updates
        for (int i = 0; i < _experts.Length; i++)
            _experts[i].LearnGated(input, label, inferenceSteps, LastGateWeights[i]);
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Classifies <paramref name="input"/> via minimum-energy winner-take-all:
    ///   i* = argmin_i  min_c  F_i(x, one-hot(c))
    /// The expert with the lowest winning energy makes the final prediction.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        float bestEnergy = float.MaxValue;
        int   bestPred   = 0;

        for (int i = 0; i < _experts.Length; i++)
        {
            var (cls, energy) = _experts[i].PredictWithEnergy(input, inferenceSteps);
            if (energy < bestEnergy) { bestEnergy = energy; bestPred = cls; }
        }

        return bestPred;
    }

    /// <summary>
    /// Classifies <paramref name="input"/> and also returns per-expert
    /// predictions and energies for diagnostics.
    /// </summary>
    public MoEPrediction PredictDetailed(DenseTensor<float> input, int inferenceSteps)
    {
        var expertCls     = new int[_experts.Length];
        var expertEnergy  = new float[_experts.Length];
        float bestEnergy  = float.MaxValue;
        int   winnerIdx   = 0;

        for (int i = 0; i < _experts.Length; i++)
        {
            (expertCls[i], expertEnergy[i]) = _experts[i].PredictWithEnergy(input, inferenceSteps);
            if (expertEnergy[i] < bestEnergy) { bestEnergy = expertEnergy[i]; winnerIdx = i; }
        }

        return new MoEPrediction(expertCls[winnerIdx], winnerIdx, expertCls, expertEnergy);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>
    /// Average weight norm across all layers and all experts.
    /// </summary>
    public float AverageWeightNorm()
    {
        float sum = 0f;
        foreach (var e in _experts) sum += e.AverageWeightNorm();
        return sum / _experts.Length;
    }

    /// <summary>
    /// Per-expert average weight norm array (length = ExpertCount).
    /// </summary>
    public float[] ExpertWeightNorms()
    {
        var norms = new float[_experts.Length];
        for (int i = 0; i < _experts.Length; i++)
            norms[i] = _experts[i].AverageWeightNorm();
        return norms;
    }

    /// <summary>
    /// Flushes all expert stores.  Required when LazyPersistence is enabled.
    /// </summary>
    public void FlushAll()
    {
        foreach (var e in _experts) e.FlushStore();
    }

    // ── Gating ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Computes gate weights in-place:
    ///   gates[i] = softmax(−β · energies[i])
    /// with optional minimum weight floor and renormalisation.
    /// </summary>
    public static void ComputeGates(
        float[] energies, float[] gates,
        float   temperature, float minGate)
    {
        int n = energies.Length;

        // Numerically-stable softmax over −β·F_i
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            float v = -temperature * energies[i];
            if (v > max) max = v;
        }

        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            gates[i] = MathF.Exp(-temperature * energies[i] - max);
            sum += gates[i];
        }
        for (int i = 0; i < n; i++) gates[i] /= sum;

        // Apply minimum gate floor: clamp values below minGate, then renormalise.
        // Due to renormalisation the final values may be epsilon below minGate;
        // the guarantee is approximate (within floating-point rounding).
        if (minGate > 0f)
        {
            sum = 0f;
            for (int i = 0; i < n; i++) { gates[i] = MathF.Max(minGate, gates[i]); sum += gates[i]; }
            for (int i = 0; i < n; i++) gates[i] /= sum;
        }
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var e in _experts) e.Dispose();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MoEPrediction  —  detailed per-expert prediction result
// ─────────────────────────────────────────────────────────────────────────────

public sealed record MoEPrediction(
    int     PredictedClass,
    int     WinningExpert,
    int[]   ExpertClasses,
    float[] ExpertEnergies);
