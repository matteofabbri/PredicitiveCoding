using PredictiveCoding.Networks;
using PredictiveCoding.Core;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// EnsembleNetwork  —  Bayesian model averaging over PC networks
//
// Each member is an independent Network that may have a different depth,
// width, activation function, or learning rate.  Architectural diversity
// is the key property that makes the ensemble more robust than any single
// member: each member makes different errors, and averaging cancels them.
//
// Learning:
//   Every member learns independently on the same (input, label) pair with
//   its own weight update.  There is no gating or routing — all members are
//   always trained.
//
// Prediction (ProbabilityAveraging, default):
//   For each member i, compute the per-class energy vector F_i ∈ ℝ^C:
//     F_i(c) = F_i(x, one-hot(c))
//   Convert to a probability distribution:
//     p_i(c) = softmax(−β · F_i)(c)
//   Average uniformly across members:
//     p(c) = (1/N) Σ_i p_i(c)
//   Predict argmax_c p(c).
//   A member with a flat energy landscape (uncertain) contributes a near-
//   uniform p_i and therefore has little influence on the final decision.
//
// Prediction (EnergyAveraging):
//   F_avg(c) = (1/N) Σ_i F_i(c),  predict argmin_c F_avg(c).
//   Stays fully within the PC free-energy framework; works best when all
//   members have comparable energy scales.
//
// References:
//   Hansen & Salamon (1990) "Neural network ensembles", IEEE PAMI.
//   Lakshminarayanan et al. (2017) "Deep Ensembles", NeurIPS.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class EnsembleNetwork : IDisposable
{
    private readonly Network[]       _members;
    private readonly EnsembleConfig  _config;
    private readonly int             _numClasses;

    // ── Public state ─────────────────────────────────────────────────────────

    public int MemberCount => _members.Length;

    public EnsembleConfig Configuration => _config;

    // ── Construction ─────────────────────────────────────────────────────────

    public EnsembleNetwork(EnsembleConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _members = new Network[config.MemberConfigs.Length];
        for (int i = 0; i < _members.Length; i++)
        {
            var memberCfg = config.MemberConfigs[i] with
            {
                Path = Path.Combine(config.BasePath, $"member_{i}"),
            };
            _members[i] = new Network(memberCfg, new Random(rng.Next()));
        }

        _numClasses = config.MemberConfigs[0].LayerSizes[^1];
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs an independent learning step on every member.
    /// All members are always updated — no gating or routing.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        foreach (var m in _members)
            m.Learn(input, label, inferenceSteps);
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Aggregated prediction using the mode specified in
    /// <see cref="EnsembleConfig.PredictionMode"/>.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps) =>
        _config.PredictionMode == EnsemblePredictionMode.ProbabilityAveraging
            ? PredictByProbabilityAveraging(input, inferenceSteps)
            : PredictByEnergyAveraging(input, inferenceSteps);

    /// <summary>
    /// Returns the full aggregated prediction result with per-member
    /// class energies, averaged probabilities or averaged energies,
    /// and per-member argmin predictions for diagnostics.
    /// </summary>
    public EnsemblePrediction PredictDetailed(DenseTensor<float> input, int inferenceSteps)
    {
        var memberEnergies = new float[_members.Length][];
        var memberPreds    = new int[_members.Length];

        for (int i = 0; i < _members.Length; i++)
        {
            memberEnergies[i] = _members[i].ClassEnergies(input, inferenceSteps);
            memberPreds[i]    = ArgMin(memberEnergies[i]);
        }

        int finalPred;
        float[]? aggregatedProbs    = null;
        float[]? aggregatedEnergies = null;

        if (_config.PredictionMode == EnsemblePredictionMode.ProbabilityAveraging)
        {
            aggregatedProbs = AverageProbabilities(memberEnergies, _config.SoftmaxTemperature);
            finalPred       = ArgMax(aggregatedProbs);
        }
        else
        {
            aggregatedEnergies = AverageEnergies(memberEnergies);
            finalPred          = ArgMin(aggregatedEnergies);
        }

        return new EnsemblePrediction(
            finalPred, memberPreds, memberEnergies,
            aggregatedProbs, aggregatedEnergies);
    }

    // ── Aggregation internals ─────────────────────────────────────────────────

    private int PredictByProbabilityAveraging(DenseTensor<float> input, int inferenceSteps)
    {
        var memberEnergies = new float[_members.Length][];
        for (int i = 0; i < _members.Length; i++)
            memberEnergies[i] = _members[i].ClassEnergies(input, inferenceSteps);

        return ArgMax(AverageProbabilities(memberEnergies, _config.SoftmaxTemperature));
    }

    private int PredictByEnergyAveraging(DenseTensor<float> input, int inferenceSteps)
    {
        var memberEnergies = new float[_members.Length][];
        for (int i = 0; i < _members.Length; i++)
            memberEnergies[i] = _members[i].ClassEnergies(input, inferenceSteps);

        return ArgMin(AverageEnergies(memberEnergies));
    }

    /// <summary>
    /// Converts each member's energy vector to softmax(−β·F_i) then averages.
    /// </summary>
    private static float[] AverageProbabilities(float[][] memberEnergies, float beta)
    {
        int n = memberEnergies.Length;
        int c = memberEnergies[0].Length;
        var avg = new float[c];

        var probs = new float[c];
        foreach (var energies in memberEnergies)
        {
            // Numerically-stable softmax over −β·F_i
            float max = float.NegativeInfinity;
            for (int k = 0; k < c; k++)
            {
                float v = -beta * energies[k];
                if (v > max) max = v;
            }

            float sum = 0f;
            for (int k = 0; k < c; k++)
            {
                probs[k] = MathF.Exp(-beta * energies[k] - max);
                sum      += probs[k];
            }

            for (int k = 0; k < c; k++)
                avg[k] += probs[k] / sum;
        }

        for (int k = 0; k < c; k++) avg[k] /= n;
        return avg;
    }

    private static float[] AverageEnergies(float[][] memberEnergies)
    {
        int n = memberEnergies.Length;
        int c = memberEnergies[0].Length;
        var avg = new float[c];

        foreach (var energies in memberEnergies)
            for (int k = 0; k < c; k++)
                avg[k] += energies[k];

        for (int k = 0; k < c; k++) avg[k] /= n;
        return avg;
    }

    private static int ArgMin(float[] values)
    {
        int   best = 0;
        float min  = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] < min) { min = values[i]; best = i; }
        return best;
    }

    private static int ArgMax(float[] values)
    {
        int   best = 0;
        float max  = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > max) { max = values[i]; best = i; }
        return best;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>Average weight norm across all members.</summary>
    public float AverageWeightNorm()
    {
        float sum = 0f;
        foreach (var m in _members) sum += m.AverageWeightNorm();
        return sum / _members.Length;
    }

    /// <summary>Per-member average weight norms (length = MemberCount).</summary>
    public float[] MemberWeightNorms()
    {
        var norms = new float[_members.Length];
        for (int i = 0; i < _members.Length; i++)
            norms[i] = _members[i].AverageWeightNorm();
        return norms;
    }

    /// <summary>
    /// Flushes all member stores.  Required when LazyPersistence is enabled.
    /// </summary>
    public void FlushAll()
    {
        foreach (var m in _members) m.FlushStore();
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var m in _members) m.Dispose();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EnsemblePrediction  —  detailed prediction result from PredictDetailed
// ─────────────────────────────────────────────────────────────────────────────

/// <param name="PredictedClass">Final ensemble prediction.</param>
/// <param name="MemberPredictions">Per-member argmin-energy class prediction.</param>
/// <param name="MemberClassEnergies">Per-member per-class energy vectors.</param>
/// <param name="AggregatedProbabilities">
/// Averaged softmax distributions (non-null with ProbabilityAveraging).
/// </param>
/// <param name="AggregatedEnergies">
/// Averaged energy vector (non-null with EnergyAveraging).
/// </param>
public sealed record EnsemblePrediction(
    int       PredictedClass,
    int[]     MemberPredictions,
    float[][] MemberClassEnergies,
    float[]?  AggregatedProbabilities,
    float[]?  AggregatedEnergies);
