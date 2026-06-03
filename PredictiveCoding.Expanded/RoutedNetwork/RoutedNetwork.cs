using PredictiveCoding.Core;
using PredictiveCoding.Networks;
using PredictiveCoding.Expanded.Networks;

namespace PredictiveCoding.Expanded.RoutedNetwork;

// ─────────────────────────────────────────────────────────────────────────────
// RoutedNetwork  —  PC ensemble with learned router + PE-based gating
//
// Gating at each learning step:
//
//   1. Probe all N experts for their free energies F_i(x, y).
//   2. PE-based gates:   pe_i   = softmax(−β · F_i)
//   3. Router inference: router_i = softmax(router(x) / β)
//   4. Mixed gates:      g_i    = (1−α)·router_i + α·pe_i  [renormalised]
//   5. Router learns to predict pe-gates from x alone:
//        router.Learn(x, pe_gates)  — supervised PC step
//   6. Top-k experts receive gated weight updates.
//
// At prediction time only the router runs, selecting top-k experts without
// probing all N — amortising the cost over training.
//
// References:
//   Shazeer et al. (2017) MoE — expert selection motivation.
//   PC-native energy gating (Jacobs 1991 + Friston 2005 free energy).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class RoutedNetwork : IDisposable
{
    private readonly Network[]           _experts;
    private readonly Network             _router;
    private readonly RoutedNetworkConfig _config;

    public int                  ExpertCount    => _experts.Length;
    public RoutedNetworkConfig  Configuration  => _config;

    /// <summary>Mixed gate weights computed on the most recent <see cref="Learn"/> call.</summary>
    public float[] LastGateWeights    { get; private set; }

    /// <summary>Raw PE energies from the most recent <see cref="Learn"/> call.</summary>
    public float[] LastExpertEnergies { get; private set; }

    /// <summary>Router-predicted gate weights from the most recent <see cref="Learn"/> call.</summary>
    public float[] LastRouterWeights  { get; private set; }

    public RoutedNetwork(RoutedNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _experts = new Network[config.ExpertCount];
        for (int i = 0; i < config.ExpertCount; i++)
        {
            var cfg = config.ExpertConfig with
            {
                Path = Path.Combine(config.BasePath, $"expert_{i}"),
            };
            _experts[i] = new Network(cfg, new Random(rng.Next()));
        }

        var routerCfg = config.RouterConfig with
        {
            Path = Path.Combine(config.BasePath, "router"),
        };
        _router = new Network(routerCfg, new Random(rng.Next()));

        LastGateWeights    = new float[config.ExpertCount];
        LastExpertEnergies = new float[config.ExpertCount];
        LastRouterWeights  = new float[config.ExpertCount];
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        // Phase 1: probe all experts (no weight update yet).
        for (int i = 0; i < _experts.Length; i++)
            LastExpertEnergies[i] = _experts[i].InferEnergy(input, label, inferenceSteps);

        // PE-based gates: softmax(−β · F_i).
        MixtureOfExperts.ComputeGates(
            LastExpertEnergies, LastGateWeights,
            _config.GatingTemperature, _config.MinGateWeight);

        // Router-based gates from input alone.
        var routerLogits = _router.InferGetTop(input, inferenceSteps);
        ComputeRouterGates(routerLogits, LastRouterWeights, _config.GatingTemperature);

        // Mix: g_i = (1−α)·router_i + α·pe_i, then renormalise.
        float alpha = _config.PeMixAlpha;
        for (int i = 0; i < _experts.Length; i++)
            LastGateWeights[i] = (1f - alpha) * LastRouterWeights[i] + alpha * LastGateWeights[i];
        Renormalize(LastGateWeights, _config.MinGateWeight);

        // Phase 2: train router to predict the mixed gates from input.
        var gateTarget = new DenseTensor<float>(_experts.Length);
        for (int i = 0; i < _experts.Length; i++)
            gateTarget.Buffer[i] = LastGateWeights[i];
        _router.Learn(input, gateTarget, inferenceSteps);

        // Phase 3: gated weight update for top-k experts only.
        int[] topK = GetTopK(LastGateWeights, _config.TopK);
        for (int i = 0; i < _experts.Length; i++)
        {
            float gw = Array.IndexOf(topK, i) >= 0 ? LastGateWeights[i] : 0f;
            _experts[i].LearnGated(input, label, inferenceSteps, gw);
        }
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Classifies using the router to select top-k experts, then picks the
    /// expert with the lowest winning energy.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
        => PredictDetailed(input, inferenceSteps).PredictedClass;

    public RoutedPrediction PredictDetailed(DenseTensor<float> input, int inferenceSteps)
    {
        var routerLogits = _router.InferGetTop(input, inferenceSteps);
        var routerGates  = new float[_experts.Length];
        ComputeRouterGates(routerLogits, routerGates, _config.GatingTemperature);

        int[] activeExperts = GetTopK(routerGates, _config.TopK);

        float bestEnergy = float.MaxValue;
        int   bestClass  = 0;
        int   winnerIdx  = activeExperts[0];

        var expertCls    = new int[_experts.Length];
        var expertEnergy = new float[_experts.Length];

        for (int rank = 0; rank < activeExperts.Length; rank++)
        {
            int i = activeExperts[rank];
            (expertCls[i], expertEnergy[i]) = _experts[i].PredictWithEnergy(input, inferenceSteps);
            if (expertEnergy[i] < bestEnergy)
            {
                bestEnergy = expertEnergy[i];
                bestClass  = expertCls[i];
                winnerIdx  = i;
            }
        }

        return new RoutedPrediction(bestClass, winnerIdx, activeExperts, expertCls, expertEnergy, routerGates);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public float AverageWeightNorm()
    {
        float sum = 0f;
        foreach (var e in _experts) sum += e.AverageWeightNorm();
        return sum / _experts.Length;
    }

    public void FlushAll()
    {
        foreach (var e in _experts) e.FlushStore();
        _router.FlushStore();
    }

    // ── Gating helpers ───────────────────────────────────────────────────────

    private static void ComputeRouterGates(DenseTensor<float> logits, float[] gates, float temperature)
    {
        int   n   = logits.Length;
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (logits.Buffer[i] > max) max = logits.Buffer[i];

        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            gates[i] = MathF.Exp((logits.Buffer[i] - max) / temperature);
            sum += gates[i];
        }
        for (int i = 0; i < n; i++) gates[i] /= sum;
    }

    private static void Renormalize(float[] gates, float minGate)
    {
        float sum = 0f;
        for (int i = 0; i < gates.Length; i++) { gates[i] = MathF.Max(minGate, gates[i]); sum += gates[i]; }
        for (int i = 0; i < gates.Length; i++) gates[i] /= sum;
    }

    // Simple O(n·k) top-k selection; n = ExpertCount is small (≤ 32 typically).
    private static int[] GetTopK(float[] gates, int k)
    {
        var indices = new int[k];
        var used    = new bool[gates.Length];
        for (int rank = 0; rank < k; rank++)
        {
            int best = -1;
            for (int i = 0; i < gates.Length; i++)
                if (!used[i] && (best < 0 || gates[i] > gates[best])) best = i;
            indices[rank] = best;
            used[best]    = true;
        }
        return indices;
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var e in _experts) e.Dispose();
        _router.Dispose();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoutedPrediction  —  detailed per-expert prediction result
// ─────────────────────────────────────────────────────────────────────────────

public sealed record RoutedPrediction(
    int     PredictedClass,
    int     WinningExpert,
    int[]   ActiveExperts,
    int[]   ExpertClasses,
    float[] ExpertEnergies,
    float[] RouterGates);
