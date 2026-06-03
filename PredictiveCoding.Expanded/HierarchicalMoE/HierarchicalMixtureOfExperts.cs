using PredictiveCoding.Core;
using PredictiveCoding.Networks;
using PredictiveCoding.Expanded.Networks;

namespace PredictiveCoding.Expanded.HierarchicalMoE;

// ─────────────────────────────────────────────────────────────────────────────
// HierarchicalMixtureOfExperts  —  two-level energy-gated mixture of PC experts
//
// Experts are partitioned into groups.  Learning runs two nested energy gates:
//
//   1. Probe every expert's free energy F_{g,e} = InferEnergy(x, y).
//   2. Within-group gate  w_{g,e} = softmax_e(−β₂ F_{g,e}).
//   3. Group energy       F_g     = Σ_e w_{g,e} F_{g,e}.
//   4. Across-group gate  π_g     = softmax_g(−β₁ F_g).
//   5. Responsibility     r_{g,e} = π_g · w_{g,e}, used to scale each expert's
//      gated weight update.
//
// Prediction is hierarchical winner-take-all: pick the group whose best expert
// has the lowest winning energy, then return that expert's class.  Specialisation
// is emergent and unsupervised — only free energy decides routing.
//
// References:
//   Jordan & Jacobs (1994); Jacobs et al. (1991).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class HierarchicalMixtureOfExperts : IDisposable
{
    private readonly Network[][]            _groups;   // [group][expert]
    private readonly HierarchicalMoEConfig _config;

    public int GroupCount      => _config.GroupCount;
    public int ExpertsPerGroup => _config.ExpertsPerGroup;
    public HierarchicalMoEConfig Configuration => _config;

    /// <summary>Across-group gate π_g from the most recent <see cref="Learn"/> call.</summary>
    public float[] LastGroupGates { get; }

    /// <summary>Responsibilities r_{g,e} from the most recent <see cref="Learn"/> call.</summary>
    public float[][] LastResponsibilities { get; }

    public HierarchicalMixtureOfExperts(HierarchicalMoEConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _groups = new Network[config.GroupCount][];
        for (int g = 0; g < config.GroupCount; g++)
        {
            _groups[g] = new Network[config.ExpertsPerGroup];
            for (int e = 0; e < config.ExpertsPerGroup; e++)
            {
                var expertCfg = config.ExpertConfig with
                {
                    Path = Path.Combine(config.BasePath, $"group_{g}", $"expert_{e}"),
                };
                _groups[g][e] = new Network(expertCfg, new Random(rng.Next()));
            }
        }

        LastGroupGates       = new float[config.GroupCount];
        LastResponsibilities = new float[config.GroupCount][];
        for (int g = 0; g < config.GroupCount; g++)
            LastResponsibilities[g] = new float[config.ExpertsPerGroup];
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// One two-level gated learning step.  Computes within-group and across-group
    /// gates from the experts' free energies, then updates every expert scaled by
    /// its responsibility r_{g,e} = π_g · w_{g,e}.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        int G = _config.GroupCount, E = _config.ExpertsPerGroup;

        // Phase 1: probe energies (no weight update).
        var energies = new float[G][];
        for (int g = 0; g < G; g++)
        {
            energies[g] = new float[E];
            for (int e = 0; e < E; e++)
                energies[g][e] = _groups[g][e].InferEnergy(input, label, inferenceSteps);
        }

        // Within-group gates + soft-min group energies.
        var groupEnergy = new float[G];
        for (int g = 0; g < G; g++)
        {
            MixtureOfExperts.ComputeGates(
                energies[g], LastResponsibilities[g],
                _config.ExpertGatingTemperature, _config.MinGateWeight);

            float fg = 0f;
            for (int e = 0; e < E; e++) fg += LastResponsibilities[g][e] * energies[g][e];
            groupEnergy[g] = fg;
        }

        // Across-group gate.
        MixtureOfExperts.ComputeGates(
            groupEnergy, LastGroupGates, _config.GroupGatingTemperature, 0f);

        // Phase 2: responsibility-scaled updates  r_{g,e} = π_g · w_{g,e}.
        for (int g = 0; g < G; g++)
            for (int e = 0; e < E; e++)
            {
                float r = LastGroupGates[g] * LastResponsibilities[g][e];
                LastResponsibilities[g][e] = r;
                _groups[g][e].LearnGated(input, label, inferenceSteps, r);
            }
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Hierarchical winner-take-all: choose the group whose best expert attains
    /// the lowest winning free energy, then return that expert's predicted class.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        float bestEnergy = float.MaxValue;
        int   bestClass  = 0;

        for (int g = 0; g < _config.GroupCount; g++)
            for (int e = 0; e < _config.ExpertsPerGroup; e++)
            {
                var (cls, energy) = _groups[g][e].PredictWithEnergy(input, inferenceSteps);
                if (energy < bestEnergy) { bestEnergy = energy; bestClass = cls; }
            }

        return bestClass;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public void FlushAll()
    {
        foreach (var group in _groups)
            foreach (var e in group) e.FlushStore();
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var group in _groups)
            foreach (var e in group) e.Dispose();
    }
}
