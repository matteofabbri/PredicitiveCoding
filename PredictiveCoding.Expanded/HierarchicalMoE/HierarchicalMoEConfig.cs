using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.HierarchicalMoE;

// ─────────────────────────────────────────────────────────────────────────────
// HierarchicalMoEConfig  —  configuration for a two-level mixture of PC experts
//
// Experts are arranged into groups.  Two energy-based gates compete at different
// levels (Jordan & Jacobs 1994):
//
//   within a group:   w_{g,e} = softmax_e(−β₂ · F_{g,e})
//   group energy:     F_g     = Σ_e w_{g,e} · F_{g,e}
//   across groups:    π_g     = softmax_g(−β₁ · F_g)
//   responsibility:   r_{g,e} = π_g · w_{g,e}
//
// Each expert then learns scaled by r_{g,e}.  The hierarchy lets groups
// specialise on coarse task structure while experts within a group specialise on
// fine structure — all driven purely by free-energy competition, with no learned
// router and no supervision of the routing itself.
//
// References:
//   Jordan & Jacobs (1994) "Hierarchical Mixtures of Experts and the EM
//     Algorithm", Neural Computation.
//   Jacobs et al. (1991); Shazeer et al. (2017) — modern (Switch) MoE analogue.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record HierarchicalMoEConfig
{
    /// <summary>Architecture shared by every expert PCN.</summary>
    public NetworkConfig ExpertConfig { get; init; } = new();

    /// <summary>Number of expert groups (top-level gate fans out over these).</summary>
    public int GroupCount { get; init; } = 2;

    /// <summary>Number of experts inside each group.</summary>
    public int ExpertsPerGroup { get; init; } = 2;

    /// <summary>β₁: inverse temperature of the across-group gate.</summary>
    public float GroupGatingTemperature { get; init; } = 1f;

    /// <summary>β₂: inverse temperature of the within-group gate.</summary>
    public float ExpertGatingTemperature { get; init; } = 1f;

    /// <summary>Floor applied to within-group gates to prevent expert collapse.</summary>
    public float MinGateWeight { get; init; } = 0f;

    public string BasePath { get; init; } = "./brain/hmoe";

    public int TotalExperts => GroupCount * ExpertsPerGroup;

    public void Validate()
    {
        if (GroupCount < 2)
            throw new InvalidOperationException("GroupCount must be >= 2.");
        if (ExpertsPerGroup < 2)
            throw new InvalidOperationException("ExpertsPerGroup must be >= 2.");
        if (GroupGatingTemperature <= 0f)
            throw new InvalidOperationException("GroupGatingTemperature must be > 0.");
        if (ExpertGatingTemperature <= 0f)
            throw new InvalidOperationException("ExpertGatingTemperature must be > 0.");
        if (MinGateWeight < 0f)
            throw new InvalidOperationException("MinGateWeight must be >= 0.");
        if (MinGateWeight >= 1f / ExpertsPerGroup)
            throw new InvalidOperationException(
                $"MinGateWeight ({MinGateWeight}) must be < 1/ExpertsPerGroup ({1f / ExpertsPerGroup:G3}).");
    }
}
