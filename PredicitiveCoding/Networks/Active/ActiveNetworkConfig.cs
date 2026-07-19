using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Active;

// ─────────────────────────────────────────────────────────────────────────────
// ActiveNetworkConfig  —  configuration for an Active Predictive-Coding agent
//
// The agent couples a generative world model (a PCN that predicts the next
// observation from the current observation and a chosen action) with a policy
// that selects actions to minimise expected future prediction error.  This
// closes the perception–action loop in the free-energy framework: actions are
// chosen so the agent's preferred (goal) observations become self-fulfilling
// predictions (Friston et al. 2017; Rao et al. 2024).
//
//   World model topology:
//     LayerSizes[0]  = ObservationDim + ActionCount   (observation ⊕ one-hot action)
//     LayerSizes[^1] = ObservationDim                 (predicted next observation)
//
// References:
//   Friston, FitzGerald, Rigoli, Schwartenbeck, Pezzulo (2017)
//     "Active Inference: A Process Theory", Neural Computation.
//   Rao, Gklezakos, Sathish (2024) "Active Predictive Coding", Neural Computation.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record ActiveNetworkConfig
{
    /// <summary>
    /// World-model PC topology.  LayerSizes[0] must equal
    /// ObservationDim + ActionCount and LayerSizes[^1] must equal ObservationDim.
    /// </summary>
    public NetworkConfig WorldModelConfig { get; init; } = new();

    /// <summary>Dimensionality of an observation vector.</summary>
    public int ObservationDim { get; init; }

    /// <summary>Number of discrete actions available to the agent.</summary>
    public int ActionCount { get; init; }

    /// <summary>
    /// Pragmatic weight λ_goal: how strongly the agent prefers reaching the goal
    /// observation.  Multiplies the goal-distance term in expected free energy.
    /// </summary>
    public float GoalPrecision { get; init; } = 1f;

    /// <summary>
    /// Epistemic weight λ_explore: bonus for actions whose predicted outcome is
    /// far from the current observation (novelty-seeking / exploration).  Set to
    /// 0 for a purely goal-directed (exploitative) agent.
    /// </summary>
    public float ExplorationWeight { get; init; } = 0f;

    public string BasePath { get; init; } = "./brain/active";

    public void Validate()
    {
        if (ObservationDim < 1)
            throw new InvalidOperationException("ObservationDim must be >= 1.");
        if (ActionCount < 2)
            throw new InvalidOperationException("ActionCount must be >= 2.");

        int expectedIn  = ObservationDim + ActionCount;
        int expectedOut = ObservationDim;
        if (WorldModelConfig.LayerSizes[0] != expectedIn)
            throw new InvalidOperationException(
                $"WorldModelConfig.LayerSizes[0] ({WorldModelConfig.LayerSizes[0]}) must equal " +
                $"ObservationDim + ActionCount ({expectedIn}).");
        if (WorldModelConfig.LayerSizes[^1] != expectedOut)
            throw new InvalidOperationException(
                $"WorldModelConfig.LayerSizes[^1] ({WorldModelConfig.LayerSizes[^1]}) must equal " +
                $"ObservationDim ({expectedOut}).");
        if (GoalPrecision < 0f)
            throw new InvalidOperationException("GoalPrecision must be >= 0.");
        if (ExplorationWeight < 0f)
            throw new InvalidOperationException("ExplorationWeight must be >= 0.");
    }
}
