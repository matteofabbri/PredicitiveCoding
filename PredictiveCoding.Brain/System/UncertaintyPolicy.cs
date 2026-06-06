namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// UncertaintyPolicy  —  uncertainty-driven action selection.
//
// When many object hypotheses remain (high uncertainty) the policy strongly
// prefers the exploration action so the agent moves to an informative new
// sensor position.  As hypotheses narrow toward one the policy shifts toward
// the exploitation action (fine-grained local probing).
//
// Formally the probability of selecting the exploration action is:
//
//   P(explore) = clamp( (|hypotheses| - 1) / uncertaintyScale, 0, 1 )
//
// so P(explore) = 0 when |hypotheses| ≤ 1 and P(explore) → 1 as |hypotheses|
// grows past uncertaintyScale.
//
// Reference: Leadholm et al. (2025) P9 §4; active-inference literature.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Inference policy that prefers exploration when belief is uncertain.</summary>
public sealed class UncertaintyPolicy : IInferencePolicy
{
    private readonly Random _rng;

    /// <summary>Motor command to use when the agent needs to explore broadly.
    /// Defaults to <see cref="PointCloudEnvironment.MoveRandom"/> (2).</summary>
    public int ExplorationAction { get; }

    /// <summary>Motor command to use when the agent believes it has converged.
    /// Defaults to <see cref="PointCloudEnvironment.MoveNext"/> (0).</summary>
    public int ExploitationAction { get; }

    /// <summary>Number of hypotheses at which P(explore) reaches 1.0.
    /// A smaller value makes the policy switch to exploration sooner.</summary>
    public float UncertaintyScale { get; }

    /// <param name="explorationAction">Motor command for broad exploration. Default 2 (random).</param>
    /// <param name="exploitationAction">Motor command for fine-grained probing. Default 0 (next).</param>
    /// <param name="uncertaintyScale">Hypothesis count at which P(explore) = 1.0. Default 5.</param>
    /// <param name="seed">RNG seed. Default 42.</param>
    public UncertaintyPolicy(
        int explorationAction = PointCloudEnvironment.MoveRandom,
        int exploitationAction = PointCloudEnvironment.MoveNext,
        float uncertaintyScale = 5f,
        int seed = 42)
    {
        if (uncertaintyScale <= 0f)
            throw new ArgumentOutOfRangeException(nameof(uncertaintyScale), "Must be positive.");
        ExplorationAction = explorationAction;
        ExploitationAction = exploitationAction;
        UncertaintyScale = uncertaintyScale;
        _rng = new Random(seed);
    }

    /// <inheritdoc/>
    public int SelectAction(IReadOnlyCollection<int> currentHypotheses, int availableActions)
    {
        ArgumentNullException.ThrowIfNull(currentHypotheses);
        if (availableActions <= 0)
            throw new ArgumentOutOfRangeException(nameof(availableActions), "Must be at least 1.");

        // P(explore) grows linearly with the number of remaining hypotheses.
        double pExplore = Math.Clamp(
            (currentHypotheses.Count - 1.0) / UncertaintyScale, 0.0, 1.0);

        int chosen = _rng.NextDouble() < pExplore ? ExplorationAction : ExploitationAction;
        return chosen % availableActions;
    }
}
