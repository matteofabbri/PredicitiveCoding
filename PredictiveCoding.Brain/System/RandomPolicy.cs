namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// RandomPolicy  —  uniform random motor command selection.
//
// A baseline inference policy that ignores the current hypothesis set and
// picks a motor command uniformly at random.  Useful as a comparison
// against uncertainty-driven policies and for unit tests that need
// deterministic, seed-controlled action sequences.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Inference policy that picks motor commands uniformly at random.</summary>
public sealed class RandomPolicy : IInferencePolicy
{
    private readonly Random _rng;

    /// <param name="seed">RNG seed. Default 42.</param>
    public RandomPolicy(int seed = 42) => _rng = new Random(seed);

    /// <inheritdoc/>
    public int SelectAction(IReadOnlyCollection<int> currentHypotheses, int availableActions)
    {
        ArgumentNullException.ThrowIfNull(currentHypotheses);
        if (availableActions <= 0)
            throw new ArgumentOutOfRangeException(nameof(availableActions),
                "Must be at least 1.");
        return _rng.Next(availableActions);
    }
}
