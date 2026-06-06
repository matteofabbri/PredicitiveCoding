namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// SensorimotorAgent  —  closes the sense → infer → act loop.
//
// Wraps a ThousandBrainsSystem and an IInferencePolicy to drive an IEnvironment
// through recognition episodes.
//
// Recognition episode (RunEpisode)
//   1. Reset the system (clear temporal context + hypothesis sets).
//   2. Observe → update system belief.
//   3. If a unique hypothesis remains → done.
//   4. Policy selects a motor command → environment moves.
//   5. Repeat until convergence or step budget exhausted.
//
// Learning (Learn / LearnAll)
//   Samples a trajectory over the target object, collects (feature, location)
//   pairs, and feeds them to the system as a training episode.
//
// Reference: Leadholm et al. (2025) P9 §3-4.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Agent that drives a <see cref="ThousandBrainsSystem"/> through
/// sensorimotor recognition and learning episodes.</summary>
public sealed class SensorimotorAgent
{
    private readonly ThousandBrainsSystem _system;
    private readonly IInferencePolicy _policy;

    /// <summary>The underlying <see cref="ThousandBrainsSystem"/> the agent operates on.</summary>
    public ThousandBrainsSystem System => _system;

    /// <summary>The inference policy used to choose motor commands.</summary>
    public IInferencePolicy Policy => _policy;

    /// <param name="system">Pre-configured (and optionally pre-trained) brain system.</param>
    /// <param name="policy">Action-selection strategy to use during inference episodes.</param>
    public SensorimotorAgent(ThousandBrainsSystem system, IInferencePolicy policy)
    {
        ArgumentNullException.ThrowIfNull(system);
        ArgumentNullException.ThrowIfNull(policy);
        _system = system;
        _policy = policy;
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// <summary>Runs a full recognition episode on the current object in
    /// <paramref name="env"/>. The environment must already have had
    /// <see cref="IEnvironment.PresentObject"/> called before this method.
    /// Returns an <see cref="AgentResult"/> describing the outcome.</summary>
    /// <param name="env">The environment to interact with.</param>
    /// <param name="maxSteps">Maximum sense-act iterations. Default 20.</param>
    public AgentResult RunEpisode(IEnvironment env, int maxSteps = 20)
    {
        ArgumentNullException.ThrowIfNull(env);
        if (maxSteps < 1)
            throw new ArgumentOutOfRangeException(nameof(maxSteps), "Must be at least 1.");

        _system.Reset();
        var history = new List<IReadOnlyList<int>>(maxSteps);

        for (int step = 0; step < maxSteps; step++)
        {
            var (feature, location) = env.Observe();
            _system.ProcessObservation(feature, location);

            IReadOnlyList<int> hypotheses = [.. _system.CurrentHypotheses];
            history.Add(hypotheses);

            // Converged: exactly one hypothesis remaining.
            if (hypotheses.Count == 1)
                return new AgentResult(hypotheses[0], step + 1, true, history);

            // No candidates at all — cannot narrow further.
            if (hypotheses.Count == 0) break;

            int action = _policy.SelectAction(hypotheses, env.AvailableActions);
            env.MoveTo(action);
        }

        IReadOnlyCollection<int> final = _system.CurrentHypotheses;
        return new AgentResult(
            final.Count == 1 ? final.First() : null,
            history.Count, false, history);
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>Samples <paramref name="observationSteps"/> observations from
    /// <paramref name="objectId"/> in <paramref name="env"/> by moving randomly
    /// between steps, then registers the collected trajectory with the system.</summary>
    /// <param name="env">The environment containing the target object.</param>
    /// <param name="objectId">Id of the object to learn.</param>
    /// <param name="observationSteps">How many points to sample. Default 6.</param>
    public void Learn(IEnvironment env, int objectId, int observationSteps = 6)
    {
        ArgumentNullException.ThrowIfNull(env);
        if (observationSteps < 1)
            throw new ArgumentOutOfRangeException(nameof(observationSteps), "Must be at least 1.");

        env.PresentObject(objectId);
        env.Reset();

        var obs = new List<(Sdr Feature, Sdr Location)>(observationSteps);
        for (int s = 0; s < observationSteps; s++)
        {
            obs.Add(env.Observe());
            if (s < observationSteps - 1)
                // Use last available action (by convention the most exploratory one).
                env.MoveTo(env.AvailableActions - 1);
        }

        _system.TeachObject(obs);
    }

    /// <summary>Calls <see cref="Learn"/> for every object in <paramref name="objectIds"/>.</summary>
    /// <param name="env">The environment.</param>
    /// <param name="objectIds">Object ids to teach the system, in order.</param>
    /// <param name="observationStepsPerObject">Points sampled per object. Default 6.</param>
    public void LearnAll(
        IEnvironment env,
        IReadOnlyList<int> objectIds,
        int observationStepsPerObject = 6)
    {
        ArgumentNullException.ThrowIfNull(env);
        ArgumentNullException.ThrowIfNull(objectIds);
        foreach (int id in objectIds)
            Learn(env, id, observationStepsPerObject);
    }

    /// <summary>Resets the underlying system's temporal context and hypothesis sets.
    /// Does not erase learned object models.</summary>
    public void Reset() => _system.Reset();
}
