namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// BehaviorSequence  —  a behaviour / sequence represented as an object in a
// one-dimensional reference frame.
//
// Thousand Brains claim (Hawkins 2021): the cortex represents behaviours the same
// way it represents physical objects — as features arranged in a reference frame.
// Here the frame is 1-D: the "location" is the position along the sequence and the
// "feature" is the state SDR observed there. ToObjectPoints emits exactly the
// (location, feature) pairs an ObjectFrame / CorticalColumn already consumes, so a
// behaviour is learned and recognised by the same machinery as a mug or a cup.
//
// PredictNext gives sequence prediction within the frame; MatchScore recognises a
// stored behaviour from a (possibly partial) observed trajectory.
//
// Reference: Hawkins (2021) ch. 7 (behaviours as objects in reference frames);
// Hawkins et al. (2019) (location-conditioned feature prediction).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class BehaviorSequence
{
    private readonly List<Sdr> _steps = [];

    /// <summary>Identifier of this behaviour.</summary>
    public int BehaviorId { get; }

    /// <summary>The ordered state SDRs that make up the behaviour.</summary>
    public IReadOnlyList<Sdr> Steps => _steps;

    /// <summary>Number of steps in the sequence.</summary>
    public int Length => _steps.Count;

    public BehaviorSequence(int behaviorId) => BehaviorId = behaviorId;

    /// <summary>Appends a state to the end of the behaviour. Returns this instance so
    /// calls can be chained.</summary>
    public BehaviorSequence Append(Sdr state)
    {
        ArgumentNullException.ThrowIfNull(state);
        _steps.Add(state);
        return this;
    }

    /// <summary>The state at <paramref name="step"/>, or null if out of range.</summary>
    public Sdr? At(int step) => step >= 0 && step < _steps.Count ? _steps[step] : null;

    /// <summary>The predicted next state after <paramref name="currentStep"/>, or null
    /// at (or past) the end of the sequence.</summary>
    public Sdr? PredictNext(int currentStep) => At(currentStep + 1);

    /// <summary>Emits the behaviour as (location, feature) pairs in its 1-D reference
    /// frame: location = <paramref name="positionEncoder"/>.Encode(stepIndex),
    /// feature = the state at that step. Ready for ObjectMemory / CorticalColumn.</summary>
    public IReadOnlyList<(Sdr Location, Sdr Feature)> ToObjectPoints(ScalarEncoder positionEncoder)
    {
        ArgumentNullException.ThrowIfNull(positionEncoder);
        var pts = new List<(Sdr, Sdr)>(_steps.Count);
        for (int i = 0; i < _steps.Count; i++)
            pts.Add((positionEncoder.Encode(i), _steps[i]));
        return pts;
    }

    /// <summary>Fraction of <paramref name="observed"/> states that match the stored
    /// state at the same position (overlap ≥ <paramref name="minOverlap"/>), in
    /// <c>[0,1]</c>. 1 = the observed trajectory is a prefix of this behaviour;
    /// higher scores pick the recognised behaviour out of a set.</summary>
    public double MatchScore(IReadOnlyList<Sdr> observed, int minOverlap)
    {
        ArgumentNullException.ThrowIfNull(observed);
        if (observed.Count == 0) return 0.0;

        int matches = 0;
        int n = Math.Min(observed.Count, _steps.Count);
        for (int i = 0; i < n; i++)
        {
            var (a, b) = (observed[i], _steps[i]);
            if (a.Size == b.Size && a.Matches(b, minOverlap)) matches++;
        }
        return (double)matches / observed.Count;
    }
}
