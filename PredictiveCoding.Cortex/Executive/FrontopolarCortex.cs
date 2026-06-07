namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// FrontopolarCortex (BA10)  —  cognitive branching / metacognitive control
//
// The most anterior prefrontal pole holds a *pending* alternative goal in mind while
// the current one is being pursued, and switches to it when it becomes worth more —
// the "cognitive branching" that lets behaviour explore an option without losing the
// thread of the main task. Here it keeps a held alternative value and switches only
// when it exceeds the active goal by a margin (hysteresis prevents thrashing). A pure
// controller that sits above `PrefrontalCortex`.
//
// References: Koechlin & Hyafil (2007) frontopolar cognitive branching;
//             Boorman et al. (2009) BA10 tracking alternative options.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class FrontopolarCortex
{
    private float _pendingValue;
    private bool _hasPending;

    /// <summary>The active goal must be beaten by at least this margin to switch.</summary>
    public float SwitchMargin { get; }

    /// <summary>Whether an alternative goal is currently being held.</summary>
    public bool HasPending => _hasPending;

    /// <summary>The value of the held alternative (0 when none).</summary>
    public float PendingValue => _pendingValue;

    public FrontopolarCortex(float switchMargin = 0.1f)
    {
        if (switchMargin < 0f) throw new ArgumentOutOfRangeException(nameof(switchMargin));
        SwitchMargin = switchMargin;
    }

    /// <summary>Holds an alternative goal's value in the branch buffer.</summary>
    public void Hold(float value)
    {
        _pendingValue = value;
        _hasPending = true;
    }

    /// <summary>True when the held alternative beats the active goal by the margin
    /// (i.e. it is worth branching to it).</summary>
    public bool ShouldSwitch(float activeGoalValue)
        => _hasPending && _pendingValue > activeGoalValue + SwitchMargin;

    /// <summary>Clears the held alternative (e.g. after switching or abandoning it).</summary>
    public void Clear()
    {
        _hasPending = false;
        _pendingValue = 0f;
    }
}
