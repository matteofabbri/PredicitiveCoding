namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// TemporoparietalJunction (TPJ)  —  ventral attention "circuit breaker"
//
// The right TPJ is the core of the stimulus-driven (ventral) attention network: when
// a behaviourally relevant, unexpected stimulus appears outside the current focus it
// interrupts ("breaks the circuit" of) ongoing top-down attention and triggers a
// reorienting toward the new stimulus. Here it compares bottom-up salience against
// the strength of current top-down focus and emits a reorienting signal only when
// salience wins — strong focus raises the bar a surprise must clear. A pure
// controller that gates `DefaultModeNetwork` / `PrefrontalCortex` attention.
//
// References: Corbetta & Shulman (2002) dorsal vs ventral attention;
//             Corbetta, Patel & Shulman (2008) TPJ as a circuit breaker.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TemporoparietalJunction
{
    /// <summary>How much current top-down focus raises the salience needed to reorient.</summary>
    public float ReorientThreshold { get; }

    public TemporoparietalJunction(float reorientThreshold = 1.5f)
    {
        if (reorientThreshold < 0f) throw new ArgumentOutOfRangeException(nameof(reorientThreshold));
        ReorientThreshold = reorientThreshold;
    }

    /// <summary>The reorienting drive: max(0, bottomUpSalience − threshold·topDownFocus).
    /// Zero while ongoing focus suppresses the distractor; positive when a salient
    /// surprise breaks through.</summary>
    public float ReorientSignal(float bottomUpSalience, float topDownFocus)
        => MathF.Max(0f, bottomUpSalience - ReorientThreshold * MathF.Max(0f, topDownFocus));

    /// <summary>True when the bottom-up stimulus should interrupt current focus.</summary>
    public bool ShouldReorient(float bottomUpSalience, float topDownFocus)
        => ReorientSignal(bottomUpSalience, topDownFocus) > 0f;
}
