namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// VotingTopology  —  lateral connection patterns between cortical columns.
//
// Controls which columns receive lateral signals from which neighbours.
// AllToAll is the most expressive; Ring and Local are more biologically
// plausible for spatially organised column arrays.
// ─────────────────────────────────────────────────────────────────────────────

public enum VotingTopology
{
    /// <summary>Every column broadcasts to every other column (all-to-all).</summary>
    AllToAll,

    /// <summary>Each column receives lateral input only from its two immediate
    /// ring-neighbours (wraps around at the ends).</summary>
    Ring,

    /// <summary>Each column receives lateral input from columns within a fixed
    /// radius (<see cref="LateralVoting.LocalRadius"/>).</summary>
    Local,
}
