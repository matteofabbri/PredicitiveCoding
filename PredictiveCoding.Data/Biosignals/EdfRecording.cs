namespace PredictiveCoding.Data.Biosignals;

/// <summary>
/// A parsed EDF (European Data Format) recording: its start time and channels.
/// </summary>
public sealed class EdfRecording
{
    /// <summary>Recording start time, or null when the header date/time is unset.</summary>
    public DateTime? StartTime { get; }

    /// <summary>The recorded channels.</summary>
    public IReadOnlyList<EdfSignal> Signals { get; }

    public EdfRecording(DateTime? startTime, IReadOnlyList<EdfSignal> signals)
    {
        ArgumentNullException.ThrowIfNull(signals);
        StartTime = startTime;
        Signals = signals;
    }

    /// <summary>Returns the first channel whose label matches (case-insensitive), or null.</summary>
    public EdfSignal? Signal(string label)
        => Signals.FirstOrDefault(s => string.Equals(s.Label, label, StringComparison.OrdinalIgnoreCase));
}
