using System.ComponentModel;
using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml.Media;
using Windows.UI;
using PredictiveCoding.OmniMind.Engine;

namespace PredictiveCoding.OmniMind;

/// <summary>
/// View-model for one provider tile in the live fan-out dashboard. Colour and
/// detail text reflect the probe's lifecycle (queued → probing → result).
/// </summary>
public sealed class ProbeTile : INotifyPropertyChanged
{
    public string Name { get; }
    public string Category { get; }

    private string _detail = "queued";
    public string Detail { get => _detail; private set => Set(ref _detail, value); }

    private Brush _accent = Make(0x3A, 0x40, 0x4F);
    public Brush Accent { get => _accent; private set => Set(ref _accent, value); }

    public ProbeTile(string name, string category)
    {
        Name = name;
        Category = category;
    }

    public void SetRunning()
    {
        Detail = "probing…";
        Accent = Make(0xF5, 0xA6, 0x23);   // amber
    }

    public void SetResult(ProbeResult r, long ms)
    {
        Accent = r.State switch
        {
            ProbeState.Ok      => Make(0x34, 0xC7, 0x59),   // green
            ProbeState.Empty   => Make(0x5A, 0x7A, 0x8A),   // slate
            ProbeState.Skipped => Make(0x8E, 0x8E, 0x93),   // grey
            ProbeState.Failed  => Make(0xFF, 0x45, 0x3A),   // red
            _                  => Make(0x3A, 0x40, 0x4F),
        };

        string head = r.State switch
        {
            ProbeState.Ok      => $"{r.ItemCount} items",
            ProbeState.Empty   => "no data",
            ProbeState.Skipped => "needs key",
            ProbeState.Failed  => "failed",
            _                  => "",
        };
        Detail = $"{head} · {r.Detail} · {ms} ms";
    }

    private static SolidColorBrush Make(byte r, byte g, byte b) => new(Color.FromArgb(255, r, g, b));

    public event PropertyChangedEventHandler? PropertyChanged;

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
