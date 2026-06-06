using Microsoft.UI.Xaml;

namespace PredictiveCoding.OmniMind;

/// <summary>
/// Application entry point for the OmniMind demo (unpackaged WinUI 3).
/// </summary>
public partial class App : Application
{
    private Window? _window;

    public App()
    {
        InitializeComponent();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new MainWindow();
        _window.Activate();
    }
}
