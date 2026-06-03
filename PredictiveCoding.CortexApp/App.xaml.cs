using Microsoft.UI.Xaml;

namespace PredictiveCoding.CortexApp;

/// <summary>
/// Application entry point for the Visual Cortex demo (unpackaged WinUI 3).
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
