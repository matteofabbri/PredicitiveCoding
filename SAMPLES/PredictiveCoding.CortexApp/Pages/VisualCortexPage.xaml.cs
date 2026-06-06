using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace PredictiveCoding.CortexApp.Pages;

public sealed partial class VisualCortexPage : Page
{
    private readonly DispatcherQueue _dispatcher;

    public VisualCortexPage()
    {
        InitializeComponent();
        _dispatcher = DispatcherQueue.GetForCurrentThread();
    }

    private async void RunButton_Click(object sender, RoutedEventArgs e)
    {
        int epochs = Math.Max(1, (int)EpochsBox.Value);
        int steps  = Math.Max(1, (int)StepsBox.Value);

        RunButton.IsEnabled = false;
        Busy.IsActive       = true;
        MetricsText.Text    = string.Empty;

        try
        {
            var result = await Task.Run(() =>
                CortexDemo.Run(epochs, steps, msg => _dispatcher.TryEnqueue(() => StatusText.Text = msg)));

            InputImage.Source = ImageRender.ToBitmap(result.Input);
            ReconImage.Source = ImageRender.ToBitmap(result.Reconstruction);

            bool   correct = result.TrueClass == result.PredClass;
            string areas   = string.Join("\n", result.AreaEnergies.Select((a, i) => $"  area {i} (V{i + 2}):  F = {a:F4}"));

            MetricsText.Text =
                $"True class : {CortexDemo.ClassNames[result.TrueClass]}\n" +
                $"Predicted  : {CortexDemo.ClassNames[result.PredClass]}   {(correct ? "✓ correct" : "✗ wrong")}\n\n" +
                $"Reconstruction error : {result.ReconError:F4}\n\n" +
                $"Free energy by area:\n  V1 column:  F = {result.V1Energy:F4}\n{areas}";

            StatusText.Text = "Done.";
        }
        catch (Exception ex)
        {
            StatusText.Text  = "Error.";
            MetricsText.Text = ex.ToString();
        }
        finally
        {
            Busy.IsActive       = false;
            RunButton.IsEnabled = true;
        }
    }
}
