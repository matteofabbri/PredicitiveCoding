using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using PredictiveCoding.CortexApp.Examples;

namespace PredictiveCoding.CortexApp.Pages;

public sealed partial class SpikingPage : Page
{
    public SpikingPage() => InitializeComponent();

    private async void RunButton_Click(object sender, RoutedEventArgs e)
    {
        int epochs = Math.Max(1, (int)EpochsBox.Value);
        int ticks  = Math.Max(1, (int)TicksBox.Value);

        RunButton.IsEnabled = false;
        Busy.IsActive       = true;
        ResultText.Text     = "Training…";

        try
        {
            var r = await Task.Run(() => SpikingExample.Run(epochs, ticks));
            ResultText.Text =
                $"Test accuracy       : {r.Accuracy * 100f:F1}%\n" +
                $"Avg. free energy    : {r.AverageEnergy:F4}\n\n" +
                $"Epochs              : {r.TrainEpochs}\n" +
                $"Spike ticks / pass  : {r.Ticks}\n" +
                $"Classes             : 3 synthetic prototypes (16-D), local Hebbian learning.";
        }
        catch (Exception ex)
        {
            ResultText.Text = "Error: " + ex.Message;
        }
        finally
        {
            Busy.IsActive       = false;
            RunButton.IsEnabled = true;
        }
    }
}
