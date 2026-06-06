using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using PredictiveCoding.CortexApp.Examples;

namespace PredictiveCoding.CortexApp.Pages;

public sealed partial class HopfieldPage : Page
{
    public HopfieldPage() => InitializeComponent();

    private async void RunButton_Click(object sender, RoutedEventArgs e)
    {
        int epochs = Math.Max(1, (int)EpochsBox.Value);
        int steps  = Math.Max(1, (int)StepsBox.Value);

        RunButton.IsEnabled = false;
        Busy.IsActive       = true;
        StatusText.Text     = "Training and recalling…";

        try
        {
            var result = await Task.Run(() => HopfieldExample.Run(epochs, steps));

            OriginalImage.Source  = ImageRender.ToBitmap(result.Original);
            CorruptedImage.Source = ImageRender.ToBitmap(result.Corrupted);
            RecalledImage.Source  = ImageRender.ToBitmap(result.Recalled);
            StatusText.Text       = $"Recalled vs original cosine similarity: {result.Similarity:F3}";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Error: " + ex.Message;
        }
        finally
        {
            Busy.IsActive       = false;
            RunButton.IsEnabled = true;
        }
    }
}
