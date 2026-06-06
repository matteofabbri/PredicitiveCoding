using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using PredictiveCoding.CortexApp.Examples;

namespace PredictiveCoding.CortexApp.Pages;

public sealed partial class ActiveInferencePage : Page
{
    public ActiveInferencePage() => InitializeComponent();

    private async void RunButton_Click(object sender, RoutedEventArgs e)
    {
        int explore = Math.Max(10, (int)ExploreBox.Value);
        int steps   = Math.Max(1, (int)StepsBox.Value);

        RunButton.IsEnabled = false;
        Busy.IsActive       = true;
        ResultText.Text     = "Exploring and planning…";

        try
        {
            var r = await Task.Run(() => ActiveInferenceExample.Run(explore, steps));

            var path = string.Join("  →  ", r.Trajectory.Select(p => $"({p.X:F2},{p.Y:F2})"));
            ResultText.Text =
                $"Start (0.10, 0.10)  →  Goal (0.90, 0.90)\n\n" +
                $"Goal reached     : {(r.Reached ? "✓ yes" : "✗ no")}\n" +
                $"Final distance   : {r.FinalDistance:F3}\n" +
                $"Steps taken      : {r.Trajectory.Count - 1}\n" +
                $"Exploration steps: {r.ExploreSteps}\n\n" +
                $"Trajectory:\n{path}";
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
