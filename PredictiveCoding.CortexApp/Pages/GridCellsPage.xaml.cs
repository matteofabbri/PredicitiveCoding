using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using PredictiveCoding.CortexApp.Examples;

namespace PredictiveCoding.CortexApp.Pages;

public sealed partial class GridCellsPage : Page
{
    public GridCellsPage() => InitializeComponent();

    private async void RunButton_Click(object sender, RoutedEventArgs e)
    {
        RunButton.IsEnabled = false;
        Busy.IsActive       = true;
        try
        {
            var result = await Task.Run(() => GridCellsExample.Run(resolution: 96));

            Map0.Source = ImageRender.ToBitmap(result.Maps[0]);
            Map1.Source = ImageRender.ToBitmap(result.Maps[1]);
            Map2.Source = ImageRender.ToBitmap(result.Maps[2]);
            Label0.Text = $"cell {result.CellIndices[0]}";
            Label1.Text = $"cell {result.CellIndices[1]}";
            Label2.Text = $"cell {result.CellIndices[2]}";
        }
        finally
        {
            Busy.IsActive       = false;
            RunButton.IsEnabled = true;
        }
    }
}
