using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using PredictiveCoding.CortexApp.Pages;

namespace PredictiveCoding.CortexApp;

/// <summary>
/// Shell window: a NavigationView whose entries each open a separate example page.
/// </summary>
public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Nav.SelectedItem = Nav.MenuItems[0];   // triggers navigation to the first page
    }

    private void Nav_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (args.SelectedItem is not NavigationViewItem { Tag: string tag }) return;

        var page = tag switch
        {
            "visual"   => typeof(VisualCortexPage),
            "hopfield" => typeof(HopfieldPage),
            "grid"     => typeof(GridCellsPage),
            "spiking"  => typeof(SpikingPage),
            "active"   => typeof(ActiveInferencePage),
            _          => typeof(VisualCortexPage),
        };
        ContentFrame.Navigate(page);
    }
}
