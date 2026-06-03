using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media.Imaging;
using PredictiveCoding.Core;

namespace PredictiveCoding.CortexApp;

/// <summary>
/// Main window: drives <see cref="CortexDemo"/> on a background thread and renders
/// the input image, the cortex's top-down reconstruction, and the metrics.
/// </summary>
public sealed partial class MainWindow : Window
{
    private readonly DispatcherQueue _dispatcher;

    public MainWindow()
    {
        InitializeComponent();
        _dispatcher = DispatcherQueue.GetForCurrentThread();
    }

    private async void RunButton_Click(object sender, RoutedEventArgs e)
    {
        int epochs = (int)EpochsBox.Value;
        int steps  = (int)StepsBox.Value;
        if (epochs < 1) epochs = 1;
        if (steps  < 1) steps  = 1;

        RunButton.IsEnabled = false;
        Busy.IsActive       = true;
        MetricsText.Text     = string.Empty;

        try
        {
            var result = await Task.Run(() =>
                CortexDemo.Run(epochs, steps, msg => _dispatcher.TryEnqueue(() => StatusText.Text = msg)));

            InputImage.Source = ToBitmap(result.Input);
            ReconImage.Source = ToBitmap(result.Reconstruction);

            string trueName = CortexDemo.ClassNames[result.TrueClass];
            string predName = CortexDemo.ClassNames[result.PredClass];
            bool   correct  = result.TrueClass == result.PredClass;

            var areas = string.Join("\n", result.AreaEnergies
                .Select((a, i) => $"  area {i} (V{i + 2}):  F = {a:F4}"));

            MetricsText.Text =
                $"True class : {trueName}\n" +
                $"Predicted  : {predName}   {(correct ? "✓ correct" : "✗ wrong")}\n\n" +
                $"Reconstruction error : {result.ReconError:F4}\n\n" +
                $"Free energy by area:\n" +
                $"  V1 column:  F = {result.V1Energy:F4}\n" +
                $"{areas}";

            StatusText.Text = "Done.";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Error.";
            MetricsText.Text = ex.ToString();
        }
        finally
        {
            Busy.IsActive       = false;
            RunButton.IsEnabled = true;
        }
    }

    // ── Rendering ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Renders a [C,H,W] tensor to a WriteableBitmap, min-max normalised to [0,1]
    /// for display.  Grayscale (C==1) is broadcast to RGB.
    /// </summary>
    private static WriteableBitmap ToBitmap(DenseTensor<float> img)
    {
        int c = img.Dimensions[0];
        int h = img.Dimensions[1];
        int w = img.Dimensions[2];

        float min = float.MaxValue, max = float.MinValue;
        foreach (float v in img.Buffer)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float range = max - min;
        float scale = range > 1e-6f ? 1f / range : 0f;

        var wb   = new WriteableBitmap(w, h);
        var bgra = new byte[w * h * 4];
        int plane = h * w;

        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int pix = y * w + x;
            float r = Norm(img.Buffer[pix], min, scale);
            float g = c > 1 ? Norm(img.Buffer[plane + pix], min, scale) : r;
            float b = c > 2 ? Norm(img.Buffer[2 * plane + pix], min, scale) : r;

            int o = pix * 4;
            bgra[o + 0] = (byte)(b * 255f);
            bgra[o + 1] = (byte)(g * 255f);
            bgra[o + 2] = (byte)(r * 255f);
            bgra[o + 3] = 255;
        }

        using (var stream = wb.PixelBuffer.AsStream())
            stream.Write(bgra, 0, bgra.Length);
        wb.Invalidate();
        return wb;
    }

    private static float Norm(float v, float min, float scale)
    {
        float n = (v - min) * scale;
        return n < 0f ? 0f : n > 1f ? 1f : n;
    }
}
