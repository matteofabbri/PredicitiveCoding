namespace PredictiveCoding.Vision.Counting;

/// <summary>
/// Sliding-window <b>median</b> of the per-frame people count. A single dropped or
/// spurious detection changes the raw count by one for one frame; taking the median
/// over the last few frames keeps the displayed number steady without lagging.
/// </summary>
public sealed class CountSmoother
{
    private readonly int[] _window;
    private int _filled;
    private int _head;

    public CountSmoother(int windowSize = 5)
    {
        if (windowSize < 1) windowSize = 1;
        _window = new int[windowSize];
    }

    /// <summary>Records a raw per-frame count and returns the smoothed (median) value.</summary>
    public int Add(int value)
    {
        _window[_head] = value;
        _head = (_head + 1) % _window.Length;
        if (_filled < _window.Length) _filled++;

        Span<int> copy = stackalloc int[_filled];
        for (int i = 0; i < _filled; i++) copy[i] = _window[i];
        copy.Sort();
        return copy[_filled / 2];
    }
}
