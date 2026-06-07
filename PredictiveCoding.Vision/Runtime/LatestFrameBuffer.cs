using System.Diagnostics.CodeAnalysis;

namespace PredictiveCoding.Vision.Runtime;

/// <summary>
/// A single-slot, thread-safe hand-off that always holds the <b>most recent</b> item.
/// Publishing while an item is still pending drops the old one (counted in
/// <see cref="Drops"/>, and handed to the optional <c>onDrop</c> callback so frames
/// can be disposed). This is how the capture thread is decoupled from a slower
/// inference thread: under load, stale frames are dropped instead of queued, so the
/// pipeline always works on fresh input and latency stays bounded.
/// </summary>
public sealed class LatestFrameBuffer<T> : IDisposable where T : class
{
    private readonly Action<T>? _onDrop;
    private readonly object _lock = new();
    private T? _pending;

    /// <summary>Total items published.</summary>
    public long Published { get; private set; }

    /// <summary>Items overwritten before they were consumed (dropped under load).</summary>
    public long Drops { get; private set; }

    /// <param name="onDrop">Invoked with each dropped item (e.g. to dispose a frame).</param>
    public LatestFrameBuffer(Action<T>? onDrop = null) => _onDrop = onDrop;

    /// <summary>Stores <paramref name="item"/> as the latest, dropping any unconsumed one.</summary>
    public void Publish(T item)
    {
        ArgumentNullException.ThrowIfNull(item);
        T? dropped = null;
        lock (_lock)
        {
            if (_pending is not null) { dropped = _pending; Drops++; }
            _pending = item;
            Published++;
        }
        if (dropped is not null) _onDrop?.Invoke(dropped);
    }

    /// <summary>Takes the pending item if there is one.</summary>
    public bool TryTake([MaybeNullWhen(false)] out T item)
    {
        lock (_lock)
        {
            if (_pending is null) { item = default; return false; }
            item = _pending;
            _pending = null;
            return true;
        }
    }

    /// <summary>Drops any pending item (handed to <c>onDrop</c>).</summary>
    public void Clear()
    {
        T? dropped;
        lock (_lock) { dropped = _pending; _pending = null; }
        if (dropped is not null) _onDrop?.Invoke(dropped);
    }

    /// <summary>Disposes the buffer, releasing any unconsumed item via <c>onDrop</c>.</summary>
    public void Dispose() => Clear();
}
