using System.Collections.Concurrent;
using THEQULT.Indexes;

namespace PredictiveCoding.Benchmark;

/// <summary>An in-memory <see cref="IByteStore"/> so a benchmark's gallery index needs no disk.</summary>
public sealed class InMemoryByteStore : IByteStore
{
    private readonly ConcurrentDictionary<long, byte[]> _data = new();

    public bool TryRead(long id, out byte[] data)
    {
        if (_data.TryGetValue(id, out byte[]? stored))
        {
            data = stored;
            return true;
        }
        data = [];
        return false;
    }

    public void Write(long id, byte[] data) => _data[id] = data;
    public void Remove(long id) => _data.TryRemove(id, out _);
    public void Checkpoint() { }
    public void Dispose() => _data.Clear();
}
