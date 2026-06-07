using Microsoft.Extensions.Options;
using PredictiveCoding.OmniMind.Web.Models;
using Sperone.Core.Collections;
using Sperone.Core.Indexes;

namespace PredictiveCoding.OmniMind.Web.Services;

/// <summary>
/// Run-history persistence backed by the <b>Sperone</b> embedded document store.
/// Each <see cref="RunRecord"/> is a JSON document in the <c>runs</c> collection,
/// indexed by <c>createdAt</c> (scalar, for recency range/sort) and <c>topic</c>
/// (full-text, for search). Durable: a checkpoint is flushed after every save and
/// recovered on startup. Registered as a singleton (one database per directory).
/// </summary>
public sealed class HistoryStore : IDisposable
{
    private const string RunsCollection = "runs";

    private readonly Database _db;
    private readonly object _writeGate = new();

    public HistoryStore(IOptions<OmniMindOptions> options)
    {
        string dir = options.Value.DataDirectory;
        Directory.CreateDirectory(dir);

        _db = new Database(new DatabaseOptions { Directory = dir, RecoverExisting = true });
        // Register the collection now so Query<T> works even before the first save.
        _db.GetCollection<RunRecord>(RunsCollection);
    }

    /// <summary>Persists a run and flushes a durable checkpoint.</summary>
    public void Save(RunRecord run)
    {
        ArgumentNullException.ThrowIfNull(run);
        lock (_writeGate)
        {
            Collection<RunRecord> col = _db.GetCollection<RunRecord>(RunsCollection);
            col.Upsert(ObjectId.FromUtf8(run.Id), run,
                IndexEntry.Scalar("createdAt", run.CreatedAtUnixMs),
                IndexEntry.Text("topic", run.Topic));
            _db.Checkpoint();
        }
    }

    /// <summary>The most recent runs, newest first.</summary>
    public IReadOnlyList<RunRecord> ListRecent(int max = 100)
    {
        CollectionQuery<RunRecord> q = _db.Query<RunRecord>(RunsCollection);
        return q.ScalarRange("createdAt", 0L, long.MaxValue)
                .Select(m => m.Value)
                .OrderByDescending(r => r.CreatedAtUnixMs)
                .Take(max)
                .ToList();
    }

    /// <summary>A single run by id, or <see langword="null"/> if it does not exist.</summary>
    public RunRecord? Get(string id)
    {
        if (string.IsNullOrWhiteSpace(id)) return null;
        return _db.Query<RunRecord>(RunsCollection).TryGet(ObjectId.FromUtf8(id), out RunRecord v) ? v : null;
    }

    /// <summary>Full-text search over topics, newest first. Empty query falls back to recents.</summary>
    public IReadOnlyList<RunRecord> Search(string? topic, int max = 100)
    {
        if (string.IsNullOrWhiteSpace(topic)) return ListRecent(max);
        return _db.Query<RunRecord>(RunsCollection)
                  .Text("topic", topic)
                  .Select(m => m.Value)
                  .OrderByDescending(r => r.CreatedAtUnixMs)
                  .Take(max)
                  .ToList();
    }

    /// <summary>Total number of stored runs.</summary>
    public long Count => _db.Query<RunRecord>(RunsCollection).Count;

    /// <summary>Flush a durable checkpoint (used on shutdown).</summary>
    public void Checkpoint()
    {
        lock (_writeGate) _db.Checkpoint();
    }

    public void Dispose() => _db.Dispose();
}
