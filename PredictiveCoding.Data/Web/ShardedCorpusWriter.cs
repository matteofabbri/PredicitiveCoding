using System.Text;

namespace PredictiveCoding.Data.Web;

// ─────────────────────────────────────────────────────────────────────────────
// ShardedCorpusWriter  —  the corpus that the crawler sediments (ROADMAP-Scale C1).
//
// An append-only folder of UTF-8 text shards (shard-00000.txt, shard-00001.txt,
// …): every Append writes one document as its lines, shards rotate at a size
// threshold, and reopening the same folder continues after the last shard —
// the accumulation only ever grows. The format is deliberately nothing but
// text lines, so the LLM side reads it back with CorpusSource.FromFolder
// (and applies its hygiene decorators at read time) with no coupling between
// the projects.
//
// Thread-safe: the crawler may append from concurrent fetches.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ShardedCorpusWriter : IDisposable
{
    private const string Prefix = "shard-";

    private readonly string _directory;
    private readonly long _maxShardBytes;
    private readonly object _gate = new();

    private StreamWriter? _writer;
    private int _shardIndex;
    private long _currentShardBytes;

    /// <summary>Documents appended by THIS writer instance.</summary>
    public long DocumentsWritten { get; private set; }

    /// <summary>UTF-8 bytes appended by THIS writer instance.</summary>
    public long BytesWritten { get; private set; }

    /// <summary>The shard folder (created if missing).</summary>
    public string Directory => _directory;

    public ShardedCorpusWriter(string directory, long maxShardBytes = 64L * 1024 * 1024)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (maxShardBytes < 1024) throw new ArgumentOutOfRangeException(nameof(maxShardBytes), "Shards under 1 KB make no sense.");
        _directory = directory;
        _maxShardBytes = maxShardBytes;
        System.IO.Directory.CreateDirectory(directory);

        // Resume after the last existing shard (append-only: never rewrite).
        var existing = System.IO.Directory.GetFiles(directory, Prefix + "*.txt");
        if (existing.Length > 0)
        {
            Array.Sort(existing, StringComparer.Ordinal);
            string last = existing[^1];
            _shardIndex = int.Parse(Path.GetFileNameWithoutExtension(last).AsSpan(Prefix.Length));
            _currentShardBytes = new FileInfo(last).Length;
            if (_currentShardBytes >= _maxShardBytes) { _shardIndex++; _currentShardBytes = 0; }
        }
    }

    /// <summary>
    /// Appends one document: each line of <paramref name="lines"/> becomes a
    /// corpus line (empty ones are skipped). Returns the bytes written.
    /// </summary>
    public long Append(IEnumerable<string> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);
        lock (_gate)
        {
            long before = BytesWritten;
            bool any = false;
            foreach (string raw in lines)
            {
                if (string.IsNullOrWhiteSpace(raw)) continue;
                string line = raw.ReplaceLineEndings(" ").Trim();   // one corpus line per block
                if (line.Length == 0) continue;

                var w = CurrentWriter();
                w.WriteLine(line);
                long bytes = Encoding.UTF8.GetByteCount(line) + Environment.NewLine.Length;
                _currentShardBytes += bytes;
                BytesWritten += bytes;
                any = true;
                if (_currentShardBytes >= _maxShardBytes) Rotate();
            }
            if (any) DocumentsWritten++;
            return BytesWritten - before;
        }
    }

    /// <summary>Appends a single-block document.</summary>
    public long Append(string text) => Append([text]);

    /// <summary>Flushes the current shard to disk (Append itself is buffered).</summary>
    public void Flush()
    {
        lock (_gate) _writer?.Flush();
    }

    private StreamWriter CurrentWriter()
    {
        if (_writer is null)
        {
            string path = Path.Combine(_directory, $"{Prefix}{_shardIndex:D5}.txt");
            _writer = new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read),
                                       new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
        return _writer;
    }

    private void Rotate()
    {
        _writer?.Flush();
        _writer?.Dispose();
        _writer = null;
        _shardIndex++;
        _currentShardBytes = 0;
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _writer?.Flush();
            _writer?.Dispose();
            _writer = null;
        }
    }
}
