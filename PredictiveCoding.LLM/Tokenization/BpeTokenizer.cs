using System.Collections.Concurrent;
using System.Text;
using PredictiveCoding.LLM.Data;

namespace PredictiveCoding.LLM.Tokenization;

// ─────────────────────────────────────────────────────────────────────────────
// BpeTokenizer  —  trainable byte-level Byte-Pair Encoding.
//
// Starts from the 256 byte tokens (+ BOS/EOS/PAD/UNK) and learns a sequence of
// merges of the most frequent adjacent token pair. Two training modes:
//
//   • Train(corpus, numMerges)  —  the original whole-stream mode: merges
//     operate over the whole byte stream, one full corpus pass per merge.
//     Fine for toy vocabularies (hundreds of merges), kept bit-identical for
//     every existing model.
//
//   • TrainFromCorpus(source, targetVocab)  —  the ROADMAP-Scale B2 mode: the
//     corpus is split into word-level PIECES (GPT-2 style: an optional leading
//     space + a run of letters | digits | symbols, or a whitespace run), BPE
//     runs over the UNIQUE pieces weighted by frequency, and each merge updates
//     only the pieces that contain the pair (priority queue + incremental
//     counts). That is what makes a 16k-32k vocab on a multi-hundred-MB crawl
//     minutes instead of weeks — merges never cross piece boundaries, exactly
//     like every production tokenizer. Encoding then works piece-by-piece with
//     a cache, so long texts no longer pay the whole-stream greedy rescan.
//
// Decoding expands each token back to its byte sequence — lossless in both
// modes. The save format keeps the merge table only; a mode flag in the magic
// ("BPE1" whole-stream, "BPE2" word-split) reconstructs the right encoder.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class BpeTokenizer : ITokenizer
{
    private const int ByteCount     = 256;
    private const int FirstMerge    = ByteCount + 4;   // ids 256..259 are specials
    private const int PieceCacheCap = 100_000;

    private readonly Dictionary<(int, int), int> _rank   = new();   // pair → merge priority
    private readonly Dictionary<(int, int), int> _merged = new();   // pair → new token id
    private readonly Dictionary<int, byte[]>     _bytes  = new();   // token id → byte expansion
    private int  _vocabSize = FirstMerge;
    private bool _wordSplit;

    // Piece → encoded ids, so repeated words (most of real text) encode once.
    // Concurrent: the parallel trainer tokenizes from several workers. The size
    // is tracked separately — ConcurrentDictionary.Count takes every bucket
    // lock, and paying that per cache miss made encoding quadratic.
    private readonly ConcurrentDictionary<string, int[]> _pieceCache = new();
    private int _pieceCacheSize;

    public int VocabSize => _vocabSize;
    public int Bos => ByteCount + 0;
    public int Eos => ByteCount + 1;
    public int Pad => ByteCount + 2;
    public int Unk => ByteCount + 3;

    /// <summary>True when trained with <see cref="TrainFromCorpus"/>: merges live inside word pieces.</summary>
    public bool WordSplit => _wordSplit;

    public BpeTokenizer()
    {
        for (int b = 0; b < ByteCount; b++) _bytes[b] = [(byte)b];
    }

    // ── Training (whole-stream, toy scale) ───────────────────────────────────

    /// <summary>
    /// Learns up to <paramref name="numMerges"/> merges from <paramref name="corpus"/>,
    /// whole-stream (one full pass per merge — toy scale only).
    /// Stops early if no adjacent pair occurs more than once.
    /// </summary>
    public void Train(string corpus, int numMerges)
    {
        if (_wordSplit)
            throw new InvalidOperationException("This tokenizer was trained word-split (TrainFromCorpus); whole-stream Train would mix modes.");

        var ids = new List<int>(Encoding.UTF8.GetBytes(corpus).Select(b => (int)b));

        for (int m = 0; m < numMerges; m++)
        {
            var counts = new Dictionary<(int, int), int>();
            for (int i = 0; i < ids.Count - 1; i++)
            {
                var pair = (ids[i], ids[i + 1]);
                counts[pair] = counts.GetValueOrDefault(pair) + 1;
            }
            if (counts.Count == 0) break;

            var best = counts.MaxBy(kv => kv.Value);
            if (best.Value < 2) break;

            RecordMerge(best.Key, _rank.Count);
            ids = MergePair(ids, best.Key, _merged[best.Key]);
        }
    }

    // ── Training (word-split, real scale — ROADMAP-Scale B2) ─────────────────

    /// <summary>
    /// Learns a real vocabulary from a <see cref="CorpusSource"/>: one streaming
    /// pass counts unique word pieces, then BPE merges run over the piece table
    /// with incremental pair counts — cost per merge is proportional to the
    /// pieces actually touched, so 16k-32k vocabularies are minutes, not weeks.
    /// Merges never cross piece boundaries. Deterministic: ties break on the
    /// smallest pair, independent of hash order.
    /// </summary>
    /// <param name="corpus">Line source (files, folders, mixes — Pro Phase 4).</param>
    /// <param name="targetVocab">Final vocab size, base tokens included (260 + merges).</param>
    /// <param name="minPairFrequency">Stop when the best pair occurs fewer times than this.</param>
    /// <param name="onProgress">(mergesDone, mergesTarget), every 256 merges and at the end.</param>
    public void TrainFromCorpus(CorpusSource corpus, int targetVocab,
                                int minPairFrequency = 2, Action<int, int>? onProgress = null)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (targetVocab <= FirstMerge)
            throw new ArgumentOutOfRangeException(nameof(targetVocab),
                $"targetVocab ({targetVocab}) must exceed the {FirstMerge} base tokens.");
        if (minPairFrequency < 1)
            throw new ArgumentOutOfRangeException(nameof(minPairFrequency));
        if (_rank.Count > 0)
            throw new InvalidOperationException("TrainFromCorpus needs a fresh tokenizer — merges already learned.");

        _wordSplit = true;

        // Pass 1 — stream the corpus once, counting unique pieces (span lookup:
        // no substring allocation for the repeats, which are almost all of text).
        var pieceCounts = new Dictionary<string, long>();
        var byPiece     = pieceCounts.GetAlternateLookup<ReadOnlySpan<char>>();
        foreach (string line in corpus.Lines())
            foreach ((int s, int len) in Pieces(line))
            {
                var span = line.AsSpan(s, len);
                byPiece[span] = byPiece.TryGetValue(span, out long c) ? c + 1 : 1;
            }

        // Piece table: byte-id sequences + weights. Single-byte pieces can never
        // contribute a pair, so they are not carried.
        var seqs    = new List<int[]>(pieceCounts.Count);
        var weights = new List<long>(pieceCounts.Count);
        foreach (var (piece, count) in pieceCounts)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(piece);
            if (bytes.Length < 2) continue;
            var seq = new int[bytes.Length];
            for (int i = 0; i < bytes.Length; i++) seq[i] = bytes[i];
            seqs.Add(seq);
            weights.Add(count);
        }
        pieceCounts.Clear();

        // Pair statistics + inverted index pair → pieces containing it.
        var pairCount = new Dictionary<(int, int), long>();
        var pairWords = new Dictionary<(int, int), HashSet<int>>();

        void AddWordPairs(int w, HashSet<(int, int)>? grown)
        {
            int[] s = seqs[w]; long c = weights[w];
            for (int i = 0; i < s.Length - 1; i++)
            {
                var pair = (s[i], s[i + 1]);
                pairCount[pair] = pairCount.GetValueOrDefault(pair) + c;
                if (!pairWords.TryGetValue(pair, out var set)) pairWords[pair] = set = [];
                set.Add(w);
                grown?.Add(pair);
            }
        }

        void RemoveWordPairs(int w)
        {
            int[] s = seqs[w]; long c = weights[w];
            for (int i = 0; i < s.Length - 1; i++)
            {
                var pair = (s[i], s[i + 1]);
                long left = pairCount[pair] - c;
                if (left <= 0) pairCount.Remove(pair); else pairCount[pair] = left;
                if (pairWords.TryGetValue(pair, out var set))
                {
                    set.Remove(w);
                    if (set.Count == 0) pairWords.Remove(pair);
                }
            }
        }

        for (int w = 0; w < seqs.Count; w++) AddWordPairs(w, grown: null);

        // Max-heap with lazy invalidation: priorities are (−count, a, b), so the
        // tie-break is the smallest pair — fully deterministic. A popped entry
        // whose stored count no longer matches is stale: re-queued fresh (counts
        // only ever decrease, so stored priorities are upper bounds).
        var pq = new PriorityQueue<(int a, int b), (long negCount, int a, int b)>();
        foreach (var (pair, c) in pairCount)
            if (c >= minPairFrequency)
                pq.Enqueue(pair, (-c, pair.Item1, pair.Item2));

        int mergesTarget = targetVocab - FirstMerge;
        int done = 0;
        var grownPairs = new HashSet<(int, int)>();

        while (done < mergesTarget && pq.TryDequeue(out var pair, out var pri))
        {
            long current = pairCount.GetValueOrDefault(pair);
            if (current != -pri.negCount)
            {
                if (current >= minPairFrequency) pq.Enqueue(pair, (-current, pair.a, pair.b));
                continue;                                   // stale entry
            }
            if (current < minPairFrequency) break;          // true best is too rare → done

            RecordMerge(pair, done);
            int newId = _merged[pair];

            grownPairs.Clear();
            int[] touched = pairWords.TryGetValue(pair, out var host) ? [.. host] : [];
            foreach (int w in touched)
            {
                RemoveWordPairs(w);
                seqs[w] = MergePairArray(seqs[w], pair, newId);
                AddWordPairs(w, grownPairs);
            }
            foreach (var p in grownPairs)
            {
                long c = pairCount.GetValueOrDefault(p);
                if (c >= minPairFrequency) pq.Enqueue(p, (-c, p.Item1, p.Item2));
            }

            done++;
            if (done % 256 == 0) onProgress?.Invoke(done, mergesTarget);
        }
        onProgress?.Invoke(done, mergesTarget);
    }

    private void RecordMerge((int, int) pair, int rank)
    {
        int newId = _vocabSize++;
        _rank[pair]   = rank;
        _merged[pair] = newId;
        _bytes[newId] = [.. _bytes[pair.Item1], .. _bytes[pair.Item2]];
    }

    // ── Piece splitter (GPT-2 style, hand-rolled) ────────────────────────────
    //
    // A piece is: an optional single leading space + a maximal run of letters,
    // digits or symbols — or a whitespace run. The single space directly before
    // a visible character opens that piece (" word"), the rest of a whitespace
    // run stays its own piece. Surrogates (emoji) fall into the symbol class;
    // byte-level BPE underneath keeps everything lossless.

    private static int Category(char c) =>
        char.IsWhiteSpace(c) ? 0 : char.IsLetter(c) ? 1 : char.IsDigit(c) ? 2 : 3;

    private static IEnumerable<(int Start, int Length)> Pieces(string text)
    {
        int n = text.Length;
        int i = 0;
        while (i < n)
        {
            int start = i;

            if (char.IsWhiteSpace(text[i]))
            {
                int j = i;
                while (j < n && char.IsWhiteSpace(text[j])) j++;
                bool glueLast = j < n && text[j - 1] == ' ';   // " word" convention
                int wsEnd = glueLast ? j - 1 : j;
                if (wsEnd > start) yield return (start, wsEnd - start);
                i = wsEnd;
                if (!glueLast) continue;
                start = i;                                     // piece opens at the glued space
            }

            int k = text[i] == ' ' ? i + 1 : i;
            int cat = Category(text[k]);
            k++;
            while (k < n && Category(text[k]) == cat) k++;
            yield return (start, k - start);
            i = k;
        }
    }

    // ── Encoding / decoding ──────────────────────────────────────────────────

    public IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        text ??= string.Empty;

        List<int> ids;
        if (_wordSplit)
        {
            ids = new List<int>(text.Length / 3 + 2);
            var bySpan = _pieceCache.GetAlternateLookup<ReadOnlySpan<char>>();
            foreach ((int s, int len) in Pieces(text))
                ids.AddRange(EncodePiece(text.AsSpan(s, len), bySpan));
        }
        else
        {
            ids = ApplyMergesGreedy(new List<int>(Encoding.UTF8.GetBytes(text).Select(b => (int)b)));
        }

        if (!addBos && !addEos) return ids;
        var wrapped = new List<int>(ids.Count + 2);
        if (addBos) wrapped.Add(Bos);
        wrapped.AddRange(ids);
        if (addEos) wrapped.Add(Eos);
        return wrapped;
    }

    private int[] EncodePiece(ReadOnlySpan<char> pieceSpan,
                              ConcurrentDictionary<string, int[]>.AlternateLookup<ReadOnlySpan<char>> bySpan)
    {
        if (bySpan.TryGetValue(pieceSpan, out int[]? cached)) return cached;

        string piece = pieceSpan.ToString();
        byte[] utf8 = Encoding.UTF8.GetBytes(piece);
        var raw = new List<int>(utf8.Length);
        foreach (byte b in utf8) raw.Add(b);
        int[] arr = [.. ApplyMergesGreedy(raw)];

        if (Volatile.Read(ref _pieceCacheSize) >= PieceCacheCap)
        {
            _pieceCache.Clear();
            Volatile.Write(ref _pieceCacheSize, 0);   // advisory count — races are benign
        }
        if (_pieceCache.TryAdd(piece, arr)) Interlocked.Increment(ref _pieceCacheSize);
        return arr;
    }

    // Greedily merge the highest-priority (lowest-rank) adjacent pair until none
    // apply — the historical whole-stream algorithm, shared by both modes (the
    // word-split mode just runs it per piece, where sequences are a few bytes).
    private List<int> ApplyMergesGreedy(List<int> ids)
    {
        while (ids.Count >= 2)
        {
            int bestRank = int.MaxValue;
            (int, int) bestPair = default;
            for (int i = 0; i < ids.Count - 1; i++)
            {
                var pair = (ids[i], ids[i + 1]);
                if (_rank.TryGetValue(pair, out int r) && r < bestRank) { bestRank = r; bestPair = pair; }
            }
            if (bestRank == int.MaxValue) break;
            ids = MergePair(ids, bestPair, _merged[bestPair]);
        }
        return ids;
    }

    private static List<int> MergePair(List<int> ids, (int, int) pair, int newId)
    {
        var outp = new List<int>(ids.Count);
        for (int i = 0; i < ids.Count;)
        {
            if (i < ids.Count - 1 && ids[i] == pair.Item1 && ids[i + 1] == pair.Item2)
            {
                outp.Add(newId);
                i += 2;
            }
            else { outp.Add(ids[i]); i++; }
        }
        return outp;
    }

    private static int[] MergePairArray(int[] ids, (int, int) pair, int newId)
    {
        var outp = new List<int>(ids.Length);
        for (int i = 0; i < ids.Length;)
        {
            if (i < ids.Length - 1 && ids[i] == pair.Item1 && ids[i + 1] == pair.Item2)
            {
                outp.Add(newId);
                i += 2;
            }
            else { outp.Add(ids[i]); i++; }
        }
        return [.. outp];
    }

    public string Decode(IReadOnlyList<int> tokens, bool skipSpecial = true)
    {
        var bytes = new List<byte>();
        foreach (int id in tokens)
        {
            if (id < 0 || id >= _vocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"token id {id} out of range [0,{_vocabSize}).");
            if (id >= ByteCount && id < FirstMerge) { if (!skipSpecial) { } continue; }  // special
            bytes.AddRange(_bytes[id]);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    private const int FileMagic   = 0x42504531;   // "BPE1" — whole-stream merges
    private const int FileMagicV2 = 0x42504532;   // "BPE2" — word-split merges (Scale B2)

    /// <summary>
    /// Saves the learned merges (in rank order) to a binary file. The 256 byte tokens
    /// and the 4 specials are implicit, so only the merge table needs persisting; the
    /// byte expansions are rebuilt deterministically on load. The magic carries the
    /// training mode, so a reloaded tokenizer encodes exactly like the original.
    /// </summary>
    public void Save(string path)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write(_wordSplit ? FileMagicV2 : FileMagic);
        w.Write(_vocabSize);

        var merges = _rank.OrderBy(kv => kv.Value).ToList();   // rank 0,1,2,… = merge priority
        w.Write(merges.Count);
        foreach (var kv in merges)
        {
            w.Write(kv.Key.Item1);
            w.Write(kv.Key.Item2);
            w.Write(_merged[kv.Key]);
        }
    }

    /// <summary>Reloads a tokenizer saved with <see cref="Save"/> (either mode).</summary>
    public static BpeTokenizer Load(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        int magic = r.ReadInt32();
        if (magic != FileMagic && magic != FileMagicV2)
            throw new InvalidDataException("Not a valid BPE tokenizer file.");

        int vocabSize = r.ReadInt32();
        int count     = r.ReadInt32();

        var tok = new BpeTokenizer { _wordSplit = magic == FileMagicV2 };
        for (int m = 0; m < count; m++)
        {
            int a = r.ReadInt32(), b = r.ReadInt32(), id = r.ReadInt32();
            tok._rank[(a, b)]   = m;
            tok._merged[(a, b)] = id;
            tok._bytes[id]      = [.. tok._bytes[a], .. tok._bytes[b]];   // a,b already known (rank order)
        }
        tok._vocabSize = vocabSize;
        return tok;
    }
}
