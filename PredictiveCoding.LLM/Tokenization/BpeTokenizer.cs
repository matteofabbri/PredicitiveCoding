using System.Text;

namespace PredictiveCoding.LLM.Tokenization;

// ─────────────────────────────────────────────────────────────────────────────
// BpeTokenizer  —  trainable byte-level Byte-Pair Encoding.
//
// Starts from the 256 byte tokens (+ BOS/EOS/PAD/UNK) and learns a sequence of
// merges of the most frequent adjacent token pair, exactly like GPT-2's BPE but
// without word pre-tokenisation (merges operate over the whole byte stream).
// Encoding greedily applies the learned merges in priority (rank) order; decoding
// expands each token back to its byte sequence — lossless for the byte tokens.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class BpeTokenizer : ITokenizer
{
    private const int ByteCount  = 256;
    private const int FirstMerge = ByteCount + 4;   // ids 256..259 are specials

    private readonly Dictionary<(int, int), int> _rank   = new();   // pair → merge priority
    private readonly Dictionary<(int, int), int> _merged = new();   // pair → new token id
    private readonly Dictionary<int, byte[]>     _bytes  = new();   // token id → byte expansion
    private int _vocabSize = FirstMerge;

    public int VocabSize => _vocabSize;
    public int Bos => ByteCount + 0;
    public int Eos => ByteCount + 1;
    public int Pad => ByteCount + 2;
    public int Unk => ByteCount + 3;

    public BpeTokenizer()
    {
        for (int b = 0; b < ByteCount; b++) _bytes[b] = [(byte)b];
    }

    // ── Training ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Learns up to <paramref name="numMerges"/> merges from <paramref name="corpus"/>.
    /// Stops early if no adjacent pair occurs more than once.
    /// </summary>
    public void Train(string corpus, int numMerges)
    {
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

            int newId = _vocabSize++;
            _rank[best.Key]   = m;
            _merged[best.Key] = newId;
            _bytes[newId]     = [.. _bytes[best.Key.Item1], .. _bytes[best.Key.Item2]];

            ids = MergePair(ids, best.Key, newId);
        }
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

    // ── Encoding / decoding ──────────────────────────────────────────────────────

    public IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>(Encoding.UTF8.GetBytes(text ?? string.Empty).Select(b => (int)b));

        // Greedily merge the highest-priority (lowest-rank) adjacent pair until none apply.
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

        if (!addBos && !addEos) return ids;
        var wrapped = new List<int>(ids.Count + 2);
        if (addBos) wrapped.Add(Bos);
        wrapped.AddRange(ids);
        if (addEos) wrapped.Add(Eos);
        return wrapped;
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
}
