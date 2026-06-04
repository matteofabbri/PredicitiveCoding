using System.Text;

namespace PredictiveCoding.LLM.Tokenization;

// ─────────────────────────────────────────────────────────────────────────────
// WordPieceTokenizer  —  greedy longest-match subword tokenizer (BERT-style).
//
// Text is split on whitespace into words; each word is tokenised by greedily
// matching the longest vocabulary piece, with "##" marking non-initial pieces.
// The vocabulary is built from a corpus (all single characters are seeded for
// full coverage, then the most frequent substrings are added up to VocabSize).
// A word that cannot be fully covered becomes a single UNK, as in BERT.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class WordPieceTokenizer : ITokenizer
{
    private readonly List<string>             _pieces = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"];
    private readonly Dictionary<string, int>  _initial = new();   // word-start piece → id
    private readonly Dictionary<string, int>  _cont    = new();   // continuation piece (no "##") → id
    private int _maxPieceLen = 1;

    public int VocabSize => _pieces.Count;
    public int Pad => 0;
    public int Unk => 1;
    public int Bos => 2;
    public int Eos => 3;

    // ── Training ───────────────────────────────────────────────────────────────

    public void Train(string corpus, int vocabSize, int maxPieceLen = 8)
    {
        _maxPieceLen = Math.Max(1, maxPieceLen);
        var words = corpus.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        var initCount = new Dictionary<string, int>();
        var contCount = new Dictionary<string, int>();
        foreach (var word in words)
            for (int start = 0; start < word.Length; start++)
                for (int len = 1; len <= Math.Min(_maxPieceLen, word.Length - start); len++)
                {
                    string sub = word.Substring(start, len);
                    var bucket = start == 0 ? initCount : contCount;
                    bucket[sub] = bucket.GetValueOrDefault(sub) + 1;
                }

        // Seed every single character (both positions) so any seen char is coverable.
        foreach (char ch in corpus)
            if (!char.IsWhiteSpace(ch))
            {
                AddInitial(ch.ToString());
                AddCont(ch.ToString());
            }

        // Add the most frequent multi-char pieces until the budget is hit.
        foreach (var (sub, _) in initCount.Where(kv => kv.Key.Length > 1).OrderByDescending(kv => kv.Value))
        {
            if (_pieces.Count >= vocabSize) break;
            AddInitial(sub);
        }
        foreach (var (sub, _) in contCount.Where(kv => kv.Key.Length > 1).OrderByDescending(kv => kv.Value))
        {
            if (_pieces.Count >= vocabSize) break;
            AddCont(sub);
        }
    }

    private void AddInitial(string s)
    {
        if (_initial.ContainsKey(s)) return;
        _initial[s] = _pieces.Count;
        _pieces.Add(s);
    }

    private void AddCont(string s)
    {
        if (_cont.ContainsKey(s)) return;
        _cont[s] = _pieces.Count;
        _pieces.Add("##" + s);
    }

    // ── Encoding / decoding ──────────────────────────────────────────────────────

    public IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>();
        if (addBos) ids.Add(Bos);

        foreach (var word in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            EncodeWord(word, ids);

        if (addEos) ids.Add(Eos);
        return ids;
    }

    private void EncodeWord(string word, List<int> ids)
    {
        var pieces = new List<int>();
        int pos = 0;
        bool start = true;
        while (pos < word.Length)
        {
            int matchLen = 0, matchId = -1;
            int maxLen = Math.Min(_maxPieceLen, word.Length - pos);
            for (int len = maxLen; len >= 1; len--)
            {
                string sub = word.Substring(pos, len);
                var dict = start ? _initial : _cont;
                if (dict.TryGetValue(sub, out int id)) { matchLen = len; matchId = id; break; }
            }
            if (matchId < 0) { ids.Add(Unk); return; }   // whole word → UNK (BERT behaviour)
            pieces.Add(matchId);
            pos  += matchLen;
            start = false;
        }
        ids.AddRange(pieces);
    }

    public string Decode(IReadOnlyList<int> tokens, bool skipSpecial = true)
    {
        var sb = new StringBuilder();
        bool first = true;
        foreach (int id in tokens)
        {
            if (id < 0 || id >= _pieces.Count)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"token id {id} out of range.");
            if (id <= Eos) continue;                      // special tokens

            string piece = _pieces[id];
            if (piece.StartsWith("##", StringComparison.Ordinal))
                sb.Append(piece, 2, piece.Length - 2);
            else
            {
                if (!first) sb.Append(' ');
                sb.Append(piece);
            }
            first = false;
        }
        return sb.ToString();
    }
}
