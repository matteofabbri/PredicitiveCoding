using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Data;

// ─────────────────────────────────────────────────────────────────────────────
// TextDataset  —  tokenised corpus sliced into (input, target) training windows.
//
// Tokenises a corpus once, then yields fixed-length windows where the target is
// the input shifted by one position (next-token prediction).  Ready to feed a
// future trainer.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TextDataset
{
    private readonly int[] _ids;
    private readonly int   _seqLen;

    public int TokenCount => _ids.Length;
    public int SeqLen     => _seqLen;

    public TextDataset(string corpus, ITokenizer tokenizer, int seqLen)
    {
        if (seqLen < 1) throw new ArgumentOutOfRangeException(nameof(seqLen));
        _ids    = tokenizer.Encode(corpus).ToArray();
        _seqLen = seqLen;
    }

    /// <summary>Loads a corpus from a UTF-8 text file.</summary>
    public static TextDataset FromFile(string path, ITokenizer tokenizer, int seqLen)
        => new(File.ReadAllText(path), tokenizer, seqLen);

    /// <summary>Number of windows produced for the given stride.</summary>
    public int WindowCount(int stride)
    {
        if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride));
        int last = _ids.Length - _seqLen - 1;
        return last < 0 ? 0 : last / stride + 1;
    }

    /// <summary>
    /// Enumerates (input, target) windows of length SeqLen; target is input shifted
    /// by one token (next-token prediction).
    /// </summary>
    public IEnumerable<(int[] input, int[] target)> Windows(int stride)
    {
        if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride));
        for (int start = 0; start + _seqLen + 1 <= _ids.Length; start += stride)
        {
            var input  = new int[_seqLen];
            var target = new int[_seqLen];
            Array.Copy(_ids, start,     input,  0, _seqLen);
            Array.Copy(_ids, start + 1, target, 0, _seqLen);
            yield return (input, target);
        }
    }
}
