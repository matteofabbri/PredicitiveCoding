using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Data;

// ─────────────────────────────────────────────────────────────────────────────
// StreamingTextDataset  —  windowed dataset over a corpus too large to hold in RAM.
//
// Reads the source lazily (line by line), tokenises incrementally into a small
// sliding buffer, and yields fixed-length (input, target) windows as they become
// available.  Consumed prefixes are dropped, so memory stays bounded regardless
// of corpus size.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class StreamingTextDataset
{
    private readonly IEnumerable<string> _lines;
    private readonly ITokenizer          _tokenizer;
    private readonly int                 _seqLen;
    private readonly int                 _stride;

    public StreamingTextDataset(IEnumerable<string> lines, ITokenizer tokenizer, int seqLen, int stride)
    {
        if (seqLen < 1) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride));
        _lines     = lines;
        _tokenizer = tokenizer;
        _seqLen    = seqLen;
        _stride    = stride;
    }

    /// <summary>Streams a UTF-8 text file line by line (constant memory).</summary>
    public static StreamingTextDataset FromFile(string path, ITokenizer tokenizer, int seqLen, int stride)
        => new(File.ReadLines(path), tokenizer, seqLen, stride);

    /// <summary>Lazily enumerates (input, target) windows; target is input shifted by one.</summary>
    public IEnumerable<(int[] input, int[] target)> Windows()
    {
        var buffer = new List<int>();
        int start  = 0;
        int trimAt = Math.Max(_seqLen * 4, 1024);   // drop consumed prefix past this point

        foreach (string line in _lines)
        {
            foreach (int id in _tokenizer.Encode(line + "\n")) buffer.Add(id);

            while (buffer.Count - start >= _seqLen + 1)
            {
                yield return Window(buffer, start);
                start += _stride;
            }

            if (start > trimAt)
            {
                buffer.RemoveRange(0, start);
                start = 0;
            }
        }

        while (buffer.Count - start >= _seqLen + 1)
        {
            yield return Window(buffer, start);
            start += _stride;
        }
    }

    private (int[] input, int[] target) Window(List<int> buffer, int start)
    {
        var input  = new int[_seqLen];
        var target = new int[_seqLen];
        buffer.CopyTo(start,     input,  0, _seqLen);
        buffer.CopyTo(start + 1, target, 0, _seqLen);
        return (input, target);
    }
}
