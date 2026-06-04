using System.Text;

namespace PredictiveCoding.LLM.Tokenization;

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer  —  text ↔ token-id conversion for the language model
//
// A byte-level tokenizer is used: it is lossless, language-agnostic and needs no
// training corpus.  The first 256 ids map to raw UTF-8 bytes; four special tokens
// follow (BOS, EOS, PAD, UNK).  A more sophisticated BPE/WordPiece tokenizer can
// implement the same ITokenizer interface later without touching the model.
// ─────────────────────────────────────────────────────────────────────────────

public interface ITokenizer
{
    int VocabSize { get; }
    int Bos { get; }
    int Eos { get; }
    int Pad { get; }
    int Unk { get; }

    /// <summary>Encodes text to token ids, optionally wrapping with BOS/EOS.</summary>
    IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false);

    /// <summary>Decodes token ids back to text, optionally dropping special tokens.</summary>
    string Decode(IReadOnlyList<int> tokens, bool skipSpecial = true);
}

public sealed class ByteTokenizer : ITokenizer
{
    private const int ByteCount = 256;

    public int VocabSize => ByteCount + 4;
    public int Bos => ByteCount + 0;   // 256
    public int Eos => ByteCount + 1;   // 257
    public int Pad => ByteCount + 2;   // 258
    public int Unk => ByteCount + 3;   // 259

    public IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        var bytes = Encoding.UTF8.GetBytes(text ?? string.Empty);
        var ids   = new List<int>(bytes.Length + 2);
        if (addBos) ids.Add(Bos);
        foreach (byte b in bytes) ids.Add(b);   // 0..255 map to themselves
        if (addEos) ids.Add(Eos);
        return ids;
    }

    public string Decode(IReadOnlyList<int> tokens, bool skipSpecial = true)
    {
        var bytes = new List<byte>(tokens.Count);
        foreach (int id in tokens)
        {
            if (id < 0 || id >= VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"token id {id} out of range [0,{VocabSize}).");
            if (id < ByteCount) bytes.Add((byte)id);
            else if (!skipSpecial) { /* special tokens have no byte form; skip */ }
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }
}
