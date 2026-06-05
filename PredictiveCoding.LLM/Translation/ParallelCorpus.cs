using System.Text;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Translation;

// ─────────────────────────────────────────────────────────────────────────────
// ParallelCorpus  —  reads parallel sentence pairs and prepares them for training.
//
// Expected format: a tab-separated file with one pair per line —
//     <target-tag>\t<source text>\t<target text>
// e.g.  ita\tHello world\tCiao mondo
//
// This matches the easily-exported Tatoeba / OPUS sentence-pair format (after a
// trivial column reshuffle). The target tag drives the many-to-target model: the
// same source can appear with several tags, each mapping to a different language.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record SentencePair(string Source, string Target, string TargetTag);

public static class ParallelCorpus
{
    /// <summary>Streams pairs from a TSV; blank lines and '#' comments are skipped.</summary>
    public static IEnumerable<SentencePair> ReadTsv(
        string path, int tagCol = 0, int srcCol = 1, int tgtCol = 2, char sep = '\t')
    {
        int need = Math.Max(tagCol, Math.Max(srcCol, tgtCol));
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line) || line[0] == '#') continue;
            var cols = line.Split(sep);
            if (cols.Length <= need) continue;

            string tag = cols[tagCol].Trim(), src = cols[srcCol].Trim(), tgt = cols[tgtCol].Trim();
            if (tag.Length == 0 || src.Length == 0 || tgt.Length == 0) continue;
            yield return new SentencePair(src, tgt, tag);
        }
    }

    /// <summary>Trains a shared byte-level BPE over the formatted text of every pair.</summary>
    public static BpeTokenizer TrainTokenizer(IEnumerable<SentencePair> pairs, int numMerges)
    {
        var sb = new StringBuilder();
        foreach (var p in pairs)
            sb.Append(TranslationFormat.PromptText(p.Source, p.TargetTag)).Append(p.Target).Append('\n');

        var tok = new BpeTokenizer();
        tok.Train(sb.ToString(), numMerges);
        return tok;
    }

    /// <summary>Tokenises every pair into a (tokens, targets) example, dropping any longer than maxSeq.</summary>
    public static List<(int[] tokens, int[] targets)> BuildExamples(
        IEnumerable<SentencePair> pairs, BpeTokenizer tok, int maxSeq)
    {
        var list = new List<(int[], int[])>();
        foreach (var p in pairs)
        {
            var ex = TranslationFormat.BuildExample(tok, p.Source, p.Target, p.TargetTag);
            if (ex.tokens.Length <= maxSeq) list.Add(ex);          // tokens length == sequence length fed to Forward
        }
        return list;
    }
}
