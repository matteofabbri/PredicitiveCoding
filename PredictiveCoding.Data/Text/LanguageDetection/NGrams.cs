using System.Text;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// NGrams  —  shared text normalisation + character n-gram extraction used by the
// trainable n-gram language classifier (training and inference must agree).
//
// Normalisation: lower-case, keep letters (accents included — they are strong
// language signals), turn every run of non-letters into a single space, and wrap
// the text in spaces. n-grams therefore include word-boundary spaces (" de ",
// "es ", "tion") which are highly discriminative between languages.
// ─────────────────────────────────────────────────────────────────────────────

internal static class NGrams
{
    public static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length + 2);
        sb.Append(' ');
        bool space = true;
        foreach (char c in text)
        {
            if (char.IsLetter(c)) { sb.Append(char.ToLowerInvariant(c)); space = false; }
            else if (!space) { sb.Append(' '); space = true; }
        }
        if (!space) sb.Append(' ');
        return sb.ToString();
    }

    /// <summary>Counts character n-grams of order [minN, maxN] over the normalised text.</summary>
    public static Dictionary<string, int> Extract(string text, int minN, int maxN)
    {
        var counts = new Dictionary<string, int>();
        string s = Normalize(text);
        if (s.Length == 0) return counts;

        for (int n = minN; n <= maxN; n++)
            for (int i = 0; i + n <= s.Length; i++)
            {
                if (n == 1 && s[i] == ' ') continue;                 // the bare-space unigram carries no signal
                string g = s.Substring(i, n);
                counts[g] = counts.GetValueOrDefault(g) + 1;
            }
        return counts;
    }
}
