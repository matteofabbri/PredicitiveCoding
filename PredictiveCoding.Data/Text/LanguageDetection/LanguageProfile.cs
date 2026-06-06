using System.IO.Compression;
using System.Text;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageProfile  —  the offline language-detection model.
//
// A character-n-gram naïve-Bayes model (Nakatani-Shuyo style): for each 1–3 char
// n-gram (keyed by its CaseSensitiveHash32) the per-language probability. The data
// is the model bundled in Catalyst, extracted once into a compact deflate-compressed
// file by tools/CatalystImport and embedded here, so detection runs fully offline
// with no Catalyst dependency at runtime.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class LanguageProfile
{
    private const uint Magic = 0x50434C44;   // "PCLD"

    /// <summary>Language labels, indexed by the language index used in <see cref="Probabilities"/>.</summary>
    public string[] Languages { get; }

    /// <summary>n-gram hash → array of (language index, probability).</summary>
    public IReadOnlyDictionary<int, (int Lang, double Prob)[]> Probabilities { get; }

    private LanguageProfile(string[] languages, Dictionary<int, (int, double)[]> probabilities)
    {
        Languages = languages;
        Probabilities = probabilities;
    }

    /// <summary>Loads the model embedded in this assembly (the default offline profile).</summary>
    public static LanguageProfile LoadEmbedded()
    {
        var assembly = typeof(LanguageProfile).Assembly;
        string name = Array.Find(assembly.GetManifestResourceNames(),
                          n => n.EndsWith("langdetect.bin", StringComparison.Ordinal))
                      ?? throw new InvalidOperationException(
                          "Embedded language model 'langdetect.bin' not found. Run tools/CatalystImport and rebuild.");
        using var stream = assembly.GetManifestResourceStream(name)!;
        return Load(stream);
    }

    /// <summary>Loads a model from a file produced by tools/CatalystImport.</summary>
    public static LanguageProfile Load(string path)
    {
        using var fs = File.OpenRead(path);
        return Load(fs);
    }

    /// <summary>Loads a model from a (deflate-compressed) stream.</summary>
    public static LanguageProfile Load(Stream compressed)
    {
        using var deflate = new DeflateStream(compressed, CompressionMode.Decompress);
        using var br = new BinaryReader(deflate, Encoding.UTF8);

        if (br.ReadUInt32() != Magic)
            throw new InvalidDataException("Not a PredictiveCoding language profile.");
        int format = br.ReadInt32();
        if (format != 1)
            throw new InvalidDataException($"Unsupported language-profile format {format}.");

        int langCount = br.ReadInt32();
        var languages = new string[langCount];
        for (int i = 0; i < langCount; i++) languages[i] = br.ReadString();

        int entryCount = br.ReadInt32();
        var probabilities = new Dictionary<int, (int, double)[]>(entryCount);
        for (int e = 0; e < entryCount; e++)
        {
            int hash = br.ReadInt32();
            int k = br.ReadInt32();
            var perLang = new (int, double)[k];
            for (int j = 0; j < k; j++)
            {
                int langIndex = br.ReadInt32();
                double prob = br.ReadDouble();
                perLang[j] = (langIndex, prob);
            }
            probabilities[hash] = perLang;
        }
        return new LanguageProfile(languages, probabilities);
    }
}
