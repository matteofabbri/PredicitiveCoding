using System.IO.Compression;
using System.Reflection;
using System.Text;
using Catalyst.Models;
using Mosaik.Core;

// ─────────────────────────────────────────────────────────────────────────────
// CatalystImport  —  extracts Catalyst's embedded language-detection model into a
// compact, portable file that PredictiveCoding loads natively.
//
// Catalyst ships the model as an embedded resource (Nakatani-Shuyo n-gram naive
// Bayes): LanguageDetectorModel { Dictionary<int,Dictionary<Language,double>>
// WordLanguageProbabilities; List<Language> Languages }. The int keys are
// CaseSensitiveHash32 hashes of 1–3 char n-grams. We dump exactly those two
// structures; the native detector recomputes the identical hashes at runtime.
//
// Usage:  dotnet run -- <output-file>
// ─────────────────────────────────────────────────────────────────────────────

const uint Magic = 0x50434C44;   // "PCLD"
const int  Format = 1;

string outPath = args.Length > 0
    ? args[0]
    : Path.Combine(AppContext.BaseDirectory, "langdetect.bin");

Storage.Current ??= new DiskStorage("catalyst-models");

Console.WriteLine("Loading Catalyst language detector (embedded, offline)…");
var detector = await LanguageDetector.FromStoreAsync(Language.Any, 0, "");
var model = (LanguageDetectorModel)GetData(detector);

var langs = model.Languages;
var wlp   = model.WordLanguageProbabilities;
Console.WriteLine($"  languages : {langs.Count}");
Console.WriteLine($"  n-grams   : {wlp.Count}");

var langIndex = new Dictionary<Language, int>();
for (int i = 0; i < langs.Count; i++) langIndex[langs[i]] = i;

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
using (var fs = File.Create(outPath))
using (var deflate = new DeflateStream(fs, CompressionLevel.SmallestSize))
using (var bw = new BinaryWriter(deflate, Encoding.UTF8))
{
    bw.Write(Magic);
    bw.Write(Format);

    bw.Write(langs.Count);
    foreach (var lang in langs) bw.Write(lang.ToString());   // stable, reversible label

    bw.Write(wlp.Count);
    foreach (var (hash, perLang) in wlp)
    {
        bw.Write(hash);
        bw.Write(perLang.Count);
        foreach (var (lang, prob) in perLang)
        {
            bw.Write(langIndex[lang]);
            bw.Write(prob);
        }
    }
}

var info = new FileInfo(outPath);
Console.WriteLine($"Wrote {outPath} ({info.Length / 1024.0:N1} KB)");
return 0;

// Catalyst stores the model on the (protected) StorableObject<,>.Data member.
static object GetData(object detector)
{
    for (var t = detector.GetType(); t is not null; t = t.BaseType)
    {
        var p = t.GetProperty("Data", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (p?.GetValue(detector) is { } pv) return pv;
        var f = t.GetField("Data", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (f?.GetValue(detector) is { } fv) return fv;
    }
    throw new InvalidOperationException("Could not locate LanguageDetector.Data via reflection.");
}
