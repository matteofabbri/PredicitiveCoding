using System.Text;
using PredictiveCoding.Data.Text;

Console.OutputEncoding = Encoding.UTF8;

var langdetect = LanguageDetector.CreateDefault(seed: 42);   // imported langdetect (53 profiles)
var ngram = NGramLanguageClassifier.CreateDefault();          // native n-gram naive-Bayes

(string label, string text)[] samples =
[
    ("EN", "This is a short English text used to test automatic language detection."),
    ("IT", "Questo è un breve testo in italiano per verificare il riconoscimento automatico della lingua."),
    ("FR", "Ceci est un court texte en français pour tester la détection automatique de la langue."),
    ("DE", "Dies ist ein kurzer deutscher Text, um die automatische Spracherkennung zu testen."),
    ("ES", "Este es un breve texto en español para probar la detección automática del idioma."),
    ("PT", "Este é um pequeno texto em português para testar a detecção automática do idioma."),
];

Console.WriteLine($"{"sample",-8}{"langdetect",-22}{"n-gram classifier",-22}");
Console.WriteLine(new string('-', 52));
foreach (var (label, text) in samples)
    Console.WriteLine($"{label,-8}{langdetect.Detect(text),-22}{ngram.Predict(text),-22}");

Console.WriteLine("\nlangdetect DetectAll (ES sample):");
foreach (var c in langdetect.DetectAll(samples[4].text))
    Console.WriteLine($"   {c.Language,-12} {c.Probability:P1}");

Console.WriteLine("\nShort/hard snippets (langdetect vs n-gram classifier):");
foreach (var t in new[] { "Hola, ¿qué tal estás hoy?", "Merci beaucoup, à bientôt!", "Grazie mille, a presto!", "Muito obrigado e até logo!" })
    Console.WriteLine($"   {langdetect.Detect(t),-12} {ngram.Predict(t),-12} <- {t}");
