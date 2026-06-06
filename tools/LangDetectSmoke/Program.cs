using System.Text;
using PredictiveCoding.Data.Text;

Console.OutputEncoding = Encoding.UTF8;

var detector = LanguageDetector.CreateDefault(seed: 42);

(string label, string text)[] samples =
[
    ("EN", "This is a short English text used to test automatic language detection."),
    ("IT", "Questo è un breve testo in italiano per verificare il riconoscimento automatico della lingua."),
    ("FR", "Ceci est un court texte en français pour tester la détection automatique de la langue."),
    ("DE", "Dies ist ein kurzer deutscher Text, um die automatische Spracherkennung zu testen."),
    ("ES", "Este es un breve texto en español para probar la detección automática del idioma."),
    ("PT", "Este é um pequeno texto em português para testar a detecção automática do idioma."),
];

Console.WriteLine("=== Detect (offline, native) ===");
foreach (var (label, text) in samples)
    Console.WriteLine($"  [{label}] -> {detector.Detect(text)}");

Console.WriteLine("\n=== DetectAll (IT sample, with probabilities) ===");
foreach (var c in detector.DetectAll(samples[1].text))
    Console.WriteLine($"   {c.Language,-14} {c.Probability:P1}");
