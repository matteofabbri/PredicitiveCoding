using PredictiveCoding.Data.Text;

namespace PredicitiveCoding.UnitTests;

// The embedded language detector: the original langdetect n-gram profiles
// (53 languages) imported from Maddalena.Cognitive, where they ran in production.
// Guards the failure that forced the import: with the previous Catalyst-based
// model, clean ~200-char Italian paragraphs came back "English" with probability
// 1.0, so the WebCrawler language filter was unusable. Detection is randomised —
// every test seeds it.
public class LanguageDetectionTests
{
    private static LanguageDetector NewDetector() => LanguageDetector.CreateDefault(seed: 42);

    // ── The reported case: short paragraphs of the main European languages ──────

    [Theory]
    [InlineData("Italian",
        "La biblioteca comunale rimane aperta tutti i giorni fino alle venti. I lettori possono consultare " +
        "liberamente gli scaffali, chiedere consiglio ai bibliotecari e partecipare agli incontri organizzati ogni settimana.")]
    [InlineData("English",
        "The public library stays open every day until eight in the evening. Readers can browse the shelves " +
        "freely, ask the librarians for advice and take part in the meetings that are organised every single week.")]
    [InlineData("German",
        "Die Stadtbibliothek bleibt jeden Tag bis zwanzig Uhr geöffnet. Die Leser können frei in den Regalen " +
        "stöbern, die Bibliothekare um Rat fragen und an den wöchentlich organisierten Treffen teilnehmen.")]
    [InlineData("French",
        "La bibliothèque municipale reste ouverte tous les jours jusqu'à vingt heures. Les lecteurs peuvent " +
        "parcourir librement les rayons, demander conseil aux bibliothécaires et participer aux rencontres organisées chaque semaine.")]
    [InlineData("Spanish",
        "El sábado por la mañana el mercado llena la plaza de puestos de colores: fruta de temporada, queso de " +
        "montaña y flores recién cortadas. La gente se detiene a charlar y vuelve a casa con las bolsas llenas.")]
    [InlineData("Portuguese",
        "No sábado de manhã o mercado enche a praça de bancas coloridas: fruta da época, queijo da serra e " +
        "flores acabadas de colher. As pessoas param para conversar e voltam para casa com os sacos cheios.")]
    [InlineData("Dutch",
        "Op zaterdagochtend vult de markt het plein met kleurrijke kramen: fruit van het seizoen, bergkaas en " +
        "vers gesneden bloemen. Mensen blijven staan om te kletsen en lopen met volle tassen naar huis.")]
    public void Detect_ShortParagraph_NamesTheRightLanguage(string expected, string text)
        => Assert.Equal(expected, NewDetector().Detect(text));

    [Theory]
    [InlineData("Italian", "Grazie mille, a presto!")]
    [InlineData("French", "Merci beaucoup, à bientôt !")]
    [InlineData("German", "Vielen Dank und bis bald!")]
    [InlineData("English", "Thanks a lot, see you soon!")]
    [InlineData("Spanish", "Hola, ¿qué tal estás hoy?")]
    public void Detect_ShortSnippet_NamesTheRightLanguage(string expected, string text)
        => Assert.Equal(expected, NewDetector().Detect(text));

    // ── Beyond the Latin script (the old 7-language classifier had no answer here) ──

    [Theory]
    [InlineData("Russian",
        "Библиотека в центре города открыта каждый день до восьми вечера. Читатели могут свободно выбирать " +
        "книги и советоваться с библиотекарями.")]
    [InlineData("Greek",
        "Η δημοτική βιβλιοθήκη είναι ανοιχτή κάθε μέρα μέχρι τις οκτώ το βράδυ. Οι αναγνώστες μπορούν να " +
        "διαλέγουν ελεύθερα βιβλία.")]
    [InlineData("Chinese",
        "市立图书馆每天开放到晚上八点。读者可以自由浏览书架，也可以向图书管理员咨询问题。")]
    [InlineData("Romanian",
        "Biblioteca municipală rămâne deschisă în fiecare zi până la ora douăzeci. Cititorii pot răsfoi liber " +
        "rafturile și pot cere sfatul bibliotecarilor.")]
    public void Detect_OtherScriptsAndNeighbours_NamesTheRightLanguage(string expected, string text)
        => Assert.Equal(expected, NewDetector().Detect(text));

    // ── API contract ────────────────────────────────────────────────────────────

    [Fact]
    public void Detect_EmptyOrLetterless_IsUnknown()
    {
        var detector = NewDetector();
        Assert.Equal("Unknown", detector.Detect(""));
        Assert.Equal("Unknown", detector.Detect("   "));
        Assert.Equal("Unknown", detector.Detect("1234 5678 !!"));
        Assert.Null(detector.DetectInfo(""));
    }

    [Fact]
    public void DetectAll_RanksCandidates_HighestFirst()
    {
        var ranked = NewDetector().DetectAll(
            "Questa è una frase italiana scritta per controllare l'ordinamento dei candidati.");

        Assert.NotEmpty(ranked);
        Assert.Equal("Italian", ranked[0].Language);
        for (int i = 1; i < ranked.Count; i++)
            Assert.True(ranked[i - 1].Probability >= ranked[i].Probability);
        Assert.All(ranked, c => Assert.InRange(c.Probability, 0.0, 1.0));
    }

    [Fact]
    public void DetectInfo_ReturnsFullIsoMetadata()
    {
        LanguageInfo? info = NewDetector().DetectInfo(
            "Questo è un breve testo in italiano per verificare il riconoscimento della lingua.");

        Assert.NotNull(info);
        Assert.Equal("Italian", info.Name);
        Assert.Equal("it", info.Iso6391);
        Assert.Equal("ita", info.Iso6393);
        Assert.Equal("Indo-European", info.Family);
    }

    [Fact]
    public void Languages_ExposesTheIsoTable()
    {
        var table = LanguageDetector.Languages;
        Assert.True(table.Count > 150);
        Assert.Contains(table, l => l.Iso6391 == "it" && l.Name == "Italian");
        Assert.Contains(table, l => l.Iso6391 == "sv" && l.Name == "Swedish");
    }

    [Fact]
    public void Detect_SameSeed_IsDeterministic()
    {
        const string text = "Ett kort stycke på svenska för att kontrollera att resultatet är stabilt.";
        var first = LanguageDetector.CreateDefault(seed: 123).DetectAll(text);
        var second = LanguageDetector.CreateDefault(seed: 123).DetectAll(text);

        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
        {
            Assert.Equal(first[i].Language, second[i].Language);
            Assert.Equal(first[i].Probability, second[i].Probability, precision: 12);
        }
        Assert.Equal("Swedish", first[0].Language);
    }
}
