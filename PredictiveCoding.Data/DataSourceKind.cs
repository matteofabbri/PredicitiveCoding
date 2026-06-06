namespace PredictiveCoding.Data;

/// <summary>
/// Identifies every data source available in <c>PredictiveCoding.Data</c>.
/// Sources broadly fall into three output shapes: media/numeric data that
/// becomes a <see cref="PredictiveCoding.Core.DenseTensor{T}"/>, and text /
/// structured data that becomes records (input for the LLM tokenizers).
/// </summary>
public enum DataSourceKind
{
    // ── Media & signals (→ DenseTensor) ───────────────────────────────────────

    /// <summary>Audio files via NAudio.</summary>
    Audio,
    /// <summary>Video files via OpenCvSharp.</summary>
    Video,
    /// <summary>Still images via SixLabors.ImageSharp.</summary>
    Image,
    /// <summary>Animated images (GIF/APNG) via ImageSharp.</summary>
    AnimatedImage,
    /// <summary>Live webcam capture via OpenCvSharp.</summary>
    Webcam,
    /// <summary>Live microphone capture via NAudio.</summary>
    Microphone,
    /// <summary>Audio spectrogram / MFCC features via NWaves.</summary>
    Spectrogram,
    /// <summary>MIDI files via DryWetMidi.</summary>
    Midi,
    /// <summary>DICOM medical images via fo-dicom.</summary>
    Dicom,
    /// <summary>NIfTI neuroimaging volumes.</summary>
    Nifti,
    /// <summary>EDF biosignals (EEG/EMG/ECG).</summary>
    Edf,

    // ── Tabular & numeric (→ DenseTensor) ──────────────────────────────────────

    /// <summary>CSV / TSV files via CsvHelper.</summary>
    Csv,
    /// <summary>Excel .xlsx workbooks via ClosedXML.</summary>
    Excel,
    /// <summary>Apache Parquet files via Parquet.Net.</summary>
    Parquet,
    /// <summary>SQLite databases via Microsoft.Data.Sqlite.</summary>
    Sqlite,
    /// <summary>MNIST / Fashion-MNIST datasets (IDX).</summary>
    Mnist,
    /// <summary>Weather time series via Open-Meteo.</summary>
    Weather,
    /// <summary>Financial OHLCV series via Stooq.</summary>
    Finance,
    /// <summary>Cryptocurrency prices / candles via Coinbase.</summary>
    Crypto,
    /// <summary>World Bank open-data indicators.</summary>
    WorldBank,

    // ── Dataset catalogues (R / Python teaching &amp; benchmark data → tables) ──────

    /// <summary>R's bundled / package datasets via the Rdatasets CSV mirror (iris, mtcars, …).</summary>
    RDatasets,
    /// <summary>Python's standard datasets via the seaborn-data repository (penguins, tips, titanic, …).</summary>
    SeabornDatasets,

    // ── Documents (local → text) ───────────────────────────────────────────────

    /// <summary>Plain-text corpus / directory.</summary>
    TextCorpus,
    /// <summary>PDF text via PdfPig.</summary>
    Pdf,
    /// <summary>Word .docx via the Open XML SDK.</summary>
    Docx,
    /// <summary>EPUB ebooks via VersOne.Epub.</summary>
    Epub,
    /// <summary>Email messages (.eml / MIME) via MimeKit.</summary>
    Email,
    /// <summary>HTML / web pages via AngleSharp.</summary>
    Html,

    // ── Web, APIs & knowledge (→ records / text) ───────────────────────────────

    /// <summary>YouTube metadata, captions and media via YoutubeExplode.</summary>
    YouTube,
    /// <summary>RSS / Atom feeds via FeedReader.</summary>
    Rss,
    /// <summary>Wikidata entities, search and SPARQL.</summary>
    Wikidata,
    /// <summary>Wikipedia summaries, extracts and search.</summary>
    Wikipedia,
    /// <summary>arXiv papers via the Atom API.</summary>
    Arxiv,
    /// <summary>PubMed / NCBI via the E-utilities.</summary>
    PubMed,
    /// <summary>Hacker News via the Firebase API.</summary>
    HackerNews,
    /// <summary>Reddit via the public JSON API.</summary>
    Reddit,
    /// <summary>Mastodon via the public REST API.</summary>
    Mastodon,
    /// <summary>Stack Exchange (Stack Overflow, …) search.</summary>
    StackExchange,
    /// <summary>OpenAlex scholarly works.</summary>
    OpenAlex,
    /// <summary>ScienceDirect / Elsevier via the official Content API.</summary>
    ScienceDirect,
    /// <summary>Unpaywall legal open-access lookup by DOI.</summary>
    Unpaywall,
    /// <summary>OpenStreetMap geographic data via Overpass.</summary>
    OpenStreetMap,
    /// <summary>ConceptNet commonsense knowledge.</summary>
    ConceptNet,
    /// <summary>GitHub repositories, READMEs and search via Octokit.</summary>
    GitHub,
    /// <summary>Hugging Face Hub datasets via the datasets-server.</summary>
    HuggingFace,
    /// <summary>Generic SPARQL endpoint (DBpedia, Wikidata, …).</summary>
    Sparql,

    // ── Local model servers (→ text / vectors) ─────────────────────────────────

    /// <summary>Local LLM via LM Studio's OpenAI-compatible server (text generation &amp; embeddings).</summary>
    LmStudio,

    // ── Storage & streaming ────────────────────────────────────────────────────

    /// <summary>Blob storage download by URL (S3 / Azure / GCS).</summary>
    Blob,
    /// <summary>MQTT sensor / IoT streams via MQTTnet.</summary>
    Mqtt,
}
