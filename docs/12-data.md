# 12. Data — `PredictiveCoding.Data`

One project that turns **45+ external data sources** into the two shapes the
rest of the solution consumes:

- **media / numeric data** → `DenseTensor<float>` (feeds the PC networks), and
- **text / structured records** (feeds the LLM tokenizers and the corpora).

Every source is reachable through the `DataSourceKind` enum, which groups them
by output shape. A per-source table with the library each one uses lives in
[`PredictiveCoding.Data/DATA_SOURCES.md`](../PredictiveCoding.Data/DATA_SOURCES.md).

## The catalogue (`DataSourceKind`)

| Group | Sources |
|-------|---------|
| **Media & signals** → tensor | Audio (NAudio), Video (OpenCvSharp), Image (in-house `PredictiveCoding.Imaging`), AnimatedImage GIF/APNG, Webcam, Microphone, Spectrogram/MFCC (NWaves), MIDI, DICOM (fo-dicom), NIfTI, EDF biosignals |
| **Tabular & numeric** → tensor | CSV/TSV, Excel (ClosedXML), Parquet, SQLite, MNIST/Fashion-MNIST (IDX), Weather (Open-Meteo), Finance OHLCV (Stooq), Crypto (Coinbase), World Bank |
| **Dataset catalogues** → tables | Rdatasets, seaborn-data, vega-datasets, plotly/datasets, OpenML (by id) |
| **Documents** → text | TextCorpus, PDF (PdfPig), DOCX (Open XML), EPUB, Email (.eml/MimeKit), HTML (AngleSharp) |
| **Web, APIs & knowledge** → records/text | YouTube (YoutubeExplode), RSS/Atom, Wikidata, Wikipedia, arXiv, PubMed, Hacker News, Reddit, Mastodon, Stack Exchange, OpenAlex, ScienceDirect, Unpaywall, OpenStreetMap, ConceptNet, GitHub (Octokit), Hugging Face datasets, generic SPARQL |
| **Local model servers** | LM Studio (OpenAI-compatible text generation & embeddings) |
| **Storage & streaming** | Blob by URL (S3/Azure/GCS), MQTT (MQTTnet) |

The **dataset catalogues** deserve a note: `Datasets/` ships one small source
per teaching catalogue (`RDatasetsSource`, `SeabornSource`, `VegaDatasetsSource`,
`PlotlyDatasetsSource`, `OpenMLSource`) plus `DatasetTable` and `MnistDataset`,
so iris / penguins / cars / MNIST-class data is one HTTP GET away without any
Python or R runtime.

## The web → corpus pipeline

The piece the LLM pretraining path is built on (`Web/`):

- **`WebCrawler`** — polite breadth-first crawling from seeds (`CrawlOptions`,
  page budget); fetch failures and "thin" pages are counted, not fatal.
- **`HtmlSource` / `IHtmlSource`** — AngleSharp-based **block-level** text
  extraction: page chrome is stripped and short boilerplate blocks are dropped,
  so what reaches the corpus is article text.
- **`ShardedCorpusWriter`** — appends every page to an **append-only shard
  folder** (`shard-*.txt`); the corpus *sediments* over time and can exceed
  memory because nothing is ever re-read as a whole.

Downstream, the LLM side adds the hygiene decorators
(`CorpusSource.FromFolder(dir).WithBoilerplateStripped().Deduplicated()` — see
[doc 6](06-llm.md)). Demos **12** (crawler) and **13** (pretraining run) in
`PredictiveCoding.Main` wire the whole path together.

## Language detection (`Text/`)

Two production-grade classifiers, both offline (profiles embedded as resources
— no service calls):

- **`LanguageDetector`** — the classic **langdetect** n-gram profiles
  (53 languages, Nakatani Shuyo, Apache 2.0) imported verbatim: `Detect`,
  `DetectAll` with probabilities.
- **`NGramLanguageClassifier`** — a native, *trainable* n-gram naive-Bayes
  classifier seeded from `language-samples.tsv`.

Demo **7** in Main pits a PC network against a naive-Bayes baseline on exactly
this problem. `tools/LangDetectSmoke` (see [doc 16](16-tools-and-cli.md)) is a
quick side-by-side of the two classifiers.

## Politeness & ethics

- **arXiv** goes through the official Atom API with a descriptive User-Agent
  and rate limiting — no scraping of the HTML site.
- **Sci-Hub was deliberately not implemented** (it distributes copyrighted
  papers by bypassing access controls). **Unpaywall** covers the same need —
  free full-text PDFs by DOI — using only legally deposited open-access copies.

## Dependencies

`PredictiveCoding.Data` references only the core library and `PredictiveCoding.Imaging`;
everything else is NuGet (AngleSharp, NAudio, OpenCvSharp, Octokit, PdfPig,
ClosedXML, Parquet.Net, MQTTnet, YoutubeExplode, …) — the full list is in the
[csproj](../PredictiveCoding.Data/PredictiveCoding.Data.csproj) and mirrored in
`DATA_SOURCES.md`.
