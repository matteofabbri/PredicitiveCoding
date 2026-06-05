# PredictiveCoding.Data — data sources

Roadmap of data sources for `PredictiveCoding.Data`. Each source either yields a
`DenseTensor<float>` (media / numeric) or normalised records / text (which feed
the tokenizers in `PredictiveCoding.LLM`).

| Source | Implemented |
|--------|:-----------:|
| Audio files (NAudio) | ✅ |
| Video files (OpenCvSharp) | ✅ |
| YouTube (YoutubeExplode) | ✅ |
| RSS / Atom feeds (CodeHollow.FeedReader) | ✅ |
| Wikidata (HTTP + SPARQL) | ✅ |
| Still images (SixLabors.ImageSharp) | ✅ |
| CSV / TSV tabular (CsvHelper) | ✅ |
| MNIST / Fashion-MNIST (IDX parser) | ✅ |
| Biosignals — EEG/EMG/ECG (EDF parser) | ✅ |
| HTML / web pages (AngleSharp) | ✅ |
| Webcam capture (OpenCvSharp) | ✅ |
| Microphone capture (NAudio) | ✅ |
| Animated GIF / APNG (ImageSharp) | ✅ |
| Audio spectrogram / MFCC (NWaves) | ✅ |
| MIDI (Melanchall.DryWetMidi) | ✅ |
| DICOM medical images (fo-dicom) | ✅ |
| NIfTI volumes (parser) | ❌ |
| Plain-text corpus / directory | ✅ |
| PDF text (UglyToad.PdfPig) | ✅ |
| DOCX (DocumentFormat.OpenXml) | ✅ |
| EPUB ebooks (VersOne.Epub) | ✅ |
| Email — IMAP / .eml (MailKit / MimeKit) | ✅ |
| Wikipedia (MediaWiki API) | ✅ |
| arXiv (Atom API) | ✅ |
| PubMed / NCBI (E-utilities) | ✅ |
| GitHub (Octokit) | ✅ |
| Hacker News (Firebase API) | ✅ |
| Reddit (public JSON API) | ✅ |
| Stack Exchange (API) | ✅ |
| Mastodon (public REST API) | ✅ |
| DBpedia / generic SPARQL | ✅ |
| OpenAlex / Crossref / Semantic Scholar | ✅ |
| OpenStreetMap (Overpass API) | ✅ |
| WordNet / ConceptNet | ❌ |
| World Bank / Eurostat open data | ✅ |
| Finance — OHLCV (Yahoo / Stooq) | ✅ |
| Cryptocurrency (Binance / Coinbase) | ✅ |
| Weather (Open-Meteo) | ✅ |
| Parquet / Arrow (Parquet.Net) | ✅ |
| Excel XLSX (ClosedXML) | ✅ |
| Hugging Face Datasets (datasets-server) | ❌ |
| SQL / SQLite (Microsoft.Data.Sqlite / Dapper) | ✅ |
| Blob storage — S3 / Azure Blob | ❌ |
| IMU / sensors (MQTT, MQTTnet) | ✅ |
