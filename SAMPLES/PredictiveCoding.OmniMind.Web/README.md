# PredictiveCoding.OmniMind.Web

OmniMind as a **Blazor Server** web app. Enter a topic and it fans out across
~24 live external providers (Wikipedia, arXiv, PubMed, GitHub, Reddit, markets,
weather, …), fuses what they return into a corpus, trains the project's own
`DecoderLM` transformer **from scratch** on it live, then samples a continuation —
streaming every step (source tiles, loss curve, generated tokens) to the browser
over SignalR. Every run is persisted to **run history**.

## Run

```sh
dotnet run --project SAMPLES/PredictiveCoding.OmniMind.Web
# → http://localhost:5240
```

- **/** — Ignite: the live experience (dashboard, loss chart, generated text).
- **/history** — past runs, searchable by topic.
- **/history/{id}** — replay a stored run.

Optional API keys (improve some sources) via environment variables:
`GITHUB_TOKEN`, `ELSEVIER_API_KEY`, `UNPAYWALL_EMAIL`.

## Persistence — Sperone

Run history is stored with [**Sperone**](../../../SPERONE), the embedded FASTER-backed
document store (`Sperone.Core`, referenced cross-repo). Runs live in a `runs`
collection indexed by `createdAt` (recency) and `topic` (full-text search); a
durable checkpoint is flushed after every run.

Data directory (configurable in `appsettings.json` → `OmniMind:DataDirectory`):
default `%USERPROFILE%/.predictivecoding/omnimind`.

## Configuration (`appsettings.json` → `OmniMind`)

| Key | Default | Meaning |
|-----|---------|---------|
| `DataDirectory` | `~/.predictivecoding/omnimind` | Sperone database location |
| `SourceTimeoutSeconds` | 25 | per-source fan-out timeout |
| `MaxConcurrentTrainings` | 1 | cap on simultaneous (CPU-bound) trainings |
| `Training.*` | Steps 160, Dim 80, … | from-scratch learning hyper-parameters |
| `Generation.*` | 220 tokens, temp 0.9, top-k 40 | sampling parameters |

## Layout

- `Engine/` — `KnowledgeFanOut`, `MindTrainer` (pure C#, shared with the original demo).
- `Services/` — `MindService` (orchestration + streaming), `HistoryStore` (Sperone).
- `Models/` — `RunRecord`, `IgniteUpdate`, `OmniMindOptions`.
- `Components/` — Blazor pages (Ignite, History, RunDetail) + `SourceTile`, `LossChart`.
