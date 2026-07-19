using PredictiveCoding.Core;
using PredictiveCoding.LLM;
using PredictiveCoding.Networks.DeepRecurrent;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// WernickeArea  —  language comprehension (sequence → meaning)
//
// Wernicke's area maps an incoming stream of speech/word units onto meaning.  The
// "comprehension" stream of the dual-stream model integrates the token sequence
// over time into a semantic representation.  Implemented over a DeepRecurrentNetwork
// that consumes a sequence of token vectors and emits a semantic latent.
//
// LLM MODE (Cortex ROADMAP opening, 2026-07-19): construct with a PredictiveLlm
// and the REAL language model stands in as the comprehension stream — meaning is
// its embedding (mean final hidden state), surprise is its prediction error (the
// framework's universal currency), and Listen() is online learning while hearing.
// The toy vector mode stays the default; the two modes never mix in one instance.
//
// References: Hickok & Poeppel (2007) dual-stream speech processing; Friston (2008).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record WernickeAreaConfig
{
    /// <summary>Temporal sequence model.  Stages[0].LayerSizes[0] = token-vector dim;
    /// Stages[^1].LayerSizes[^1] = semantic latent dim.</summary>
    public DeepRecurrentNetworkConfig Sequence { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/wernicke";

    public void Validate() => Sequence.Validate();
}

public sealed class WernickeArea : IDisposable
{
    private readonly DeepRecurrentNetwork? _seq;
    private readonly PredictiveLlm?        _llm;   // not owned: caller keeps/disposes nothing (plain object)

    /// <summary>True when the real language model is the comprehension stream.</summary>
    public bool UsesLanguageModel => _llm is not null;

    public int TokenDim    => _seq?.InputDim ?? _llm!.Config.Vocab;
    public int SemanticDim => _seq?.TopLatentDim ?? _llm!.Config.Dim;

    public WernickeArea(WernickeAreaConfig config, Random rng)
    {
        config.Validate();
        _seq = new DeepRecurrentNetwork(config.Sequence with { BasePath = config.BasePath }, rng);
    }

    /// <summary>LLM mode: the trained PredictiveLlm IS the comprehension stream.</summary>
    public WernickeArea(PredictiveLlm languageModel) => _llm = languageModel;

    // ── Toy vector mode (unchanged) ──────────────────────────────────────────

    /// <summary>Learns one utterance (token sequence); returns the final semantic latent.</summary>
    public DenseTensor<float> LearnUtterance(IReadOnlyList<DenseTensor<float>> tokens, int steps)
    {
        var seq = RequireSequenceMode();
        if (tokens.Count == 0) throw new ArgumentException("At least one token is required.", nameof(tokens));
        seq.Reset();
        DenseTensor<float> top = null!;
        foreach (var tok in tokens) top = seq.Step(tok, steps);
        return top;
    }

    /// <summary>Comprehends a token sequence into a semantic latent (inference, no learning).</summary>
    public DenseTensor<float> Comprehend(IReadOnlyList<DenseTensor<float>> tokens, int steps)
    {
        var seq = RequireSequenceMode();
        if (tokens.Count == 0) throw new ArgumentException("At least one token is required.", nameof(tokens));
        seq.Reset();
        DenseTensor<float> top = null!;
        foreach (var tok in tokens) top = seq.Encode(tok, steps);
        return top;
    }

    // ── LLM mode: text in, meaning/surprise out ──────────────────────────────

    /// <summary>
    /// Comprehends text into the semantic latent (the LLM embedding, [SemanticDim]) —
    /// the meaning vector the rest of the cortex consumes (hippocampus, PFC, and
    /// <see cref="BrocaArea.ProduceFromMeaning"/> as its latent plan).
    /// </summary>
    public DenseTensor<float> Comprehend(string text)
    {
        var llm = RequireLlmMode();
        return new DenseTensor<float>(llm.Embed(text), llm.Config.Dim);
    }

    /// <summary>
    /// The prediction error of the utterance in nats/token — the framework's
    /// universal currency: low = understood/expected, high = novel/anomalous.
    /// </summary>
    public float Surprise(string text) => (float)RequireLlmMode().Evaluate(text).NllPerToken;

    /// <summary>
    /// Hears AND learns (the PC symmetry, in the comprehension stream): one online
    /// learning step on the utterance. Returns surprise before/after.
    /// </summary>
    public OnlineLearnReport Listen(string text, int steps = 1) => RequireLlmMode().Learn(text, steps);

    private DeepRecurrentNetwork RequireSequenceMode()
        => _seq ?? throw new InvalidOperationException(
            "This WernickeArea runs on a PredictiveLlm — use the text-based Comprehend/Surprise/Listen.");

    private PredictiveLlm RequireLlmMode()
        => _llm ?? throw new InvalidOperationException(
            "This WernickeArea runs on the toy sequence network — construct it with a PredictiveLlm for text.");

    public void FlushAll() => _seq?.FlushAll();
    public void Dispose()  => _seq?.Dispose();
}
