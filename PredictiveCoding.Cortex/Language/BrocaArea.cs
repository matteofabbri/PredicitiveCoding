using PredictiveCoding.Core;
using PredictiveCoding.LLM;
using PredictiveCoding.Networks.Generative;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// BrocaArea  —  language production (meaning → sequence)
//
// Broca's area drives speech production: it turns an intended meaning into an
// articulatory/word sequence.  The "production" stream of the dual-stream model is
// a generative model that decodes a semantic latent into an utterance; pairs with
// MotorCortex for articulation.  Implemented over a GenerativeNetwork whose latent
// IS the semantic code.
//
// LLM MODE (Cortex ROADMAP opening, 2026-07-19): construct with a PredictiveLlm
// and the REAL language model is the production stream — Produce(prompt) generates
// text, and ProduceFromMeaning injects a semantic latent (typically
// WernickeArea.Comprehend's embedding) as the model's LATENT PLAN (2026 roadmap
// Phase 8) before decoding: meaning conditions production top-down, exactly the
// dual-stream story. The plan is restored afterwards — stateless between calls.
//
// References: Hickok & Poeppel (2007); Friston (2009) active inference of action.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record BrocaAreaConfig
{
    /// <summary>Generative model of utterances.  LatentDim = semantic (meaning) dim.</summary>
    public GenerativeNetworkConfig Generative { get; init; } = new();

    /// <summary>Std-dev of the latent walk used by <see cref="BrocaArea.ProduceSequence"/>.</summary>
    public float DriftScale { get; init; } = 0.3f;

    public string BasePath { get; init; } = "./cortex/broca";

    public void Validate()
    {
        if (DriftScale < 0f) throw new InvalidOperationException("DriftScale must be >= 0.");
    }
}

public sealed class BrocaArea : IDisposable
{
    private readonly GenerativeNetwork? _gen;
    private readonly BrocaAreaConfig?   _config;
    private readonly PredictiveLlm?     _llm;

    /// <summary>True when the real language model is the production stream.</summary>
    public bool UsesLanguageModel => _llm is not null;

    public int SemanticDim  => _gen?.LatentDim ?? _llm!.Config.Dim;
    public int UtteranceDim => _gen?.InputDim ?? _llm!.Config.Vocab;

    public BrocaArea(BrocaAreaConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        _gen    = new GenerativeNetwork(config.Generative with { BasePath = config.BasePath }, rng);
    }

    /// <summary>LLM mode: the trained PredictiveLlm IS the production stream.</summary>
    public BrocaArea(PredictiveLlm languageModel) => _llm = languageModel;

    // ── LLM mode: meaning/prompt in, text out ────────────────────────────────

    /// <summary>Produces a continuation of <paramref name="prompt"/> (greedy/speculative).</summary>
    public string Produce(string prompt, int maxNewTokens = 48)
        => RequireLlmMode().Generate(prompt, maxNewTokens, new GenerationOptions { StopAtEos = false });

    /// <summary>
    /// Top-down production FROM MEANING: injects <paramref name="semantic"/>
    /// (typically <see cref="WernickeArea.Comprehend(string)"/>'s embedding,
    /// [SemanticDim]) as the model's latent plan, scaled by
    /// <paramref name="planGain"/>, then decodes; the plan is restored after.
    /// Small gains condition, large gains overwhelm — the Phase-8 lesson.
    /// Requires a model built with <c>UsePlan = true</c>.
    /// </summary>
    public string ProduceFromMeaning(DenseTensor<float> semantic, string prompt, int maxNewTokens = 48, float planGain = 0.1f)
    {
        var llm = RequireLlmMode();
        if (!llm.Config.UsePlan)
            throw new InvalidOperationException("ProduceFromMeaning needs a model built with UsePlan = true (the latent plan is the meaning channel).");
        if (semantic.Length != llm.Config.Dim)
            throw new ArgumentException($"semantic has {semantic.Length} dims; expected SemanticDim ({llm.Config.Dim}).");

        var saved = llm.Model.GetPlan();
        try
        {
            var plan = new float[semantic.Length];
            for (int i = 0; i < plan.Length; i++) plan[i] = saved[i] + planGain * semantic.Buffer[i];
            llm.Model.SetPlan(plan);
            return llm.Generate(prompt, maxNewTokens, new GenerationOptions { StopAtEos = false });
        }
        finally { llm.Model.SetPlan(saved); }
    }

    private PredictiveLlm RequireLlmMode()
        => _llm ?? throw new InvalidOperationException(
            "This BrocaArea runs on the toy generative network — construct it with a PredictiveLlm for text.");

    private GenerativeNetwork RequireGenerativeMode()
        => _gen ?? throw new InvalidOperationException(
            "This BrocaArea runs on a PredictiveLlm — use the text-based Produce/ProduceFromMeaning.");

    // ── Toy generative mode (unchanged) ──────────────────────────────────────

    /// <summary>Learns the generative model of utterances (awake speech experience).</summary>
    public void LearnUtterance(DenseTensor<float> utterance, int steps) => RequireGenerativeMode().Learn(utterance, steps);

    /// <summary>Produces an utterance from a semantic latent (top-down generation).</summary>
    public DenseTensor<float> Produce(DenseTensor<float> semantic, int steps)
    {
        var gen = RequireGenerativeMode();
        if (semantic.Length != gen.LatentDim)
            throw new ArgumentException($"semantic has {semantic.Length} dims; expected SemanticDim ({gen.LatentDim}).");
        return gen.Decode(semantic, steps);
    }

    /// <summary>
    /// Produces a smoothly-varying sequence of utterances by drifting the semantic
    /// latent by DriftScale each step (a spoken phrase / prosodic contour).
    /// </summary>
    public List<DenseTensor<float>> ProduceSequence(DenseTensor<float> semantic, int length, int steps, Random rng)
    {
        var gen = RequireGenerativeMode();
        if (length < 1) throw new ArgumentOutOfRangeException(nameof(length));
        if (semantic.Length != gen.LatentDim)
            throw new ArgumentException($"semantic has {semantic.Length} dims; expected SemanticDim ({gen.LatentDim}).");

        var seq = new List<DenseTensor<float>>(length);
        var z   = semantic.Clone();
        for (int t = 0; t < length; t++)
        {
            seq.Add(gen.Decode(z, steps));
            if (_config!.DriftScale > 0f)
                for (int d = 0; d < z.Length; d++)
                    z.Buffer[d] += RandomUtils.SampleNormal(0f, _config.DriftScale, rng);
        }
        return seq;
    }

    public void FlushAll() => _gen?.FlushStore();
    public void Dispose()  => _gen?.Dispose();
}
