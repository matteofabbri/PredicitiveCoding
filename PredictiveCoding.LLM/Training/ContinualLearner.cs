using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Eval;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// ContinualLearner  —  robust learn-while-using (roadmap-Pro Phase 6).
//
// PredictiveLlm.Learn absorbs text but forgets old domains (catastrophic
// forgetting) and spends the same effort on the trivial and the surprising.
// This wraps a model with the three standard antidotes, expressed the PC way:
//
//   · Replay buffer — each absorb mixes k windows of PAST text (a rolling ring)
//     into the step, so the gradient keeps pulling toward what was already
//     learned. The cheap, decisive forgetting antidote.
//   · Surprise-proportional lr — the step is scaled by how surprising the text
//     is relative to a running average (bounded): learn hard from the
//     unexpected, barely from the already-predicted. Precision-weighting on the
//     weight side — the same "trust the error in proportion to its size" move
//     the core network makes in the states.
//   · Forgetting guardrail — every N absorbs, a fixed yardstick's perplexity is
//     re-measured; drift past a threshold fires a hook (warn / rollback).
//
// Deterministic throughout: replay is sampled by a moving cursor, not RNG.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Knobs for <see cref="ContinualLearner"/>.</summary>
public sealed record ContinualOptions
{
    /// <summary>Window length in tokens (clamped to MaxSeq).</summary>
    public int SeqLen { get; init; } = 32;
    /// <summary>Window stride in tokens.</summary>
    public int Stride { get; init; } = 16;
    /// <summary>Base learning rate before the surprise scale.</summary>
    public float BaseLr { get; init; } = 0.002f;
    /// <summary>Gradient steps per absorbed text.</summary>
    public int Steps { get; init; } = 1;
    public float ClipNorm { get; init; } = 1f;

    /// <summary>Rolling replay-buffer capacity in windows (0 = replay off).</summary>
    public int ReplayCapacity { get; init; } = 128;
    /// <summary>Replay windows mixed into each absorb step.</summary>
    public int ReplayPerStep { get; init; } = 4;

    /// <summary>Clamp on the surprise-driven lr multiplier (lr = BaseLr · clamp(surprise/avg, lo, hi)).</summary>
    public float MinLrScale { get; init; } = 0.25f;
    public float MaxLrScale { get; init; } = 4f;
    /// <summary>EMA decay for the running surprise average (0 disables the scale → lr = BaseLr).</summary>
    public float SurpriseEmaDecay { get; init; } = 0.9f;

    /// <summary>Re-measure the yardstick every N absorbs (0 = never). Requires a yardstick.</summary>
    public int GuardrailEvery { get; init; } = 0;
    /// <summary>Relative ppl drift over baseline that fires <see cref="ContinualLearner.OnDrift"/>.</summary>
    public double DriftThreshold { get; init; } = 0.25;
}

/// <summary>What one absorb did.</summary>
/// <param name="SurpriseBefore">Mean loss on the new text before the step.</param>
/// <param name="SurpriseAfter">Mean loss on the new text after the step.</param>
/// <param name="EffectiveLr">The surprise-scaled learning rate used.</param>
/// <param name="ReplayWindows">How many past windows were mixed in.</param>
public sealed record ContinualStep(float SurpriseBefore, float SurpriseAfter, float EffectiveLr, int ReplayWindows);

/// <summary>Guardrail report: the yardstick drifted past the threshold.</summary>
/// <param name="BaselinePerplexity">Yardstick ppl when the learner was created.</param>
/// <param name="CurrentPerplexity">Yardstick ppl now.</param>
/// <param name="RelativeDrift">(current − baseline) / baseline.</param>
/// <param name="AbsorbCount">Absorbs done so far.</param>
public sealed record ContinualDrift(double BaselinePerplexity, double CurrentPerplexity, double RelativeDrift, int AbsorbCount);

public sealed class ContinualLearner
{
    private readonly PredictiveLlm    _llm;
    private readonly ContinualOptions _opts;
    private readonly int              _seqLen;
    private readonly Adam             _optimizer = new();

    private readonly List<(int[] input, int[] target)> _replay = [];
    private int _replayCursor;

    private readonly List<(int[] input, int[] target)>? _yardstick;
    private readonly double _baselinePerplexity = double.NaN;

    private float _surpriseAvg = -1f;   // running EMA of surprise (−1 = unset)
    private int   _absorbCount;

    /// <summary>Fires when the yardstick perplexity drifts past the threshold (guardrail).</summary>
    public Action<ContinualDrift>? OnDrift { get; set; }

    /// <summary>Absorbs done so far.</summary>
    public int AbsorbCount => _absorbCount;

    /// <summary>Windows currently in the replay buffer.</summary>
    public int ReplaySize => _replay.Count;

    public ContinualLearner(PredictiveLlm llm, ContinualOptions? options = null, string? yardstickText = null)
    {
        _llm    = llm ?? throw new ArgumentNullException(nameof(llm));
        _opts   = options ?? new ContinualOptions();
        _seqLen = Math.Min(_opts.SeqLen, llm.Config.MaxSeq);

        if (!string.IsNullOrEmpty(yardstickText))
        {
            _yardstick = Windows(yardstickText);
            if (_yardstick.Count > 0)
                _baselinePerplexity = PerplexityEvaluator.Evaluate(llm.Model, _yardstick).Perplexity;
        }
    }

    /// <summary>The yardstick perplexity measured at construction (NaN if no yardstick).</summary>
    public double BaselinePerplexity => _baselinePerplexity;

    /// <summary>Re-measures the yardstick perplexity now (NaN if no yardstick).</summary>
    public double MeasureYardstick()
        => _yardstick is { Count: > 0 } ? PerplexityEvaluator.Evaluate(_llm.Model, _yardstick).Perplexity : double.NaN;

    /// <summary>
    /// Absorbs <paramref name="text"/>: surprise-scaled lr, new windows mixed with
    /// replayed past windows, then the new windows enter the ring. Only ever learns
    /// from observed text — never the model's own output. Runs the guardrail on the
    /// configured cadence.
    /// </summary>
    public ContinualStep Absorb(string text)
    {
        var fresh = Windows(text);
        if (fresh.Count == 0) fresh = [WholeWindow(text)];
        var ps = _llm.Model.Parameters();

        float before = MeanLoss(fresh);

        // Surprise-proportional lr (bounded). The EMA seeds from the first surprise
        // so the very first absorb runs at BaseLr (scale 1).
        float scale = 1f;
        if (_opts.SurpriseEmaDecay > 0f)
        {
            if (_surpriseAvg < 0f) _surpriseAvg = before;
            scale = Math.Clamp(before / MathF.Max(_surpriseAvg, 1e-6f), _opts.MinLrScale, _opts.MaxLrScale);
        }
        float lr = _opts.BaseLr * scale;

        // Batch = fresh windows + a deterministic slice of the replay ring (sampled
        // before the fresh windows are pushed, so it only ever holds PAST text).
        var replaySlice = TakeReplay(_opts.ReplayPerStep);
        var batch = new List<(int[] input, int[] target)>(fresh.Count + replaySlice.Count);
        batch.AddRange(fresh);
        batch.AddRange(replaySlice);

        for (int s = 0; s < _opts.Steps; s++)
            Trainer.Step(ps, batch, LossFn, _optimizer, lr, _opts.ClipNorm);

        float after = MeanLoss(fresh);

        // Push the fresh windows into the ring, and refresh the EMA.
        PushReplay(fresh);
        if (_opts.SurpriseEmaDecay > 0f)
            _surpriseAvg = _opts.SurpriseEmaDecay * _surpriseAvg + (1f - _opts.SurpriseEmaDecay) * before;

        _absorbCount++;
        RunGuardrail();

        return new ContinualStep(before, after, lr, replaySlice.Count);

        Tensor LossFn(int[] t, int[] y, bool training) => _llm.Model.Loss(t, y, ignoreIndex: -1, training);
    }

    // ── internals ──────────────────────────────────────────────────────────────

    private void RunGuardrail()
    {
        if (_opts.GuardrailEvery <= 0 || _yardstick is not { Count: > 0 }) return;
        if (_absorbCount % _opts.GuardrailEvery != 0) return;

        double now   = PerplexityEvaluator.Evaluate(_llm.Model, _yardstick).Perplexity;
        double drift  = (now - _baselinePerplexity) / _baselinePerplexity;
        if (drift > _opts.DriftThreshold)
            OnDrift?.Invoke(new ContinualDrift(_baselinePerplexity, now, drift, _absorbCount));
    }

    // Deterministic ring sample: up to `count` windows via a moving cursor,
    // skipping the `exclude` most-recently-added (the fresh windows just about to
    // be added — avoids trivially replaying what we're already training on).
    private List<(int[] input, int[] target)> TakeReplay(int count)
    {
        var slice = new List<(int[], int[])>();
        if (_opts.ReplayCapacity == 0 || count <= 0 || _replay.Count == 0) return slice;

        int available = _replay.Count;
        int take = Math.Min(count, available);
        for (int i = 0; i < take; i++)
        {
            _replayCursor = (_replayCursor + 1) % available;
            slice.Add(_replay[_replayCursor]);
        }
        return slice;
    }

    private void PushReplay(List<(int[] input, int[] target)> windows)
    {
        if (_opts.ReplayCapacity == 0) return;
        foreach (var w in windows)
        {
            if (_replay.Count < _opts.ReplayCapacity) _replay.Add(w);
            else
            {
                _replay[_replayCursor % _opts.ReplayCapacity] = w;   // overwrite oldest-ish, ring style
                _replayCursor = (_replayCursor + 1) % _opts.ReplayCapacity;
            }
        }
    }

    private List<(int[] input, int[] target)> Windows(string text) =>
        new StreamingTextDataset(text.Split('\n'), _llm.Tokenizer, _seqLen, _opts.Stride).Windows().ToList();

    private (int[] input, int[] target) WholeWindow(string text)
    {
        var ids = _llm.Tokenizer.Encode(text, addBos: true);
        if (ids.Count < 2) ids = [_llm.Tokenizer.Bos, .. ids, _llm.Tokenizer.Eos];
        int n = Math.Min(ids.Count - 1, _seqLen);
        var input = new int[n]; var target = new int[n];
        for (int i = 0; i < n; i++) { input[i] = ids[i]; target[i] = ids[i + 1]; }
        return (input, target);
    }

    private float MeanLoss(List<(int[] input, int[] target)> ws)
    {
        double sum = 0d;
        foreach (var (t, y) in ws) sum += _llm.Model.Loss(t, y, ignoreIndex: -1).Data[0];
        return (float)(sum / ws.Count);
    }
}
