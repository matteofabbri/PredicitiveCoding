using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// RecurrentTemporalLayer  —  the O(1)-cache inference twin of the trainable
// TemporalBlock (roadmap Phase 5).
//
//   a = σ(Wa·x + ba)    decay gate
//   i = σ(Wi·x + bi)    input gate
//   h ← a ⊙ h + i ⊙ (Wu·x + bu)
//   out = Wo·h + bo
//
// Where attention drags a growing transcript (the KV cache), this layer keeps a
// single fixed-size state vector — the predictive-coding way: persistent state,
// not a replay buffer. Interleaved with attention via
// TransformerConfig.BlockPattern (Nemotron 3 / Mamba-hybrid style).
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Fixed-size recurrent state: the whole "cache" is one vector.</summary>
public sealed class RecurrentStateCache : ILayerCache
{
    public float[] State  { get; }
    public int     Length { get; private set; }   // steps absorbed so far

    public RecurrentStateCache(int dim) => State = new float[dim];
    private RecurrentStateCache(float[] state, int length) { State = state; Length = length; }

    public long BytesUsed => (long)State.Length * sizeof(float);   // constant, by construction

    public void NoteStep() => Length++;

    public ILayerCache Clone() => new RecurrentStateCache((float[])State.Clone(), Length);
}

/// <summary>Gated linear recurrence as a drop-in ISelfAttention replacement.</summary>
public sealed class RecurrentTemporalLayer : ISelfAttention
{
    private readonly Linear _wa, _wi, _wu, _wo;
    private readonly int    _dim;

    internal Linear Wa => _wa;   // serving-path import hooks (ROADMAP-Pro Phase 1)
    internal Linear Wi => _wi;
    internal Linear Wu => _wu;
    internal Linear Wo => _wo;

    public RecurrentTemporalLayer(TransformerConfig cfg, Random rng)
    {
        _dim = cfg.EmbedDim;
        _wa  = new Linear(_dim, _dim, rng);
        _wi  = new Linear(_dim, _dim, rng);
        _wu  = new Linear(_dim, _dim, rng);
        _wo  = new Linear(_dim, _dim, rng);
    }

    public ILayerCache CreateCache() => new RecurrentStateCache(_dim);

    public DenseTensor<float> Forward(DenseTensor<float> x, ILayerCache cacheBase, int position)
    {
        var cache = (RecurrentStateCache)cacheBase;
        var h     = cache.State;

        float[] a = _wa.Forward(x).Buffer;
        float[] i = _wi.Forward(x).Buffer;
        float[] u = _wu.Forward(x).Buffer;

        var mixed = new DenseTensor<float>(_dim);
        for (int j = 0; j < _dim; j++)
        {
            float aj = 1f / (1f + MathF.Exp(-a[j]));
            float ij = 1f / (1f + MathF.Exp(-i[j]));
            h[j] = aj * h[j] + ij * u[j];
            mixed.Buffer[j] = h[j];
        }
        cache.NoteStep();

        return _wo.Forward(mixed);
    }
}
