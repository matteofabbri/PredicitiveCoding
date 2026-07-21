using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Model;

namespace PredictiveCoding.LLM.Serving;

// ─────────────────────────────────────────────────────────────────────────────
// GptExporter  —  train on DecoderLM, serve on GptModel (ROADMAP-Pro Phase 1).
//
// The trainable stack re-runs the whole sequence per generated token (O(n²·d));
// the inference stack decodes incrementally through the KV cache (O(n·d)) and
// carries every Phase-4 efficiency feature. This exporter maps the trained
// weights 1:1 onto the inference stack:
//
//   wemb (+ absorbed plan)      → Embedding table
//   per-head Wq/Wk/Wv/Wo        → stacked Q/K/V/O Linears (transposed)
//   TemporalBlock gates         → RecurrentTemporalLayer Linears
//   MoE router + experts        → MoeFeedForward (identical gate semantics)
//   LayerNorms, FFNs, LM head   → their inference twins
//
// Declared subset: NumLoops must be 1 (the served stack is single-pass); the
// t+2 head is skipped (serving decodes greedily off the main head — speculative
// serving arrives with Phase 2's sampling rework); the latent plan is absorbed
// into the embedding table (it was an additive conditioning on every position).
//
// The parameter walk is positional over DecoderLM.Parameters() — guarded twice:
// every tensor's shape is checked against expectations, and the walk must end
// exactly at the last parameter, so any layout drift fails loudly.
// ─────────────────────────────────────────────────────────────────────────────

public static class GptExporter
{
    public static GptModel Export(DecoderLM model)
    {
        var c = model.Config;
        if (c.NumLoops > 1)
            throw new NotSupportedException(
                "Looped models (NumLoops > 1) cannot be exported: the serving stack is single-pass. " +
                "Train with NumLoops = 1 for serving (declared out in ROADMAP-Pro Phase 1).");

        var cfg = new TransformerConfig
        {
            VocabSize = c.Vocab, MaxSeqLen = c.MaxSeq, EmbedDim = c.Dim, NumHeads = c.NumHeads,
            NumLayers = c.NumBlocks, FeedForwardDim = c.FfDim, LayerNormEps = c.Eps,
            TieEmbeddings = false,
            // The block family comes FROM the trained config — the Scale presets
            // (and CreatePc) are Llama-style, and a served stack hard-wired to
            // LayerNorm + sinusoidal would silently compute a different model.
            Positional = c.UseRope ? PositionalKind.Rotary : PositionalKind.Sinusoidal,
            Norm = c.UseRmsNorm ? NormKind.RmsNorm : NormKind.LayerNorm,
            BlockPattern = c.BlockPattern, NumExperts = c.NumExperts, MoeTopK = c.MoeTopK,
        };
        // RMSNorm carries a gamma and NO beta, so the parameter walk below must
        // consume one tensor per norm instead of two — get this wrong and every
        // subsequent Next() reads the neighbouring weight.
        bool rms = c.UseRmsNorm;
        var gpt = new GptModel(cfg, new Random(0));   // random init, every weight overwritten below

        var ps = model.Parameters();
        int cursor = 0;
        Tensor Next(int rows, int cols, string what)
        {
            if (cursor >= ps.Length)
                throw new InvalidOperationException($"Exporter ran out of parameters at '{what}' — layout drift.");
            var t = ps[cursor++];
            if (t.Rows != rows || t.Cols != cols)
                throw new InvalidOperationException(
                    $"Exporter shape mismatch at '{what}': expected [{rows},{cols}], found [{t.Rows},{t.Cols}].");
            return t;
        }

        int dim = c.Dim, hd = c.HeadDim, ff = c.FfDim, heads = c.NumHeads;

        var wemb = Next(c.Vocab, dim, "wemb");

        for (int b = 0; b < c.NumBlocks; b++)
        {
            var block    = gpt.Blocks[b];
            bool temporal = c.BlockPattern is { } p && p[b] == 'T';

            CopyNorm(block.Ln1, Next(1, dim, "ln1G"), rms ? null : Next(1, dim, "ln1B"));

            if (temporal)
            {
                var rec = (RecurrentTemporalLayer)block.Attn;
                CopyLinear(rec.Wa, Next(dim, dim, "wa"), Next(1, dim, "ba"));
                CopyLinear(rec.Wi, Next(dim, dim, "wi"), Next(1, dim, "bi"));
                CopyLinear(rec.Wu, Next(dim, dim, "wu"), Next(1, dim, "bu"));
                CopyLinear(rec.Wo, Next(dim, dim, "wo"), Next(1, dim, "bo"));
            }
            else
            {
                var attn = (MultiHeadAttention)block.Attn;
                var qW = attn.Q.WeightTensor.Buffer;   // [H·hd, dim]
                var kW = attn.K.WeightTensor.Buffer;
                var vW = attn.V.WeightTensor.Buffer;
                var oW = attn.O.WeightTensor.Buffer;   // [dim, H·hd]
                for (int h = 0; h < heads; h++)
                {
                    var wq = Next(dim, hd, "wq"); var wk = Next(dim, hd, "wk");
                    var wv = Next(dim, hd, "wv"); var wo = Next(hd, dim, "wo");
                    for (int j = 0; j < hd; j++)
                        for (int i = 0; i < dim; i++)
                        {
                            int row = (h * hd + j) * dim + i;
                            qW[row] = wq.Data[i * hd + j];         // [dim,hd] → transposed slice
                            kW[row] = wk.Data[i * hd + j];
                            vW[row] = wv.Data[i * hd + j];
                        }
                    for (int d = 0; d < dim; d++)
                        for (int j = 0; j < hd; j++)
                            oW[d * (heads * hd) + h * hd + j] = wo.Data[j * dim + d];
                }
                // Q/K/V/O biases stay zero — the trainable heads have none.
            }

            CopyNorm(block.Ln2, Next(1, dim, "ln2G"), rms ? null : Next(1, dim, "ln2B"));

            if (c.NumExperts > 0 && !temporal)
            {
                var moe = (MoeFeedForward)block.Ffn;
                CopyLinear(moe.Router, Next(dim, c.NumExperts, "router"), bias: null);
                for (int e = 0; e < c.NumExperts; e++)
                {
                    CopyLinear(moe.Experts[e].Fc1, Next(dim, ff, "ew1"), Next(1, ff, "eb1"));
                    CopyLinear(moe.Experts[e].Fc2, Next(ff, dim, "ew2"), Next(1, dim, "eb2"));
                }
            }
            else
            {
                var ffn = (FeedForward)block.Ffn;
                CopyLinear(ffn.Fc1, Next(dim, ff, "w1"), Next(1, ff, "b1"));
                CopyLinear(ffn.Fc2, Next(ff, dim, "w2"), Next(1, dim, "b2"));
            }
        }

        CopyNorm(gpt.FinalNorm, Next(1, dim, "lnFG"), rms ? null : Next(1, dim, "lnFB"));
        CopyLinear(gpt.LmHead!, Next(dim, c.Vocab, "wout"), bias: null);

        if (c.MtpWeight > 0f) Next(dim, c.Vocab, "woutMtp");        // main head serves; t+2 head skipped
        float[]? plan = c.UsePlan ? Next(1, dim, "plan").Data : null;

        if (cursor != ps.Length)
            throw new InvalidOperationException(
                $"Exporter left {ps.Length - cursor} parameter(s) unconsumed — layout drift.");

        // Embedding table = wemb rows, with the additive plan absorbed into
        // every row (the plan conditioned every position identically).
        var table = gpt.EmbeddingTable.Table.Buffer;
        for (int t = 0; t < c.Vocab; t++)
            for (int i = 0; i < dim; i++)
                table[t * dim + i] = wemb.Data[t * dim + i] + (plan?[i] ?? 0f);

        return gpt;
    }

    // Trainable convention y = x·W with W [in, out]; Linear computes y = W'·x
    // with W' [out, in] — copy transposed, and the bias when the trainable side
    // has one (else the Linear's zero bias stays).
    private static void CopyLinear(Linear lin, Tensor w, Tensor? bias)
    {
        int inDim = w.Rows, outDim = w.Cols;
        if (lin.InputDim != inDim || lin.OutputDim != outDim)
            throw new InvalidOperationException(
                $"Linear shape mismatch: exporter [{outDim},{inDim}] vs layer [{lin.OutputDim},{lin.InputDim}].");
        var dst = lin.WeightTensor.Buffer;
        for (int o = 0; o < outDim; o++)
            for (int i = 0; i < inDim; i++)
                dst[o * inDim + i] = w.Data[i * outDim + o];
        if (bias is not null)
            bias.Data.AsSpan(0, outDim).CopyTo(lin.BiasTensor.Buffer);
    }

    // The trainable side has a beta only under LayerNorm; RMSNorm scales and
    // nothing else. A mismatch between the two sides means the walk went out of
    // phase — say so here rather than importing a weight into the wrong slot.
    private static void CopyNorm(INormLayer norm, Tensor gamma, Tensor? beta)
    {
        switch (norm)
        {
            case RmsNorm rn when beta is null:
                gamma.Data.AsSpan().CopyTo(rn.Gamma);
                break;
            case LayerNorm ln when beta is not null:
                gamma.Data.AsSpan().CopyTo(ln.Gamma);
                beta.Data.AsSpan().CopyTo(ln.Beta);
                break;
            default:
                throw new InvalidOperationException(
                    $"Norm import mismatch: serving layer is {norm.GetType().Name} but the exporter " +
                    $"{(beta is null ? "found no beta" : "found a beta")} — trained and served configs disagree.");
        }
    }
}
