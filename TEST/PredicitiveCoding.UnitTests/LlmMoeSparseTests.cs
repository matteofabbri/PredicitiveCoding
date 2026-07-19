using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Eval;

namespace PredicitiveCoding.UnitTests;

// FLOP-sparse MoE execution (the post-Phase-6 opening): each expert runs only
// on the rows routed to it via Ag.GatherRows / Ag.ScatterRows, instead of the
// masked-dense "every expert computes every row" formulation. Certified here:
// the two formulations are bit-identical (forward AND gradients), the new ops
// pass a numeric gradient check, and the probe counts exactly rows·topK row
// executions per MoE block.
public class LlmMoeSparseTests
{
    private static DecoderLmConfig Cfg(int experts, int topK, int blocks = 1) =>
        new(13, 8, NumBlocks: blocks, NumHeads: 1, FfDim: 16, MaxSeq: 8,
            NumExperts: experts, MoeTopK: topK);

    // ── The gather/scatter ops earn their gradients ──────────────────────────

    [Fact]
    public void GatherScatter_PassNumericGradientCheck()
    {
        const int M = 5, N = 3;
        int[] gatherIdx  = [0, 2, 2, 4];       // duplicate: gather backward must accumulate
        int[] scatterIdx = [1, 0, 3, 2];
        int[] targets    = [0, 2, 1, 0, 2];
        var rng  = new Random(17);
        var xRaw = new float[M * N];
        for (int i = 0; i < xRaw.Length; i++) xRaw[i] = (float)(rng.NextDouble() * 2 - 1);

        (Tensor x, Tensor loss) Build()
        {
            var x   = Tensor.From(xRaw, M, N);
            var sub = Ag.GatherRows(x, gatherIdx);                        // [4, N]
            var y   = Ag.ScatterRows(Ag.Gelu(sub), scatterIdx, M);        // [M, N]
            return (x, Ag.Add(Ag.CrossEntropy(y, targets), Ag.Mean(x)));
        }

        var (tx, loss) = Build();
        loss.Backward();

        for (int k = 0; k < xRaw.Length; k++)
        {
            const float eps = 1e-3f;
            float keep = xRaw[k];
            xRaw[k] = keep + eps; float up   = Build().loss.Data[0];
            xRaw[k] = keep - eps; float down = Build().loss.Data[0];
            xRaw[k] = keep;

            float numeric = (up - down) / (2 * eps);
            Assert.True(MathF.Abs(numeric - tx.Grad[k]) <= 1e-2f * MathF.Max(1f, MathF.Abs(numeric)),
                $"x[{k}]: numeric {numeric:F4} vs autograd {tx.Grad[k]:F4}");
        }
    }

    [Fact]
    public void ScatterRows_DuplicateIndices_SumLikeTheBackward()
    {
        var src = Tensor.From([1f, 2f, 10f, 20f], 2, 2);
        var o   = Ag.ScatterRows(src, [1, 1], 3);
        Assert.Equal([0f, 0f, 11f, 22f, 0f, 0f], o.Data);
    }

    // ── Sparse execution ≡ masked-dense, bit for bit ─────────────────────────

    [Fact]
    public void SparseMoe_MatchesMaskedDense_ForwardAndGradients()
    {
        const int M = 6, D = 4, F = 8, E = 3, TopK = 1;
        int[] targets = [0, 2, 1, 3, 0, 2];
        var rng = new Random(29);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }

        var h2Raw = Rand(M * D);
        var rRaw  = Rand(D * E);
        var w1Raw = new float[E][]; var b1Raw = new float[E][];
        var w2Raw = new float[E][]; var b2Raw = new float[E][];
        for (int e = 0; e < E; e++)
        {
            w1Raw[e] = Rand(D * F); b1Raw[e] = Rand(F);
            w2Raw[e] = Rand(F * D); b2Raw[e] = Rand(D);
        }

        // Builds either formulation on fresh leaves over the SAME raw values.
        (Tensor h2, Tensor r, Tensor[] w1, Tensor[] w2, Tensor loss) Build(bool sparse)
        {
            var h2 = Tensor.From(h2Raw, M, D);
            var r  = Tensor.From(rRaw, D, E);
            var w1 = new Tensor[E]; var b1 = new Tensor[E];
            var w2 = new Tensor[E]; var b2 = new Tensor[E];
            for (int e = 0; e < E; e++)
            {
                w1[e] = Tensor.From(w1Raw[e], D, F); b1[e] = Tensor.From(b1Raw[e], 1, F);
                w2[e] = Tensor.From(w2Raw[e], F, D); b2[e] = Tensor.From(b2Raw[e], 1, D);
            }

            var gates = Ag.SoftmaxRows(Ag.MatMul(h2, r));
            var mask  = TopKMask(gates.Data, M, E, TopK);
            var kept  = Ag.Mul(gates, mask);

            Tensor? outSum = null;
            for (int e = 0; e < E; e++)
            {
                Tensor contribution;
                if (sparse)
                {
                    var routed = RoutedRows(mask.Data, M, E, e);
                    if (routed.Length == 0) continue;
                    var sub     = Ag.GatherRows(h2, routed);
                    var expert  = Ag.AddBias(Ag.MatMul(Ag.Gelu(Ag.AddBias(Ag.MatMul(sub, w1[e]), b1[e])), w2[e]), b2[e]);
                    var gateCol = Ag.GatherRows(Ag.Column(kept, e), routed);
                    contribution = Ag.ScatterRows(Ag.ScaleRows(expert, gateCol), routed, M);
                }
                else
                {
                    var expert = Ag.AddBias(Ag.MatMul(Ag.Gelu(Ag.AddBias(Ag.MatMul(h2, w1[e]), b1[e])), w2[e]), b2[e]);
                    contribution = Ag.ScaleRows(expert, Ag.Column(kept, e));
                }
                outSum = outSum is null ? contribution : Ag.Add(outSum, contribution);
            }

            return (h2, r, w1, w2, Ag.CrossEntropy(outSum!, targets));
        }

        var dense  = Build(sparse: false);
        var lean   = Build(sparse: true);
        dense.loss.Backward();
        lean.loss.Backward();

        Assert.Equal(dense.loss.Data[0], lean.loss.Data[0]);
        Assert.Equal(dense.h2.Grad, lean.h2.Grad);
        Assert.Equal(dense.r.Grad, lean.r.Grad);
        for (int e = 0; e < E; e++)
        {
            Assert.Equal(dense.w1[e].Grad, lean.w1[e].Grad);
            Assert.Equal(dense.w2[e].Grad, lean.w2[e].Grad);
        }
    }

    // ── The probe certifies the FLOP saving: rows·topK, never rows·E ─────────

    [Fact]
    public void Probe_CountsExactlyRowsTimesTopK_PerMoeBlock()
    {
        int[] tokens = [1, 2, 3, 4, 5];
        var probe = new ActivationProbe();

        new DecoderLM(Cfg(experts: 4, topK: 2), new Random(11)).Forward(tokens, probe: probe);
        Assert.Equal(5L * 2, probe.MoeRowsComputed);            // not 5·4: two experts never ran

        probe.Reset();
        Assert.Equal(0L, probe.MoeRowsComputed);

        new DecoderLM(Cfg(experts: 4, topK: 2, blocks: 2), new Random(11)).Forward(tokens, probe: probe);
        Assert.Equal(5L * 2 * 2, probe.MoeRowsComputed);

        probe.Reset();
        new DecoderLM(Cfg(experts: 0, topK: 1), new Random(11)).Forward(tokens, probe: probe);
        Assert.Equal(0L, probe.MoeRowsComputed);                // dense FFN never touches the meter
    }

    // ── Test-local replicas of the Block's private routing helpers ───────────

    private static Tensor TopKMask(float[] gates, int rows, int experts, int topK)
    {
        var mask = new Tensor(rows, experts);
        for (int i = 0; i < rows; i++)
            for (int chosen = 0; chosen < topK; chosen++)
            {
                int best = -1; float bestVal = float.NegativeInfinity;
                for (int j = 0; j < experts; j++)
                {
                    if (mask.Data[i * experts + j] != 0f) continue;
                    if (gates[i * experts + j] > bestVal) { bestVal = gates[i * experts + j]; best = j; }
                }
                mask.Data[i * experts + best] = 1f;
            }
        return mask;
    }

    private static int[] RoutedRows(float[] mask, int rows, int experts, int expert)
    {
        var routed = new List<int>();
        for (int i = 0; i < rows; i++) if (mask[i * experts + expert] != 0f) routed.Add(i);
        return [.. routed];
    }
}
