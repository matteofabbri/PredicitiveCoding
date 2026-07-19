using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Phase 6 of PredictiveCoding.LLM/ROADMAP.md: sparse mixture-of-experts FFN —
// router → softmax gates → hard top-k → gate-weighted expert sum, plus the
// Switch-style load-balancing loss and active-parameter accounting.
public class LlmMoeTests
{
    private static DecoderLmConfig Cfg(int experts = 0, int topK = 1, float balance = 0.01f) =>
        new(13, 8, NumBlocks: 1, NumHeads: 1, FfDim: 16, MaxSeq: 8,
            NumExperts: experts, MoeTopK: topK, MoeBalanceWeight: balance);

    // ── Off means off ────────────────────────────────────────────────────────

    [Fact]
    public void MoeOff_IsBitIdenticalToVanilla()
    {
        var a = new DecoderLM(Cfg(), new Random(3));
        var b = new DecoderLM(Cfg(experts: 0), new Random(3));

        Assert.Equal(a.ParameterCount, b.ParameterCount);
        Assert.Equal(a.ParameterCount, a.ActiveParameterCountPerToken);   // dense: everything is active

        var la = a.Forward([1, 2, 3]);
        var lb = b.Forward([1, 2, 3]);
        for (int i = 0; i < la.Data.Length; i++)
            Assert.Equal(la.Data[i], lb.Data[i]);
    }

    [Fact]
    public void InvalidMoeConfig_Throws()
    {
        Assert.Throws<ArgumentException>(() => Cfg(experts: 4, topK: 0).Validate());
        Assert.Throws<ArgumentException>(() => Cfg(experts: 4, topK: 5).Validate());
        Assert.Throws<ArgumentException>(() => Cfg(experts: -1).Validate());
    }

    // ── Active-parameter accounting (the Step-3.5-Flash metric) ─────────────

    [Fact]
    public void ActiveParams_CountOnlyTheRoutedExperts()
    {
        var cfg = Cfg(experts: 4, topK: 1);
        var lm  = new DecoderLM(cfg, new Random(5));

        long perExpert = 8 * 16 + 16 + 16 * 8 + 8;                        // w1 + b1 + w2 + b2
        Assert.Equal(lm.ParameterCount - 1 * (4 - 1) * perExpert, lm.ActiveParameterCountPerToken);
        Assert.True(lm.ActiveParameterCountPerToken < lm.ParameterCount);

        var all = new DecoderLM(Cfg(experts: 4, topK: 4), new Random(5));
        Assert.Equal(all.ParameterCount, all.ActiveParameterCountPerToken);   // k = E ⇒ everything active
    }

    // ── The balance loss has an exact fixed point ────────────────────────────

    [Fact]
    public void UniformRouter_BalanceLossContributesExactlyOnePerBlock()
    {
        // All-zero parameters ⇒ uniform gates AND uniform logits, so:
        //   loss = ln(vocab) + balanceWeight · (1.0 per MoE block).
        var cfg = Cfg(experts: 4, topK: 1, balance: 0.5f);
        var lm  = new DecoderLM(cfg, new Random(7));
        foreach (var p in lm.Parameters()) Array.Clear(p.Data);

        float loss     = lm.Loss([1, 2, 3, 4], [2, 3, 4, 5], ignoreIndex: -1).Data[0];
        float expected = MathF.Log(13f) + 0.5f * 1f;
        Assert.Equal(expected, loss, precision: 3);

        // Same model with the balance term disabled: pure uniform CE.
        var off = new DecoderLM(Cfg(experts: 4, topK: 1, balance: 0f), new Random(7));
        foreach (var p in off.Parameters()) Array.Clear(p.Data);
        Assert.Equal(MathF.Log(13f), off.Loss([1, 2, 3, 4], [2, 3, 4, 5], ignoreIndex: -1).Data[0], precision: 3);
    }

    // ── It learns ────────────────────────────────────────────────────────────

    [Fact]
    public void MoeModel_Training_ReducesLoss()
    {
        var lm  = new DecoderLM(Cfg(experts: 4, topK: 1), new Random(4));
        var opt = new Adam();
        int[] toks = [1, 2, 3, 4, 5], tgts = [2, 3, 4, 5, 6];

        float first = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];
        for (int it = 0; it < 150; it++)
        {
            lm.Loss(toks, tgts, ignoreIndex: -1, training: true).Backward();
            opt.Step(lm.Parameters(), 0.01f);
        }
        float last = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];

        Assert.True(last < first * 0.5f, $"MoE model did not learn: {first:F3} → {last:F3}");
    }

    // ── FLOP sparsity: skipped experts really cost nothing ───────────────────

    [Fact]
    public void FlopSparse_RunsOnlyTheRoutedRows()
    {
        var probe = new PredictiveCoding.LLM.Eval.ActivationProbe();
        int[] tokens = [1, 5, 9, 2];

        // Top-1 of 4 experts: each row runs exactly one expert FFN.
        var k1 = new DecoderLM(Cfg(experts: 4, topK: 1), new Random(11));
        k1.Forward(tokens, training: false, probe);
        Assert.Equal(4, probe.MoeRowsComputed);                   // rows·K, not rows·E (16)

        // k = E: every expert runs on every row — dense again, by choice.
        probe.Reset();
        var kAll = new DecoderLM(Cfg(experts: 4, topK: 4), new Random(11));
        kAll.Forward(tokens, training: false, probe);
        Assert.Equal(4 * 4, probe.MoeRowsComputed);

        probe.Reset();
        var k2 = new DecoderLM(Cfg(experts: 4, topK: 2), new Random(11));
        k2.Forward(tokens, training: false, probe);
        Assert.Equal(4 * 2, probe.MoeRowsComputed);
    }

    [Fact]
    public void GatherScatter_PassNumericGradientCheck()
    {
        const int Rows = 4, Cols = 3;
        int[] routed  = [2, 0, 3];                                // out of order on purpose
        int[] targets = [0, 2, 1, 0];
        var rng  = new Random(17);
        var xRaw = new float[Rows * Cols];
        for (int i = 0; i < xRaw.Length; i++) xRaw[i] = (float)(rng.NextDouble() * 2 - 1);

        // loss = CE( Scatter(Gather(x, routed), routed, Rows), targets ):
        // routed rows pass through, the rest are zeros.
        (Tensor x, Tensor loss) Build()
        {
            var x = Tensor.From(xRaw, Rows, Cols);
            var y = Ag.ScatterRows(Ag.GatherRows(x, routed), routed, Rows);
            return (x, Ag.CrossEntropy(y, targets));
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

    // ── The router ops earn their gradients ──────────────────────────────────

    [Fact]
    public void RouterOps_PassNumericGradientCheck()
    {
        const int M = 4, N = 3, E = 2;
        int[] targets = [0, 2, 1, 0];
        var rng = new Random(13);
        float[] xRaw = new float[M * N], rRaw = new float[M * E];
        for (int i = 0; i < xRaw.Length; i++) xRaw[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < rRaw.Length; i++) rRaw[i] = (float)(rng.NextDouble() * 2 - 1);

        // loss = CE( ScaleRows(x, Column(SoftmaxRows(r), 0)), targets ) + Mean(x)
        (Tensor x, Tensor r, Tensor loss) Build()
        {
            var x = Tensor.From(xRaw, M, N);
            var r = Tensor.From(rRaw, M, E);
            var scaled = Ag.ScaleRows(x, Ag.Column(Ag.SoftmaxRows(r), 0));
            return (x, r, Ag.Add(Ag.CrossEntropy(scaled, targets), Ag.Mean(x)));
        }

        var (tx, tr, loss) = Build();
        loss.Backward();

        foreach (var (raw, grads, name) in new[] { (xRaw, tx.Grad, "x"), (rRaw, tr.Grad, "r") })
        {
            for (int k = 0; k < raw.Length; k += 3)
            {
                const float eps = 1e-3f;
                float keep = raw[k];
                raw[k] = keep + eps; float up   = Build().loss.Data[0];
                raw[k] = keep - eps; float down = Build().loss.Data[0];
                raw[k] = keep;

                float numeric = (up - down) / (2 * eps);
                Assert.True(MathF.Abs(numeric - grads[k]) <= 1e-2f * MathF.Max(1f, MathF.Abs(numeric)),
                    $"{name}[{k}]: numeric {numeric:F4} vs autograd {grads[k]:F4}");
            }
        }
    }
}
