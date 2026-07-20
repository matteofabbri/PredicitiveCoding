using System.Collections.Generic;
using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// PcDecoderTrainer — Incremental Predictive Coding (iPC) per un DecoderLM.
//
// Allena il transformer NON con backprop globale ma con Predictive Coding a
// confine-di-blocco: fra i blocchi vivono variabili latenti μ_b (lo "stato"
// del residual stream); ogni blocco predice il proprio latente dal precedente,
// e impara SOLO dal proprio errore di predizione locale ε_b = μ_b − f_b(μ_{b-1}).
//
// Il rilassamento dei latenti è LEGGERO (RelaxSteps, default 1 = iPC): l'equilibrio
// pieno del PC è instabile/biased sui transformer, mentre il rilassamento leggero
// resta vicino al feedforward e impara alla pari del backprop (verificato).
//
// Segni (derivati dall'energia F = Σ ½‖ε‖²):
//   rilassamento:  μ_b ← μ_b − η·(ε_b + topdown_b)     (F scende)
//   pesi blocco:   θ_b += (lr/seq)·VJP_{f_b}(ε_b)       (ogni blocco predice il suo latente)
// dove topdown_b è il messaggio dall'alto: il grad che atterra sul latente
// detachato passando l'errore del blocco sopra (o, in cima, il gradiente della
// cross-entropy). Embedding / unembedding / norm finale sono allenati dal
// gradiente della CE (la loro formulazione PC è a parte).
//
// v1: preset densi RMSNorm+RoPE (Small/Base), NumLoops=1, senza MoE/MTP/plan.
// L'"ancora" Adam periodica (che rimette in riga il PC) si aggancia sopra a parte.
// ─────────────────────────────────────────────────────────────────────────────
public sealed class PcDecoderTrainer
{
    private readonly DecoderLM _m;
    private readonly float _eps;

    /// <summary>Passi di rilassamento del latente per weight-update (1 = iPC, consigliato).</summary>
    public int RelaxSteps { get; init; } = 1;

    /// <summary>Learning-rate del rilassamento dei latenti.</summary>
    public float EtaState { get; init; } = 1.0f;

    /// <summary>Learning-rate degli update dei pesi (blocchi + mappe I/O).</summary>
    public float LrWeights { get; init; } = 1.5f;

    // ── Ancora Adam (opzionale) ─────────────────────────────────────────────
    // Ogni AnchorEvery step, un breve burst di backprop+Adam "rimette in riga"
    // il PC (il suo gradiente locale è biased: l'ancora ri-centra sulla loss
    // vera). m/v di Adam RE-INIZIALIZZATI a ogni scatto (niente stato stantio).
    /// <summary>Ogni quanti step un burst di ancora (0 = spenta).</summary>
    public int AnchorEvery { get; init; } = 0;
    /// <summary>Passi di Adam per scatto d'ancora.</summary>
    public int AnchorBurst { get; init; } = 3;
    /// <summary>Learning-rate dell'ancora Adam.</summary>
    public float AnchorLr { get; init; } = 0.01f;
    private long _steps;

    public PcDecoderTrainer(DecoderLM model)
    {
        var c = model.Config;
        if (c.NumLoops != 1 || c.NumExperts != 0 || c.MtpWeight != 0f || c.UsePlan || c.BlockPattern is not null)
            throw new NotSupportedException("PcDecoderTrainer v1: preset denso semplice (NumLoops=1, niente MoE/MTP/plan).");
        if (!c.UseRope)
            throw new NotSupportedException("PcDecoderTrainer v1 assume UseRope (posizioni in q/k).");
        if (!c.UseRmsNorm)
            throw new NotSupportedException("PcDecoderTrainer v1 assume UseRmsNorm.");
        _m = model;
        _eps = c.Eps;
    }

    private Tensor Fwd(Tensor x, int b) =>
        _m.Blocks[b].Forward(x, _m.AttnScale, 0f, training: false, _m.Rng, _eps, probe: null, b, auxLosses: null);

    /// <summary>
    /// Un passo di training iPC su una sequenza. Ritorna la cross-entropy misurata
    /// prima dell'aggiornamento delle mappe I/O (per il logging).
    /// </summary>
    public float Step(int[] tokens, int[] targets)
    {
        int B = _m.Blocks.Count, dim = _m.Config.Dim, L = tokens.Length;

        // Latenti: μ_0 = embedding, μ_b warm-start = feedforward.
        var mu = new float[B + 1][];
        mu[0] = (float[])Ag.GatherRows(_m.Wemb, tokens).Data.Clone();
        for (int b = 1; b <= B; b++)
            mu[b] = (float[])Fwd(Tensor.From(mu[b - 1], L, dim), b - 1).Data.Clone();

        // Rilassamento leggero dei latenti μ_1..μ_B.
        for (int t = 0; t < RelaxSteps; t++)
            for (int b = 1; b <= B; b++)
            {
                var fPrev = Fwd(Tensor.From(mu[b - 1], L, dim), b - 1).Data;
                var eps = new float[L * dim];
                for (int i = 0; i < eps.Length; i++) eps[i] = mu[b][i] - fPrev[i];

                float[] topdown;
                if (b < B)
                {
                    var hc = Tensor.From(mu[b], L, dim);
                    var f = Fwd(hc, b);
                    var eNext = new float[L * dim]; var fn = f.Data;
                    for (int i = 0; i < eNext.Length; i++) eNext[i] = mu[b + 1][i] - fn[i];
                    f.BackwardFrom(eNext);
                    topdown = (float[])hc.Grad.Clone();
                }
                else
                {
                    var hc = Tensor.From(mu[b], L, dim);
                    var loss = Ag.CrossEntropy(Ag.MatMul(Ag.RmsNorm(hc, _m.LnFG, _eps), _m.Wout), targets, -1);
                    loss.Backward();
                    topdown = (float[])hc.Grad.Clone();
                }
                for (int i = 0; i < mu[b].Length; i++) mu[b][i] -= EtaState * (eps[i] + topdown[i]);
            }

        // Update pesi dei blocchi: θ_b += (lr/L)·VJP(ε_b).
        for (int b = 1; b <= B; b++)
        {
            var f = Fwd(Tensor.From(mu[b - 1], L, dim), b - 1);
            var eps = new float[L * dim];
            for (int i = 0; i < eps.Length; i++) eps[i] = mu[b][i] - f.Data[i];
            f.BackwardFrom(eps);
            foreach (var p in _m.Blocks[b - 1].Parameters())
                for (int i = 0; i < p.Data.Length; i++) p.Data[i] += (LrWeights / L) * p.Grad[i];
        }

        // Mappe I/O (embedding, unembedding, norm finale) via gradiente CE.
        foreach (var p in _m.Parameters()) Array.Clear(p.Grad);
        var ce = _m.Loss(tokens, targets, ignoreIndex: -1);
        ce.Backward();
        UpdateIo(_m.Wemb); UpdateIo(_m.Wout); UpdateIo(_m.LnFG);

        _steps++;
        if (AnchorEvery > 0 && _steps % AnchorEvery == 0) Anchor(tokens, targets);
        return ce.Data[0];
    }

    // L'ancora: un breve burst di backprop+Adam sulla finestra corrente, che
    // ri-centra i pesi sulla loss vera (il gradiente PC è biased al prim'ordine).
    private void Anchor(int[] tokens, int[] targets)
    {
        var opt = new Adam();                                   // m/v freschi ad ogni scatto
        var win = new List<(int[] input, int[] target)> { (tokens, targets) };
        for (int k = 0; k < AnchorBurst; k++)
            Trainer.Step(_m.Parameters(), win, (t, y, tr) => _m.Loss(t, y, ignoreIndex: -1, tr), opt, AnchorLr, clipNorm: 1f);
    }

    private void UpdateIo(Tensor p)
    {
        for (int i = 0; i < p.Data.Length; i++) p.Data[i] -= LrWeights * p.Grad[i];
    }
}
