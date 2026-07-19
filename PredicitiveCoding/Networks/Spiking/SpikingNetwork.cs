using PredictiveCoding.Core;

namespace PredictiveCoding.Networks.Spiking;

// ─────────────────────────────────────────────────────────────────────────────
// SpikingNetwork  —  hierarchical predictive coding with leaky integrate-and-fire
//
// A self-contained, in-memory spiking PCN (no FasterKV persistence — spiking
// state is transient by nature).  Each layer holds a population of LIF neurons
// whose membrane potential V_l integrates two predictive-coding drives:
//
//   pred_{l-1} = f(W_l r_l + b_l)              top-down generative prediction
//   ε_{l-1}    = r_{l-1} − pred_{l-1}          error as a firing-rate difference
//   drive_l    = W_lᵀ (ε_{l-1} ⊙ f'(·)) − ε_l  bottom-up minus own error
//   V_l       += Δt·(−λ V_l + η_V · drive_l)   leaky integration
//   spike, reset, then r_l ← ρ r_l + (1−ρ)·spike
//
// After the membrane dynamics settle, weights follow the local Hebbian PC rule
//   ΔW_l = η_W · (ε_{l-1} ⊙ f'(·)) ⊗ r_l,   Δb_l = η_W · (ε_{l-1} ⊙ f'(·))
// so learning stays fully local (no backprop), as required for neuromorphic
// implementation.  Classification is energy-based, identical to the rate model:
//   c* = argmin_c F(x, one-hot(c)),  F = Σ_l ½‖ε_l‖².
//
// References:
//   Boerlin, Machens & Denève (2013); Rao & Ballard (1999); Friston (2005).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class SpikingNetwork
{
    private readonly SpikingNetworkConfig   _config;
    private readonly int                    _L;
    private readonly DenseTensor<float>[]   _w;   // _w[l] : [size_{l-1}, size_l]  for l = 1..L-1
    private readonly DenseTensor<float>[]   _b;   // _b[l] : [size_{l-1}]          for l = 1..L-1

    public SpikingNetworkConfig Configuration => _config;
    public int InputDim  => _config.LayerSizes[0];
    public int OutputDim => _config.LayerSizes[^1];

    public SpikingNetwork(SpikingNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        _L = config.LayerSizes.Length;

        _w = new DenseTensor<float>[_L];
        _b = new DenseTensor<float>[_L];
        for (int l = 1; l < _L; l++)
        {
            int fanIn  = config.LayerSizes[l];      // r_l
            int fanOut = config.LayerSizes[l - 1];  // pred for layer below
            _w[l] = RandomUtils.InitWeights(fanIn, fanOut, config.Activation, rng); // [fanOut, fanIn]
            _b[l] = DenseTensor<float>.Zeros(fanOut);
        }
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs the spiking dynamics with the top layer free and returns the settled
    /// output-layer firing rates r_{L-1}.  Transient — no weights are modified.
    /// </summary>
    public DenseTensor<float> Infer(DenseTensor<float> input)
    {
        var r = Settle(input, clampTop: false, label: null);
        return r[_L - 1];
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// One supervised learning step: clamp input and label rates, settle the
    /// spiking dynamics, then apply the local Hebbian weight/bias updates.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label)
    {
        CheckSize(input, InputDim, nameof(input));
        CheckSize(label, OutputDim, nameof(label));

        var r = Settle(input, clampTop: true, label: label);

        float eta = _config.WeightLearningRate;
        for (int l = 1; l < _L; l++)
        {
            var (delta, _) = LayerErrorAndPrediction(r, l);
            // ΔW_l = η · δ ⊗ r_l ;  Δb_l = η · δ
            var dw = TensorUtils.OuterProduct(delta, r[l]);
            TensorUtils.InPlaceAxpy(eta, dw, _w[l]);
            TensorUtils.InPlaceAxpy(eta, delta, _b[l]);
        }
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Energy-based classification: c* = argmin_c F(x, one-hot(c)).
    /// </summary>
    public int Predict(DenseTensor<float> input)
    {
        CheckSize(input, InputDim, nameof(input));

        float bestEnergy = float.MaxValue;
        int   bestClass  = 0;
        var   oneHot     = DenseTensor<float>.Zeros(OutputDim);

        for (int c = 0; c < OutputDim; c++)
        {
            oneHot.Zero();
            oneHot.Buffer[c] = 1f;
            var   r = Settle(input, clampTop: true, label: oneHot);
            float f = Energy(r);
            if (f < bestEnergy) { bestEnergy = f; bestClass = c; }
        }
        return bestClass;
    }

    /// <summary>Free energy F = Σ_l ½‖ε_l‖² after settling with the top layer free.</summary>
    public float InferEnergy(DenseTensor<float> input)
        => Energy(Settle(input, clampTop: false, label: null));

    // ── Core spiking dynamics ──────────────────────────────────────────────────

    /// <summary>
    /// Integrates the LIF membrane dynamics for the configured number of ticks and
    /// returns the settled firing rates per layer.
    /// </summary>
    private DenseTensor<float>[] Settle(DenseTensor<float> input, bool clampTop, DenseTensor<float>? label)
    {
        CheckSize(input, InputDim, nameof(input));

        var V = new DenseTensor<float>[_L];
        var r = new DenseTensor<float>[_L];
        for (int l = 0; l < _L; l++)
        {
            V[l] = DenseTensor<float>.Zeros(_config.LayerSizes[l]);
            r[l] = DenseTensor<float>.Zeros(_config.LayerSizes[l]);
        }

        float dt   = _config.Dt;
        float leak = _config.Leak;
        float gain = _config.DriveRate;
        float thr  = _config.Threshold;
        float rho  = _config.RateDecay;

        for (int tick = 0; tick < _config.Ticks; tick++)
        {
            // Clamp boundary rates (input always; label when supervised).
            input.CopyTo(r[0]);
            if (clampTop && label != null) label.CopyTo(r[_L - 1]);

            // Snapshot all errors/deltas from the current rates (Jacobi update).
            var deltas = new DenseTensor<float>[_L]; // deltas[l] lives on layer l-1
            var errs   = new DenseTensor<float>[_L]; // errs[l] = ε_l (layer l's own error)
            for (int l = 0; l < _L; l++) errs[l] = DenseTensor<float>.Zeros(_config.LayerSizes[l]);
            for (int l = 1; l < _L; l++)
            {
                var (delta, errBelow) = LayerErrorAndPrediction(r, l);
                deltas[l]  = delta;
                errs[l - 1] = errBelow;
            }
            // errs[L-1] stays zero (top layer receives no prediction from above).

            // Integrate membranes for free layers, emit spikes, update rates.
            for (int l = 1; l < _L; l++)
            {
                if (clampTop && l == _L - 1) continue; // label-clamped

                var bottomUp = TensorUtils.MatVecMulT(_w[l], deltas[l]); // size_l
                for (int i = 0; i < V[l].Length; i++)
                {
                    float drive = bottomUp.Buffer[i] - errs[l].Buffer[i];
                    V[l].Buffer[i] += dt * (-leak * V[l].Buffer[i] + gain * drive);

                    float spike = 0f;
                    if (V[l].Buffer[i] >= thr) { spike = 1f; V[l].Buffer[i] -= thr; }
                    r[l].Buffer[i] = rho * r[l].Buffer[i] + (1f - rho) * spike;
                }
            }
        }

        // Re-clamp boundaries so the returned rates reflect the clamped values.
        input.CopyTo(r[0]);
        if (clampTop && label != null) label.CopyTo(r[_L - 1]);
        return r;
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    /// <summary>
    /// For layer <paramref name="l"/> (1..L-1) computes the precision-weighted
    /// delta δ = ε_{l-1} ⊙ f'(W_l r_l + b_l) and the raw error ε_{l-1}.
    /// </summary>
    private (DenseTensor<float> delta, DenseTensor<float> errBelow) LayerErrorAndPrediction(
        DenseTensor<float>[] r, int l)
    {
        var pre  = TensorUtils.ElementwiseAdd(TensorUtils.MatVecMul(_w[l], r[l]), _b[l]);
        var pred = ActivationFn.Apply(pre, _config.Activation);
        var err  = TensorUtils.ElementwiseSub(r[l - 1], pred);            // ε_{l-1}
        var fp   = ActivationFn.Derivative(pre, pred, _config.Activation);
        var delta = TensorUtils.ElementwiseMul(err, fp);
        return (delta, err);
    }

    private float Energy(DenseTensor<float>[] r)
    {
        float energy = 0f;
        for (int l = 1; l < _L; l++)
        {
            var (_, errBelow) = LayerErrorAndPrediction(r, l);
            energy += 0.5f * TensorUtils.SumSquares(errBelow);
        }
        return energy;
    }

    private static void CheckSize(DenseTensor<float> t, int expected, string name)
    {
        if (t.Length != expected)
            throw new ArgumentException($"{name} has {t.Length} elements; expected {expected}.");
    }
}
