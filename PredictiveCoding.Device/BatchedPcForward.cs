using PredictiveCoding.Core;
using PredictiveCoding.LLM.Infrastructure;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// BatchedPcForward  —  the predictive-coding free inference reshaped from B
// independent matrix×vector settlings into per-step GEMMs over the whole batch
// (ROADMAP-MixedPrecision Phase 5, "batched PC forward on GPU").
//
// Per settling step and per layer the two heavy products become
//     Pre[l]  = M[l] · W_lᵀ        [B, lower]    (top-down prediction)
//     WᵀΔ[l]  = Δ[l] · W_l         [B, size]     (state-update gradient)
// and both route through Gemm.Provider — CpuGemmProvider by default, tensor
// cores when the provider is a CudaGemmProvider, and the mixed (16-bit
// operand / fp32 accumulate) kernels when `mixed` is set. This is exactly the
// batching the roadmap caveat calls for: a single matrix×vector on the GPU is
// transfer-dominated, a [B, n]·[n, m] GEMM is not.
//
// Semantics: InferGetTop(input, steps, resetState: true) for every sample —
// free layers start from zero, top layer unclamped, flat top prior. Scalar
// per-layer precision, activity regularisation and the FFG warm-start are
// reproduced; snapshots with diagonal precision / informative priors are
// already rejected at export.
// ─────────────────────────────────────────────────────────────────────────────

public static class BatchedPcForward
{
    /// <summary>
    /// Batched twin of <c>Network.InferGetTop(x, steps, resetState: true)</c>:
    /// settles every input in the batch simultaneously and returns the top
    /// latent per sample. With <paramref name="mixed"/> the GEMMs run on the
    /// provider's mixed path (Half operands, float32 accumulation).
    /// </summary>
    public static DenseTensor<float>[] InferGetTopBatch(
        GenerativeSnapshot            model,
        IReadOnlyList<DenseTensor<float>> inputs,
        int                           inferenceSteps,
        bool                          mixed = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Count == 0) throw new ArgumentException("Empty batch.", nameof(inputs));
        if (inferenceSteps < 1) throw new ArgumentOutOfRangeException(nameof(inferenceSteps));

        int levels = model.LayerSizes.Length;
        int batch  = inputs.Count;

        // M[l] : [batch, size_l] states; layer 0 clamped to the inputs.
        var m = new float[levels][];
        for (int l = 0; l < levels; l++) m[l] = new float[batch * model.LayerSizes[l]];
        for (int i = 0; i < batch; i++)
        {
            if (inputs[i].Length != model.LayerSizes[0])
                throw new ArgumentException(
                    $"Input {i} has {inputs[i].Length} elements; the model expects {model.LayerSizes[0]}.");
            inputs[i].Buffer.CopyTo(m[0], i * model.LayerSizes[0]);
        }

        // W_l is [lower, size]; the prediction GEMM needs W_lᵀ [size, lower].
        var wT = new float[levels - 1][];
        for (int l = 1; l < levels; l++)
            wT[l - 1] = Transpose(model.Weights[l - 1].Buffer, model.LayerSizes[l - 1], model.LayerSizes[l]);

        float[] Mul(float[] a, int rows, int inner, float[] b, int cols) => mixed
            ? Gemm.Provider.MatMulMixed(a, rows, inner, b, cols)
            : Gemm.MatMul(a, rows, inner, b, cols);

        // Pre[l] / Pred[l] : [batch, size_{l-1}] top-down products of layer l.
        var pre  = new float[levels][];
        var pred = new float[levels][];

        void TopDown(int l)
        {
            int size = model.LayerSizes[l], lower = model.LayerSizes[l - 1];
            pre[l] = Mul(m[l], batch, size, wT[l - 1], lower);
            AddBiasRows(pre[l], model.Biases[l - 1].Buffer, batch, lower);
            pred[l] = Activate(pre[l], batch, lower, model.Activation);
        }

        // FFG warm-start: one top-down sweep seeding each hidden state.
        if (model.UseFeedforwardGistInit)
            for (int l = levels - 1; l >= 1; l--)
            {
                TopDown(l);
                if (l > 1) Array.Copy(pred[l], m[l - 1], pred[l].Length);
            }

        float eta    = model.StateLearningRate;
        float actReg = model.ActivityRegularization;

        for (int step = 0; step < inferenceSteps; step++)
        {
            for (int l = levels - 1; l >= 1; l--)
                TopDown(l);

            // E[l] = Π_l (M[l] − Pred[l+1]) for l = 0..L−2; the free top layer
            // has a flat prior, so E[L−1] = 0 and its gradient is −WᵀΔ alone.
            var e = new float[levels][];
            for (int l = 0; l < levels - 1; l++)
            {
                int n = m[l].Length;
                var err = new float[n];
                float pi = model.Precisions[l];
                for (int i = 0; i < n; i++) err[i] = pi * (m[l][i] - pred[l + 1][i]);
                e[l] = err;
            }

            for (int l = 1; l < levels; l++)
            {
                int size = model.LayerSizes[l], lower = model.LayerSizes[l - 1];

                // Δ = f'(Pre) ⊙ E[l−1], then WᵀΔ as one GEMM: Δ[B,lower] · W[lower,size].
                var delta = Derive(pre[l], pred[l], batch, lower, model.Activation);
                for (int i = 0; i < delta.Length; i++) delta[i] *= e[l - 1][i];
                var wTDelta = Mul(delta, batch, lower, model.Weights[l - 1].Buffer, size);

                var state = m[l];
                var eSelf = l < levels - 1 ? e[l] : null;        // top: ε = 0
                for (int i = 0; i < state.Length; i++)
                {
                    float grad = (eSelf?[i] ?? 0f) - wTDelta[i];
                    if (actReg > 0f) grad += actReg * MathF.Sign(state[i]);
                    state[i] -= eta * grad;
                }
            }
        }

        int top = model.LayerSizes[^1];
        var outputs = new DenseTensor<float>[batch];
        for (int i = 0; i < batch; i++)
        {
            var t = new DenseTensor<float>(top);
            Array.Copy(m[^1], i * top, t.Buffer, 0, top);
            outputs[i] = t;
        }
        return outputs;
    }

    // ── helpers on raw [rows, cols] buffers ──────────────────────────────────

    private static float[] Transpose(float[] w, int rows, int cols)
    {
        var t = new float[w.Length];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[c * rows + r] = w[r * cols + c];
        return t;
    }

    private static void AddBiasRows(float[] mat, float[] bias, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            int off = r * cols;
            for (int c = 0; c < cols; c++) mat[off + c] += bias[c];
        }
    }

    // ActivationFn works element-wise on the buffer, so a [rows, cols] matrix
    // is just a long vector to it.
    private static float[] Activate(float[] preAct, int rows, int cols, ActivationType type)
    {
        var wrapped = new DenseTensor<float>(rows * cols);
        preAct.CopyTo(wrapped.Buffer, 0);
        return ActivationFn.Apply(wrapped, type).Buffer;
    }

    private static float[] Derive(float[] preAct, float[] activated, int rows, int cols, ActivationType type)
    {
        var p = new DenseTensor<float>(rows * cols);
        preAct.CopyTo(p.Buffer, 0);
        var a = new DenseTensor<float>(rows * cols);
        activated.CopyTo(a.Buffer, 0);
        return ActivationFn.Derivative(p, a, type).Buffer;
    }
}
