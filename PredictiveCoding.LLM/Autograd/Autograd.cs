using System.Numerics;

namespace PredictiveCoding.LLM.Autograd;

// ─────────────────────────────────────────────────────────────────────────────
// Autograd  —  a minimal reverse-mode automatic-differentiation engine.
//
// Tensors are 2-D (rows × cols). Each op records its parents and a closure that
// pushes gradients back into them. Calling Backward() on a scalar loss builds the
// topological order, seeds dLoss/dLoss = 1 and runs the closures in reverse,
// accumulating ∂loss/∂· in every tensor's Grad. This is the gradient machinery a
// transformer needs to be trained (the surrounding PredictiveCoding library is
// otherwise backprop-free); see LLM_Roadmap.md.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Tensor
{
    public int Rows { get; }
    public int Cols { get; }
    public float[] Data { get; }
    public float[] Grad { get; }

    internal Action?   BackwardFn;
    internal Tensor[]  Parents = [];

    public Tensor(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new float[rows * cols];
        Grad = new float[rows * cols];
    }

    public static Tensor From(float[] data, int rows, int cols)
    {
        var t = new Tensor(rows, cols);
        Array.Copy(data, t.Data, data.Length);
        return t;
    }

    /// <summary>Reverse pass from this (scalar) tensor: fills Grad on all ancestors.</summary>
    public void Backward()
    {
        var topo    = new List<Tensor>();
        var visited = new HashSet<Tensor>();
        Build(this, topo, visited);

        foreach (var t in topo) Array.Clear(t.Grad);
        Grad[0] = 1f;                                   // seed dL/dL
        for (int i = topo.Count - 1; i >= 0; i--) topo[i].BackwardFn?.Invoke();
    }

    private static void Build(Tensor t, List<Tensor> topo, HashSet<Tensor> visited)
    {
        if (!visited.Add(t)) return;
        foreach (var p in t.Parents) Build(p, topo, visited);
        topo.Add(t);
    }
}

/// <summary>Differentiable operations producing new tensors wired for backprop.</summary>
public static class Ag
{
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        int m = a.Rows, k = a.Cols, n = b.Cols;
        var o = new Tensor(m, n) { Parents = [a, b] };
        float[] A = a.Data, B = b.Data, C = o.Data;

        // Forward C = A·B in i-k-j order: the inner axpy over j is contiguous and
        // SIMD-vectorised, and rows of C are independent so they parallelise cleanly.
        RunRows(m, i =>
        {
            var crow = C.AsSpan(i * n, n);
            crow.Clear();
            int abase = i * k;
            for (int l = 0; l < k; l++)
            {
                float av = A[abase + l];
                if (av != 0f) AddScaled(crow, B.AsSpan(l * n, n), av);          // crow += av · B[l,:]
            }
        });

        o.BackwardFn = () =>
        {
            float[] dC = o.Grad, dA = a.Grad, dB = b.Grad;

            // dA = dC·Bᵀ : dA[i,l] += dot(dC[i,:], B[l,:]).  Parallel over i (distinct rows of dA).
            RunRows(m, i =>
            {
                var dcrow = dC.AsSpan(i * n, n);
                int abase = i * k;
                for (int l = 0; l < k; l++)
                    dA[abase + l] += Dot(dcrow, B.AsSpan(l * n, n));
            });

            // dB = Aᵀ·dC : dB[l,:] += A[i,l]·dC[i,:].  Parallel over l (distinct rows of dB) → race-free.
            RunRows(k, l =>
            {
                var dbrow = dB.AsSpan(l * n, n);
                for (int i = 0; i < m; i++)
                {
                    float av = A[i * k + l];
                    if (av != 0f) AddScaled(dbrow, dC.AsSpan(i * n, n), av);     // dbrow += av · dC[i,:]
                }
            });
        };
        return o;
    }

    // ── SIMD / parallel helpers for MatMul ───────────────────────────────────────

    /// <summary>Runs <paramref name="body"/> over rows, parallel when the work amortises the overhead.</summary>
    private static void RunRows(int rows, Action<int> body)
    {
        const int ParallelThreshold = 64;       // small matmuls (e.g. gradient-checks) stay serial + deterministic
        if (rows >= ParallelThreshold) Parallel.For(0, rows, body);
        else for (int i = 0; i < rows; i++) body(i);
    }

    /// <summary>acc += scale · x, SIMD-vectorised.</summary>
    private static void AddScaled(Span<float> acc, ReadOnlySpan<float> x, float scale)
    {
        int n = acc.Length, i = 0, w = Vector<float>.Count;
        var vs = new Vector<float>(scale);
        for (; i <= n - w; i += w)
        {
            var r = new Vector<float>(acc.Slice(i, w)) + vs * new Vector<float>(x.Slice(i, w));
            r.CopyTo(acc.Slice(i, w));
        }
        for (; i < n; i++) acc[i] += scale * x[i];
    }

    /// <summary>Dot product of two equal-length spans, SIMD-vectorised.</summary>
    private static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        int n = x.Length, i = 0, w = Vector<float>.Count;
        var acc = Vector<float>.Zero;
        for (; i <= n - w; i += w)
            acc += new Vector<float>(x.Slice(i, w)) * new Vector<float>(y.Slice(i, w));
        float s = Vector.Dot(acc, Vector<float>.One);
        for (; i < n; i++) s += x[i] * y[i];
        return s;
    }

    public static Tensor Transpose(Tensor x)
    {
        int m = x.Rows, n = x.Cols;
        var o = new Tensor(n, m) { Parents = [x] };
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) o.Data[j * m + i] = x.Data[i * n + j];
        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) x.Grad[i * n + j] += o.Grad[j * m + i];
        };
        return o;
    }

    /// <summary>Elementwise add of equally-shaped tensors (residual connections).</summary>
    public static Tensor Add(Tensor a, Tensor b)
    {
        var o = new Tensor(a.Rows, a.Cols) { Parents = [a, b] };
        for (int i = 0; i < o.Data.Length; i++) o.Data[i] = a.Data[i] + b.Data[i];
        o.BackwardFn = () =>
        {
            for (int i = 0; i < o.Data.Length; i++) { a.Grad[i] += o.Grad[i]; b.Grad[i] += o.Grad[i]; }
        };
        return o;
    }

    /// <summary>Adds a row-vector bias b[1,n] to every row of x[m,n].</summary>
    public static Tensor AddBias(Tensor x, Tensor b)
    {
        int m = x.Rows, n = x.Cols;
        var o = new Tensor(m, n) { Parents = [x, b] };
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) o.Data[i * n + j] = x.Data[i * n + j] + b.Data[j];
        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) { x.Grad[i * n + j] += o.Grad[i * n + j]; b.Grad[j] += o.Grad[i * n + j]; }
        };
        return o;
    }

    public static Tensor Scale(Tensor x, float s)
    {
        var o = new Tensor(x.Rows, x.Cols) { Parents = [x] };
        for (int i = 0; i < o.Data.Length; i++) o.Data[i] = x.Data[i] * s;
        o.BackwardFn = () => { for (int i = 0; i < o.Data.Length; i++) x.Grad[i] += s * o.Grad[i]; };
        return o;
    }

    public static Tensor Gelu(Tensor x)
    {
        const float c = 0.7978845608f, a = 0.044715f;
        var o = new Tensor(x.Rows, x.Cols) { Parents = [x] };
        for (int i = 0; i < o.Data.Length; i++)
        {
            float v = x.Data[i];
            o.Data[i] = 0.5f * v * (1f + MathF.Tanh(c * (v + a * v * v * v)));
        }
        o.BackwardFn = () =>
        {
            for (int i = 0; i < o.Data.Length; i++)
            {
                float v = x.Data[i];
                float u = c * (v + a * v * v * v);
                float t = MathF.Tanh(u);
                float dudv = c * (1f + 3f * a * v * v);
                float dg = 0.5f * (1f + t) + 0.5f * v * (1f - t * t) * dudv;
                x.Grad[i] += dg * o.Grad[i];
            }
        };
        return o;
    }

    /// <summary>
    /// Inverted dropout: in training, zeroes each element with probability p and
    /// scales survivors by 1/(1−p); in inference it is the identity.
    /// </summary>
    public static Tensor Dropout(Tensor x, float p, bool training, Random rng)
    {
        var o = new Tensor(x.Rows, x.Cols) { Parents = [x] };
        if (!training || p <= 0f)
        {
            Array.Copy(x.Data, o.Data, x.Data.Length);
            o.BackwardFn = () => { for (int i = 0; i < x.Data.Length; i++) x.Grad[i] += o.Grad[i]; };
            return o;
        }

        float scale = 1f / (1f - p);
        var   mask  = new float[x.Data.Length];
        for (int i = 0; i < x.Data.Length; i++)
        {
            mask[i]   = rng.NextDouble() < p ? 0f : scale;
            o.Data[i] = x.Data[i] * mask[i];
        }
        o.BackwardFn = () => { for (int i = 0; i < x.Data.Length; i++) x.Grad[i] += mask[i] * o.Grad[i]; };
        return o;
    }

    /// <summary>Row-wise softmax with a causal mask: row i attends to columns ≤ i.</summary>
    public static Tensor SoftmaxRowsCausal(Tensor x)
    {
        int m = x.Rows, n = x.Cols;
        var o = new Tensor(m, n) { Parents = [x] };
        for (int i = 0; i < m; i++)
        {
            int last = Math.Min(i, n - 1);
            float max = float.NegativeInfinity;
            for (int j = 0; j <= last; j++) max = MathF.Max(max, x.Data[i * n + j]);
            float sum = 0f;
            for (int j = 0; j <= last; j++) { float e = MathF.Exp(x.Data[i * n + j] - max); o.Data[i * n + j] = e; sum += e; }
            float inv = sum > 0f ? 1f / sum : 0f;
            for (int j = 0; j <= last; j++) o.Data[i * n + j] *= inv;
        }
        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++)
            {
                int last = Math.Min(i, n - 1);
                float dot = 0f;
                for (int j = 0; j <= last; j++) dot += o.Data[i * n + j] * o.Grad[i * n + j];
                for (int j = 0; j <= last; j++)
                    x.Grad[i * n + j] += o.Data[i * n + j] * (o.Grad[i * n + j] - dot);
            }
        };
        return o;
    }

    /// <summary>Per-row layer normalisation with learnable γ[1,n] and β[1,n].</summary>
    public static Tensor LayerNorm(Tensor x, Tensor gamma, Tensor beta, float eps)
    {
        int m = x.Rows, n = x.Cols;
        var o      = new Tensor(m, n) { Parents = [x, gamma, beta] };
        var xhat   = new float[m * n];
        var invStd = new float[m];

        for (int i = 0; i < m; i++)
        {
            float mean = 0f;
            for (int j = 0; j < n; j++) mean += x.Data[i * n + j];
            mean /= n;
            float var = 0f;
            for (int j = 0; j < n; j++) { float d = x.Data[i * n + j] - mean; var += d * d; }
            var /= n;
            invStd[i] = 1f / MathF.Sqrt(var + eps);
            for (int j = 0; j < n; j++)
            {
                float xh = (x.Data[i * n + j] - mean) * invStd[i];
                xhat[i * n + j] = xh;
                o.Data[i * n + j] = xh * gamma.Data[j] + beta.Data[j];
            }
        }

        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++)
            {
                float sumDy = 0f, sumDyXhat = 0f;
                for (int j = 0; j < n; j++)
                {
                    float dy = o.Grad[i * n + j] * gamma.Data[j];
                    sumDy     += dy;
                    sumDyXhat += dy * xhat[i * n + j];
                }
                for (int j = 0; j < n; j++)
                {
                    float dy = o.Grad[i * n + j] * gamma.Data[j];
                    x.Grad[i * n + j] += invStd[i] * (dy - sumDy / n - xhat[i * n + j] * sumDyXhat / n);
                    gamma.Grad[j] += o.Grad[i * n + j] * xhat[i * n + j];
                    beta.Grad[j]  += o.Grad[i * n + j];
                }
            }
        };
        return o;
    }

    /// <summary>Mean next-token cross-entropy of logits[m,vocab] against targets[m]; returns a scalar.</summary>
    public static Tensor CrossEntropy(Tensor logits, int[] targets)
    {
        int m = logits.Rows, vocab = logits.Cols;
        var o     = new Tensor(1, 1) { Parents = [logits] };
        var probs = new float[m * vocab];
        double loss = 0d;

        for (int i = 0; i < m; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < vocab; j++) max = MathF.Max(max, logits.Data[i * vocab + j]);
            float sum = 0f;
            for (int j = 0; j < vocab; j++) { float e = MathF.Exp(logits.Data[i * vocab + j] - max); probs[i * vocab + j] = e; sum += e; }
            float inv = 1f / sum;
            for (int j = 0; j < vocab; j++) probs[i * vocab + j] *= inv;
            loss += -MathF.Log(probs[i * vocab + targets[i]] + 1e-12f);
        }
        o.Data[0] = (float)(loss / m);

        o.BackwardFn = () =>
        {
            float scale = o.Grad[0] / m;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < vocab; j++)
                {
                    float g = probs[i * vocab + j] - (j == targets[i] ? 1f : 0f);
                    logits.Grad[i * vocab + j] += scale * g;
                }
        };
        return o;
    }

    /// <summary>
    /// Masked next-token cross-entropy: like <see cref="CrossEntropy(Tensor,int[])"/>
    /// but every row whose target equals <paramref name="ignoreIndex"/> is skipped
    /// (no loss, no gradient), and the mean is taken over the counted rows only.
    /// This is what sequence-to-sequence translation training needs: the loss must
    /// fall on the target-side tokens, not on the source/prompt that only conditions.
    /// </summary>
    public static Tensor CrossEntropy(Tensor logits, int[] targets, int ignoreIndex)
    {
        int m = logits.Rows, vocab = logits.Cols;
        var o       = new Tensor(1, 1) { Parents = [logits] };
        var probs   = new float[m * vocab];
        double loss = 0d;
        int    counted = 0;

        for (int i = 0; i < m; i++)
        {
            if (targets[i] == ignoreIndex) continue;             // masked position: no loss
            float max = float.NegativeInfinity;
            for (int j = 0; j < vocab; j++) max = MathF.Max(max, logits.Data[i * vocab + j]);
            float sum = 0f;
            for (int j = 0; j < vocab; j++) { float e = MathF.Exp(logits.Data[i * vocab + j] - max); probs[i * vocab + j] = e; sum += e; }
            float inv = 1f / sum;
            for (int j = 0; j < vocab; j++) probs[i * vocab + j] *= inv;
            loss += -MathF.Log(probs[i * vocab + targets[i]] + 1e-12f);
            counted++;
        }
        o.Data[0] = counted > 0 ? (float)(loss / counted) : 0f;

        o.BackwardFn = () =>
        {
            if (counted == 0) return;
            float scale = o.Grad[0] / counted;
            for (int i = 0; i < m; i++)
            {
                if (targets[i] == ignoreIndex) continue;         // masked rows contribute no gradient
                for (int j = 0; j < vocab; j++)
                {
                    float g = probs[i * vocab + j] - (j == targets[i] ? 1f : 0f);
                    logits.Grad[i * vocab + j] += scale * g;
                }
            }
        };
        return o;
    }

    // ── Temporal-state ops (roadmap Phase 5) ─────────────────────────────────

    /// <summary>Elementwise product a ⊙ b (same shape).</summary>
    public static Tensor Mul(Tensor a, Tensor b)
    {
        var o = new Tensor(a.Rows, a.Cols) { Parents = [a, b] };
        for (int i = 0; i < o.Data.Length; i++) o.Data[i] = a.Data[i] * b.Data[i];
        o.BackwardFn = () =>
        {
            for (int i = 0; i < o.Data.Length; i++)
            {
                a.Grad[i] += o.Grad[i] * b.Data[i];
                b.Grad[i] += o.Grad[i] * a.Data[i];
            }
        };
        return o;
    }

    /// <summary>Elementwise logistic sigmoid σ(x); dσ = σ·(1−σ).</summary>
    public static Tensor Sigmoid(Tensor x)
    {
        var o = new Tensor(x.Rows, x.Cols) { Parents = [x] };
        for (int i = 0; i < o.Data.Length; i++) o.Data[i] = 1f / (1f + MathF.Exp(-x.Data[i]));
        o.BackwardFn = () =>
        {
            for (int i = 0; i < o.Data.Length; i++)
                x.Grad[i] += o.Grad[i] * o.Data[i] * (1f - o.Data[i]);
        };
        return o;
    }

    /// <summary>Row-wise softmax over full rows (the MoE router's gate distribution).</summary>
    public static Tensor SoftmaxRows(Tensor x)
    {
        int m = x.Rows, n = x.Cols;
        var o = new Tensor(m, n) { Parents = [x] };
        for (int i = 0; i < m; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < n; j++) max = MathF.Max(max, x.Data[i * n + j]);
            float sum = 0f;
            for (int j = 0; j < n; j++) { float e = MathF.Exp(x.Data[i * n + j] - max); o.Data[i * n + j] = e; sum += e; }
            float inv = 1f / sum;
            for (int j = 0; j < n; j++) o.Data[i * n + j] *= inv;
        }
        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++)
            {
                float dot = 0f;
                for (int j = 0; j < n; j++) dot += o.Grad[i * n + j] * o.Data[i * n + j];
                for (int j = 0; j < n; j++) x.Grad[i * n + j] += o.Data[i * n + j] * (o.Grad[i * n + j] - dot);
            }
        };
        return o;
    }

    /// <summary>Column <paramref name="j"/> of x as an [m, 1] tensor; backward scatters into that column.</summary>
    public static Tensor Column(Tensor x, int j)
    {
        int m = x.Rows, n = x.Cols;
        var o = new Tensor(m, 1) { Parents = [x] };
        for (int i = 0; i < m; i++) o.Data[i] = x.Data[i * n + j];
        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++) x.Grad[i * n + j] += o.Grad[i];
        };
        return o;
    }

    /// <summary>o[i,·] = x[i,·] · s[i], with s an [m, 1] per-row scale; gradients flow to both.</summary>
    public static Tensor ScaleRows(Tensor x, Tensor s)
    {
        int m = x.Rows, n = x.Cols;
        var o = new Tensor(m, n) { Parents = [x, s] };
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) o.Data[i * n + j] = x.Data[i * n + j] * s.Data[i];
        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++)
            {
                float acc = 0f;
                for (int j = 0; j < n; j++)
                {
                    x.Grad[i * n + j] += o.Grad[i * n + j] * s.Data[i];
                    acc += o.Grad[i * n + j] * x.Data[i * n + j];
                }
                s.Grad[i] += acc;
            }
        };
        return o;
    }

    /// <summary>Mean of every element, as a [1, 1] scalar.</summary>
    public static Tensor Mean(Tensor x)
    {
        var o = new Tensor(1, 1) { Parents = [x] };
        double sum = 0d;
        for (int i = 0; i < x.Data.Length; i++) sum += x.Data[i];
        o.Data[0] = (float)(sum / x.Data.Length);
        o.BackwardFn = () =>
        {
            float g = o.Grad[0] / x.Data.Length;
            for (int i = 0; i < x.Data.Length; i++) x.Grad[i] += g;
        };
        return o;
    }

    /// <summary>
    /// Gated linear recurrence over the rows (time): h_t = a_t ⊙ h_{t−1} + b_t
    /// with h_{−1} = 0 — the diagonal state-space scan hybrid architectures
    /// build on. The backward pass is BPTT run inside the op:
    ///   g_t = ∂L/∂h_t + a_{t+1} ⊙ g_{t+1},   ∂L/∂b_t = g_t,   ∂L/∂a_t = g_t ⊙ h_{t−1}.
    /// </summary>
    public static Tensor RecurrentScan(Tensor a, Tensor b)
    {
        int T = a.Rows, d = a.Cols;
        var o = new Tensor(T, d) { Parents = [a, b] };

        for (int i = 0; i < d; i++) o.Data[i] = b.Data[i];               // h_0 (empty past)
        for (int t = 1; t < T; t++)
            for (int i = 0; i < d; i++)
                o.Data[t * d + i] = a.Data[t * d + i] * o.Data[(t - 1) * d + i] + b.Data[t * d + i];

        o.BackwardFn = () =>
        {
            var g = new float[d];                                        // carries g_{t+1} down the time axis
            for (int t = T - 1; t >= 0; t--)
            {
                for (int i = 0; i < d; i++)
                {
                    float gt = o.Grad[t * d + i] + (t + 1 < T ? a.Data[(t + 1) * d + i] * g[i] : 0f);
                    b.Grad[t * d + i] += gt;
                    if (t > 0) a.Grad[t * d + i] += gt * o.Data[(t - 1) * d + i];
                    g[i] = gt;
                }
            }
        };
        return o;
    }
}
