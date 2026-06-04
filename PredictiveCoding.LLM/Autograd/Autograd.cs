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
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float s = 0f;
                for (int l = 0; l < k; l++) s += a.Data[i * k + l] * b.Data[l * n + j];
                o.Data[i * n + j] = s;
            }

        o.BackwardFn = () =>
        {
            for (int i = 0; i < m; i++)
                for (int l = 0; l < k; l++)
                {
                    float ga = 0f;
                    for (int j = 0; j < n; j++) ga += o.Grad[i * n + j] * b.Data[l * n + j];
                    a.Grad[i * k + l] += ga;
                }
            for (int l = 0; l < k; l++)
                for (int j = 0; j < n; j++)
                {
                    float gb = 0f;
                    for (int i = 0; i < m; i++) gb += a.Data[i * k + l] * o.Grad[i * n + j];
                    b.Grad[l * n + j] += gb;
                }
        };
        return o;
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
}
