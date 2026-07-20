namespace PredictiveCoding.Device.Kernels;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceMath  —  the non-GEMM tensor ops the LLM needs, as GPU kernels.
//
// Each op is a small CUDA C kernel compiled once (Lazy<CudaKernel>) and launched
// on device pointers, so a forward/backward step can stay resident on the card
// instead of bouncing to the CPU around each cuBLAS matmul. Pointers are the
// device addresses the library's DeviceTensor / cudart allocator hands out.
//
// Kernels operate IN PLACE where the math allows (GELU) or with explicit row
// semantics (softmax, RMSNorm over the last dimension of a [rows, dim] tensor).
// ─────────────────────────────────────────────────────────────────────────────

public static class DeviceMath
{
    // ── GELU (elementwise, in place) ──
    private static readonly Lazy<CudaKernel> GeluK = new(() => new CudaKernel(GeluSrc, "gelu"));
    private const string GeluSrc = @"
extern ""C"" __global__ void gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(t));
    }
}";

    /// <summary>In-place GELU (tanh approximation) over <paramref name="count"/> elements.</summary>
    public static void GeluInPlace(IntPtr x, int count) => GeluK.Value.Launch1D(count, 256, x, count);

    // ── Row-wise softmax (in place) — one block per row, numerically stable ──
    private static readonly Lazy<CudaKernel> SoftmaxK = new(() => new CudaKernel(SoftmaxSrc, "softmax_rows"));
    private const string SoftmaxSrc = @"
extern ""C"" __global__ void softmax_rows(float* x, int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* r = x + (long)row * dim;
    // max
    float m = -3.402823466e38f;
    for (int j = 0; j < dim; j++) m = fmaxf(m, r[j]);
    // exp + sum
    float s = 0.0f;
    for (int j = 0; j < dim; j++) { float e = expf(r[j] - m); r[j] = e; s += e; }
    // normalise
    float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int j = 0; j < dim; j++) r[j] *= inv;
}";

    /// <summary>In-place softmax over the last dim of a row-major [rows, dim] tensor.</summary>
    public static void SoftmaxRows(IntPtr x, int rows, int dim)
        => SoftmaxK.Value.Launch((uint)rows, 1, x, rows, dim);

    // ── RMSNorm (in place) — one block per row: y = x / rms(x) * gamma ──
    private static readonly Lazy<CudaKernel> RmsNormK = new(() => new CudaKernel(RmsNormSrc, "rmsnorm_rows"));
    private const string RmsNormSrc = @"
extern ""C"" __global__ void rmsnorm_rows(float* x, const float* gamma, int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* r = x + (long)row * dim;
    float ss = 0.0f;
    for (int j = 0; j < dim; j++) ss += r[j] * r[j];
    float inv = rsqrtf(ss / dim + eps);
    for (int j = 0; j < dim; j++) r[j] = r[j] * inv * gamma[j];
}";

    /// <summary>In-place RMSNorm over the last dim of [rows, dim], scaled by <paramref name="gamma"/> [dim].</summary>
    public static void RmsNormRows(IntPtr x, IntPtr gamma, int rows, int dim, float eps)
        => RmsNormK.Value.Launch((uint)rows, 1, x, gamma, rows, dim, eps);

    // ── RoPE (in place) — interleaved-pair rotary, matching Model.RotaryEmbedding ──
    // x is [rows, heads*headDim]; token at row r gets position basePos + r; within
    // each head the pairs (2i, 2i+1) are rotated by angle = position / 10000^(2i/headDim).
    private static readonly Lazy<CudaKernel> RopeK = new(() => new CudaKernel(RopeSrc, "rope"));
    private const string RopeSrc = @"
extern ""C"" __global__ void rope(float* x, int rows, int heads, int headDim, int basePos) {
    int half    = headDim / 2;
    int perHead = half;
    long total  = (long)rows * heads * perHead;
    long idx    = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row  = (int)(idx / ((long)heads * perHead));
    int rem  = (int)(idx % ((long)heads * perHead));
    int head = rem / perHead;
    int i    = rem % perHead;
    int position = basePos + row;
    float freq  = 1.0f / powf(10000.0f, (float)(2 * i) / headDim);
    float angle = position * freq;
    float c = cosf(angle), s = sinf(angle);
    long base = (long)row * heads * headDim + (long)head * headDim;
    long a = base + 2 * i, b = a + 1;
    float x0 = x[a], x1 = x[b];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
}";

    /// <summary>
    /// In-place rotary position embedding on [rows, heads*headDim]: token in row r
    /// uses position <paramref name="basePos"/> + r. Matches <c>Model.RotaryEmbedding</c>.
    /// </summary>
    public static void RopeInPlace(IntPtr x, int rows, int heads, int headDim, int basePos)
        => RopeK.Value.Launch1D(rows * heads * (headDim / 2), 256, x, rows, heads, headDim, basePos);

    // ── Single-query multi-head attention (flash-style online softmax) ──
    // One query per head attends to n cached K/V positions; matches the library's
    // MultiHeadAttention.AttendOnline (basic case: score = dot(q,k)·scale, no ALiBi/
    // softcap). One thread per head; headDim ≤ 128 (register-resident accumulator).
    // Q,O: [heads, headDim]; K,V: [n, heads, headDim] (MHA: kv head = q head).
    private static readonly Lazy<CudaKernel> AttnK = new(() => new CudaKernel(AttnSrc, "attn_1q"));
    private const string AttnSrc = @"
extern ""C"" __global__ void attn_1q(const float* Q, const float* K, const float* V, float* O,
                                     int heads, int headDim, int n, float scale) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    const float* q = Q + (long)h * headDim;
    float acc[128];
    for (int i = 0; i < headDim; i++) acc[i] = 0.0f;
    float m = -3.402823466e38f, l = 0.0f;
    for (int p = 0; p < n; p++) {
        const float* k = K + ((long)p * heads + h) * headDim;
        const float* v = V + ((long)p * heads + h) * headDim;
        float s = 0.0f;
        for (int i = 0; i < headDim; i++) s += q[i] * k[i];
        s *= scale;
        float mNew = fmaxf(m, s);
        float sc = expf(m - mNew);
        float pe = expf(s - mNew);
        l = l * sc + pe;
        for (int i = 0; i < headDim; i++) acc[i] = acc[i] * sc + pe * v[i];
        m = mNew;
    }
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    float* o = O + (long)h * headDim;
    for (int i = 0; i < headDim; i++) o[i] = acc[i] * inv;
}";

    /// <summary>
    /// Single-query attention on the device: for each of <paramref name="heads"/> heads, the
    /// query [headDim] attends to <paramref name="n"/> cached keys/values (online softmax),
    /// writing the context to <paramref name="o"/> [heads, headDim]. <paramref name="scale"/>
    /// is 1/√headDim. headDim ≤ 128.
    /// </summary>
    public static void AttentionSingleQuery(IntPtr q, IntPtr k, IntPtr v, IntPtr o, int heads, int headDim, int n, float scale)
    {
        if (headDim > 128) throw new ArgumentOutOfRangeException(nameof(headDim), "attn_1q kernel supports headDim ≤ 128.");
        AttnK.Value.Launch1D(heads, 64, q, k, v, o, heads, headDim, n, scale);
    }

    // ── Backward helpers (Phase A, Point 3) ──

    // Transpose [rows, cols] → [cols, rows]. Needed for the Linear gradients
    // (dW = xᵀ·dy, dx = dy·Wᵀ) since the device MatMul computes C = A·B only.
    private static readonly Lazy<CudaKernel> TransposeK = new(() => new CudaKernel(TransposeSrc, "transpose"));
    private const string TransposeSrc = @"
extern ""C"" __global__ void transpose(const float* in, float* out, int rows, int cols) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * cols;
    if (idx >= total) return;
    int r = (int)(idx / cols);
    int c = (int)(idx % cols);
    out[(long)c * rows + r] = in[idx];
}";

    /// <summary>Transposes a row-major [rows, cols] tensor into <paramref name="dst"/> [cols, rows].</summary>
    public static void Transpose(IntPtr src, IntPtr dst, int rows, int cols)
        => TransposeK.Value.Launch1D(rows * cols, 256, src, dst, rows, cols);

    // GELU backward: grad ← grad · gelu'(preact), elementwise (tanh-approx derivative).
    private static readonly Lazy<CudaKernel> GeluBwdK = new(() => new CudaKernel(GeluBwdSrc, "gelu_bwd"));
    private const string GeluBwdSrc = @"
extern ""C"" __global__ void gelu_bwd(float* grad, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = 0.7978845608028654f;
        float u = c * (v + 0.044715f * v * v * v);
        float t = tanhf(u);
        float du = c * (1.0f + 3.0f * 0.044715f * v * v);
        float deriv = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du;
        grad[i] = grad[i] * deriv;
    }
}";

    /// <summary>
    /// In-place GELU backward: turns an upstream gradient into the pre-activation
    /// gradient — <paramref name="grad"/> ← grad · gelu'(<paramref name="preact"/>).
    /// </summary>
    public static void GeluBackwardInPlace(IntPtr grad, IntPtr preact, int count)
        => GeluBwdK.Value.Launch1D(count, 256, grad, preact, count);

    // Softmax backward (in place): grad ← y ⊙ (grad − Σ(grad⊙y)) per row, where y is
    // the softmax OUTPUT of the forward pass. One block per row.
    private static readonly Lazy<CudaKernel> SoftmaxBwdK = new(() => new CudaKernel(SoftmaxBwdSrc, "softmax_bwd"));
    private const string SoftmaxBwdSrc = @"
extern ""C"" __global__ void softmax_bwd(float* grad, const float* y, int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* g = grad + (long)row * dim;
    const float* yy = y + (long)row * dim;
    float dot = 0.0f;
    for (int j = 0; j < dim; j++) dot += g[j] * yy[j];
    for (int j = 0; j < dim; j++) g[j] = yy[j] * (g[j] - dot);
}";

    /// <summary>In-place softmax backward over [rows, dim]; <paramref name="y"/> is the forward softmax output.</summary>
    public static void SoftmaxBackwardRows(IntPtr grad, IntPtr y, int rows, int dim)
        => SoftmaxBwdK.Value.Launch((uint)rows, 1, grad, y, rows, dim);

    // RMSNorm backward (in place) → dx (grad w.r.t. the input; dgamma is a separate
    // cross-row reduction, not done here). Recomputes inv from x. One block per row.
    //   dx_i = inv·γ_i·g_i − (inv³·x_i/dim)·Σ_j g_j·γ_j·x_j
    private static readonly Lazy<CudaKernel> RmsNormBwdK = new(() => new CudaKernel(RmsNormBwdSrc, "rmsnorm_bwd"));
    private const string RmsNormBwdSrc = @"
extern ""C"" __global__ void rmsnorm_bwd(float* grad, const float* x, const float* gamma, int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* g = grad + (long)row * dim;
    const float* xr = x + (long)row * dim;
    float ss = 0.0f;
    for (int j = 0; j < dim; j++) ss += xr[j] * xr[j];
    float inv = rsqrtf(ss / dim + eps);
    float S = 0.0f;
    for (int j = 0; j < dim; j++) S += g[j] * gamma[j] * xr[j];
    float inv3 = inv * inv * inv;
    for (int j = 0; j < dim; j++) g[j] = inv * gamma[j] * g[j] - (inv3 * xr[j] / dim) * S;
}";

    /// <summary>In-place RMSNorm backward → dx over [rows, dim] (needs the forward input x and gamma).</summary>
    public static void RmsNormBackwardRows(IntPtr grad, IntPtr x, IntPtr gamma, int rows, int dim, float eps)
        => RmsNormBwdK.Value.Launch((uint)rows, 1, grad, x, gamma, rows, dim, eps);

    // RoPE backward (in place): the rotation is orthogonal, so the input gradient is
    // the rotation by −θ (transpose). Same indexing as the forward rope kernel.
    private static readonly Lazy<CudaKernel> RopeBwdK = new(() => new CudaKernel(RopeBwdSrc, "rope_bwd"));
    private const string RopeBwdSrc = @"
extern ""C"" __global__ void rope_bwd(float* g, int rows, int heads, int headDim, int basePos) {
    int half    = headDim / 2;
    int perHead = half;
    long total  = (long)rows * heads * perHead;
    long idx    = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row  = (int)(idx / ((long)heads * perHead));
    int rem  = (int)(idx % ((long)heads * perHead));
    int head = rem / perHead;
    int i    = rem % perHead;
    int position = basePos + row;
    float freq  = 1.0f / powf(10000.0f, (float)(2 * i) / headDim);
    float angle = position * freq;
    float c = cosf(angle), s = sinf(angle);
    long base = (long)row * heads * headDim + (long)head * headDim;
    long a = base + 2 * i, b = a + 1;
    float g0 = g[a], g1 = g[b];
    g[a] =  g0 * c + g1 * s;   // Rᵀ(θ) = R(−θ)
    g[b] = -g0 * s + g1 * c;
}";

    /// <summary>In-place RoPE backward on [rows, heads*headDim] — the transpose (−θ rotation) of the forward.</summary>
    public static void RopeBackwardInPlace(IntPtr grad, int rows, int heads, int headDim, int basePos)
        => RopeBwdK.Value.Launch1D(rows * heads * (headDim / 2), 256, grad, rows, heads, headDim, basePos);
}
