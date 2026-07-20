namespace PredictiveCoding.Device.Kernels;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceTrainKernels  —  the training-step kernels (ROADMAP-Scale A1/A3).
//
// Everything here queues on a caller-supplied CUDA stream and NEVER blocks:
// a full forward+backward+Adam step is one uninterrupted chain of kernel and
// cuBLAS launches, and the host synchronises once — when it reads the loss.
//
// The attention trio is the A3 optimisation: block per (query, head) with K/V
// tiles staged in shared memory (+1 padded stride — an unpadded headDim stride
// puts a warp's serial dots all on one smem bank, a measured 28% step
// regression), per-position scores computed serially from smem by one thread
// each, and the online-softmax reduction run once per TILE instead of once per
// position. The backward replays softmax weights from the saved logsumexp
// (flash style) — no [seq, seq] matrix ever materialises. Layouts:
//   Q,K,V,O,dO : [seq, heads, headDim]   (row = position, heads packed)
//   L,D        : [seq, heads]            (logsumexp / dot(dO,O) per row-head)
// ─────────────────────────────────────────────────────────────────────────────

public static class DeviceTrainKernels
{
    /// <summary>Largest supported head dim (shared-memory tiles are sized for it).</summary>
    public const int MaxHeadDim = 128;

    private static void RequireHeadDim(int headDim)
    {
        if (headDim < 2 || headDim > MaxHeadDim || (headDim & (headDim - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim must be a power of two in [2, {MaxHeadDim}], got {headDim} (the block reduction needs it).");
    }

    // ── Embedding ────────────────────────────────────────────────────────────

    private static readonly Lazy<CudaKernel> GatherK = new(() => new CudaKernel(GatherSrc, "embed_gather"));
    private const string GatherSrc = @"
extern ""C"" __global__ void embed_gather(const float* W, const int* tok, float* out, int seq, int dim) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)seq * dim) return;
    int i = (int)(idx / dim), j = (int)(idx % dim);
    out[idx] = W[(long)tok[i] * dim + j];
}";

    /// <summary>out[i,:] = W[tokens[i],:] — the device twin of Ag.GatherRows on the embedding.</summary>
    public static void EmbedGather(IntPtr stream, IntPtr w, IntPtr tokens, IntPtr outp, int seq, int dim)
        => GatherK.Value.Launch1DOnStream(stream, seq * dim, 256, w, tokens, outp, seq, dim);

    private static readonly Lazy<CudaKernel> ScatterK = new(() => new CudaKernel(ScatterSrc, "embed_scatter"));
    private const string ScatterSrc = @"
extern ""C"" __global__ void embed_scatter(float* dW, const int* tok, const float* g, int seq, int dim) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)seq * dim) return;
    int i = (int)(idx / dim), j = (int)(idx % dim);
    atomicAdd(&dW[(long)tok[i] * dim + j], g[idx]);
}";

    /// <summary>dW[tokens[i],:] += g[i,:] (atomic — repeated tokens accumulate).</summary>
    public static void EmbedScatterAdd(IntPtr stream, IntPtr dW, IntPtr tokens, IntPtr g, int seq, int dim)
        => ScatterK.Value.Launch1DOnStream(stream, seq * dim, 256, dW, tokens, g, seq, dim);

    // ── Bias ─────────────────────────────────────────────────────────────────

    private static readonly Lazy<CudaKernel> BiasK = new(() => new CudaKernel(BiasSrc, "bias_add"));
    private const string BiasSrc = @"
extern ""C"" __global__ void bias_add(float* x, const float* b, int rows, int dim) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * dim) return;
    x[idx] += b[idx % dim];
}";

    /// <summary>x[i,:] += b (row-broadcast bias).</summary>
    public static void BiasAdd(IntPtr stream, IntPtr x, IntPtr b, int rows, int dim)
        => BiasK.Value.Launch1DOnStream(stream, rows * dim, 256, x, b, rows, dim);

    // Column reductions tile over (256 columns × 64 rows) with a final
    // atomicAdd — coalesced reads and full occupancy even at dim ≪ GPU width
    // (the thread-per-column version left the card ~97% idle).
    private const int ColRowChunk = 64;

    private static readonly Lazy<CudaKernel> ColSumK = new(() => new CudaKernel(ColSumSrc, "col_sum_acc"));
    private const string ColSumSrc = @"
extern ""C"" __global__ void col_sum_acc(float* db, const float* g, int rows, int dim, int colTiles) {
    int colTile = blockIdx.x % colTiles;
    int rowTile = blockIdx.x / colTiles;
    int j = colTile * blockDim.x + threadIdx.x;
    if (j >= dim) return;
    int r0 = rowTile * 64;
    int r1 = r0 + 64; if (r1 > rows) r1 = rows;
    float s = 0.0f;
    for (int i = r0; i < r1; i++) s += g[(long)i * dim + j];
    atomicAdd(&db[j], s);
}";

    /// <summary>db += column sums of g [rows, dim] — the bias gradient (accumulating).</summary>
    public static void ColumnSumAccumulate(IntPtr stream, IntPtr db, IntPtr g, int rows, int dim)
    {
        int colTiles = (dim + 255) / 256;
        int rowTiles = (rows + ColRowChunk - 1) / ColRowChunk;
        ColSumK.Value.LaunchOnStream(stream, (uint)(colTiles * rowTiles), 256, db, g, rows, dim, colTiles);
    }

    // ── GELU (out of place: the backward needs the pre-activation) ───────────

    private static readonly Lazy<CudaKernel> GeluFwdK = new(() => new CudaKernel(GeluFwdSrc, "gelu_fwd"));
    private const string GeluFwdSrc = @"
extern ""C"" __global__ void gelu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.0f + tanhf(t));
    }
}";

    /// <summary>y = gelu(x), out of place (x stays for the backward).</summary>
    public static void GeluForward(IntPtr stream, IntPtr x, IntPtr y, int count)
        => GeluFwdK.Value.Launch1DOnStream(stream, count, 256, x, y, count);

    private static readonly Lazy<CudaKernel> GeluBwdK = new(() => new CudaKernel(GeluBwdSrc, "gelu_bwd_s"));
    private const string GeluBwdSrc = @"
extern ""C"" __global__ void gelu_bwd_s(float* grad, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = 0.7978845608028654f;
        float u = c * (v + 0.044715f * v * v * v);
        float t = tanhf(u);
        float du = c * (1.0f + 3.0f * 0.044715f * v * v);
        grad[i] *= 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du;
    }
}";

    /// <summary>grad ← grad · gelu'(x), in place, on the stream.</summary>
    public static void GeluBackward(IntPtr stream, IntPtr grad, IntPtr x, int count)
        => GeluBwdK.Value.Launch1DOnStream(stream, count, 256, grad, x, count);

    // ── Elementwise add (residuals / gradient fan-in), on the stream ─────────

    private static readonly Lazy<CudaKernel> AddK = new(() => new CudaKernel(AddSrc, "add_inplace_s"));
    private const string AddSrc = @"
extern ""C"" __global__ void add_inplace_s(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}";

    /// <summary>dst += src on the stream (residuals and gradient accumulation).</summary>
    public static void AddInPlace(IntPtr stream, IntPtr dst, IntPtr src, int count)
        => AddK.Value.Launch1DOnStream(stream, count, 256, dst, src, count);

    // ── RMSNorm (training version: out of place, saves 1/rms per row) ────────

    private static readonly Lazy<CudaKernel> RmsFwdK = new(() => new CudaKernel(RmsFwdSrc, "rms_fwd"));
    private const string RmsFwdSrc = @"
extern ""C"" __global__ void rms_fwd(const float* x, const float* gamma, float* y, float* inv,
                                     int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (long)row * dim;
    float* yr = y + (long)row * dim;
    __shared__ float red[256];
    float ss = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) ss += xr[j] * xr[j];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    __shared__ float invSh;
    if (threadIdx.x == 0) { invSh = rsqrtf(red[0] / dim + eps); inv[row] = invSh; }
    __syncthreads();
    for (int j = threadIdx.x; j < dim; j += blockDim.x) yr[j] = xr[j] * invSh * gamma[j];
}";

    /// <summary>y = rmsnorm(x)·γ per row; saves inv[row] = 1/rms for the backward.</summary>
    public static void RmsNormForward(IntPtr stream, IntPtr x, IntPtr gamma, IntPtr y, IntPtr inv,
                                      int rows, int dim, float eps)
        => RmsFwdK.Value.LaunchOnStream(stream, (uint)rows, 256, x, gamma, y, inv, rows, dim, eps);

    private static readonly Lazy<CudaKernel> RmsBwdK = new(() => new CudaKernel(RmsBwdSrc, "rms_bwd_dx"));
    private const string RmsBwdSrc = @"
extern ""C"" __global__ void rms_bwd_dx(const float* dY, const float* x, const float* gamma, const float* inv,
                                        float* dX, int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* g  = dY + (long)row * dim;
    const float* xr = x  + (long)row * dim;
    float* dx = dX + (long)row * dim;
    __shared__ float red[256];
    float s = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) s += g[j] * gamma[j] * xr[j];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float S = red[0];
    float iv = inv[row];
    float inv3 = iv * iv * iv;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        dx[j] = iv * gamma[j] * g[j] - inv3 * xr[j] / dim * S;
}";

    /// <summary>dX = RMSNorm backward (input gradient); needs x, γ and the saved inv.</summary>
    public static void RmsNormBackwardDx(IntPtr stream, IntPtr dY, IntPtr x, IntPtr gamma, IntPtr inv,
                                         IntPtr dX, int rows, int dim)
        => RmsBwdK.Value.LaunchOnStream(stream, (uint)rows, 256, dY, x, gamma, inv, dX, rows, dim);

    private static readonly Lazy<CudaKernel> RmsDgK = new(() => new CudaKernel(RmsDgSrc, "rms_dgamma"));
    private const string RmsDgSrc = @"
extern ""C"" __global__ void rms_dgamma(const float* dY, const float* x, const float* inv,
                                        float* dGamma, int rows, int dim, int colTiles) {
    int colTile = blockIdx.x % colTiles;
    int rowTile = blockIdx.x / colTiles;
    int j = colTile * blockDim.x + threadIdx.x;
    if (j >= dim) return;
    int r0 = rowTile * 64;
    int r1 = r0 + 64; if (r1 > rows) r1 = rows;
    float s = 0.0f;
    for (int i = r0; i < r1; i++) s += dY[(long)i * dim + j] * x[(long)i * dim + j] * inv[i];
    atomicAdd(&dGamma[j], s);
}";

    /// <summary>dγ += Σ_rows dY·x·inv — the RMSNorm gamma gradient (accumulating).</summary>
    public static void RmsNormDgamma(IntPtr stream, IntPtr dY, IntPtr x, IntPtr inv, IntPtr dGamma, int rows, int dim)
    {
        int colTiles = (dim + 255) / 256;
        int rowTiles = (rows + ColRowChunk - 1) / ColRowChunk;
        RmsDgK.Value.LaunchOnStream(stream, (uint)(colTiles * rowTiles), 256, dY, x, inv, dGamma, rows, dim, colTiles);
    }

    // ── RoPE on the packed [seq, heads, headDim] layout, on the stream ───────

    private static readonly Lazy<CudaKernel> RopeK = new(() => new CudaKernel(RopeSrc, "rope_s"));
    private const string RopeSrc = @"
extern ""C"" __global__ void rope_s(float* x, int rows, int heads, int headDim, int basePos, int backward) {
    int half = headDim / 2;
    long total = (long)rows * heads * half;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row  = (int)(idx / ((long)heads * half));
    int rem  = (int)(idx % ((long)heads * half));
    int head = rem / half;
    int i    = rem % half;
    float freq  = 1.0f / powf(10000.0f, (float)(2 * i) / headDim);
    float angle = (basePos + row) * freq;
    float c = cosf(angle), s = sinf(angle);
    if (backward != 0) s = -s;                       // Rᵀ(θ) = R(−θ)
    long off = ((long)row * heads + head) * headDim + 2 * i;
    float x0 = x[off], x1 = x[off + 1];
    x[off]     = x0 * c - x1 * s;
    x[off + 1] = x0 * s + x1 * c;
}";

    /// <summary>In-place RoPE (forward or its transpose) on [seq, heads, headDim], on the stream.</summary>
    public static void Rope(IntPtr stream, IntPtr x, int rows, int heads, int headDim, int basePos, bool backward)
        => RopeK.Value.Launch1DOnStream(stream, rows * heads * (headDim / 2), 256,
                                        x, rows, heads, headDim, basePos, backward ? 1 : 0);

    // ── Causal multi-query attention, fused + tiled (A3) ─────────────────────

    // Tile size: P positions of K/V staged in shared memory per iteration, at a
    // +1 padded stride (headDim+1) so a warp's serial dots hit distinct smem
    // banks. P·headDim is fixed at 4096 floats (16 KB before padding), so P
    // adapts: headDim 64 → P 64, headDim 128 → P 32; capped at 256.
    private static uint AttnBlockDim(int headDim)
        => (uint)Math.Max(headDim, Math.Min(4096 / headDim, 256));

    private static readonly Lazy<CudaKernel> AttnFwdK = new(() => new CudaKernel(AttnFwdSrc, "attn_fwd"));
    private const string AttnFwdSrc = @"
extern ""C"" __global__ void attn_fwd(const float* Q, const float* K, const float* V,
                                      float* O, float* L,
                                      int seq, int heads, int headDim, float scale) {
    int q = blockIdx.x / heads;
    int h = blockIdx.x % heads;
    int tid = threadIdx.x;
    __shared__ float qs[128];
    __shared__ float Kt[4352];                // P × (headDim+1), padded stride
    __shared__ float Vt[4352];
    __shared__ float pr[256];                 // per-position score → prob
    __shared__ float red[256];
    __shared__ float mSh, lSh;
    int P = 4096 / headDim; if (P > 256) P = 256;
    int stride = headDim + 1;

    if (tid < headDim) qs[tid] = Q[((long)q * heads + h) * headDim + tid];
    if (tid == 0) { mSh = -3.402823466e38f; lSh = 0.0f; }
    float acc = 0.0f;                         // component tid of the running context
    __syncthreads();

    int total = q + 1;                        // causal: keys 0..q
    for (int base_ = 0; base_ < total; base_ += P) {
        int cnt = total - base_; if (cnt > P) cnt = P;
        // coalesced cooperative load of the K/V tile (padded rows)
        for (int idx = tid; idx < cnt * headDim; idx += blockDim.x) {
            int p = idx / headDim, i = idx % headDim;
            long off = ((long)(base_ + p) * heads + h) * headDim + i;
            Kt[p * stride + i] = K[off];
            Vt[p * stride + i] = V[off];
        }
        __syncthreads();
        // one thread per tile position: serial dot from smem, no syncs
        float s = -3.402823466e38f;
        if (tid < cnt) {
            float d = 0.0f;
            for (int i = 0; i < headDim; i++) d += qs[i] * Kt[tid * stride + i];
            s = d * scale;
        }
        red[tid] = s;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {                 // tile max
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        float mOld = mSh;
        float mNew = fmaxf(mOld, red[0]);
        float p_ = tid < cnt ? expf(s - mNew) : 0.0f;
        pr[tid] = p_;
        red[tid] = p_;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {                 // tile sum
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        float corr = expf(mOld - mNew);
        if (tid == 0) { lSh = lSh * corr + red[0]; mSh = mNew; }
        if (tid < headDim) {                  // context update, component tid
            float add = 0.0f;
            for (int p = 0; p < cnt; p++) add += pr[p] * Vt[p * stride + tid];
            acc = acc * corr + add;
        }
        __syncthreads();
    }
    if (tid < headDim) {
        float l = lSh;
        O[((long)q * heads + h) * headDim + tid] = l > 0.0f ? acc / l : 0.0f;
    }
    if (tid == 0) L[(long)q * heads + h] = mSh + logf(lSh);
}";

    /// <summary>
    /// Fused causal attention forward, tiled: block per (query, head), K/V
    /// staged in shared memory P positions at a time — scores are per-thread
    /// serial dots from smem (padded stride, bank-conflict free), and the
    /// online-softmax reduction runs once per TILE instead of once per
    /// position. Writes O and the per-row logsumexp L (the backward replays
    /// weights from it).
    /// </summary>
    public static void AttentionForward(IntPtr stream, IntPtr q, IntPtr k, IntPtr v, IntPtr o, IntPtr l,
                                        int seq, int heads, int headDim, float scale)
    {
        RequireHeadDim(headDim);
        AttnFwdK.Value.LaunchOnStream(stream, (uint)(seq * heads), AttnBlockDim(headDim), q, k, v, o, l, seq, heads, headDim, scale);
    }

    private static readonly Lazy<CudaKernel> AttnDotK = new(() => new CudaKernel(AttnDotSrc, "attn_dot"));
    private const string AttnDotSrc = @"
extern ""C"" __global__ void attn_dot(const float* dO, const float* O, float* D, int seq, int heads, int headDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq * heads) return;
    const float* a = dO + (long)idx * headDim;
    const float* b = O  + (long)idx * headDim;
    float s = 0.0f;
    for (int i = 0; i < headDim; i++) s += a[i] * b[i];
    D[idx] = s;
}";

    /// <summary>D[q,h] = dot(dO, O) — the flash-backward correction term.</summary>
    public static void AttentionDotDoO(IntPtr stream, IntPtr dO, IntPtr o, IntPtr d, int seq, int heads, int headDim)
        => AttnDotK.Value.Launch1DOnStream(stream, seq * heads, 128, dO, o, d, seq, heads, headDim);

    private static readonly Lazy<CudaKernel> AttnBwdQK = new(() => new CudaKernel(AttnBwdQSrc, "attn_bwd_dq"));
    private const string AttnBwdQSrc = @"
extern ""C"" __global__ void attn_bwd_dq(const float* Q, const float* K, const float* V,
                                         const float* dO, const float* L, const float* D,
                                         float* dQ, int seq, int heads, int headDim, float scale) {
    int q = blockIdx.x / heads;
    int h = blockIdx.x % heads;
    int tid = threadIdx.x;
    __shared__ float qs[128];
    __shared__ float dos[128];
    __shared__ float Kt[4352];
    __shared__ float Vt[4352];
    __shared__ float ds[256];                 // per-position dS
    int P = 4096 / headDim; if (P > 256) P = 256;
    int stride = headDim + 1;

    long rowOff = ((long)q * heads + h) * headDim;
    if (tid < headDim) { qs[tid] = Q[rowOff + tid]; dos[tid] = dO[rowOff + tid]; }
    float Lq = L[(long)q * heads + h];
    float Dq = D[(long)q * heads + h];
    float acc = 0.0f;                         // component tid of dq
    __syncthreads();

    int total = q + 1;
    for (int base_ = 0; base_ < total; base_ += P) {
        int cnt = total - base_; if (cnt > P) cnt = P;
        for (int idx = tid; idx < cnt * headDim; idx += blockDim.x) {
            int p = idx / headDim, i = idx % headDim;
            long off = ((long)(base_ + p) * heads + h) * headDim + i;
            Kt[p * stride + i] = K[off];
            Vt[p * stride + i] = V[off];
        }
        __syncthreads();
        if (tid < cnt) {                      // one thread per tile position
            float s = 0.0f, dA = 0.0f;
            for (int i = 0; i < headDim; i++) {
                s  += qs[i]  * Kt[tid * stride + i];
                dA += dos[i] * Vt[tid * stride + i];
            }
            float a = expf(s * scale - Lq);   // softmax weight replayed from L
            ds[tid] = a * (dA - Dq);
        } else ds[tid] = 0.0f;
        __syncthreads();
        if (tid < headDim) {                  // dq update, component tid
            float add = 0.0f;
            for (int p = 0; p < cnt; p++) add += ds[p] * Kt[p * stride + tid];
            acc += add;
        }
        __syncthreads();
    }
    if (tid < headDim) dQ[rowOff + tid] = scale * acc;
}";

    private static readonly Lazy<CudaKernel> AttnBwdKVK = new(() => new CudaKernel(AttnBwdKVSrc, "attn_bwd_dkv"));
    private const string AttnBwdKVSrc = @"
extern ""C"" __global__ void attn_bwd_dkv(const float* Q, const float* K, const float* V,
                                          const float* dO, const float* L, const float* D,
                                          float* dK, float* dV, int seq, int heads, int headDim, float scale) {
    int p = blockIdx.x / heads;
    int h = blockIdx.x % heads;
    int tid = threadIdx.x;
    __shared__ float ks[128];
    __shared__ float vs[128];
    __shared__ float Qt[4352];
    __shared__ float dOt[4352];
    __shared__ float aArr[256];
    __shared__ float dsArr[256];
    int P = 4096 / headDim; if (P > 256) P = 256;
    int stride = headDim + 1;

    long pOff = ((long)p * heads + h) * headDim;
    if (tid < headDim) { ks[tid] = K[pOff + tid]; vs[tid] = V[pOff + tid]; }
    float dka = 0.0f, dva = 0.0f;             // components tid of dk, dv
    __syncthreads();

    for (int base_ = p; base_ < seq; base_ += P) {
        int cnt = seq - base_; if (cnt > P) cnt = P;
        for (int idx = tid; idx < cnt * headDim; idx += blockDim.x) {
            int t = idx / headDim, i = idx % headDim;
            long off = ((long)(base_ + t) * heads + h) * headDim + i;
            Qt[t * stride + i]  = Q[off];
            dOt[t * stride + i] = dO[off];
        }
        __syncthreads();
        if (tid < cnt) {                      // one thread per tile query
            int qq = base_ + tid;
            float s = 0.0f, dA = 0.0f;
            for (int i = 0; i < headDim; i++) {
                s  += Qt[tid * stride + i]  * ks[i];
                dA += dOt[tid * stride + i] * vs[i];
            }
            float a = expf(s * scale - L[(long)qq * heads + h]);
            aArr[tid]  = a;
            dsArr[tid] = a * (dA - D[(long)qq * heads + h]);
        } else { aArr[tid] = 0.0f; dsArr[tid] = 0.0f; }
        __syncthreads();
        if (tid < headDim) {                  // dk/dv update, component tid
            float addK = 0.0f, addV = 0.0f;
            for (int t = 0; t < cnt; t++) {
                addK += dsArr[t] * Qt[t * stride + tid];
                addV += aArr[t]  * dOt[t * stride + tid];
            }
            dka += addK; dva += addV;
        }
        __syncthreads();
    }
    if (tid < headDim) { dK[pOff + tid] = scale * dka; dV[pOff + tid] = dva; }
}";

    /// <summary>
    /// Fused causal attention backward (flash style, tiled like the forward):
    /// replays the softmax weights from the saved logsumexp — dQ from a block
    /// per (query, head), dK/dV from a block per (key, head). Call
    /// <see cref="AttentionDotDoO"/> first for D.
    /// </summary>
    public static void AttentionBackward(IntPtr stream, IntPtr q, IntPtr k, IntPtr v, IntPtr dO,
                                         IntPtr l, IntPtr d, IntPtr dQ, IntPtr dK, IntPtr dV,
                                         int seq, int heads, int headDim, float scale)
    {
        RequireHeadDim(headDim);
        AttnBwdQK.Value.LaunchOnStream(stream, (uint)(seq * heads), AttnBlockDim(headDim),
                                       q, k, v, dO, l, d, dQ, seq, heads, headDim, scale);
        AttnBwdKVK.Value.LaunchOnStream(stream, (uint)(seq * heads), AttnBlockDim(headDim),
                                        q, k, v, dO, l, d, dK, dV, seq, heads, headDim, scale);
    }

    // ── Fused softmax cross-entropy: loss per row + dLogits in one pass ──────

    private static readonly Lazy<CudaKernel> CeK = new(() => new CudaKernel(CeSrc, "ce_loss_grad"));
    private const string CeSrc = @"
extern ""C"" __global__ void ce_loss_grad(const float* logits, const int* targets,
                                          float* dLogits, float* lossRows,
                                          int rows, int vocab, float invRows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* lr = logits + (long)row * vocab;
    float* dr = dLogits + (long)row * vocab;
    __shared__ float red[256];
    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) m = fmaxf(m, lr[j]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads();
    float s = 0.0f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) s += expf(lr[j] - mx);
    red[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float sum = red[0];
    float logSum = logf(sum);
    int t = targets[row];
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        float p = expf(lr[j] - mx) / sum;
        dr[j] = (p - (j == t ? 1.0f : 0.0f)) * invRows;
    }
    if (threadIdx.x == 0) lossRows[row] = -(lr[t] - mx - logSum);
}";

    /// <summary>
    /// Fused softmax cross-entropy: per row of logits [rows, vocab] writes the
    /// mean-loss gradient dLogits = (softmax − onehot(target))/rows and the row
    /// loss −log p(target). One kernel, no [rows, vocab] probability tape.
    /// </summary>
    public static void CrossEntropyLossGrad(IntPtr stream, IntPtr logits, IntPtr targets,
                                            IntPtr dLogits, IntPtr lossRows, int rows, int vocab)
        => CeK.Value.LaunchOnStream(stream, (uint)rows, 256, logits, targets, dLogits, lossRows, rows, vocab, 1f / rows);

    // ── bf16 conversions (A2 mixed compute) ──────────────────────────────────
    //
    // bf16 is the top 16 bits of an IEEE float (same exponent range as fp32 —
    // no loss scaling needed). Round-to-nearest-even done by hand, so no CUDA
    // headers are required under NVRTC.

    private static readonly Lazy<CudaKernel> F32ToBf16K = new(() => new CudaKernel(F32ToBf16Src, "f32_to_bf16"));
    private const string F32ToBf16Src = @"
extern ""C"" __global__ void f32_to_bf16(const float* src, unsigned short* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int u = __float_as_uint(src[i]);
    u += 0x7FFFu + ((u >> 16) & 1u);          // round to nearest even
    dst[i] = (unsigned short)(u >> 16);
}";

    /// <summary>dst (bf16) = src (fp32), round-to-nearest-even.</summary>
    public static void FloatToBf16(IntPtr stream, IntPtr src, IntPtr dst, int count)
        => F32ToBf16K.Value.Launch1DOnStream(stream, count, 256, src, dst, count);

    private static readonly Lazy<CudaKernel> Bf16ToF32K = new(() => new CudaKernel(Bf16ToF32Src, "bf16_to_f32"));
    private const string Bf16ToF32Src = @"
extern ""C"" __global__ void bf16_to_f32(const unsigned short* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __uint_as_float(((unsigned int)src[i]) << 16);
}";

    /// <summary>dst (fp32) = src (bf16) — exact (bf16 ⊂ fp32).</summary>
    public static void Bf16ToFloat(IntPtr stream, IntPtr src, IntPtr dst, int count)
        => Bf16ToF32K.Value.Launch1DOnStream(stream, count, 256, src, dst, count);

    // ── AdamW, on device (the exact CPU Optimizers.cs maths) ─────────────────

    private static readonly Lazy<CudaKernel> AdamK = new(() => new CudaKernel(AdamSrc, "adam_step"));
    private const string AdamSrc = @"
extern ""C"" __global__ void adam_step(float* w, const float* g, float* m, float* v, int n,
                                       float lr, float beta1, float beta2, float eps,
                                       float weightDecay, float bc1, float bc2, float gradScale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i] * gradScale;
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi; v[i] = vi;
    float mHat = mi / bc1;
    float vHat = vi / bc2;
    float wi = w[i] - lr * mHat / (sqrtf(vHat) + eps);
    if (weightDecay != 0.0f) wi -= lr * weightDecay * wi;
    w[i] = wi;
}";

    /// <summary>
    /// One AdamW update on a device parameter buffer — decoupled weight decay,
    /// bias corrections bc1/bc2 = 1−βᵗ computed by the host, gradScale folds the
    /// micro-batch mean (1/windows) into the read. Mirrors
    /// <c>PredictiveCoding.LLM.Training.Adam</c> exactly.
    /// </summary>
    public static void AdamStep(IntPtr stream, IntPtr w, IntPtr g, IntPtr m, IntPtr v, int count,
                                float lr, float beta1, float beta2, float eps, float weightDecay,
                                float bc1, float bc2, float gradScale = 1f)
        => AdamK.Value.Launch1DOnStream(stream, count, 256, w, g, m, v, count, lr, beta1, beta2, eps, weightDecay, bc1, bc2, gradScale);
}
