using System.Runtime.InteropServices;
using PredictiveCoding.Device.Allocators;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Kernels;
using PredictiveCoding.Device.Streams;
using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.Device.Training;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceDecoderTrainer  —  the ROADMAP-Scale A1 wiring: a full training step of
// the DecoderLM resident on the GPU.
//
// Weights, Adam moments, gradients and every activation live in device buffers
// allocated ONCE (the arena) and reused every step; a Step() is one unbroken
// chain of kernel + cuBLAS launches on a single stream, and the host blocks
// exactly once — to read the scalar loss. The per-op host round-trips that made
// "just enable CUDA" a lie are gone by construction.
//
// Supported config: the Scale preset family — dense attention blocks (no MoE /
// loops / MTP / plan / temporal), RMSNorm + RoPE, dropout 0. The CPU DecoderLM
// with the same config is the reference twin: parity is asserted by test, and
// ImportFrom/ExportTo move weights losslessly in both directions.
//
// Head packing: the CPU model keeps per-head Wq/Wk/Wv [dim, headDim] and
// Wo [headDim, dim]. On device the heads are packed — Wq/Wk/Wv concatenate as
// columns into [dim, dim] (so n1·Wq_all yields [seq, heads, headDim] rows) and
// Wo stacks as rows into [dim, dim] (so attnO·Wo_stack sums the heads) — the
// exact same maths, one GEMM per projection instead of one per head.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class DeviceDecoderTrainer : IDisposable
{
    private readonly DecoderLmConfig _cfg;
    private readonly int _deviceId;
    private readonly CudaStreamHandle _stream;
    private readonly IDeviceAllocator _alloc = CudaAllocator.Shared;
    private readonly CudaGemmProvider _gemm;

    private readonly int _maxSeq, _dim, _ffn, _vocab, _heads, _headDim, _blocks;
    private readonly float _attnScale;

    // AdamW state (identical to LLM.Training.Adam defaults).
    private readonly float _beta1, _beta2, _epsAdam, _weightDecay;
    private int _t;

    /// <summary>Cumulative token positions trained on this device instance.</summary>
    public long TokensSeen { get; private set; }

    /// <summary>Total device memory this trainer allocated (weights + Adam + arena), in bytes.</summary>
    public long AllocatedBytes { get; private set; }

    // ── Parameter buffers ────────────────────────────────────────────────────

    private sealed class Param
    {
        public IntPtr W, G, M, V;
        public int Count;
    }

    private readonly Param _wemb, _gF, _wout;
    private sealed class BlockParams
    {
        public Param G1 = null!, Wq = null!, Wk = null!, Wv = null!, Wo = null!;
        public Param G2 = null!, W1 = null!, B1 = null!, W2 = null!, B2 = null!;
        public IEnumerable<Param> All() { yield return G1; yield return Wq; yield return Wk; yield return Wv; yield return Wo; yield return G2; yield return W1; yield return B1; yield return W2; yield return B2; }
    }
    private readonly BlockParams[] _bp;

    // ── Activation arena (sized once for MaxSeq, reused every step) ──────────

    private sealed class BlockActs
    {
        public IntPtr N1, Q, K, V, AttnO, XMid, N2, FfPre, FfAct;   // forward tape
        public IntPtr Inv1, Inv2, L;                                // [S], [S], [S,H]
    }
    private readonly BlockActs[] _acts;
    private readonly IntPtr[] _x;                       // B+1 block inputs [S,D]
    private IntPtr _nF, _invF, _logits, _dLogits, _lossRows;
    private IntPtr _dX, _dXAlt, _dN, _dQ, _dK, _dV, _dAttnO, _dFf, _dot;   // backward workspace
    private IntPtr _tokDev, _tgtDev;                    // int[S]
    private IntPtr _tokPin, _tgtPin, _lossPin;          // pinned host staging
    private readonly float[] _lossHost;                 // reusable managed view of _lossPin

    private readonly List<(IntPtr ptr, nuint bytes)> _owned = new();
    private bool _disposed;

    /// <summary>True when this config can run on the device path (and a GPU is present).</summary>
    public static bool Supports(DecoderLmConfig cfg) =>
        DeviceRuntime.CudaAvailable
        && cfg.UseRmsNorm && cfg.UseRope
        && cfg.BlockPattern is null && cfg.NumExperts == 0 && cfg.NumLoops == 1
        && cfg.MtpWeight == 0f && !cfg.UsePlan && cfg.Dropout == 0f
        && cfg.HeadDim >= 2 && cfg.HeadDim <= DeviceTrainKernels.MaxHeadDim
        && (cfg.HeadDim & (cfg.HeadDim - 1)) == 0;

    public DeviceDecoderTrainer(DecoderLM model, float beta1 = 0.9f, float beta2 = 0.999f,
                                float epsAdam = 1e-8f, float weightDecay = 0f, int deviceId = 0)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!Supports(model.Config))
            throw new NotSupportedException(
                "DeviceDecoderTrainer needs a CUDA GPU and a dense RMSNorm+RoPE config (the Scale presets).");

        _cfg = model.Config;
        _deviceId = deviceId;
        _gemm = new CudaGemmProvider(deviceId);
        _stream = CudaStreamManager.Shared.ThreadStream(deviceId);
        _beta1 = beta1; _beta2 = beta2; _epsAdam = epsAdam; _weightDecay = weightDecay;

        _maxSeq = _cfg.MaxSeq; _dim = _cfg.Dim; _ffn = _cfg.FfDim; _vocab = _cfg.Vocab;
        _heads = _cfg.NumHeads; _headDim = _cfg.HeadDim; _blocks = _cfg.NumBlocks;
        _attnScale = 1f / MathF.Sqrt(_headDim);

        // Parameters (W + G + Adam M/V each).
        _wemb = NewParam(_vocab * _dim);
        _bp   = new BlockParams[_blocks];
        for (int b = 0; b < _blocks; b++)
            _bp[b] = new BlockParams
            {
                G1 = NewParam(_dim),        Wq = NewParam(_dim * _dim), Wk = NewParam(_dim * _dim),
                Wv = NewParam(_dim * _dim), Wo = NewParam(_dim * _dim),
                G2 = NewParam(_dim),        W1 = NewParam(_dim * _ffn), B1 = NewParam(_ffn),
                W2 = NewParam(_ffn * _dim), B2 = NewParam(_dim),
            };
        _gF   = NewParam(_dim);
        _wout = NewParam(_dim * _vocab);

        // Activation arena.
        int S = _maxSeq, D = _dim, F = _ffn, H = _heads;
        _acts = new BlockActs[_blocks];
        for (int b = 0; b < _blocks; b++)
            _acts[b] = new BlockActs
            {
                N1 = Buf(S * D), Q = Buf(S * D), K = Buf(S * D), V = Buf(S * D),
                AttnO = Buf(S * D), XMid = Buf(S * D), N2 = Buf(S * D),
                FfPre = Buf(S * F), FfAct = Buf(S * F),
                Inv1 = Buf(S), Inv2 = Buf(S), L = Buf(S * H),
            };
        _x = new IntPtr[_blocks + 1];
        for (int b = 0; b <= _blocks; b++) _x[b] = Buf(S * D);
        _nF = Buf(S * D); _invF = Buf(S);
        _logits = Buf(S * _vocab); _dLogits = Buf(S * _vocab); _lossRows = Buf(S);
        _dX = Buf(S * D); _dXAlt = Buf(S * D); _dN = Buf(S * D);
        _dQ = Buf(S * D); _dK = Buf(S * D); _dV = Buf(S * D);
        _dAttnO = Buf(S * D); _dFf = Buf(S * F); _dot = Buf(S * H);
        _tokDev = Buf(S); _tgtDev = Buf(S);

        _tokPin  = _alloc.AllocatePinnedHost((nuint)(S * sizeof(int)));
        _tgtPin  = _alloc.AllocatePinnedHost((nuint)(S * sizeof(int)));
        _lossPin = _alloc.AllocatePinnedHost((nuint)(S * sizeof(float)));
        _lossHost = new float[S];

        ImportFrom(model);
        ZeroGrads();
    }

    private Param NewParam(int count)
    {
        var p = new Param { Count = count, W = Buf(count), G = Buf(count), M = Buf(count), V = Buf(count) };
        Memset(p.M, count); Memset(p.V, count);
        return p;
    }

    private IntPtr Buf(int floats)
    {
        nuint bytes = (nuint)floats * sizeof(float);
        IntPtr p = _alloc.Allocate(_deviceId, bytes, _stream);
        _owned.Add((p, bytes));
        AllocatedBytes += (long)bytes;
        return p;
    }

    private void Memset(IntPtr p, int floats)
        => NativeCheck.Ok(Cuda.cudaMemsetAsync(p, 0, (nuint)floats * sizeof(float), _stream.Handle), "memsetAsync");

    // ── Weight transfer (CPU per-head layout ⇄ device packed layout) ─────────

    private IEnumerable<Param> AllParams()
    {
        yield return _wemb;
        foreach (var bp in _bp) foreach (var p in bp.All()) yield return p;
        yield return _gF;
        yield return _wout;
    }

    /// <summary>Uploads the model's weights into the device buffers (packing the heads).</summary>
    public void ImportFrom(DecoderLM model)
    {
        var ps = model.Parameters();
        int i = 0;
        Tensor Next() => ps[i++];

        Upload(_wemb.W, Next().Data);                                  // wemb
        for (int b = 0; b < _blocks; b++)
        {
            Upload(_bp[b].G1.W, Next().Data);                          // ln1 gamma
            var wq = new Tensor[_heads]; var wk = new Tensor[_heads];
            var wv = new Tensor[_heads]; var wo = new Tensor[_heads];
            for (int h = 0; h < _heads; h++) { wq[h] = Next(); wk[h] = Next(); wv[h] = Next(); wo[h] = Next(); }
            Upload(_bp[b].Wq.W, PackColumns(wq)); Upload(_bp[b].Wk.W, PackColumns(wk));
            Upload(_bp[b].Wv.W, PackColumns(wv)); Upload(_bp[b].Wo.W, PackRows(wo));
            Upload(_bp[b].G2.W, Next().Data);                          // ln2 gamma
            Upload(_bp[b].W1.W, Next().Data); Upload(_bp[b].B1.W, Next().Data);
            Upload(_bp[b].W2.W, Next().Data); Upload(_bp[b].B2.W, Next().Data);
        }
        Upload(_gF.W, Next().Data);                                    // final gamma
        Upload(_wout.W, Next().Data);                                  // wout
        if (i != ps.Length) throw new InvalidOperationException($"Parameter walk mismatch: {i} vs {ps.Length}.");
        CudaStreamManager.Shared.Synchronize(_stream);
    }

    /// <summary>Downloads the device weights back into the model (unpacking the heads).</summary>
    public void ExportTo(DecoderLM model)
    {
        CudaStreamManager.Shared.Synchronize(_stream);
        var ps = model.Parameters();
        int i = 0;
        Tensor Next() => ps[i++];

        Download(_wemb.W, Next().Data);
        for (int b = 0; b < _blocks; b++)
        {
            Download(_bp[b].G1.W, Next().Data);
            var packedQ = new float[_dim * _dim]; Download(_bp[b].Wq.W, packedQ);
            var packedK = new float[_dim * _dim]; Download(_bp[b].Wk.W, packedK);
            var packedV = new float[_dim * _dim]; Download(_bp[b].Wv.W, packedV);
            var packedO = new float[_dim * _dim]; Download(_bp[b].Wo.W, packedO);
            for (int h = 0; h < _heads; h++)
            {
                UnpackColumn(packedQ, Next().Data, h); UnpackColumn(packedK, Next().Data, h);
                UnpackColumn(packedV, Next().Data, h); UnpackRow(packedO, Next().Data, h);
            }
            Download(_bp[b].G2.W, Next().Data);
            Download(_bp[b].W1.W, Next().Data); Download(_bp[b].B1.W, Next().Data);
            Download(_bp[b].W2.W, Next().Data); Download(_bp[b].B2.W, Next().Data);
        }
        Download(_gF.W, Next().Data);
        Download(_wout.W, Next().Data);
        if (i != ps.Length) throw new InvalidOperationException($"Parameter walk mismatch: {i} vs {ps.Length}.");
    }

    // Per-head [dim, headDim] blocks → side-by-side columns of [dim, heads·headDim].
    private float[] PackColumns(Tensor[] perHead)
    {
        var packed = new float[_dim * _dim];
        for (int h = 0; h < _heads; h++)
            for (int r = 0; r < _dim; r++)
                Array.Copy(perHead[h].Data, r * _headDim, packed, r * _dim + h * _headDim, _headDim);
        return packed;
    }

    // Per-head [headDim, dim] blocks → stacked rows of [heads·headDim, dim].
    private float[] PackRows(Tensor[] perHead)
    {
        var packed = new float[_dim * _dim];
        for (int h = 0; h < _heads; h++)
            Array.Copy(perHead[h].Data, 0, packed, h * _headDim * _dim, _headDim * _dim);
        return packed;
    }

    private void UnpackColumn(float[] packed, float[] dst, int h)
    {
        for (int r = 0; r < _dim; r++)
            Array.Copy(packed, r * _dim + h * _headDim, dst, r * _headDim, _headDim);
    }

    private void UnpackRow(float[] packed, float[] dst, int h)
        => Array.Copy(packed, h * _headDim * _dim, dst, 0, _headDim * _dim);

    private void Upload(IntPtr dst, float[] src)
    {
        var hh = GCHandle.Alloc(src, GCHandleType.Pinned);
        try { NativeCheck.Ok(Cuda.cudaMemcpyAsync(dst, hh.AddrOfPinnedObject(), (nuint)src.Length * sizeof(float), (int)MemcpyKind.HostToDevice, _stream.Handle), "h2d"); CudaStreamManager.Shared.Synchronize(_stream); }
        finally { hh.Free(); }
    }

    private void Download(IntPtr src, float[] dst)
    {
        var hh = GCHandle.Alloc(dst, GCHandleType.Pinned);
        try { NativeCheck.Ok(Cuda.cudaMemcpy(hh.AddrOfPinnedObject(), src, (nuint)dst.Length * sizeof(float), (int)MemcpyKind.DeviceToHost), "d2h"); }
        finally { hh.Free(); }
    }

    // ── The step ─────────────────────────────────────────────────────────────

    /// <summary>
    /// One AdamW training step on a single window: forward, fused-attention
    /// backward and the optimizer update — all queued on one stream, weights
    /// never leaving the card. Returns the mean cross-entropy of the window.
    /// </summary>
    public float Step(int[] tokens, int[] targets, float lr)
    {
        float loss = Accumulate(tokens, targets);
        ApplyAdam(lr, gradScale: 1f);
        return loss;
    }

    /// <summary>
    /// Gradient-accumulating step: every window's gradients sum on device, the
    /// single AdamW update reads them scaled by 1/windows (the micro-batch
    /// mean). Returns the mean loss across windows.
    /// </summary>
    public float Step(IReadOnlyList<(int[] tokens, int[] targets)> windows, float lr)
    {
        if (windows.Count == 0) throw new ArgumentException("Need at least one window.");
        float sum = 0f;
        foreach (var (tok, tgt) in windows) sum += Accumulate(tok, tgt);
        ApplyAdam(lr, gradScale: 1f / windows.Count);
        return sum / windows.Count;
    }

    // Forward + backward of one window; gradients ACCUMULATE into the param G buffers.
    private float Accumulate(int[] tokens, int[] targets)
    {
        int S = tokens.Length;
        if (S < 2 || S > _maxSeq) throw new ArgumentOutOfRangeException(nameof(tokens), $"seq {S} outside [2, {_maxSeq}].");
        if (targets.Length != S) throw new ArgumentException("tokens/targets length mismatch.");

        var stream = _stream; IntPtr sh = stream.Handle;

        // Tokens/targets in via pinned staging (the only host→device data of the step).
        Marshal.Copy(tokens, 0, _tokPin, S);
        Marshal.Copy(targets, 0, _tgtPin, S);
        NativeCheck.Ok(Cuda.cudaMemcpyAsync(_tokDev, _tokPin, (nuint)(S * sizeof(int)), (int)MemcpyKind.HostToDevice, sh), "tok h2d");
        NativeCheck.Ok(Cuda.cudaMemcpyAsync(_tgtDev, _tgtPin, (nuint)(S * sizeof(int)), (int)MemcpyKind.HostToDevice, sh), "tgt h2d");

        // ── Forward ──
        DeviceTrainKernels.EmbedGather(sh, _wemb.W, _tokDev, _x[0], S, _dim);
        for (int b = 0; b < _blocks; b++)
        {
            var a = _acts[b]; var p = _bp[b];
            DeviceTrainKernels.RmsNormForward(sh, _x[b], p.G1.W, a.N1, a.Inv1, S, _dim, _cfg.Eps);
            _gemm.MatMulDeviceEx(a.N1, p.Wq.W, a.Q, S, _dim, _dim, false, false, 0f, stream);
            _gemm.MatMulDeviceEx(a.N1, p.Wk.W, a.K, S, _dim, _dim, false, false, 0f, stream);
            _gemm.MatMulDeviceEx(a.N1, p.Wv.W, a.V, S, _dim, _dim, false, false, 0f, stream);
            DeviceTrainKernels.Rope(sh, a.Q, S, _heads, _headDim, 0, backward: false);
            DeviceTrainKernels.Rope(sh, a.K, S, _heads, _headDim, 0, backward: false);
            DeviceTrainKernels.AttentionForward(sh, a.Q, a.K, a.V, a.AttnO, a.L, S, _heads, _headDim, _attnScale);

            // xMid = x + attnO·Wo   (copy then GEMM with β=1: the residual for free)
            NativeCheck.Ok(Cuda.cudaMemcpyAsync(a.XMid, _x[b], (nuint)(S * _dim * sizeof(float)), (int)MemcpyKind.DeviceToDevice, sh), "resid d2d");
            _gemm.MatMulDeviceEx(a.AttnO, p.Wo.W, a.XMid, S, _dim, _dim, false, false, 1f, stream);

            DeviceTrainKernels.RmsNormForward(sh, a.XMid, p.G2.W, a.N2, a.Inv2, S, _dim, _cfg.Eps);
            _gemm.MatMulDeviceEx(a.N2, p.W1.W, a.FfPre, S, _dim, _ffn, false, false, 0f, stream);
            DeviceTrainKernels.BiasAdd(sh, a.FfPre, p.B1.W, S, _ffn);
            DeviceTrainKernels.GeluForward(sh, a.FfPre, a.FfAct, S * _ffn);

            // x_{b+1} = xMid + ffAct·W2 + b2
            NativeCheck.Ok(Cuda.cudaMemcpyAsync(_x[b + 1], a.XMid, (nuint)(S * _dim * sizeof(float)), (int)MemcpyKind.DeviceToDevice, sh), "resid d2d");
            _gemm.MatMulDeviceEx(a.FfAct, p.W2.W, _x[b + 1], S, _ffn, _dim, false, false, 1f, stream);
            DeviceTrainKernels.BiasAdd(sh, _x[b + 1], p.B2.W, S, _dim);
        }
        DeviceTrainKernels.RmsNormForward(sh, _x[_blocks], _gF.W, _nF, _invF, S, _dim, _cfg.Eps);
        _gemm.MatMulDeviceEx(_nF, _wout.W, _logits, S, _dim, _vocab, false, false, 0f, stream);
        DeviceTrainKernels.CrossEntropyLossGrad(sh, _logits, _tgtDev, _dLogits, _lossRows, S, _vocab);

        // ── Backward ──
        _gemm.MatMulDeviceEx(_nF, _dLogits, _wout.G, _dim, S, _vocab, true, false, 1f, stream);      // dWout += nFᵀ·dL
        _gemm.MatMulDeviceEx(_dLogits, _wout.W, _dN, S, _vocab, _dim, false, true, 0f, stream);      // dnF = dL·Woutᵀ
        DeviceTrainKernels.RmsNormDgamma(sh, _dN, _x[_blocks], _invF, _gF.G, S, _dim);
        DeviceTrainKernels.RmsNormBackwardDx(sh, _dN, _x[_blocks], _gF.W, _invF, _dX, S, _dim);

        for (int b = _blocks - 1; b >= 0; b--)
        {
            var a = _acts[b]; var p = _bp[b];

            // FFN half: x_{b+1} = xMid + gelu(n2·W1 + b1)·W2 + b2, upstream grad in _dX.
            DeviceTrainKernels.ColumnSumAccumulate(sh, p.B2.G, _dX, S, _dim);
            _gemm.MatMulDeviceEx(a.FfAct, _dX, p.W2.G, _ffn, S, _dim, true, false, 1f, stream);      // dW2 += ffActᵀ·dX
            _gemm.MatMulDeviceEx(_dX, p.W2.W, _dFf, S, _dim, _ffn, false, true, 0f, stream);         // dFfAct = dX·W2ᵀ
            DeviceTrainKernels.GeluBackward(sh, _dFf, a.FfPre, S * _ffn);                            // → dFfPre (in place)
            DeviceTrainKernels.ColumnSumAccumulate(sh, p.B1.G, _dFf, S, _ffn);
            _gemm.MatMulDeviceEx(a.N2, _dFf, p.W1.G, _dim, S, _ffn, true, false, 1f, stream);        // dW1 += n2ᵀ·dFfPre
            _gemm.MatMulDeviceEx(_dFf, p.W1.W, _dN, S, _ffn, _dim, false, true, 0f, stream);         // dn2 = dFfPre·W1ᵀ
            DeviceTrainKernels.RmsNormDgamma(sh, _dN, a.XMid, a.Inv2, p.G2.G, S, _dim);
            DeviceTrainKernels.RmsNormBackwardDx(sh, _dN, a.XMid, p.G2.W, a.Inv2, _dXAlt, S, _dim);
            DeviceTrainKernels.AddInPlace(sh, _dXAlt, _dX, S * _dim);                                // dXMid = residual + norm path

            // Attention half: xMid = x_b + attnO·Wo, upstream grad now in _dXAlt.
            _gemm.MatMulDeviceEx(a.AttnO, _dXAlt, p.Wo.G, _dim, S, _dim, true, false, 1f, stream);   // dWo += attnOᵀ·dXMid
            _gemm.MatMulDeviceEx(_dXAlt, p.Wo.W, _dAttnO, S, _dim, _dim, false, true, 0f, stream);   // dAttnO
            DeviceTrainKernels.AttentionDotDoO(sh, _dAttnO, a.AttnO, _dot, S, _heads, _headDim);
            DeviceTrainKernels.AttentionBackward(sh, a.Q, a.K, a.V, _dAttnO, a.L, _dot, _dQ, _dK, _dV, S, _heads, _headDim, _attnScale);
            DeviceTrainKernels.Rope(sh, _dQ, S, _heads, _headDim, 0, backward: true);
            DeviceTrainKernels.Rope(sh, _dK, S, _heads, _headDim, 0, backward: true);
            _gemm.MatMulDeviceEx(a.N1, _dQ, p.Wq.G, _dim, S, _dim, true, false, 1f, stream);         // dWq += n1ᵀ·dQ
            _gemm.MatMulDeviceEx(a.N1, _dK, p.Wk.G, _dim, S, _dim, true, false, 1f, stream);
            _gemm.MatMulDeviceEx(a.N1, _dV, p.Wv.G, _dim, S, _dim, true, false, 1f, stream);
            _gemm.MatMulDeviceEx(_dQ, p.Wq.W, _dN, S, _dim, _dim, false, true, 0f, stream);          // dn1  = dQ·Wqᵀ
            _gemm.MatMulDeviceEx(_dK, p.Wk.W, _dN, S, _dim, _dim, false, true, 1f, stream);          //      + dK·Wkᵀ
            _gemm.MatMulDeviceEx(_dV, p.Wv.W, _dN, S, _dim, _dim, false, true, 1f, stream);          //      + dV·Wvᵀ
            DeviceTrainKernels.RmsNormDgamma(sh, _dN, _x[b], a.Inv1, p.G1.G, S, _dim);
            DeviceTrainKernels.RmsNormBackwardDx(sh, _dN, _x[b], p.G1.W, a.Inv1, _dX, S, _dim);
            DeviceTrainKernels.AddInPlace(sh, _dX, _dXAlt, S * _dim);                                // dx_b = norm path + residual
        }
        DeviceTrainKernels.EmbedScatterAdd(sh, _wemb.G, _tokDev, _dX, S, _dim);

        // The single synchronisation of the step: read the scalar loss.
        NativeCheck.Ok(Cuda.cudaMemcpyAsync(_lossPin, _lossRows, (nuint)(S * sizeof(float)), (int)MemcpyKind.DeviceToHost, sh), "loss d2h");
        CudaStreamManager.Shared.Synchronize(stream);

        Marshal.Copy(_lossPin, _lossHost, 0, S);
        float total = 0f;
        for (int i = 0; i < S; i++) total += _lossHost[i];
        TokensSeen += S;
        return total / S;
    }

    private void ApplyAdam(float lr, float gradScale)
    {
        _t++;
        float bc1 = 1f - MathF.Pow(_beta1, _t);
        float bc2 = 1f - MathF.Pow(_beta2, _t);
        foreach (var p in AllParams())
        {
            DeviceTrainKernels.AdamStep(_stream.Handle, p.W, p.G, p.M, p.V, p.Count,
                                        lr, _beta1, _beta2, _epsAdam, _weightDecay, bc1, bc2, gradScale);
            Memset(p.G, p.Count);          // grads consumed → ready for the next accumulation
        }
    }

    private void ZeroGrads()
    {
        foreach (var p in AllParams()) Memset(p.G, p.Count);
        CudaStreamManager.Shared.Synchronize(_stream);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        CudaStreamManager.Shared.Synchronize(_stream);
        foreach (var (ptr, bytes) in _owned) _alloc.Free(_deviceId, ptr, bytes, _stream);
        _alloc.FreePinnedHost(_tokPin, (nuint)(_maxSeq * sizeof(int)));
        _alloc.FreePinnedHost(_tgtPin, (nuint)(_maxSeq * sizeof(int)));
        _alloc.FreePinnedHost(_lossPin, (nuint)(_maxSeq * sizeof(float)));
    }
}
