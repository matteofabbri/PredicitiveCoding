using PredictiveCoding.LLM.Eval;

namespace PredictiveCoding.LLM.Autograd;

// ─────────────────────────────────────────────────────────────────────────────
// DecoderLM  —  a configurable multi-block, multi-head decoder transformer built
// on the autograd engine (the trainable counterpart of the inference-only
// GptModel). It generalises TinyTransformerLM from a single single-head block to
// N blocks × H heads, and adds sinusoidal positional encoding + a final norm.
//
// Multi-head without a concat op: each head h owns its own Q/K/V projections into
// head-space AND its own output projection Wo_h [headDim, dim]; the per-head
// outputs are summed. That sum is exactly  concat(head_1..head_H) · Wo  with a
// block-structured Wo, so it is real multi-head attention expressed only with the
// ops the engine already has (MatMul, Transpose, Scale, SoftmaxRowsCausal,
// LayerNorm, Gelu, Add, AddBias, Dropout).
//
// Forward (pre-norm + residual, GPT-style):
//   x  = Wemb[tokens] + positional      (row gather ≡ onehot(tokens)·Wemb)
//   repeat NumLoops times (weight-tied; + per-loop bias when NumLoops > 1):
//     for each block:
//       x += Dropout( Σ_h  softmaxCausal((h1·Wq_h)(h1·Wk_h)ᵀ·scale)·(h1·Wv_h) · Wo_h )
//       x += Dropout( gelu(LN(x)·W1 + b1)·W2 + b2 )
//   logits = LN_f(x)·Wout
//
// NumLoops > 1 is looped/weight-tied depth (roadmap Phase 2): the stack is
// iterated toward a fixed point instead of stacked deeper, and at inference the
// loop can exit early once a pass stops changing the state (LoopExitTau).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record DecoderLmConfig(
    int Vocab,
    int Dim,
    int NumBlocks,
    int NumHeads,
    int FfDim,
    int MaxSeq,
    float Dropout = 0f,
    float Eps = 1e-5f,
    int NumLoops = 1,
    float LoopExitTau = 0f,
    float MtpWeight = 0f,
    int MtpHorizon = 2,
    string? BlockPattern = null,
    int NumExperts = 0,
    int MoeTopK = 1,
    float MoeBalanceWeight = 0.01f,
    bool UsePlan = false,
    bool UseRmsNorm = false,
    bool UseRope = false)
{
    public int HeadDim => Dim / NumHeads;

    public void Validate()
    {
        if (Vocab <= 0 || Dim <= 0 || NumBlocks <= 0 || NumHeads <= 0 || FfDim <= 0 || MaxSeq <= 0)
            throw new ArgumentException("All DecoderLmConfig sizes must be positive.");
        if (Dim % NumHeads != 0)
            throw new ArgumentException($"Dim ({Dim}) must be divisible by NumHeads ({NumHeads}).");
        if (NumLoops < 1)
            throw new ArgumentException($"NumLoops ({NumLoops}) must be >= 1.");
        if (LoopExitTau < 0f)
            throw new ArgumentException($"LoopExitTau ({LoopExitTau}) must be >= 0 (0 disables the early exit).");
        if (MtpWeight < 0f)
            throw new ArgumentException($"MtpWeight ({MtpWeight}) must be >= 0 (0 disables multi-token prediction).");
        if (MtpWeight > 0f && MtpHorizon < 2)
            throw new ArgumentException($"MtpHorizon ({MtpHorizon}) must be >= 2 when MtpWeight is on (2 = predict t+2).");
        if (BlockPattern is { } p)
        {
            if (p.Length != NumBlocks)
                throw new ArgumentException($"BlockPattern length ({p.Length}) must equal NumBlocks ({NumBlocks}).");
            foreach (char c in p)
                if (c != 'A' && c != 'T')
                    throw new ArgumentException($"BlockPattern may only contain 'A' (attention) or 'T' (temporal), found '{c}'.");
        }
        if (NumExperts < 0)
            throw new ArgumentException($"NumExperts ({NumExperts}) must be >= 0 (0 = dense feed-forward).");
        if (NumExperts > 0 && (MoeTopK < 1 || MoeTopK > NumExperts))
            throw new ArgumentException($"MoeTopK ({MoeTopK}) must be in [1, NumExperts ({NumExperts})].");
        if (MoeBalanceWeight < 0f)
            throw new ArgumentException($"MoeBalanceWeight ({MoeBalanceWeight}) must be >= 0 (0 disables the balance loss).");
        if (UseRope && HeadDim % 2 != 0)
            throw new ArgumentException($"UseRope needs an even head dim, got {HeadDim}.");
        if ((UseRmsNorm || UseRope) && BlockPattern is not null)
            throw new ArgumentException("UseRmsNorm/UseRope are the Scale training path — attention blocks only (BlockPattern must be null).");
    }

    // ── Scale presets (ROADMAP-Scale B1) ─────────────────────────────────────
    //
    // Plain GPT-2-style dense decoders: every block is attention, dense FFN
    // (4×dim), head dim 64, no loops/MTP/MoE/plan — the 2026 toy stays the
    // showcase of every mechanism; these are the configs meant to be TRAINED
    // for real on one RTX 4070, where pretraining throughput and a clean
    // GPT-2 comparison matter more than mechanism coverage. Any 2026 feature
    // can still be switched on per-run via `with { … }` (it is a record).
    //
    // Parameters (dense, untied embeddings — see DenseParameterFormula):
    //   2·vocab·dim + blocks·(4·dim² + 2·dim·ffn + ffn + norm·dim) + tail·dim
    //   (norm = 3 with RMSNorm, 5 with LayerNorm; tail = 1 vs 2)
    //   Small: 26.8M @ vocab 16k   (39.4M @ 32k)
    //   Base:  135.3M @ vocab 32k  (GPT-2-small class: 768×12, ctx 1024)
    //
    // Presets run RMSNorm + RoPE (ROADMAP-Scale A1): the Llama-style block the
    // device training path implements — CPU and GPU share one architecture.

    /// <summary>~30M-class preset (dim 384, 8 blocks, 6 heads, ffn 1536, RMSNorm+RoPE): 26.8M parameters at a 16k vocab.</summary>
    public static DecoderLmConfig Small(int vocab, int maxSeq = 512) =>
        new(vocab, Dim: 384, NumBlocks: 8, NumHeads: 6, FfDim: 1536, MaxSeq: maxSeq,
            UseRmsNorm: true, UseRope: true);

    /// <summary>GPT-2-small-class preset (dim 768, 12 blocks, 12 heads, ffn 3072, RMSNorm+RoPE): 135.3M parameters at a 32k vocab.</summary>
    public static DecoderLmConfig Base(int vocab, int maxSeq = 1024) =>
        new(vocab, Dim: 768, NumBlocks: 12, NumHeads: 12, FfDim: 3072, MaxSeq: maxSeq,
            UseRmsNorm: true, UseRope: true);

    /// <summary>
    /// Exact parameter count for a DENSE config (no loops/MTP/MoE/plan) without
    /// instantiating the tensors — must match <see cref="DecoderLM.ParameterCount"/>;
    /// the B3 memory budget builds on it.
    /// </summary>
    public long DenseParameterFormula()
    {
        if (NumExperts > 0 || NumLoops > 1 || MtpWeight > 0f || UsePlan || BlockPattern is not null)
            throw new InvalidOperationException("DenseParameterFormula only covers the plain dense-attention config.");
        long normPerBlock = UseRmsNorm ? 2L * Dim : 4L * Dim;               // 2 gammas vs 2 gamma+beta pairs
        long perBlock = 4L * Dim * Dim + 2L * Dim * FfDim + FfDim + Dim + normPerBlock;
        long finalNorm = UseRmsNorm ? Dim : 2L * Dim;
        return 2L * Vocab * Dim + NumBlocks * perBlock + finalNorm;
    }
}

public sealed class DecoderLM
{
    public DecoderLmConfig Config { get; }

    private readonly float          _attnScale;
    private readonly Random         _rng;
    private readonly Tensor         _wemb;     // [vocab, dim]
    private readonly IDecoderBlock[] _blocks;
    private readonly Tensor[]? _loopBias;      // [NumLoops] × [1, dim] — loop-step conditioning; null when NumLoops == 1
    private readonly Tensor    _lnFG;          // final norm gamma
    private readonly Tensor?   _lnFB;          // final norm beta — null under RMSNorm
    private readonly Tensor    _wout;          // [dim, vocab]
    private readonly Tensor?   _woutMtp;       // [dim, vocab] — aux head predicting t+MtpHorizon; null when MtpWeight == 0
    private readonly Tensor?   _plan;          // [1, dim] — latent plan conditioning (roadmap Phase 8); null when UsePlan == false

    public DecoderLM(DecoderLmConfig cfg, Random rng)
    {
        cfg.Validate();
        Config     = cfg;
        _rng       = rng;
        _attnScale = 1f / MathF.Sqrt(cfg.HeadDim);

        _wemb   = AgInit.Randn(cfg.Vocab, cfg.Dim, rng);
        _blocks = new IDecoderBlock[cfg.NumBlocks];
        for (int b = 0; b < cfg.NumBlocks; b++)
            _blocks[b] = cfg.BlockPattern is { } p && p[b] == 'T'
                ? new TemporalBlock(cfg, rng)
                : new Block(cfg, rng);
        if (cfg.NumLoops > 1)
        {
            // Zero-initialised, so a fresh looped model starts as a pure
            // weight-tied iteration; only allocated when looping, so the
            // NumLoops == 1 parameter list (and old checkpoints) are untouched.
            _loopBias = new Tensor[cfg.NumLoops];
            for (int k = 0; k < cfg.NumLoops; k++) _loopBias[k] = new Tensor(1, cfg.Dim);
        }
        _lnFG = AgInit.Ones(1, cfg.Dim);
        _lnFB = cfg.UseRmsNorm ? null : new Tensor(1, cfg.Dim);
        _wout = AgInit.Randn(cfg.Dim, cfg.Vocab, rng);
        // Multi-token prediction head (roadmap Phase 3), only allocated when on
        // so the vanilla parameter list (and old checkpoints) are untouched.
        if (cfg.MtpWeight > 0f)
            _woutMtp = AgInit.Randn(cfg.Dim, cfg.Vocab, rng);
        // Latent plan vector (roadmap Phase 8): zero-initialised, so a fresh
        // plan model starts exactly as the unconditioned one.
        if (cfg.UsePlan)
            _plan = new Tensor(1, cfg.Dim);
    }

    /// <summary>All trainable tensors in a fixed order (must match for Checkpoint save/load).</summary>
    public Tensor[] Parameters()
    {
        var ps = new List<Tensor> { _wemb };
        foreach (var b in _blocks) ps.AddRange(b.Parameters());
        if (_loopBias is not null) ps.AddRange(_loopBias);
        ps.Add(_lnFG);
        if (_lnFB is not null) ps.Add(_lnFB);
        ps.Add(_wout);
        if (_woutMtp is not null) ps.Add(_woutMtp);
        if (_plan is not null) ps.Add(_plan);
        return [.. ps];
    }

    // ── Predictive-coding (iPC) training hooks — used by Training/PcDecoderTrainer ──
    internal float AttnScale => _attnScale;
    internal Random Rng => _rng;
    internal Tensor Wemb => _wemb;
    internal Tensor Wout => _wout;
    internal Tensor LnFG => _lnFG;
    internal IReadOnlyList<IDecoderBlock> Blocks => _blocks;

    /// <summary>
    /// Forward pass: tokens → logits [seq, vocab]. Dropout active only when
    /// training. An optional <see cref="ActivationProbe"/> records per-block
    /// activation statistics and per-head attention-sink shares; null (the
    /// default) leaves the pass untouched.
    /// </summary>
    public Tensor Forward(int[] tokens, bool training = false, ActivationProbe? probe = null)
        => Ag.MatMul(Hidden(tokens, training, probe, auxLosses: null), _wout);   // [seq, vocab]

    /// <summary>
    /// Forward pass returning both heads (roadmap Phase 3): the main next-token
    /// logits and, when <c>MtpWeight &gt; 0</c>, the auxiliary logits predicting
    /// the token at t+MtpHorizon from the same final hidden state (null when
    /// multi-token prediction is off). Predicting further into the future is
    /// the same idea as the core's temporal path, one level up.
    /// </summary>
    public (Tensor main, Tensor? mtp) ForwardMtp(int[] tokens, bool training = false, ActivationProbe? probe = null)
        => Heads(tokens, training, probe, auxLosses: null);

    /// <summary>
    /// The shared trunk's final hidden state [seq, Dim] — the representation the
    /// output heads read from. Used as the retrieval key for episodic memory
    /// (roadmap-Pro Phase 7); no gradient is needed by the caller.
    /// </summary>
    public Tensor HiddenStates(int[] tokens) => Hidden(tokens, training: false, probe: null, auxLosses: null);

    private (Tensor main, Tensor? mtp) Heads(int[] tokens, bool training, ActivationProbe? probe, List<Tensor>? auxLosses)
    {
        var xf = Hidden(tokens, training, probe, auxLosses);
        return (Ag.MatMul(xf, _wout), _woutMtp is null ? null : Ag.MatMul(xf, _woutMtp));
    }

    /// <summary>
    /// Compute odometer: token positions fed to this model's forwards, lifetime
    /// (every Loss/Evaluate/Generate/Embed pass adds its sequence length — a
    /// full-reforward generation pays O(n²) here). Not synchronized: parallel
    /// training replicas are separate instances, each counts its own.
    /// </summary>
    public long ForwardTokens { get; private set; }

    // Embedding → looped block stack → final layer-norm: the shared trunk under
    // every head.
    private Tensor Hidden(int[] tokens, bool training, ActivationProbe? probe, List<Tensor>? auxLosses)
    {
        int seq = tokens.Length;
        if (seq > Config.MaxSeq)
            throw new ArgumentException($"Sequence length {seq} exceeds MaxSeq {Config.MaxSeq}.");
        ForwardTokens += seq;

        // Embedding as a row gather — the same maths as onehot(tokens)·Wemb
        // (forward copies row t; backward scatter-adds dx into row t) without
        // materialising a [seq, vocab] one-hot: at a real vocab (16k-32k,
        // ROADMAP-Scale B1) the one-hot matmul would dominate the whole step.
        var x = Ag.GatherRows(_wemb, tokens);                              // [seq, dim]
        if (!Config.UseRope)                                               // with RoPE, positions live in q/k
            x = Ag.Add(x, AgPositional.Sinusoidal(seq, Config.Dim));       // + positions
        if (_plan is not null)
            x = Ag.AddBias(x, _plan);                                      // latent plan conditioning (Phase 8)

        // Weight-tied depth (roadmap Phase 2): the same block stack is applied
        // NumLoops times — depth becomes iteration toward a fixed point, the
        // discrete cousin of the core network's iterative settling. Gradients
        // through the reused parameters accumulate once per use (the engine
        // += s into Grad), so training the tied loop needs nothing special.
        int      loops     = Config.NumLoops;
        int      loopsUsed = 0;
        float[]? previous  = null;                                          // x after the previous loop pass

        for (int k = 0; k < loops; k++)
        {
            if (_loopBias is not null)
                x = Ag.AddBias(x, _loopBias[k]);          // loop-step conditioning: which pass is this?

            for (int b = 0; b < _blocks.Length; b++)
            {
                x = _blocks[b].Forward(x, _attnScale, Config.Dropout, training, _rng, Config.Eps, probe, b, auxLosses);
                probe?.RecordBlock(b, x.Data);
            }
            loopsUsed = k + 1;

            // Early exit (inference only): stop looping once a pass changes the
            // hidden state by less than LoopExitTau, relative — Phase 1's
            // adaptive effort, applied to the LM. Tau = 0 disables it.
            if (!training && Config.LoopExitTau > 0f && k < loops - 1)
            {
                if (previous is not null && RelativeChange(previous, x.Data) < Config.LoopExitTau)
                    break;
                previous = (float[])x.Data.Clone();
            }
        }
        if (probe is not null) probe.LoopsUsed = loopsUsed;

        return Config.UseRmsNorm
            ? Ag.RmsNorm(x, _lnFG, Config.Eps)
            : Ag.LayerNorm(x, _lnFG, _lnFB!, Config.Eps);
    }

    // ‖now − before‖ / (‖before‖ + tiny), over the flattened hidden state.
    private static float RelativeChange(float[] before, float[] now)
    {
        double diff = 0d, norm = 0d;
        for (int i = 0; i < now.Length; i++)
        {
            double d = now[i] - before[i];
            diff += d * d;
            norm += (double)before[i] * before[i];
        }
        return (float)(Math.Sqrt(diff) / (Math.Sqrt(norm) + 1e-12));
    }

    /// <summary>
    /// Masked next-token cross-entropy; rows with target == ignoreIndex are
    /// skipped. With <c>MtpWeight &gt; 0</c> the auxiliary multi-token head is
    /// trained too: total = CE(main, t+1) + MtpWeight · CE(aux, t+MtpHorizon),
    /// where the aux targets are the same target stream shifted MtpHorizon − 1
    /// further into the future (missing tail rows are masked out).
    /// </summary>
    public Tensor Loss(int[] tokens, int[] targets, int ignoreIndex, bool training = false)
    {
        // Collect the per-block router balance terms only when they can matter.
        var aux = Config.NumExperts > 0 && Config.MoeBalanceWeight > 0f ? new List<Tensor>() : null;

        var (main, mtp) = Heads(tokens, training, probe: null, aux);
        var loss = Ag.CrossEntropy(main, targets, ignoreIndex);

        if (mtp is not null)
        {
            var auxTargets = MtpTargets(targets, ignoreIndex);
            bool anyValid  = false;
            foreach (int t in auxTargets) if (t != ignoreIndex) { anyValid = true; break; }
            if (anyValid)                                  // else: sequence too short for the horizon
                loss = Ag.Add(loss, Ag.Scale(Ag.CrossEntropy(mtp, auxTargets, ignoreIndex), Config.MtpWeight));
        }

        if (aux is { Count: > 0 })
        {
            var balance = aux[0];
            for (int i = 1; i < aux.Count; i++) balance = Ag.Add(balance, aux[i]);
            loss = Ag.Add(loss, Ag.Scale(balance, Config.MoeBalanceWeight));
        }
        return loss;
    }

    /// <summary>Total trainable scalars.</summary>
    public long ParameterCount
    {
        get
        {
            long n = 0;
            foreach (var p in Parameters()) n += p.Data.Length;
            return n;
        }
    }

    /// <summary>
    /// Parameters active per token (the Step-3.5-Flash headline metric): with
    /// sparse experts, each token runs only MoeTopK of the NumExperts expert
    /// FFNs, so the inactive experts' weights don't count. Equals
    /// <see cref="ParameterCount"/> for dense models.
    /// </summary>
    public long ActiveParameterCountPerToken
    {
        get
        {
            if (Config.NumExperts == 0) return ParameterCount;
            long perExpert = (long)Config.Dim * Config.FfDim + Config.FfDim
                           + (long)Config.FfDim * Config.Dim + Config.Dim;
            int attentionBlocks = Config.BlockPattern is { } p ? p.Count(c => c == 'A') : Config.NumBlocks;
            return ParameterCount - attentionBlocks * (Config.NumExperts - Config.MoeTopK) * perExpert;
        }
    }

    // ── Latent planning (roadmap Phase 8) ───────────────────────────────────

    /// <summary>
    /// Test-time settling of the latent plan on the evidence: freezes every
    /// weight and runs <paramref name="steps"/> plain gradient steps on the
    /// plan vector ALONE, minimising the next-token cross-entropy of the
    /// prompt — infer the latent from what was observed, then decode under it.
    /// This is the predictive-coding reading of "plan in latent space, then
    /// generate". Returns the per-step loss trace (the settling curve).
    /// </summary>
    public float[] SettlePlan(int[] promptTokens, int steps, float learningRate)
    {
        if (_plan is null)
            throw new InvalidOperationException("SettlePlan needs a model built with UsePlan = true.");
        if (promptTokens.Length < 2)
            throw new ArgumentException("Settling needs a prompt of at least 2 tokens (one prediction).");
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "steps must be >= 1.");

        var tokens  = promptTokens[..^1];
        var targets = promptTokens[1..];

        var trace = new float[steps];
        for (int k = 0; k < steps; k++)
        {
            var loss = Ag.CrossEntropy(Forward(tokens), targets);
            loss.Backward();                                  // fills every Grad; only the plan moves
            for (int i = 0; i < _plan.Data.Length; i++)
                _plan.Data[i] -= learningRate * _plan.Grad[i];
            trace[k] = loss.Data[0];
        }
        return trace;
    }

    /// <summary>Snapshot of the current plan vector (empty when UsePlan is off).</summary>
    public float[] GetPlan() => _plan is null ? [] : (float[])_plan.Data.Clone();

    /// <summary>Overwrites the plan vector (length must match Dim) — for save/restore and ablations.</summary>
    public void SetPlan(float[] plan)
    {
        if (_plan is null)
            throw new InvalidOperationException("SetPlan needs a model built with UsePlan = true.");
        if (plan.Length != _plan.Data.Length)
            throw new ArgumentException($"Plan length ({plan.Length}) must equal Dim ({_plan.Data.Length}).");
        Array.Copy(plan, _plan.Data, plan.Length);
    }

    /// <summary>
    /// The aux-head target stream: the main targets shifted MtpHorizon − 1
    /// further into the future; rows whose future falls past the window are
    /// masked with <paramref name="ignoreIndex"/>.
    /// </summary>
    public int[] MtpTargets(int[] targets, int ignoreIndex)
    {
        int shift = Config.MtpHorizon - 1;
        var aux   = new int[targets.Length];
        for (int i = 0; i < targets.Length; i++)
            aux[i] = i + shift < targets.Length ? targets[i + shift] : ignoreIndex;
        return aux;
    }

    // ── decoder blocks ───────────────────────────────────────────────────────────

    internal interface IDecoderBlock
    {
        IEnumerable<Tensor> Parameters();
        Tensor Forward(Tensor x, float attnScale, float dropout, bool training, Random rng, float eps,
                       ActivationProbe? probe, int blockIndex, List<Tensor>? auxLosses);
    }

    private sealed class Block : IDecoderBlock
    {
        private readonly int      _heads;
        private readonly bool     _useRms, _useRope;           // ROADMAP-Scale A1: the Llama-style block
        private readonly Tensor   _ln1G, _ln2G;
        private readonly Tensor?  _ln1B, _ln2B;                // null under RMSNorm (gamma-only)
        private readonly Tensor[] _wq, _wk, _wv, _wo;          // per-head projections
        private readonly Tensor?  _w1, _b1, _w2, _b2;          // dense feed-forward (null when MoE)
        private readonly Tensor?   _router;                    // [dim, E] — MoE gate (roadmap Phase 6)
        private readonly Tensor[]? _ew1, _eb1, _ew2, _eb2;     // per-expert feed-forwards
        private readonly int       _topK;

        public Block(DecoderLmConfig cfg, Random rng)
        {
            _heads   = cfg.NumHeads;
            _useRms  = cfg.UseRmsNorm;
            _useRope = cfg.UseRope;
            _ln1G = AgInit.Ones(1, cfg.Dim); _ln1B = _useRms ? null : new Tensor(1, cfg.Dim);
            _ln2G = AgInit.Ones(1, cfg.Dim); _ln2B = _useRms ? null : new Tensor(1, cfg.Dim);

            _wq = new Tensor[_heads]; _wk = new Tensor[_heads];
            _wv = new Tensor[_heads]; _wo = new Tensor[_heads];
            for (int h = 0; h < _heads; h++)
            {
                _wq[h] = AgInit.Randn(cfg.Dim, cfg.HeadDim, rng);
                _wk[h] = AgInit.Randn(cfg.Dim, cfg.HeadDim, rng);
                _wv[h] = AgInit.Randn(cfg.Dim, cfg.HeadDim, rng);
                _wo[h] = AgInit.Randn(cfg.HeadDim, cfg.Dim, rng);
            }

            if (cfg.NumExperts > 0)
            {
                // Sparse experts replace the dense FFN entirely (dense tensors
                // stay null, so NumExperts = 0 keeps the old parameter list).
                int e   = cfg.NumExperts;
                _topK   = cfg.MoeTopK;
                _router = AgInit.Randn(cfg.Dim, e, rng);
                _ew1 = new Tensor[e]; _eb1 = new Tensor[e];
                _ew2 = new Tensor[e]; _eb2 = new Tensor[e];
                for (int k = 0; k < e; k++)
                {
                    _ew1[k] = AgInit.Randn(cfg.Dim, cfg.FfDim, rng); _eb1[k] = new Tensor(1, cfg.FfDim);
                    _ew2[k] = AgInit.Randn(cfg.FfDim, cfg.Dim, rng); _eb2[k] = new Tensor(1, cfg.Dim);
                }
            }
            else
            {
                _w1 = AgInit.Randn(cfg.Dim, cfg.FfDim, rng); _b1 = new Tensor(1, cfg.FfDim);
                _w2 = AgInit.Randn(cfg.FfDim, cfg.Dim, rng); _b2 = new Tensor(1, cfg.Dim);
            }
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return _ln1G; if (_ln1B is not null) yield return _ln1B;
            for (int h = 0; h < _heads; h++)
            {
                yield return _wq[h]; yield return _wk[h]; yield return _wv[h]; yield return _wo[h];
            }
            yield return _ln2G; if (_ln2B is not null) yield return _ln2B;
            if (_router is not null)
            {
                yield return _router;
                for (int k = 0; k < _ew1!.Length; k++)
                {
                    yield return _ew1[k]; yield return _eb1![k]; yield return _ew2![k]; yield return _eb2![k];
                }
            }
            else
            {
                yield return _w1!; yield return _b1!; yield return _w2!; yield return _b2!;
            }
        }

        public Tensor Forward(Tensor x, float attnScale, float dropout, bool training, Random rng, float eps,
                              ActivationProbe? probe, int blockIndex, List<Tensor>? auxLosses)
        {
            // Multi-head causal self-attention (pre-norm + residual).
            var h1 = _useRms ? Ag.RmsNorm(x, _ln1G, eps) : Ag.LayerNorm(x, _ln1G, _ln1B!, eps);
            Tensor? attn = null;
            for (int h = 0; h < _heads; h++)
            {
                var q       = Ag.MatMul(h1, _wq[h]);                                  // [seq, headDim]
                var k       = Ag.MatMul(h1, _wk[h]);
                var v       = Ag.MatMul(h1, _wv[h]);
                if (_useRope) { q = Ag.Rope(q); k = Ag.Rope(k); }                     // positions live in q/k
                var scores  = Ag.Scale(Ag.MatMul(q, Ag.Transpose(k)), attnScale);     // [seq, seq]
                var probs   = Ag.SoftmaxRowsCausal(scores);
                probe?.RecordCausalAttention(blockIndex, h, probs.Data, probs.Rows);
                var headOut = Ag.MatMul(probs, v);                                    // [seq, headDim]
                var proj    = Ag.MatMul(headOut, _wo[h]);                             // [seq, dim]
                attn = attn is null ? proj : Ag.Add(attn, proj);
            }
            x = Ag.Add(x, Ag.Dropout(attn!, dropout, training, rng));

            // Feed-forward (pre-norm + residual): dense, or sparse experts.
            var h2 = _useRms ? Ag.RmsNorm(x, _ln2G, eps) : Ag.LayerNorm(x, _ln2G, _ln2B!, eps);
            var ff = _router is null ? DenseFfn(h2) : MoeFfn(h2, auxLosses, probe);
            return Ag.Add(x, Ag.Dropout(ff, dropout, training, rng));
        }

        private Tensor DenseFfn(Tensor h2)
            => Ag.AddBias(Ag.MatMul(Ag.Gelu(Ag.AddBias(Ag.MatMul(h2, _w1!), _b1!)), _w2!), _b2!);

        // Sparse mixture of experts (roadmap Phase 6, FLOP-sparse execution).
        // Router → softmax gates → hard top-k mask (a CONSTANT built from the
        // gate values, so gradients flow only through the surviving gates and
        // the selected experts). Each expert then runs ONLY on the rows routed
        // to it (GatherRows → FFN_e → gate-scale → ScatterRows); an expert with
        // no rows costs nothing at all. Skipped contributions were exact zeros
        // in the masked-dense formulation, so the outputs are bit-identical.
        private Tensor MoeFfn(Tensor h2, List<Tensor>? auxLosses, ActivationProbe? probe)
        {
            int e    = _ew1!.Length;
            int rows = h2.Rows;
            var gates = Ag.SoftmaxRows(Ag.MatMul(h2, _router!));                      // [seq, E]
            var mask  = TopKMask(gates.Data, rows, e, _topK);                         // constant 0/1
            var kept  = Ag.Mul(gates, mask);

            Tensor? outSum = null;
            for (int k = 0; k < e; k++)
            {
                var routed = RoutedRows(mask.Data, rows, e, k);
                if (routed.Length == 0) continue;                                     // the FLOP saving
                if (probe is not null) probe.MoeRowsComputed += routed.Length;

                var sub     = Ag.GatherRows(h2, routed);
                var expert  = Ag.AddBias(Ag.MatMul(Ag.Gelu(Ag.AddBias(Ag.MatMul(sub, _ew1[k]), _eb1![k])), _ew2![k]), _eb2![k]);
                var gateCol = Ag.GatherRows(Ag.Column(kept, k), routed);              // [r, 1]
                var placed  = Ag.ScatterRows(Ag.ScaleRows(expert, gateCol), routed, rows);
                outSum = outSum is null ? placed : Ag.Add(outSum, placed);
            }

            // Switch-style load balancing: L = E · Σ_e f_e · P_e, with f_e the
            // (constant) fraction of row-selections expert e received and P_e
            // its mean gate probability. Uniform routing ⇒ L = 1 exactly.
            if (auxLosses is not null)
            {
                var fractions = SelectionFractions(mask.Data, h2.Rows, e, _topK);
                Tensor? bal = null;
                for (int k = 0; k < e; k++)
                {
                    if (fractions[k] == 0f) continue;
                    var term = Ag.Scale(Ag.Mean(Ag.Column(gates, k)), e * fractions[k]);
                    bal = bal is null ? term : Ag.Add(bal, term);
                }
                if (bal is not null) auxLosses.Add(bal);
            }

            return outSum!;
        }

        // 0/1 keep-mask with the k largest gates per row. A plain constant
        // tensor (no parents): the discrete routing decision is not a
        // differentiable quantity.
        private static Tensor TopKMask(float[] gates, int rows, int experts, int topK)
        {
            var mask = new Tensor(rows, experts);
            for (int i = 0; i < rows; i++)
            {
                for (int chosen = 0; chosen < topK; chosen++)
                {
                    int best = -1;
                    float bestVal = float.NegativeInfinity;
                    for (int j = 0; j < experts; j++)
                    {
                        if (mask.Data[i * experts + j] != 0f) continue;
                        float g = gates[i * experts + j];
                        if (g > bestVal) { bestVal = g; best = j; }
                    }
                    mask.Data[i * experts + best] = 1f;
                }
            }
            return mask;
        }

        private static float[] SelectionFractions(float[] mask, int rows, int experts, int topK)
        {
            var f = new float[experts];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < experts; j++)
                    if (mask[i * experts + j] != 0f) f[j] += 1f;
            for (int j = 0; j < experts; j++) f[j] /= rows * topK;
            return f;
        }

        private static int[] RoutedRows(float[] mask, int rows, int experts, int expert)
        {
            int count = 0;
            for (int i = 0; i < rows; i++) if (mask[i * experts + expert] != 0f) count++;
            var routed = new int[count];
            int w = 0;
            for (int i = 0; i < rows; i++) if (mask[i * experts + expert] != 0f) routed[w++] = i;
            return routed;
        }
    }

    // ── temporal-state block (roadmap Phase 5) ──────────────────────────────────
    //
    // Replaces attention with a gated diagonal linear recurrence:
    //   a_t = σ(h1·Wa + ba)        decay gate   (how much past survives)
    //   i_t = σ(h1·Wi + bi)        input gate   (how much present enters)
    //   u_t = h1·Wu + bu           input projection
    //   s_t = a_t ⊙ s_{t−1} + i_t ⊙ u_t        (Ag.RecurrentScan)
    //   out = s·Wo + bo
    // The persistent state s is the LM-side sibling of the core network's
    // temporal hidden state: O(1) memory instead of a growing transcript.
    // Causality is inherent — the scan only ever looks backwards.

    private sealed class TemporalBlock : IDecoderBlock
    {
        private readonly Tensor _ln1G, _ln1B, _ln2G, _ln2B;
        private readonly Tensor _wa, _ba, _wi, _bi, _wu, _bu, _wo, _bo;   // recurrence
        private readonly Tensor _w1, _b1, _w2, _b2;                       // feed-forward

        public TemporalBlock(DecoderLmConfig cfg, Random rng)
        {
            _ln1G = AgInit.Ones(1, cfg.Dim); _ln1B = new Tensor(1, cfg.Dim);
            _ln2G = AgInit.Ones(1, cfg.Dim); _ln2B = new Tensor(1, cfg.Dim);

            _wa = AgInit.Randn(cfg.Dim, cfg.Dim, rng); _ba = new Tensor(1, cfg.Dim);
            _wi = AgInit.Randn(cfg.Dim, cfg.Dim, rng); _bi = new Tensor(1, cfg.Dim);
            _wu = AgInit.Randn(cfg.Dim, cfg.Dim, rng); _bu = new Tensor(1, cfg.Dim);
            _wo = AgInit.Randn(cfg.Dim, cfg.Dim, rng); _bo = new Tensor(1, cfg.Dim);

            _w1 = AgInit.Randn(cfg.Dim, cfg.FfDim, rng); _b1 = new Tensor(1, cfg.FfDim);
            _w2 = AgInit.Randn(cfg.FfDim, cfg.Dim, rng); _b2 = new Tensor(1, cfg.Dim);
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return _ln1G; yield return _ln1B;
            yield return _wa; yield return _ba; yield return _wi; yield return _bi;
            yield return _wu; yield return _bu; yield return _wo; yield return _bo;
            yield return _ln2G; yield return _ln2B;
            yield return _w1; yield return _b1; yield return _w2; yield return _b2;
        }

        public Tensor Forward(Tensor x, float attnScale, float dropout, bool training, Random rng, float eps,
                              ActivationProbe? probe, int blockIndex, List<Tensor>? auxLosses)
        {
            // Gated recurrence (pre-norm + residual). attnScale is unused —
            // there is no attention here, and no sink to meter. The FFN stays
            // dense (MoE lives in the attention blocks for now).
            var h1 = Ag.LayerNorm(x, _ln1G, _ln1B, eps);
            var a  = Ag.Sigmoid(Ag.AddBias(Ag.MatMul(h1, _wa), _ba));
            var i  = Ag.Sigmoid(Ag.AddBias(Ag.MatMul(h1, _wi), _bi));
            var u  = Ag.AddBias(Ag.MatMul(h1, _wu), _bu);
            var s  = Ag.RecurrentScan(a, Ag.Mul(i, u));
            var t  = Ag.AddBias(Ag.MatMul(s, _wo), _bo);
            x = Ag.Add(x, Ag.Dropout(t, dropout, training, rng));

            // Feed-forward (pre-norm + residual), same shape as the attention block.
            var h2  = Ag.LayerNorm(x, _ln2G, _ln2B, eps);
            var ff1 = Ag.Gelu(Ag.AddBias(Ag.MatMul(h2, _w1), _b1));
            var ff2 = Ag.Dropout(Ag.AddBias(Ag.MatMul(ff1, _w2), _b2), dropout, training, rng);
            return Ag.Add(x, ff2);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AgInit  —  shared parameter initialisers for autograd models.
// ─────────────────────────────────────────────────────────────────────────────

internal static class AgInit
{
    public static Tensor Randn(int rows, int cols, Random rng)
    {
        var t = new Tensor(rows, cols);
        int n = t.Data.Length;

        // Piccolo: sequenziale (il parallelismo non ripaga l'overhead).
        if (n < 1 << 16)
        {
            for (int i = 0; i < n; i++) t.Data[i] = 0.02f * (float)NextGaussian(rng);
            return t;
        }

        // Grande: Box-Muller in PARALLELO. Nascere con centinaia di milioni di pesi
        // su un core solo costava minuti (Log/Cos per elemento). Ogni partizione ha
        // il SUO Random (Random non e' thread-safe) con seed derivato UNA volta dal
        // master -> resta riproducibile (stesso master seed -> stesso modello), ma
        // l'init sfrutta tutti i core: da minuti a secondi.
        int baseSeed = rng.Next();
        int parts = Math.Max(1, Environment.ProcessorCount);
        int chunk = (n + parts - 1) / parts;
        System.Threading.Tasks.Parallel.For(0, parts, p =>
        {
            var local = new Random(baseSeed + p);
            int start = p * chunk, end = Math.Min(start + chunk, n);
            for (int i = start; i < end; i++) t.Data[i] = 0.02f * (float)NextGaussian(local);
        });
        return t;
    }

    public static Tensor Ones(int rows, int cols)
    {
        var t = new Tensor(rows, cols);
        Array.Fill(t.Data, 1f);
        return t;
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
