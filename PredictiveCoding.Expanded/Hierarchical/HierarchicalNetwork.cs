using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.Hierarchical;

// ─────────────────────────────────────────────────────────────────────────────
// HierarchicalNetwork  —  deep modular predictive-coding hierarchy
//
// A stack of N autonomous generative PCNs (modules).  Module 0 maps the raw
// datum to a latent vector; every higher module maps the latent of the module
// below to its own, more abstract latent.  Crucially, the higher modules do NOT
// predict the data directly — they predict the *latent states* of the modules
// underneath them.  This yields the multi-level abstraction of a hierarchical
// VAE while keeping every weight update local (no global backprop).
//
//   Encode :  x → enc_0 → z_0 → enc_1 → z_1 → … → z_{N-1}
//   Decode :  z_{N-1} → dec_{N-1} → ẑ_{N-2} → … → dec_0 → x̂
//
// Learning is greedy and layer-wise (Hinton et al. 2006 DBN-style): each module
// learns to autoencode the (frozen) latent produced by the module below it.
// This avoids backprop-through-time across the whole stack and lets each level
// converge independently.
//
// References:
//   Rao & Ballard (1999); Friston (2005) — hierarchical PC theory.
//   Hinton, Osindero & Teh (2006) — greedy layer-wise unsupervised learning.
//   Sønderby et al. (2016) Ladder VAE — hierarchical latent variables.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class HierarchicalNetwork : IDisposable
{
    private readonly Network[]                 _modules;
    private readonly HierarchicalNetworkConfig _config;

    /// <summary>Number of stacked modules.</summary>
    public int ModuleCount => _modules.Length;

    /// <summary>Dimensionality of the raw input (module 0, layer[0]).</summary>
    public int InputDim => _modules[0].Configuration.LayerSizes[0];

    /// <summary>Dimensionality of the most abstract latent (top module, layer[^1]).</summary>
    public int TopLatentDim => _modules[^1].Configuration.LayerSizes[^1];

    public HierarchicalNetworkConfig Configuration => _config;

    public HierarchicalNetwork(HierarchicalNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _modules = new Network[config.Modules.Length];
        for (int k = 0; k < config.Modules.Length; k++)
        {
            var netCfg = config.Modules[k] with
            {
                Path = Path.Combine(config.BasePath, $"module_{k}"),
            };
            _modules[k] = new Network(netCfg, new Random(rng.Next()));
        }
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes <paramref name="input"/> all the way up the stack and returns the
    /// top-module latent z_{N-1}.
    /// </summary>
    public DenseTensor<float> Encode(DenseTensor<float> input, int inferenceSteps)
    {
        var cur = input;
        for (int k = 0; k < _modules.Length; k++)
            cur = _modules[k].InferGetTop(cur, inferenceSteps);
        return cur;
    }

    /// <summary>
    /// Encodes <paramref name="input"/> and returns the latent emitted by every
    /// module.  Index k holds z_k (the top-layer state of module k); the last
    /// element is the most abstract representation.
    /// </summary>
    public DenseTensor<float>[] EncodeAll(DenseTensor<float> input, int inferenceSteps)
    {
        var latents = new DenseTensor<float>[_modules.Length];
        var cur     = input;
        for (int k = 0; k < _modules.Length; k++)
        {
            cur        = _modules[k].InferGetTop(cur, inferenceSteps);
            latents[k] = cur;
        }
        return latents;
    }

    // ── Decoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes a top-module latent back down to the input space by running each
    /// module's generative sweep in turn.
    /// </summary>
    public DenseTensor<float> Decode(DenseTensor<float> topLatent, int inferenceSteps)
    {
        var cur = topLatent;
        for (int k = _modules.Length - 1; k >= 0; k--)
            cur = _modules[k].Generate(cur, inferenceSteps);
        return cur;
    }

    // ── Reconstruction ───────────────────────────────────────────────────────

    /// <summary>Full round-trip: x → z_{N-1} → x̂.</summary>
    public DenseTensor<float> Reconstruct(DenseTensor<float> input, int inferenceSteps)
        => Decode(Encode(input, inferenceSteps), inferenceSteps);

    /// <summary>Half mean-squared reconstruction error ½‖x − Reconstruct(x)‖².</summary>
    public float ReconstructionError(DenseTensor<float> input, int inferenceSteps)
    {
        var flat  = input.Rank == 1 ? input : TensorUtils.Reshape(input, InputDim);
        var x_hat = Reconstruct(flat, inferenceSteps);
        return 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(flat, x_hat));
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Greedy layer-wise unsupervised learning pass.  Module 0 learns to encode
    /// the raw datum; each higher module then learns to encode the (current)
    /// latent emitted by the module below it.  Every module is updated with the
    /// local Hebbian PC rule — there is no global backprop across the stack.
    /// </summary>
    public void Learn(DenseTensor<float> input, int inferenceSteps)
    {
        var cur = input;
        for (int k = 0; k < _modules.Length; k++)
        {
            _modules[k].LearnUnsupervised(cur, inferenceSteps);
            // Re-encode with the just-updated weights so the module above sees the
            // freshest representation of the level below.
            cur = _modules[k].InferGetTop(cur, inferenceSteps);
        }
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>PC free energy of module <paramref name="moduleIndex"/> after its last inference.</summary>
    public float ModuleEnergy(int moduleIndex) => _modules[moduleIndex].ComputeEnergy();

    public void FlushAll()
    {
        foreach (var m in _modules) m.FlushStore();
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var m in _modules) m.Dispose();
    }
}
