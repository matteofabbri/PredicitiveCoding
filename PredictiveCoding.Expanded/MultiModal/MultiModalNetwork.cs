using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.MultiModal;

// ─────────────────────────────────────────────────────────────────────────────
// MultiModalNetwork  —  multi-sensory PCN with cross-modal completion
//
// Architecture:
//   • N modality-specific PCNs, each mapping a raw input to a latent vector.
//   • One associative PCN that maps the concatenated latent to a label (or
//     task-specific output).
//
// Training (per sample):
//   1. Each present modality network runs LearnUnsupervised — learns its own
//      internal representation without supervision.
//   2. All modality latents are concatenated (missing → zero-padded).
//   3. The associative network runs supervised Learn on the joint latent.
//   4. If TrainAssociativeGeneratively=true, the associative network also
//      runs LearnUnsupervised to learn to reconstruct the full joint latent.
//
// Prediction with partial modalities:
//   Missing modalities contribute zero vectors; the associative network
//   classifies from the remaining evidence.
//
// Cross-modal generation (requires TrainAssociativeGeneratively=true):
//   source → z_src → [z_src | 0 | …] → assoc.InferGetTop → z_assoc
//   z_assoc → assoc.Generate → z_full_hat → slice z_target → target.Generate
//
// References:
//   Rao & Ballard (1999); Friston (2005) — PC hierarchy theory.
//   Denk et al. (2023) Multisensory integration as predictive coding.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class MultiModalNetwork : IDisposable
{
    private readonly Network[]               _modalities;
    private readonly Network                 _associative;
    private readonly MultiModalNetworkConfig _config;
    private readonly int[]                   _latentDims;    // top-layer size per modality
    private readonly int[]                   _latentOffsets; // start offset in concat vector

    public int ModalityCount               => _modalities.Length;
    public MultiModalNetworkConfig Configuration => _config;

    public MultiModalNetwork(MultiModalNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _modalities = new Network[config.Modalities.Length];
        for (int i = 0; i < config.Modalities.Length; i++)
        {
            var m = config.Modalities[i];
            var netCfg = m.NetworkConfig with
            {
                Path = Path.Combine(config.BasePath, $"modality_{m.Name}"),
            };
            _modalities[i] = new Network(netCfg, new Random(rng.Next()));
        }

        var assocCfg = config.AssociativeConfig with
        {
            Path = Path.Combine(config.BasePath, "associative"),
        };
        _associative = new Network(assocCfg, new Random(rng.Next()));

        _latentDims    = config.Modalities.Select(m => m.NetworkConfig.LayerSizes[^1]).ToArray();
        _latentOffsets = new int[_latentDims.Length];
        for (int i = 1; i < _latentDims.Length; i++)
            _latentOffsets[i] = _latentOffsets[i - 1] + _latentDims[i - 1];
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes all provided modalities and concatenates their latent vectors.
    /// Modalities absent from <paramref name="inputs"/> are zero-padded.
    /// Keys are modality indices (0-based).
    /// </summary>
    public DenseTensor<float> EncodeAll(
        IReadOnlyDictionary<int, DenseTensor<float>> inputs,
        int inferenceSteps)
    {
        var parts = new DenseTensor<float>[_modalities.Length];
        for (int i = 0; i < _modalities.Length; i++)
        {
            parts[i] = inputs.TryGetValue(i, out var x)
                ? _modalities[i].InferGetTop(x, inferenceSteps)
                : DenseTensor<float>.Zeros(_latentDims[i]);
        }
        return TensorUtils.Concat(parts);
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Full supervised learning pass:
    ///   • Each present modality network updates its generative weights
    ///     (unsupervised PC learning on its own input).
    ///   • The associative network learns the joint latent → label mapping.
    ///   • If TrainAssociativeGeneratively is true, the associative network
    ///     also learns to reconstruct the full concatenated latent.
    /// </summary>
    public void Learn(
        IReadOnlyDictionary<int, DenseTensor<float>> modalInputs,
        DenseTensor<float>                            label,
        int                                           inferenceSteps)
    {
        foreach (var (idx, x) in modalInputs)
            _modalities[idx].LearnUnsupervised(x, inferenceSteps);

        var zFull = EncodeAll(modalInputs, inferenceSteps);
        _associative.Learn(zFull, label, inferenceSteps);

        if (_config.TrainAssociativeGeneratively)
            _associative.LearnUnsupervised(zFull, inferenceSteps);
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Classifies via the associative network given any non-empty subset of
    /// modalities.  Missing modalities contribute zero latent vectors.
    /// </summary>
    public int Predict(
        IReadOnlyDictionary<int, DenseTensor<float>> modalInputs,
        int inferenceSteps)
    {
        var zFull = EncodeAll(modalInputs, inferenceSteps);
        return _associative.Predict(zFull, inferenceSteps);
    }

    /// <summary>Predict with free-energy diagnostics.</summary>
    public (int cls, float energy) PredictWithEnergy(
        IReadOnlyDictionary<int, DenseTensor<float>> modalInputs,
        int inferenceSteps)
    {
        var zFull = EncodeAll(modalInputs, inferenceSteps);
        return _associative.PredictWithEnergy(zFull, inferenceSteps);
    }

    // ── Cross-modal generation ────────────────────────────────────────────────

    /// <summary>
    /// Generates a reconstruction of <paramref name="targetModalityIndex"/> from
    /// <paramref name="sourceInput"/> alone.
    ///
    /// Requires <see cref="MultiModalNetworkConfig.TrainAssociativeGeneratively"/> = true
    /// so the associative network has learned to invert its own input space.
    ///
    /// Pipeline:
    ///   source input
    ///     → source latent z_src
    ///     → concat [z_src | 0 | …]
    ///     → associative.InferGetTop  → z_assoc (top-layer state)
    ///     → associative.Generate     → z_full_hat (reconstructed concat latent)
    ///     → slice target portion     → z_target
    ///     → target modality.Generate → x_target_hat
    /// </summary>
    public DenseTensor<float> CrossModalGenerate(
        int                sourceModalityIndex,
        DenseTensor<float> sourceInput,
        int                targetModalityIndex,
        int                inferenceSteps)
    {
        if (!_config.TrainAssociativeGeneratively)
            throw new InvalidOperationException(
                "CrossModalGenerate requires TrainAssociativeGeneratively = true in the config.");

        var partialInputs = new Dictionary<int, DenseTensor<float>>
        {
            [sourceModalityIndex] = sourceInput,
        };

        var zPartial   = EncodeAll(partialInputs, inferenceSteps);
        var zAssocTop  = _associative.InferGetTop(zPartial, inferenceSteps);
        var zFullHat   = _associative.Generate(zAssocTop, inferenceSteps);

        int offset = _latentOffsets[targetModalityIndex];
        int dim    = _latentDims[targetModalityIndex];
        var zTarget = new DenseTensor<float>(dim);
        for (int d = 0; d < dim; d++)
            zTarget.Buffer[d] = zFullHat.Buffer[offset + d];

        return _modalities[targetModalityIndex].Generate(zTarget, inferenceSteps);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public string ModalityName(int index) => _config.Modalities[index].Name;

    public void FlushAll()
    {
        foreach (var m in _modalities) m.FlushStore();
        _associative.FlushStore();
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var m in _modalities) m.Dispose();
        _associative.Dispose();
    }
}
