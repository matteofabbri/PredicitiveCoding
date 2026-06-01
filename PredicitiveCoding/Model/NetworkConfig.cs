using PredictiveCoding.Core;

namespace PredictiveCoding.Model;

public sealed record NetworkConfig
{
    /// <summary>
    /// Layer sizes from input to output, e.g. [784, 256, 128, 10].
    /// </summary>
    public int[] LayerSizes { get; init; } = [20, 64, 32, 4];

    /// <summary>Non-linearity of the generative model f(W μ + b).</summary>
    public ActivationType Activation { get; init; } = ActivationType.Tanh;

    /// <summary>
    /// Learning rate for hidden-state updates during inference (η_state).
    /// Typically much larger than WeightLearningRate.
    /// </summary>
    public float StateLearningRate { get; init; } = 0.1f;

    /// <summary>
    /// Learning rate for generative-weight updates (η_weight).
    /// Controls synaptic plasticity speed.
    /// </summary>
    public float WeightLearningRate { get; init; } = 0.001f;

    /// <summary>
    /// L2 weight-decay coefficient λ (Huang &amp; Rao 2011).
    /// Applied as ΔW -= λ · W during each weight update. Default = 0.
    /// </summary>
    public float WeightDecay { get; init; } = 0f;

    /// <summary>
    /// When true, each layer learns a per-unit diagonal precision vector
    /// instead of a scalar (Huang &amp; Rao 2011, diagonal Σ_l).
    /// </summary>
    public bool UseDiagonalPrecision { get; init; } = false;

    /// <summary>
    /// Optional shape of the raw input tensor, e.g. [28, 28] for a grayscale image
    /// or [3, 32, 32] for an RGB image.  When set, <see cref="Network.Infer"/> and
    /// <see cref="Network.Learn"/> automatically flatten the input to a 1-D vector
    /// before processing.  <see cref="LayerSizes"/>[0] must equal the product of
    /// all dimensions, e.g. 784 for [28, 28].
    /// Leave null (default) for inputs that are already 1-D.
    /// </summary>
    public int[]? InputShape { get; init; } = null;

    /// <summary>
    /// Total number of input units: product of InputShape if set, else LayerSizes[0].
    /// </summary>
    public int InputSize => InputShape is null
        ? LayerSizes[0]
        : InputShape.Aggregate(1, (acc, d) => acc * d);

    public void Validate()
    {
        if (LayerSizes.Length < 2)
            throw new InvalidOperationException("LayerSizes must have at least 2 elements.");
        if (StateLearningRate < 0)
            throw new InvalidOperationException("StateLearningRate must be >= 0.");
        if (WeightLearningRate < 0)
            throw new InvalidOperationException("WeightLearningRate must be >= 0.");
        if (WeightDecay < 0)
            throw new InvalidOperationException("WeightDecay must be >= 0.");
        if (InputShape is not null)
        {
            if (InputShape.Any(d => d <= 0))
                throw new InvalidOperationException("All InputShape dimensions must be > 0.");
            if (InputSize != LayerSizes[0])
                throw new InvalidOperationException(
                    $"InputShape product ({InputSize}) must equal LayerSizes[0] ({LayerSizes[0]}).");
        }
    }
}
