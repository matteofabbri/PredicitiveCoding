using PredictiveCoding.Core;

namespace PredictiveCoding.Networks;

public sealed record NetworkConfig
{

    public string Path { get; set; } = "./data";

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
    /// When true, LayerStore keeps live Layer objects resident in memory and
    /// defers serialization + FasterKV persistence until an explicit Flush().
    /// Reads are served from the resident objects, avoiding per-sample
    /// serialization. Dramatically improves throughput for large weight
    /// matrices (e.g. CIFAR-10) where per-sample disk writes would saturate I/O.
    /// In eager mode (false), every write is serialized to FasterKV immediately
    /// and reads go straight through FasterKV.
    /// </summary>
    public bool LazyPersistence { get; init; } = false;

    /// <summary>
    /// When true, runs a single top-down forward pass before the iterative
    /// inference loop to seed hidden-layer states (Feedforward Gist init,
    /// Haider et al. 2024).  Propagates the clamped top state downward via
    ///   μ_l ← pred_{l+1} = f(W_{l+1} μ_{l+1} + b_{l+1})
    /// giving a warm start that reduces the inference steps needed to converge.
    /// Most effective when the top layer is clamped (training and prediction).
    /// </summary>
    public bool UseFeedforwardGistInit { get; init; } = false;

    /// <summary>
    /// L1 activity-regularisation coefficient λ_act (Keller &amp; Mrsic-Flogel 2018).
    /// Adds a sparsity gradient to the hidden-state update:
    ///   grad_l += λ_act · sign(μ_l)
    /// Encourages distributed sparse representations without changing the
    /// generative-model weights.  Default = 0 (disabled).
    /// </summary>
    public float ActivityRegularization { get; init; } = 0f;

    /// <summary>
    /// Optional shape of the raw input tensor, e.g. [28, 28] for a grayscale image
    /// or [3, 32, 32] for an RGB image.  When set, <see cref="Network.Infer"/> and
    /// <see cref="Network.Learn"/> automatically flatten the input before processing.
    /// The product must equal LayerSizes[0].  Leave null for 1-D inputs.
    /// </summary>
    public int[]? InputShape { get; init; } = null;

    /// <summary>
    /// Optional shape of the raw output/label tensor, e.g. [28, 28] for image
    /// reconstruction or [4, 4] for a 2-D label map.  When set,
    /// <see cref="Network.Infer"/> and <see cref="Network.Learn"/> automatically
    /// flatten the label before clamping the top layer.
    /// The product must equal LayerSizes[^1].  Leave null for 1-D labels.
    /// </summary>
    public int[]? OutputShape { get; init; } = null;

    /// <summary>Total input units: product of InputShape if set, else LayerSizes[0].</summary>
    public int InputSize => InputShape is null
        ? LayerSizes[0]
        : InputShape.Aggregate(1, (acc, d) => acc * d);

    /// <summary>Total output units: product of OutputShape if set, else LayerSizes[^1].</summary>
    public int OutputSize => OutputShape is null
        ? LayerSizes[^1]
        : OutputShape.Aggregate(1, (acc, d) => acc * d);

    public void Validate()
    {
        Path = System.IO.Path.GetFullPath(Path);

        if(!System.IO.Directory.Exists(Path))
            System.IO.Directory.CreateDirectory(Path);

        if (LayerSizes.Length < 2)
            throw new InvalidOperationException("LayerSizes must have at least 2 elements.");
        if (StateLearningRate < 0)
            throw new InvalidOperationException("StateLearningRate must be >= 0.");
        if (WeightLearningRate < 0)
            throw new InvalidOperationException("WeightLearningRate must be >= 0.");
        if (WeightDecay < 0)
            throw new InvalidOperationException("WeightDecay must be >= 0.");
        if (ActivityRegularization < 0)
            throw new InvalidOperationException("ActivityRegularization must be >= 0.");
        if (InputShape is not null)
        {
            if (InputShape.Any(d => d <= 0))
                throw new InvalidOperationException("All InputShape dimensions must be > 0.");
            if (InputSize != LayerSizes[0])
                throw new InvalidOperationException(
                    $"InputShape product ({InputSize}) must equal LayerSizes[0] ({LayerSizes[0]}).");
        }
        if (OutputShape is not null)
        {
            if (OutputShape.Any(d => d <= 0))
                throw new InvalidOperationException("All OutputShape dimensions must be > 0.");
            if (OutputSize != LayerSizes[^1])
                throw new InvalidOperationException(
                    $"OutputShape product ({OutputSize}) must equal LayerSizes[^1] ({LayerSizes[^1]}).");
        }
    }
}
