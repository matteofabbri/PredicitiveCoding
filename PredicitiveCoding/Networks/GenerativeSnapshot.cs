using PredictiveCoding.Core;

namespace PredictiveCoding.Networks;

/// <summary>
/// Read-only copy of a <see cref="Network"/>'s generative model — layer sizes,
/// activation, per-layer weights/biases/precisions and the inference
/// hyper-parameters — for out-of-core consumers. The batched (GEMM-shaped,
/// GPU-routable) forward in <c>PredictiveCoding.Device</c> is the intended
/// customer: it needs the model but not the store.
/// </summary>
/// <param name="Weights">Index l−1 holds W_l [LowerSize × Size], l = 1..L−1.</param>
/// <param name="Biases">Index l−1 holds b_l [LowerSize].</param>
/// <param name="Precisions">Scalar Π_l per layer, l = 0..L−1.</param>
public sealed record GenerativeSnapshot(
    int[]                LayerSizes,
    ActivationType       Activation,
    float                StateLearningRate,
    float                ActivityRegularization,
    bool                 UseFeedforwardGistInit,
    DenseTensor<float>[] Weights,
    DenseTensor<float>[] Biases,
    float[]              Precisions);
