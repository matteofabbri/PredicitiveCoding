using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Datasets;

/// <summary>Which MNIST-format dataset to download.</summary>
public enum MnistVariant
{
    /// <summary>The original handwritten-digit MNIST.</summary>
    Mnist,

    /// <summary>Zalando's Fashion-MNIST (clothing images, same format).</summary>
    FashionMnist,
}

/// <summary>
/// A loaded MNIST-format split: images as a <c>[count, rows, cols]</c> tensor and
/// the matching integer labels.
/// </summary>
public sealed class MnistData
{
    /// <summary>Images, shape <c>[count, rows, cols]</c>, values 0..1 when normalised.</summary>
    public DenseTensor<float> Images { get; }

    /// <summary>Class label per image.</summary>
    public IReadOnlyList<int> Labels { get; }

    /// <summary>Number of examples.</summary>
    public int Count => Labels.Count;

    public MnistData(DenseTensor<float> images, IReadOnlyList<int> labels)
    {
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(labels);
        if (images.Rank != 3)
            throw new ArgumentException("Images must be a 3-D tensor [count, rows, cols].", nameof(images));
        if (images.Dimensions[0] != labels.Count)
            throw new ArgumentException(
                $"Image count {images.Dimensions[0]} does not match label count {labels.Count}.", nameof(labels));

        Images = images;
        Labels = labels;
    }
}
