namespace PredictiveCoding.Brain;

/// <summary>
/// Encodes a value of type <typeparamref name="T"/> into a fixed-size
/// <see cref="Sdr"/>. Encoders are the entry point of the Thousand-Brains
/// pipeline: they turn raw inputs (scalars, categories, coordinates, …) into the
/// sparse, semantically-overlapping codes that columns operate on.
/// </summary>
public interface ISdrEncoder<in T>
{
    /// <summary>Dimensionality of the produced SDRs.</summary>
    int Size { get; }

    /// <summary>Encodes <paramref name="value"/> into an SDR of length <see cref="Size"/>.</summary>
    Sdr Encode(T value);
}
