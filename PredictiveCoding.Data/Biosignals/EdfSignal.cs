using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Biosignals;

/// <summary>
/// One channel of an EDF recording: physical-unit samples plus its metadata.
/// </summary>
/// <param name="Label">Channel label (e.g. "EEG Fpz-Cz").</param>
/// <param name="PhysicalDimension">Physical unit (e.g. "uV").</param>
/// <param name="SamplingRate">Samples per second.</param>
/// <param name="Samples">1-D tensor of samples in physical units.</param>
public sealed record EdfSignal(
    string Label,
    string PhysicalDimension,
    double SamplingRate,
    DenseTensor<float> Samples);
