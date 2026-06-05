namespace PredictiveCoding.Benchmark;

/// <summary>Where to get a dataset and whether it needs registration.</summary>
public sealed record AcquisitionInfo(string Url, string Notes, bool Gated);
