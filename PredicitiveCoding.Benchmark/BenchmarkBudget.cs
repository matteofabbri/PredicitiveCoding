namespace PredictiveCoding.Benchmark;

/// <summary>Caps that keep a benchmark fast enough for a demo run.</summary>
public sealed record BenchmarkBudget(int MaxIdentities = 400, int MaxImages = 14000, int MaxPairs = 6000, int MaxProbesPerIdentity = 4);
