using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for the Thousand-Brains Phase 0 substrate: SDRs and encoders.</summary>
public class BrainSdrTests
{
    // ── Sdr core ─────────────────────────────────────────────────────────────

    [Fact]
    public void Construction_SortsDeduplicatesAndCounts()
    {
        var s = new Sdr(100, [5, 5, 1, 3, 1]);
        Assert.Equal(100, s.Size);
        Assert.Equal(new[] { 1, 3, 5 }, s.ActiveBits);
        Assert.Equal(3, s.Cardinality);
        Assert.True(MathF.Abs(s.Sparsity - 0.03f) < 1e-6f);
        Assert.True(s.IsActive(3));
        Assert.False(s.IsActive(2));
    }

    [Fact]
    public void Construction_RejectsOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sdr(10, [10]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sdr(10, [-1]));
    }

    [Fact]
    public void Overlap_IdenticalAndDisjoint()
    {
        var a = new Sdr(20, [1, 2, 3, 4]);
        var b = new Sdr(20, [3, 4, 5, 6]);
        Assert.Equal(2, a.Overlap(b));
        Assert.Equal(4, a.Overlap(a));
        Assert.Equal(0, a.Overlap(new Sdr(20, [10, 11])));
        Assert.Equal(1f, a.OverlapScore(a));
    }

    [Fact]
    public void Overlap_SizeMismatchThrows()
        => Assert.Throws<ArgumentException>(() => new Sdr(10, [1]).Overlap(new Sdr(11, [1])));

    [Fact]
    public void UnionAndIntersect()
    {
        var a = new Sdr(20, [1, 2, 3]);
        var b = new Sdr(20, [3, 4]);
        Assert.Equal(new[] { 1, 2, 3, 4 }, a.Union(b).ActiveBits);
        Assert.Equal(new[] { 3 }, a.Intersect(b).ActiveBits);
    }

    [Fact]
    public void Random_HasRequestedCardinality_IsInRange_AndDeterministic()
    {
        var s1 = Sdr.Random(2048, 40, new Random(1));
        var s2 = Sdr.Random(2048, 40, new Random(1));
        Assert.Equal(40, s1.Cardinality);
        Assert.All(s1.ActiveBits, b => Assert.InRange(b, 0, 2047));
        Assert.Equal(s1.ActiveBits, s2.ActiveBits);   // same seed → same SDR
    }

    [Fact]
    public void AddNoise_PreservesCardinality_AndMovesExactlyTheNoiseBits()
    {
        var rng = new Random(7);
        var s = Sdr.Random(1000, 50, rng);
        var noisy = s.AddNoise(0.2f, rng);   // move 10 of 50 bits

        Assert.Equal(50, noisy.Cardinality);
        Assert.Equal(40, s.Overlap(noisy));  // new bits are disjoint from the originals
    }

    [Fact]
    public void Subsample_TakesSubsetOfActiveBits()
    {
        var s = Sdr.Random(500, 30, new Random(3));
        var sub = s.Subsample(10, new Random(3));
        Assert.Equal(10, sub.Cardinality);
        Assert.All(sub.ActiveBits, b => Assert.True(s.IsActive(b)));
    }

    [Fact]
    public void FromDense_And_ToDense_RoundTrip()
    {
        var s = Sdr.FromDense([0.1f, 0.9f, 0.6f, 0.0f, 1.0f], threshold: 0.5f);
        Assert.Equal(new[] { 1, 2, 4 }, s.ActiveBits);

        var dense = new Sdr(6, [0, 4, 5]).ToDense();
        Assert.Equal(new float[] { 1, 0, 0, 0, 1, 1 }, dense);
    }

    [Fact]
    public void Equality_ByValue()
    {
        Assert.Equal(new Sdr(20, [1, 2, 3]), new Sdr(20, [3, 2, 1, 1]));
        Assert.NotEqual(new Sdr(20, [1, 2, 3]), new Sdr(20, [1, 2, 4]));
        Assert.NotEqual(new Sdr(20, [1, 2, 3]), new Sdr(21, [1, 2, 3]));
    }

    // ── Encoders ─────────────────────────────────────────────────────────────

    [Fact]
    public void ScalarEncoder_WindowSemanticsOverlapAndClamp()
    {
        var enc = new ScalarEncoder(0, 100, n: 100, w: 21);
        Assert.Equal(100, enc.Size);

        var lo = enc.Encode(0);
        var hi = enc.Encode(100);
        Assert.Equal(21, lo.Cardinality);
        Assert.Equal(0, lo.Overlap(hi));                       // extremes share nothing

        var mid = enc.Encode(50);
        var near = enc.Encode(52);
        var far = enc.Encode(90);
        Assert.True(mid.Overlap(near) > mid.Overlap(far));     // closer values overlap more

        Assert.Equal(lo.ActiveBits, enc.Encode(-25).ActiveBits);   // clamp low
        Assert.Equal(hi.ActiveBits, enc.Encode(500).ActiveBits);   // clamp high
    }

    [Fact]
    public void CategoryEncoder_BlocksAreDisjointAndStable()
    {
        var enc = new CategoryEncoder(4, w: 8);
        Assert.Equal(32, enc.Size);

        var c0 = enc.Encode(0);
        var c1 = enc.Encode(1);
        Assert.Equal(8, c0.Cardinality);
        Assert.Equal(0, c0.Overlap(c1));                 // distinct categories disjoint
        Assert.Equal(8, c0.Overlap(enc.Encode(0)));      // same category identical
        Assert.Throws<ArgumentOutOfRangeException>(() => enc.Encode(4));
    }
}
