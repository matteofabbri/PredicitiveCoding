using PredictiveCoding.Core;
using PredictiveCoding.Networks.Hopfield;

namespace PredicitiveCoding.UnitTests;

public class HopfieldMemoryTests
{
    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    private static float L2(DenseTensor<float> a, DenseTensor<float> b)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++) { float d = a.Buffer[i] - b.Buffer[i]; s += d * d; }
        return s;
    }

    [Fact]
    public void EmptyMemory_ReturnsQueryUnchanged()
    {
        var mem = new HopfieldMemory(4);
        var q   = Vec(1, 2, 3, 4);
        var r   = mem.Retrieve(q, beta: 8f, iterations: 3);
        Assert.Equal(q.Buffer, r.Buffer);
    }

    [Fact]
    public void Store_IncrementsCount()
    {
        var mem = new HopfieldMemory(4);
        Assert.Equal(0, mem.Count);
        mem.Store(Vec(1, 0, 0, 0));
        mem.Store(Vec(0, 1, 0, 0));
        Assert.Equal(2, mem.Count);
    }

    [Fact]
    public void Retrieve_CompletesNoisyCue_ToNearestStoredPattern()
    {
        var mem = new HopfieldMemory(8);
        var p0  = Vec(1, 1, 1, 1, 0, 0, 0, 0);
        var p1  = Vec(0, 0, 0, 0, 1, 1, 1, 1);
        var p2  = Vec(1, 0, 1, 0, 1, 0, 1, 0);
        mem.Store(p0); mem.Store(p1); mem.Store(p2);

        var noisy = Vec(0.9f, 1.1f, 0.8f, 1.2f, 0.1f, -0.1f, 0.05f, 0f); // corrupted p0
        var r     = mem.Retrieve(noisy, beta: 16f, iterations: 5);

        Assert.True(L2(r, p0) < L2(r, p1));
        Assert.True(L2(r, p0) < L2(r, p2));
    }

    [Fact]
    public void Retrieve_WrongDim_Throws()
    {
        var mem = new HopfieldMemory(4);
        mem.Store(Vec(1, 0, 0, 0));
        Assert.Throws<ArgumentException>(() => mem.Retrieve(Vec(1, 2, 3), 8f, 2));
    }
}
