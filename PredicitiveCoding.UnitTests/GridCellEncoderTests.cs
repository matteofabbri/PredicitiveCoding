using PredictiveCoding.Cortex;

namespace PredicitiveCoding.UnitTests;

public class GridCellEncoderTests
{
    private static GridCellEncoder Make() =>
        new(modules: 3, cellsPerModule: 16, baseFrequency: 0.5f, frequencyRatio: 1.4f, new Random(42));

    [Fact]
    public void CodeDim_IsModulesTimesCells()
    {
        var enc = Make();
        Assert.Equal(3 * 16, enc.CodeDim);
    }

    [Fact]
    public void Encode_ReturnsCodeDimVector_InUnitRange()
    {
        var enc  = Make();
        var code = enc.Encode(1.3f, -2.1f);

        Assert.Equal(enc.CodeDim, code.Length);
        Assert.All(code.Buffer, v => Assert.InRange(v, 0f, 1f));
    }

    [Fact]
    public void Encode_IsDeterministic_ForSamePosition()
    {
        var enc = Make();
        var a   = enc.Encode(0.7f, 0.4f);
        var b   = enc.Encode(0.7f, 0.4f);
        Assert.Equal(a.Buffer, b.Buffer);
    }

    [Fact]
    public void Encode_DiffersForDistantPositions()
    {
        var enc = Make();
        var a   = enc.Encode(0f, 0f);
        var b   = enc.Encode(5f, 5f);

        float diff = 0f;
        for (int i = 0; i < a.Length; i++) diff += MathF.Abs(a.Buffer[i] - b.Buffer[i]);
        Assert.True(diff > 1e-3f, "Grid code should change across distant positions.");
    }
}
