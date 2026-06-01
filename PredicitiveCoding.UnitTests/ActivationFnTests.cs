using PredictiveCoding.Core;

namespace PredicitiveCoding.UnitTests;

public class ActivationFnTests
{
    private static DenseTensor<float> Vec(params float[] values) =>
        new(values, values.Length);

    // ── Apply ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Apply_Linear_PassesThrough()
    {
        var x = Vec(-1f, 0f, 1f, 2f);
        var z = ActivationFn.Apply(x, ActivationType.Linear);
        Assert.Equal(x.Buffer, z.Buffer);
    }

    [Fact]
    public void Apply_Tanh_CorrectValues()
    {
        var x = Vec(0f, 1f, -1f);
        var z = ActivationFn.Apply(x, ActivationType.Tanh);
        Assert.Equal(0f,               z[0], precision: 5);
        Assert.Equal(MathF.Tanh(1f),  z[1], precision: 5);
        Assert.Equal(MathF.Tanh(-1f), z[2], precision: 5);
    }

    [Fact]
    public void Apply_ReLU_ZerosNegatives()
    {
        var x = Vec(-2f, 0f, 3f);
        var z = ActivationFn.Apply(x, ActivationType.ReLU);
        Assert.Equal(0f, z[0], precision: 5);
        Assert.Equal(0f, z[1], precision: 5);
        Assert.Equal(3f, z[2], precision: 5);
    }

    [Fact]
    public void Apply_Sigmoid_InUnitInterval()
    {
        var x = Vec(-3f, 0f, 3f);
        var z = ActivationFn.Apply(x, ActivationType.Sigmoid);
        Assert.True(z[0] > 0f && z[0] < 0.5f);
        Assert.Equal(0.5f, z[1], precision: 5);
        Assert.True(z[2] > 0.5f && z[2] < 1f);
    }

    // ── Derivative ───────────────────────────────────────────────────────────

    [Fact]
    public void Derivative_Linear_AllOnes()
    {
        var x = Vec(-1f, 0f, 2f);
        var d = ActivationFn.Derivative(x, null, ActivationType.Linear);
        Assert.All(d.Buffer, v => Assert.Equal(1f, v, precision: 5));
    }

    [Fact]
    public void Derivative_Tanh_CorrectFormula()
    {
        // f'(x) = 1 - tanh(x)^2
        var x = Vec(0f, 1f);
        var y = ActivationFn.Apply(x, ActivationType.Tanh);
        var d = ActivationFn.Derivative(x, y, ActivationType.Tanh);
        Assert.Equal(1f - MathF.Tanh(0f) * MathF.Tanh(0f), d[0], precision: 5);
        Assert.Equal(1f - MathF.Tanh(1f) * MathF.Tanh(1f), d[1], precision: 5);
    }

    [Fact]
    public void Derivative_ReLU_StepFunction()
    {
        var x = Vec(-1f, 0f, 1f);
        var d = ActivationFn.Derivative(x, null, ActivationType.ReLU);
        Assert.Equal(0f, d[0], precision: 5);
        Assert.Equal(0f, d[1], precision: 5);
        Assert.Equal(1f, d[2], precision: 5);
    }

    [Fact]
    public void Derivative_Sigmoid_CorrectFormula()
    {
        // f'(x) = σ(x)(1 - σ(x))
        var x = Vec(0f, 2f);
        var sig = ActivationFn.Apply(x, ActivationType.Sigmoid);
        var d   = ActivationFn.Derivative(x, sig, ActivationType.Sigmoid);
        Assert.Equal(sig[0] * (1f - sig[0]), d[0], precision: 5);
        Assert.Equal(sig[1] * (1f - sig[1]), d[1], precision: 5);
    }

    // ── Softmax ───────────────────────────────────────────────────────────────

    [Fact]
    public void Softmax_SumsToOne()
    {
        var x = Vec(1f, 2f, 3f, 4f);
        var s = ActivationFn.Softmax(x);
        float sum = s.Buffer.Sum();
        Assert.Equal(1f, sum, precision: 5);
    }

    [Fact]
    public void Softmax_LargestInputHasLargestOutput()
    {
        var x = Vec(0.1f, 0.9f, 0.3f);
        var s = ActivationFn.Softmax(x);
        Assert.Equal(TensorUtils.Argmax(x), TensorUtils.Argmax(s));
    }
}
