using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

public class PCLayerTests
{
    private static Layer MakeLayer(int size, int lowerSize, ActivationType act = ActivationType.Linear)
    {
        var layer = new Layer
        {
            Index              = 1,
            Size               = size,
            LowerSize          = lowerSize,
            ActivationType     = act,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(42));
        return layer;
    }

    // ── Initialize ────────────────────────────────────────────────────────────

    [Fact]
    public void Initialize_InputLayer_HasNoWeights()
    {
        var layer = new Layer { Index = 0, Size = 4, LowerSize = 0, ActivationType = ActivationType.Linear };
        layer.Initialize(new Random(1));

        Assert.Null(layer.GenerativeWeights);
        Assert.Null(layer.Bias);
        Assert.NotNull(layer.Mu);
        Assert.Equal(4, layer.Mu.Length);
    }

    [Fact]
    public void Initialize_HiddenLayer_HasWeightsAndBias()
    {
        var layer = MakeLayer(size: 4, lowerSize: 3);

        Assert.NotNull(layer.GenerativeWeights);
        Assert.NotNull(layer.Bias);
        Assert.Equal(3, layer.GenerativeWeights!.Dimensions[0]); // lowerSize rows
        Assert.Equal(4, layer.GenerativeWeights!.Dimensions[1]); // size cols
        Assert.Equal(3, layer.Bias!.Length);
    }

    // ── ComputePrediction ─────────────────────────────────────────────────────

    [Fact]
    public void ComputePrediction_Linear_MatchesMatVecMul()
    {
        var layer = MakeLayer(size: 2, lowerSize: 2, act: ActivationType.Linear);

        // Set known weights: W = I (2×2), bias = 0, mu = [1,2]
        var w = layer.GenerativeWeights!;
        w.Zero();
        w[0, 0] = 1f; w[1, 1] = 1f;
        layer.Bias!.Zero();
        layer.Mu[0] = 1f; layer.Mu[1] = 2f;

        layer.ComputePrediction();

        Assert.Equal(1f, layer.Prediction![0], precision: 5);
        Assert.Equal(2f, layer.Prediction![1], precision: 5);
    }

    // ── SetPredictionError ────────────────────────────────────────────────────

    [Fact]
    public void SetPredictionError_NullPrediction_ZeroesError()
    {
        var layer = MakeLayer(size: 3, lowerSize: 0);
        layer.PredictionError.Fill(5f);

        layer.SetPredictionError(null);

        Assert.All(layer.PredictionError.Buffer, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void SetPredictionError_KnownValues_CorrectDifference()
    {
        var layer = MakeLayer(size: 2, lowerSize: 0);
        layer.Mu[0] = 3f; layer.Mu[1] = 5f;
        var pred = new DenseTensor<float>([1f, 2f], 2);

        layer.SetPredictionError(pred);

        Assert.Equal(2f, layer.PredictionError[0], precision: 5);
        Assert.Equal(3f, layer.PredictionError[1], precision: 5);
    }

    // ── LocalEnergy ───────────────────────────────────────────────────────────

    [Fact]
    public void LocalEnergy_HalfSumSquaresOfError()
    {
        var layer = MakeLayer(size: 2, lowerSize: 0);
        layer.PredictionError[0] = 3f;
        layer.PredictionError[1] = 4f;

        float energy = layer.LocalEnergy();

        Assert.Equal(12.5f, energy, precision: 5); // 0.5 * (9 + 16)
    }

    // ── IsClamped ─────────────────────────────────────────────────────────────

    [Fact]
    public void UpdateState_Clamped_DoesNotModifyMu()
    {
        var layer = MakeLayer(size: 2, lowerSize: 2);
        layer.Mu[0] = 7f; layer.Mu[1] = 8f;
        layer.IsClamped = true;

        layer.UpdateState(null, etaState: 0.5f);

        Assert.Equal(7f, layer.Mu[0]);
        Assert.Equal(8f, layer.Mu[1]);
    }
}
