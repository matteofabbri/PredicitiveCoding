using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

// ─────────────────────────────────────────────────────────────────────────────
// Tests for the precision-weighting and prior-mean features derived from:
//   Bogacz, R. (2017). A tutorial on the free-energy framework for modelling
//   perception and learning. Journal of Mathematical Psychology, 76, 198–211.
//
// Key equations tested
// ──────────────────────────────────────────────────────────────────────────────
//   Eq. 6:  ε_x = Π_x (x − g(v)),   ε_v = Π_v (v − v_p)
//   Eq. 11: v* = (θ^T Π_x θ + Π_v)^{-1} θ^T Π_x x   [Bayesian posterior mean]
//   Sec 2.4.2: ∂F/∂Π = ½||residual||² − ½ n / Π      [precision learning]
//
// Analytical fixed point for a [1,1] linear network with weight θ, prior
// precision Π_v and prior mean v_p, sensory precision Π_x and input x:
//
//   v* = (θ Π_x x + Π_v v_p) / (θ² Π_x + Π_v)      (derived from dv/dt = 0)
//
// Special cases
//   Π_x = Π_v = 1, v_p = 0 → v* = θ x / (θ² + 1)   MAP / posterior mean
//   Π_v → 0 (flat)          → v* = x / θ              MLE (current default)
//   Π_v → ∞                 → v* = v_p                prior dominates
// ─────────────────────────────────────────────────────────────────────────────

public class PrecisionTests
{
    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Builds a [1,1] linear network with a known weight θ, sets the top-layer
    /// prior mean and both layer precisions, then runs inference and returns μ_1*.
    /// </summary>
    private static float InferOneLayerNet(
        float theta,
        float input,
        float piSensory,
        float piPrior,
        float priorMean  = 0f,
        int   steps      = 600,
        float stateLR    = 0.05f)
    {
        var net = new Network(
            new NetworkConfig { LayerSizes = [1, 1], Activation = ActivationType.Linear, StateLearningRate = stateLR, WeightLearningRate = 0f },
            new Random(0));

        net.Layers[1].GenerativeWeights![0, 0] = theta;
        net.Layers[1].Bias![0]                  = 0f;
        net.Layers[1].Mu[0]                     = 0f;

        // Set precision and prior on the appropriate layers
        net.Layers[0].Precision = piSensory;
        net.Layers[1].Precision = piPrior;
        net.Layers[1].PriorMean = Vec(priorMean);

        net.Infer(Vec(input), inferenceSteps: steps);
        return net.Layers[1].Mu[0];
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Eq. 6  Precision-weighted error ε_l = Π_l · (μ_l − prediction)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 6: SetPredictionError must multiply the residual by Precision.
    /// With Π=3 and residual=2, PredictionError[0] should be 6.
    /// </summary>
    [Fact]
    public void SetPredictionError_ScalesByPrecision()
    {
        var layer = new Layer
        {
            Index = 0, Size = 1, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.Mu[0]       = 5f;
        layer.Precision   = 3f;
        var pred = Vec(3f); // residual = 5 - 3 = 2

        layer.SetPredictionError(pred);

        Assert.Equal(6f, layer.PredictionError[0], precision: 5); // Π · residual = 3 · 2
    }

    /// <summary>
    /// Unit precision (Π=1) must leave the residual unchanged — backward
    /// compatible with the original implementation.
    /// </summary>
    [Fact]
    public void SetPredictionError_UnitPrecision_EqualsRawResidual()
    {
        var layer = new Layer
        {
            Index = 0, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.Mu[0] = 4f; layer.Mu[1] = 1f;
        var pred = Vec(1f, 3f); // residuals = [3, -2]

        layer.SetPredictionError(pred);

        Assert.Equal(3f,  layer.PredictionError[0], precision: 5);
        Assert.Equal(-2f, layer.PredictionError[1], precision: 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Eq. 6  Prior mean  ε_v = Π_v (v − v_p)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 6: the top-layer prediction error with an informative prior
    /// is ε_v = Π_v (μ − v_p), not zero.
    /// </summary>
    [Fact]
    public void TopLayer_WithPriorMean_ComputesNonZeroError()
    {
        var layer = new Layer
        {
            Index = 1, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.Mu[0] = 3f; layer.Mu[1] = 1f;
        layer.PriorMean = Vec(1f, 4f);  // v_p = [1, 4]
        layer.Precision = 2f;           // Π_v = 2

        layer.SetPredictionError(null); // top layer

        // ε_v = Π (μ − v_p) = 2 · [3-1, 1-4] = [4, -6]
        Assert.Equal(4f,  layer.PredictionError[0], precision: 5);
        Assert.Equal(-6f, layer.PredictionError[1], precision: 5);
    }

    /// <summary>
    /// With null PriorMean (flat prior), the top-layer error is still zero —
    /// preserving backward compatibility.
    /// </summary>
    [Fact]
    public void TopLayer_FlatPrior_ErrorIsZero()
    {
        var layer = new Layer
        {
            Index = 1, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
            Precision = 5f,
        };
        layer.Initialize(new Random(0));
        layer.Mu[0] = 99f; layer.Mu[1] = -99f;

        layer.SetPredictionError(null);

        Assert.Equal(0f, layer.PredictionError[0], precision: 5);
        Assert.Equal(0f, layer.PredictionError[1], precision: 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Eq. 1 / LocalEnergy  F_l = ½ ||ε_l||² / Π_l
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 1: LocalEnergy = ½ Π ||residual||² = ½ ||ε||² / Π.
    /// With ε = Π · residual stored in PredictionError, the formula
    /// ½ ||ε||² / Π recovers the true free-energy contribution.
    /// </summary>
    [Fact]
    public void LocalEnergy_WithPrecision_CorrectBogaczFormula()
    {
        var layer = new Layer
        {
            Index = 0, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
            Precision = 4f,
        };
        layer.Initialize(new Random(0));

        // Inject ε directly (simulating that SetPredictionError was called)
        layer.PredictionError[0] = 4f;  // ε = Π · residual = 4 · 1 = 4
        layer.PredictionError[1] = 8f;  // ε = 4 · 2 = 8

        // True F = ½ Π (1² + 2²) = ½ · 4 · 5 = 10
        // Via formula: ½ ||ε||² / Π = ½ (16 + 64) / 4 = 10
        Assert.Equal(10f, layer.LocalEnergy(), precision: 5);
    }

    /// <summary>
    /// LocalEnergy with Π=1 (default) must match the old formula ½ ||ε||²,
    /// ensuring no regression for existing callers.
    /// </summary>
    [Fact]
    public void LocalEnergy_UnitPrecision_MatchesOldFormula()
    {
        var layer = new Layer
        {
            Index = 0, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.PredictionError[0] = 3f;
        layer.PredictionError[1] = 4f;

        Assert.Equal(12.5f, layer.LocalEnergy(), precision: 5); // ½(9+16)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Eq. 11  Bayesian posterior mean
    // v* = (θ^T Π_x θ + Π_v)^{-1} θ^T Π_x x
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 11: with unit precisions and zero prior mean, inference must
    /// converge to the Bayesian posterior mean v* = θ x / (θ² + 1),
    /// NOT the MLE v* = x / θ that a flat prior would produce.
    ///
    /// For θ=1, x=4: v* = 4/2 = 2  (vs. MLE = 4).
    /// For θ=2, x=3: v* = 6/5 = 1.2 (vs. MLE = 1.5).
    /// </summary>
    [Theory]
    [InlineData(1f, 4f, 2.0f)]    // v* = θx/(θ²+1) = 4/2
    [InlineData(2f, 3f, 1.2f)]    // v* = 6/5
    [InlineData(0.5f, 4f, 1.6f)]  // v* = 2/1.25
    public void Inference_UnitPrecisions_ConvergesToBayesianPosterior(
        float theta, float x, float expectedV)
    {
        float v = InferOneLayerNet(theta, x, piSensory: 1f, piPrior: 1f, priorMean: 0f);
        Assert.Equal(expectedV, v, precision: 1);
    }

    /// <summary>
    /// Bogacz eq. 11: with a non-zero prior mean v_p, the posterior mean is
    ///   v* = (θ Π_x x + Π_v v_p) / (θ² Π_x + Π_v)
    ///
    /// For θ=1, Π_x=Π_v=1, x=6, v_p=2:
    ///   v* = (6 + 2) / (1 + 1) = 4.
    /// </summary>
    [Fact]
    public void Inference_NonZeroPriorMean_ShiftsFixedPoint()
    {
        float v = InferOneLayerNet(theta: 1f, input: 6f,
            piSensory: 1f, piPrior: 1f, priorMean: 2f);
        Assert.Equal(4f, v, precision: 1);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Eq. 11 — Precision ratio controls data / prior balance
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 11: high sensory precision (Π_x >> Π_v) makes v* ≈ x/θ
    /// (data dominates; network ignores the prior).
    /// </summary>
    [Fact]
    public void HighSensoryPrecision_FixedPoint_NearMLE()
    {
        // θ=1, Π_x=100, Π_v=1, v_p=0, x=3
        // v* = 100·3 / (100+1) ≈ 2.97 ≈ 3 = x/θ (MLE)
        // Stable step size: η < 2/(θ²·100+1)=2/101 ≈ 0.0198 → use 0.005
        float v = InferOneLayerNet(theta: 1f, input: 3f,
            piSensory: 100f, piPrior: 1f, priorMean: 0f,
            steps: 800, stateLR: 0.005f);
        Assert.Equal(3f, v, precision: 1);
    }

    /// <summary>
    /// Bogacz eq. 11: high prior precision (Π_v >> Π_x) drives v* toward v_p
    /// regardless of the sensory input (prior dominates).
    /// </summary>
    [Fact]
    public void HighPriorPrecision_FixedPoint_NearPriorMean()
    {
        // θ=1, Π_x=1, Π_v=100, v_p=7, x=3
        // v* = (3 + 100·7) / (1 + 100) = 703/101 ≈ 6.96 ≈ 7 = v_p
        // Stable step size: η < 2/(1+100)=2/101 ≈ 0.0198 → use 0.005
        float v = InferOneLayerNet(theta: 1f, input: 3f,
            piSensory: 1f, piPrior: 100f, priorMean: 7f,
            steps: 800, stateLR: 0.005f);
        Assert.Equal(7f, v, precision: 1);
    }

    /// <summary>
    /// With equal sensory and prior precision, the posterior mean is exactly
    /// halfway between the MLE estimate and the prior mean.
    ///
    /// For θ=1, Π_x=Π_v, x=0, v_p=4:
    ///   v* = (0 + 4) / 2 = 2.   (half-way to prior)
    /// </summary>
    [Fact]
    public void EqualPrecisions_PosteriorMean_HalfwayBetweenMLEAndPrior()
    {
        // θ=1, Π_x=Π_v=1, x=0, v_p=4 → v* = 2
        float v = InferOneLayerNet(theta: 1f, input: 0f,
            piSensory: 1f, piPrior: 1f, priorMean: 4f);
        Assert.Equal(2f, v, precision: 1);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2.4.2  Precision learning
    // ∂F/∂Π = ½ ||residual||² − ½ n/Π   →   Π* = n / ||residual||²
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz section 2.4.2: UpdatePrecision performs gradient descent on the
    /// free energy w.r.t. Π.  After many updates starting from a mismatched
    /// precision, Π must converge toward the MLE value  n / ||residual||².
    /// </summary>
    [Fact]
    public void UpdatePrecision_ConvergesToMLEPrecision()
    {
        const int n = 1;             // 1-D layer
        const float trueResidual = 2f;

        var layer = new Layer
        {
            Index = 0, Size = n, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
            Precision = 5f,  // start far from optimum
        };
        layer.Initialize(new Random(0));

        // Fix a constant prediction error corresponding to residual = 2
        // ε = Π · residual  →  stored in PredictionError dynamically per update
        for (int i = 0; i < 2000; i++)
        {
            // Simulate what SetPredictionError would store:
            layer.PredictionError[0] = layer.Precision * trueResidual;
            layer.UpdatePrecision(eta: 0.01f);
        }

        // MLE optimum: Π* = n / ||residual||² = 1 / 4 = 0.25
        Assert.Equal(0.25f, layer.Precision, precision: 2);
    }

    /// <summary>
    /// Bogacz section 2.4.2: precision learning reduces the free energy over
    /// time (gradient descent on F w.r.t. Π must decrease F).
    /// </summary>
    [Fact]
    public void UpdatePrecision_ReducesFreeEnergyOverTime()
    {
        var net   = new Network(
            new NetworkConfig { LayerSizes = [2, 3], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(11));
        var input = Vec(1f, -1f);

        // Run inference to a steady state
        net.Infer(input, inferenceSteps: 200);

        float energyBefore = net.ComputeEnergy();

        // Perform precision updates on the input layer
        for (int i = 0; i < 50; i++)
        {
            net.Infer(input, inferenceSteps: 5);
            net.Layers[0].UpdatePrecision(eta: 0.01f);
        }

        float energyAfter = net.ComputeEnergy();

        Assert.True(energyAfter < energyBefore,
            $"Precision learning should reduce F: before={energyBefore:F4}, after={energyAfter:F4}");
    }

    /// <summary>
    /// Bogacz section 2.4.2: when the prediction error is small (network
    /// explains the input well), precision learning should increase Π
    /// (higher confidence in predictions — more precise model).
    /// </summary>
    [Fact]
    public void UpdatePrecision_SmallResidual_IncreasesConfidence()
    {
        var layer = new Layer
        {
            Index = 0, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
            Precision = 1f,
        };
        layer.Initialize(new Random(0));

        // Very small residual → MLE optimum Π* = n/||residual||² >> 1
        // Inject small ε = Π · residual = 1 · 0.1 = 0.1 per element
        layer.PredictionError[0] = 0.1f;
        layer.PredictionError[1] = 0.1f;

        float piInitial = layer.Precision;
        layer.UpdatePrecision(eta: 0.1f);

        Assert.True(layer.Precision > piInitial,
            $"Precision should increase when residuals are small: {piInitial:F4} → {layer.Precision:F4}");
    }

    /// <summary>
    /// Bogacz section 2.4.2: when the prediction error is large (poor model
    /// fit), precision learning should decrease Π (less confident model).
    /// </summary>
    [Fact]
    public void UpdatePrecision_LargeResidual_DecreasesConfidence()
    {
        var layer = new Layer
        {
            Index = 0, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
            Precision = 5f,
        };
        layer.Initialize(new Random(0));

        // Large residual → MLE Π* = n/||residual||² << 5
        layer.PredictionError[0] = 10f;  // ε = Π · residual = 5 · 2
        layer.PredictionError[1] = 10f;

        float piInitial = layer.Precision;
        layer.UpdatePrecision(eta: 0.1f);

        Assert.True(layer.Precision < piInitial,
            $"Precision should decrease when residuals are large: {piInitial:F4} → {layer.Precision:F4}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Diagonal precision  (Huang & Rao 2011, diagonal Σ_l)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// InitDiagonalPrecision must create a per-unit vector of size Size,
    /// all initialised to 1 (unit precision), and the scalar Precision must
    /// be overridden in SetPredictionError.
    /// </summary>
    [Fact]
    public void DiagonalPrecision_Init_AllOnes()
    {
        var layer = new Layer
        {
            Index = 0, Size = 3, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.InitDiagonalPrecision();

        Assert.NotNull(layer.DiagonalPrecision);
        Assert.Equal(3, layer.DiagonalPrecision!.Length);
        Assert.All(layer.DiagonalPrecision.Buffer, v => Assert.Equal(1f, v));
    }

    /// <summary>
    /// SetPredictionError with diagonal precision must scale each unit
    /// independently: ε[i] = Π[i] · (μ[i] − pred[i]).
    /// </summary>
    [Fact]
    public void DiagonalPrecision_SetPredictionError_ScalesPerUnit()
    {
        var layer = new Layer
        {
            Index = 0, Size = 3, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.InitDiagonalPrecision();
        // Override per-unit precisions: [2, 3, 1]
        layer.DiagonalPrecision![0] = 2f;
        layer.DiagonalPrecision![1] = 3f;
        layer.DiagonalPrecision![2] = 1f;

        layer.Mu[0] = 5f; layer.Mu[1] = 4f; layer.Mu[2] = 2f;
        var pred = Vec(3f, 1f, 2f); // residuals = [2, 3, 0]

        layer.SetPredictionError(pred);

        Assert.Equal(4f,  layer.PredictionError[0], precision: 5); // 2 · 2
        Assert.Equal(9f,  layer.PredictionError[1], precision: 5); // 3 · 3
        Assert.Equal(0f,  layer.PredictionError[2], precision: 5); // 1 · 0
    }

    /// <summary>
    /// LocalEnergy with diagonal precision:
    ///   F_l = ½ Σ_i ε[i]² / Π[i]
    /// </summary>
    [Fact]
    public void DiagonalPrecision_LocalEnergy_PerUnitFormula()
    {
        var layer = new Layer
        {
            Index = 0, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.InitDiagonalPrecision();
        layer.DiagonalPrecision![0] = 2f;
        layer.DiagonalPrecision![1] = 4f;

        // Inject ε directly: ε = [4, 8]  (residuals = [2, 2])
        layer.PredictionError[0] = 4f;
        layer.PredictionError[1] = 8f;

        // F = ½(4²/2 + 8²/4) = ½(8 + 16) = 12
        Assert.Equal(12f, layer.LocalEnergy(), precision: 5);
    }

    /// <summary>
    /// UpdateDiagonalPrecision must converge each per-unit precision to its
    /// MLE fixed point: Π[i]* = 1 / residual[i]²
    /// (Huang & Rao 2011, diagonal Σ learning).
    /// </summary>
    [Fact]
    public void DiagonalPrecision_Update_ConvergesToMLEPerUnit()
    {
        float[] trueResiduals = [2f, 0.5f]; // expected Π* = [0.25, 4.0]

        var layer = new Layer
        {
            Index = 0, Size = 2, LowerSize = 0,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(0));
        layer.InitDiagonalPrecision();
        layer.DiagonalPrecision![0] = 5f;
        layer.DiagonalPrecision![1] = 0.1f;

        for (int i = 0; i < 10000; i++)
        {
            for (int j = 0; j < 2; j++)
                layer.PredictionError[j] = layer.DiagonalPrecision![j] * trueResiduals[j];
            layer.UpdateDiagonalPrecision(eta: 0.05f);
        }

        Assert.Equal(0.25f, layer.DiagonalPrecision![0], precision: 2); // 1/2² = 0.25
        Assert.Equal(4.0f,  layer.DiagonalPrecision![1], precision: 2); // 1/0.5² = 4.0
    }

    // ─────────────────────────────────────────────────────────────────────────
    // L2 weight regularisation  (Huang & Rao 2011, λ||W||² term)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// With WeightDecay > 0, repeated weight updates with zero Hebbian signal
    /// must shrink all weights toward zero (L2 decay).
    /// </summary>
    [Fact]
    public void WeightDecay_ShrinksWeightsOverTime()
    {
        var layer = new Layer
        {
            Index = 1, Size = 2, LowerSize = 2,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
            WeightDecay = 0.1f,
        };
        layer.Initialize(new Random(0));

        float normBefore = layer.WeightNorm();

        // lowerLayerError = 0 → Hebbian term is zero; only decay acts
        var zeroError = new DenseTensor<float>(new float[2], 2);
        layer.ComputePrediction();

        for (int i = 0; i < 100; i++)
            layer.UpdateWeights(zeroError, etaWeight: 0.01f);

        float normAfter = layer.WeightNorm();
        Assert.True(normAfter < normBefore,
            $"Weight decay should shrink norms: before={normBefore:F4}, after={normAfter:F4}");
    }

    /// <summary>
    /// WeightDecay = 0 must behave identically to the old code (no decay).
    /// </summary>
    [Fact]
    public void WeightDecay_Zero_NoEffect()
    {
        var layer = new Layer
        {
            Index = 1, Size = 2, LowerSize = 2,
            ActivationType = ActivationType.Linear,
            StateLearningRate = 0.05f, WeightLearningRate = 0.01f,
            WeightDecay = 0f,
        };
        layer.Initialize(new Random(42));
        float normBefore = layer.WeightNorm();

        var zeroError = new DenseTensor<float>(new float[2], 2);
        layer.ComputePrediction();
        layer.UpdateWeights(zeroError, etaWeight: 0.01f);

        Assert.Equal(normBefore, layer.WeightNorm(), precision: 5);
    }
}
