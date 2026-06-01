using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

// ─────────────────────────────────────────────────────────────────────────────
// Tests grounded in:
//   Bogacz, R. (2017). A tutorial on the free-energy framework for modelling
//   perception and learning. Journal of Mathematical Psychology, 76, 198–211.
//   https://doi.org/10.1016/j.jmp.2015.11.003
//
// Equation numbers refer to that paper unless otherwise noted.
// W&B = Whittington & Bogacz (2017), cited as the hierarchical extension.
//
// Implementation notes
// ─────────────────────────────────────────────────────────────────────────────
// Our network uses a flat (uninformative) prior at the top layer: ε_top is
// forced to zero each inference step, so ε_v = 0 rather than ε_v = μ - μ_prior.
// This is the limiting case of Bogacz eq. 5 with Π_v → 0 (infinite prior
// variance), making inference converge to the MLE rather than the MAP/posterior.
// Tests that involve the prior are annotated accordingly.
// ─────────────────────────────────────────────────────────────────────────────

public class BogaczTutorialTests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    /// <summary>
    /// Creates a network and sets weight/bias of layer[1] to exact values,
    /// leaving layer state as initialised.
    /// </summary>
    private static Network TwoLayerLinear(float weight, float bias = 0f,
        float stateLR = 0.05f, float weightLR = 0.01f)
    {
        var net = new Network(
            new NetworkConfig { LayerSizes = [1, 1], Activation = ActivationType.Linear, StateLearningRate = stateLR, WeightLearningRate = weightLR },
            new Random(0));
        net.Layers[1].GenerativeWeights![0, 0] = weight;
        net.Layers[1].Bias![0]                 = bias;
        net.Layers[1].Mu[0]                    = 0f; // start from zero
        return net;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2 / eq. 12  Free-energy decomposition
    // F = ½ Σ_l || ε_l ||²
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 12: F = ½ Σ_{l=0}^{L-2} || ε_l ||²
    ///
    /// We set prediction errors manually and verify ComputeEnergy() matches
    /// the closed-form sum.  The top-layer error is always zero (flat prior),
    /// so it contributes nothing.
    /// </summary>
    [Fact]
    public void FreeEnergy_EqualsSumOfSquaredPredictionErrors_DividedByTwo()
    {
        var net = new Network(
            new NetworkConfig { LayerSizes = [2, 3, 2], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(1));

        // Inject known prediction errors into layers 0 and 1
        net.Layers[0].PredictionError[0] = 2f;
        net.Layers[0].PredictionError[1] = 4f;
        net.Layers[1].PredictionError[0] = 1f;
        net.Layers[1].PredictionError[1] = 2f;
        net.Layers[1].PredictionError[2] = 2f;

        float expected = 0.5f * (4f + 16f)        // ½ || ε_0 ||²
                       + 0.5f * (1f + 4f + 4f);   // ½ || ε_1 ||²

        Assert.Equal(expected, net.ComputeEnergy(), precision: 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2.3 / eq. 5  Inference = gradient descent on F
    // dv/dt = g'(v) ε_x − ε_v  →  F decreases each step
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz section 2.3: because each inference step moves μ in the
    /// direction of −∂F/∂μ (gradient descent), the free energy must not
    /// increase between steps — provided the step size is small enough.
    ///
    /// We use stateLR = 0.02 to stay comfortably within the stable region
    /// for the random weight scales produced by Xavier initialisation.
    /// </summary>
    [Fact]
    public void Inference_FreeEnergy_NonIncreasing_PerStep()
    {
        var net   = new Network(
            new NetworkConfig { LayerSizes = [3, 4, 2], Activation = ActivationType.Linear, StateLearningRate = 0.02f, WeightLearningRate = 0f },
            new Random(7));
        var input = Vec(1f, -0.5f, 0.8f);

        // Prime the network once so it has a meaningful starting energy
        net.Infer(input, inferenceSteps: 1);
        float prev = net.ComputeEnergy();

        int regressions = 0;
        for (int step = 0; step < 80; step++)
        {
            net.Infer(input, inferenceSteps: 1);
            float current = net.ComputeEnergy();

            // Allow a tiny numerical tolerance for floating-point rounding
            if (current > prev + 1e-4f) regressions++;
            prev = current;
        }

        // At most 1 marginal regression from rounding — never a sustained climb
        Assert.True(regressions <= 1,
            $"Energy increased in {regressions} out of 80 steps.");
    }

    /// <summary>
    /// Bogacz section 2.3: 50 inference steps must produce strictly lower
    /// free energy than a single step, for a non-trivial input.
    /// </summary>
    [Fact]
    public void Inference_MoreStepsProduceLowerFreeEnergy()
    {
        // Clone-like setup: two identical networks, same seed, same input
        var rng   = new Random(42);
        var net1  = new Network(
            new NetworkConfig { LayerSizes = [3, 4, 2], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(42));
        var net2  = new Network(
            new NetworkConfig { LayerSizes = [3, 4, 2], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(42));
        var input = Vec(1f, 0f, -1f);

        net1.Infer(input, inferenceSteps: 1);
        net2.Infer(input, inferenceSteps: 50);

        Assert.True(net2.ComputeEnergy() < net1.ComputeEnergy(),
            "50 inference steps should produce lower energy than 1 step.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2.3 / eq. 10  Fixed point: ∂F/∂μ_l = 0
    // g'(v*) ε_x = ε_v  (for our flat prior: W^T ε_{l−1} = ε_l)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 10: at the inference fixed point the gradient ∂F/∂μ_l
    /// vanishes for every free layer, meaning one more update step leaves
    /// μ essentially unchanged.
    ///
    /// For linear activation and flat prior (ε_top = 0) the fixed-point
    /// condition is W_l^T ε_{l−1} − ε_l = 0 for each hidden layer l.
    /// </summary>
    [Fact]
    public void Inference_FixedPoint_GradientVanishes()
    {
        var net   = new Network(
            new NetworkConfig { LayerSizes = [2, 4, 2], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(3));
        var input = Vec(1f, -1f);

        // Run until approximate convergence
        net.Infer(input, inferenceSteps: 400);

        // Record μ of the hidden layer
        float[] muBefore = (float[])net.Layers[1].Mu.Buffer.Clone();

        // One more step
        net.Infer(input, inferenceSteps: 1);

        float maxChange = 0f;
        for (int i = 0; i < net.Layers[1].Mu.Length; i++)
            maxChange = MathF.Max(maxChange,
                MathF.Abs(net.Layers[1].Mu.Buffer[i] - muBefore[i]));

        Assert.True(maxChange < 1e-3f,
            $"Hidden state still moving by {maxChange:E2} — not at fixed point.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 3 / eq. 11  MLE fixed point (flat prior)
    // With g(v) = θv and v_p → ∞, eq. 11 reduces to v* = x / θ
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz section 3: with a flat (uninformative) prior the inference fixed
    /// point is the maximum-likelihood estimate v* = x / θ.
    ///
    /// For a [1→1] linear network with weight θ and input x:
    ///   F(v) = ½(x − θv)²  (no prior term)
    ///   ∂F/∂v = −θ(x − θv)  →  v* = x / θ
    ///
    /// Test cases:
    ///   θ=1, x=3  →  v* = 3.0
    ///   θ=2, x=3  →  v* = 1.5
    /// </summary>
    [Theory]
    [InlineData(1f, 3f, 3.0f)]
    [InlineData(2f, 3f, 1.5f)]
    [InlineData(0.5f, 2f, 4.0f)]
    public void Inference_Linear_FlatPrior_ConvergesToMLE(
        float theta, float x, float expectedV)
    {
        // stateLR=0.1 is stable for θ≤2 (condition: η < 2/θ²) and fast enough
        // that 500 steps fully converges even for the slowest case (θ=0.5).
        var net   = TwoLayerLinear(theta, stateLR: 0.1f);
        var input = Vec(x);

        net.Infer(input, inferenceSteps: 500);

        Assert.Equal(expectedV, net.Layers[1].Mu[0], precision: 1);
    }

    /// <summary>
    /// Bogacz eq. 3 / section 3: at the MLE fixed point, the sensory
    /// prediction error ε_x = x − g(v*) collapses to zero.
    ///
    /// This captures the core predictive-coding intuition: the network
    /// "explains away" the sensory signal via top-down predictions.
    /// </summary>
    [Fact]
    public void Inference_AtFixedPoint_SensoryPredictionErrorNearZero()
    {
        var net   = TwoLayerLinear(1f, stateLR: 0.05f);
        var input = Vec(2.5f);

        net.Infer(input, inferenceSteps: 500);

        // Recompute predictions so ε_0 is current
        net.Layers[1].ComputePrediction();
        net.Layers[0].SetPredictionError(net.Layers[1].Prediction);

        float errMag = MathF.Abs(net.Layers[0].PredictionError[0]);
        Assert.True(errMag < 0.01f,
            $"Sensory error |ε_0| = {errMag:F4} should be near zero at fixed point.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2.4 / eq. 7  Learning rule: Hebbian prediction-error weighting
    // dθ/dt = ε_x · v   (1-D)
    // ΔW_l  = η · ε_{l−1} ⊗ μ_l^T   (hierarchical, W&B eq. 20)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 7 / W&B eq. 20: one weight-update step changes W by
    ///   ΔW_l = η_W · f'(pre_l) ⊙ ε_{l−1}  ⊗  μ_l^T
    ///
    /// For linear activation f'=1, so ΔW = η_W · ε_{l−1} ⊗ μ_l^T.
    /// This is a pure Hebbian rule modulated by the prediction error.
    /// </summary>
    [Fact]
    public void WeightUpdate_IsHebbianPredictionError_Rule()
    {
        const float eta = 0.01f;

        // Hand-craft a layer in a known state
        var layer = new Layer
        {
            Index              = 1,
            Size               = 2,
            LowerSize          = 2,
            ActivationType     = ActivationType.Linear,
            StateLearningRate  = 0.05f,
            WeightLearningRate = eta,
        };
        layer.Initialize(new Random(0));

        // Set exact μ_1 and ε_0
        layer.Mu[0] = 3f; layer.Mu[1] = 5f;
        var lowerError = Vec(1f, -2f);  // ε_0

        // Set pre-activation so f'(pre) = 1 (linear: always 1)
        // W does not matter for the test; only ΔW is checked
        float[] wBefore = (float[])layer.GenerativeWeights!.Buffer.Clone();

        layer.UpdateWeights(lowerError, eta);

        // Expected ΔW = eta * ε_0 ⊗ μ_1^T
        // ε_0 = [1, -2],  μ_1 = [3, 5]
        // ΔW[i,j] = eta * ε_0[i] * μ_1[j]
        float[,] expected =
        {
            {  eta * 1f * 3f,  eta * 1f * 5f  },
            {  eta * (-2f) * 3f, eta * (-2f) * 5f }
        };

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                float actual = layer.GenerativeWeights!.Buffer[i * 2 + j] - wBefore[i * 2 + j];
                Assert.Equal(expected[i, j], actual, precision: 5);
            }
    }

    /// <summary>
    /// Bogacz eq. 7: the weight update and the state update share the same
    /// prediction error signal ε_x (biological plausibility — one neuron
    /// type drives both learning and inference).
    ///
    /// Verify that, after one weight update, the prediction at layer 0
    /// has moved closer to the clamped input compared to before.
    /// </summary>
    [Fact]
    public void WeightUpdate_ImprovesPrediction_AfterOneStep()
    {
        var net   = TwoLayerLinear(0.5f, stateLR: 0f, weightLR: 0.5f);
        var input = Vec(4f);

        // Fix μ_0 and μ_1 to known values so errBefore is fully determined
        net.Layers[0].Mu[0] = 4f;
        net.Layers[0].IsClamped = true;
        net.Layers[1].Mu[0] = 2f;

        // Compute baseline prediction error
        net.Layers[1].ComputePrediction();
        net.Layers[0].SetPredictionError(net.Layers[1].Prediction);
        float errBefore = MathF.Abs(net.Layers[0].PredictionError[0]);

        // Single learning step: inference (to refresh errors) + weight update
        net.Learn(input, Vec(0f), inferenceSteps: 1);

        net.Layers[1].ComputePrediction();
        net.Layers[0].SetPredictionError(net.Layers[1].Prediction);
        float errAfter = MathF.Abs(net.Layers[0].PredictionError[0]);

        Assert.True(errAfter < errBefore,
            $"Prediction error should decrease: before={errBefore:F4}, after={errAfter:F4}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2.4  Learning minimises free energy over time
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz section 2.4: iterating the joint inference + weight-update step
    /// minimises the free energy F(μ, θ) jointly over states and parameters.
    ///
    /// After many Learn() calls the post-inference free energy must be
    /// significantly lower than after the first epoch.
    /// </summary>
    [Fact]
    public void Learning_FreeEnergy_DecreasesOverEpochs()
    {
        var net   = new Network(
            new NetworkConfig { LayerSizes = [3, 4, 2], Activation = ActivationType.Tanh, StateLearningRate = 0.05f, WeightLearningRate = 0.01f },
            new Random(99));
        var input = Vec(1f, 0f, -1f);
        var label = Vec(1f, 0f);

        // First epoch energy
        net.Learn(input, label, inferenceSteps: 10);
        float energyEpoch1 = net.ComputeEnergy();

        // Train for more epochs
        for (int e = 1; e < 100; e++)
            net.Learn(input, label, inferenceSteps: 10);

        float energyFinal = net.ComputeEnergy();

        Assert.True(energyFinal < energyEpoch1 * 0.5f,
            $"Energy after 100 epochs ({energyFinal:F4}) should be " +
            $"< 50 % of epoch-1 energy ({energyEpoch1:F4}).");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 4 / W&B eq. 14  Hierarchical inference
    // dμ_l/dt = W_l^T f'(pre_l) ε_{l−1} − ε_l
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// W&B (2017) eq. 14 / Bogacz section 4: for a linear, L-layer network
    /// at the fixed point, the prediction at every layer must equal the state
    /// of the layer below (zero sensory error cascades upward).
    ///
    /// We verify that after convergence, pred_l ≈ μ_{l-1} for each l ≥ 1.
    /// </summary>
    [Fact]
    public void HierarchicalInference_Converges_PredictionsMatchStates()
    {
        // Use a [2,2,2] network with identity weights so the fixed point is
        // analytically tractable.  With W_1 = W_2 = I and b = 0:
        //   dμ_1/dt = (μ_0 − μ_1) − (μ_1 − μ_2) = 0  →  μ_1* = μ_0
        //   dμ_2/dt = (μ_1 − μ_2) = 0                 →  μ_2* = μ_0
        // so pred_1 = W_1 μ_1* = μ_0 exactly.
        var net = new Network(
            new NetworkConfig { LayerSizes = [2, 2, 2], Activation = ActivationType.Linear, StateLearningRate = 0.1f, WeightLearningRate = 0f },
            new Random(5));

        // Set both weight matrices to identity
        net.Layers[1].GenerativeWeights![0, 0] = 1f; net.Layers[1].GenerativeWeights![0, 1] = 0f;
        net.Layers[1].GenerativeWeights![1, 0] = 0f; net.Layers[1].GenerativeWeights![1, 1] = 1f;
        net.Layers[1].Bias!.Zero();
        net.Layers[2].GenerativeWeights![0, 0] = 1f; net.Layers[2].GenerativeWeights![0, 1] = 0f;
        net.Layers[2].GenerativeWeights![1, 0] = 0f; net.Layers[2].GenerativeWeights![1, 1] = 1f;
        net.Layers[2].Bias!.Zero();

        var input = Vec(1f, -1f);

        // Eigenvalue analysis: convergence factor ≈ max(|1-0.038η|, |1-0.262η|) with η=0.1
        // → ~0.974^300 ≈ 3e-4, well within tolerance 0.01
        net.Infer(input, inferenceSteps: 300);

        // Recompute predictions fresh
        for (int l = net.LayerCount - 1; l >= 1; l--)
            net.Layers[l].ComputePrediction();

        // pred_1 (generated by layer 1) should ≈ μ_0
        for (int i = 0; i < net.Layers[0].Size; i++)
        {
            float pred  = net.Layers[1].Prediction![i];
            float state = net.Layers[0].Mu[i];
            Assert.True(MathF.Abs(pred - state) < 0.01f,
                $"Layer-0 mismatch at [{i}]: pred={pred:F4}, μ_0={state:F4}");
        }
    }

    /// <summary>
    /// Bogacz section 4: in a hierarchical network, the free energy is the
    /// sum of local energies over all layers except the top (eq. 12 extended).
    ///
    /// Verify: ComputeEnergy() = Σ_{l=0}^{L-2} LocalEnergy(l)
    /// </summary>
    [Fact]
    public void HierarchicalFreeEnergy_EqualsSumOfLayerLocalEnergies()
    {
        var net   = new Network(
            new NetworkConfig { LayerSizes = [2, 3, 2], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(13));
        var input = Vec(0.5f, -0.5f);

        net.Infer(input, inferenceSteps: 10);

        float sumLocal = 0f;
        for (int l = 0; l < net.LayerCount - 1; l++)
            sumLocal += net.Layers[l].LocalEnergy();

        Assert.Equal(sumLocal, net.ComputeEnergy(), precision: 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 4  Top-down prediction suppresses sensory error
    // (predictive coding core: "explaining away" Rao & Ballard 1999)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz section 4 / Rao & Ballard (1999): the top-down prediction
    /// g(μ_{l+1}) actively reduces ε_l.  Without a top-down prediction
    /// (i.e. with a random / uninformed higher layer), ε_0 is large;
    /// after sufficient inference it shrinks as μ_1 adjusts to explain x.
    ///
    /// We verify that inference strictly reduces || ε_0 || compared to
    /// the raw residual before any inference.
    /// </summary>
    [Fact]
    public void TopDownPrediction_ReducesSensoryPredictionError()
    {
        var net   = new Network(
            new NetworkConfig { LayerSizes = [3, 4, 2], Activation = ActivationType.Tanh, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(17));
        var input = Vec(1f, 0f, -1f);

        // Measure || ε_0 || without any inference (just clamp + compute errors)
        net.Layers[0].Mu[0] = input[0];
        net.Layers[0].Mu[1] = input[1];
        net.Layers[0].Mu[2] = input[2];
        net.Layers[0].IsClamped = true;
        for (int l = net.LayerCount - 1; l >= 1; l--)
            net.Layers[l].ComputePrediction();
        net.Layers[0].SetPredictionError(net.Layers[1].Prediction);
        float errBefore = MathF.Sqrt(TensorUtils.SumSquares(net.Layers[0].PredictionError));

        // Run inference (state updates only, weights frozen)
        net.Infer(input, inferenceSteps: 200);
        for (int l = net.LayerCount - 1; l >= 1; l--)
            net.Layers[l].ComputePrediction();
        net.Layers[0].SetPredictionError(net.Layers[1].Prediction);
        float errAfter = MathF.Sqrt(TensorUtils.SumSquares(net.Layers[0].PredictionError));

        Assert.True(errAfter < errBefore,
            $"Inference should reduce || ε_0 ||: before={errBefore:F4}, after={errAfter:F4}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Complementary: bias update follows the same Hebbian rule (Bogacz eq. 7)
    // Δb = η · ε_{l−1}   (bias acts like a weight with constant pre-synaptic 1)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz eq. 7: bias update Δb = η · ε_x (same signal as ΔW, with
    /// implicit presynaptic activity = 1).
    ///
    /// We verify that the bias shift after one update step equals
    /// η · lowerLayerError exactly.
    /// </summary>
    [Fact]
    public void BiasUpdate_ProportionalToPredictionError()
    {
        const float eta = 0.01f;

        var layer = new Layer
        {
            Index              = 1,
            Size               = 2,
            LowerSize          = 2,
            ActivationType     = ActivationType.Linear,
            StateLearningRate  = 0.05f,
            WeightLearningRate = eta,
        };
        layer.Initialize(new Random(0));
        layer.Mu[0] = 1f; layer.Mu[1] = 1f;

        var lowerError   = Vec(3f, -1f);
        float[] bBefore  = (float[])layer.Bias!.Buffer.Clone();

        layer.UpdateWeights(lowerError, eta);

        // Δb[i] = eta * ε_0[i]  (f'=1 for Linear)
        Assert.Equal(eta * 3f,  layer.Bias!.Buffer[0] - bBefore[0], precision: 5);
        Assert.Equal(eta * -1f, layer.Bias!.Buffer[1] - bBefore[1], precision: 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2.3  Energy landscape: lower energy ↔ better fit
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Bogacz section 2.3: the free energy is proportional to the negative
    /// log-likelihood under a Gaussian generative model.  A state that
    /// perfectly explains the input (ε = 0 everywhere) achieves F = 0,
    /// the global minimum.
    ///
    /// We construct such a perfect state and verify ComputeEnergy() = 0.
    /// </summary>
    [Fact]
    public void FreeEnergy_IsZero_WhenAllPredictionErrorsAreZero()
    {
        var net = new Network(
            new NetworkConfig { LayerSizes = [2, 3, 2], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(0));

        // Force all prediction errors to zero
        foreach (var layer in net.Layers)
            layer.PredictionError.Zero();

        Assert.Equal(0f, net.ComputeEnergy(), precision: 5);
    }

    /// <summary>
    /// Bogacz section 2.3: the free energy is strictly positive when any
    /// prediction error is non-zero (it equals half the sum of squares).
    /// </summary>
    [Fact]
    public void FreeEnergy_IsPositive_WhenPredictionErrorIsNonZero()
    {
        var net = new Network(
            new NetworkConfig { LayerSizes = [2, 3], Activation = ActivationType.Linear, StateLearningRate = 0.05f, WeightLearningRate = 0f },
            new Random(0));

        net.Layers[0].PredictionError[0] = 1f; // any non-zero value

        Assert.True(net.ComputeEnergy() > 0f);
    }
}
