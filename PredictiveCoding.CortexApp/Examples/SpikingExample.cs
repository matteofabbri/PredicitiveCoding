using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Spiking;

namespace PredictiveCoding.CortexApp.Examples;

// Trains a spiking predictive-coding network on synthetic class prototypes and
// reports test accuracy and average free energy.
public static class SpikingExample
{
    private const int InputDim = 16;
    private const int Classes  = 3;

    public sealed record Result(float Accuracy, float AverageEnergy, int TrainEpochs, int Ticks);

    public static Result Run(int epochs, int ticks)
    {
        var rng = new Random(99);
        var net = new SpikingNetwork(
            new SpikingNetworkConfig { LayerSizes = [InputDim, 24, Classes], Ticks = ticks, WeightLearningRate = 0.02f },
            rng);

        // Fixed random class prototypes in [-1, 1].
        var protos = new DenseTensor<float>[Classes];
        for (int c = 0; c < Classes; c++)
        {
            protos[c] = new DenseTensor<float>(InputDim);
            for (int i = 0; i < InputDim; i++) protos[c].Buffer[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        for (int e = 0; e < epochs; e++)
            for (int c = 0; c < Classes; c++)
                net.Learn(Noisy(protos[c], 0.15f, rng), OneHot(c));

        int correct = 0, total = 0;
        float energy = 0f;
        for (int c = 0; c < Classes; c++)
            for (int s = 0; s < 12; s++)
            {
                var x = Noisy(protos[c], 0.2f, rng);
                if (net.Predict(x) == c) correct++;
                energy += net.InferEnergy(x);
                total++;
            }

        return new Result(correct / (float)total, energy / total, epochs, ticks);
    }

    private static DenseTensor<float> Noisy(DenseTensor<float> src, float sigma, Random rng)
    {
        var t = src.Clone();
        for (int i = 0; i < t.Length; i++) t.Buffer[i] += RandomUtils.SampleNormal(0f, sigma, rng);
        return t;
    }

    private static DenseTensor<float> OneHot(int c)
    {
        var t = DenseTensor<float>.Zeros(Classes);
        t.Buffer[c] = 1f;
        return t;
    }
}
