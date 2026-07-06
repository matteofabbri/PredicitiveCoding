using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx.Tests;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxYoloxParityTests — the in-house ONNX executor vs Microsoft.ML.OnnxRuntime
// on the real YOLOX detection model (~36 MB, downloaded once by the
// people-counting demo). One deterministic pseudo-frame goes through both
// runtimes; every graph output must agree within float accumulation-order
// tolerance. This is the "does the whole runtime actually work" test.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Numerical parity of <see cref="OnnxModel"/> against ONNX Runtime on
/// yolox_2022nov.onnx. Skipped silently when the model is not on this machine.</summary>
public class OnnxYoloxParityTests
{
    // May take tens of seconds on CPU — acceptable for an Integration-tagged test.
    [Fact]
    [Trait("Category", "Integration")]
    public void InHouseExecutor_MatchesOnnxRuntime_OnYolox()
    {
        // Downloaded by YoloPersonDetector on first use (people-counting demo).
        string path = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".predictivecoding", "vision-models", "yolox_2022nov.onnx");

        // Model not downloaded on this machine → return silently, the soft-skip
        // pattern this repo uses for tests depending on downloaded artefacts.
        if (!File.Exists(path)) return;

        var model = OnnxModel.Load(path);
        OnnxValueInfo input = Assert.Single(model.Graph.Inputs);   // YOLOX has one NCHW input

        // Declared input shape, substituting 1 for any dynamic (-1) dim;
        // for yolox_2022nov this is the static [1,3,640,640].
        int[] shape = input.Shape.Select(d => d < 0 ? 1 : d).ToArray();

        // Deterministic pseudo-frame mirroring YoloPersonDetector's preprocessing:
        // raw pixel values in 0..255, no mean/std normalisation, NCHW layout.
        var rng = new Random(42);
        float[] data = new float[Tensor.CheckedProduct(shape)];
        for (int i = 0; i < data.Length; i++) data[i] = rng.Next(0, 256);

        // Run 1 — the in-house parser + executor.
        Dictionary<string, Tensor> mine = model.Run(new() { [input.Name] = new Tensor(shape, data) });

        // Run 2 — ONNX Runtime, default CPU session, byte-identical input.
        using var options = new SessionOptions();
        using var session = new InferenceSession(path, options);
        var dense = new DenseTensor<float>(data, shape);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> ortOutputs =
            session.Run([NamedOnnxValue.CreateFromTensor(input.Name, dense)]);
        Dictionary<string, Tensor<float>> ortByName =
            ortOutputs.ToDictionary(v => v.Name, v => v.AsTensor<float>());

        // Both runtimes must produce the same output set …
        Assert.Equal(model.OutputNames.Count, ortOutputs.Count);

        // … and every output must match in shape and (within tolerance) in value.
        foreach (string name in model.OutputNames)
        {
            Assert.True(ortByName.TryGetValue(name, out var ort),
                $"ONNX Runtime produced no output named '{name}'.");

            Tensor ours = mine[name];
            Assert.Equal(ort!.Dimensions.ToArray(), ours.Shape);

            float[] expected = ort.ToArray();
            float[] actual = ours.Data;
            Assert.Equal(expected.Length, actual.Length);

            double maxAbs = 0, sumAbs = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                double d = Math.Abs((double)actual[i] - expected[i]);
                if (d > maxAbs) maxAbs = d;
                sumAbs += d;
            }
            double meanAbs = actual.Length == 0 ? 0 : sumAbs / actual.Length;

            // Kernels accumulate sums in a different order than ORT, so exact
            // equality is impossible — hence the max/mean absolute tolerances.
            Assert.True(maxAbs < 1e-2 && meanAbs < 1e-3,
                $"Output '{name}' diverges from ONNX Runtime: " +
                $"maxAbsDiff={maxAbs:E4} (limit 1e-2), meanAbsDiff={meanAbs:E4} (limit 1e-3).");
        }
    }
}
