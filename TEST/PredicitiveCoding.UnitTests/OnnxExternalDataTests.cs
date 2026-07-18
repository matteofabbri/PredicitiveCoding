using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PredictiveCoding.Onnx;
using OnnxTensor = PredictiveCoding.Onnx.Tensor;

namespace PredictiveCoding.Onnx.Tests;

/// <summary>
/// External-data support: tensors whose payload lives in a side file
/// (data_location=EXTERNAL, the format HuggingFace uses for &gt;2 GB models),
/// plus the arctic-embed integration parity when that model is on disk.
/// </summary>
public class OnnxExternalDataTests
{
    // ── Synthetic external-data model (portable, no downloads) ───────────────

    [Fact]
    public void Load_ExternalDataTensor_ReadsFromSideFile()
    {
        // Graph: y = Add(x, w) with w [2,2] stored externally at offset 8.
        string dir = Path.Combine(Path.GetTempPath(), $"pc-onnx-ext-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            float[] weights = [10f, 20f, 30f, 40f];
            byte[] side = new byte[8 + 16];                      // 8 junk bytes, then the payload
            Buffer.BlockCopy(weights, 0, side, 8, 16);
            File.WriteAllBytes(Path.Combine(dir, "w.bin"), side);

            string modelPath = Path.Combine(dir, "model.onnx");
            File.WriteAllBytes(modelPath, BuildExternalModel("w.bin", offset: 8, length: 16));

            var model = OnnxModel.Load(modelPath);

            var y = model.Run(new() { ["x"] = new OnnxTensor([2, 2], [1f, 2f, 3f, 4f]) })["y"];
            Assert.Equal(new[] { 11f, 22f, 33f, 44f }, y.Data);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Load_ExternalDataFromBareBytes_ThrowsClearly()
    {
        // Parsing bytes with no directory context cannot resolve the side file.
        byte[] model = BuildExternalModel("w.bin", offset: 0, length: 16);
        var ex = Assert.ThrowsAny<Exception>(() => OnnxModel.Load(model));
        Assert.Contains("external", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Load_ExternalDataMissingSideFile_ThrowsClearly()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"pc-onnx-ext-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string modelPath = Path.Combine(dir, "model.onnx");
            File.WriteAllBytes(modelPath, BuildExternalModel("missing.bin", offset: 0, length: 16));
            var ex = Assert.ThrowsAny<Exception>(() => OnnxModel.Load(modelPath));
            Assert.Contains("missing.bin", ex.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    // ── Arctic-embed integration (skips when the Sperone model is absent) ────

    private const string ArcticPath =
        @"D:\REPOSITORY\SPERONE\Sperone.Setup\bin\Debug\net10.0\data\models\arctic-embed-l-v2.0\model.onnx";

    [Fact]
    [Trait("Category", "Integration")]
    public void ArcticEmbed_InHouse_MatchesOnnxRuntime()
    {
        if (!File.Exists(ArcticPath)) return;   // model lives in the Sperone checkout on this machine

        const int seqLen = 8;
        long[] ids = [0, 4322, 51418, 8999, 120, 77, 6, 2];   // <s> … </s>, no pad
        long[] mask = [1, 1, 1, 1, 1, 1, 1, 1];

        var mine = OnnxModel.Load(ArcticPath).Run(new()
        {
            ["input_ids"] = new OnnxTensor([1, seqLen], ids),
            ["attention_mask"] = new OnnxTensor([1, seqLen], mask),
        })["sentence_embedding"];

        using var session = new InferenceSession(ArcticPath);
        using var ortOut = session.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(ids, [1, seqLen])),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(mask, [1, seqLen])),
        ]);
        float[] theirs = ortOut.First(o => o.Name == "sentence_embedding").AsEnumerable<float>().ToArray();

        double max = 0, dot = 0, na = 0, nb = 0;
        for (int i = 0; i < theirs.Length; i++)
        {
            max = Math.Max(max, Math.Abs(theirs[i] - mine.Data[i]));
            dot += (double)theirs[i] * mine.Data[i];
            na += (double)theirs[i] * theirs[i];
            nb += (double)mine.Data[i] * mine.Data[i];
        }
        double cos = dot / Math.Max(Math.Sqrt(na) * Math.Sqrt(nb), 1e-12);
        Assert.True(max < 1e-3 && cos > 0.99999,
            $"sentence_embedding: maxAbsDiff={max:e3}, cosine={cos:F7}");
    }

    // ── Minimal protobuf writer for the synthetic model ──────────────────────
    //
    // ModelProto{ graph=7{ node=1{Add}, initializer=5{w, EXTERNAL}, input=11{x}, output=12{y} }, opset=8 }
    private static byte[] BuildExternalModel(string location, long offset, long length)
    {
        byte[] tensor = Concat(
            PackedVarints(1, [2, 2]),                            // dims [2,2]
            Varint(2, 1),                                        // data_type FLOAT
            LengthDelimited(8, "w"u8.ToArray()),                 // name
            LengthDelimited(13, StringEntry("location", location)),
            LengthDelimited(13, StringEntry("offset", offset.ToString())),
            LengthDelimited(13, StringEntry("length", length.ToString())),
            Varint(14, 1));                                      // data_location EXTERNAL

        byte[] node = Concat(
            LengthDelimited(1, "x"u8.ToArray()), LengthDelimited(1, "w"u8.ToArray()),
            LengthDelimited(2, "y"u8.ToArray()),
            LengthDelimited(4, "Add"u8.ToArray()));

        byte[] graph = Concat(
            LengthDelimited(1, node),
            LengthDelimited(5, tensor),
            LengthDelimited(11, ValueInfo("x")),
            LengthDelimited(12, ValueInfo("y")));

        byte[] opset = Varint(2, 14);                            // version 14, default domain
        return Concat(LengthDelimited(7, graph), LengthDelimited(8, opset));
    }

    private static byte[] StringEntry(string key, string value) => Concat(
        LengthDelimited(1, System.Text.Encoding.UTF8.GetBytes(key)),
        LengthDelimited(2, System.Text.Encoding.UTF8.GetBytes(value)));

    private static byte[] ValueInfo(string name)
        => LengthDelimited(1, System.Text.Encoding.UTF8.GetBytes(name));

    private static byte[] Varint(int field, long value)
        => [.. RawVarint((ulong)(field << 3)), .. RawVarint((ulong)value)];

    private static byte[] LengthDelimited(int field, byte[] payload)
        => [.. RawVarint((ulong)(field << 3 | 2)), .. RawVarint((ulong)payload.Length), .. payload];

    private static byte[] PackedVarints(int field, long[] values)
        => LengthDelimited(field, values.SelectMany(v => RawVarint((ulong)v)).ToArray());

    private static byte[] RawVarint(ulong v)
    {
        var bytes = new List<byte>();
        do
        {
            byte b = (byte)(v & 0x7F);
            v >>= 7;
            if (v != 0) b |= 0x80;
            bytes.Add(b);
        } while (v != 0);
        return [.. bytes];
    }

    private static byte[] Concat(params byte[][] parts)
        => parts.SelectMany(p => p).ToArray();
}
