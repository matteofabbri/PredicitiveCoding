using PredictiveCoding.Onnx;
using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx.Tests;

/// <summary>Regression tests for the defects confirmed by the adversarial review:
/// MaxPool ceil_mode right-padding windows, auto_pad SAME resolution, lazy
/// initializer materialisation, and the loud guards for unsupported attribute forms.</summary>
public class OnnxReviewFixTests
{
    // ── MaxPool ceil_mode: windows starting in the right padding are dropped ──

    [Fact]
    public void MaxPool_CeilMode_DropsWindowsStartingInRightPadding()
    {
        // PyTorch nn.MaxPool2d(2, 2, padding=1, ceil_mode=True) on 3×3:
        // ceil((3+2-2)/2)+1 = 3, but the third window would START at row 3 ≥ size+padLo=4? No:
        // start = 2·2 − 1 = 3 ≥ 3+1 = 4 is false… actual rule: (outDim−1)·stride ≥ size+padLo
        // → 4 ≥ 4 → drop → 2×2 (ORT/PyTorch agree). Old code compared against size+padLo+padHi=5
        // and kept a third row of −infinity.
        var x = new Tensor([1, 1, 3, 3], [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f]);

        var y = Ops.MaxPool(x, kernel: [2, 2], strides: [2, 2], pads: [1, 1, 1, 1], ceilMode: true);

        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape);
        Assert.Equal(new[] { 1f, 3f, 7f, 9f }, y.Data);
        Assert.DoesNotContain(float.NegativeInfinity, y.Data);
    }

    // ── auto_pad: SAME_UPPER conv keeps the spatial size and pads correctly ──

    [Fact]
    public void Conv_AutoPadSameUpper_PreservesSpatialSize()
    {
        // 3×3 all-ones kernel, stride 1, SAME → output 4×4; on an all-ones input each
        // output pixel counts its in-bounds neighbours: corners 4, edges 6, centre 9.
        var graph = new OnnxGraph
        {
            OpsetVersion = 11,
            Nodes =
            [
                new OnnxNode
                {
                    Name = "conv", OpType = "Conv",
                    Inputs = ["x", "w"], Outputs = ["y"],
                    Attributes = new()
                    {
                        ["auto_pad"] = new OnnxAttribute { Name = "auto_pad", Type = OnnxAttributeType.String, S = "SAME_UPPER" },
                        ["strides"] = new OnnxAttribute { Name = "strides", Type = OnnxAttributeType.Ints, Ints = [1, 1] },
                    },
                },
            ],
            Initializers =
            {
                ["w"] = new OnnxTensorData { Name = "w", DataType = OnnxDataType.Float, Dims = [1, 1, 3, 3], Floats = new float[9].Select(_ => 1f).ToArray() },
            },
            Inputs = [new OnnxValueInfo { Name = "x", Shape = [1, 1, 4, 4] }],
            Outputs = [new OnnxValueInfo { Name = "y" }],
        };
        var x = new Tensor([1, 1, 4, 4], new float[16].Select(_ => 1f).ToArray());

        var y = Executor.Run(graph, new() { ["x"] = x })["y"];

        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Shape);
        float[] expected =
        {
            4, 6, 6, 4,
            6, 9, 9, 6,
            6, 9, 9, 6,
            4, 6, 6, 4,
        };
        Assert.Equal(expected, y.Data);
    }

    [Fact]
    public void Conv_AutoPadUnknown_ThrowsLoudly()
    {
        var graph = SingleNodeGraph("Conv", ["x", "w"],
            new() { ["auto_pad"] = new OnnxAttribute { Name = "auto_pad", Type = OnnxAttributeType.String, S = "SAME_SIDEWAYS" } });
        graph.Initializers["w"] = new OnnxTensorData { Name = "w", DataType = OnnxDataType.Float, Dims = [1, 1, 1, 1], Floats = [1f] };

        var ex = Assert.Throws<NotSupportedException>(
            () => Executor.Run(graph, new() { ["x"] = new Tensor([1, 1, 2, 2], [1f, 2f, 3f, 4f]) }));
        Assert.Contains("auto_pad", ex.Message);
    }

    // ── Lazy initializers: unused unsupported-dtype weights must not block Run ──

    [Fact]
    public void Run_UnusedUnsupportedInitializer_DoesNotBlockExecution()
    {
        var graph = SingleNodeGraph("Relu", ["x"]);
        // A double (data_type 11) leftover no node consumes: Floats/Longs both null.
        graph.Initializers["dead_weight"] = new OnnxTensorData { Name = "dead_weight", DataType = 11, Dims = [2] };

        var y = Executor.Run(graph, new() { ["x"] = new Tensor([2], [-1f, 2f]) })["y"];

        Assert.Equal(new[] { 0f, 2f }, y.Data);
    }

    [Fact]
    public void Run_UsedUnsupportedInitializer_StillThrowsClearly()
    {
        var graph = SingleNodeGraph("Add", ["x", "bad"]);
        graph.Initializers["bad"] = new OnnxTensorData { Name = "bad", DataType = 11, Dims = [2] };

        var ex = Assert.Throws<NotSupportedException>(
            () => Executor.Run(graph, new() { ["x"] = new Tensor([2], [1f, 2f]) }));
        Assert.Contains("bad", ex.Message);
    }

    [Fact]
    public void Run_GraphOutputAliasingInitializer_IsReturned()
    {
        // Degenerate but legal: a graph output that IS an initializer (no node produces it).
        var graph = new OnnxGraph
        {
            Initializers = { ["c"] = new OnnxTensorData { Name = "c", DataType = OnnxDataType.Float, Dims = [2], Floats = [7f, 8f] } },
            Outputs = [new OnnxValueInfo { Name = "c" }],
        };

        var y = Executor.Run(graph, new())["c"];

        Assert.Equal(new[] { 7f, 8f }, y.Data);
    }

    // ── Loud guards for attribute forms the kernels would otherwise mis-run ──

    [Fact]
    public void Reshape_AllowZero_ThrowsLoudly()
    {
        var graph = SingleNodeGraph("Reshape", ["x", "shape"],
            new() { ["allowzero"] = new OnnxAttribute { Name = "allowzero", Type = OnnxAttributeType.Int, I = 1 } });
        graph.Initializers["shape"] = new OnnxTensorData { Name = "shape", DataType = OnnxDataType.Int64, Dims = [2], Longs = [0, 4] };

        Assert.Throws<NotSupportedException>(
            () => Executor.Run(graph, new() { ["x"] = new Tensor([2, 2], [1f, 2f, 3f, 4f]) }));
    }

    [Fact]
    public void MaxPool_Dilations_ThrowsLoudly()
    {
        var graph = SingleNodeGraph("MaxPool", ["x"],
            new()
            {
                ["kernel_shape"] = new OnnxAttribute { Name = "kernel_shape", Type = OnnxAttributeType.Ints, Ints = [2, 2] },
                ["dilations"] = new OnnxAttribute { Name = "dilations", Type = OnnxAttributeType.Ints, Ints = [2, 2] },
            });

        Assert.Throws<NotSupportedException>(
            () => Executor.Run(graph, new() { ["x"] = new Tensor([1, 1, 4, 4], new float[16]) }));
    }

    [Fact]
    public void Softmax_PreOpset13_NonLastAxis_ThrowsLoudly()
    {
        var graph = SingleNodeGraph("Softmax", ["x"],
            new() { ["axis"] = new OnnxAttribute { Name = "axis", Type = OnnxAttributeType.Int, I = 0 } });
        graph.OpsetVersion = 11;

        Assert.Throws<NotSupportedException>(
            () => Executor.Run(graph, new() { ["x"] = new Tensor([2, 3], new float[6]) }));
    }

    [Fact]
    public void Softmax_PreOpset13_LastAxis_StillRuns()
    {
        // axis defaults to 1 pre-opset-13, which IS the last dim of a rank-2 tensor:
        // flatten-at-axis and single-axis semantics coincide → must keep working.
        var graph = SingleNodeGraph("Softmax", ["x"]);
        graph.OpsetVersion = 11;

        var y = Executor.Run(graph, new() { ["x"] = new Tensor([1, 2], [0f, 0f]) })["y"];

        Assert.Equal(0.5f, y.Data[0], 5);
        Assert.Equal(0.5f, y.Data[1], 5);
    }

    private static OnnxGraph SingleNodeGraph(string opType, string[] inputs,
                                             Dictionary<string, OnnxAttribute>? attrs = null) => new()
    {
        Nodes =
        [
            new OnnxNode { Name = "n0", OpType = opType, Inputs = inputs, Outputs = ["y"], Attributes = attrs ?? [] },
        ],
        Inputs = [new OnnxValueInfo { Name = "x" }],
        Outputs = [new OnnxValueInfo { Name = "y" }],
    };
}
