using PredictiveCoding.Onnx;
using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx.Tests;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxTransformerOpsTests — the transformer-era kernels added for
// arctic-embed-l-v2.0 (XLM-RoBERTa, opset 14): Pow/Erf/Clip, int64 elementwise
// arithmetic, Equal/Not/Where/Expand/Cast, ReduceMean/ReduceL2/CumSum, and an
// Executor mini-graph reproducing the XLM-R position-id pattern.
//
// Expectations are hand-computed from the ONNX spec, independent of the kernel
// implementations (which are written separately) — the calls compile against
// the exact signatures pinned by Executor.Dispatch. Bool tensors follow the
// runtime convention: int64 payload holding 0/1, no third payload kind.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Spec-derived tests for the ops needed by the XLM-RoBERTa text embedder:
/// LayerNorm decomposition (ReduceMean/Sub/Pow/Sqrt), GELU-Erf, attention-mask
/// logic (Equal/Not/Cast/Where/Expand) and the pad-aware CumSum position ids.</summary>
public class OnnxTransformerOpsTests
{
    // ── Pow ──────────────────────────────────────────────────────────────────

    [Fact]
    public void Pow_ScalarExponent_Squares()
    {
        var x = new Tensor([2, 3], [1f, 2f, 3f, 4f, 5f, 6f]);
        var e = new Tensor([1], [2f]);

        var y = Ops.Pow(x, e);

        Assert.Equal(new[] { 2, 3 }, y.Shape);
        AssertClose([1f, 4f, 9f, 16f, 25f, 36f], y);
    }

    [Fact]
    public void Pow_Broadcast_2x1_Pow_3()
    {
        // base [2,1] = {2,3}, exp [3] = {1,2,3} → [2,3]:
        // row 0: 2^1=2, 2^2=4, 2^3=8;  row 1: 3^1=3, 3^2=9, 3^3=27.
        var b = new Tensor([2, 1], [2f, 3f]);
        var e = new Tensor([3], [1f, 2f, 3f]);

        var y = Ops.Pow(b, e);

        Assert.Equal(new[] { 2, 3 }, y.Shape);
        AssertClose([2f, 4f, 8f, 3f, 9f, 27f], y);
    }

    [Fact]
    public void Pow_Int64Base_Throws()
    {
        var b = new Tensor([2], [2L, 3L]);
        var e = new Tensor([1], [2f]);

        AssertThrowsGuard(() => Ops.Pow(b, e));
    }

    // ── Erf ──────────────────────────────────────────────────────────────────

    [Fact]
    public void Erf_KnownValues()
    {
        // Reference values (Abramowitz & Stegun / scipy.special.erf):
        // erf(0)=0, erf(±1)=±0.8427007929, erf(0.5)=0.5204998778, erf(±3)=±0.9999779095.
        var x = new Tensor([6], [0f, 1f, -1f, 0.5f, 3f, -3f]);

        var y = Ops.Erf(x);

        AssertClose(
            [0f, 0.8427007929f, -0.8427007929f, 0.5204998778f, 0.9999779095f, -0.9999779095f],
            y, tol: 1e-6f);
    }

    [Fact]
    public void Erf_Saturates_At_PlusMinusTen()
    {
        // erf(10) = 1 − 2.1e-45, which rounds to exactly 1 in float32.
        var y = Ops.Erf(new Tensor([2], [10f, -10f]));

        Assert.Equal(1f, y.Data[0]);
        Assert.Equal(-1f, y.Data[1]);
    }

    [Fact]
    public void Erf_IsOddFunction_BitExact()
    {
        // erf(−x) must equal −erf(x) bit-for-bit, or GELU(x)+GELU(−x) ≠ x drifts.
        float[] vals = [0.25f, 0.75f, 1.5f, 2.5f];
        var pos = Ops.Erf(new Tensor([4], vals));
        var neg = Ops.Erf(new Tensor([4], [-0.25f, -0.75f, -1.5f, -2.5f]));

        for (int i = 0; i < vals.Length; i++)
            Assert.Equal(-pos.Data[i], neg.Data[i]);
    }

    // ── Clip ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Clip_MinAndMaxTensors()
    {
        var x = new Tensor([5], [-2f, -1f, 0f, 1f, 2f]);
        var min = new Tensor([], [-1f]);             // rank-0 scalar per the spec
        var max = new Tensor([], [1f]);

        var y = Ops.Clip(x, min, max, float.NegativeInfinity, float.PositiveInfinity);

        AssertClose([-1f, -1f, 0f, 1f, 1f], y);
    }

    [Fact]
    public void Clip_OnlyMinTensor()
    {
        var x = new Tensor([5], [-2f, -1f, 0f, 1f, 2f]);

        var y = Ops.Clip(x, new Tensor([], [0f]), null, float.NegativeInfinity, float.PositiveInfinity);

        AssertClose([0f, 0f, 0f, 1f, 2f], y);
    }

    [Fact]
    public void Clip_OnlyMaxTensor()
    {
        var x = new Tensor([5], [-2f, -1f, 0f, 1f, 2f]);

        var y = Ops.Clip(x, null, new Tensor([], [0f]), float.NegativeInfinity, float.PositiveInfinity);

        AssertClose([-2f, -1f, 0f, 0f, 0f], y);
    }

    [Fact]
    public void Clip_AttributeFallback_WhenTensorSlotsNull()
    {
        // Opset-6 form: min/max come as attributes, no tensor inputs.
        var x = new Tensor([5], [-2f, -1f, 0f, 1f, 2f]);

        var y = Ops.Clip(x, null, null, -1f, 1f);

        AssertClose([-1f, -1f, 0f, 1f, 1f], y);
    }

    [Fact]
    public void Clip_NoBoundsAnywhere_IsNoOp()
    {
        // Absent attributes surface as ±infinity (Executor's GetFloat defaults) → identity.
        var x = new Tensor([4], [-1e30f, -1f, 1f, 1e30f]);

        var y = Ops.Clip(x, null, null, float.NegativeInfinity, float.PositiveInfinity);

        Assert.Equal(x.Data, y.Data);
    }

    // ── Int64 elementwise arithmetic ─────────────────────────────────────────

    [Fact]
    public void Add_Int64_BroadcastScalar()
    {
        var y = Ops.Add(new Tensor([3], [1L, 2L, 3L]), new Tensor([1], [10L]));

        Assert.True(y.IsInt64);
        Assert.Equal(new[] { 3 }, y.Shape);
        Assert.Equal(new long[] { 11, 12, 13 }, y.I64!);
    }

    [Fact]
    public void Sub_Int64()
    {
        var y = Ops.Sub(new Tensor([3], [5L, 3L, 1L]), new Tensor([3], [1L, 2L, 3L]));

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 4, 1, -2 }, y.I64!);
    }

    [Fact]
    public void Mul_Int64_BroadcastRow()
    {
        // [2,3] * [3]: row 0 {1,2,3}·{2,3,4} = {2,6,12}; row 1 {4,5,6}·{2,3,4} = {8,15,24}.
        var y = Ops.Mul(new Tensor([2, 3], [1L, 2L, 3L, 4L, 5L, 6L]), new Tensor([3], [2L, 3L, 4L]));

        Assert.True(y.IsInt64);
        Assert.Equal(new[] { 2, 3 }, y.Shape);
        Assert.Equal(new long[] { 2, 6, 12, 8, 15, 24 }, y.I64!);
    }

    [Fact]
    public void Div_Int64_TruncatesTowardZero()
    {
        // C# integer division semantics: 7/2 = 3 and −7/2 = −3 (not −4).
        var y = Ops.Div(new Tensor([2], [7L, -7L]), new Tensor([1], [2L]));

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 3, -3 }, y.I64!);
    }

    [Fact]
    public void Add_MixedFloatAndInt64_Throws()
    {
        AssertThrowsGuard(() => Ops.Add(new Tensor([2], [1f, 2f]), new Tensor([2], [1L, 2L])));
    }

    // ── Equal ────────────────────────────────────────────────────────────────

    [Fact]
    public void Equal_Int64_SameShape()
    {
        var a = new Tensor([1, 4], [5L, 99L, 1L, 1L]);
        var b = new Tensor([1, 4], [5L, 1L, 1L, 2L]);

        var y = Ops.Equal(a, b);

        Assert.True(y.IsInt64);                      // bool = int64 0/1 by convention
        Assert.Equal(new[] { 1, 4 }, y.Shape);
        Assert.Equal(new long[] { 1, 0, 1, 0 }, y.I64!);
    }

    [Fact]
    public void Equal_Int64_BroadcastScalar()
    {
        var y = Ops.Equal(new Tensor([2, 3], [1L, 2L, 3L, 2L, 2L, 2L]), new Tensor([], [2L]));

        Assert.True(y.IsInt64);
        Assert.Equal(new[] { 2, 3 }, y.Shape);
        Assert.Equal(new long[] { 0, 1, 0, 1, 1, 1 }, y.I64!);
    }

    [Fact]
    public void Equal_Float_IsExactComparison()
    {
        // 1.0000001f is the neighbouring float of 1f → not equal; −0f == 0f per IEEE 754.
        var a = new Tensor([3], [1f, 1.0000001f, -0f]);
        var b = new Tensor([3], [1f, 1f, 0f]);

        var y = Ops.Equal(a, b);

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 1, 0, 1 }, y.I64!);
    }

    [Fact]
    public void Equal_MixedKinds_Throws()
    {
        AssertThrowsGuard(() => Ops.Equal(new Tensor([2], [1f, 2f]), new Tensor([2], [1L, 2L])));
    }

    // ── Not ──────────────────────────────────────────────────────────────────

    [Fact]
    public void Not_FlipsZeroOne()
    {
        var y = Ops.Not(new Tensor([4], [0L, 1L, 1L, 0L]));

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 1, 0, 0, 1 }, y.I64!);
    }

    [Fact]
    public void Not_FloatInput_Throws()
    {
        AssertThrowsGuard(() => Ops.Not(new Tensor([2], [0f, 1f])));
    }

    // ── Where ────────────────────────────────────────────────────────────────

    [Fact]
    public void Where_ThreeWayBroadcast_Float()
    {
        // cond [2,1] = {1,0}, x [1,3] = {1,2,3}, y [1] = {9} → [2,3]:
        // row 0 (cond 1) takes x → {1,2,3}; row 1 (cond 0) takes y → {9,9,9}.
        var cond = new Tensor([2, 1], [1L, 0L]);
        var x = new Tensor([1, 3], [1f, 2f, 3f]);
        var y = new Tensor([1], [9f]);

        var r = Ops.Where(cond, x, y);

        Assert.Equal(new[] { 2, 3 }, r.Shape);
        Assert.Equal(new[] { 1f, 2f, 3f, 9f, 9f, 9f }, r.Data);
    }

    [Fact]
    public void Where_SameShape_Float()
    {
        var r = Ops.Where(new Tensor([4], [1L, 0L, 1L, 0L]),
                          new Tensor([4], [1f, 2f, 3f, 4f]),
                          new Tensor([4], [-1f, -2f, -3f, -4f]));

        Assert.Equal(new[] { 1f, -2f, 3f, -4f }, r.Data);
    }

    [Fact]
    public void Where_Int64Payload()
    {
        var r = Ops.Where(new Tensor([4], [1L, 0L, 1L, 0L]),
                          new Tensor([4], [10L, 20L, 30L, 40L]),
                          new Tensor([4], [-1L, -2L, -3L, -4L]));

        Assert.True(r.IsInt64);
        Assert.Equal(new long[] { 10, -2, 30, -4 }, r.I64!);
    }

    // ── Expand ───────────────────────────────────────────────────────────────

    [Fact]
    public void Expand_IsBidirectionalBroadcast()
    {
        // [3,1] expanded with shape [2,1,4] → [2,3,4]: the given shape's middle 1
        // is overridden by the input's 3 (Expand broadcasts BOTH ways, unlike Reshape).
        var x = new Tensor([3, 1], [1f, 2f, 3f]);

        var y = Ops.Expand(x, new Tensor([3], [2L, 1L, 4L]));

        Assert.Equal(new[] { 2, 3, 4 }, y.Shape);
        float[] batch = [1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f, 3f, 3f, 3f, 3f];
        Assert.Equal(batch.Concat(batch).ToArray(), y.Data);
    }

    [Fact]
    public void Expand_OneToFive()
    {
        var y = Ops.Expand(new Tensor([1], [7f]), new Tensor([1], [5L]));

        Assert.Equal(new[] { 5 }, y.Shape);
        Assert.Equal(new[] { 7f, 7f, 7f, 7f, 7f }, y.Data);
    }

    [Fact]
    public void Expand_IncompatibleDims_Throws()
    {
        // 3 vs 4: neither is 1 → not broadcastable.
        AssertThrowsGuard(() => Ops.Expand(new Tensor([3], [1f, 2f, 3f]), new Tensor([1], [4L])));
    }

    [Fact]
    public void Expand_Int64PayloadPreserved()
    {
        var y = Ops.Expand(new Tensor([2, 1], [1L, 2L]), new Tensor([2], [2L, 2L]));

        Assert.True(y.IsInt64);
        Assert.Equal(new[] { 2, 2 }, y.Shape);
        Assert.Equal(new long[] { 1, 1, 2, 2 }, y.I64!);
    }

    // ── Cast ─────────────────────────────────────────────────────────────────
    // ONNX TensorProto.DataType codes: 1=float, 6=int32, 7=int64, 9=bool, 11=double.

    [Fact]
    public void Cast_FloatToInt64_TruncatesTowardZero()
    {
        var y = Ops.Cast(new Tensor([4], [2.9f, -2.9f, 0f, 1.5f]), 7);

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 2, -2, 0, 1 }, y.I64!);
    }

    [Fact]
    public void Cast_Int64ToFloat()
    {
        var y = Ops.Cast(new Tensor([3], [3L, -4L, 5L]), 1);

        Assert.False(y.IsInt64);
        Assert.Equal(new[] { 3f, -4f, 5f }, y.Data);
    }

    [Fact]
    public void Cast_FloatToBool_ProducesInt64ZeroOne()
    {
        // bool = int64 0/1 by convention: 0f → 0, any non-zero (incl. negative) → 1.
        var y = Ops.Cast(new Tensor([3], [0f, -3f, 2f]), 9);

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 0, 1, 1 }, y.I64!);
    }

    [Fact]
    public void Cast_Int64ToInt32_StaysInt64Payload()
    {
        // The runtime widens int32 to int64, so this cast keeps the payload kind.
        var y = Ops.Cast(new Tensor([2], [5L, -6L]), 6);

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 5, -6 }, y.I64!);
    }

    [Fact]
    public void Cast_UnsupportedTarget_ThrowsNotSupported()
    {
        Assert.Throws<NotSupportedException>(() => Ops.Cast(new Tensor([1], [1f]), 11));
    }

    // ── ReduceMean ───────────────────────────────────────────────────────────
    // Fixture [2,4] = {1..8}: row means 2.5 and 6.5; grand mean 36/8 = 4.5.

    private static Tensor Mean24() => new([2, 4], [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]);

    [Fact]
    public void ReduceMean_Axis1_KeepDims()
    {
        var y = Ops.ReduceMean(Mean24(), null, [1L], true, false);

        Assert.Equal(new[] { 2, 1 }, y.Shape);
        AssertClose([2.5f, 6.5f], y);
    }

    [Fact]
    public void ReduceMean_Axis1_NoKeepDims()
    {
        var y = Ops.ReduceMean(Mean24(), null, [1L], false, false);

        Assert.Equal(new[] { 2 }, y.Shape);
        AssertClose([2.5f, 6.5f], y);
    }

    [Fact]
    public void ReduceMean_BothAxes_KeepDims()
    {
        var y = Ops.ReduceMean(Mean24(), null, [0L, 1L], true, false);

        Assert.Equal(new[] { 1, 1 }, y.Shape);
        AssertClose([4.5f], y);
    }

    [Fact]
    public void ReduceMean_NoAxes_ReducesAll()
    {
        // Empty axes + noop=false → reduce over ALL dims; keepdims=false → rank-0 scalar.
        var y = Ops.ReduceMean(Mean24(), null, [], false, false);

        Assert.Empty(y.Shape);
        Assert.Single(y.Data);
        AssertClose([4.5f], y);
    }

    [Fact]
    public void ReduceMean_NegativeAxis()
    {
        var y = Ops.ReduceMean(Mean24(), null, [-1L], true, false);

        Assert.Equal(new[] { 2, 1 }, y.Shape);
        AssertClose([2.5f, 6.5f], y);
    }

    [Fact]
    public void ReduceMean_NoopWithEmptyAxes_ReturnsInputUnchanged()
    {
        var x = Mean24();

        var y = Ops.ReduceMean(x, null, [], true, true);

        Assert.Equal(x.Shape, y.Shape);
        Assert.Equal(x.Data, y.Data);
    }

    [Fact]
    public void ReduceMean_AxesAsTensorInput()
    {
        // Opset-18 form: axes arrive as an int64 tensor input instead of the attribute.
        var y = Ops.ReduceMean(Mean24(), new Tensor([1], [1L]), [], true, false);

        Assert.Equal(new[] { 2, 1 }, y.Shape);
        AssertClose([2.5f, 6.5f], y);
    }

    // ── ReduceL2 ─────────────────────────────────────────────────────────────

    [Fact]
    public void ReduceL2_Vector345()
    {
        // sqrt(3² + 4² + 0² + 0²) = 5 — the 3-4-5 triangle.
        var y = Ops.ReduceL2(new Tensor([1, 4], [3f, 4f, 0f, 0f]), null, [1L], true, false);

        Assert.Equal(new[] { 1, 1 }, y.Shape);
        AssertClose([5f], y);
    }

    // ── CumSum ───────────────────────────────────────────────────────────────

    private static Tensor Cum4() => new([4], [1f, 2f, 3f, 4f]);
    private static Tensor Axis0() => new([1], [0L]);

    [Fact]
    public void CumSum_Basic()
    {
        var y = Ops.CumSum(Cum4(), Axis0(), false, false);

        AssertClose([1f, 3f, 6f, 10f], y);
    }

    [Fact]
    public void CumSum_Exclusive()
    {
        // Each element is the sum of everything BEFORE it: {0, 1, 1+2, 1+2+3}.
        var y = Ops.CumSum(Cum4(), Axis0(), true, false);

        AssertClose([0f, 1f, 3f, 6f], y);
    }

    [Fact]
    public void CumSum_Reverse()
    {
        // Suffix sums: {10, 9, 7, 4}.
        var y = Ops.CumSum(Cum4(), Axis0(), false, true);

        AssertClose([10f, 9f, 7f, 4f], y);
    }

    [Fact]
    public void CumSum_ReverseExclusive()
    {
        // Sum of everything AFTER each element: {2+3+4, 3+4, 4, 0}.
        var y = Ops.CumSum(Cum4(), Axis0(), true, true);

        AssertClose([9f, 7f, 4f, 0f], y);
    }

    [Fact]
    public void CumSum_Int64Payload()
    {
        var y = Ops.CumSum(new Tensor([4], [1L, 2L, 3L, 4L]), Axis0(), false, false);

        Assert.True(y.IsInt64);
        Assert.Equal(new long[] { 1, 3, 6, 10 }, y.I64!);
    }

    [Fact]
    public void CumSum_AxisAsScalarTensor()
    {
        // Rank-0 int64 axis (the shape real exports use) must work like [1].
        var y = Ops.CumSum(Cum4(), new Tensor([], [0L]), false, false);

        AssertClose([1f, 3f, 6f, 10f], y);
    }

    [Fact]
    public void CumSum_2D_Axis0()
    {
        // [2,3] {1,2,3 / 4,5,6} down the columns → {1,2,3 / 5,7,9}.
        var y = Ops.CumSum(new Tensor([2, 3], [1f, 2f, 3f, 4f, 5f, 6f]), Axis0(), false, false);

        Assert.Equal(new[] { 2, 3 }, y.Shape);
        AssertClose([1f, 2f, 3f, 5f, 7f, 9f], y);
    }

    [Fact]
    public void CumSum_2D_Axis1()
    {
        // Along the rows → {1,3,6 / 4,9,15}.
        var y = Ops.CumSum(new Tensor([2, 3], [1f, 2f, 3f, 4f, 5f, 6f]), new Tensor([1], [1L]), false, false);

        Assert.Equal(new[] { 2, 3 }, y.Shape);
        AssertClose([1f, 3f, 6f, 4f, 9f, 15f], y);
    }

    // ── Executor mini-graph: the XLM-R position-id pattern ───────────────────

    [Fact]
    public void Executor_XlmRobertaPositionIdPattern()
    {
        // XLM-R computes position ids padding-aware (create_position_ids_from_input_ids):
        //   mask   = Cast(Not(Equal(input_ids, pad)))   → {1,1,0,0}   for ids {5,99,1,1}, pad 1
        //   cumsum = CumSum(mask, axis=1)               → {1,2,2,2}
        //   masked = cumsum * mask                      → {1,2,0,0}
        //   posids = masked + pad                       → {2,3,1,1}   (pads share position pad_id)
        var graph = new OnnxGraph
        {
            OpsetVersion = 14,
            Nodes =
            [
                new OnnxNode { Name = "eq", OpType = "Equal", Inputs = ["input_ids", "pad"], Outputs = ["is_pad"] },
                new OnnxNode { Name = "not", OpType = "Not", Inputs = ["is_pad"], Outputs = ["not_pad"] },
                new OnnxNode
                {
                    Name = "cast", OpType = "Cast", Inputs = ["not_pad"], Outputs = ["mask"],
                    Attributes = new() { ["to"] = new OnnxAttribute { Name = "to", Type = OnnxAttributeType.Int, I = 7 } },
                },
                new OnnxNode { Name = "cumsum", OpType = "CumSum", Inputs = ["mask", "axis"], Outputs = ["positions"] },
                new OnnxNode { Name = "mul", OpType = "Mul", Inputs = ["positions", "mask"], Outputs = ["masked"] },
                new OnnxNode { Name = "add", OpType = "Add", Inputs = ["masked", "pad"], Outputs = ["position_ids"] },
            ],
            Initializers =
            {
                ["pad"] = new OnnxTensorData { Name = "pad", DataType = OnnxDataType.Int64, Dims = [], Longs = [1] },
                ["axis"] = new OnnxTensorData { Name = "axis", DataType = OnnxDataType.Int64, Dims = [], Longs = [1] },
            },
            Inputs = [new OnnxValueInfo { Name = "input_ids", Shape = [1, 4] }],
            Outputs = [new OnnxValueInfo { Name = "position_ids" }],
        };
        var inputIds = new Tensor([1, 4], [5L, 99L, 1L, 1L]);

        var y = Executor.Run(graph, new() { ["input_ids"] = inputIds })["position_ids"];

        Assert.True(y.IsInt64);
        Assert.Equal(new[] { 1, 4 }, y.Shape);
        Assert.Equal(new long[] { 2, 3, 1, 1 }, y.I64!);
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// <summary>Element-wise float comparison with absolute tolerance (default 1e-5).</summary>
    private static void AssertClose(float[] expected, Tensor actual, float tol = 1e-5f)
    {
        Assert.False(actual.IsInt64);
        Assert.Equal(expected.Length, actual.Data.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual.Data[i]) <= tol,
                $"Element {i}: expected {expected[i]}, got {actual.Data[i]} (tolerance {tol}).");
    }

    /// <summary>Asserts the call fails with a DELIBERATE guard exception (Argument /
    /// NotSupported / InvalidOperation) — an accidental NRE or index-out-of-range
    /// still fails the test, without over-pinning which guard type the kernel picked.</summary>
    private static void AssertThrowsGuard(Func<object?> act)
    {
        var ex = Record.Exception(() => { act(); });
        Assert.NotNull(ex);
        Assert.True(ex is ArgumentException or NotSupportedException or InvalidOperationException,
            $"Expected a deliberate guard exception, got {ex!.GetType().Name}: {ex.Message}");
    }
}
