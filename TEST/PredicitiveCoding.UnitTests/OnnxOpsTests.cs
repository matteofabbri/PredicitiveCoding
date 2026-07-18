using PredictiveCoding.Onnx;
using PredictiveCoding.Onnx.Proto;
using Xunit;

namespace PredictiveCoding.Onnx.Tests;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxOpsTests — independent checks on the in-house ONNX kernels (Ops) and the
// graph Executor.
//
// Every expected value is hand-computed from the ONNX operator spec (opset
// 11-13 semantics) and written out literally — the tests never derive an
// expectation from the runtime itself, so they remain an independent oracle
// for the kernel implementations. Kernel call shapes mirror exactly what
// Executor.Dispatch passes (including empty long[] for absent attributes).
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Hand-computed tests for the Ops kernels (elementwise, conv, pooling,
/// linear algebra, shape plumbing, int64 flow) and the graph Executor.</summary>
public class OnnxOpsTests
{
    private const float Tol = 1e-5f;

    // ── helpers ───────────────────────────────────────────────────────────────

    private static Tensor T(int[] shape, params float[] data) => new(shape, data);

    private static Tensor I(int[] shape, params long[] data) => new(shape, data);

    private static void AssertTensor(Tensor actual, int[] expectedShape, params float[] expected)
    {
        Assert.False(actual.IsInt64);
        Assert.Equal(expectedShape, actual.Shape);
        Assert.Equal(expected.Length, actual.Data.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.Data[i]) <= Tol,
                $"element [{i}]: expected {expected[i]}, got {actual.Data[i]} (shape {actual.ShapeString})");
    }

    private static void AssertI64(Tensor actual, int[] expectedShape, params long[] expected)
    {
        Assert.True(actual.IsInt64, "expected an int64 tensor");
        Assert.Equal(expectedShape, actual.Shape);
        Assert.Equal(expected, actual.I64);
    }

    // ── Add / Mul broadcasting ────────────────────────────────────────────────

    [Fact]
    public void Add_SameShape_FastPath()
        => AssertTensor(Ops.Add(T([2, 2], 1, 2, 3, 4), T([2, 2], 10, 20, 30, 40)),
                        [2, 2], 11, 22, 33, 44);

    [Fact]
    public void Add_Broadcast_RowVector_2x3_Plus_3()
        // [1,2,3;4,5,6] + [10,20,30] per row → [11,22,33;14,25,36]
        => AssertTensor(Ops.Add(T([2, 3], 1, 2, 3, 4, 5, 6), T([3], 10, 20, 30)),
                        [2, 3], 11, 22, 33, 14, 25, 36);

    [Fact]
    public void Add_Broadcast_2x1_Plus_1x3_Gives_2x3()
        // column [1;2] + row [10,20,30] → [[1+10,1+20,1+30],[2+10,2+20,2+30]]
        => AssertTensor(Ops.Add(T([2, 1], 1, 2), T([1, 3], 10, 20, 30)),
                        [2, 3], 11, 21, 31, 12, 22, 32);

    [Fact]
    public void Add_Broadcast_Scalar()
        => AssertTensor(Ops.Add(T([2, 3], 1, 2, 3, 4, 5, 6), T([1], 100)),
                        [2, 3], 101, 102, 103, 104, 105, 106);

    [Fact]
    public void Add_IncompatibleShapes_Throws()
        // trailing dims 3 vs 2 — not broadcastable
        => Assert.ThrowsAny<Exception>(() => Ops.Add(T([2, 3], 1, 2, 3, 4, 5, 6), T([2], 1, 2)));

    [Fact]
    public void Mul_Broadcast_RowVector()
        // [1,2,3;4,5,6] * [2,3,4] per row → [2,6,12;8,15,24]
        => AssertTensor(Ops.Mul(T([2, 3], 1, 2, 3, 4, 5, 6), T([3], 2, 3, 4)),
                        [2, 3], 2, 6, 12, 8, 15, 24);

    [Fact]
    public void Mul_IncompatibleShapes_Throws()
        => Assert.ThrowsAny<Exception>(() => Ops.Mul(T([2, 3], 1, 2, 3, 4, 5, 6), T([4], 1, 2, 3, 4)));

    // ── Sub / Div ─────────────────────────────────────────────────────────────

    [Fact]
    public void Sub_Basic()
        => AssertTensor(Ops.Sub(T([3], 5, 7, 9), T([3], 1, 2, 3)), [3], 4, 5, 6);

    [Fact]
    public void Div_Basic()
        => AssertTensor(Ops.Div(T([3], 6, 8, 9), T([3], 2, 4, 3)), [3], 3, 2, 3);

    // ── Activations ───────────────────────────────────────────────────────────

    [Fact]
    public void Sigmoid_KnownValues()
        // σ(x) = 1/(1+e^-x): σ(-1)=0.26894142, σ(0)=0.5, σ(1)=0.73105858, σ(2)=0.88079708
        => AssertTensor(Ops.Sigmoid(T([4], -1, 0, 1, 2)),
                        [4], 0.26894142f, 0.5f, 0.73105858f, 0.88079708f);

    [Fact]
    public void Relu_ClampsNegatives()
        => AssertTensor(Ops.Relu(T([4], -2, -0.5f, 0, 3)), [4], 0, 0, 0, 3);

    [Fact]
    public void Exp_KnownValues()
        // e^0=1, e^1=2.71828183, e^-1=0.36787944
        => AssertTensor(Ops.Exp(T([3], 0, 1, -1)), [3], 1f, 2.71828183f, 0.36787944f);

    [Fact]
    public void Sqrt_KnownValues()
        => AssertTensor(Ops.Sqrt(T([4], 0, 1, 4, 2.25f)), [4], 0, 1, 2, 1.5f);

    [Fact]
    public void PRelu_PerChannelSlope_OnNchw()
    {
        // x [1,2,2,2], slope [2,1,1] broadcasts over the channel axis of NCHW:
        // ch0 slope 0.1 → [-1*0.1, 2, -3*0.1, 4]; ch1 slope 0.5 → [-5*0.5, 6, -7*0.5, 8]
        var x = T([1, 2, 2, 2], -1, 2, -3, 4, -5, 6, -7, 8);
        var slope = T([2, 1, 1], 0.1f, 0.5f);
        AssertTensor(Ops.PRelu(x, slope),
                     [1, 2, 2, 2], -0.1f, 2, -0.3f, 4, -2.5f, 6, -3.5f, 8);
    }

    // ── Softmax ───────────────────────────────────────────────────────────────

    [Fact]
    public void Softmax_LastAxis_On2x3()
    {
        // row [1,2,3]: e=[2.7182818, 7.3890561, 20.0855369], sum=30.1928749
        //   → [0.09003057, 0.24472847, 0.66524096]
        // row [1000,1001,1002]: same offsets — a stable kernel (max-subtraction)
        //   must give the identical result instead of overflowing e^1000 to +inf.
        var y = Ops.Softmax(T([2, 3], 1, 2, 3, 1000, 1001, 1002), -1);
        AssertTensor(y, [2, 3],
                     0.09003057f, 0.24472847f, 0.66524096f,
                     0.09003057f, 0.24472847f, 0.66524096f);

        Assert.All(y.Data, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(1f, y.Data[0] + y.Data[1] + y.Data[2], Tol);   // rows sum to 1
        Assert.Equal(1f, y.Data[3] + y.Data[4] + y.Data[5], Tol);
    }

    [Fact]
    public void Softmax_Axis0_On2x3()
    {
        // Opset-13 semantics (matching the dispatcher's axis=-1 default): softmax
        // ALONG axis 0, i.e. per column. Columns (1,4),(2,5),(3,6) each differ by
        // 3 → [1/(1+e³), e³/(1+e³)] = [0.04742587, 0.95257413] in every column.
        var y = Ops.Softmax(T([2, 3], 1, 2, 3, 4, 5, 6), 0);
        AssertTensor(y, [2, 3],
                     0.04742587f, 0.04742587f, 0.04742587f,
                     0.95257413f, 0.95257413f, 0.95257413f);
    }

    // ── Conv ──────────────────────────────────────────────────────────────────

    [Fact]
    public void Conv_Identity1x1Kernel_EmptyAttributeDefaults()
    {
        // 1×1 kernel of weight 1 on C=1: pure identity. strides/pads/dilations are
        // passed EMPTY — exactly what Executor passes when the attributes are
        // absent — so the kernel must default them to 1/0/1.
        var x = T([1, 1, 2, 2], 1, 2, 3, 4);
        var w = T([1, 1, 1, 1], 1);
        AssertTensor(Ops.Conv(x, w, null, [], [], [], 1), [1, 1, 2, 2], 1, 2, 3, 4);
    }

    [Fact]
    public void Conv_1x1Kernel_MixesInputChannels()
    {
        // out = 1.0*ch0 + 0.5*ch1: [1+5, 2+10, 3+15, 4+20] = [6,12,18,24]
        var x = T([1, 2, 2, 2], 1, 2, 3, 4, 10, 20, 30, 40);
        var w = T([1, 2, 1, 1], 1f, 0.5f);
        AssertTensor(Ops.Conv(x, w, null, [1, 1], [0, 0, 0, 0], [1, 1], 1),
                     [1, 1, 2, 2], 6, 12, 18, 24);
    }

    [Fact]
    public void Conv_3x3SumKernel_Pads1_On4x4()
    {
        // x = 1..16 row-major 4×4, w = 3×3 of ones, pads [1,1,1,1] → each output
        // is the sum of the zero-padded 3×3 neighbourhood. Hand-computed:
        //   row0: 1+2+5+6=14 | 1+2+3+5+6+7=24 | 2+3+4+6+7+8=30 | 3+4+7+8=22
        //   row1: 1+2+5+6+9+10=33 | (1..3)+(5..7)+(9..11)=54 | (2..4)+(6..8)+(10..12)=63 | 3+4+7+8+11+12=45
        //   row2: 5+6+9+10+13+14=57 | (5..7)+(9..11)+(13..15)=90 | (6..8)+(10..12)+(14..16)=99 | 7+8+11+12+15+16=69
        //   row3: 9+10+13+14=46 | 9+10+11+13+14+15=72 | 10+11+12+14+15+16=78 | 11+12+15+16=54
        var x = T([1, 1, 4, 4],
                  1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16);
        var w = T([1, 1, 3, 3], 1, 1, 1, 1, 1, 1, 1, 1, 1);
        AssertTensor(Ops.Conv(x, w, null, [1, 1], [1, 1, 1, 1], [1, 1], 1),
                     [1, 1, 4, 4],
                     14, 24, 30, 22,
                     33, 54, 63, 45,
                     57, 90, 99, 69,
                     46, 72, 78, 54);
    }

    [Fact]
    public void Conv_Stride2_2x2SumKernel()
    {
        // 2×2 ones kernel, stride 2, no pads on the 1..16 grid → 2×2 output:
        //   1+2+5+6=14 | 3+4+7+8=22 | 9+10+13+14=46 | 11+12+15+16=54
        var x = T([1, 1, 4, 4],
                  1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16);
        var w = T([1, 1, 2, 2], 1, 1, 1, 1);
        AssertTensor(Ops.Conv(x, w, null, [2, 2], [0, 0, 0, 0], [1, 1], 1),
                     [1, 1, 2, 2], 14, 22, 46, 54);
    }

    [Fact]
    public void Conv_Dilation2_2x2SumKernel()
    {
        // 2×2 ones kernel with dilation 2 spans 3×3, so out[i,j] =
        // x[i,j]+x[i,j+2]+x[i+2,j]+x[i+2,j+2] and the output is 2×2:
        //   1+3+9+11=24 | 2+4+10+12=28 | 5+7+13+15=40 | 6+8+14+16=44
        var x = T([1, 1, 4, 4],
                  1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16);
        var w = T([1, 1, 2, 2], 1, 1, 1, 1);
        AssertTensor(Ops.Conv(x, w, null, [1, 1], [0, 0, 0, 0], [2, 2], 1),
                     [1, 1, 2, 2], 24, 28, 40, 44);
    }

    [Fact]
    public void Conv_Group2_EachOutputSeesOnlyItsInputChannels()
    {
        // C=4, M=2, group=2: W is [M, C/group, 1, 1] = [2,2,1,1].
        // Output ch0 (group 0) sees input ch 0,1: 1*10 + 2*100 = 210.
        // Output ch1 (group 1) sees input ch 2,3: 3*1000 + 4*10000 = 43000.
        // (If grouping were ignored, ch0 would also pick up 3 and 4.)
        var x = T([1, 4, 1, 1], 1, 2, 3, 4);
        var w = T([2, 2, 1, 1], 10, 100, 1000, 10000);
        AssertTensor(Ops.Conv(x, w, null, [1, 1], [0, 0, 0, 0], [1, 1], 2),
                     [1, 2, 1, 1], 210, 43000);
    }

    [Fact]
    public void Conv_BiasAdded()
    {
        // 1×1 kernel weight 2, bias 10: 2*x + 10 → [12,14,16,18]
        var x = T([1, 1, 2, 2], 1, 2, 3, 4);
        var w = T([1, 1, 1, 1], 2);
        var b = T([1], 10);
        AssertTensor(Ops.Conv(x, w, b, [1, 1], [0, 0, 0, 0], [1, 1], 1),
                     [1, 1, 2, 2], 12, 14, 16, 18);
    }

    // ── MaxPool / AveragePool / GlobalAveragePool ─────────────────────────────

    [Fact]
    public void MaxPool_2x2Stride2_On4x4()
    {
        // Non-overlapping 2×2 windows on the 1..16 grid:
        //   max(1,2,5,6)=6 | max(3,4,7,8)=8 | max(9,10,13,14)=14 | max(11,12,15,16)=16
        var x = T([1, 1, 4, 4],
                  1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16);
        AssertTensor(Ops.MaxPool(x, [2, 2], [2, 2], [0, 0, 0, 0], false),
                     [1, 1, 2, 2], 6, 8, 14, 16);
    }

    [Fact]
    public void MaxPool_3x3Stride1_Pads1_SamePadding()
    {
        // k=3, s=1, pads [1,1,1,1] on the 1..16 grid → 4×4 output; padding cells
        // never win a max. out[i,j] = max of the clamped 3×3 window centred at (i,j):
        //   row0: 6  7  8  8      row1: 10 11 12 12
        //   row2: 14 15 16 16     row3: 14 15 16 16
        var x = T([1, 1, 4, 4],
                  1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16);
        AssertTensor(Ops.MaxPool(x, [3, 3], [1, 1], [1, 1, 1, 1], false),
                     [1, 1, 4, 4],
                     6, 7, 8, 8,
                     10, 11, 12, 12,
                     14, 15, 16, 16,
                     14, 15, 16, 16);
    }

    [Fact]
    public void MaxPool_CeilMode_GrowsOutput_WindowClamped()
    {
        // 5×5 (values 1..25), k=2, s=2, no pads.
        // floor: (5-2)/2+1 = 2 → 2×2:  max(1,2,6,7)=7   max(3,4,8,9)=9
        //                              max(11,12,16,17)=17 max(13,14,18,19)=19
        // ceil:  ceil((5-2)/2)+1 = 3 → 3×3, last window starts at 4 and is clamped:
        //   7  9  max(5,10)=10
        //   17 19 max(15,20)=20
        //   max(21,22)=22 max(23,24)=24 max(25)=25
        var x = T([1, 1, 5, 5],
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25);

        AssertTensor(Ops.MaxPool(x, [2, 2], [2, 2], [0, 0, 0, 0], false),
                     [1, 1, 2, 2], 7, 9, 17, 19);

        AssertTensor(Ops.MaxPool(x, [2, 2], [2, 2], [0, 0, 0, 0], true),
                     [1, 1, 3, 3],
                     7, 9, 10,
                     17, 19, 20,
                     22, 24, 25);
    }

    [Fact]
    public void AveragePool_CountIncludePad_TrueVsFalse()
    {
        // x = [1,2;3,4], k=2, s=1, pads [1,1,1,1] → 3×3 output. Window sums and
        // valid-cell counts (kernel area is always 4):
        //   sums:   1  3  2      counts: 1 2 1
        //           4 10  6              2 4 2
        //           3  7  4              1 2 1
        // exclude pad → sum/count; include pad → sum/4.
        var x = T([1, 1, 2, 2], 1, 2, 3, 4);

        AssertTensor(Ops.AveragePool(x, [2, 2], [1, 1], [1, 1, 1, 1], false),
                     [1, 1, 3, 3],
                     1f, 1.5f, 2f,
                     2f, 2.5f, 3f,
                     3f, 3.5f, 4f);

        AssertTensor(Ops.AveragePool(x, [2, 2], [1, 1], [1, 1, 1, 1], true),
                     [1, 1, 3, 3],
                     0.25f, 0.75f, 0.5f,
                     1f, 2.5f, 1.5f,
                     0.75f, 1.75f, 1f);
    }

    [Fact]
    public void GlobalAveragePool_PerChannelMeans()
    {
        // ch0 mean(1,2,3,4)=2.5, ch1 mean(10,20,30,40)=25 → shape [1,2,1,1]
        var x = T([1, 2, 2, 2], 1, 2, 3, 4, 10, 20, 30, 40);
        AssertTensor(Ops.GlobalAveragePool(x), [1, 2, 1, 1], 2.5f, 25f);
    }

    // ── Gemm / MatMul / BatchNorm ─────────────────────────────────────────────

    [Fact]
    public void Gemm_Plain()
    {
        // A[2,3]=[1,2,3;4,5,6], B[3,2]=[7,8;9,10;11,12]:
        //   [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        //   [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        var a = T([2, 3], 1, 2, 3, 4, 5, 6);
        var b = T([3, 2], 7, 8, 9, 10, 11, 12);
        AssertTensor(Ops.Gemm(a, b, null, 1f, 1f, false, false),
                     [2, 2], 58, 64, 139, 154);
    }

    [Fact]
    public void Gemm_TransB()
    {
        // B' [2,3] = [7,9,11;8,10,12] is the transpose of the B above, so
        // transB=true must reproduce [58,64;139,154].
        var a = T([2, 3], 1, 2, 3, 4, 5, 6);
        var bt = T([2, 3], 7, 9, 11, 8, 10, 12);
        AssertTensor(Ops.Gemm(a, bt, null, 1f, 1f, false, true),
                     [2, 2], 58, 64, 139, 154);
    }

    [Fact]
    public void Gemm_AlphaBeta_WithCVectorBroadcast()
    {
        // Y = 2*A·B + 3*C with C=[1,-1] broadcast over rows of the [2,2] product:
        //   [2*58+3*1, 2*64+3*(-1)]  = [119, 125]
        //   [2*139+3*1, 2*154+3*(-1)] = [281, 305]
        var a = T([2, 3], 1, 2, 3, 4, 5, 6);
        var b = T([3, 2], 7, 8, 9, 10, 11, 12);
        var c = T([2], 1, -1);
        AssertTensor(Ops.Gemm(a, b, c, 2f, 3f, false, false),
                     [2, 2], 119, 125, 281, 305);
    }

    [Fact]
    public void MatMul_2D()
        // same product as Gemm_Plain
        => AssertTensor(Ops.MatMul(T([2, 3], 1, 2, 3, 4, 5, 6),
                                   T([3, 2], 7, 8, 9, 10, 11, 12)),
                        [2, 2], 58, 64, 139, 154);

    [Fact]
    public void MatMul_Batched()
    {
        // batch 0: [1,2,3;4,5,6] × [7,8;9,10;11,12] = [58,64;139,154]
        // batch 1: [1,0,0;0,1,0] × [1,2;3,4;5,6]    = [1,2;3,4] (row selection)
        var a = T([2, 2, 3],
                  1, 2, 3, 4, 5, 6,
                  1, 0, 0, 0, 1, 0);
        var b = T([2, 3, 2],
                  7, 8, 9, 10, 11, 12,
                  1, 2, 3, 4, 5, 6);
        AssertTensor(Ops.MatMul(a, b), [2, 2, 2],
                     58, 64, 139, 154,
                     1, 2, 3, 4);
    }

    [Fact]
    public void BatchNorm_PerChannel()
    {
        // y = scale*(x-mean)/sqrt(var+eps) + bias, eps=1e-5.
        // ch0 (scale 1, bias 0, mean 2, var 4): (x-2)/2 → [-0.5, 0, 0.5, 1]
        // ch1 (scale 2, bias 1, mean 6, var 9): 2*(x-6)/3+1 → [1/3, 1, 5/3, 7/3]
        // (the eps shifts each value by < 1.3e-6 — inside the 1e-5 tolerance)
        var x = T([1, 2, 2, 2], 1, 2, 3, 4, 5, 6, 7, 8);
        var scale = T([2], 1, 2);
        var bias = T([2], 0, 1);
        var mean = T([2], 2, 6);
        var variance = T([2], 4, 9);
        AssertTensor(Ops.BatchNorm(x, scale, bias, mean, variance, 1e-5f),
                     [1, 2, 2, 2],
                     -0.5f, 0f, 0.5f, 1f,
                     0.33333333f, 1f, 1.66666667f, 2.33333333f);
    }

    // ── Reshape / Flatten / Transpose / Concat ────────────────────────────────

    [Fact]
    public void Reshape_Explicit_PreservesRowMajorData()
        => AssertTensor(Ops.Reshape(T([2, 3], 1, 2, 3, 4, 5, 6), I([2], 3, 2)),
                        [3, 2], 1, 2, 3, 4, 5, 6);

    [Fact]
    public void Reshape_Zero_CopiesInputDim()
        // 0 in the target copies the corresponding input dim (allowzero=0 default):
        // [2,6] with target [0,2,3] → [2,2,3]
        => AssertTensor(Ops.Reshape(T([2, 6], 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                                    I([3], 0, 2, 3)),
                        [2, 2, 3], 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

    [Fact]
    public void Reshape_MinusOne_Inferred()
        // 6 elements with target [3,-1] → -1 resolves to 2
        => AssertTensor(Ops.Reshape(T([2, 3], 1, 2, 3, 4, 5, 6), I([2], 3, -1)),
                        [3, 2], 1, 2, 3, 4, 5, 6);

    [Fact]
    public void Reshape_ProductMismatch_Throws()
        // 6 elements cannot fill [4,2]
        => Assert.ThrowsAny<Exception>(
               () => Ops.Reshape(T([2, 3], 1, 2, 3, 4, 5, 6), I([2], 4, 2)));

    [Fact]
    public void Flatten_Axis0_1_MinusOne()
    {
        // [2,3,4]: axis 0 → [1,24]; axis 1 → [2,12]; axis -1 (= rank-1 = 2) → [6,4].
        // Row-major data never moves.
        var data = new float[24];
        for (int i = 0; i < 24; i++) data[i] = i;
        var x = new Tensor([2, 3, 4], data);

        AssertTensor(Ops.Flatten(x, 0), [1, 24], data);
        AssertTensor(Ops.Flatten(x, 1), [2, 12], data);
        AssertTensor(Ops.Flatten(x, -1), [6, 4], data);
    }

    [Fact]
    public void Transpose_DefaultPerm_ReversesAxes()
        // empty perm (what Executor passes when the attribute is absent) = reverse:
        // [1,2,3;4,5,6]ᵀ = [1,4;2,5;3,6]
        => AssertTensor(Ops.Transpose(T([2, 3], 1, 2, 3, 4, 5, 6), []),
                        [3, 2], 1, 4, 2, 5, 3, 6);

    [Fact]
    public void Transpose_ExplicitPerm_NchwToNhwc()
    {
        // x [1,2,2,2] with ch0=[1,2;3,4], ch1=[5,6;7,8]; perm [0,2,3,1] gives
        // out[n,h,w,c] = x[n,c,h,w] → data order (h,w,c): [1,5, 2,6, 3,7, 4,8]
        var x = T([1, 2, 2, 2], 1, 2, 3, 4, 5, 6, 7, 8);
        AssertTensor(Ops.Transpose(x, [0, 2, 3, 1]),
                     [1, 2, 2, 2], 1, 5, 2, 6, 3, 7, 4, 8);
    }

    [Fact]
    public void Concat_Axis1()
        // [1,2;3,4] ++ [5;6] along axis 1 → [1,2,5;3,4,6]
        => AssertTensor(Ops.Concat([T([2, 2], 1, 2, 3, 4), T([2, 1], 5, 6)], 1),
                        [2, 3], 1, 2, 5, 3, 4, 6);

    [Fact]
    public void Concat_NegativeAxis()
        // axis -1 on rank 2 = axis 1: [1,2] ++ [3,4,5] → [1,2,3,4,5]
        => AssertTensor(Ops.Concat([T([1, 2], 1, 2), T([1, 3], 3, 4, 5)], -1),
                        [1, 5], 1, 2, 3, 4, 5);

    // ── Slice ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Slice_Basic_DefaultAxesAndSteps()
        // x = 0..9, starts [2], ends [5], axes/steps omitted → [2,3,4]
        => AssertTensor(Ops.Slice(T([10], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                  I([1], 2), I([1], 5), null, null),
                        [3], 2, 3, 4);

    [Fact]
    public void Slice_NegativeStartEnd()
        // starts [-4] → 6, ends [-1] → 9 (exclusive) → [6,7,8]
        => AssertTensor(Ops.Slice(T([10], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                  I([1], -4), I([1], -1), I([1], 0), null),
                        [3], 6, 7, 8);

    [Fact]
    public void Slice_Step2_EndClampedToDim()
        // ends 100 must clamp to 10; step 2 from 0 → [0,2,4,6,8]
        => AssertTensor(Ops.Slice(T([10], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                  I([1], 0), I([1], 100), I([1], 0), I([1], 2)),
                        [5], 0, 2, 4, 6, 8);

    [Fact]
    public void Slice_NegativeStep_PartialReverse()
        // start 8, end 2 (exclusive), step -2 → indices 8,6,4 → [8,6,4]
        => AssertTensor(Ops.Slice(T([10], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                  I([1], 8), I([1], 2), I([1], 0), I([1], -2)),
                        [3], 8, 6, 4);

    [Fact]
    public void Slice_NegativeStep_FullReverse()
        // start -1 → 9, end -11 (before element 0), step -1 → 9,8,...,0
        => AssertTensor(Ops.Slice(T([10], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                  I([1], -1), I([1], -11), I([1], 0), I([1], -1)),
                        [10], 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    [Fact]
    public void Slice_AxesSubset_OnlyNamedAxisSliced()
        // x [2,3] = [1,2,3;4,5,6]; slice cols 1..3 only (axes=[1]) → [2,3;5,6]
        => AssertTensor(Ops.Slice(T([2, 3], 1, 2, 3, 4, 5, 6),
                                  I([1], 1), I([1], 3), I([1], 1), null),
                        [2, 2], 2, 3, 5, 6);

    // ── Shape / Unsqueeze / Squeeze ───────────────────────────────────────────

    [Fact]
    public void Shape_ReturnsInt64Dims()
        => AssertI64(Ops.Shape(new Tensor([2, 3, 4])), [3], 2, 3, 4);

    [Fact]
    public void Unsqueeze_Axes0And3_TensorAndAttributeForms()
    {
        // [2,3] with axes [0,3] (positions in the OUTPUT) → [1,2,3,1]
        var x = T([2, 3], 1, 2, 3, 4, 5, 6);

        // opset-13 tensor form
        AssertTensor(Ops.Unsqueeze(x, I([2], 0, 3), []), [1, 2, 3, 1], 1, 2, 3, 4, 5, 6);
        // opset-11 attribute form
        AssertTensor(Ops.Unsqueeze(x, null, [0, 3]), [1, 2, 3, 1], 1, 2, 3, 4, 5, 6);
    }

    [Fact]
    public void Squeeze_SpecificAxis_TensorAndAttributeForms()
    {
        // [1,2,1,3], squeeze axis 2 only → [1,2,3]
        var x = T([1, 2, 1, 3], 1, 2, 3, 4, 5, 6);

        AssertTensor(Ops.Squeeze(x, I([1], 2), []), [1, 2, 3], 1, 2, 3, 4, 5, 6);
        AssertTensor(Ops.Squeeze(x, null, [2]), [1, 2, 3], 1, 2, 3, 4, 5, 6);
    }

    [Fact]
    public void Squeeze_NoAxes_RemovesAllOnes()
        // no axes at all → every dim == 1 goes: [1,2,1,3] → [2,3]
        => AssertTensor(Ops.Squeeze(T([1, 2, 1, 3], 1, 2, 3, 4, 5, 6), null, []),
                        [2, 3], 1, 2, 3, 4, 5, 6);

    // ── Int64 flow ────────────────────────────────────────────────────────────

    [Fact]
    public void Int64Flow_ShapeSliceConcat_StaysInt64()
    {
        // The classic reshape-target build: Shape → Slice the spatial dims →
        // Concat with a leading -1. Every intermediate must stay int64.
        var x = new Tensor([1, 3, 4, 4]);                                   // 48 zeros

        var shape = Ops.Shape(x);
        AssertI64(shape, [4], 1, 3, 4, 4);

        var hw = Ops.Slice(shape, I([1], 2), I([1], 4), I([1], 0), null);   // dims 2..4
        AssertI64(hw, [2], 4, 4);

        var target = Ops.Concat([I([1], -1), hw], 0);
        AssertI64(target, [3], -1, 4, 4);

        // and the target actually drives Reshape: 48 elements → [-1,4,4] = [3,4,4]
        var reshaped = Ops.Reshape(x, target);
        Assert.Equal(new[] { 3, 4, 4 }, reshaped.Shape);
    }

    // ── Executor ──────────────────────────────────────────────────────────────

    /// <summary>Conv(1×1, weight 2 as initializer) → Sigmoid, built in code:
    /// the smallest graph that exercises initializer loading, node chaining and
    /// attribute-less dispatch (Conv defaults come from empty GetInts results).</summary>
    private static OnnxGraph BuildConvSigmoidGraph()
    {
        var graph = new OnnxGraph
        {
            Name = "silu-ish",
            Nodes =
            [
                new OnnxNode { Name = "conv", OpType = "Conv", Inputs = ["x", "w"], Outputs = ["c"] },
                new OnnxNode { Name = "sig", OpType = "Sigmoid", Inputs = ["c"], Outputs = ["y"] },
            ],
            Inputs = [new OnnxValueInfo { Name = "x", Shape = [1, 1, 2, 2] }],
            Outputs = [new OnnxValueInfo { Name = "y" }],
        };
        graph.Initializers["w"] = new OnnxTensorData
        {
            Name = "w",
            DataType = OnnxDataType.Float,
            Dims = [1, 1, 1, 1],
            Floats = [2f],
        };
        return graph;
    }

    [Fact]
    public void Executor_ConvSigmoidGraph_EndToEnd()
    {
        // conv: 2*x = [0,2,-2,4]; sigmoid: [0.5, σ(2)=0.88079708,
        // σ(-2)=0.11920292, σ(4)=0.98201379]
        var graph = BuildConvSigmoidGraph();
        var outputs = Executor.Run(graph, new() { ["x"] = T([1, 1, 2, 2], 0, 1, -1, 2) });

        Assert.Single(outputs);
        AssertTensor(outputs["y"], [1, 1, 2, 2],
                     0.5f, 0.88079708f, 0.11920292f, 0.98201379f);
    }

    [Fact]
    public void Executor_MissingGraphInput_ThrowsArgumentException()
    {
        var graph = BuildConvSigmoidGraph();
        var ex = Assert.Throws<ArgumentException>(
            () => Executor.Run(graph, new Dictionary<string, Tensor>()));
        Assert.Contains("x", ex.Message);
    }

    [Fact]
    public void Executor_UnsupportedOp_ThrowsNamingTheOp()
    {
        var graph = new OnnxGraph
        {
            Nodes = [new OnnxNode { Name = "lstm0", OpType = "LSTM", Inputs = ["x"], Outputs = ["y"] }],
            Inputs = [new OnnxValueInfo { Name = "x", Shape = [1] }],
            Outputs = [new OnnxValueInfo { Name = "y" }],
        };
        var ex = Assert.Throws<NotSupportedException>(
            () => Executor.Run(graph, new() { ["x"] = T([1], 1) }));
        Assert.Contains("LSTM", ex.Message);
    }
}
