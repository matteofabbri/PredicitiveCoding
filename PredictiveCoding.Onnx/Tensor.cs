namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — the runtime value flowing through the in-house ONNX executor.
//
// Float32 is the primary payload (Data); a tensor may instead carry int64
// values (I64) — ONNX uses int64 tensors for shapes and Slice/Reshape
// parameters, never for heavy math, so both live in one class rather than a
// generic hierarchy. Layout is row-major (C order), matching ONNX.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Tensor
{
    public int[] Shape { get; }
    public float[] Data { get; }

    /// <summary>Set instead of <see cref="Data"/> when the tensor holds int64 values
    /// (shape/index inputs of Reshape, Slice, …). <see cref="Data"/> is empty then.</summary>
    public long[]? I64 { get; }

    public bool IsInt64 => I64 is not null;
    public int Count => IsInt64 ? I64!.Length : Data.Length;

    public Tensor(int[] shape)
    {
        Shape = shape;
        Data = new float[CheckedProduct(shape)];
    }

    public Tensor(int[] shape, float[] data)
    {
        if (data.Length != CheckedProduct(shape))
            throw new ArgumentException($"Data length {data.Length} does not match shape [{string.Join(",", shape)}].");
        Shape = shape;
        Data = data;
    }

    public Tensor(int[] shape, long[] i64)
    {
        if (i64.Length != CheckedProduct(shape))
            throw new ArgumentException($"Data length {i64.Length} does not match shape [{string.Join(",", shape)}].");
        Shape = shape;
        Data = [];
        I64 = i64;
    }

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

    /// <summary>Shape as "[1,3,640,640]" for diagnostics.</summary>
    public string ShapeString => $"[{string.Join(",", Shape)}]";

    /// <summary>Element count of a shape, validating against overflow/negatives.</summary>
    public static int CheckedProduct(ReadOnlySpan<int> shape)
    {
        long n = 1;
        foreach (int d in shape)
        {
            if (d < 0) throw new ArgumentException($"Negative dimension {d} in shape.");
            n *= d;
            if (n > int.MaxValue) throw new ArgumentException("Tensor too large.");
        }
        return (int)n;
    }

    /// <summary>Row-major strides for this tensor's shape.</summary>
    public int[] Strides()
    {
        var s = new int[Rank];
        int acc = 1;
        for (int i = Rank - 1; i >= 0; i--)
        {
            s[i] = acc;
            acc *= Shape[i];
        }
        return s;
    }
}
