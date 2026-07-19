using System.Buffers;
using System.Numerics.Tensors;

namespace PredictiveCoding.Core;

// ─────────────────────────────────────────────────────────────────────────────
// TensorUtils
//
// Libreria di operazioni tensoriali per il framework di Predictive Coding.
// Tutte le operazioni dense usano System.Numerics.Tensors.TensorPrimitives
// per accelerazione SIMD automatica.
//
// Convenzioni:
//   • Vettori: DenseTensor<float> con Rank == 1, Dimensions[0] == N
//   • Matrici: DenseTensor<float> con Rank == 2, layout row-major
//              Dimensions[0] = righe (m), Dimensions[1] = colonne (k)
//   • MatVecMul(W[m,k], v[k]) → y[m] = W @ v
//   • MatVecMulT(W[m,k], v[m]) → y[k] = W^T @ v
//   • OuterProduct(a[m], b[k]) → C[m,k] dove C[i,j] = a[i]*b[j]
// ─────────────────────────────────────────────────────────────────────────────

public static class TensorUtils
{
    // ── Prodotto matrice-vettore  y = W @ v ─────────────────────────────────
    //
    // W: [m, k]  v: [k]  →  y: [m]
    // y[i] = sum_j W[i,j] * v[j]   ≡ dot product di ogni riga di W con v
    //
    public static DenseTensor<float> MatVecMul(DenseTensor<float> w, DenseTensor<float> v)
    {
        ShapeCheck.Matrix(w, w.Dimensions[0], w.Dimensions[1], nameof(w));
        ShapeCheck.Vector(v, w.Dimensions[1], nameof(v));

        int m = w.Dimensions[0];
        int k = w.Dimensions[1];

        var y = new DenseTensor<float>(m);
        var vSpan = v.Buffer.AsSpan();

        for (int i = 0; i < m; i++)
            // Dot-product della riga i di W con v, usando SIMD via TensorPrimitives
            y.Buffer[i] = TensorPrimitives.Dot(w.Buffer.AsSpan(i * k, k), vSpan);

        return y;
    }

    // ── Prodotto matrice-trasposta-vettore  y = W^T @ v ─────────────────────
    //
    // W: [m, k]  v: [m]  →  y: [k]
    // y[j] = sum_i W[i,j] * v[i]
    //
    // Utilizzato nel calcolo del gradiente dello stato:
    //   dF/dμ_l ∋  -W_l^T @ δ_{l-1}   (Rao & Ballard eq. 2.3)
    //
    public static DenseTensor<float> MatVecMulT(DenseTensor<float> w, DenseTensor<float> v)
    {
        ShapeCheck.Matrix(w, w.Dimensions[0], w.Dimensions[1], nameof(w));
        ShapeCheck.Vector(v, w.Dimensions[0], nameof(v));

        int m = w.Dimensions[0];
        int k = w.Dimensions[1];

        var y = new DenseTensor<float>(k);   // inizializzato a 0

        for (int i = 0; i < m; i++)
        {
            float vi = v.Buffer[i];
            // y += W[i,:] * v[i]   →  y[j] += W[i,j] * v[i]
            TensorPrimitives.MultiplyAdd(
                w.Buffer.AsSpan(i * k, k),
                vi,
                y.Buffer.AsSpan(),
                y.Buffer.AsSpan());
        }

        return y;
    }

    // ── Prodotto esterno  C = a ⊗ b ─────────────────────────────────────────
    //
    // a: [m]  b: [k]  →  C: [m, k]
    // C[i,j] = a[i] * b[j]
    //
    // Usato nell'aggiornamento dei pesi:
    //   ΔW_l = η_W * (δ_{l-1} ⊗ μ_l)   (Rao & Ballard eq. 2.4)
    //
    public static DenseTensor<float> OuterProduct(DenseTensor<float> a, DenseTensor<float> b)
    {
        if (a.Rank != 1) throw new ArgumentException("a deve essere un vettore (Rank=1)");
        if (b.Rank != 1) throw new ArgumentException("b deve essere un vettore (Rank=1)");

        int m = a.Dimensions[0];
        int k = b.Dimensions[0];

        var c = new DenseTensor<float>(m, k);
        var bSpan = b.Buffer.AsSpan();

        for (int i = 0; i < m; i++)
            // Riga i di C = a[i] * b  (scalare × vettore)
            TensorPrimitives.Multiply(bSpan, a.Buffer[i], c.Buffer.AsSpan(i * k, k));

        return c;
    }

    // ── Addizione element-wise  z = x + y ───────────────────────────────────

    public static DenseTensor<float> ElementwiseAdd(DenseTensor<float> x, DenseTensor<float> y)
    {
        ShapeCheck.SameLength(x, y);
        var z = DenseTensor<float>.Like(x);
        TensorPrimitives.Add(x.Buffer.AsSpan(), y.Buffer.AsSpan(), z.Buffer.AsSpan());
        return z;
    }

    // ── Sottrazione element-wise  z = x - y ─────────────────────────────────

    public static DenseTensor<float> ElementwiseSub(DenseTensor<float> x, DenseTensor<float> y)
    {
        ShapeCheck.SameLength(x, y);
        var z = DenseTensor<float>.Like(x);
        TensorPrimitives.Subtract(x.Buffer.AsSpan(), y.Buffer.AsSpan(), z.Buffer.AsSpan());
        return z;
    }

    // ── Moltiplicazione element-wise  z = x ⊙ y ─────────────────────────────

    public static DenseTensor<float> ElementwiseMul(DenseTensor<float> x, DenseTensor<float> y)
    {
        ShapeCheck.SameLength(x, y);
        var z = DenseTensor<float>.Like(x);
        TensorPrimitives.Multiply(x.Buffer.AsSpan(), y.Buffer.AsSpan(), z.Buffer.AsSpan());
        return z;
    }

    // ── Moltiplicazione scalare  z = alpha * x ──────────────────────────────

    public static DenseTensor<float> ScalarMul(DenseTensor<float> x, float alpha)
    {
        var z = DenseTensor<float>.Like(x);
        TensorPrimitives.Multiply(x.Buffer.AsSpan(), alpha, z.Buffer.AsSpan());
        return z;
    }

    // ── Negazione  z = -x ───────────────────────────────────────────────────

    public static DenseTensor<float> Negate(DenseTensor<float> x)
    {
        var z = DenseTensor<float>.Like(x);
        TensorPrimitives.Negate(x.Buffer.AsSpan(), z.Buffer.AsSpan());
        return z;
    }

    // ── Operazioni in-place ─────────────────────────────────────────────────

    /// <summary>dst += alpha * src  (AXPY: alpha times X Plus Y)</summary>
    public static void InPlaceAxpy(float alpha, DenseTensor<float> src, DenseTensor<float> dst)
    {
        ShapeCheck.SameLength(src, dst);
        // dst[i] = alpha * src[i] + dst[i]
        TensorPrimitives.MultiplyAdd(src.Buffer.AsSpan(), alpha, dst.Buffer.AsSpan(), dst.Buffer.AsSpan());
    }

    /// <summary>dst += src element-wise</summary>
    public static void InPlaceAdd(DenseTensor<float> src, DenseTensor<float> dst)
    {
        ShapeCheck.SameLength(src, dst);
        TensorPrimitives.Add(dst.Buffer.AsSpan(), src.Buffer.AsSpan(), dst.Buffer.AsSpan());
    }

    // ── Metriche scalari ────────────────────────────────────────────────────

    /// <summary>||x||^2 = sum_i x[i]^2</summary>
    public static float SumSquares(DenseTensor<float> x) =>
        TensorPrimitives.SumOfSquares(x.Buffer.AsSpan());

    /// <summary>sum_i x[i]</summary>
    public static float Sum(DenseTensor<float> x) =>
        TensorPrimitives.Sum(x.Buffer.AsSpan());

    /// <summary>max_i x[i] → indice argmax</summary>
    public static int Argmax(DenseTensor<float> x)
    {
        int best = 0;
        float bestVal = x.Buffer[0];
        for (int i = 1; i < x.Length; i++)
        {
            if (x.Buffer[i] > bestVal)
            {
                bestVal = x.Buffer[i];
                best = i;
            }
        }
        return best;
    }

    // ── Reshape ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Restituisce una vista dello stesso buffer con nuove dimensioni.
    /// Il numero totale di elementi deve essere invariato.
    /// </summary>
    public static DenseTensor<float> Reshape(DenseTensor<float> x, params int[] newDims)
    {
        int newSize = 1;
        foreach (int d in newDims) newSize *= d;
        if (newSize != x.Length)
            throw new ArgumentException(
                $"Reshape impossibile: {x.Length} elementi → {newSize}");
        return new DenseTensor<float>(x.Buffer, newDims);
    }

    // ── Trasposizione 2D ────────────────────────────────────────────────────

    /// <summary>
    /// Trasposizione di una matrice: W[m,k] → W^T[k,m]
    /// (Crea un nuovo buffer, non una vista.)
    /// </summary>
    public static DenseTensor<float> Transpose2D(DenseTensor<float> w)
    {
        ShapeCheck.Matrix(w, w.Dimensions[0], w.Dimensions[1]);
        int m = w.Dimensions[0], k = w.Dimensions[1];
        var result = new DenseTensor<float>(k, m);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                result[j, i] = w[i, j];
        return result;
    }

    // ── Concatenazione di vettori ───────────────────────────────────────────

    /// <summary>Concatena una sequenza di vettori 1D in un unico vettore.</summary>
    public static DenseTensor<float> Concat(IReadOnlyList<DenseTensor<float>> parts)
    {
        int total = 0;
        foreach (var p in parts) total += p.Length;

        var result = new DenseTensor<float>(total);
        int offset = 0;
        foreach (var p in parts)
        {
            p.Buffer.AsSpan().CopyTo(result.Buffer.AsSpan(offset));
            offset += p.Length;
        }
        return result;
    }

    // ── Clip ─────────────────────────────────────────────────────────────────

    /// <summary>Clip element-wise in [min, max].</summary>
    public static DenseTensor<float> Clip(DenseTensor<float> x, float min, float max)
    {
        var z = DenseTensor<float>.Like(x);
        TensorPrimitives.Clamp(x.Buffer.AsSpan(), min, max, z.Buffer.AsSpan());
        return z;
    }

    /// <summary>Aggiunge bias (vettore 1D) ad ogni elemento di x (vettore 1D).</summary>
    public static DenseTensor<float> AddBias(DenseTensor<float> x, DenseTensor<float> bias)
        => ElementwiseAdd(x, bias);

    // ─────────────────────────────────────────────────────────────────────────
    // Precisione mista (mixed precision)
    //
    // I pesi sono memorizzati in Half (16 bit → metà memoria e banda), ma i
    // prodotti scalari si ACCUMULANO in float (32 bit) per non perdere precisione
    // sulle somme lunghe. È lo schema "mixed precision" standard: storage leggero,
    // accumulo pieno. Sulla CPU il guadagno è memoria/banda; su GPU (cuBLAS /
    // tensor core) anche il throughput di calcolo.
    // ─────────────────────────────────────────────────────────────────────────

    // ── Prodotto matrice-vettore in precisione mista  y = W @ v ─────────────
    //
    // W: [m, k] in Half   v: [k] in float   →  y: [m] in float
    // y[i] = sum_j (float)W[i,j] * v[j]     (conversione SIMD + accumulo in float)
    //
    public static DenseTensor<float> MatVecMulMixed(DenseTensor<Half> w, DenseTensor<float> v)
    {
        ArgumentNullException.ThrowIfNull(w);
        ArgumentNullException.ThrowIfNull(v);
        if (w.Rank != 2)
            throw new InvalidOperationException($"Matrice 'w' attesa di rank 2, trovata {w}");
        ShapeCheck.Vector(v, w.Dimensions[1], nameof(v));

        int m = w.Dimensions[0];
        int k = w.Dimensions[1];

        var y = new DenseTensor<float>(m);
        var vSpan = v.Buffer.AsSpan();

        // Buffer riusato per la riga convertita Half→float: la conversione è
        // vettorizzata (SIMD) e il Dot accumula in float.
        float[] rowBuffer = ArrayPool<float>.Shared.Rent(k);
        try
        {
            Span<float> rowF = rowBuffer.AsSpan(0, k);
            for (int i = 0; i < m; i++)
            {
                TensorPrimitives.ConvertToSingle(w.Buffer.AsSpan(i * k, k), rowF);
                y.Buffer[i] = TensorPrimitives.Dot(rowF, vSpan);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuffer);
        }

        return y;
    }

    // ── Prodotto matrice-trasposta-vettore in precisione mista  y = W^T @ v ──
    //
    // W: [m, k] in Half   v: [m] in float   →  y: [k] in float
    // y[j] = sum_i (float)W[i,j] * v[i]     (conversione SIMD + accumulo in float)
    //
    // Complemento di MatVecMulMixed: entrambi consumano la matrice pesi W in Half.
    // Usato nel gradiente di stato del predictive coding (−W^T @ δ).
    //
    public static DenseTensor<float> MatVecMulTMixed(DenseTensor<Half> w, DenseTensor<float> v)
    {
        ArgumentNullException.ThrowIfNull(w);
        ArgumentNullException.ThrowIfNull(v);
        if (w.Rank != 2)
            throw new InvalidOperationException($"Matrice 'w' attesa di rank 2, trovata {w}");
        ShapeCheck.Vector(v, w.Dimensions[0], nameof(v));

        int m = w.Dimensions[0];
        int k = w.Dimensions[1];

        var y = new DenseTensor<float>(k);   // inizializzato a 0
        var ySpan = y.Buffer.AsSpan();

        // Buffer riusato per la riga convertita Half→float; l'accumulo è in float.
        float[] rowBuffer = ArrayPool<float>.Shared.Rent(k);
        try
        {
            Span<float> rowF = rowBuffer.AsSpan(0, k);
            for (int i = 0; i < m; i++)
            {
                // y += W[i,:] * v[i]
                TensorPrimitives.ConvertToSingle(w.Buffer.AsSpan(i * k, k), rowF);
                TensorPrimitives.MultiplyAdd(rowF, v.Buffer[i], ySpan, ySpan);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuffer);
        }

        return y;
    }

    // ── Conversioni di precisione  float ↔ Half ──────────────────────────────

    /// <summary>Copia un tensore float in un nuovo tensore Half (storage 16 bit), stessa forma.</summary>
    public static DenseTensor<Half> ToHalf(DenseTensor<float> x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var half = new Half[x.Buffer.Length];
        TensorPrimitives.ConvertToHalf(x.Buffer.AsSpan(), half);
        return new DenseTensor<Half>(half, x.Dimensions);
    }

    /// <summary>Copia un tensore Half in un nuovo tensore float, stessa forma.</summary>
    public static DenseTensor<float> ToFloat(DenseTensor<Half> x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var f = new float[x.Buffer.Length];
        TensorPrimitives.ConvertToSingle(x.Buffer.AsSpan(), f);
        return new DenseTensor<float>(f, x.Dimensions);
    }

    // ── Conversioni di precisione  float ↔ bfloat16 ──────────────────────────
    //
    // bfloat16 = i 16 bit alti di un float32: stesso esponente (stesso range,
    // niente overflow ai valori grandi, a differenza di Half), mantissa a 8 bit.
    // Arrotondamento round-to-nearest-even; i NaN restano NaN.

    /// <summary>Converte float32 → bit bfloat16 (round-to-nearest-even).</summary>
    public static void FloatToBf16Bits(ReadOnlySpan<float> src, Span<ushort> dst)
    {
        if (dst.Length < src.Length) throw new ArgumentException("dst too short.", nameof(dst));
        for (int i = 0; i < src.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(src[i]);
            if (float.IsNaN(src[i]))
            {
                dst[i] = (ushort)((bits >> 16) | 0x0040);          // keep it NaN
                continue;
            }
            uint rounding = 0x7FFFu + ((bits >> 16) & 1u);         // round to nearest even
            dst[i] = (ushort)((bits + rounding) >> 16);
        }
    }

    /// <summary>Converte bit bfloat16 → float32 (esatta: shift dei 16 bit alti).</summary>
    public static void Bf16BitsToFloat(ReadOnlySpan<ushort> src, Span<float> dst)
    {
        if (dst.Length < src.Length) throw new ArgumentException("dst too short.", nameof(dst));
        for (int i = 0; i < src.Length; i++)
            dst[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
    }
}
