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
}
