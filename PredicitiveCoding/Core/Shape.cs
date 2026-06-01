using System.Numerics;

namespace PredictiveCoding.Core;

// ─────────────────────────────────────────────────────────────────────────────
// DenseTensor<T>
//
// Tensore denso N-dimensionale in layout row-major (C-order).
// Backed da un array piatto T[]; tutte le operazioni matematiche usano
// System.Numerics.Tensors.TensorPrimitives per SIMD automatico.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Tensore denso N-dimensionale ad alto controllo (row-major, layout contiguo).
/// </summary>
public sealed class DenseTensor<T> where T : struct, INumber<T>
{
    // ── Dati ────────────────────────────────────────────────────────────────
    public T[]   Buffer     { get; }
    public int[] Dimensions { get; }

    // ── Proprietà derivate ─────────────────────────────────────────────────
    public int Rank   => Dimensions.Length;
    public int Length => Buffer.Length;

    // ── Costruttori ─────────────────────────────────────────────────────────

    /// <summary>Crea un tensore di soli zeri con le dimensioni specificate.</summary>
    public DenseTensor(params int[] dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        if (dimensions.Length == 0)
            throw new ArgumentException("Almeno una dimensione è richiesta.", nameof(dimensions));

        Dimensions = (int[])dimensions.Clone();
        int size = 1;
        foreach (int d in dimensions)
        {
            if (d <= 0)
                throw new ArgumentOutOfRangeException(nameof(dimensions), $"Dimensione non valida: {d}");
            checked { size *= d; }
        }
        Buffer = new T[size];
    }

    /// <summary>Crea un tensore avvolgendo un buffer esistente.</summary>
    public DenseTensor(T[] buffer, params int[] dimensions)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentNullException.ThrowIfNull(dimensions);

        Dimensions = (int[])dimensions.Clone();
        int size = 1;
        foreach (int d in dimensions)
        {
            if (d <= 0)
                throw new ArgumentOutOfRangeException(nameof(dimensions));
            size *= d;
        }
        if (buffer.Length != size)
            throw new ArgumentException(
                $"Lunghezza buffer {buffer.Length} non corrisponde al prodotto delle dimensioni {size}.");
        Buffer = buffer;
    }

    // ── Indicizzatori ───────────────────────────────────────────────────────

    /// <summary>Accesso lineare (per vettori).</summary>
    public T this[int i]
    {
        get => Buffer[i];
        set => Buffer[i] = value;
    }

    /// <summary>Accesso 2D row-major (per matrici).</summary>
    public T this[int row, int col]
    {
        get => Buffer[row * Dimensions[1] + col];
        set => Buffer[row * Dimensions[1] + col] = value;
    }

    // ── Operazioni utili ────────────────────────────────────────────────────

    public DenseTensor<T> Clone()
    {
        var clone = new DenseTensor<T>(Dimensions);
        Buffer.CopyTo(clone.Buffer, 0);
        return clone;
    }

    public void CopyTo(DenseTensor<T> target)
    {
        ShapeCheck.SameShape(this, target);
        Buffer.CopyTo(target.Buffer, 0);
    }

    /// <summary>Azzera tutti gli elementi in-place.</summary>
    public void Zero() => Array.Clear(Buffer, 0, Buffer.Length);

    /// <summary>Riempie tutti gli elementi con il valore specificato.</summary>
    public void Fill(T value) => Array.Fill(Buffer, value);

    // ── Factory methods ─────────────────────────────────────────────────────
    public static DenseTensor<T> Zeros(params int[] dims) => new(dims);

    public static DenseTensor<T> Like(DenseTensor<T> other) => new(other.Dimensions);

    public override string ToString() =>
        $"DenseTensor<{typeof(T).Name}>[{string.Join(", ", Dimensions)}]  L={Length}";
}


// ─────────────────────────────────────────────────────────────────────────────
// ShapeCheck
//
// Classe di validazione delle forme. Lancia ArgumentException con messaggi
// descrittivi se le forme non corrispondono, facilitando il debug.
// ─────────────────────────────────────────────────────────────────────────────

public static class ShapeCheck
{
    public static void SameShape<T>(DenseTensor<T> a, DenseTensor<T> b, string? nameB = null)
        where T : struct, INumber<T>
    {
        if (!a.Dimensions.AsSpan().SequenceEqual(b.Dimensions))
            throw new InvalidOperationException(
                $"Mismatch di forma: {a} vs {b} ({nameB ?? "b"})");
    }

    /// <summary>
    /// Verifica che la matrice W abbia forma [rows, cols] attesa.
    /// </summary>
    public static void Matrix(DenseTensor<float> w, int rows, int cols, string name = "W")
    {
        if (w.Rank != 2 || w.Dimensions[0] != rows || w.Dimensions[1] != cols)
            throw new InvalidOperationException(
                $"Matrice '{name}' attesa [{rows},{cols}], trovata {w}");
    }

    /// <summary>
    /// Verifica che il vettore v abbia lunghezza size.
    /// </summary>
    public static void Vector(DenseTensor<float> v, int size, string name = "v")
    {
        if (v.Rank != 1 || v.Dimensions[0] != size)
            throw new InvalidOperationException(
                $"Vettore '{name}' atteso [{size}], trovato {v}");
    }

    /// <summary>
    /// Verifica che due vettori abbiano la stessa dimensione.
    /// </summary>
    public static void SameLength(DenseTensor<float> a, DenseTensor<float> b,
        string nameA = "a", string nameB = "b")
    {
        if (a.Length != b.Length)
            throw new InvalidOperationException(
                $"Lunghezze diverse: '{nameA}'={a.Length}, '{nameB}'={b.Length}");
    }
}
