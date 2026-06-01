namespace PredictiveCoding.Core;

// ─────────────────────────────────────────────────────────────────────────────
// RandomUtils
//
// Inizializzazione casuale di tensori per il framework di predictive coding.
// Usa Box-Muller per campionare da distribuzioni normali.
//
// Xavier/Glorot: appropriato per Tanh e Sigmoid (mantiene varianza della
//   segnale attraverso i layer a init).
// He: appropriato per ReLU (compensa il dimezzamento della varianza).
// ─────────────────────────────────────────────────────────────────────────────

public static class RandomUtils
{
    // ── Campionamento normale (Box-Muller) ───────────────────────────────────

    /// <summary>
    /// Campiona un singolo valore da N(mean, std²) tramite trasformata Box-Muller.
    /// </summary>
    public static float SampleNormal(float mean, float std, Random rng)
    {
        // Trasformata di Box-Muller: z = sqrt(-2 ln u1) * cos(2π u2)
        double u1 = 1.0 - rng.NextDouble();  // (0, 1]
        double u2 = rng.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + std * (float)z;
    }

    /// <summary>
    /// Crea un tensore riempito con campioni da N(mean, std²).
    /// </summary>
    public static DenseTensor<float> SampleNormal(int[] shape, float mean, float std, Random rng)
    {
        var t = new DenseTensor<float>(shape);
        for (int i = 0; i < t.Length; i++)
            t.Buffer[i] = SampleNormal(mean, std, rng);
        return t;
    }

    // ── Inizializzazione Xavier/Glorot ───────────────────────────────────────

    /// <summary>
    /// Inizializzazione di Glorot (Xavier) per una matrice [fanOut, fanIn]:
    ///   W ~ U[-limit, +limit]   con limit = sqrt(6 / (fanIn + fanOut))
    ///
    /// Riferimento: Glorot & Bengio (2010) — "Understanding the difficulty of
    /// training deep feedforward neural networks"
    /// </summary>
    public static DenseTensor<float> XavierUniform(int fanIn, int fanOut, Random rng)
    {
        float limit = MathF.Sqrt(6f / (fanIn + fanOut));
        var w = new DenseTensor<float>(fanOut, fanIn);
        for (int i = 0; i < w.Length; i++)
            w.Buffer[i] = (float)(rng.NextDouble() * 2 * limit - limit);
        return w;
    }

    /// <summary>
    /// Inizializzazione di Glorot (Xavier) Normale:
    ///   W ~ N(0, σ²)   con σ = sqrt(2 / (fanIn + fanOut))
    /// </summary>
    public static DenseTensor<float> XavierNormal(int fanIn, int fanOut, Random rng)
    {
        float std = MathF.Sqrt(2f / (fanIn + fanOut));
        return SampleNormal([fanOut, fanIn], 0f, std, rng);
    }

    // ── Inizializzazione He ───────────────────────────────────────────────────

    /// <summary>
    /// Inizializzazione He per ReLU:
    ///   W ~ N(0, 2/fanIn)
    ///
    /// Riferimento: He et al. (2015) — "Delving Deep into Rectifiers"
    /// </summary>
    public static DenseTensor<float> HeNormal(int fanIn, int fanOut, Random rng)
    {
        float std = MathF.Sqrt(2f / fanIn);
        return SampleNormal([fanOut, fanIn], 0f, std, rng);
    }

    // ── Factory unificata ─────────────────────────────────────────────────────

    /// <summary>
    /// Sceglie automaticamente l'inizializzazione appropriata in base all'attivazione.
    /// </summary>
    public static DenseTensor<float> InitWeights(int fanIn, int fanOut,
        ActivationType activation, Random rng)
    {
        return activation switch
        {
            ActivationType.ReLU    => HeNormal(fanIn, fanOut, rng),
            ActivationType.Linear  => XavierUniform(fanIn, fanOut, rng),
            ActivationType.Tanh    => XavierNormal(fanIn, fanOut, rng),
            ActivationType.Sigmoid => XavierNormal(fanIn, fanOut, rng),
            _ => XavierNormal(fanIn, fanOut, rng)
        };
    }

    /// <summary>
    /// Crea un vettore di bias inizializzato a zero (pratica standard).
    /// </summary>
    public static DenseTensor<float> ZeroBias(int size) => DenseTensor<float>.Zeros(size);

    /// <summary>
    /// Shuffle in-place di un array di indici (Fisher-Yates).
    /// </summary>
    public static void Shuffle<T>(T[] array, Random rng)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
