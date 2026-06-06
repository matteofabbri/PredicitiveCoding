namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// SpatialPooler  —  HTM spatial pooler.
//
// Maps an input SDR to a fixed-sparsity output SDR over a population of columns.
// Each column has a fixed random set of *potential* synapses onto the input, each
// with a permanence; a synapse is *connected* when its permanence ≥ threshold.
// A column's overlap is the number of connected synapses whose input bit is on.
// The top-k columns by overlap win (k-winners-take-all → stable sparsity), and
// Hebbian learning nudges the winners' permanences toward the current input.
//
// Reference: Hawkins & Ahmad (2016); Cui, Ahmad & Hawkins (2017) "HTM Spatial Pooler".
// ─────────────────────────────────────────────────────────────────────────────

public sealed record SpatialPoolerConfig
{
    /// <summary>Number of input bits (must match the encoder output size).</summary>
    public required int InputSize { get; init; }

    /// <summary>Number of columns (output bits).</summary>
    public required int Columns { get; init; }

    /// <summary>Target number of active (winning) columns. 0 ⇒ ~2% of <see cref="Columns"/>.</summary>
    public int ActiveColumns { get; init; }

    /// <summary>Fraction of the input each column can potentially connect to.</summary>
    public float PotentialPct { get; init; } = 0.5f;

    public float ConnectedPermanence { get; init; } = 0.5f;
    public float PermanenceIncrement { get; init; } = 0.05f;
    public float PermanenceDecrement { get; init; } = 0.02f;

    /// <summary>A column must have at least this overlap to be eligible to win.</summary>
    public int StimulusThreshold { get; init; } = 1;

    public int Seed { get; init; } = 42;
}

public sealed class SpatialPooler
{
    private readonly SpatialPoolerConfig _cfg;
    private readonly int[][] _potential;   // column → potential input bits
    private readonly float[][] _perm;      // column → permanences (aligned with _potential)
    private readonly int _activeTarget;

    public int Columns => _cfg.Columns;
    public int InputSize => _cfg.InputSize;
    public int ActiveColumns => _activeTarget;

    public SpatialPooler(SpatialPoolerConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        if (cfg.InputSize <= 0 || cfg.Columns <= 0)
            throw new ArgumentException("InputSize and Columns must be positive.");
        _cfg = cfg;
        _activeTarget = cfg.ActiveColumns > 0
            ? Math.Min(cfg.ActiveColumns, cfg.Columns)
            : Math.Max(1, (int)Math.Round(cfg.Columns * 0.02));

        var rng = new Random(cfg.Seed);
        int potentialCount = Math.Clamp((int)Math.Round(cfg.PotentialPct * cfg.InputSize), 1, cfg.InputSize);

        _potential = new int[cfg.Columns][];
        _perm = new float[cfg.Columns][];
        for (int c = 0; c < cfg.Columns; c++)
        {
            int[] bits = SampleDistinct(cfg.InputSize, potentialCount, rng);
            var perms = new float[bits.Length];
            for (int i = 0; i < bits.Length; i++)
            {
                // Initialise close to the connection threshold so learning can push
                // synapses across it in either direction.
                float p = cfg.ConnectedPermanence + (float)(rng.NextDouble() * 0.2 - 0.1);
                perms[i] = Math.Clamp(p, 0f, 1f);
            }
            _potential[c] = bits;
            _perm[c] = perms;
        }
    }

    /// <summary>Computes the active columns for <paramref name="input"/>, optionally learning.</summary>
    public Sdr Compute(Sdr input, bool learn)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Size != _cfg.InputSize)
            throw new ArgumentException($"Input size {input.Size} != InputSize {_cfg.InputSize}.", nameof(input));

        var overlaps = new int[_cfg.Columns];
        for (int c = 0; c < _cfg.Columns; c++)
        {
            int[] pot = _potential[c];
            float[] pm = _perm[c];
            int ov = 0;
            for (int i = 0; i < pot.Length; i++)
                if (pm[i] >= _cfg.ConnectedPermanence && input.IsActive(pot[i])) ov++;
            overlaps[c] = ov;
        }

        int[] winners = TopColumns(overlaps, _activeTarget, _cfg.StimulusThreshold);

        if (learn)
        {
            foreach (int c in winners)
            {
                int[] pot = _potential[c];
                float[] pm = _perm[c];
                for (int i = 0; i < pot.Length; i++)
                {
                    pm[i] = input.IsActive(pot[i])
                        ? Math.Clamp(pm[i] + _cfg.PermanenceIncrement, 0f, 1f)
                        : Math.Clamp(pm[i] - _cfg.PermanenceDecrement, 0f, 1f);
                }
            }
        }

        return new Sdr(_cfg.Columns, winners);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static int[] TopColumns(int[] overlaps, int k, int stimulusThreshold)
    {
        var eligible = new List<int>();
        for (int c = 0; c < overlaps.Length; c++)
            if (overlaps[c] >= stimulusThreshold) eligible.Add(c);

        eligible.Sort((a, b) =>
        {
            int cmp = overlaps[b].CompareTo(overlaps[a]);   // higher overlap first
            return cmp != 0 ? cmp : a.CompareTo(b);          // ties: lower index first
        });

        int take = Math.Min(k, eligible.Count);
        var result = new int[take];
        for (int i = 0; i < take; i++) result[i] = eligible[i];
        return result;
    }

    private static int[] SampleDistinct(int range, int count, Random rng)
    {
        count = Math.Min(count, range);
        var set = new HashSet<int>(count);
        while (set.Count < count) set.Add(rng.Next(range));
        var arr = new int[set.Count];
        set.CopyTo(arr);
        Array.Sort(arr);
        return arr;
    }
}
