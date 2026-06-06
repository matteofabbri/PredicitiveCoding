using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for the Thousand-Brains Phase 2 sequence memory: Spatial Pooler and Temporal Memory.</summary>
public class BrainSequenceTests
{
    // ── Spatial Pooler ───────────────────────────────────────────────────────

    [Fact]
    public void SpatialPooler_FixedSparsity_Deterministic_Discriminative()
    {
        var sp = new SpatialPooler(new SpatialPoolerConfig
        {
            InputSize = 200,
            Columns = 128,
            ActiveColumns = 10,
            Seed = 1,
        });

        var input = Sdr.Random(200, 40, new Random(5));
        var o1 = sp.Compute(input, learn: false);
        var o2 = sp.Compute(input, learn: false);

        Assert.Equal(10, o1.Cardinality);     // k-winners-take-all
        Assert.Equal(o1, o2);                  // deterministic without learning

        var other = Sdr.Random(200, 40, new Random(6));
        Assert.NotEqual(o1, sp.Compute(other, learn: false));   // different input → different code

        for (int i = 0; i < 30; i++) sp.Compute(input, learn: true);
        Assert.Equal(10, sp.Compute(input, learn: false).Cardinality);   // still a valid sparse code
    }

    // ── Temporal Memory ──────────────────────────────────────────────────────

    [Fact]
    public void TemporalMemory_NovelInput_BurstsWithFullAnomaly()
    {
        var tm = new TemporalMemory(new TemporalMemoryConfig
        {
            Columns = 32,
            CellsPerColumn = 4,
            ActivationThreshold = 2,
            MinThreshold = 1,
            MaxNewSynapseCount = 8,
            Seed = 1,
        });

        var s = Sdr.Random(32, 8, new Random(2));
        var r = tm.Compute(s, learn: true);

        Assert.Equal(1.0f, r.Anomaly);                 // nothing was predicted
        Assert.Equal(8 * 4, r.ActiveCells.Count);      // every cell of every active column bursts
        Assert.Empty(r.PredictiveCells);               // no segments yet
    }

    [Fact]
    public void TemporalMemory_LearnsRepeatingSequence_AnomalyFalls_AndPredictsNext()
    {
        const int columns = 64, cells = 8, w = 12;
        var tm = new TemporalMemory(new TemporalMemoryConfig
        {
            Columns = columns,
            CellsPerColumn = cells,
            ActivationThreshold = 3,
            MinThreshold = 2,
            ConnectedPermanence = 0.5f,
            InitialPermanence = 0.35f,
            PermanenceIncrement = 0.15f,
            PermanenceDecrement = 0.02f,
            MaxNewSynapseCount = w,
            Seed = 1,
        });

        var rng = new Random(42);
        Sdr[] sequence =
        [
            Sdr.Random(columns, w, rng),
            Sdr.Random(columns, w, rng),
            Sdr.Random(columns, w, rng),
            Sdr.Random(columns, w, rng),
        ];

        const int reps = 30;
        var anomalies = new List<float>();
        for (int rep = 0; rep < reps; rep++)
            foreach (var element in sequence)
                anomalies.Add(tm.Compute(element, learn: true).Anomaly);

        Assert.Equal(1.0f, anomalies[0]);   // first ever element is a surprise

        float lastCycle = anomalies.Skip(anomalies.Count - sequence.Length).Average();
        Assert.True(lastCycle < 0.3f, $"sequence not learned: last-cycle anomaly {lastCycle:F2}");

        // After learning, presenting S0 (from a clean context) predicts S1's columns.
        tm.Reset();
        var prediction = tm.Compute(sequence[0], learn: false);
        var predictedColumns = new HashSet<int>(prediction.PredictiveCells.Select(c => c / cells));
        int hits = sequence[1].ActiveBits.Count(predictedColumns.Contains);
        Assert.True(hits >= (int)(0.8 * w), $"predicted only {hits}/{w} of the next element's columns");
    }

    [Fact]
    public void SpatialPooler_To_TemporalMemory_PipelineRuns()
    {
        var encoder = new ScalarEncoder(0, 10, n: 120, w: 21);
        var sp = new SpatialPooler(new SpatialPoolerConfig { InputSize = 120, Columns = 128, ActiveColumns = 12, Seed = 3 });
        var tm = new TemporalMemory(new TemporalMemoryConfig
        {
            Columns = 128,
            CellsPerColumn = 8,
            ActivationThreshold = 3,
            MinThreshold = 2,
            MaxNewSynapseCount = 12,
            Seed = 3,
        });

        float firstAnomaly = 0f, lastAnomaly = 0f;
        for (int rep = 0; rep < 20; rep++)
        {
            for (int v = 0; v < 5; v++)
            {
                var cols = sp.Compute(encoder.Encode(v), learn: true);
                var r = tm.Compute(cols, learn: true);
                if (rep == 0 && v == 0) firstAnomaly = r.Anomaly;
                lastAnomaly = r.Anomaly;
            }
        }

        Assert.Equal(1.0f, firstAnomaly);
        Assert.True(lastAnomaly < firstAnomaly, "temporal memory did not learn the encoded scalar sequence");
    }
}
