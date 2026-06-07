using PredictiveCoding.Core;
using PredictiveCoding.Cortex;

namespace PredicitiveCoding.UnitTests;

// Group G — subcortical sensory / attention: InferiorColliculus, SuperiorColliculus, Pulvinar.
public class CortexSubcorticalTests
{
    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    // ── InferiorColliculus ──────────────────────────────────────────────────────

    [Fact]
    public void InferiorColliculus_LateralInhibition_CancelsFlat_SharpensPeak()
    {
        var ic = new InferiorColliculus(channels: 6, surroundRadius: 1);

        var flat = ic.Process(Vec(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f));
        Assert.Equal(6, flat.Length);
        Assert.All(flat.Buffer, v => Assert.True(MathF.Abs(v) < 1e-3f));   // constant ⇒ ≈0

        var peak = ic.Process(Vec(0, 0, 0, 1, 0, 0));
        Assert.True(peak.Buffer[3] > 0f);                                  // the peak survives
        Assert.True(peak.Buffer[2] < 0f);                                  // flanks are inhibited

        Assert.Throws<ArgumentException>(() => ic.Process(Vec(1, 2, 3)));
    }

    // ── SuperiorColliculus ──────────────────────────────────────────────────────

    [Fact]
    public void SuperiorColliculus_FusesModalities_AndOrientsWithInhibitionOfReturn()
    {
        var sc = new SuperiorColliculus(cellCount: 4);
        float[] visual = [0.1f, 0.9f, 0.2f, 0.0f];
        float[] audio = [0.0f, 0.8f, 0.1f, 0.3f];

        var priority = sc.PriorityMap(visual, audio);
        Assert.Equal(4, priority.Length);
        Assert.Equal(1.7f, priority[1], 3);                                // 0.9 + 0.8

        Assert.Equal(1, sc.Orient(visual, audio, new HashSet<int>()));     // cell 1 is most salient

        var scan = sc.Scan(visual, audio, 4);
        Assert.Equal(4, scan.Count);
        Assert.Equal(4, scan.Distinct().Count());                          // IOR ⇒ all distinct
        Assert.Equal(1, scan[0]);
        Assert.NotEqual(1, sc.Orient(visual, audio, new HashSet<int> { 1 }));
    }

    // ── Pulvinar ────────────────────────────────────────────────────────────────

    [Fact]
    public void Pulvinar_AllocatesPrecisionBySalience_MeanStaysAtBase()
    {
        var pulvinar = new Pulvinar(temperature: 1f, basePrecision: 1f);

        var equal = pulvinar.Allocate([0.5f, 0.5f, 0.5f, 0.5f]);
        Assert.All(equal, p => Assert.Equal(1f, p, 3));                    // uniform ⇒ base each

        var alloc = pulvinar.Allocate([0f, 3f, 0f]);
        Assert.True(alloc[1] > alloc[0] && alloc[1] > alloc[2]);           // salient area wins
        Assert.Equal(1f, (alloc[0] + alloc[1] + alloc[2]) / 3f, 3);        // mean = base precision

        Assert.Empty(pulvinar.Allocate([]));
    }
}
