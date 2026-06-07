using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Group A — Neuromodulation: DopaminergicSystem, LocusCoeruleus, NucleusBasalis, RapheNuclei.
// Pure controllers (no persistence) except the one Network-integration test for LC.Modulate.
public class CortexNeuromodulationTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"neuromod_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    // ── DopaminergicSystem (VTA / SNc) ──────────────────────────────────────────

    [Fact]
    public void DopaminergicSystem_Rpe_TerminalAndDiscountedForms()
    {
        var da = new DopaminergicSystem(discount: 0.9f);
        Assert.True(da.RewardPredictionError(reward: 1f, valueCurrent: 0f) > 0f);       // terminal δ = r − V
        Assert.Equal(1f + 0.9f * 0.5f - 0.2f, da.RewardPredictionError(1f, 0.2f, 0.5f), 4);  // δ = r + γ·V′ − V
    }

    [Fact]
    public void DopaminergicSystem_BaselineTracksAverageReward_AndRpeShrinks()
    {
        var da = new DopaminergicSystem(baselineLearningRate: 0.5f);
        float first = da.Step(1f);            // δ = 1 − 0 = 1; baseline → 0.5
        Assert.Equal(1f, first, 4);
        Assert.Equal(0.5f, da.Baseline, 4);
        Assert.True(da.Step(1f) < first);     // δ = 1 − 0.5 = 0.5 < 1
    }

    [Fact]
    public void DopaminergicSystem_ModulateLearningRate_ScalesWithSurprise()
    {
        var da = new DopaminergicSystem(tonicLevel: 0.1f);
        Assert.True(da.ModulateLearningRate(0.01f, rpe: 2.0f) > da.ModulateLearningRate(0.01f, rpe: 0.1f));
        Assert.Equal(0.01f * 0.1f, da.ModulateLearningRate(0.01f, 0f), 6);   // zero surprise → tonic floor
    }

    [Fact]
    public void DopaminergicSystem_RejectsBadDiscount()
        => Assert.Throws<ArgumentOutOfRangeException>(() => new DopaminergicSystem(discount: 1.5f));

    // ── LocusCoeruleus (noradrenaline) ──────────────────────────────────────────

    [Fact]
    public void LocusCoeruleus_FirstObservationSeedsBaseline_NoBurst()
    {
        var lc = new LocusCoeruleus();
        var (burst, gain) = lc.Process(10f);
        Assert.False(burst);
        Assert.Equal(lc.TonicGain, gain, 3);
        Assert.Equal(10f, lc.ExpectedSurprise, 3);
    }

    [Fact]
    public void LocusCoeruleus_SurpriseSpikeFiresPhasicBurst()
    {
        var lc = new LocusCoeruleus(adaptation: 0.1f, phasicThreshold: 1.5f, phasicGain: 4f, tonicGain: 1f);
        lc.Process(10f);                          // seed baseline at 10
        var (burst, gain) = lc.Process(100f);     // 100 > 1.5·10 ⇒ burst
        Assert.True(burst);
        Assert.Equal(4f, gain, 3);
    }

    [Fact]
    public void LocusCoeruleus_SteadySurpriseDoesNotBurst()
    {
        var lc = new LocusCoeruleus();
        lc.Process(10f);
        var (burst, gain) = lc.Process(11f);      // within threshold
        Assert.False(burst);
        Assert.Equal(lc.TonicGain, gain, 3);
    }

    [Fact]
    public void LocusCoeruleus_RejectsBadThreshold()
        => Assert.Throws<ArgumentOutOfRangeException>(() => new LocusCoeruleus(phasicThreshold: 0.9f));

    [Fact]
    public void LocusCoeruleus_Modulate_ReadsEnergyAndWritesGainAcrossRealNetworks()
    {
        using var a0 = new Network(new NetworkConfig { LayerSizes = [4, 2], Path = Path.Combine(_tmp, "a0") }, new Random(0));
        using var a1 = new Network(new NetworkConfig { LayerSizes = [4, 2], Path = Path.Combine(_tmp, "a1") }, new Random(1));
        var areas = new[] { a0, a1 };

        Assert.True(LocusCoeruleus.GlobalSurprise(areas) >= 0f);

        var lc   = new LocusCoeruleus();
        float g  = lc.Modulate(areas);            // first call seeds baseline → tonic gain (no burst)
        Assert.Equal(lc.TonicGain, g, 3);
    }

    // ── NucleusBasalis (acetylcholine) ──────────────────────────────────────────

    [Fact]
    public void NucleusBasalis_HigherEnergyAreaGetsMoreAcetylcholine()
    {
        var nb  = new NucleusBasalis(areaCount: 2, gain: 1f);
        var ach = nb.Update([1f, 5f]);            // first call seeds EMA directly
        Assert.Equal(2, ach.Length);
        Assert.True(ach[1] > ach[0]);
        Assert.Equal(5f, nb.ExpectedUncertainty[1], 3);
    }

    [Fact]
    public void NucleusBasalis_ModulateLearningRates_HigherForUncertainArea()
    {
        var nb    = new NucleusBasalis(areaCount: 2, gain: 1f);
        var rates = nb.ModulateLearningRates(0.01f, [1f, 5f]);
        Assert.True(rates[1] > rates[0]);
        Assert.Equal(0.01f * (1f + 1f), rates[0], 5);   // η = base·(1 + ACh); ACh[0] = gain·1 = 1
    }

    [Fact]
    public void NucleusBasalis_EmaSmoothsAcrossUpdates()
    {
        var nb = new NucleusBasalis(areaCount: 1, adaptation: 0.5f, gain: 1f);
        nb.Update([0f]);                          // seed at 0
        nb.Update([10f]);                         // EMA: 0 + 0.5·(10 − 0) = 5
        Assert.Equal(5f, nb.ExpectedUncertainty[0], 3);
    }

    [Fact]
    public void NucleusBasalis_RejectsWrongLength()
        => Assert.Throws<ArgumentException>(() => new NucleusBasalis(2).Update([1f]));

    // ── RapheNuclei (serotonin) ─────────────────────────────────────────────────

    [Fact]
    public void RapheNuclei_LevelClampsToUnitInterval()
    {
        var r = new RapheNuclei();
        r.Level = 2f;  Assert.Equal(1f, r.Level, 3);
        r.Level = -1f; Assert.Equal(0f, r.Level, 3);
    }

    [Fact]
    public void RapheNuclei_DiscountFactorAndHorizonIncreaseWithSerotonin()
    {
        var lo = new RapheNuclei(0f);
        var hi = new RapheNuclei(1f);
        Assert.Equal(0.5f,  lo.DiscountFactor(0.5f, 0.99f), 3);
        Assert.Equal(0.99f, hi.DiscountFactor(0.5f, 0.99f), 3);
        Assert.True(hi.TimeHorizon() > lo.TimeHorizon());
    }
}
