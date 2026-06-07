using PredictiveCoding.Cortex;

namespace PredicitiveCoding.UnitTests;

// Group G — reward / aversion / defence: NucleusAccumbens, Habenula, PeriaqueductalGray.
public class CortexRewardTests
{
    // ── NucleusAccumbens ────────────────────────────────────────────────────────

    [Fact]
    public void NucleusAccumbens_SelectsHighestValue_DopamineSharpensGating()
    {
        var nacc = new NucleusAccumbens(temperature: 1f);
        float[] values = [0.2f, 0.8f];

        Assert.Equal(1, nacc.Select(values, dopamine: 0f));

        var lowDopamine = nacc.GateWeights(values, dopamine: 0f);
        var highDopamine = nacc.GateWeights(values, dopamine: 2f);
        Assert.Equal(1f, lowDopamine[0] + lowDopamine[1], 3);              // weights sum to 1
        Assert.True(highDopamine[1] > lowDopamine[1]);                     // dopamine ⇒ more peaked
    }

    // ── Habenula ────────────────────────────────────────────────────────────────

    [Fact]
    public void Habenula_SuppressesDopamineOnDisappointment_NotOnGoodOutcome()
    {
        var habenula = new Habenula(suppressionGain: 1f);

        Assert.Equal(0.4f, habenula.NegativePredictionError(reward: 0.2f, expected: 0.6f), 3);
        Assert.Equal(0f, habenula.NegativePredictionError(reward: 1.0f, expected: 0.6f), 3);   // better ⇒ 0

        Assert.Equal(0.6f, habenula.SuppressDopamine(dopamine: 1f, reward: 0.2f, expected: 0.6f), 3); // 1 − 0.4
        Assert.Equal(1.0f, habenula.SuppressDopamine(dopamine: 1f, reward: 0.9f, expected: 0.6f), 3); // unchanged
    }

    // ── PeriaqueductalGray ──────────────────────────────────────────────────────

    [Fact]
    public void PeriaqueductalGray_SelectsDefenceByThreatImminence()
    {
        var pag = new PeriaqueductalGray(threatThreshold: 0.1f);

        Assert.Equal(DefensiveResponse.None, pag.Respond(threat: 0.05f, proximity: 1f));   // sub-threshold
        Assert.Equal(DefensiveResponse.Freeze, pag.Respond(threat: 0.8f, proximity: 0.1f)); // distal
        Assert.Equal(DefensiveResponse.Flight, pag.Respond(threat: 0.8f, proximity: 0.5f)); // intermediate
        Assert.Equal(DefensiveResponse.Fight, pag.Respond(threat: 0.8f, proximity: 0.95f)); // contact
    }
}
