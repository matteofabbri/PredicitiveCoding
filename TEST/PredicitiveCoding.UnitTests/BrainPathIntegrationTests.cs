using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for motor-driven grid-cell path integration (PathIntegrator).</summary>
public class BrainPathIntegrationTests
{
    private static PoseEncoder Encoder() => new(-10f, 10f, -10f, 10f,
        positionBits: 100, positionActiveBits: 10, orientationBuckets: 8, orientationActiveBits: 5);

    [Fact]
    public void Step_AccumulatesWorldFrameDisplacement()
    {
        var pi = new PathIntegrator(Encoder());
        pi.Step(1f, 0f);
        pi.Step(1f, 0f);
        pi.Step(1f, 2f);

        Assert.Equal(3f, pi.X, 1e-4f);
        Assert.Equal(2f, pi.Y, 1e-4f);
    }

    [Fact]
    public void ClosedLoop_ReturnsToStartingLocation()
    {
        var pi = new PathIntegrator(Encoder());

        // Walk a 5×5 square: forward, turn 90°, ×4.
        pi.StepForward(5f); pi.Turn(90f);
        pi.StepForward(5f); pi.Turn(90f);
        pi.StepForward(5f); pi.Turn(90f);
        pi.StepForward(5f);

        Assert.Equal(0f, pi.X, 1e-3f);
        Assert.Equal(0f, pi.Y, 1e-3f);
    }

    [Fact]
    public void LocationCode_HasEncoderSize_AndNearbyPosesOverlapMore()
    {
        var enc = Encoder();
        var pi = new PathIntegrator(enc);

        var atOrigin = pi.LocationCode;
        Assert.Equal(enc.Size, atOrigin.Size);

        pi.Reset(0.3f, 0f);
        var near = pi.LocationCode;
        pi.Reset(9f, 9f);
        var far = pi.LocationCode;

        Assert.True(atOrigin.Overlap(near) > atOrigin.Overlap(far));
    }

    [Fact]
    public void Turn_NormalisesHeadingIntoZeroTo360()
    {
        var pi = new PathIntegrator(Encoder());
        pi.Turn(370f);
        Assert.Equal(10f, pi.HeadingDeg, 1e-4f);

        pi.Turn(-20f);
        Assert.Equal(350f, pi.HeadingDeg, 1e-4f);
    }
}
