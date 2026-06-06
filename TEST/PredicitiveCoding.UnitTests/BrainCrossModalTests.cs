using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for cross-modal voting into a shared object id (CrossModalSystem).</summary>
public class BrainCrossModalTests
{
    [Fact]
    public void CrossModalSystem_HeterogeneousModalities_RecogniseSharedObject()
    {
        // Two modalities with DIFFERENT feature sizes (vision 100-d, touch 40-d) that
        // share a location space and a common object-id space.
        const int visionSize = 100, touchSize = 40, locSize = 80;
        const int nObjects = 4, landmarks = 6;
        var rng = new Random(11);

        var sharedLocations = Enumerable.Range(0, landmarks)
            .Select(_ => Sdr.Random(locSize, 8, rng)).ToList();

        // Per object: a vision feature and a touch feature at each shared location.
        var vision = new List<(Sdr, Sdr)>[nObjects];
        var touch = new List<(Sdr, Sdr)>[nObjects];
        for (int o = 0; o < nObjects; o++)
        {
            vision[o] = []; touch[o] = [];
            for (int l = 0; l < landmarks; l++)
            {
                vision[o].Add((Sdr.Random(visionSize, 20, rng), sharedLocations[l]));
                touch[o].Add((Sdr.Random(touchSize, 8, rng), sharedLocations[l]));
            }
        }

        var sys = new CrossModalSystem(
        [
            new CorticalColumnConfig { InputSize = visionSize, Columns = 128, CellsPerColumn = 4, Sparsity = 0.05f, MinOverlap = 3, Seed = 1 },
            new CorticalColumnConfig { InputSize = touchSize,  Columns = 128, CellsPerColumn = 4, Sparsity = 0.05f, MinOverlap = 3, Seed = 2 },
        ]);

        for (int o = 0; o < nObjects; o++)
            sys.LearnObject([vision[o], touch[o]]);

        Assert.Equal(2, sys.ModalityCount);
        Assert.Equal(nObjects, sys.ObjectCount);

        // Recognise object 2 by voting vision+touch over its landmarks.
        const int target = 2;
        sys.Reset();
        ConsensusResult? consensus = null;
        for (int l = 0; l < landmarks; l++)
        {
            consensus = sys.VoteStep([vision[target][l], touch[target][l]]);
            if (consensus.Candidates.Count == 1) break;
        }

        Assert.NotNull(consensus);
        Assert.Contains(target, consensus!.Candidates);
        Assert.True(consensus.Candidates.Count < nObjects);
    }

    [Fact]
    public void CrossModalSystem_RequiresAtLeastTwoModalities()
    {
        Assert.Throws<ArgumentException>(() => new CrossModalSystem(
        [
            new CorticalColumnConfig { InputSize = 40, Columns = 64 },
        ]));
    }
}
