using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for the Thousand-Brains Phase 5: lateral voting across cortical columns.</summary>
public class BrainVotingTests
{
    // ── LateralVoting ────────────────────────────────────────────────────────

    [Fact]
    public void LateralVoting_AllToAll_EachColumnReceivesUnionOfAllOthers()
    {
        // 3 columns, each with its own 4-bit SDR (size=8, 4 active).
        var sdrs = new[]
        {
            new Sdr(8, [0, 1, 2, 3]),
            new Sdr(8, [2, 3, 4, 5]),
            new Sdr(8, [4, 5, 6, 7]),
        };
        var lateral = new LateralVoting(3, VotingTopology.AllToAll);
        var broadcast = lateral.Broadcast(sdrs);

        // Column 0 should receive union of columns 1 and 2 = {2,3,4,5,6,7}
        var expected0 = sdrs[1].Union(sdrs[2]);
        Assert.Equal(expected0, broadcast[0]);

        // Column 1 should receive union of columns 0 and 2 = {0,1,2,3,4,5,6,7}
        var expected1 = sdrs[0].Union(sdrs[2]);
        Assert.Equal(expected1, broadcast[1]);

        // Column 2 should receive union of columns 0 and 1 = {0,1,2,3,4,5}
        var expected2 = sdrs[0].Union(sdrs[1]);
        Assert.Equal(expected2, broadcast[2]);
    }

    [Fact]
    public void LateralVoting_Ring_EachColumnReceivesExactlyTwoNeighbours()
    {
        var lateral = new LateralVoting(5, VotingTopology.Ring);

        // Column 0: ring neighbours are 4 (left) and 1 (right)
        Assert.Equal(new[] { 1, 4 }, lateral.Neighbors(0).OrderBy(x => x).ToArray());

        // Column 2: ring neighbours are 1 and 3
        Assert.Equal(new[] { 1, 3 }, lateral.Neighbors(2).OrderBy(x => x).ToArray());

        // Column 4: ring neighbours are 3 and 0
        Assert.Equal(new[] { 0, 3 }, lateral.Neighbors(4).OrderBy(x => x).ToArray());
    }

    // ── Consensus ────────────────────────────────────────────────────────────

    [Fact]
    public void Consensus_IntersectionNarrowsToSharedCandidate()
    {
        // 3 columns, each unsure between {A,B}, {A,C}, {A,D}.
        // With unanimity threshold: only A (present in all three) survives.
        var activities = new[]
        {
            new ColumnActivity(0, Sdr.Empty(4), [0, 1], false, 1f),  // object 0 and 1
            new ColumnActivity(1, Sdr.Empty(4), [0, 2], false, 1f),  // object 0 and 2
            new ColumnActivity(2, Sdr.Empty(4), [0, 3], false, 1f),  // object 0 and 3
        };
        var consensus = new Consensus(minAgreementFraction: 1.0f);
        var result = consensus.Compute(activities, round: 1);

        Assert.True(result.IsReached);
        Assert.Equal([0], result.Candidates.ToArray());
        Assert.Equal(1.0f, result.AgreementScore, precision: 4);
        Assert.Equal(1, result.Round);
    }

    // ── MultiColumnSystem end-to-end ─────────────────────────────────────────

    [Fact]
    public void MultiColumnSystem_TwoColumns_ConvergeToSameObjectFasterThanSingleColumn()
    {
        // Setup: 3 objects, each with 6 (location, feature) observations.
        // Two columns each learn the same 3 objects but observe different points.
        // During inference, each column sees 1 observation → alone: 3 candidates.
        // With voting (intersection) → should converge to 1.

        var rng = new Random(99);
        const int inputSize = 100;
        const int columns = 64;
        const int locationSize = 80;
        const int activeLoc = 8;

        var cfg = new CorticalColumnConfig
        {
            InputSize = inputSize,
            Columns = columns,
            CellsPerColumn = 4,
            Sparsity = 0.05f,
            MinOverlap = 2,
            Seed = 7,
        };

        // Build three distinct objects: each object has 6 unique feature/location pairs.
        const int nObjects = 3;
        const int pointsPerObject = 6;

        var objectPoints = new List<(Sdr feature, Sdr location)>[nObjects];
        for (int o = 0; o < nObjects; o++)
        {
            objectPoints[o] = [];
            for (int p = 0; p < pointsPerObject; p++)
            {
                var feat = Sdr.Random(inputSize, 20, rng);
                var loc = Sdr.Random(locationSize, activeLoc, rng);
                objectPoints[o].Add((feat, loc));
            }
        }

        // Column 0 learns points 0-2 of each object; column 1 learns points 3-5.
        var sys = new MultiColumnSystem(
            [cfg with { Seed = 7 }, cfg with { Seed = 8 }],
            VotingTopology.AllToAll,
            minAgreementFraction: 0.5f);

        for (int o = 0; o < nObjects; o++)
        {
            var obs0 = objectPoints[o].Take(3).ToList();
            var obs1 = objectPoints[o].Skip(3).Take(3).ToList();
            sys.LearnObject([obs0, obs1]);
        }

        Assert.Equal(nObjects, sys.ObjectCount);

        // Inference: each column observes only the FIRST point of object 1.
        // A single column alone with 1 observation may not converge (all 3 objects are candidates).
        // After one VoteStep, the intersection should identify object 1.
        sys.Reset();
        var inferInput = new (Sdr Feature, Sdr Location)[]
        {
            objectPoints[1][0],   // col0 sees point 0 of obj 1
            objectPoints[1][3],   // col1 sees point 3 of obj 1
        };
        var consensus = sys.VoteStep(inferInput);

        // The voting consensus must include object 1.
        Assert.Contains(1, consensus.Candidates);
        // The consensus must have narrowed down (not all 3 objects).
        Assert.True(consensus.Candidates.Count < nObjects,
            $"Expected fewer than {nObjects} candidates but got {consensus.Candidates.Count}.");
    }
}
