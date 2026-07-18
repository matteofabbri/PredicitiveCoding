using System.Diagnostics;
using PredictiveCoding.Brain;
using PredictiveCoding.Brain.Viz;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// ThousandBrainsDemo  —  Phase 9 evaluation & visual demo for PredictiveCoding.Brain.
//
// Four parts:
//   1. Recognition benchmark — teaches a set of synthetic objects to a single
//      cortical column and to a population of voting columns, then reports top-1
//      accuracy and mean steps-to-recognise under clean / occlusion / sensor-noise
//      / pose-jitter conditions (RecognitionBenchmark).
//   2. Live sensing & voting — narrates one recognition episode step by step,
//      showing a single column's hypothesis set narrowing over sensorimotor steps
//      versus a voting population converging in far fewer steps.
//   3. Pose-invariant recognition — absolute location matching vs displacement
//      structure after a rigid transform.
//   4. Image sensor scan — a real-image sensor patch scanning a procedural
//      texture, recording the raw descriptor at every fixation.
//
// Each part also records its data into a VizData; the demo ends by writing a
// self-contained web page (HtmlVisualizer) and opening it in the browser.
// Together these demonstrate the central Thousand-Brains claim: many columns that
// each model objects in their own reference frame and vote recognise objects
// rapidly and robustly (Leadholm et al. 2025; Hawkins 2021).
// ─────────────────────────────────────────────────────────────────────────────

public static class ThousandBrainsDemo
{
    public static void Run()
    {
        Console.WriteLine();
        Console.WriteLine("╔══════════════════════════════════════════════════════╗");
        Console.WriteLine("║   Thousand Brains  ·  sensorimotor object recognition ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════╝");
        Console.WriteLine();

        var benchmark = RunBenchmark();
        Console.WriteLine();
        var episode = RunLiveRecognition();
        Console.WriteLine();
        var poseInvariance = RunPoseInvariance();
        Console.WriteLine();
        var imageScan = RunImageScan();

        var data = new VizData
        {
            Benchmark = benchmark.Conditions
                .Select(c => new VizBenchmarkRow(
                    c.Condition,
                    c.SingleColumn.Top1Accuracy, c.SingleColumn.MeanSteps,
                    c.MultiColumn.Top1Accuracy, c.MultiColumn.MeanSteps))
                .ToList(),
            Episode        = episode,
            PoseInvariance = poseInvariance,
            ImageScan      = imageScan,
        };
        SaveVisualization(data);
    }

    // ── Part 1: benchmark ─────────────────────────────────────────────────────

    private static BenchmarkResult RunBenchmark()
    {
        var cfg = new BenchmarkConfig
        {
            ObjectCount           = 10,
            LocationsPerObject    = 12,
            FeatureVocabularySize = 5,
            Columns               = 4,
            ProbesPerObject       = 3,
            Seed                  = 42,
        };

        Console.WriteLine($"  Dataset : {cfg.ObjectCount} objects · {cfg.LocationsPerObject} landmarks each ·");
        Console.WriteLine($"            shared vocabulary of {cfg.FeatureVocabularySize} features (ambiguous landmarks)");
        Console.WriteLine($"  Models  : 1 cortical column   vs   {cfg.Columns} columns voting (unanimous consensus)");
        Console.WriteLine();
        Console.WriteLine("  Running benchmark …");

        var result = RecognitionBenchmark.Run(cfg);

        Console.WriteLine();
        Console.WriteLine($"  {"Condition",-12} │ {"single column",-20} │ {"voting (" + result.Columns + " cols)",-20}");
        Console.WriteLine($"  {"",-12} │ {"acc",-7}{"steps",-13} │ {"acc",-7}{"steps",-13}");
        Console.WriteLine("  ─────────────┼──────────────────────┼──────────────────────");
        foreach (var c in result.Conditions)
        {
            Console.WriteLine(
                $"  {c.Condition,-12} │ {c.SingleColumn.Top1Accuracy,-7:P0}{c.SingleColumn.MeanSteps,-13:F1} │ " +
                $"{c.MultiColumn.Top1Accuracy,-7:P0}{c.MultiColumn.MeanSteps,-13:F1}");
        }
        Console.WriteLine();
        Console.WriteLine("  → Voting matches single-column accuracy but converges in far fewer");
        Console.WriteLine("    sensorimotor steps: each round pools evidence from several columns.");
        return result;
    }

    // ── Part 2: live sensing & voting ─────────────────────────────────────────

    private static VizEpisode RunLiveRecognition()
    {
        Console.WriteLine("  ── Live recognition episode ─────────────────────────────");
        Console.WriteLine();

        const int columns = 4;
        var ds = SyntheticObjectDataset.Generate(
            objectCount: 6, locationsPerObject: 10, featureVocabularySize: 4, seed: 7);
        var pose = new PoseEncoder(-ds.Extent - 1, ds.Extent + 1, -ds.Extent - 1, ds.Extent + 1,
            positionBits: 100, positionActiveBits: 10, orientationBuckets: 8, orientationActiveBits: 5);

        var single = new CorticalColumn(ColumnConfig(ds.FeatureSize, seed: 7));
        var voting = new MultiColumnSystem(
            Enumerable.Range(0, columns).Select(c => ColumnConfig(ds.FeatureSize, seed: 8 + c)).ToList(),
            VotingTopology.AllToAll, minAgreementFraction: 1.0f);

        foreach (var obj in ds.Objects)
        {
            var obs = Encode(obj.Points, pose);
            single.LearnObject(obs);
            voting.LearnObject(Enumerable.Repeat(obs, columns).ToList());
        }

        // Pick the object the single column finds hardest (most steps) so the
        // gradual narrowing — and the voting speed-up — are clearest.
        var target = ds.Objects[0];
        int hardest = -1;
        foreach (var obj in ds.Objects)
        {
            int steps = StepsToConverge(single.Infer(Encode(obj.Points, pose)));
            if (steps > hardest) { hardest = steps; target = obj; }
        }

        var probe = Encode(target.Points, pose);
        Console.WriteLine($"  Presenting object #{target.Id}  ({ds.ObjectCount} objects learned, {probe.Count} landmarks).");
        Console.WriteLine();

        // Record the episode for the visual demo alongside the narration.
        var episode = new VizEpisode
        {
            TargetId      = target.Id,
            ObjectCount   = ds.ObjectCount,
            ColumnsVoting = columns,
            Objects       = ds.Objects
                .Select(o => new VizObject(o.Id, o.Points
                    .Select(pt => new VizPoint(pt.X, pt.Y, FeatureId(pt.Feature, ds.Vocabulary)))
                    .ToList()))
                .ToList(),
        };

        // Single column: one landmark per step.
        Console.WriteLine("  Single column — one landmark per step:");
        var outs = single.Infer(probe);
        for (int i = 0; i < outs.Count; i++)
        {
            var o = outs[i];
            episode.SingleSteps.Add([.. o.ObjectHypotheses]);
            Console.WriteLine($"    step {i + 1,2}: {o.ObjectHypotheses.Count,2} hypotheses {Bar(o.ObjectHypotheses.Count, ds.ObjectCount)}"
                + (o.Converged ? $"  → recognised #{o.ObjectHypotheses[0]}" : ""));
            if (o.Converged) break;
        }

        // Voting population: `columns` landmarks per round.
        Console.WriteLine();
        Console.WriteLine($"  Voting population — {columns} columns sense {columns} landmarks per round:");
        voting.Reset();
        int p = probe.Count;
        for (int round = 0; round < 12; round++)
        {
            var inputs = new (Sdr, Sdr)[columns];
            for (int c = 0; c < columns; c++) inputs[c] = probe[(round * columns + c) % p];
            var consensus = voting.VoteStep(inputs);
            episode.VotingRounds.Add([.. consensus.Candidates]);
            Console.WriteLine($"    round {round + 1,2}: {consensus.Candidates.Count,2} candidates {Bar(consensus.Candidates.Count, ds.ObjectCount)}"
                + (consensus.Candidates.Count == 1 ? $"  → recognised #{consensus.Candidates[0]}" : ""));
            if (consensus.Candidates.Count == 1) break;
        }

        Console.WriteLine();
        Console.WriteLine("  Same object, same evidence — the voting population settles on the");
        Console.WriteLine("  identity in a fraction of the steps. That is the Thousand Brains idea.");
        return episode;
    }

    // ── Part 3: pose-invariant recognition ────────────────────────────────────

    private static VizPoseInvariance RunPoseInvariance()
    {
        Console.WriteLine("  ── Pose-invariant recognition ───────────────────────────");
        Console.WriteLine();

        var ds = SyntheticObjectDataset.Generate(
            objectCount: 8, locationsPerObject: 12, featureVocabularySize: 5, seed: 21);
        float bound = ds.Extent * 4 + 12f;
        var pose = new PoseEncoder(-bound, bound, -bound, bound,
            positionBits: 100, positionActiveBits: 10, orientationBuckets: 8, orientationActiveBits: 5);

        // Absolute recogniser (location-based column) vs pose-invariant recogniser
        // (displacement structure), taught the same objects.
        var column = new CorticalColumn(ColumnConfig(ds.FeatureSize, seed: 21));
        var displacement = new DisplacementRecognizer(minOverlap: 12, distanceTolerance: 0.6f);
        foreach (var o in ds.Objects)
        {
            column.LearnObject(Encode(o.Points, pose));
            displacement.Learn(o.Id, o.Points);
        }

        var move = new Pose2D(8f, -6f, 70f);   // rotate 70°, translate (8, −6)
        var viz = new VizPoseInvariance { Dx = 8f, Dy = -6f, ThetaDeg = 70f };
        int columnCorrect = 0, displacementCorrect = 0;
        foreach (var o in ds.Objects)
        {
            var moved = o.Points.Select(p => p.Transformed(move)).ToList();

            var hyp = column.Infer(Encode(moved, pose))[^1].ObjectHypotheses;
            bool columnOk = hyp.Count == 1 && hyp[0] == o.Id;
            if (columnOk) columnCorrect++;

            bool displacementOk = displacement.Recognize(moved) == o.Id;
            if (displacementOk) displacementCorrect++;

            viz.Rows.Add(new VizPoseRow(o.Id, columnOk, displacementOk));
        }

        int n = ds.ObjectCount;
        Console.WriteLine("  Each object moved by a rigid transform (rotate 70°, translate (8,−6)):");
        Console.WriteLine();
        Console.WriteLine($"    absolute cortical column   : {100.0 * columnCorrect / n,5:F0}%  ({columnCorrect}/{n})");
        Console.WriteLine($"    displacement recogniser    : {100.0 * displacementCorrect / n,5:F0}%  ({displacementCorrect}/{n})");
        Console.WriteLine();
        Console.WriteLine("  Absolute location matching breaks when the object moves; recognising by");
        Console.WriteLine("  the displacement structure between features is invariant to rigid pose —");
        Console.WriteLine("  the role of displacement cells in the Thousand Brains theory.");
        return viz;
    }

    // ── Part 4: image sensor scan ─────────────────────────────────────────────

    // A procedural 96×96 texture image with three visibly different regions —
    // left third vertical stripes, middle third flat mid-grey, right third
    // diagonal stripes — scanned row-major by an ImageSensorEnvironment patch
    // sensor, recording the raw descriptor (dominant orientation category,
    // energy share, mean luminance) at every fixation for the visual demo.
    private static VizImageScan RunImageScan()
    {
        Console.WriteLine("  ── Image sensor scan ────────────────────────────────────");
        Console.WriteLine();
        Console.WriteLine("  Scanning a procedural 96×96 texture with a 16 px sensor patch …");

        const int size = 96;
        const int patchSize = 16;
        const int stride = 8;
        const int orientationBins = 8;

        // Left third: vertical stripes sin(x); middle: flat mid-grey; right
        // third: diagonal stripes sin(x+y). All values in [0, 1].
        var image = new float[size, size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                image[y, x] =
                    x < size / 3     ? 0.5f + 0.5f * MathF.Sin(2f * MathF.PI * x / 8f)
                  : x < 2 * size / 3 ? 0.5f
                  :                    0.5f + 0.5f * MathF.Sin(2f * MathF.PI * (x + y) / 8f);
            }
        }

        var env = new ImageSensorEnvironment(
            patchSize: patchSize, stride: stride, orientationBins: orientationBins);
        env.AddObject(0, image);
        env.PresentObject(0);

        // Navigate from the centre to fixation (0, 0): clamped moves, so
        // GridWidth lefts + GridHeight ups are always enough.
        for (int i = 0; i < env.GridWidth; i++) env.MoveTo(ImageSensorEnvironment.MoveLeft);
        for (int i = 0; i < env.GridHeight; i++) env.MoveTo(ImageSensorEnvironment.MoveUp);

        // Row-major scan: across each row with MoveRight, then MoveDown and
        // MoveLeft back to column 0 for the next row.
        var scan = new VizImageScan
        {
            Width           = size,
            Height          = size,
            Pixels          = new float[size * size],
            GridCols        = env.GridWidth,
            GridRows        = env.GridHeight,
            PatchSize       = patchSize,
            Stride          = stride,
            OrientationBins = orientationBins,
        };
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                scan.Pixels[y * size + x] = image[y, x];

        for (int row = 0; row < env.GridHeight; row++)
        {
            for (int col = 0; col < env.GridWidth; col++)
            {
                var (c, r) = env.SensorPosition;
                var (category, share, luminance) = env.CurrentDescriptor();
                env.Observe();
                scan.Fixations.Add(new VizFixation(c, r, category, share, luminance));
                if (col < env.GridWidth - 1) env.MoveTo(ImageSensorEnvironment.MoveRight);
            }
            if (row < env.GridHeight - 1)
            {
                env.MoveTo(ImageSensorEnvironment.MoveDown);
                for (int i = 0; i < env.GridWidth - 1; i++)
                    env.MoveTo(ImageSensorEnvironment.MoveLeft);
            }
        }

        return scan;
    }

    // ── Web page export ───────────────────────────────────────────────────────

    private static void SaveVisualization(VizData data)
    {
        string dir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".predictivecoding");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "thousand-brains-demo.html");
        HtmlVisualizer.Save(path, data);

        Console.WriteLine();
        Console.WriteLine($"  Visual demo written to  {path}");

        // UseShellExecute hands the document to the shell ASYNCHRONOUSLY: if this
        // process exits right away, the handoff can die before the browser claims
        // the page (console apps vanish at end-of-Main). Hold the process open
        // until the user is done looking.
        try { Process.Start(new ProcessStartInfo(path) { UseShellExecute = true }); } catch { }
        Console.WriteLine("  Opening it in the browser — press ENTER to exit.");
        Console.ReadLine();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Maps a dataset feature Sdr to the index of its vocabulary entry. The
    // dataset draws every feature directly from the vocabulary, so reference
    // equality resolves it; max overlap is the (equivalent) fallback.
    private static int FeatureId(Sdr feature, IReadOnlyList<Sdr> vocabulary)
    {
        int best = 0, bestOverlap = -1;
        for (int v = 0; v < vocabulary.Count; v++)
        {
            if (ReferenceEquals(vocabulary[v], feature)) return v;
            int overlap = vocabulary[v].Overlap(feature);
            if (overlap > bestOverlap) { best = v; bestOverlap = overlap; }
        }
        return best;
    }

    private static IReadOnlyList<(Sdr Feature, Sdr Location)> Encode(
        IReadOnlyList<FeaturePoint> points, PoseEncoder pose)
    {
        var obs = new List<(Sdr, Sdr)>(points.Count);
        foreach (var pt in points)
            obs.Add((pt.Feature, pose.EncodeXY(pt.X, pt.Y)));
        return obs;
    }

    private static int StepsToConverge(IReadOnlyList<CorticalColumnOutput> outputs)
    {
        for (int i = 0; i < outputs.Count; i++)
            if (outputs[i].Converged) return i + 1;
        return outputs.Count;
    }

    private static CorticalColumnConfig ColumnConfig(int featureSize, int seed) => new()
    {
        InputSize      = featureSize,
        Columns        = 256,
        CellsPerColumn = 4,
        Sparsity       = 0.04f,
        MinOverlap     = 4,
        Seed           = seed,
    };

    private static string Bar(int count, int max)
    {
        int filled = Math.Clamp(count, 0, max);
        return "[" + new string('█', filled) + new string('·', max - filled) + "]";
    }
}
