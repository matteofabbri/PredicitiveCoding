using System.Text;
using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for object-model persistence (SdrCodec + ObjectModelStore).</summary>
public class BrainPersistenceTests
{
    [Fact]
    public void SdrCodec_RoundTrips_IncludingEmpty()
    {
        var rng = new Random(1);
        var sdr = Sdr.Random(128, 16, rng);
        Assert.Equal(sdr, SdrCodec.Decode(SdrCodec.Encode(sdr)));

        var empty = Sdr.Empty(64);
        Assert.Equal(empty, SdrCodec.Decode(SdrCodec.Encode(empty)));
    }

    [Fact]
    public void ObjectModelStore_RoundTrips_ObjectsAndPoints()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 5, locationsPerObject: 9, seed: 2);

        var sw = new StringWriter();
        ObjectModelStore.Save(sw, ds.Objects);
        var loaded = ObjectModelStore.Load(new StringReader(sw.ToString()));

        Assert.Equal(ds.ObjectCount, loaded.Count);
        for (int o = 0; o < ds.ObjectCount; o++)
        {
            Assert.Equal(ds.Objects[o].Id, loaded[o].Id);
            Assert.Equal(ds.Objects[o].Points, loaded[o].Points);   // FeaturePoint structural equality
        }
    }

    [Fact]
    public void ObjectModelStore_RoundTripsThroughFile()
    {
        var ds = SyntheticObjectDataset.Generate(objectCount: 3, locationsPerObject: 6, seed: 3);
        string path = Path.Combine(Path.GetTempPath(), $"brain_models_{Guid.NewGuid():N}.txt");
        try
        {
            ObjectModelStore.SaveToFile(path, ds.Objects);
            var loaded = ObjectModelStore.LoadFromFile(path);
            Assert.Equal(ds.ObjectCount, loaded.Count);
            Assert.Equal(ds.Objects[1].Points, loaded[1].Points);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ObjectModelStore_LoadedModels_AreStillRecognisable()
    {
        // Persistence is meaningful only if a reloaded model still works: load it
        // into a pose-invariant recogniser and recognise a transformed probe.
        var ds = SyntheticObjectDataset.Generate(objectCount: 6, locationsPerObject: 12, seed: 4);

        var sw = new StringWriter();
        ObjectModelStore.Save(sw, ds.Objects);
        var loaded = ObjectModelStore.Load(new StringReader(sw.ToString()));

        var rec = new DisplacementRecognizer(minOverlap: 12, distanceTolerance: 0.6f);
        foreach (var o in loaded) rec.Learn(o.Id, o.Points);

        var pose = new Pose2D(4f, -3f, 90f);
        var probe = ds.Objects[2].Points.Select(p => p.Transformed(pose)).ToList();
        Assert.Equal(2, rec.Recognize(probe));
    }
}
