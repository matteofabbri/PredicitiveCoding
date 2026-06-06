using System.Globalization;

namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ObjectModelStore  —  persistence for learned sensorimotor object models.
//
// Saves and reloads objects as their feature-point clouds (id → list of
// FeaturePoints), the portable form that can rehydrate any recogniser: encode +
// CorticalColumn.LearnObject, DisplacementRecognizer.Learn, or a CompositionLibrary
// primitive. This is the Phase 7 "persistence of learned object models" task
// (deferred to Phase 9). The format is a small line-based text file so models are
// diff-friendly and human-inspectable; SDRs use SdrCodec.
//
// Reference: Leadholm et al. (2025) P9 (persistent object models).
// ─────────────────────────────────────────────────────────────────────────────

public static class ObjectModelStore
{
    private const string Header = "# PredictiveCoding.Brain object models v1";

    /// <summary>Writes <paramref name="objects"/> to <paramref name="writer"/>.</summary>
    public static void Save(TextWriter writer, IReadOnlyList<SyntheticObject> objects)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(objects);

        writer.WriteLine(Header);
        foreach (var obj in objects)
        {
            writer.WriteLine($"object {obj.Id.ToString(CultureInfo.InvariantCulture)} {obj.Points.Count.ToString(CultureInfo.InvariantCulture)}");
            foreach (var pt in obj.Points)
                writer.WriteLine(
                    $"{pt.X.ToString(CultureInfo.InvariantCulture)} " +
                    $"{pt.Y.ToString(CultureInfo.InvariantCulture)} " +
                    SdrCodec.Encode(pt.Feature));
        }
    }

    /// <summary>Reads objects previously written by <see cref="Save"/>.</summary>
    public static IReadOnlyList<SyntheticObject> Load(TextReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);

        var objects = new List<SyntheticObject>();
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            line = line.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;

            string[] head = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (head.Length != 3 || head[0] != "object")
                throw new FormatException($"Expected 'object <id> <count>', got '{line}'.");

            int id = int.Parse(head[1], CultureInfo.InvariantCulture);
            int count = int.Parse(head[2], CultureInfo.InvariantCulture);

            var points = new List<FeaturePoint>(count);
            for (int i = 0; i < count; i++)
            {
                string? p = reader.ReadLine()
                    ?? throw new FormatException($"Object {id}: expected {count} points, file ended early.");
                int sp1 = p.IndexOf(' ');
                int sp2 = p.IndexOf(' ', sp1 + 1);
                if (sp1 < 0 || sp2 < 0)
                    throw new FormatException($"Malformed point line: '{p}'.");

                float x = float.Parse(p.AsSpan(0, sp1), CultureInfo.InvariantCulture);
                float y = float.Parse(p.AsSpan(sp1 + 1, sp2 - sp1 - 1), CultureInfo.InvariantCulture);
                var feature = SdrCodec.Decode(p[(sp2 + 1)..]);
                points.Add(new FeaturePoint(x, y, feature));
            }
            objects.Add(new SyntheticObject(id, points));
        }
        return objects;
    }

    /// <summary>Saves <paramref name="objects"/> to a file.</summary>
    public static void SaveToFile(string path, IReadOnlyList<SyntheticObject> objects)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var writer = new StreamWriter(path);
        Save(writer, objects);
    }

    /// <summary>Loads objects from a file written by <see cref="SaveToFile"/>.</summary>
    public static IReadOnlyList<SyntheticObject> LoadFromFile(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var reader = new StreamReader(path);
        return Load(reader);
    }
}
