using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint  —  save / resume trained parameters.
//
// Serialises the model's parameter tensors (shape + fp32 data) to a binary file
// and reloads them in place, so training can be resumed or a trained model can be
// run later.
// ─────────────────────────────────────────────────────────────────────────────

public static class Checkpoint
{
    private const int Magic = 0x4C4C4D31;   // "LLM1"

    public static void Save(string path, IReadOnlyList<Tensor> parameters)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write(Magic);
        w.Write(parameters.Count);
        foreach (var p in parameters)
        {
            w.Write(p.Rows);
            w.Write(p.Cols);
            foreach (float f in p.Data) w.Write(f);
        }
    }

    /// <summary>Loads parameters in place; tensor count and shapes must match.</summary>
    public static void Load(string path, IReadOnlyList<Tensor> parameters)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        if (r.ReadInt32() != Magic)
            throw new InvalidDataException("Not a valid LLM checkpoint.");
        int count = r.ReadInt32();
        if (count != parameters.Count)
            throw new InvalidDataException($"Checkpoint has {count} tensors; model has {parameters.Count}.");

        foreach (var p in parameters)
        {
            int rows = r.ReadInt32(), cols = r.ReadInt32();
            if (rows != p.Rows || cols != p.Cols)
                throw new InvalidDataException($"Shape mismatch: checkpoint [{rows},{cols}] vs model [{p.Rows},{p.Cols}].");
            for (int i = 0; i < p.Data.Length; i++) p.Data[i] = r.ReadSingle();
        }
    }
}
