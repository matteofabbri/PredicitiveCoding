using System.Buffers.Binary;
using System.Runtime.InteropServices;
using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// OptimizerCheckpoint  —  persist / restore Adam's moment state (roadmap-Pro
// Phase 5, "true resume": moments serialized next to the weights, so a resumed
// run CONTINUES the optimizer instead of restarting it).
//
// A plain flat file (the weights already own the FASTER store): the step counter
// t, then one [len][m fp32][v fp32] record per parameter, in the fixed
// Model.Parameters() order. Load installs the moments back onto the same
// parameter tensors and restores t.
// ─────────────────────────────────────────────────────────────────────────────

public static class OptimizerCheckpoint
{
    private const uint Magic = 0x4F50_5431;   // "OPT1"

    public static void Save(string path, IReadOnlyList<Tensor> parameters, Adam optimizer)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var fs = File.Create(path);
        Span<byte> hdr = stackalloc byte[12];
        BinaryPrimitives.WriteUInt32LittleEndian(hdr[..4], Magic);
        BinaryPrimitives.WriteInt32LittleEndian(hdr[4..8], optimizer.StepCount);
        BinaryPrimitives.WriteInt32LittleEndian(hdr[8..12], parameters.Count);
        fs.Write(hdr);

        var len = new byte[4];
        foreach (var p in parameters)
        {
            var moments = optimizer.MomentsFor(p);
            // A parameter never stepped has no moments yet — persist zeros so the
            // record count always matches the parameter count.
            float[] m = moments?.m ?? new float[p.Data.Length];
            float[] v = moments?.v ?? new float[p.Data.Length];

            BinaryPrimitives.WriteInt32LittleEndian(len, p.Data.Length);
            fs.Write(len);
            fs.Write(MemoryMarshal.AsBytes(m.AsSpan()));
            fs.Write(MemoryMarshal.AsBytes(v.AsSpan()));
        }
    }

    /// <summary>Restores moments + step counter onto <paramref name="optimizer"/>; shapes must match.</summary>
    public static void Load(string path, IReadOnlyList<Tensor> parameters, Adam optimizer)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var span = bytes.AsSpan();
        if (BinaryPrimitives.ReadUInt32LittleEndian(span[..4]) != Magic)
            throw new InvalidDataException($"'{path}' is not an optimizer checkpoint.");
        int t     = BinaryPrimitives.ReadInt32LittleEndian(span[4..8]);
        int count = BinaryPrimitives.ReadInt32LittleEndian(span[8..12]);
        if (count != parameters.Count)
            throw new InvalidDataException(
                $"Optimizer checkpoint has {count} parameters; model has {parameters.Count}.");

        int pos = 12;
        for (int i = 0; i < parameters.Count; i++)
        {
            int len = BinaryPrimitives.ReadInt32LittleEndian(span.Slice(pos, 4)); pos += 4;
            if (len != parameters[i].Data.Length)
                throw new InvalidDataException(
                    $"Optimizer moment length {len} ≠ parameter {i} length {parameters[i].Data.Length}.");
            var m = new float[len];
            var v = new float[len];
            span.Slice(pos, len * 4).CopyTo(MemoryMarshal.AsBytes(m.AsSpan())); pos += len * 4;
            span.Slice(pos, len * 4).CopyTo(MemoryMarshal.AsBytes(v.AsSpan())); pos += len * 4;
            optimizer.SetMoments(parameters[i], m, v);
        }
        optimizer.RestoreStepCount(t);
    }
}
