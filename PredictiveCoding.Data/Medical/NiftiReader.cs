using System.IO.Compression;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Medical;

/// <summary>A NIfTI volume: its native dimensions and the voxels as a tensor.</summary>
/// <remarks><see cref="Voxels"/> is row-major <c>[nz, ny, nx]</c> (z slowest, x fastest).</remarks>
public sealed record NiftiVolume(IReadOnlyList<int> Dimensions, DenseTensor<float> Voxels);

/// <summary>
/// Reader for NIfTI-1 neuroimaging volumes (.nii / .nii.gz). Parses the 348-byte
/// header and the voxel array, applying the scl_slope/scl_inter scaling.
/// Little-endian files are supported.
/// </summary>
public static class NiftiReader
{
    /// <summary>Reads a .nii or .nii.gz file.</summary>
    public static NiftiVolume Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("NIfTI file not found.", path);

        byte[] bytes = File.ReadAllBytes(path);
        if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
            bytes = Gunzip(bytes);
        return Parse(bytes);
    }

    /// <summary>Parses NIfTI-1 bytes (header + voxels).</summary>
    public static NiftiVolume Parse(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length < 352)
            throw new FormatException("NIfTI data is too short for a header.");
        if (BitConverter.ToInt32(data, 0) != 348)
            throw new FormatException("Unsupported NIfTI header (expected a little-endian NIfTI-1, sizeof_hdr 348).");

        int nx = Math.Max(1, (int)BitConverter.ToInt16(data, 42));
        int ny = Math.Max(1, (int)BitConverter.ToInt16(data, 44));
        int nz = Math.Max(1, (int)BitConverter.ToInt16(data, 46));
        short datatype = BitConverter.ToInt16(data, 70);
        float voxOffset = BitConverter.ToSingle(data, 108);
        float sclSlope = BitConverter.ToSingle(data, 112);
        float sclInter = BitConverter.ToSingle(data, 116);

        int start = voxOffset > 0 ? (int)voxOffset : 352;
        int voxelCount = checked(nx * ny * nz);
        int bytesPerVoxel = BytesPerVoxel(datatype);
        if (data.Length < start + (long)voxelCount * bytesPerVoxel)
            throw new FormatException("NIfTI voxel data is truncated.");

        float slope = sclSlope != 0 ? sclSlope : 1f;
        var values = new float[voxelCount];
        for (int i = 0; i < voxelCount; i++)
        {
            int offset = start + i * bytesPerVoxel;
            float raw = datatype switch
            {
                2 => data[offset],                                  // uint8
                4 => BitConverter.ToInt16(data, offset),            // int16
                8 => BitConverter.ToInt32(data, offset),            // int32
                16 => BitConverter.ToSingle(data, offset),          // float32
                64 => (float)BitConverter.ToDouble(data, offset),   // float64
                256 => (sbyte)data[offset],                         // int8
                512 => BitConverter.ToUInt16(data, offset),         // uint16
                _ => throw new FormatException($"Unsupported NIfTI datatype code {datatype}."),
            };
            values[i] = raw * slope + sclInter;
        }

        return new NiftiVolume([nx, ny, nz], new DenseTensor<float>(values, nz, ny, nx));
    }

    private static int BytesPerVoxel(short datatype) => datatype switch
    {
        2 or 256 => 1,
        4 or 512 => 2,
        8 or 16 => 4,
        64 => 8,
        _ => throw new FormatException($"Unsupported NIfTI datatype code {datatype}."),
    };

    private static byte[] Gunzip(byte[] compressed)
    {
        using var input = new MemoryStream(compressed);
        using var gzip = new GZipStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        gzip.CopyTo(output);
        return output.ToArray();
    }
}
