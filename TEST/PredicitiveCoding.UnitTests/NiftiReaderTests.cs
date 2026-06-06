using PredictiveCoding.Data.Medical;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the NIfTI-1 reader: build a tiny .nii in memory and parse it.</summary>
public sealed class NiftiReaderTests
{
    private static byte[] BuildNifti(int nx, int ny, int nz, float[] voxels, float slope = 0f, float inter = 0f)
    {
        var data = new byte[352 + voxels.Length * 4];
        BitConverter.GetBytes(348).CopyTo(data, 0);            // sizeof_hdr
        BitConverter.GetBytes((short)3).CopyTo(data, 40);      // dim[0] = 3 dimensions
        BitConverter.GetBytes((short)nx).CopyTo(data, 42);     // dim[1]
        BitConverter.GetBytes((short)ny).CopyTo(data, 44);     // dim[2]
        BitConverter.GetBytes((short)nz).CopyTo(data, 46);     // dim[3]
        BitConverter.GetBytes((short)16).CopyTo(data, 70);     // datatype = float32
        BitConverter.GetBytes((short)32).CopyTo(data, 72);     // bitpix
        BitConverter.GetBytes(352f).CopyTo(data, 108);         // vox_offset
        BitConverter.GetBytes(slope).CopyTo(data, 112);        // scl_slope
        BitConverter.GetBytes(inter).CopyTo(data, 116);        // scl_inter
        data[344] = (byte)'n';
        data[345] = (byte)'+';
        data[346] = (byte)'1';
        data[347] = 0;                                         // magic "n+1\0"
        Buffer.BlockCopy(voxels, 0, data, 352, voxels.Length * 4);
        return data;
    }

    [Fact]
    public void Parse_ReadsDimensionsAndVoxels()
    {
        float[] voxels = [1, 2, 3, 4, 5, 6, 7, 8];
        NiftiVolume volume = NiftiReader.Parse(BuildNifti(2, 2, 2, voxels));

        Assert.Equal([2, 2, 2], volume.Dimensions);
        Assert.Equal(new[] { 2, 2, 2 }, volume.Voxels.Dimensions); // [nz, ny, nx]
        Assert.Equal(voxels, volume.Voxels.Buffer);
    }

    [Fact]
    public void Parse_AppliesSlopeAndIntercept()
    {
        float[] voxels = [1, 2, 3, 4, 5, 6, 7, 8];
        NiftiVolume volume = NiftiReader.Parse(BuildNifti(2, 2, 2, voxels, slope: 2f, inter: 1f));

        Assert.Equal([3f, 5f, 7f, 9f, 11f, 13f, 15f, 17f], volume.Voxels.Buffer); // raw*2 + 1
    }

    [Fact]
    public void Parse_TooShort_Throws()
        => Assert.Throws<FormatException>(() => NiftiReader.Parse(new byte[100]));

    [Fact]
    public void Parse_BadHeader_Throws()
    {
        var data = new byte[352];
        BitConverter.GetBytes(999).CopyTo(data, 0); // not 348
        Assert.Throws<FormatException>(() => NiftiReader.Parse(data));
    }
}
