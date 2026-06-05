using FellowOakDicom;
using FellowOakDicom.Imaging;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Medical;

/// <summary>Pixel data decoded from a DICOM image.</summary>
/// <remarks><see cref="Pixels"/> is <c>[frames, rows, columns]</c> in raw stored values.</remarks>
public sealed record DicomImageData(int Frames, int Rows, int Columns, DenseTensor<float> Pixels);

/// <summary>
/// Reads pixel data from DICOM medical-imaging files with fo-dicom.
/// </summary>
public static class DicomReader
{
    /// <summary>Reads the pixel data of a DICOM file.</summary>
    public static DicomImageData Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("DICOM file not found.", path);
        return Extract(DicomFile.Open(path).Dataset);
    }

    private static DicomImageData Extract(DicomDataset dataset)
    {
        var pixelData = DicomPixelData.Create(dataset);
        int rows = pixelData.Height;
        int columns = pixelData.Width;
        int frames = Math.Max(1, pixelData.NumberOfFrames);
        int bitsAllocated = pixelData.BitsAllocated;

        var data = new float[checked(frames * rows * columns)];
        int offset = 0;
        for (int f = 0; f < frames; f++)
        {
            byte[] frame = pixelData.GetFrame(f).Data;
            int pixelsPerFrame = rows * columns;
            for (int p = 0; p < pixelsPerFrame; p++)
            {
                data[offset + p] = bitsAllocated == 16
                    ? (ushort)(frame[p * 2] | (frame[p * 2 + 1] << 8))
                    : frame[p];
            }
            offset += pixelsPerFrame;
        }

        var tensor = new DenseTensor<float>(data, frames, rows, columns);
        return new DicomImageData(frames, rows, columns, tensor);
    }
}
