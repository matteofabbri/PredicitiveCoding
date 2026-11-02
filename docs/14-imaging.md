# 14. Imaging — `PredictiveCoding.Imaging`

The solution's own image layer: five small types that give the whole vision
stack (and `PredictiveCoding.Data`) pixel I/O with **no native binaries and no
licence gate**.

## Types

| Type | Role |
|------|------|
| **`RgbImage`** | A 24-bit RGB image: flat row-major byte buffer (3 bytes/pixel) plus the operations the vision stack needs — `Load`, `SaveAsPng`/JPEG, `Resize`, `Crop`, grayscale, `FromPixelData` (zero-copy wrap). Implements `IDisposable` as a no-op so streaming call sites can treat frames uniformly. |
| **`Rgb24`** | The 3-byte pixel struct (`MemoryMarshal.Cast` gives span views over a whole image). |
| **`NetpbmDecoder`** | In-house fallback decoder for PGM/PPM (Netpbm) images. |
| **`GifEncoder`** | In-house animated-GIF writer (used to export demo animations). |

Decoding normally goes through **StbImageSharp** (pure C#, public domain) with
the Netpbm decoder as fallback; encoding through **StbImageWriteSharp** and the
in-house `GifEncoder`.

## Why it exists

`PredictiveCoding.Vision`, `PredictiveCoding.FaceRecognition`, the
`DetectorDistiller` tool and several data sources all need to move pixels
around. Sharing one tiny managed implementation keeps those projects free of
System.Drawing GDI+ hazards and of heavyweight imaging dependencies on their
hot paths. (SixLabors ImageSharp is still used *inside* `PredictiveCoding.Data`
for animated formats — a different concern, isolated there.)
