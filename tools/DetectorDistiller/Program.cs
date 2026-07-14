using System.Drawing;
using PredictiveCoding.FaceRecognition;
using PredictiveCoding.Imaging;
using PredictiveCoding.Vision.Capture;
using PredictiveCoding.Vision.Detectors;

// ─────────────────────────────────────────────────────────────────────────────
// DetectorDistiller
//
// Trains the in-house, download-free detectors by DISTILLING the heavy ONNX
// "teacher" models that the repo otherwise depends on at inference:
//
//   YOLOX      → labels person boxes      → trains an in-house HOG person detector
//   RetinaFace → labels face boxes        → trains an in-house HOG face detector
//   ArcFace    → labels face embeddings   → trains an in-house face embedder (regress
//                                            face-HOG → ArcFace 512-D, i.e. distillation)
//
// The teachers supply the labels once (offline). After this runs, the produced
// models (`hog-person.txt`, `hog-face.txt`, `face-embedder.txt`) drive
// HogPersonDetector / LinearRegressor with NO ONNX runtime and NO model download.
//
// Usage:
//   DetectorDistiller --source <video|imageFolder> [--frames N] [--neg K]
//                     [--models <onnxDir>] [--out <outDir>]
// ─────────────────────────────────────────────────────────────────────────────

string? source = Arg("--source");
if (source is null)
{
    Console.WriteLine("Distil YOLOX / RetinaFace / ArcFace into in-house HOG detectors + a face embedder.");
    Console.WriteLine();
    Console.WriteLine("  Usage: DetectorDistiller --source <video|imageFolder> [--frames N] [--neg K]");
    Console.WriteLine("                           [--models <onnxDir>] [--out <outDir>]");
    Console.WriteLine();
    Console.WriteLine("  The ONNX teacher models are downloaded on first use; outputs are plain text");
    Console.WriteLine("  loadable by HogPersonDetector (LinearModel.LoadFromFile) and LinearRegressor.");
    return 0;
}

string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
string modelsDir = Arg("--models") ?? Path.Combine(home, ".predictivecoding", "vision-models");
string outDir = Arg("--out") ?? Path.Combine(home, ".predictivecoding", "vision-inhouse");
int maxFrames = int.TryParse(Arg("--frames"), out int mf) ? mf : 200;
int negPerFrame = int.TryParse(Arg("--neg"), out int ng) ? ng : 5;
Directory.CreateDirectory(outDir);

Console.WriteLine("╔══════════════════════════════════════════════════════╗");
Console.WriteLine("║   Detector Distiller · teachers → in-house students  ║");
Console.WriteLine("╚══════════════════════════════════════════════════════╝");
Console.WriteLine($"  Source     : {source}");
Console.WriteLine($"  ONNX models: {modelsDir}  (downloaded on first use)");
Console.WriteLine($"  Output     : {outDir}");
Console.WriteLine($"  Budget     : {maxFrames} frames, {negPerFrame} negatives/frame");
Console.WriteLine();

var rng = new Random(7);
var hogPerson = new HogDescriptor(64, 128);
var hogFace = new HogDescriptor(48, 48);
var trainerPerson = new HogTrainer(hogPerson);
var trainerFace = new HogTrainer(hogFace);
var embedSamples = new List<(float[] Features, float[] Target)>();

using var yolo = new YoloPersonDetector(modelsDir);
using var faceDetector = new FaceCountDetector(modelsDir);
using var arcFace = new ArcFaceOnnxEmbedder(modelsDir, useAlignment: false);

int frameNo = 0;
foreach (RgbImage frame in Frames(source, maxFrames))
{
    using (frame)
    {
        // ── Persons: YOLOX teacher → in-house person detector ────────────────
        var persons = SafeDetect(yolo, frame);
        var personRects = persons.Select(ToRect).ToList();
        foreach (Rectangle b in personRects) trainerPerson.AddPositive(frame, b);
        trainerPerson.AddRandomNegatives(frame, personRects, negPerFrame, rng);

        // ── Faces: RetinaFace teacher → in-house face detector ───────────────
        var faces = SafeDetect(faceDetector, frame);
        var faceRects = faces.Select(ToRect).ToList();
        foreach (Rectangle b in faceRects) trainerFace.AddPositive(frame, b);
        trainerFace.AddRandomNegatives(frame, faceRects, negPerFrame, rng);

        // ── ArcFace teacher → distilled in-house face embedder ───────────────
        foreach (Rectangle b in faceRects)
        {
            float[]? embedding = EmbedFace(arcFace, frame, b, outDir);
            if (embedding is not null)
                embedSamples.Add((hogFace.FromCrop(frame, b), embedding));
        }
    }

    if (++frameNo % 20 == 0)
        Console.WriteLine($"  {frameNo,4} frames | person +{trainerPerson.PositiveCount} | face +{trainerFace.PositiveCount} | emb {embedSamples.Count}");
}

Console.WriteLine();
Console.WriteLine($"  Collected over {frameNo} frames:");
Console.WriteLine($"    person  {trainerPerson.PositiveCount} pos / {trainerPerson.NegativeCount} neg");
Console.WriteLine($"    face    {trainerFace.PositiveCount} pos / {trainerFace.NegativeCount} neg");
Console.WriteLine($"    arcface {embedSamples.Count} distillation pairs");
Console.WriteLine();

if (trainerPerson.PositiveCount > 0)
{
    string path = Path.Combine(outDir, "hog-person.txt");
    trainerPerson.Train("person", epochs: 20).SaveToFile(path);
    Console.WriteLine($"  ✓ in-house person detector → {path}");
}
if (trainerFace.PositiveCount > 0)
{
    string path = Path.Combine(outDir, "hog-face.txt");
    trainerFace.Train("face", epochs: 20).SaveToFile(path);
    Console.WriteLine($"  ✓ in-house face detector   → {path}");
}
if (embedSamples.Count >= 10)
{
    string path = Path.Combine(outDir, "face-embedder.txt");
    var reg = new LinearRegressor(hogFace.Dimension, arcFace.Dimension);
    reg.Fit(embedSamples, epochs: 30, learningRate: 1e-3f);
    reg.SaveToFile(path);
    Console.WriteLine($"  ✓ in-house face embedder   → {path}  (distilled from ArcFace)");
}

Console.WriteLine();
Console.WriteLine("  Load the person detector with no ONNX:");
Console.WriteLine("    var hog = new HogDescriptor(64,128);");
Console.WriteLine("    var model = LinearModel.LoadFromFile(\"hog-person.txt\");");
Console.WriteLine("    IPersonDetector detector = new HogPersonDetector(hog, model);");
return 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static IReadOnlyList<Detection> SafeDetect(IPersonDetector detector, RgbImage frame)
{
    try { return detector.Detect(frame); }
    catch (Exception ex) { Console.WriteLine($"  [warn] {detector.Name}: {ex.Message}"); return []; }
}

static Rectangle ToRect(Detection d)
{
    int x = (int)MathF.Floor(d.X1), y = (int)MathF.Floor(d.Y1);
    int w = Math.Max(1, (int)MathF.Ceiling(d.X2 - d.X1));
    int h = Math.Max(1, (int)MathF.Ceiling(d.Y2 - d.Y1));
    return new Rectangle(x, y, w, h);
}

static float[]? EmbedFace(ArcFaceOnnxEmbedder embedder, RgbImage frame, Rectangle box, string scratchDir)
{
    Rectangle rect = Rectangle.Intersect(box, frame.Bounds);
    if (rect.Width < 4 || rect.Height < 4) return null;

    string tmp = Path.Combine(scratchDir, $"_face_{Guid.NewGuid():N}.png");
    try
    {
        using (RgbImage crop = frame.Crop(rect))
            crop.SaveAsPng(tmp);
        float[] e = embedder.Embed(tmp);
        return e.Length == embedder.Dimension ? e : null;
    }
    catch (Exception ex) { Console.WriteLine($"  [warn] ArcFace embed: {ex.Message}"); return null; }
    finally { try { if (File.Exists(tmp)) File.Delete(tmp); } catch { /* best effort */ } }
}

static IEnumerable<RgbImage> Frames(string source, int max)
{
    int n = 0;
    if (Directory.Exists(source))
    {
        string[] exts = [".jpg", ".jpeg", ".png", ".bmp"];
        foreach (string file in Directory.EnumerateFiles(source).OrderBy(f => f))
        {
            if (n >= max) yield break;
            if (!exts.Contains(Path.GetExtension(file).ToLowerInvariant())) continue;
            RgbImage? img = null;
            try { img = RgbImage.Load(file); } catch { img = null; }
            if (img is null) continue;
            n++;
            yield return img;
        }
    }
    else
    {
        foreach (RgbImage frame in FrameStream.Read(source))
        {
            if (n >= max) { frame.Dispose(); yield break; }
            n++;
            yield return frame;
        }
    }
}

string? Arg(string name)
{
    for (int i = 0; i < args.Length - 1; i++)
        if (args[i] == name) return args[i + 1];
    return null;
}
