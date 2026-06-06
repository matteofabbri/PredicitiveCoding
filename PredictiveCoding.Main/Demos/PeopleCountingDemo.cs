using PredictiveCoding.Vision;
using PredictiveCoding.Vision.Capture;
using PredictiveCoding.Vision.Counting;
using PredictiveCoding.Vision.Detectors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.Main.Demos;

// ════════════════════════════════════════════════════════════════════════════
// PeopleCountingDemo
//
//   Live people counting on a camera feed or video file — the first milestone of
//   docs/10-people-counting-roadmap.md (Phases 0–2: capture → detect → count).
//
//     • Source   : webcam index ("0"), a video-file path, or an rtsp:// URL.
//     • Detector : YOLOX full-body (COCO "person", Apache-2.0, auto-downloaded)
//                  or RetinaFace face/head count (already bundled, no download).
//     • Output   : live per-frame count in the console, sliding-median smoothed;
//                  optionally saves annotated frames (green boxes) to disk.
//
//   Multi-object tracking and unique counting (Phases 3–4) are the next step.
// ════════════════════════════════════════════════════════════════════════════
public static class PeopleCountingDemo
{
    private static readonly string ModelsDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "vision-models");

    private static readonly string OutputDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "vision-output");

    public static void Run()
    {
        Console.WriteLine("\n  ── Live people counting ───────────────────────────────────");
        Console.WriteLine("  Counts people in a live camera feed or a video file.\n");

        Console.Write("  Source — webcam index, video path, or rtsp:// URL [0]: ");
        string source = Console.ReadLine()?.Trim() is { Length: > 0 } s ? s : "0";

        Console.WriteLine("\n  Detector:");
        Console.WriteLine("    1  YOLOX full-body  (COCO 'person', Apache-2.0, downloads ~36 MB on first use)");
        Console.WriteLine("    2  RetinaFace faces (already bundled, counts faces pointing at the camera)");
        Console.Write("  Choice [1]: ");
        string det = Console.ReadLine()?.Trim() is { Length: > 0 } d ? d : "1";

        Console.Write("\n  Process every Nth frame (1 = every frame) [2]: ");
        int stride = int.TryParse(Console.ReadLine(), out int st) && st > 0 ? st : 2;

        Console.Write("  Save annotated frames to disk? [y/N]: ");
        bool save = Console.ReadLine()?.Trim().StartsWith("y", StringComparison.OrdinalIgnoreCase) == true;

        using IPersonDetector detector = det == "2"
            ? new FaceCountDetector(ModelsDir)
            : new YoloPersonDetector(ModelsDir);

        Console.WriteLine($"\n  Detector : {detector.Name}");
        Console.WriteLine($"  Source   : {source}");
        if (save)
        {
            Directory.CreateDirectory(OutputDir);
            Console.WriteLine($"  Saving   : {OutputDir}");
        }
        Console.WriteLine("\n  Warming up the detector… (first frame also triggers any model download)");
        Console.WriteLine("  Press any key to stop.\n");

        var smoother = new CountSmoother(windowSize: 5);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int frame = 0, peak = 0;
        long total = 0;

        try
        {
            foreach (Image<Rgb24> img in FrameStream.Read(source, stride))
            {
                using (img)
                {
                    IReadOnlyList<Detection> dets = detector.Detect(img);
                    int raw = dets.Count;
                    int smooth = smoother.Add(raw);
                    peak = Math.Max(peak, raw);
                    total += raw;
                    frame++;

                    double fps = frame / Math.Max(1e-3, sw.Elapsed.TotalSeconds);
                    Console.Write(
                        $"\r  frame {frame,5} │ people now: {smooth,3} (raw {raw,3}) │ peak {peak,3} │ {fps,5:F1} fps   ");

                    if (save)
                    {
                        Annotate.DrawBoxes(img, dets);
                        img.SaveAsJpeg(Path.Combine(OutputDir, $"frame_{frame:D5}_n{raw}.jpg"));
                    }
                }

                if (KeyPressed()) break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n\n  [stopped] {ex.Message}");
            Console.WriteLine("  Tip: for a webcam try index 0; for a file pass a full path; for an IP cam an rtsp:// URL.");
        }

        Console.WriteLine("\n");
        Console.WriteLine($"  Frames processed : {frame}");
        if (frame > 0) Console.WriteLine($"  Average count    : {(double)total / frame:F1}");
        Console.WriteLine($"  Peak count       : {peak}");
        Console.WriteLine("\n  Done.");
    }

    // Non-blocking "any key" check; harmless no-op when stdin is redirected.
    private static bool KeyPressed()
    {
        try
        {
            if (!Console.KeyAvailable) return false;
            Console.ReadKey(intercept: true);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
